#!/usr/bin/env python3
"""Within-phase LoRA-drift diagnostics: how far did the adapter actually move?

``compare_lora.py`` answers the *cross-iteration* question (final checkpoint of iter
k vs iter k-1) and materializes the full ΔW = (α/r)·B@A, which OOMs on a 14B adapter
held for several checkpoints at once. This script answers the complementary
*within-phase* question — does the policy move across the steps_NNNN checkpoints of a
single training phase — and does it with **constant memory** by never forming ΔW.

For a LoRA layer with A:(r,in), B:(out,r), scaling s = α/r, ΔW = s·B·A. All metrics
reduce to r×r (=rank×rank) Gram contractions over the raw A/B matrices:

    ‖ΔW‖_F^2          = s^2 · Σ_layers  sum( (BᵀB) ⊙ (AAᵀ) )
    ⟨ΔW₁, ΔW₂⟩_F      = s₁s₂ · Σ_layers sum( (B₁ᵀB₂) ⊙ (A₁A₂ᵀ) )
    ‖ΔW_k − ΔW_j‖_F   = sqrt( ‖ΔW_k‖² + ‖ΔW_j‖² − 2⟨ΔW_k, ΔW_j⟩ )
    cos(ΔW_k, ΔW_j)   = ⟨ΔW_k, ΔW_j⟩ / (‖ΔW_k‖·‖ΔW_j‖)

For each (iter, side) it prints, per steps_NNNN checkpoint:
    ‖ΔW‖ (cumulative drift from base), ‖ΔW_k − ΔW_{k-1}‖ (per-step change),
    rel_step (= step / ‖ΔW_{k-1}‖), cos-to-prev (direction stability),
    cos-to-final (convergence), and — if a base model is resolvable —
    ΔW/W (shift relative to the base q/k/v/o weights, the interpretable magnitude).

A negligible-shift FLAG fires when the final ΔW/W is below --negligible-ratio
(default 1e-4): the policy technically trained but moved far too little to matter.

Output is stdout tables only; nothing is written into the results dir.

Usage:
    python util/lora_drift_diagnostics.py RESULTS_DIR [RESULTS_DIR ...]
        [--base-model-dir DIR] [--negligible-ratio F]

If --base-model-dir is omitted the script auto-resolves the base model named in
adapter_config.json against $HF_HOME and ~/model_backup-style roots; if none is
found it prints a note and omits the ΔW/W column (absolute norms still shown).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util._diag_common import iter_runs, load_run_summary, note  # noqa: E402
from plotting._data import load_args_yaml  # noqa: E402

# LoRA-targeted attention projections (matches run_training.sh adapter config).
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")

_PRE = "base_model.model."
_SA = ".lora_A.weight"
_SB = ".lora_B.weight"


# ---------------------------------------------------------------------------
# Adapter loading (raw A/B only — never forms ΔW)
# ---------------------------------------------------------------------------

def _resolve_adapter_dir(steps_dir: Path) -> Path:
    """Return the sql_agent adapter dir under a steps_NNNN checkpoint.

    Handles the legacy nested sql_agent/sql_agent/ layout (mirrors compare_lora.py).
    """
    base = steps_dir / "sql_agent"
    nested = base / "sql_agent"
    if not (base / "adapter_config.json").is_file() and (nested / "adapter_config.json").is_file():
        return nested
    return base


def load_lora_layers(adapter_dir: Path) -> tuple[float, dict[str, dict[str, torch.Tensor]]]:
    """Load {layer_key: {'A','B'}} and the global scaling s=α/r for one adapter."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r, alpha = cfg.get("r"), cfg.get("lora_alpha")
    if r is None or alpha is None:
        raise ValueError(f"adapter_config.json missing r/lora_alpha: {adapter_dir}")
    scaling = alpha / r
    sd = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for k, t in sd.items():
        if not k.startswith(_PRE):
            raise ValueError(f"unexpected adapter key prefix: {k!r}")
        s = k[len(_PRE):]
        if s.endswith(_SA):
            groups.setdefault(s[: -len(_SA)], {})["A"] = t.float()
        elif s.endswith(_SB):
            groups.setdefault(s[: -len(_SB)], {})["B"] = t.float()
        else:
            raise ValueError(f"unrecognized adapter key (no lora_A/B suffix): {k!r}")
    for lk, m in groups.items():
        if set(m) != {"A", "B"}:
            raise ValueError(f"incomplete A/B pair for {lk}: {sorted(m)}")
    return scaling, groups


# ---------------------------------------------------------------------------
# Gram contractions (r×r; exact, constant memory)
# ---------------------------------------------------------------------------

def _inner(s1: float, g1: dict, s2: float, g2: dict) -> float:
    """⟨ΔW₁, ΔW₂⟩_F summed over shared layers, via r×r contractions.

    ⟨s₁B₁A₁, s₂B₂A₂⟩ = s₁s₂·tr(B₁ᵀB₂ · A₂A₁ᵀ) = s₁s₂·Σ (B₁ᵀB₂) ⊙ (A₁A₂ᵀ).
    """
    tot = 0.0
    for lk in g1:
        if lk not in g2:
            continue
        A1, B1 = g1[lk]["A"], g1[lk]["B"]
        A2, B2 = g2[lk]["A"], g2[lk]["B"]
        P = B1.transpose(0, 1) @ B2          # (r, r)  = B₁ᵀB₂
        Q = A1 @ A2.transpose(0, 1)          # (r, r)  = A₁A₂ᵀ
        tot += s1 * s2 * float((P * Q).sum())
    return tot


def _norm2(s: float, g: dict) -> float:
    return _inner(s, g, s, g)


# ---------------------------------------------------------------------------
# Base-model norm over targeted modules (the ΔW/W denominator)
# ---------------------------------------------------------------------------

def _candidate_base_dirs(base_name: str) -> list[Path]:
    """Best-effort on-disk locations for an HF base model id like 'Org/Name'."""
    cands: list[Path] = []
    p = Path(base_name)
    if p.is_dir():
        cands.append(p)
    short = base_name.split("/")[-1]
    roots = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots += [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("/scratch") / os.environ.get("USER", "") / "model_backup",
    ]
    for root in roots:
        cands.append(root / short)
        cands.append(root / f"models--{base_name.replace('/', '--')}")
    return cands


def _resolve_snapshot(d: Path) -> Path | None:
    """Resolve an HF hub cache dir to its snapshot, or return d if it holds shards."""
    if not d.is_dir():
        return None
    if list(d.glob("*.safetensors")):
        return d
    snaps = d / "snapshots"
    if snaps.is_dir():
        for snap in sorted(snaps.iterdir()):
            if list(snap.glob("*.safetensors")):
                return snap
    return None


def base_norm_over_targets(base_dir: Path, targets: tuple[str, ...]) -> tuple[float, int]:
    """‖W‖_F over self_attn.{target}.weight matrices, via lazy safetensors reads."""
    shards = sorted(base_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors under base model dir: {base_dir}")
    tot2, n = 0.0, 0
    for sh in shards:
        with safe_open(str(sh), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(".weight") and "self_attn" in k and any(f".{t}.weight" in k for t in targets):
                    t = f.get_tensor(k).float()
                    tot2 += float((t * t).sum())
                    n += 1
    return math.sqrt(tot2), n


def resolve_base_norm(
    adapter_dir: Path,
    targets: tuple[str, ...],
    explicit_dir: str | None,
) -> tuple[float, str] | None:
    """Return (base_norm, source_str) or None if the base model can't be located.

    Explicit --base-model-dir is fail-fast (crashes if unusable). Auto-resolution
    is best-effort and returns None with a note on miss.
    """
    if explicit_dir is not None:
        d = _resolve_snapshot(Path(explicit_dir))
        if d is None:
            raise FileNotFoundError(f"--base-model-dir has no safetensors: {explicit_dir}")
        bn, n = base_norm_over_targets(d, targets)
        return bn, f"{d} ({n} matrices)"

    base_name = json.loads((adapter_dir / "adapter_config.json").read_text()).get(
        "base_model_name_or_path", ""
    )
    for cand in _candidate_base_dirs(base_name):
        snap = _resolve_snapshot(cand)
        if snap is not None:
            bn, n = base_norm_over_targets(snap, targets)
            return bn, f"{snap} ({n} matrices)"
    note(f"base model {base_name!r} not found on disk — omitting ΔW/W (pass --base-model-dir)")
    return None


# ---------------------------------------------------------------------------
# Per-side rendering
# ---------------------------------------------------------------------------

def _list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    ckpt_root = run_dir / "checkpoints"
    out: list[tuple[int, Path]] = []
    if not ckpt_root.is_dir():
        return out
    for d in ckpt_root.iterdir():
        if d.is_dir() and d.name.startswith("steps_"):
            try:
                out.append((int(d.name.split("_")[1]), d))
            except (ValueError, IndexError):
                continue
    out.sort(key=lambda x: x[0])
    return out


def render_side(
    n: int,
    side: str,
    run_dir: Path,
    targets: tuple[str, ...],
    base_dir_override: str | None,
    negligible_ratio: float,
) -> None:
    print(f"\n-- iter {n} / {side.upper()} --")
    cks = _list_checkpoints(run_dir)
    if not cks:
        note(f"iter_{n}/{side}: no checkpoints/steps_* found")
        return

    # Load all checkpoints' raw A/B (tiny: rank-r factors, not ΔW).
    loaded: list[tuple[int, float, dict]] = []
    for step, steps_dir in cks:
        adir = _resolve_adapter_dir(steps_dir)
        if not (adir / "adapter_model.safetensors").is_file():
            note(f"  steps_{step:04d}: missing adapter_model.safetensors — skipped")
            continue
        s, g = load_lora_layers(adir)
        loaded.append((step, s, g))
    if not loaded:
        note(f"iter_{n}/{side}: no usable adapters")
        return

    norm2 = {st: _norm2(s, g) for st, s, g in loaded}
    fst, fs, fg = loaded[-1]

    base = resolve_base_norm(_resolve_adapter_dir(cks[0][1]), targets, base_dir_override)
    base_norm = base[0] if base else None
    if base:
        print(f"  base ‖W‖ over {','.join(targets)}: {base_norm:.2f}   [{base[1]}]")

    cols = f"  {'step':>6} {'||dW||':>11} {'d_step':>10} {'rel':>7} {'cosPrev':>8} {'cosFin':>8}"
    if base_norm:
        cols += f" {'dW/W':>10}"
    print(cols)

    path_len = 0.0
    for i, (st, s, g) in enumerate(loaded):
        cum = math.sqrt(max(norm2[st], 0.0))
        if i == 0:
            dstep = cum
            rel = float("nan")
            cprev = float("nan")
        else:
            pst, ps, pg = loaded[i - 1]
            ip = _inner(s, g, ps, pg)
            dstep = math.sqrt(max(norm2[st] + norm2[pst] - 2 * ip, 0.0))
            prev_norm = math.sqrt(max(norm2[pst], 0.0))
            # Undefined against a zero-vector predecessor (e.g. the frozen warmup ckpt).
            if prev_norm < 1e-9 or cum < 1e-9:
                rel = float("nan")
                cprev = float("nan")
            else:
                rel = dstep / prev_norm
                cprev = ip / (cum * prev_norm)
        path_len += dstep
        if st == fst:
            cfin = 1.0
        else:
            cfin = _inner(s, g, fs, fg) / (math.sqrt(norm2[st] * norm2[fst]) + 1e-12)
        line = (
            f"  {st:>6} {cum:>11.5f} {dstep:>10.5f} "
            f"{rel:>7.3f} {cprev:>8.4f} {cfin:>8.4f}"
        )
        if base_norm:
            line += f" {cum / base_norm:>10.2e}"
        print(line)

    net = math.sqrt(norm2[fst])
    summary = (
        f"  net ‖dW‖ (base→final) = {net:.5f}; "
        f"cumulative path length = {path_len:.5f}"
    )
    if base_norm:
        ratio = net / base_norm
        summary += f"; final dW/W = {ratio:.2e}"
    print(summary)

    flags: list[str] = []
    if net < 1e-9:
        flags.append("FROZEN(dW~0 — actor never updated)")
    if base_norm and net / base_norm < negligible_ratio:
        flags.append(f"negligible_shift(dW/W {net / base_norm:.1e} < {negligible_ratio:.0e})")
    print(f"  FLAGS: {', '.join(flags) if flags else '(none)'}")


def render_dir(results_dir: str, targets: tuple[str, ...], base_dir: str | None, neg: float) -> None:
    summary = load_run_summary(results_dir)
    print(f"\n== {results_dir}  [{summary['honeypot_type']}] ==")
    runs = list(iter_runs(results_dir))
    if not runs:
        note(f"{results_dir}: no trained iterations found")
        return
    header_args = load_args_yaml(runs[0][2])
    print(
        f"   lr={header_args.get('lr', 'n/a')} "
        f"warmup_steps={header_args.get('warmup_steps', 'n/a')} "
        f"num_env_steps={header_args.get('num_env_steps', summary.get('num_env_steps'))}"
    )
    for n, side, run_dir in runs:
        render_side(n, side, run_dir, targets, base_dir, neg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dirs", nargs="+", help="one or more results-* directories")
    ap.add_argument("--base-model-dir", default=None,
                    help="On-disk base model dir for the ΔW/W denominator (else auto-resolve).")
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                    help="Comma-separated LoRA-targeted module names (default: q_proj,k_proj,v_proj,o_proj).")
    ap.add_argument("--negligible-ratio", type=float, default=1e-4,
                    help="Flag the run if final ΔW/W is below this (default: 1e-4).")
    args = ap.parse_args()

    targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())
    for d in args.results_dirs:
        render_dir(d, targets, args.base_model_dir, args.negligible_ratio)


if __name__ == "__main__":
    main()
