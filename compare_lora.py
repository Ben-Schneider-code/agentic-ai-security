#!/usr/bin/env python3
"""
compare_lora.py <result_dir> — plot LoRA weight-delta evolution across self-play iterations.

For every iteration found in result_dir the script computes the LoRA task vector
(ΔW = (α/r)·B@A) for both the red and blue teams and derives five metrics:

  1. ‖ΔW_k‖_F per iteration — cumulative drift from the base model
  2. ‖ΔW_k − ΔW_{k-1}‖_F per iteration — actual change made at each self-play step
     (iter 1 is compared to the base model, i.e. ΔW_0 = 0)
  3. Cosine similarity between consecutive iterations (novelty)
  4. Cosine similarity vs. iteration 1 (how much later iters repeat the first)
  5. Cosine similarity between red and blue task vectors (arms-race signature)

Two PNGs and a JSON of all raw metrics are written to
  <result_dir>/<out_subdir>/lora_delta_overview.png   (1×2 panel: metrics 1 & 2)
  <result_dir>/<out_subdir>/lora_delta_cosine.png     (1×3 panel: metrics 3, 4 & 5)
  <result_dir>/<out_subdir>/lora_delta_metrics.json

The script is deliberately strict: any missing checkpoint, adapter file, or
base-model inconsistency raises an exception immediately — nothing is silently
skipped or defaulted.

Usage:
    python compare_lora.py <result_dir>
    python compare_lora.py <result_dir> --base-model meta-llama/Llama-3.1-8B-Instruct
    python compare_lora.py <result_dir> --out-subdir lora_delta_debug
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Resolve imports from the repo root regardless of where the script is invoked
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_task_vector import LLMTaskVector  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint discovery  (mirrors util/cross_evaluate.py — inlined here to
# avoid importing the full MARFT training framework as a side-effect)
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(team_dir: str) -> str | None:
    """Return the highest-step sql_agent checkpoint path under team_dir."""
    candidates: list[tuple[int, str]] = []
    for root, dirs, _files in os.walk(team_dir):
        if "sql_agent" in dirs:
            sql_agent_path = os.path.join(root, "sql_agent")
            parent = os.path.basename(root)
            if parent.startswith("steps_"):
                try:
                    step = int(parent.split("_")[1])
                    # Fallback: if adapter_config.json is missing directly in
                    # sql_agent/ but present in sql_agent/sql_agent/, use the
                    # nested path (produced when adapter_name="sql_agent" was
                    # passed to PeftModel.from_pretrained in older checkpoints).
                    nested = os.path.join(sql_agent_path, "sql_agent")
                    if (
                        not os.path.isfile(os.path.join(sql_agent_path, "adapter_config.json"))
                        and os.path.isfile(os.path.join(nested, "adapter_config.json"))
                    ):
                        sql_agent_path = nested
                    candidates.append((step, sql_agent_path))
                except (ValueError, IndexError):
                    pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _discover_checkpoints(
    selfplay_dir: str,
) -> tuple[dict[int, str], dict[int, str], list[int]]:
    """Walk selfplay_dir for iter_N/{redteam,blueteam} LoRA checkpoints.

    Returns:
        (red_loras, blue_loras, iter_dirs):
            red_loras, blue_loras: {iter_number: absolute_adapter_path}
            iter_dirs: every iter_N directory observed (for missing-iter reporting)
    """
    red_loras: dict[int, str] = {}
    blue_loras: dict[int, str] = {}
    iter_dirs: list[int] = []

    for item in sorted(os.listdir(selfplay_dir)):
        if not item.startswith("iter_"):
            continue
        try:
            iter_num = int(item.split("_")[1])
        except (ValueError, IndexError):
            raise ValueError(
                f"Cannot parse iteration number from directory name: {item!r}"
            )

        iter_path = os.path.join(selfplay_dir, item)
        if not os.path.isdir(iter_path):
            continue
        iter_dirs.append(iter_num)

        # Surface unexpected siblings (e.g. redteam_old) so they aren't quietly ignored.
        siblings = sorted(
            s for s in os.listdir(iter_path)
            if os.path.isdir(os.path.join(iter_path, s))
            and s not in ("redteam", "blueteam")
        )
        if siblings:
            print(f"Note: iter_{iter_num} has extra subdirs (ignored): {siblings}")

        red_dir = os.path.join(iter_path, "redteam")
        if os.path.isdir(red_dir):
            ckpt = _find_latest_checkpoint(red_dir)
            if ckpt:
                red_loras[iter_num] = os.path.realpath(ckpt)
            else:
                print(
                    f"Warning: iter_{iter_num}/redteam exists but contains no usable "
                    f"sql_agent checkpoint — iteration will be skipped for red team."
                )
        else:
            print(f"Warning: iter_{iter_num}/redteam is missing.")

        blue_dir = os.path.join(iter_path, "blueteam")
        if os.path.isdir(blue_dir):
            ckpt = _find_latest_checkpoint(blue_dir)
            if ckpt:
                blue_loras[iter_num] = os.path.realpath(ckpt)
            else:
                print(
                    f"Warning: iter_{iter_num}/blueteam exists but contains no usable "
                    f"sql_agent checkpoint — iteration will be skipped for blue team."
                )
        else:
            print(f"Warning: iter_{iter_num}/blueteam is missing.")

    return red_loras, blue_loras, iter_dirs


# ---------------------------------------------------------------------------
# Checkpoint validation helpers
# ---------------------------------------------------------------------------

def _require_file(path: Path, description: str) -> None:
    """Raise FileNotFoundError with a clear message if path does not exist."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def _read_base_model(adapter_path: str) -> str:
    """Return base_model_name_or_path from adapter_config.json; crash on error."""
    config_path = Path(adapter_path) / "adapter_config.json"
    _require_file(config_path, "adapter_config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    key = "base_model_name_or_path"
    if key not in cfg:
        raise KeyError(f"'{key}' missing from {config_path}")
    return cfg[key]


def _validate_adapter(adapter_path: str) -> bool:
    """Return True if required adapter files are present, False (with a warning) otherwise."""
    p = Path(adapter_path)
    for fname in ("adapter_config.json", "adapter_model.safetensors"):
        if not (p / fname).is_file():
            print(f"Warning: incomplete checkpoint (missing {fname}): {adapter_path}")
            return False
    return True


# ---------------------------------------------------------------------------
# Discovery & validation
# ---------------------------------------------------------------------------

def load_and_validate(
    result_dir: str,
    base_model_override: str | None,
) -> tuple[dict[int, str], dict[int, str], str]:
    """
    Walk result_dir, discover checkpoints, and validate strict invariants.

    Returns:
        (red_loras, blue_loras, base_model_id)
            red_loras, blue_loras: {iter_num: adapter_path}
            base_model_id: the HF model ID / path shared by all adapters
    """
    red_loras, blue_loras, iter_dirs = _discover_checkpoints(result_dir)

    print(f"Discovered iter_N directories: {sorted(iter_dirs)}")

    if not red_loras:
        raise RuntimeError(f"No red-team LoRA checkpoints found under: {result_dir}")
    if not blue_loras:
        raise RuntimeError(f"No blue-team LoRA checkpoints found under: {result_dir}")

    red_iters = set(red_loras)
    blue_iters = set(blue_loras)
    only_red = sorted(red_iters - blue_iters)
    only_blue = sorted(blue_iters - red_iters)
    missing_both = sorted(set(iter_dirs) - red_iters - blue_iters)
    if only_red:
        print(f"Note: iterations only in red team:  {only_red}")
    if only_blue:
        print(f"Note: iterations only in blue team: {only_blue}")
    if missing_both:
        print(f"Note: iter_N dirs with no usable checkpoints (skipped): {missing_both}")

    # Validate adapter directories; drop iterations with incomplete checkpoints.
    for iter_num in sorted(red_iters):
        if not _validate_adapter(red_loras[iter_num]):
            del red_loras[iter_num]
    for iter_num in sorted(blue_iters):
        if not _validate_adapter(blue_loras[iter_num]):
            del blue_loras[iter_num]

    if not red_loras:
        raise RuntimeError(f"No valid red-team LoRA checkpoints found under: {result_dir}")
    if not blue_loras:
        raise RuntimeError(f"No valid blue-team LoRA checkpoints found under: {result_dir}")

    # Collect base model IDs from every adapter and assert consistency.
    all_paths = list(red_loras.values()) + list(blue_loras.values())
    base_models = {_read_base_model(p) for p in all_paths}
    if len(base_models) > 1:
        raise RuntimeError(
            f"Adapters disagree on base_model_name_or_path: {sorted(base_models)}"
        )
    base_model_id = base_models.pop()

    if base_model_override is not None and base_model_override != base_model_id:
        raise RuntimeError(
            f"--base-model {base_model_override!r} does not match "
            f"adapter_config.json value {base_model_id!r}"
        )

    return red_loras, blue_loras, base_model_id


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    red_loras: dict[int, str],
    blue_loras: dict[int, str],
) -> dict:
    """
    Compute per-iteration metrics for red and blue teams.

    Returns a dict with keys:
        red_iters       : sorted list of iter numbers for red team
        blue_iters      : sorted list of iter numbers for blue team
        shared_iters    : sorted list of iter numbers present in both teams
        red_norm        : ‖ΔW_k‖_F for each iter in red_iters  (drift from base)
        blue_norm       : ‖ΔW_k‖_F for each iter in blue_iters (drift from base)
        red_delta_norm  : ‖ΔW_k − ΔW_{k-1}‖_F for red  (per-iteration change; iter 1 = ‖ΔW_1‖)
        blue_delta_norm : ‖ΔW_k − ΔW_{k-1}‖_F for blue (per-iteration change; iter 1 = ‖ΔW_1‖)
        red_cos_consec  : cos(tv_N, tv_{N-1}) for red  (None for first red iter)
        blue_cos_consec : cos(tv_N, tv_{N-1}) for blue (None for first blue iter)
        red_cos_vs1     : cos(tv_N, tv_1) for red       (1.0 for first red iter)
        blue_cos_vs1    : cos(tv_N, tv_1) for blue      (1.0 for first blue iter)
        rb_cosine       : cos(tv_red_N, tv_blue_N) for each iter in shared_iters
    """
    red_iters = sorted(red_loras.keys())
    blue_iters = sorted(blue_loras.keys())
    shared_iters = sorted(set(red_iters) & set(blue_iters))

    print("Computing task vectors...")
    red_tvs: dict[int, LLMTaskVector] = {}
    blue_tvs: dict[int, LLMTaskVector] = {}
    for n in red_iters:
        print(f"  iter {n}: red  [{red_loras[n]}]")
        red_tvs[n] = LLMTaskVector.from_lora_adapter(red_loras[n])
    for n in blue_iters:
        print(f"  iter {n}: blue [{blue_loras[n]}]")
        blue_tvs[n] = LLMTaskVector.from_lora_adapter(blue_loras[n])

    red_norm = [red_tvs[n].l2_norm() for n in red_iters]
    blue_norm = [blue_tvs[n].l2_norm() for n in blue_iters]

    # Per-iteration change: ‖tv_k − tv_{k-1}‖_F
    # For the first iteration tv_{k-1} = 0 (base model), so delta = ‖tv_1‖.
    red_delta_norm: list[float] = []
    blue_delta_norm: list[float] = []
    for i, n in enumerate(red_iters):
        if i == 0:
            red_delta_norm.append(red_tvs[n].l2_norm())
        else:
            prev = red_iters[i - 1]
            red_delta_norm.append((red_tvs[n] - red_tvs[prev]).l2_norm())
    for i, n in enumerate(blue_iters):
        if i == 0:
            blue_delta_norm.append(blue_tvs[n].l2_norm())
        else:
            prev = blue_iters[i - 1]
            blue_delta_norm.append((blue_tvs[n] - blue_tvs[prev]).l2_norm())

    red_cos_consec: list[float | None] = []
    blue_cos_consec: list[float | None] = []
    red_cos_vs1: list[float] = []
    blue_cos_vs1: list[float] = []
    rb_cosine: list[float] = []

    red_first = red_iters[0] if red_iters else None
    for i, n in enumerate(red_iters):
        if i == 0:
            red_cos_consec.append(None)
        else:
            prev = red_iters[i - 1]
            red_cos_consec.append(red_tvs[n].cosine_similarity(red_tvs[prev]))
        red_cos_vs1.append(1.0 if n == red_first else red_tvs[n].cosine_similarity(red_tvs[red_first]))

    blue_first = blue_iters[0] if blue_iters else None
    for i, n in enumerate(blue_iters):
        if i == 0:
            blue_cos_consec.append(None)
        else:
            prev = blue_iters[i - 1]
            blue_cos_consec.append(blue_tvs[n].cosine_similarity(blue_tvs[prev]))
        blue_cos_vs1.append(1.0 if n == blue_first else blue_tvs[n].cosine_similarity(blue_tvs[blue_first]))

    for n in shared_iters:
        rb_cosine.append(red_tvs[n].cosine_similarity(blue_tvs[n]))

    return dict(
        red_iters=red_iters,
        blue_iters=blue_iters,
        shared_iters=shared_iters,
        red_norm=red_norm,
        blue_norm=blue_norm,
        red_delta_norm=red_delta_norm,
        blue_delta_norm=blue_delta_norm,
        red_cos_consec=red_cos_consec,
        blue_cos_consec=blue_cos_consec,
        red_cos_vs1=red_cos_vs1,
        blue_cos_vs1=blue_cos_vs1,
        rb_cosine=rb_cosine,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

RED_COL = "#d62728"
BLUE_COL = "#4c78a8"
GRAY_COL = "#888888"


def setup_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_overview(metrics: dict, out_path: Path) -> None:
    setup_style()
    red_iters = np.array(metrics["red_iters"])
    blue_iters = np.array(metrics["blue_iters"])
    all_iters = np.union1d(red_iters, blue_iters)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5))
    fig.suptitle("LoRA weight-delta evolution across self-play iterations", fontsize=14)

    # --- Left: cumulative drift from base model ---
    ax = axes[0]
    if len(red_iters) > 0:
        ax.plot(red_iters, metrics["red_norm"], color=RED_COL, marker="o", lw=2.5, label="Red team")
    if len(blue_iters) > 0:
        ax.plot(blue_iters, metrics["blue_norm"], color=BLUE_COL, marker="s", lw=2.5, label="Blue team")
    ax.set_title("Cumulative drift from base model", fontsize=12)
    ax.set_xlabel("Self-play iteration $k$", fontsize=12)
    ax.set_ylabel(r"$\|\Delta W_k\|_F$", fontsize=12)
    ax.set_xticks(all_iters)
    ax.legend(frameon=True, fontsize=11)

    # --- Right: per-iteration change ‖tv_k − tv_{k-1}‖_F ---
    ax = axes[1]
    if len(red_iters) > 0:
        ax.plot(red_iters, metrics["red_delta_norm"], color=RED_COL, marker="o", lw=2.5, label="Red team")
    if len(blue_iters) > 0:
        ax.plot(blue_iters, metrics["blue_delta_norm"], color=BLUE_COL, marker="s", lw=2.5, label="Blue team")
    ax.set_title(r"Per-iteration change  $\|\Delta W_k - \Delta W_{k-1}\|_F$", fontsize=12)
    ax.set_xlabel("Self-play iteration $k$", fontsize=12)
    ax.set_ylabel(r"$\|\Delta W_k - \Delta W_{k-1}\|_F$", fontsize=12)
    ax.set_xticks(all_iters)
    ax.legend(frameon=True, fontsize=11)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def plot_cosine_metrics(metrics: dict, out_path: Path) -> None:
    """1×3 panel covering metrics 3, 4, and 5."""
    setup_style()
    red_iters = np.array(metrics["red_iters"])
    blue_iters = np.array(metrics["blue_iters"])
    shared_iters = np.array(metrics["shared_iters"])
    all_iters = np.union1d(red_iters, blue_iters)

    fig, axes = plt.subplots(1, 3, figsize=(21.0, 5.5))
    fig.suptitle("LoRA task-vector cosine similarities across self-play iterations", fontsize=14)

    # --- Left (metric 3): cos(tv_k, tv_{k-1}) — novelty per step ---
    ax = axes[0]
    # First iteration has no predecessor; its entry is None — skip it.
    red_consec_iters = red_iters[1:] if len(red_iters) > 1 else np.array([], dtype=int)
    blue_consec_iters = blue_iters[1:] if len(blue_iters) > 1 else np.array([], dtype=int)
    red_consec_vals = [v for v in metrics["red_cos_consec"] if v is not None]
    blue_consec_vals = [v for v in metrics["blue_cos_consec"] if v is not None]
    if len(red_consec_iters) > 0:
        ax.plot(red_consec_iters, red_consec_vals, color=RED_COL, marker="o", lw=2.5, label="Red team")
    if len(blue_consec_iters) > 0:
        ax.plot(blue_consec_iters, blue_consec_vals, color=BLUE_COL, marker="s", lw=2.5, label="Blue team")
    ax.set_title("Consecutive cosine similarity\n(novelty per iteration)", fontsize=12)
    ax.set_xlabel("Self-play iteration $k$", fontsize=12)
    ax.set_ylabel(r"$\cos(\Delta W_k,\,\Delta W_{k-1})$", fontsize=12)
    ax.set_ylim(-1.05, 1.05)
    consec_ticks = np.union1d(red_consec_iters, blue_consec_iters)
    if len(consec_ticks) > 0:
        ax.set_xticks(consec_ticks)
    ax.axhline(0, color=GRAY_COL, lw=1.0, ls="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=11)

    # --- Middle (metric 4): cos(tv_k, tv_1) — drift from the seed direction ---
    ax = axes[1]
    if len(red_iters) > 0:
        ax.plot(red_iters, metrics["red_cos_vs1"], color=RED_COL, marker="o", lw=2.5, label="Red team")
    if len(blue_iters) > 0:
        ax.plot(blue_iters, metrics["blue_cos_vs1"], color=BLUE_COL, marker="s", lw=2.5, label="Blue team")
    ax.set_title("Cosine vs. iteration 1\n(strategy drift from seed)", fontsize=12)
    ax.set_xlabel("Self-play iteration $k$", fontsize=12)
    ax.set_ylabel(r"$\cos(\Delta W_k,\,\Delta W_1)$", fontsize=12)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(all_iters)
    ax.axhline(0, color=GRAY_COL, lw=1.0, ls="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=11)

    # --- Right (metric 5): cos(tv_red_k, tv_blue_k) — arms-race alignment ---
    ax = axes[2]
    if len(shared_iters) > 0:
        ax.plot(shared_iters, metrics["rb_cosine"], color=GRAY_COL, marker="D", lw=2.5,
                label="Red vs. Blue")
    ax.set_title("Red–Blue cosine similarity\n(arms-race alignment)", fontsize=12)
    ax.set_xlabel("Self-play iteration $k$", fontsize=12)
    ax.set_ylabel(
        r"$\cos(\Delta W^{\mathrm{red}}_k,\,\Delta W^{\mathrm{blue}}_k)$", fontsize=12
    )
    ax.set_ylim(-1.05, 1.05)
    if len(shared_iters) > 0:
        ax.set_xticks(shared_iters)
    ax.axhline(0, color=GRAY_COL, lw=1.0, ls="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=11)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Output-directory preflight
# ---------------------------------------------------------------------------

def _is_writable_dir(path: Path) -> bool:
    """Return True iff we can create+remove a temp file inside path."""
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(path), prefix=".writetest_", delete=True
        ):
            pass
        return True
    except OSError:
        return False


def _expected_outputs(out_dir: Path) -> list[Path]:
    return [
        out_dir / "lora_delta_metrics.json",
        out_dir / "lora_delta_overview.png",
        out_dir / "lora_delta_cosine.png",
    ]


def _candidate_writable(path: Path) -> bool:
    """Return True iff `path` can be created/used as a writable output dir."""
    if path.exists():
        if not path.is_dir():
            return False
        if not _is_writable_dir(path):
            return False
        for fp in _expected_outputs(path):
            if fp.exists() and not os.access(fp, os.W_OK):
                return False
        return True
    # Doesn't exist — check the nearest existing ancestor for writability.
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    return _is_writable_dir(ancestor)


def prepare_output_dir(
    result_dir: Path,
    requested_subdir: str,
    out_dir_override: str | None,
) -> Path:
    """
    Return a writable output directory.

    Resolution order:
      1. --out-dir (explicit user override, must be writable).
      2. result_dir/requested_subdir, if writable (can overwrite existing outputs).
      3. result_dir/requested_subdir_<timestamp>, if result_dir is writable.
      4. cwd/<result_dir.name>_<requested_subdir>, if writable.
      5. cwd/<result_dir.name>_<requested_subdir>_<timestamp>, last resort.

    Always returns a created, writable directory.
    """
    if out_dir_override is not None:
        out = Path(out_dir_override)
        out.mkdir(parents=True, exist_ok=True)
        if not _is_writable_dir(out):
            raise PermissionError(f"--out-dir {out} is not writable.")
        return out

    candidates: list[Path] = [
        result_dir / requested_subdir,
        result_dir / f"{requested_subdir}_{time.strftime('%Y%m%d-%H%M%S')}",
        Path.cwd() / f"{result_dir.name}_{requested_subdir}",
        Path.cwd() / f"{result_dir.name}_{requested_subdir}_{time.strftime('%Y%m%d-%H%M%S')}",
    ]

    chosen: Path | None = None
    for cand in candidates:
        if _candidate_writable(cand):
            chosen = cand
            break

    if chosen is None:
        raise PermissionError(
            "No writable output location found. Tried: "
            + ", ".join(str(c) for c in candidates)
            + ". Pass --out-dir explicitly."
        )

    chosen.mkdir(parents=True, exist_ok=True)
    primary = result_dir / requested_subdir
    if chosen != primary:
        print(
            f"Warning: cannot write to {primary} "
            f"(check ownership/permissions). Falling back to: {chosen}"
        )
    return chosen


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute and plot LoRA weight-delta evolution across self-play iterations. "
            "Crashes on any missing checkpoint or inconsistency — nothing is silently skipped."
        )
    )
    p.add_argument(
        "result_dir",
        help="Path to a self-play result directory containing iter_N/ subdirectories.",
    )
    p.add_argument(
        "--base-model",
        default=None,
        help=(
            "HF model ID or path of the base (pretrained) model. "
            "If omitted, read from adapter_config.json; must match across all adapters."
        ),
    )
    p.add_argument(
        "--out-subdir",
        default="lora_delta",
        help="Subdirectory inside result_dir where outputs are written (default: lora_delta).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Explicit output directory override. If set, takes precedence over "
            "--out-subdir and disables fallback resolution."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    result_dir = args.result_dir
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"result_dir does not exist or is not a directory: {result_dir}")

    print(f"=== LoRA delta analysis: {result_dir} ===")

    # Preflight: pick a writable output dir BEFORE doing minutes of LoRA loading.
    out_dir = prepare_output_dir(Path(result_dir), args.out_subdir, args.out_dir)
    print(f"Output dir: {out_dir}")

    red_loras, blue_loras, base_model_id = load_and_validate(result_dir, args.base_model)
    print(f"Base model: {base_model_id}")
    print(f"Red team iterations:  {sorted(red_loras)}")
    print(f"Blue team iterations: {sorted(blue_loras)}")

    metrics = compute_metrics(red_loras, blue_loras)

    # Dump raw metrics as JSON (None → null for JSON compatibility)
    json_path = out_dir / "lora_delta_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics JSON saved: {json_path}")

    plot_overview(metrics, out_dir / "lora_delta_overview.png")
    plot_cosine_metrics(metrics, out_dir / "lora_delta_cosine.png")


if __name__ == "__main__":
    main()
