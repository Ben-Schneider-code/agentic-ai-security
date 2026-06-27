"""
LoRA task-vector (ΔW = α/r · B@A) evolution across self-play iterations.

Three separate output files:
  lora_drift.png  — cumulative drift from base model (‖ΔW_k‖_F)
  lora_delta.png  — per-iteration change (‖ΔW_k − ΔW_{k-1}‖_F)
  lora_cosine.png — 1×3 cosine panel:
                      (a) consecutive cosine similarity (novelty per step)
                      (b) cosine vs. iteration 1 (strategy drift)
                      (c) red–blue cosine similarity (arms-race alignment)

Ported and refactored from compare_lora.py.  Imports LLMTaskVector from the
repo root via sys.path so it can be run with or without the package installed.

Can be run standalone:
    python plotting/plot_lora_diversity.py --results results-<ID>
Or imported:
    from plotting.plot_lora_diversity import plot_lora_diversity, DESCRIPTION
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Repo root (parent of plotting/) for importing llm_task_vector
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from ._data import (
        apply_paper_style, parse_results_arg,
        RED_COL, BLUE_COL, GRAY_COL, RUN_COLORS,
        FIG_SIZE_SINGLE, FIG_SIZE_1x3,
    )
except ImportError:
    from plotting._data import (
        apply_paper_style, parse_results_arg,
        RED_COL, BLUE_COL, GRAY_COL, RUN_COLORS,
        FIG_SIZE_SINGLE, FIG_SIZE_1x3,
    )

apply_paper_style()

DESCRIPTION = (
    "LoRA task-vector (ΔW = α/r · B@A) evolution across self-play iterations. "
    "lora_drift.png: cumulative Frobenius norm drift from the base model per iteration. "
    "lora_delta.png: per-iteration change ‖ΔW_k − ΔW_{k-1}‖_F. "
    "lora_cosine.png: (a) consecutive cosine similarity (novelty), "
    "(b) cosine vs. iteration 1 (strategy drift), "
    "(c) red–blue cosine similarity (arms-race alignment)."
)


# ---------------------------------------------------------------------------
# Checkpoint discovery & validation (ported from compare_lora.py)
# ---------------------------------------------------------------------------

def discover_lora_checkpoints(selfplay_dir: str) -> tuple[dict[int, str], dict[int, str]]:
    """
    Walk selfplay_dir for iter_N/{redteam,blueteam} LoRA adapter checkpoints.

    Returns:
        (red_loras, blue_loras): {iter_number: absolute_adapter_path}
    """
    red_loras: dict[int, str] = {}
    blue_loras: dict[int, str] = {}

    for item in sorted(os.listdir(selfplay_dir)):
        if not item.startswith("iter_"):
            continue
        try:
            iter_num = int(item.split("_")[1])
        except (ValueError, IndexError):
            continue

        iter_path = os.path.join(selfplay_dir, item)

        for team_name, team_dict in (("redteam", red_loras), ("blueteam", blue_loras)):
            team_dir = os.path.join(iter_path, team_name)
            if not os.path.isdir(team_dir):
                continue
            ckpt = _find_latest_checkpoint(team_dir)
            if ckpt:
                team_dict[iter_num] = os.path.realpath(ckpt)

    return red_loras, blue_loras


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
                    # Handle nested sql_agent/sql_agent produced by older PEFT saves
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


def validate_adapter(adapter_path: str) -> bool:
    """Return True if required adapter files are present, False with a warning otherwise."""
    p = Path(adapter_path)
    for fname in ("adapter_config.json", "adapter_model.safetensors"):
        if not (p / fname).is_file():
            print(f"Warning: incomplete checkpoint (missing {fname}): {adapter_path}",
                  file=sys.stderr)
            return False
    return True


def _read_base_model(adapter_path: str) -> str:
    cfg = Path(adapter_path) / "adapter_config.json"
    with open(cfg) as f:
        return json.load(f)["base_model_name_or_path"]


# ---------------------------------------------------------------------------
# Metric computation (ported from compare_lora.compute_metrics)
# ---------------------------------------------------------------------------

def compute_lora_metrics(
    red_loras: dict[int, str],
    blue_loras: dict[int, str],
) -> dict:
    """
    Compute per-iteration LoRA task-vector metrics.

    Returns a dict with keys:
        red_iters, blue_iters, shared_iters,
        red_norm, blue_norm,
        red_delta_norm, blue_delta_norm,
        red_cos_consec, blue_cos_consec,
        red_cos_vs1, blue_cos_vs1,
        rb_cosine
    """
    from llm_task_vector import LLMTaskVector  # noqa: E402

    red_iters  = sorted(red_loras.keys())
    blue_iters = sorted(blue_loras.keys())
    shared_iters = sorted(set(red_iters) & set(blue_iters))

    print("Computing task vectors…")
    red_tvs:  dict[int, LLMTaskVector] = {}
    blue_tvs: dict[int, LLMTaskVector] = {}
    for n in red_iters:
        print(f"  iter {n}: red  [{red_loras[n]}]")
        red_tvs[n]  = LLMTaskVector.from_lora_adapter(red_loras[n])
    for n in blue_iters:
        print(f"  iter {n}: blue [{blue_loras[n]}]")
        blue_tvs[n] = LLMTaskVector.from_lora_adapter(blue_loras[n])

    red_norm  = [red_tvs[n].l2_norm()  for n in red_iters]
    blue_norm = [blue_tvs[n].l2_norm() for n in blue_iters]

    # Per-iteration change: ‖tv_k − tv_{k-1}‖_F  (iter 1 compared to base = 0)
    red_delta_norm:  list[float] = []
    blue_delta_norm: list[float] = []
    for i, n in enumerate(red_iters):
        if i == 0:
            red_delta_norm.append(red_tvs[n].l2_norm())
        else:
            red_delta_norm.append((red_tvs[n] - red_tvs[red_iters[i - 1]]).l2_norm())
    for i, n in enumerate(blue_iters):
        if i == 0:
            blue_delta_norm.append(blue_tvs[n].l2_norm())
        else:
            blue_delta_norm.append((blue_tvs[n] - blue_tvs[blue_iters[i - 1]]).l2_norm())

    # Cosine similarity between consecutive iterations
    red_cos_consec:  list[float | None] = []
    blue_cos_consec: list[float | None] = []
    for i, n in enumerate(red_iters):
        if i == 0:
            red_cos_consec.append(None)
        else:
            red_cos_consec.append(red_tvs[n].cosine_similarity(red_tvs[red_iters[i - 1]]))

    for i, n in enumerate(blue_iters):
        if i == 0:
            blue_cos_consec.append(None)
        else:
            blue_cos_consec.append(blue_tvs[n].cosine_similarity(blue_tvs[blue_iters[i - 1]]))

    # Cosine vs. first iteration
    red_first  = red_iters[0]  if red_iters  else None
    blue_first = blue_iters[0] if blue_iters else None
    red_cos_vs1  = [
        1.0 if n == red_first  else red_tvs[n].cosine_similarity(red_tvs[red_first])
        for n in red_iters
    ]
    blue_cos_vs1 = [
        1.0 if n == blue_first else blue_tvs[n].cosine_similarity(blue_tvs[blue_first])
        for n in blue_iters
    ]

    # Red–blue cross-team cosine per shared iteration
    rb_cosine = [
        red_tvs[n].cosine_similarity(blue_tvs[n]) for n in shared_iters
    ]

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


def compute_lora(
    results: list[tuple[str, str]],
    base_model: str | None = None,
    **kwargs,
) -> dict:
    """Pure compute export of the LoRA task-vector metrics for a single run.

    Resolution order (no matplotlib / no file writes):
      1. Prefer the precomputed cache ``<selfplay_dir>/lora_delta/lora_delta_metrics.json``.
         If present, return ``json.load(...)`` merged with ``{"source": ...}``.
         This path NEVER imports torch.
      2. Otherwise recompute via ``compute_lora_metrics`` (the torch + adapters path),
         returning its dict merged with ``{"source": "recomputed"}``.

    The schema (``red_iters``, ``blue_iters``, ``shared_iters``, ``red_norm``,
    ``blue_norm``, ``red_delta_norm``, ``blue_delta_norm``, ``red_cos_consec``,
    ``blue_cos_consec``, ``red_cos_vs1``, ``blue_cos_vs1``, ``rb_cosine``) is identical
    in both paths, so the cached json and the recomputed dict are interchangeable.

    Args:
        results:    ``[(label, selfplay_dir), ...]``; only the first run is used.
        base_model: optional base-model id; cross-checked against adapter_config.json
                    on the recompute path only.
        **kwargs:   accepted and ignored (uniform compute-fn signature).

    Returns:
        ``{}`` when no run is available, when there are no valid checkpoints, or when
        torch / adapters are unavailable on the recompute path (never raises on those).
    """
    if not results:
        return {}
    selfplay_dir = results[0][1]

    cache = Path(selfplay_dir) / "lora_delta" / "lora_delta_metrics.json"
    if cache.is_file():
        try:
            data = json.load(open(cache))
        except (json.JSONDecodeError, OSError):
            data = None
        if isinstance(data, dict):
            return {**data, "source": "lora_delta_metrics.json"}

    # Recompute path (requires torch + adapters via llm_task_vector). Wrap the whole
    # path so any missing optional dep -> {} rather than a crash.
    try:
        red_loras, blue_loras = discover_lora_checkpoints(selfplay_dir)
        red_loras = {n: p for n, p in red_loras.items() if validate_adapter(p)}
        blue_loras = {n: p for n, p in blue_loras.items() if validate_adapter(p)}
        if not red_loras and not blue_loras:
            return {}
        if base_model is not None:
            all_paths = list(red_loras.values()) + list(blue_loras.values())
            detected = {_read_base_model(p) for p in all_paths}
            if base_model not in detected:
                raise RuntimeError(
                    f"--base-model {base_model!r} not among adapter values {sorted(detected)}"
                )
        metrics = compute_lora_metrics(red_loras, blue_loras)
    except (ImportError, ModuleNotFoundError):
        return {}
    except Exception as e:  # pragma: no cover - defensive; never crash the exporter
        print(f"  [compute_lora] recompute failed: {e}", file=sys.stderr)
        return {}
    return {**metrics, "source": "recomputed"}


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def _setup_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_lora_drift(
    all_metrics: list[tuple[str, dict]],
    out_path: Path,
) -> None:
    """
    Plot 1: Cumulative drift ‖ΔW_k‖_F per iteration for red and blue teams.

    all_metrics: [(label, metrics_dict), ...]
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)

    colors = RUN_COLORS * (len(all_metrics) // len(RUN_COLORS) + 1)

    for run_idx, (label, m) in enumerate(all_metrics):
        color = colors[run_idx]
        red_iters  = np.array(m["red_iters"])
        blue_iters = np.array(m["blue_iters"])

        if len(red_iters) > 0:
            lbl = f"{label} — red"  if len(all_metrics) > 1 else "Red team"
            ax.plot(red_iters,  m["red_norm"],  color=RED_COL  if len(all_metrics) == 1 else color,
                    marker="o", lw=2.5, label=lbl)
        if len(blue_iters) > 0:
            lbl = f"{label} — blue" if len(all_metrics) > 1 else "Blue team"
            ax.plot(blue_iters, m["blue_norm"], color=BLUE_COL if len(all_metrics) == 1 else _lighten(color),
                    marker="s", lw=2.5, label=lbl, linestyle="--")

    ax.set_title("Cumulative Drift from Base Model", fontsize=12)
    ax.set_xlabel("Self-play iteration $k$", fontsize=12)
    ax.set_ylabel(r"$\|\Delta W_k\|_F$", fontsize=12)
    all_iters = _all_iter_ticks(all_metrics)
    if all_iters:
        ax.set_xticks(all_iters)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def plot_lora_delta(
    all_metrics: list[tuple[str, dict]],
    out_path: Path,
) -> None:
    """
    Plot 2: Per-iteration change ‖ΔW_k − ΔW_{k-1}‖_F.

    all_metrics: [(label, metrics_dict), ...]
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)

    colors = RUN_COLORS * (len(all_metrics) // len(RUN_COLORS) + 1)

    for run_idx, (label, m) in enumerate(all_metrics):
        color = colors[run_idx]
        red_iters  = np.array(m["red_iters"])
        blue_iters = np.array(m["blue_iters"])

        if len(red_iters) > 0:
            lbl = f"{label} — red"  if len(all_metrics) > 1 else "Red team"
            ax.plot(red_iters,  m["red_delta_norm"],  color=RED_COL  if len(all_metrics) == 1 else color,
                    marker="o", lw=2.5, label=lbl)
        if len(blue_iters) > 0:
            lbl = f"{label} — blue" if len(all_metrics) > 1 else "Blue team"
            ax.plot(blue_iters, m["blue_delta_norm"], color=BLUE_COL if len(all_metrics) == 1 else _lighten(color),
                    marker="s", lw=2.5, label=lbl, linestyle="--")

    ax.set_title(r"Per-Iteration Change $\|\Delta W_k - \Delta W_{k-1}\|_F$", fontsize=12)
    ax.set_xlabel("Self-play iteration $k$", fontsize=12)
    ax.set_ylabel(r"$\|\Delta W_k - \Delta W_{k-1}\|_F$", fontsize=12)
    all_iters = _all_iter_ticks(all_metrics)
    if all_iters:
        ax.set_xticks(all_iters)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def plot_lora_cosine(
    all_metrics: list[tuple[str, dict]],
    out_path: Path,
) -> None:
    """
    Plot 3: 1×3 cosine panel:
      (a) consecutive cosine similarity (novelty per step)
      (b) cosine vs. iteration 1 (strategy drift)
      (c) red–blue cosine similarity (arms-race alignment)

    all_metrics: [(label, metrics_dict), ...]
    """
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE_1x3)
    colors = RUN_COLORS * (len(all_metrics) // len(RUN_COLORS) + 1)

    for run_idx, (label, m) in enumerate(all_metrics):
        color = colors[run_idx]
        red_iters   = np.array(m["red_iters"])
        blue_iters  = np.array(m["blue_iters"])
        shared_iters = np.array(m["shared_iters"])

        # --- (a) Consecutive cosine similarity ---
        ax = axes[0]
        r_consec_iters = red_iters[1:]  if len(red_iters)  > 1 else np.array([], dtype=int)
        b_consec_iters = blue_iters[1:] if len(blue_iters) > 1 else np.array([], dtype=int)
        r_consec_vals  = [v for v in m["red_cos_consec"]  if v is not None]
        b_consec_vals  = [v for v in m["blue_cos_consec"] if v is not None]

        if len(r_consec_iters) > 0:
            lbl = f"{label} — red"  if len(all_metrics) > 1 else "Red team"
            ax.plot(r_consec_iters, r_consec_vals, color=RED_COL  if len(all_metrics) == 1 else color,
                    marker="o", lw=2.5, label=lbl)
        if len(b_consec_iters) > 0:
            lbl = f"{label} — blue" if len(all_metrics) > 1 else "Blue team"
            ax.plot(b_consec_iters, b_consec_vals, color=BLUE_COL if len(all_metrics) == 1 else _lighten(color),
                    marker="s", lw=2.5, label=lbl, linestyle="--")

        # --- (b) Cosine vs. iteration 1 ---
        ax = axes[1]
        if len(red_iters) > 0:
            lbl = f"{label} — red"  if len(all_metrics) > 1 else "Red team"
            ax.plot(red_iters,  m["red_cos_vs1"],  color=RED_COL  if len(all_metrics) == 1 else color,
                    marker="o", lw=2.5, label=lbl)
        if len(blue_iters) > 0:
            lbl = f"{label} — blue" if len(all_metrics) > 1 else "Blue team"
            ax.plot(blue_iters, m["blue_cos_vs1"], color=BLUE_COL if len(all_metrics) == 1 else _lighten(color),
                    marker="s", lw=2.5, label=lbl, linestyle="--")

        # --- (c) Red–blue cosine (arms-race alignment) ---
        ax = axes[2]
        if len(shared_iters) > 0:
            lbl = label if len(all_metrics) > 1 else "Red vs. Blue"
            ax.plot(shared_iters, m["rb_cosine"], color=GRAY_COL if len(all_metrics) == 1 else color,
                    marker="D", lw=2.5, label=lbl)

    # Formatting for each panel
    all_iters = _all_iter_ticks(all_metrics)

    for ax, title, ylabel, ylabel_latex in [
        (
            axes[0],
            "Consecutive Cosine Similarity\n(novelty per iteration)",
            "cos",
            r"$\cos(\Delta W_k,\,\Delta W_{k-1})$",
        ),
        (
            axes[1],
            "Cosine vs. Iteration 1\n(strategy drift from seed)",
            "cos",
            r"$\cos(\Delta W_k,\,\Delta W_1)$",
        ),
        (
            axes[2],
            "Red–Blue Cosine Similarity\n(arms-race alignment)",
            "cos",
            r"$\cos(\Delta W^{\rm red}_k,\,\Delta W^{\rm blue}_k)$",
        ),
    ]:
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Self-play iteration $k$", fontsize=11)
        ax.set_ylabel(ylabel_latex, fontsize=11)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color=GRAY_COL, lw=1.0, ls="--", alpha=0.5)
        if all_iters:
            ax.set_xticks(all_iters)
        ax.legend(frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Top-level function
# ---------------------------------------------------------------------------

def plot_lora_diversity(
    results: list[tuple[str, str]],
    out_dir: str | Path,
    base_model: str | None = None,
    expected_iters: int | None = None,
) -> tuple[Path, Path, Path]:
    """
    Compute LoRA metrics and write three separate PNG files.

    Args:
        results:        [(label, selfplay_dir), ...]
        out_dir:        Directory where lora_drift.png, lora_delta.png,
                        lora_cosine.png are written.
        base_model:     Override base model ID (cross-checked against adapter_config.json).
        expected_iters: If set, fail loudly when either side has fewer than N valid
                        adapters. Prevents silent stale regens over a truncated run.

    Returns:
        (drift_path, delta_path, cosine_path)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[tuple[str, dict]] = []

    for label, selfplay_dir in results:
        red_loras, blue_loras = discover_lora_checkpoints(selfplay_dir)

        # Validate adapters; drop incomplete entries
        red_loras  = {n: p for n, p in red_loras.items()  if validate_adapter(p)}
        blue_loras = {n: p for n, p in blue_loras.items() if validate_adapter(p)}

        if not red_loras and not blue_loras:
            print(f"  [plot_lora_diversity] No valid adapters found in {selfplay_dir}",
                  file=sys.stderr)
            continue

        if expected_iters is not None:
            short = []
            if len(red_loras) < expected_iters:
                short.append(f"red={len(red_loras)}/{expected_iters}")
            if len(blue_loras) < expected_iters:
                short.append(f"blue={len(blue_loras)}/{expected_iters}")
            if short:
                raise RuntimeError(
                    f"[plot_lora_diversity] {selfplay_dir}: expected {expected_iters} iters "
                    f"but got {', '.join(short)}. Use --expected-iters to match actual run "
                    f"length, or check that iter_* checkpoints are present."
                )

        # Verify base model consistency if requested
        all_paths = list(red_loras.values()) + list(blue_loras.values())
        if all_paths:
            base_models = {_read_base_model(p) for p in all_paths}
            if len(base_models) > 1:
                raise RuntimeError(
                    f"Adapters in {selfplay_dir} disagree on base model: {sorted(base_models)}"
                )
            detected = base_models.pop()
            if base_model is not None and base_model != detected:
                raise RuntimeError(
                    f"--base-model {base_model!r} does not match adapter value {detected!r}"
                )

        metrics = compute_lora_metrics(red_loras, blue_loras)
        all_metrics.append((label, metrics))

    if not all_metrics:
        print("  [plot_lora_diversity] Nothing to plot — no valid checkpoints found.",
              file=sys.stderr)
        empty = out_dir / "lora_drift.png"
        return empty, out_dir / "lora_delta.png", out_dir / "lora_cosine.png"

    drift_path  = out_dir / "lora_drift.png"
    delta_path  = out_dir / "lora_delta.png"
    cosine_path = out_dir / "lora_cosine.png"

    plot_lora_drift(all_metrics, drift_path)
    plot_lora_delta(all_metrics, delta_path)
    plot_lora_cosine(all_metrics, cosine_path)

    return drift_path, delta_path, cosine_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_iter_ticks(all_metrics: list[tuple[str, dict]]) -> list[int]:
    """Collect the union of all iteration numbers across runs."""
    ticks: set[int] = set()
    for _, m in all_metrics:
        ticks.update(m.get("red_iters", []))
        ticks.update(m.get("blue_iters", []))
    return sorted(ticks)


def _lighten(hex_color: str, factor: float = 0.45) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r + (255 - r) * factor),
        int(g + (255 - g) * factor),
        int(b + (255 - b) * factor),
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot LoRA weight-delta evolution across self-play iterations."
    )
    parser.add_argument(
        "--results", nargs="+", required=True, metavar="DIR[:LABEL]",
        help="One or more selfplay result dirs, optionally with :Label suffix.",
    )
    parser.add_argument("--out-dir", default="figures/", metavar="DIR",
                        help="Directory where the three PNG files are written.")
    parser.add_argument("--base-model", default=None, metavar="MODEL_ID",
                        help="Override base model ID (checked against adapter_config.json).")
    parser.add_argument("--expected-iters", type=int, default=None, metavar="N",
                        help="Fail if either red or blue has fewer than N valid adapter "
                             "iters. Prevents silent stale regens over a truncated run.")
    args = parser.parse_args()

    results = parse_results_arg(args.results)
    drift, delta, cosine = plot_lora_diversity(
        results, args.out_dir, args.base_model, args.expected_iters,
    )
    print(f"[{DESCRIPTION}]")
    print(f"  → {drift}")
    print(f"  → {delta}")
    print(f"  → {cosine}")


if __name__ == "__main__":
    main()
