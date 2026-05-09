"""
LoRA geometry — honest version.

Two panels:
  (a) Per-team consecutive cosine cos(ΔW_k, ΔW_{k-1}). This IS real signal:
      as both teams approach a plateau, per-iteration updates become nearly
      co-linear with the previous iteration's update, so novel search
      directions are running out.

  (b) Red-vs-Blue cosine similarity as stored in `lora_delta_metrics.json`
      (mean of per-layer cosines from llm_task_vector.LLMTaskVector.cosine_similarity,
      per_layer=True). On rank-low LoRAs optimising asymmetric reward
      signals, the NULL expectation is ~0 with a very narrow spread around
      zero — so a value near 0 carries no information by itself. We plot the
      observed values against that null band and state explicitly: the
      panel is a *sanity check* ("blue is not covertly aligning to red"),
      not a positive robustness claim. A non-zero positive value would be
      interpreted as a warning; the opposite (anti-alignment) would be
      surprising but would also not imply robustness.

Source: <selfplay_dir>/lora_delta/lora_delta_metrics.json.

NOTE: the committed metrics JSON currently covers iters 1–4 only. To extend
to 1–7, re-run `compare_lora.py <selfplay_dir> --base-model Snowflake/Arctic-Text2SQL-R1-7B
--out-subdir lora_delta` (no GPU; just CPU state-dict diffs of the LoRA
adapters).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._data import (
        apply_paper_style, parse_results_arg, write_sidecar,
        RED_COL, BLUE_COL, GRAY_COL, FIG_SIZE_1x2,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style, parse_results_arg, write_sidecar,
        RED_COL, BLUE_COL, GRAY_COL, FIG_SIZE_1x2,
    )

apply_paper_style()

DESCRIPTION = (
    "LoRA geometry — honest version. Left: per-team consecutive cosine rising "
    "to ~0.98 is a REAL signal of per-iteration novelty decay. Right: Red-vs-"
    "Blue per-layer-mean cosine from lora_delta_metrics.json; plotted against "
    "the NULL band expected for independent low-rank LoRAs (values near 0 "
    "carry no positive information). Used only as a sanity check that blue "
    "is not covertly aligning to red."
)


def _null_band_half_width(num_params_per_layer_mean: float = 4096 * 4096,
                           n_layers: int = 128,
                           z: float = 2.576) -> float:
    """
    Approximate half-width of the 99 % null band for the mean of per-layer
    cosines between two independent random unit vectors of dimension d.
    Per-layer cos ~ N(0, 1/d); mean over L layers ~ N(0, 1/(L·d)).
    """
    per_layer_sd = 1.0 / math.sqrt(num_params_per_layer_mean)
    mean_sd = per_layer_sd / math.sqrt(n_layers)
    return z * mean_sd


def plot_lora_orthogonality(
    results: list[tuple[str, str]],
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    if len(results) > 1:
        print("[plot_lora_orthogonality] Multiple runs given; using first only.",
              file=sys.stderr)
    label, selfplay_dir = results[0]

    lora_path = Path(selfplay_dir) / "lora_delta" / "lora_delta_metrics.json"
    if not lora_path.is_file():
        print(f"[plot_lora_orthogonality] Missing {lora_path}", file=sys.stderr)
        fig, ax = plt.subplots(figsize=FIG_SIZE_1x2)
        ax.text(0.5, 0.5, "lora_delta_metrics.json not found",
                ha="center", va="center", transform=ax.transAxes, color=GRAY_COL)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    with open(lora_path) as f:
        m = json.load(f)

    shared = m.get("shared_iters", [])
    rb = m.get("rb_cosine", [])
    red_consec = m.get("red_cos_consec", [])
    blue_consec = m.get("blue_cos_consec", [])
    red_iters = m.get("red_iters", [])
    blue_iters = m.get("blue_iters", [])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)

    # ---- (a) Per-team consecutive cosine (real signal) ----
    def _valid(xs, ys):
        out_x, out_y = [], []
        for x, y in zip(xs, ys):
            if y is None:
                continue
            out_x.append(x)
            out_y.append(y)
        return out_x, out_y

    rx, ry = _valid(red_iters, red_consec)
    bx, by = _valid(blue_iters, blue_consec)
    if rx:
        ax_a.plot(rx, ry, "-o", color=RED_COL, linewidth=1.8, markersize=7,
                  label="Red team")
    if bx:
        ax_a.plot(bx, by, "--s", color=BLUE_COL, linewidth=1.8, markersize=7,
                  label="Blue team")
    ax_a.axhline(1.0, color=GRAY_COL, linewidth=1.0, linestyle=":")
    ax_a.axhline(0.0, color=GRAY_COL, linewidth=0.5, linestyle="-", alpha=0.5)
    ax_a.set_xticks(sorted(set(rx + bx)) if (rx or bx) else [])
    ax_a.set_ylim(0, 1.05)
    ax_a.set_xlabel("Self-play iteration $k$")
    ax_a.set_ylabel(r"cos$(\Delta W_k, \Delta W_{k-1})$")
    ax_a.set_title(
        "Per-team consecutive cosine  (REAL signal)\n"
        r"→1: updates repeating $\Rightarrow$ novelty exhausted"
    )
    ax_a.legend(loc="lower right", fontsize=10, frameon=True)
    ax_a.grid(True, axis="y", alpha=0.4)

    # ---- (b) Red-Blue cosine — plotted honestly with null band ----
    null_half = _null_band_half_width()
    # Zoom y-axis to the range of observed data plus several null widths.
    if rb:
        obs_max = max(abs(v) for v in rb)
        y_abs_max = max(obs_max * 3.0, null_half * 5.0, 1e-3)
    else:
        y_abs_max = 1e-3
    ax_b.axhspan(-null_half, null_half, color="#d9d9d9", alpha=0.55,
                 label=f"99 % null band (±{null_half:.1e})")
    ax_b.axhline(0.0, color=GRAY_COL, linewidth=0.8, linestyle=":")
    if shared and rb:
        ax_b.plot(shared, rb, "-D", color="#4d4d4d",
                  linewidth=1.8, markersize=7,
                  label=r"observed cos$(\Delta W^{red}_k, \Delta W^{blue}_k)$")
        for x, v in zip(shared, rb):
            ax_b.text(x, v + y_abs_max * 0.06, f"{v:+.1e}",
                      ha="center", fontsize=7, color="#555")
    ax_b.set_ylim(-y_abs_max, y_abs_max)
    ax_b.set_xticks(shared)
    ax_b.set_xlabel("Self-play iteration $k$")
    ax_b.set_ylabel("Mean per-layer cosine")
    ax_b.set_title(
        "Red–Blue cosine  (SANITY CHECK only)\n"
        "null is ~0 for independent low-rank LoRAs — no positive claim"
    )
    ax_b.legend(loc="upper right", fontsize=8, frameon=True)
    ax_b.grid(True, axis="y", alpha=0.4)
    # Axis-style: show scientific notation
    ax_b.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)

    fig.suptitle(
        f"LoRA geometry — {label}\n"
        "Per-team novelty decay is the substantive finding; red–blue cosine is a null check",
        fontsize=12,
    )
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    # Stderr diagnostics
    print(f"[plot_lora_orthogonality] source={lora_path}", file=sys.stderr)
    print(f"  null_band_half_width ≈ {null_half:.3e}  (99% CI, "
          "assuming per-layer cosine ~N(0, 1/d))", file=sys.stderr)
    for k, v in zip(shared, rb):
        ratio = abs(v) / null_half if null_half > 0 else 0.0
        print(f"  iter={k} rb_cosine={v:+.3e}  |v|/null = {ratio:.2f}σ-equivalent",
              file=sys.stderr)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, metavar="DIR[:LABEL]")
    ap.add_argument("--out", default="figures/lora_orthogonality.png")
    args = ap.parse_args()
    results = parse_results_arg(args.results)
    out = plot_lora_orthogonality(results, args.out)
    write_sidecar(out, DESCRIPTION, results)
    print(f"[{DESCRIPTION[:100]}...]\n  → {out}")


if __name__ == "__main__":
    main()
