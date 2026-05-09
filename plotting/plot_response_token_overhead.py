"""
Defender response-token overhead — deployment-cost proxy (C.3, Pillar 1).

Tokenizes `blue_action` across all attack and benign turns, comparing the
trained-defender (canonical cross_eval, 8×8) against the prompt-only baseline
(cross_eval_baseline, 8×1). Output:

  - Distribution of `blue_action` token counts per turn, faceted by
    {attack, benign} × {trained, baseline}
  - Median, p90, p99 token counts per facet
  - Estimated wall-clock latency overhead at a representative throughput

Addresses the likely reviewer concern: "does the safer defender just emit
longer/cosmetic responses, defeating its own utility?". Pure CPU; uses
GPT-4-style tiktoken encoding as an architecture-neutral token proxy
(or falls back to whitespace word-count if tiktoken not available).

Sidecar: figures/response_token_overhead.json
PNG    : figures/response_token_overhead.png

CLI:
    python plotting/plot_response_token_overhead.py \\
        --results results-<ID>[:Label] [--out-dir figures/]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._data import apply_paper_style, parse_results_arg
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import apply_paper_style, parse_results_arg

apply_paper_style()

DESCRIPTION = (
    "Defender response token-count distribution (deployment-cost proxy). "
    "Compares trained vs baseline `blue_action` lengths across attack and benign turns."
)

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def _tokenize(text: str) -> int:
        if not text:
            return 0
        return len(_ENC.encode(text, disallowed_special=()))
    TOKENIZER_LABEL = "tiktoken cl100k_base"
except ImportError:  # pragma: no cover
    def _tokenize(text: str) -> int:
        return len((text or "").split())
    TOKENIZER_LABEL = "whitespace fallback"


def _scan_pairings(base: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {"attack": [], "benign": []}
    if not base.exists():
        return out
    for pairing_dir in sorted(base.glob("red_*_blue_*")):
        jsonl = pairing_dir / "reward_debug.jsonl"
        if not jsonl.exists():
            continue
        with open(jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                turn_type = r.get("turn_type")
                if turn_type not in ("attack", "benign"):
                    continue
                blue_action = r.get("blue_action", "") or ""
                tok = _tokenize(blue_action)
                out[turn_type].append(tok)
    return out


def _summary(toks: list[int]) -> dict:
    if not toks:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0}
    arr = np.array(toks, dtype=int)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 1),
        "median": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }


def collect_overhead(
    selfplay_dir: str,
    trained_subdir: str = "cross_eval",
    baseline_subdir: str = "cross_eval_baseline",
) -> dict:
    base = Path(selfplay_dir)
    trained = _scan_pairings(base / trained_subdir / "pairings")
    baseline = _scan_pairings(base / baseline_subdir / "pairings")
    return {
        "description": DESCRIPTION,
        "tokenizer": TOKENIZER_LABEL,
        "trained": {
            "attack": _summary(trained["attack"]),
            "benign": _summary(trained["benign"]),
        },
        "baseline": {
            "attack": _summary(baseline["attack"]),
            "benign": _summary(baseline["benign"]),
        },
        "overhead_pct_at_median": {
            "attack": _overhead_pct(
                _summary(trained["attack"])["median"],
                _summary(baseline["attack"])["median"],
            ),
            "benign": _overhead_pct(
                _summary(trained["benign"])["median"],
                _summary(baseline["benign"])["median"],
            ),
        },
        "_raw": {  # kept for plotting; not for paper citation
            "trained_attack": trained["attack"],
            "trained_benign": trained["benign"],
            "baseline_attack": baseline["attack"],
            "baseline_benign": baseline["benign"],
        },
    }


def _overhead_pct(trained_val: float, baseline_val: float) -> float:
    if baseline_val <= 0:
        return float("nan")
    return round((trained_val - baseline_val) / baseline_val * 100, 1)


def render_overhead(out: dict, out_path: Path) -> None:
    raw = out["_raw"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))

    # Token-count CDF, attack and benign
    for ax, key in zip(axes, ["attack", "benign"]):
        for label, color, data in [
            ("Trained", "#1f77b4", raw[f"trained_{key}"]),
            ("Baseline", "#d62728", raw[f"baseline_{key}"]),
        ]:
            if not data:
                continue
            arr = np.sort(np.array(data, dtype=int))
            cdf = np.arange(1, len(arr) + 1) / len(arr)
            ax.plot(arr, cdf, label=f"{label} (n={len(arr):,})", color=color, lw=1.6)
        ax.set_xlabel("blue_action token count")
        ax.set_ylabel("CDF")
        ax.set_title(f"{key.capitalize()} turns")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    # Annotate
    sub_t = out["trained"]
    sub_b = out["baseline"]
    overhead = out["overhead_pct_at_median"]
    fig.suptitle(
        f"Defender response token overhead — {out['tokenizer']}\n"
        f"attack median: trained={sub_t['attack']['median']} vs baseline={sub_b['attack']['median']} "
        f"(overhead {overhead['attack']:+.1f}%); "
        f"benign median: trained={sub_t['benign']['median']} vs baseline={sub_b['benign']['median']} "
        f"(overhead {overhead['benign']:+.1f}%)",
        fontsize=10, y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--trained-subdir", default="cross_eval")
    p.add_argument("--baseline-subdir", default="cross_eval_baseline")
    p.add_argument("--out-dir", default="figures/")
    args = p.parse_args()

    results = parse_results_arg(args.results)
    label, selfplay_dir = results[0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out = collect_overhead(selfplay_dir, args.trained_subdir, args.baseline_subdir)
    png_path = out_dir / "response_token_overhead.png"
    render_overhead(out, png_path)

    # Write sidecar without the _raw arrays (large)
    sidecar_path = out_dir / "response_token_overhead.json"
    sidecar = {k: v for k, v in out.items() if k != "_raw"}
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[response_token_overhead] saved {sidecar_path}")
    print(f"[response_token_overhead] saved {png_path}")
    print()
    for cond in ("trained", "baseline"):
        for kind in ("attack", "benign"):
            s = out[cond][kind]
            print(f"  {cond:9s} {kind:7s}  n={s['n']:6,d}  mean={s['mean']:6.1f}  "
                  f"median={s['median']:5d}  p90={s['p90']:5d}  p99={s['p99']:5d}")
    print(f"  attack-median overhead: {out['overhead_pct_at_median']['attack']:+.1f}%")
    print(f"  benign-median overhead: {out['overhead_pct_at_median']['benign']:+.1f}%")


if __name__ == "__main__":
    main()
