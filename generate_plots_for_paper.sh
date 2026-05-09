#!/bin/bash
# Regenerate every `% cmd:`-tagged figure in methodology.tex into figures/.
#
# Prereqs: self-play and cross-evaluation have already been run for both base
# models. This script only re-renders plots from the existing artifacts; it
# does NOT launch GPU work.
#
# Usage:
#     ARCTIC_RESULTS_DIR=results-... LLAMA_RESULTS_DIR=results-... ./generate_plots_for_paper.sh
#
# Optional env vars:
#     ABLATIONS_DIR   (default: ablations) — set by run_ablations.sh
#     FIGURES_DIR     (default: figures)    — output tree, in .gitignore
#     SKIP_PER_ITER   (unset)               — skip per-iter red/blue plots (appendix)
#     PY              (default: python)     — interpreter override
#
# Flags:
#     --quick         — read cross-eval from <results>/cross_eval_quick instead
#                       of <results>/cross_eval (matches run_cross_eval.sh --quick)

# TODO: This is being subsumed by plotting/plot_paper_figures.py

set -euo pipefail

USE_QUICK_CROSS_EVAL=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) USE_QUICK_CROSS_EVAL=true; shift ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ "$USE_QUICK_CROSS_EVAL" == "true" ]]; then
    CROSS_EVAL_SUBDIR="cross_eval_quick"
else
    CROSS_EVAL_SUBDIR="cross_eval"
fi

: "${ARCTIC_RESULTS_DIR:?set to the Arctic self-play results dir (Snowflake/Arctic-Text2SQL-R1-7B)}"
: "${LLAMA_RESULTS_DIR:?set to the Llama self-play results dir (meta-llama/Llama-3.1-8B-Instruct)}"

ABLATIONS_DIR="${ABLATIONS_DIR:-ablations}"
FIGURES_DIR="${FIGURES_DIR:-figures}"
PY="${PY:-python}"

# Agent-vs-Human artifacts (produced by run_human_baseline.sh + util/jailbreak_baseline.py).
AUDIT_DIR="${AUDIT_DIR:-data/human_attack_audit}"
HUMAN_TABLES_CONFIG="${HUMAN_TABLES_CONFIG:-configs/human_baseline_tables.json}"
RL_VS_HUMAN_CONFIG="${RL_VS_HUMAN_CONFIG:-configs/rl_vs_human.json}"
DATASET="${DATASET:-new_jailbreaks.txt}"

ARCTIC_BASE="Snowflake/Arctic-Text2SQL-R1-7B"
LLAMA_BASE="meta-llama/Llama-3.1-8B-Instruct"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTIL_DIR="$SCRIPT_DIR/util"

for d in "$ARCTIC_RESULTS_DIR" "$LLAMA_RESULTS_DIR"; do
    if [[ ! -d "$d" ]]; then
        echo "ERROR: $d not found" >&2
        exit 2
    fi
done

mkdir -p "$FIGURES_DIR"/{arctic,llama,ablations,jailbreak_outcomes}
for tag in arctic llama; do
    mkdir -p "$FIGURES_DIR/$tag"/{cross_eval,redteam_per_iter,blueteam_per_iter}
done

run_model_figures() {
    local tag="$1" results_dir="$2" base_model="$3"
    local out="$FIGURES_DIR/$tag"
    echo ""
    echo "===== [$tag] ← $results_dir → $out ====="

    # 1. Self-play relative-strength trajectory
    echo "[$tag] selfplay_relative_strength"
    "$PY" "$UTIL_DIR/plot_selfplay_results.py" "$results_dir" --output-dir "$out"

    # 2. LoRA delta overview + cosine (compare_lora writes into <result>/<out-subdir>)
    echo "[$tag] lora_delta"
    local lora_tmp="$results_dir/_lora_paper"
    rm -rf "$lora_tmp"
    "$PY" "$SCRIPT_DIR/compare_lora.py" "$results_dir" \
        --base-model "$base_model" \
        --out-subdir "_lora_paper"
    cp "$lora_tmp"/lora_delta_overview.png "$out/lora_delta_overview.png"
    cp "$lora_tmp"/lora_delta_cosine.png   "$out/lora_delta_cosine.png"
    cp "$lora_tmp"/lora_delta_metrics.json "$out/lora_delta_metrics.json"
    rm -rf "$lora_tmp"

    # 3. Compute-cost analysis
    echo "[$tag] compute_cost_analysis"
    "$PY" "$UTIL_DIR/compute_cost_analysis.py" \
        --selfplay-dir "$results_dir" \
        --json-out "$out/compute_cost_analysis.json" \
        --plot-out "$out/compute_cost_analysis.png"

    # 4. Diversity diagnostics trend
    echo "[$tag] diversity_trend"
    local div_tmp="$out/_diversity_tmp"
    "$PY" "$UTIL_DIR/diversity_diagnostics.py" "$results_dir" \
        --output-dir "$div_tmp" --last-n 100 --n-gram 4
    if [[ -f "$div_tmp/trend.png" ]]; then
        cp "$div_tmp/trend.png" "$out/diversity_trend.png"
    fi
    rm -rf "$div_tmp"

    # 5. Collapse monitor (benign denial rate per iter + summary)
    echo "[$tag] collapse_monitor"
    local col_tmp="$out/_collapse_tmp"
    "$PY" "$UTIL_DIR/collapse_monitor.py" "$results_dir" \
        --output-dir "$col_tmp" --window 30 --threshold 0.7 || true
    if [[ -f "$col_tmp/summary.json" ]]; then
        cp "$col_tmp/summary.json" "$out/collapse_monitor.json"
    fi
    local last_collapse_png
    last_collapse_png=$(ls -1 "$col_tmp"/iter_*_benign_denial.png 2>/dev/null | sort -V | tail -n1 || true)
    if [[ -n "$last_collapse_png" ]]; then
        cp "$last_collapse_png" "$out/collapse_monitor.png"
    fi
    rm -rf "$col_tmp"

    # 6. Per-style PUD trend
    echo "[$tag] per_style_pud_trend"
    local pst_tmp="$out/_pstyle_tmp"
    "$PY" "$UTIL_DIR/per_style_pud_trend.py" "$results_dir" --output-dir "$pst_tmp" || true
    if [[ -f "$pst_tmp/trend.png" ]]; then
        cp "$pst_tmp/trend.png" "$out/per_style_pud_trend.png"
    fi
    rm -rf "$pst_tmp"

    # 7. Cross-evaluation (expects run_cross_eval.sh to have run)
    local ce="$results_dir/$CROSS_EVAL_SUBDIR"
    if [[ -d "$ce" ]]; then
        echo "[$tag] cross_eval ($CROSS_EVAL_SUBDIR)"
        "$PY" "$UTIL_DIR/plot_cross_eval.py" "$ce" --output-dir "$out/cross_eval"
    else
        local hint=""
        [[ "$USE_QUICK_CROSS_EVAL" == "true" ]] && hint=" --quick"
        echo "[$tag] WARN: $ce missing — cross-eval figures skipped (run run_cross_eval.sh${hint} first)"
    fi

    # 8. Per-iter red/blue training plots (appendix — skip with SKIP_PER_ITER=1)
    if [[ -z "${SKIP_PER_ITER:-}" ]]; then
        echo "[$tag] per-iter red/blue training plots"
        for it in "$results_dir"/iter_*; do
            [[ -d "$it" ]] || continue
            local n
            n=$(basename "$it" | sed 's/iter_//')

            local rt bt
            rt=$(find "$it/redteam"  -maxdepth 6 -type d -name 'run_*' 2>/dev/null | sort | head -n1 || true)
            bt=$(find "$it/blueteam" -maxdepth 6 -type d -name 'run_*' 2>/dev/null | sort | head -n1 || true)

            if [[ -n "$rt" ]]; then
                "$PY" "$UTIL_DIR/plot_redteam_results.py" "$rt" \
                    --output-dir "$out/redteam_per_iter" || true
                for ext in png pdf; do
                    local src="$out/redteam_per_iter/training_results_detailed.$ext"
                    local dst="$out/redteam_per_iter/iter_${n}_training_results_detailed.$ext"
                    [[ -f "$src" ]] && mv "$src" "$dst"
                done
            fi

            if [[ -n "$bt" ]]; then
                "$PY" "$UTIL_DIR/plot_blueteam_results.py" "$bt" \
                    --output-dir "$out/blueteam_per_iter" || true
                for ext in png pdf; do
                    local src="$out/blueteam_per_iter/blueteam_training_results.$ext"
                    local dst="$out/blueteam_per_iter/iter_${n}_blueteam_training_results.$ext"
                    [[ -f "$src" ]] && mv "$src" "$dst"
                done
            fi
        done
    else
        echo "[$tag] per-iter plots skipped (SKIP_PER_ITER set)"
    fi
}

run_model_figures arctic "$ARCTIC_RESULTS_DIR" "$ARCTIC_BASE"
run_model_figures llama  "$LLAMA_RESULTS_DIR"  "$LLAMA_BASE"

# ── Ablations ────────────────────────────────────────────────────────────────
if [[ -d "$ABLATIONS_DIR" ]]; then
    echo ""
    echo "===== [ablations] $ABLATIONS_DIR → $FIGURES_DIR/ablations ====="
    "$PY" "$UTIL_DIR/plot_ablations.py" "$ABLATIONS_DIR" \
        --output-dir "$FIGURES_DIR/ablations"
    "$PY" "$UTIL_DIR/ablation_table.py" "$ABLATIONS_DIR" \
        --output "$FIGURES_DIR/ablations/master_table.tex"
else
    echo ""
    echo "WARN: $ABLATIONS_DIR missing — ablation figures and master table skipped."
fi

# ── Agent-vs-Human tables + outcome panels ──────────────────────────────────
# Requires run_human_baseline.sh to have populated $AUDIT_DIR (JSONL reports
# per (tag, variant) cell). Plotting is a pure read of those artifacts.
if [[ -d "$AUDIT_DIR" ]]; then
    echo ""
    echo "===== [human_baseline] $AUDIT_DIR → $FIGURES_DIR/{jailbreak_outcomes,human_baseline} ====="

    mkdir -p "$FIGURES_DIR/jailbreak_outcomes" "$FIGURES_DIR/human_baseline" \
             "$FIGURES_DIR/rl_vs_human"

    "$PY" "$UTIL_DIR/plot_jailbreak_outcomes.py" \
        --audit-dir "$AUDIT_DIR" \
        --out "$FIGURES_DIR/jailbreak_outcomes"

    if [[ -f "$HUMAN_TABLES_CONFIG" ]]; then
        "$PY" "$UTIL_DIR/human_baseline_tables.py" \
            --audit-dir "$AUDIT_DIR" \
            --dataset "$DATASET" \
            --config "$HUMAN_TABLES_CONFIG" \
            --out-dir "$FIGURES_DIR/human_baseline"
    else
        echo "WARN: $HUMAN_TABLES_CONFIG missing — Q1a/Q1b/Q2 tables skipped."
    fi

    "$PY" "$UTIL_DIR/nondeterminism_audit.py" \
        --audit-dir "$AUDIT_DIR" \
        --out "$FIGURES_DIR/human_baseline/nondeterminism_audit.tex"

    if [[ -f "$RL_VS_HUMAN_CONFIG" ]]; then
        "$PY" "$UTIL_DIR/eval_rl_vs_human.py" \
            --config "$RL_VS_HUMAN_CONFIG" \
            --audit-dir "$AUDIT_DIR" \
            --out-json-dir "data/rl_vs_human" \
            --out-tex "$FIGURES_DIR/rl_vs_human/rl_vs_human_table.tex"
    else
        echo "WARN: $RL_VS_HUMAN_CONFIG missing — RL-vs-manual table skipped."
    fi
else
    echo ""
    echo "WARN: $AUDIT_DIR missing — Agent-vs-Human artifacts skipped."
    echo "      Run ./run_human_baseline.sh --manifest <...> first."
fi

cat <<EOF

============================================================
Figure generation complete. See $FIGURES_DIR/.
EOF
