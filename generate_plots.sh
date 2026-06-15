#!/bin/bash

# Convenience script to generate all the plots in the repo

# No self-play, fine-tuning (figures_longggg)
RESULTS_DIRS=(
    # "results-20260610-0349-u4zr8" # broken because text2sql was too heavily guardrailed
    "results-20260613-2213-vx8k7"
)
# Short run, identical training seeds, col-only (figures_replicates_baseline)
# RESULTS_DIRS=(
#     "results-20260603-2040-ofpbb"
#     "results-20260605-0204-9m6bf"
# )
# # Long run, not identical training seeds, col-only (figures_replicates_baseline_long)
# RESULTS_DIRS=(
#     "results-20260526-2344-wgtx3"
# )
# # Short runs, not identical training seeds, col-only
# RESULTS_DIRS=(
#     "results-20260522-1427-5c25m"
#     "results-20260522-1243-nqf8a"
#     "results-20260524-1408-73lv5"
#     "results-20260524-1408-snfb7"
#     "results-20260524-1409-6pnq4"
# )
# RESULTS_DIRS=(
#     "results-20260511-2244-js96d"
#     "results-20260511-2245-84jao"
#     "results-20260511-2245-owd3g"
# )

OUT_DIR="figures_longggg/"


echo "Generating plots for results directories:"
for dir in "${RESULTS_DIRS[@]}"; do
    echo "  - $dir"
done

# 1. Build arguments for plot_paper_figures.py (attaching :rep0, :rep1, etc.)
PAPER_FIG_ARGS=()
for i in "${!RESULTS_DIRS[@]}"; do
    PAPER_FIG_ARGS+=("${RESULTS_DIRS[$i]}:rep${i}")
done

source .venv/bin/activate

python plotting/plot_paper_figures.py \
    --results "${PAPER_FIG_ARGS[@]}" \
    --cross-eval-subdir cross_eval --out-dir $OUT_DIR --ci=false

# # 2. Run plot_results.py for each directory
# for dir in "${RESULTS_DIRS[@]}"; do
#     ./util/plot_results.py "$dir"
# done

# 3. Run honeypot_stats.py for each directory (commented out)
# for dir in "${RESULTS_DIRS[@]}"; do
#     python ./util/honeypot_stats.py --results-dir "$dir"
# done

# 4. Build multiple --results-dir arguments for compare_honeypot_stats.py
COMPARE_ARGS=()
for dir in "${RESULTS_DIRS[@]}"; do
    COMPARE_ARGS+=("--results-dir" "$dir")
done

python ./util/compare_honeypot_stats.py "${COMPARE_ARGS[@]}"
