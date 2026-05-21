#!/bin/bash

# Convenience script to generate all the plots in the repo

RESULTS_DIRS=(
    "results-20260511-2244-js96d"
    "results-20260511-2245-84jao"
    "results-20260511-2245-owd3g"
)
echo "Generating plots for results directories:"
for dir in "${RESULTS_DIRS[@]}"; do
    echo "  - $dir"
done

# 1. Build arguments for plot_paper_figures.py (attaching :rep0, :rep1, etc.)
PAPER_FIG_ARGS=()
for i in "${!RESULTS_DIRS[@]}"; do
    PAPER_FIG_ARGS+=("${RESULTS_DIRS[$i]}:rep${i}")
done

python plotting/plot_paper_figures.py \
    --results "${PAPER_FIG_ARGS[@]}" \
    --cross-eval-subdir cross_eval --out-dir figures_replicates_baseline/ --ci=false

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
