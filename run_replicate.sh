#!/bin/bash
# Run one full replicate end-to-end on a single 2-GPU pair (Docker-friendly).
#
#   1. Self-play   (./run_selfplay.sh, forwarding all flags verbatim)
#   2. Cross-eval  (./run_cross_eval.sh on the resulting run dir)
#   3. Benign-eval (./util/run_benign_eval.sh on the resulting run dir)
#
# Designed for set-and-forget execution inside one Docker container. Drop the
# command into the container, walk away, return to a results-<ID>/ tree that
# contains cross_eval/ and benign_eval/ subdirs ready for the plotting step.
#
# All flags are forwarded to run_selfplay.sh. The wrapper sniffs --base-model,
# --redteam-gpu, and --blueteam-gpu so it can wire the downstream evals to the
# same GPU pair and model.
#
# Usage:
#   ./run_replicate.sh \
#       --replicate-seed 0 \
#       --num-env-steps 200 --num-iterations 4 --horizon 5 \
#       --vanilla-size 120 --bordercase-size 20 --honeypot-type rowcol \
#       --redteam-gpu 0 --blueteam-gpu 1

set -e
set -o pipefail

BASE_MODEL="Snowflake/Arctic-Text2SQL-R1-7B"
RED_GPU=0
BLUE_GPU=1
REPLICATE_SEED=0
RESULTS_ID_OVERRIDE=""

# Sniff (don't consume) base-model / GPU / seed args for the downstream eval calls.
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
        --base-model)     BASE_MODEL="${args[$((i+1))]}" ;;
        --redteam-gpu)    RED_GPU="${args[$((i+1))]}" ;;
        --blueteam-gpu)   BLUE_GPU="${args[$((i+1))]}" ;;
        --replicate-seed) REPLICATE_SEED="${args[$((i+1))]}" ;;
        --results-id)     RESULTS_ID_OVERRIDE="${args[$((i+1))]}" ;;
    esac
done

echo "========================================"
echo "Agentic AI Security: End-to-End Replicate"
echo "========================================"
echo "Base model:  $BASE_MODEL"
echo "Red GPU:     $RED_GPU"
echo "Blue GPU:    $BLUE_GPU"
echo "Forwarded:   $*"
echo "========================================"

LOG="replicate_$$.selfplay.log"

# Phase 1 — self-play
echo "[replicate] Phase 1/3: self-play (log mirrored to $LOG)"
./run_selfplay.sh "${args[@]}" 2>&1 | tee "$LOG"

# Parse SELFPLAY_ID from the standard banner line printed at run_selfplay.sh:171.
SELFPLAY_ID=$(grep -m1 "^Selfplay Run ID:" "$LOG" | awk '{print $NF}')
[[ -z "$SELFPLAY_ID" ]] && { echo "[replicate] FAIL: could not parse SELFPLAY_ID from $LOG"; exit 1; }
RESULTS_DIR="results-${SELFPLAY_ID}"
[[ ! -d "$RESULTS_DIR" ]] && { echo "[replicate] FAIL: $RESULTS_DIR missing on disk"; exit 1; }
echo "[replicate] Self-play complete. Run dir: $RESULTS_DIR"

# Phase 2 — cross-eval (dual-GPU; same pair, same base model)
echo "[replicate] Phase 2/3: cross-eval"
./run_cross_eval.sh \
    --selfplay-dir "$RESULTS_DIR" \
    --base-model   "$BASE_MODEL" \
    --red-gpu  "$RED_GPU" \
    --blue-gpu "$BLUE_GPU" \
    --episodes    80 \
    --concurrency 32 \
    --seed       "$REPLICATE_SEED" \
    --resume

# Phase 3 — benign-eval (single GPU; reuse the blue GPU)
echo "[replicate] Phase 3/3: benign-eval"
./util/run_benign_eval.sh \
    --results-dir "$RESULTS_DIR" \
    --base-model  "$BASE_MODEL" \
    --gpu         "$BLUE_GPU" \
    --episodes    100 \
    --seed        "$REPLICATE_SEED" \
    --resume

echo "========================================"
echo "[replicate] complete. Artifacts under $RESULTS_DIR/"
echo "  - iter_*/{red,blue}team/        (training)"
echo "  - cross_eval/cross_eval_results.json"
echo "  - benign_eval/benign_eval_results.json"
echo "========================================"
