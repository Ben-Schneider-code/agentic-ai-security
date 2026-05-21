#!/bin/bash
# Ablation grid orchestrator (redesign).
#
# Runs the self-play pipeline (run_selfplay.sh) over a set of cells defined by
#   * vanilla benign size  ∈ {120, 60, 0}
#   * bordercase benign    ∈ {20, 10, 0}
#   * honeypot type        ∈ {rowcol, row, col}
#
# Two modes:
#   * --mode axis (default): one-axis-at-a-time + baseline → 7 cells.
#       Baseline:        (120, 20, rowcol)
#       Vanilla sweep:   (60,  20, rowcol), (0,   20, rowcol)
#       Bordercase sweep:(120, 10, rowcol), (120, 0,  rowcol)
#       Honeypot sweep:  (120, 20, row),    (120, 20, col)
#   * --mode full: full factorial 3×3×3 = 27 cells.
#
# Cell-level parallelism is enforced via --max-parallel-cells. Each cell is a
# self-contained run_selfplay.sh invocation pinned to a 2-GPU pair.
#
# Profile presets (consumed by run_selfplay.sh):
#   * smoke    : --num-env-steps 400  --num-iterations 2
#   * default  : --num-env-steps 1600 --num-iterations 2  (≈48 GPU-hr / cell)
#   * extended : --num-env-steps 4000 --num-iterations 3
#
# Usage:
#   ./run_ablation_grid.sh --mode axis --profile default \
#                          --gpu-pairs '0,1;2,3;4,5'
#
#   ./run_ablation_grid.sh --mode full --profile extended --gpu-pairs '0,1'

set -e
set -o pipefail

MODE="axis"
PROFILE="default"
BASE_MODEL="Snowflake/Arctic-Text2SQL-R1-7B"
HORIZON=5
GPU_PAIRS="0,1"
DRY_RUN=false
GRID_TAG=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift ;;
        --profile) PROFILE="$2"; shift ;;
        --base-model) BASE_MODEL="$2"; shift ;;
        --horizon) HORIZON="$2"; shift ;;
        --gpu-pairs) GPU_PAIRS="$2"; shift ;;  # ';'-separated pairs of 'red,blue'
        --grid-tag) GRID_TAG="$2"; shift ;;    # optional dir name suffix
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

case "$PROFILE" in
    smoke)    NUM_ENV_STEPS=400;  NUM_ITERATIONS=2 ;;
    default)  NUM_ENV_STEPS=1600; NUM_ITERATIONS=2 ;;
    extended) NUM_ENV_STEPS=4000; NUM_ITERATIONS=3 ;;
    *) echo "ERROR: --profile must be smoke|default|extended"; exit 1 ;;
esac

case "$MODE" in
    axis|full) ;;
    *) echo "ERROR: --mode must be axis|full"; exit 1 ;;
esac

# Build the cell list as JSON array printed by Python.
CELLS_JSON=$(python3 - <<PY
import json
mode = "${MODE}"

vanilla_levels    = [120, 60, 0]
bordercase_levels = [20, 10, 0]
honeypot_levels   = ["rowcol", "row", "col"]

# Baseline = (max vanilla, max bordercase, full honeypot universe).
baseline = {"vanilla": 120, "bordercase": 20, "honeypot": "rowcol"}

cells = []
seen = set()
def push(c):
    key = (c["vanilla"], c["bordercase"], c["honeypot"])
    if key in seen:
        return
    seen.add(key)
    cells.append(c)

if mode == "axis":
    push(baseline)
    for v in vanilla_levels:
        push({"vanilla": v, "bordercase": baseline["bordercase"], "honeypot": baseline["honeypot"]})
    for b in bordercase_levels:
        push({"vanilla": baseline["vanilla"], "bordercase": b, "honeypot": baseline["honeypot"]})
    for h in honeypot_levels:
        push({"vanilla": baseline["vanilla"], "bordercase": baseline["bordercase"], "honeypot": h})
else:  # full factorial
    for v in vanilla_levels:
        for b in bordercase_levels:
            for h in honeypot_levels:
                push({"vanilla": v, "bordercase": b, "honeypot": h})

print(json.dumps(cells))
PY
)

# Print the plan.
N_CELLS=$(echo "$CELLS_JSON" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")
IFS=';' read -ra PAIRS_ARR <<< "$GPU_PAIRS"
N_PAIRS=${#PAIRS_ARR[@]}

echo "========================================"
echo "Ablation Grid"
echo "========================================"
echo "Mode:              $MODE"
echo "Profile:           $PROFILE  (num_env_steps=$NUM_ENV_STEPS, num_iterations=$NUM_ITERATIONS)"
echo "Cells:             $N_CELLS"
echo "GPU pairs:         $GPU_PAIRS  ($N_PAIRS in parallel)"
echo "Base model:        $BASE_MODEL"
echo "Horizon:           $HORIZON"
echo "Dry run:           $DRY_RUN"
[[ -n "$GRID_TAG" ]] && echo "Grid tag:          $GRID_TAG"

# Output dir for the grid (collects all cell IDs in summary).
ts="$(date +%Y%m%d-%H%M%S)"
GRID_ROOT="ablations/grid_${ts}${GRID_TAG:+_$GRID_TAG}"
mkdir -p "$GRID_ROOT"
GRID_MANIFEST="${GRID_ROOT}/manifest.jsonl"

# Strict bash arrays from cells.
mapfile -t CELL_LINES < <(echo "$CELLS_JSON" | python3 -c "
import sys, json
for c in json.load(sys.stdin):
    print(f\"{c['vanilla']}|{c['bordercase']}|{c['honeypot']}\")
")

run_one_cell() {
    local cell_idx="$1"
    local cell_spec="$2"
    local pair_idx="$3"
    local v b h
    IFS='|' read -r v b h <<< "$cell_spec"
    local pair_spec="${PAIRS_ARR[$pair_idx]}"
    local red_gpu blue_gpu
    IFS=',' read -r red_gpu blue_gpu <<< "$pair_spec"

    local cell_log="${GRID_ROOT}/cell_${cell_idx}_v${v}_b${b}_${h}.log"
    local cmd=(./run_selfplay.sh
        --base-model "$BASE_MODEL"
        --num-iterations "$NUM_ITERATIONS"
        --num-env-steps "$NUM_ENV_STEPS"
        --horizon "$HORIZON"
        --vanilla-size "$v"
        --bordercase-size "$b"
        --honeypot-type "$h"
        --redteam-gpu "$red_gpu"
        --blueteam-gpu "$blue_gpu"
    )

    echo "[grid] cell ${cell_idx}: v=${v} b=${b} h=${h} GPUs(${red_gpu},${blue_gpu}) → ${cell_log}"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[dry-run] ${cmd[*]}"
        return 0
    fi

    # Append to manifest BEFORE running so partial grids still record dispatch.
    python3 - <<PY >> "$GRID_MANIFEST"
import json
print(json.dumps({"cell_idx": ${cell_idx}, "vanilla": ${v}, "bordercase": ${b}, "honeypot": "${h}", "red_gpu": ${red_gpu}, "blue_gpu": ${blue_gpu}, "log": "${cell_log}"}))
PY

    "${cmd[@]}" >"$cell_log" 2>&1
    local rc=$?
    echo "[grid] cell ${cell_idx} exit=${rc}"
    return $rc
}

# Concurrency-limited dispatcher: at most N_PAIRS cells run at once.
declare -a PIDS=()
declare -a PID_TO_PAIR=()
declare -a FREE_PAIRS=()

for ((p=0; p<N_PAIRS; p++)); do
    FREE_PAIRS+=("$p")
done

CELL_IDX=0
for cell_spec in "${CELL_LINES[@]}"; do
    # Wait until a pair is free.
    while [[ ${#FREE_PAIRS[@]} -eq 0 ]]; do
        # Block until any child finishes; then mark its pair free.
        wait -n
        # Reap finished pids and free their pair slots.
        new_pids=()
        new_pid_pairs=()
        for i in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$i]}" 2>/dev/null; then
                new_pids+=("${PIDS[$i]}")
                new_pid_pairs+=("${PID_TO_PAIR[$i]}")
            else
                FREE_PAIRS+=("${PID_TO_PAIR[$i]}")
            fi
        done
        PIDS=("${new_pids[@]}")
        PID_TO_PAIR=("${new_pid_pairs[@]}")
    done

    PAIR_IDX="${FREE_PAIRS[0]}"
    FREE_PAIRS=("${FREE_PAIRS[@]:1}")

    run_one_cell "$CELL_IDX" "$cell_spec" "$PAIR_IDX" &
    PIDS+=("$!")
    PID_TO_PAIR+=("$PAIR_IDX")
    CELL_IDX=$((CELL_IDX + 1))
done

# Wait for all remaining cells to finish.
EXIT_RC=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        EXIT_RC=1
    fi
done

echo ""
echo "========================================"
echo "Ablation Grid Complete (rc=$EXIT_RC)"
echo "========================================"
echo "Manifest:  $GRID_MANIFEST"
echo "Logs in:   $GRID_ROOT"
exit $EXIT_RC
