#!/bin/bash
# Cross-Evaluation: All n² (red_i, blue_j) pairings from a self-play run.
#
# Usage:
#   ./run_cross_eval.sh --selfplay-dir results-20260322-1641-m92p4 \
#       [--episodes 100] [--base-model <model>] \
#       [--red-gpu 0] [--blue-gpu 1] [--horizon 5] [--seed 42] \
#       [--plot-only]
#
# Requires exactly 2 GPUs (80GB each). Both vLLM servers stay running for
# the entire evaluation to keep GPUs allocated on a shared server.

set -e
set -o pipefail

echo "========================================"
echo "Agentic AI Security: Cross-Evaluation"
echo "========================================"

# --- Defaults ---
SELFPLAY_DIR=""
EPISODES=100
BASE_MODEL=""
RED_GPU=0
BLUE_GPU=1
HORIZON=5
SEED=42
PLOT_ONLY=false

# --- Parse arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --selfplay-dir) SELFPLAY_DIR="$2"; shift ;;
        --episodes) EPISODES="$2"; shift ;;
        --base-model) BASE_MODEL="$2"; shift ;;
        --red-gpu) RED_GPU="$2"; shift ;;
        --blue-gpu) BLUE_GPU="$2"; shift ;;
        --horizon) HORIZON="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --plot-only) PLOT_ONLY=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# --- Validate ---
if [[ -z "$SELFPLAY_DIR" ]]; then
    echo "ERROR: --selfplay-dir is required."
    echo "Usage: ./run_cross_eval.sh --selfplay-dir <dir> [--episodes N] [--red-gpu X] [--blue-gpu Y]"
    exit 1
fi
SELFPLAY_DIR="${SELFPLAY_DIR%/}"
if [[ ! -d "$SELFPLAY_DIR" ]]; then
    echo "ERROR: Directory not found: $SELFPLAY_DIR"
    exit 1
fi

OUTPUT_DIR="${SELFPLAY_DIR}/cross_eval"
mkdir -p "$OUTPUT_DIR"

# --- Discover iterations ---
ITERATIONS=()
for dir in "$SELFPLAY_DIR"/iter_*; do
    if [[ -d "$dir" ]]; then
        iter_num=$(basename "$dir" | sed 's/iter_//')
        ITERATIONS+=("$iter_num")
    fi
done
IFS=$'\n' ITERATIONS=($(sort -n <<<"${ITERATIONS[*]}")); unset IFS

if [[ ${#ITERATIONS[@]} -eq 0 ]]; then
    echo "ERROR: No iter_* directories found in $SELFPLAY_DIR"
    exit 1
fi
echo "Found ${#ITERATIONS[@]} iterations: ${ITERATIONS[*]}"

# --- Find latest checkpoint for a team directory ---
find_latest_checkpoint() {
    local dir="$1"
    find "$dir" -name "sql_agent" -type d 2>/dev/null | sort -V | tail -n 1
}

# --- Auto-detect base model from adapter_config.json ---
if [[ -z "$BASE_MODEL" ]]; then
    SAMPLE_ADAPTER=$(find "$SELFPLAY_DIR" -name "adapter_config.json" -type f -print -quit)
    if [[ -z "$SAMPLE_ADAPTER" ]]; then
        echo "ERROR: Could not auto-detect base model. No adapter_config.json found."
        echo "Specify --base-model explicitly."
        exit 1
    fi
    BASE_MODEL=$(python3 -c "import json; print(json.load(open('$SAMPLE_ADAPTER'))['base_model_name_or_path'])")
    echo "Auto-detected base model: $BASE_MODEL"
fi

# --- Discover all LoRA checkpoints ---
declare -A RED_LORAS
declare -A BLUE_LORAS

for iter in "${ITERATIONS[@]}"; do
    red_dir="$SELFPLAY_DIR/iter_${iter}/redteam"
    blue_dir="$SELFPLAY_DIR/iter_${iter}/blueteam"

    if [[ -d "$red_dir" ]]; then
        ckpt=$(find_latest_checkpoint "$red_dir")
        if [[ -n "$ckpt" ]]; then
            RED_LORAS[$iter]=$(realpath "$ckpt")
            echo "  Red iter_${iter}: ${RED_LORAS[$iter]}"
        else
            echo "  WARNING: No red checkpoint found for iter_${iter}, skipping red_${iter}"
        fi
    fi

    if [[ -d "$blue_dir" ]]; then
        ckpt=$(find_latest_checkpoint "$blue_dir")
        if [[ -n "$ckpt" ]]; then
            BLUE_LORAS[$iter]=$(realpath "$ckpt")
            echo "  Blue iter_${iter}: ${BLUE_LORAS[$iter]}"
        else
            echo "  WARNING: No blue checkpoint found for iter_${iter}, skipping blue_${iter}"
        fi
    fi
done

N_RED=${#RED_LORAS[@]}
N_BLUE=${#BLUE_LORAS[@]}
# +1 for base model (iter_0)
TOTAL_PAIRINGS=$(( (N_RED + 1) * (N_BLUE + 1) ))
echo ""
echo "LoRAs found: ${N_RED} red, ${N_BLUE} blue"
echo "Total pairings (including base): ${TOTAL_PAIRINGS}"
echo "Results directory: $OUTPUT_DIR"

# --- Plot-only mode: skip infrastructure, just re-plot ---
if [[ "$PLOT_ONLY" == "true" ]]; then
    echo ""
    echo "[plot-only] Aggregating results..."
    python3 util/cross_evaluate.py \
        --selfplay-dir "$SELFPLAY_DIR" \
        --base-model "$BASE_MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --aggregate-only

    echo "[plot-only] Generating plots..."
    python3 util/plot_cross_eval.py "$OUTPUT_DIR"

    echo ""
    echo "========================================"
    echo "Cross-Evaluation Plots Complete"
    echo "========================================"
    echo "Results: $OUTPUT_DIR"
    exit 0
fi

# ============================================
# Cleanup function — SIGTERM first, SIGKILL only as last resort
# Mirrors run_selfplay.sh pattern to avoid bricking GPUs
# ============================================
RED_VLLM_PID=""
BLUE_VLLM_PID=""

cleanup_vllm() {
    echo ""
    echo "[cleanup] Stopping vLLM servers..."
    for pid_var in RED_VLLM_PID BLUE_VLLM_PID; do
        pid=${!pid_var}
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "[cleanup] Sending SIGTERM to $pid_var (PID=$pid)..."
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Give vLLM time for graceful CUDA context teardown
    sleep 10
    for pid_var in RED_VLLM_PID BLUE_VLLM_PID; do
        pid=${!pid_var}
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "[cleanup] WARNING: $pid_var still alive. Sending SIGKILL — may brick GPU."
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup_vllm EXIT

# ============================================
# 1. Start Postgres + MCP
# ============================================
echo ""
echo "[1/3] Starting Postgres and MCP..."
./script/init.sh
echo "Postgres and MCP ready."

# ============================================
# 2. Start two vLLM servers with all LoRAs
# ============================================
echo ""
echo "[2/3] Starting vLLM servers..."
echo "  Red server:  GPU $RED_GPU, port 8001"
echo "  Blue server: GPU $BLUE_GPU, port 8002"

# Build --lora-modules string for red server
RED_LORA_MODULES=""
for iter in "${!RED_LORAS[@]}"; do
    RED_LORA_MODULES="${RED_LORA_MODULES} red_${iter}=${RED_LORAS[$iter]}"
done

# Build --lora-modules string for blue server
BLUE_LORA_MODULES=""
for iter in "${!BLUE_LORAS[@]}"; do
    BLUE_LORA_MODULES="${BLUE_LORA_MODULES} blue_${iter}=${BLUE_LORAS[$iter]}"
done

if [[ -z "${RED_LORA_MODULES// }" ]]; then
    echo "ERROR: No red LoRA checkpoints found. Cannot start red vLLM server."
    exit 1
fi
if [[ -z "${BLUE_LORA_MODULES// }" ]]; then
    echo "ERROR: No blue LoRA checkpoints found. Cannot start blue vLLM server."
    exit 1
fi

echo "  Red LoRA modules:$RED_LORA_MODULES"
echo "  Blue LoRA modules:$BLUE_LORA_MODULES"

LOG_DIR="/tmp/vllm_logs"
mkdir -p "$LOG_DIR"

# Start red vLLM server
echo "Starting red team vLLM server..."
CUDA_VISIBLE_DEVICES=$RED_GPU python3 -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --port 8001 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --disable-log-requests \
    --enforce-eager \
    --dtype auto \
    --enable-lora \
    --max-lora-rank 64 \
    --max-loras 2 \
    --lora-modules $RED_LORA_MODULES \
    > "$LOG_DIR/crosseval_red.log" 2>&1 &
RED_VLLM_PID=$!
echo "  Red vLLM PID: $RED_VLLM_PID"

# Start blue vLLM server
echo "Starting blue team vLLM server..."
CUDA_VISIBLE_DEVICES=$BLUE_GPU python3 -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --port 8002 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --disable-log-requests \
    --enforce-eager \
    --dtype auto \
    --enable-lora \
    --max-lora-rank 64 \
    --max-loras 2 \
    --lora-modules $BLUE_LORA_MODULES \
    > "$LOG_DIR/crosseval_blue.log" 2>&1 &
BLUE_VLLM_PID=$!
echo "  Blue vLLM PID: $BLUE_VLLM_PID"

# Wait for both servers
echo "Waiting for vLLM servers to be ready..."
TIMEOUT=600
START_TIME=$(date +%s)
RED_READY=false
BLUE_READY=false

while [[ "$RED_READY" == "false" || "$BLUE_READY" == "false" ]]; do
    ELAPSED=$(($(date +%s) - START_TIME))
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        echo "ERROR: vLLM servers did not become ready within ${TIMEOUT}s"
        echo "Check logs: $LOG_DIR/crosseval_red.log and crosseval_blue.log"
        exit 1
    fi

    # Check if processes are still alive
    if ! kill -0 $RED_VLLM_PID 2>/dev/null; then
        echo "ERROR: Red vLLM server died. Check $LOG_DIR/crosseval_red.log"
        exit 1
    fi
    if ! kill -0 $BLUE_VLLM_PID 2>/dev/null; then
        echo "ERROR: Blue vLLM server died. Check $LOG_DIR/crosseval_blue.log"
        exit 1
    fi

    # Health check
    if [[ "$RED_READY" == "false" ]]; then
        if curl -s http://localhost:8001/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('data') else 1)" 2>/dev/null; then
            RED_READY=true
            echo "  Red vLLM ready (${ELAPSED}s)"
        fi
    fi
    if [[ "$BLUE_READY" == "false" ]]; then
        if curl -s http://localhost:8002/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('data') else 1)" 2>/dev/null; then
            BLUE_READY=true
            echo "  Blue vLLM ready (${ELAPSED}s)"
        fi
    fi

    if [[ "$RED_READY" == "false" || "$BLUE_READY" == "false" ]]; then
        echo "  Waiting... (${ELAPSED}/${TIMEOUT}s)"
        sleep 5
    fi
done
echo "Both vLLM servers ready."

# ============================================
# 3. Run cross-evaluation
# ============================================
echo ""
echo "[3/3] Running cross-evaluation..."
python3 util/cross_evaluate.py \
    --selfplay-dir "$SELFPLAY_DIR" \
    --base-model "$BASE_MODEL" \
    --episodes "$EPISODES" \
    --horizon "$HORIZON" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR" \
    --red-vllm-url "http://localhost:8001/v1" \
    --blue-vllm-url "http://localhost:8002/v1" \
    --resume

echo ""
echo "Aggregating results..."
python3 util/cross_evaluate.py \
    --selfplay-dir "$SELFPLAY_DIR" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --aggregate-only

echo ""
echo "Generating plots..."
python3 util/plot_cross_eval.py "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Cross-Evaluation Complete"
echo "========================================"
echo "Results: $OUTPUT_DIR"
echo "Figures: $OUTPUT_DIR/figures/"

# Cleanup handled by EXIT trap
