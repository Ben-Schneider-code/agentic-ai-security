#!/bin/bash
# Benign-TPR sweep: every blue-team LoRA checkpoint, 200 benign episodes each.
#
# Usage:
#   ./util/run_benign_eval.sh --results-dir results-20260322-1641-m92p4 \
#       [--episodes 200] [--base-model <hf-id>] \
#       [--gpu 0] [--gpu-mem 0.50] [--max-model-len 4096] \
#       [--concurrency 8] [--seed 42] \
#       [--port 8002] [--no-include-base] [--resume] [--aggregate-only]
#
# --results-dir is the self-play run directory (contains iter_*/blueteam/).
# Output is written to <results-dir>/benign_eval/.
#
# Only the blue-team LoRAs are loaded; no red-team vLLM server is started.
# Default GPU memory fraction (0.50) is safe for a shared server A100.

set -e
set -o pipefail

# ── Progress colors ────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    _B='\033[1m'; _0='\033[0m'
    _CYN='\033[1;36m'; _GRN='\033[1;32m'; _YLW='\033[1;33m'; _RED='\033[1;31m'
else
    _B=''; _0=''; _CYN=''; _GRN=''; _YLW=''; _RED=''
fi
_step()   { printf "${_CYN}${_B}▶  %s${_0}\n"          "$*"; }
_ok()     { printf "${_GRN}${_B}✓  %s${_0}\n"          "$*"; }
_warn()   { printf "${_YLW}${_B}⚠  WARNING: %s${_0}\n" "$*"; }
_err()    { printf "${_RED}${_B}✗  ERROR: %s${_0}\n"   "$*"; }
_banner() { printf "${_B}%s${_0}\n"                     "$*"; }

_banner "========================================"
_banner "Agentic AI Security: Benign-TPR Sweep"
_banner "========================================"

# ── Defaults ───────────────────────────────────────────────────────────────────
RESULTS_DIR=""
EPISODES=200
BASE_MODEL=""
GPU=0
GPU_MEM=0.50
MAX_MODEL_LEN=4096
CONCURRENCY=8
SEED=42
PORT=8002
INCLUDE_BASE=true
RESUME=false
AGGREGATE_ONLY=false

# ── Parse arguments ────────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --results-dir)     RESULTS_DIR="$2";    shift ;;
        --episodes)        EPISODES="$2";       shift ;;
        --base-model)      BASE_MODEL="$2";     shift ;;
        --gpu)             GPU="$2";            shift ;;
        --gpu-mem)         GPU_MEM="$2";        shift ;;
        --max-model-len)   MAX_MODEL_LEN="$2";  shift ;;
        --concurrency)     CONCURRENCY="$2";    shift ;;
        --seed)            SEED="$2";           shift ;;
        --port)            PORT="$2";           shift ;;
        --no-include-base) INCLUDE_BASE=false ;;
        --resume)          RESUME=true ;;
        --aggregate-only)  AGGREGATE_ONLY=true ;;
        *) _err "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# ── Validate ───────────────────────────────────────────────────────────────────
if [[ -z "$RESULTS_DIR" ]]; then
    _err "--results-dir is required."
    echo "Usage: ./util/run_benign_eval.sh --results-dir <selfplay-run-dir> [options]"
    exit 1
fi
RESULTS_DIR="${RESULTS_DIR%/}"
if [[ ! -d "$RESULTS_DIR" ]]; then
    _err "Directory not found: $RESULTS_DIR"
    exit 1
fi

# Clamp gpu-mem at 0.90 (same guard as run_cross_eval.sh single-GPU mode)
if awk -v g="$GPU_MEM" 'BEGIN{exit !(g > 0.90)}'; then
    _warn "--gpu-mem=$GPU_MEM exceeds 0.90 cap; clamping to 0.90."
    GPU_MEM=0.90
fi

OUTPUT_DIR="${RESULTS_DIR}/benign_eval"
mkdir -p "$OUTPUT_DIR"

# ── Aggregate-only shortcut: no infra needed ───────────────────────────────────
if [[ "$AGGREGATE_ONLY" == "true" ]]; then
    _step "[aggregate-only] Rebuilding benign_eval_results.json..."
    AONLY_ARGS=(--aggregate-only)
    [[ "$INCLUDE_BASE" == "false" ]] && AONLY_ARGS+=(--no-include-base)
    python3 util/run_benign_eval.py \
        --selfplay-dir "$RESULTS_DIR" \
        --output-dir "$OUTPUT_DIR" \
        "${AONLY_ARGS[@]}"
    _ok "Aggregation complete → $OUTPUT_DIR/benign_eval_results.json"
    exit 0
fi

# ── Discover blue LoRA checkpoints ─────────────────────────────────────────────
# Mirrors the find_latest_checkpoint pattern in run_cross_eval.sh
find_latest_checkpoint() {
    local dir="$1"
    find "$dir" -name "sql_agent" -type d 2>/dev/null | sort -V | tail -n 1
}

ITERATIONS=()
for dir in "$RESULTS_DIR"/iter_*; do
    if [[ -d "$dir" ]]; then
        iter_num=$(basename "$dir" | sed 's/iter_//')
        ITERATIONS+=("$iter_num")
    fi
done
IFS=$'\n' ITERATIONS=($(sort -n <<<"${ITERATIONS[*]}")); unset IFS

if [[ ${#ITERATIONS[@]} -eq 0 ]]; then
    _err "No iter_* directories found in $RESULTS_DIR"
    exit 1
fi
echo "Found ${#ITERATIONS[@]} iterations: ${ITERATIONS[*]}"

declare -A BLUE_LORAS
for iter in "${ITERATIONS[@]}"; do
    blue_dir="$RESULTS_DIR/iter_${iter}/blueteam"
    if [[ -d "$blue_dir" ]]; then
        ckpt=$(find_latest_checkpoint "$blue_dir")
        if [[ -n "$ckpt" ]]; then
            BLUE_LORAS[$iter]=$(realpath "$ckpt")
            echo "  Blue iter_${iter}: ${BLUE_LORAS[$iter]}"
        else
            _warn "No blue checkpoint in iter_${iter}, skipping."
        fi
    fi
done

N_BLUE=${#BLUE_LORAS[@]}
if [[ $N_BLUE -eq 0 ]]; then
    _err "No blue LoRA checkpoints found in $RESULTS_DIR"
    exit 1
fi

# ── Auto-detect base model ─────────────────────────────────────────────────────
if [[ -z "$BASE_MODEL" ]]; then
    SAMPLE_ADAPTER=$(find "$RESULTS_DIR" -name "adapter_config.json" -type f -print -quit)
    if [[ -z "$SAMPLE_ADAPTER" ]]; then
        _err "Could not auto-detect base model — no adapter_config.json found."
        echo "Specify --base-model explicitly."
        exit 1
    fi
    BASE_MODEL=$(python3 -c "import json; print(json.load(open('$SAMPLE_ADAPTER'))['base_model_name_or_path'])")
    echo "Auto-detected base model: $BASE_MODEL"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "Blue LoRAs found: ${N_BLUE}"
echo "Results directory: $OUTPUT_DIR"
_step "Episodes per iteration: $EPISODES  seed=$SEED  concurrency=$CONCURRENCY"
_step "GPU: $GPU  gpu-mem: $GPU_MEM  max-model-len: $MAX_MODEL_LEN  port: $PORT"

# ── Cleanup trap ───────────────────────────────────────────────────────────────
VLLM_PID=""
cleanup_vllm() {
    echo ""
    echo "[cleanup] Stopping vLLM server..."
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[cleanup] Sending SIGTERM to vLLM (PID=$VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    sleep 10
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[cleanup] WARNING: vLLM still alive. Sending SIGKILL — may brick GPU."
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup_vllm EXIT

# ============================================================
# 1. Start Postgres + MCP
# ============================================================
echo ""
_step "[1/3] Starting Postgres and MCP..."
./script/init.sh
_ok "Postgres and MCP ready."

# ============================================================
# 2. Start vLLM server (blue-team LoRAs only)
# ============================================================
echo ""
_step "[2/3] Starting vLLM server..."

LOG_DIR="/tmp/vllm_logs"
mkdir -p "$LOG_DIR"

BLUE_LORA_MODULES=""
for iter in "${!BLUE_LORAS[@]}"; do
    BLUE_LORA_MODULES="${BLUE_LORA_MODULES} blue_${iter}=${BLUE_LORAS[$iter]}"
done

MAX_BLUE_LORAS=$(( N_BLUE > 2 ? N_BLUE : 2 ))

echo "  vLLM: GPU $GPU, port $PORT, ${N_BLUE} blue LoRAs"
echo "  LoRA modules:$BLUE_LORA_MODULES"

CUDA_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --port $PORT \
    --host 0.0.0.0 \
    --gpu-memory-utilization $GPU_MEM \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --disable-log-requests \
    --dtype auto \
    --enable-lora \
    --max-lora-rank 64 \
    --max-loras $MAX_BLUE_LORAS \
    --lora-modules $BLUE_LORA_MODULES \
    > "$LOG_DIR/benign_eval.log" 2>&1 &
VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID"

echo "Waiting for vLLM server to be ready..."
TIMEOUT=600
START_TIME=$(date +%s)
SERVER_READY=false

while [[ "$SERVER_READY" == "false" ]]; do
    ELAPSED=$(($(date +%s) - START_TIME))
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        _err "vLLM server did not become ready within ${TIMEOUT}s."
        echo "Check log: $LOG_DIR/benign_eval.log"
        exit 1
    fi

    if ! kill -0 $VLLM_PID 2>/dev/null; then
        _err "vLLM server died. Check $LOG_DIR/benign_eval.log"
        exit 1
    fi

    if curl -s "http://localhost:${PORT}/v1/models" | \
        python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('data') else 1)" 2>/dev/null; then
        SERVER_READY=true
        _ok "vLLM ready (${ELAPSED}s)"
    fi

    if [[ "$SERVER_READY" == "false" ]]; then
        echo "  Waiting... (${ELAPSED}/${TIMEOUT}s)"
        sleep 5
    fi
done

# ============================================================
# 3. Run benign sweep
# ============================================================
echo ""
_step "[3/3] Running benign-TPR sweep..."

SWEEP_ARGS=(
    --selfplay-dir "$RESULTS_DIR"
    --base-model   "$BASE_MODEL"
    --episodes     "$EPISODES"
    --seed         "$SEED"
    --output-dir   "$OUTPUT_DIR"
    --blue-vllm-url "http://localhost:${PORT}/v1"
    --concurrency  "$CONCURRENCY"
)
[[ "$INCLUDE_BASE" == "false" ]] && SWEEP_ARGS+=(--no-include-base)
[[ "$RESUME"       == "true"  ]] && SWEEP_ARGS+=(--resume)

python3 util/run_benign_eval.py "${SWEEP_ARGS[@]}"

echo ""
_banner "========================================"
_ok  "Benign-TPR Sweep Complete"
_banner "========================================"
echo "Results: $OUTPUT_DIR"
echo "Summary: $OUTPUT_DIR/benign_eval_results.json"

# Cleanup handled by EXIT trap
