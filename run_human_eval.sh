#!/bin/bash
# Evaluate a blueteam model against the human jailbreak + benign query datasets.
#
# Usage (base model, no LoRA):
#   ./run_human_eval.sh --model-id Snowflake/Arctic-Text2SQL-R1-7B --tag arctic_base
#
# Usage (trained LoRA checkpoint):
#   ./run_human_eval.sh \
#       --model-id Snowflake/Arctic-Text2SQL-R1-7B \
#       --blueteam-dir results-20260322-1641-m92p4/iter_3/blueteam \
#       --tag arctic_iter3
#
# The script:
#   1. Discovers the latest complete LoRA checkpoint in --blueteam-dir (if given)
#   2. Auto-detects base model from adapter_config.json (overridden by --model-id)
#   3. Starts a vLLM server (with or without LoRA) on --port
#   4. Initialises Postgres + MCP via ./script/init.sh
#   5. Runs util/eval_human_vs_blueteam.py and writes results to data/human_eval/<tag>/
#   6. Shuts down vLLM on exit
#
# Outputs: data/human_eval/<tag>/summary.json  (PVR_turn, PVR_conv, BRR + CIs)
#          data/human_eval/<tag>/attack_detail.jsonl
#          data/human_eval/<tag>/benign_detail.jsonl

set -e
set -o pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_ID=""
BLUETEAM_DIR=""
TAG=""
PORT=8002
GPU=0
GPU_MEM=0.90
MAX_MODEL_LEN=16384
NUM_SEEDS=10
MAX_CONCURRENT=8
ATTACK_FILE="new_jailbreaks.txt"
RESUME=""
SEED_BASE=1000
SYSTEM_PROMPT_FILE=""

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --model-id)            MODEL_ID="$2";            shift ;;
        --blueteam-dir)        BLUETEAM_DIR="$2";        shift ;;
        --tag)                 TAG="$2";                 shift ;;
        --port)                PORT="$2";                shift ;;
        --gpu)                 GPU="$2";                 shift ;;
        --gpu-mem)             GPU_MEM="$2";             shift ;;
        --max-model-len)       MAX_MODEL_LEN="$2";       shift ;;
        --num-seeds)           NUM_SEEDS="$2";           shift ;;
        --max-concurrent)      MAX_CONCURRENT="$2";      shift ;;
        --attack-file)         ATTACK_FILE="$2";         shift ;;
        --resume)              RESUME="--resume" ;;
        --seed-base)           SEED_BASE="$2";           shift ;;
        --system-prompt-file)  SYSTEM_PROMPT_FILE="$2";  shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# ── Validate ───────────────────────────────────────────────────────────────────
if [[ -z "$TAG" ]]; then
    echo "ERROR: --tag is required (short label for the output directory)."
    exit 1
fi

# ── Locate adapter if LoRA run ─────────────────────────────────────────────────
ADAPTER_PATH=""
LORA_MODULES_ARG=""
LORA_NAME_ARG=""

if [[ -n "$BLUETEAM_DIR" ]]; then
    BLUETEAM_DIR="${BLUETEAM_DIR%/}"
    if [[ ! -d "$BLUETEAM_DIR" ]]; then
        echo "ERROR: --blueteam-dir not found: $BLUETEAM_DIR"
        exit 1
    fi

    echo "Searching for latest complete checkpoint in $BLUETEAM_DIR ..."
    ADAPTER_PATH=$(python3 -c "
import pathlib, sys
root = pathlib.Path('$BLUETEAM_DIR')
if not root.is_dir():
    print('')
    sys.exit(0)

best = None
best_step = -1
for cfg in root.rglob('adapter_config.json'):
    adapter = cfg.parent
    if adapter.name != 'sql_agent':
        continue
    steps_dir = adapter.parent
    if not steps_dir.name.startswith('steps_'):
        continue
    try:
        step = int(steps_dir.name.split('_', 1)[1])
    except ValueError:
        continue
    if step > best_step:
        best_step = step
        best = adapter

print(str(best) if best is not None else '')
")
    if [[ -z "$ADAPTER_PATH" ]]; then
        echo "ERROR: No complete checkpoint (sql_agent/adapter_config.json) found in $BLUETEAM_DIR"
        exit 1
    fi
    echo "  Adapter path : $ADAPTER_PATH"

    # Auto-detect base model from adapter config (overridden if --model-id was given)
    if [[ -z "$MODEL_ID" ]]; then
        MODEL_ID=$(python3 -c "
import json
with open('$ADAPTER_PATH/adapter_config.json') as f:
    print(json.load(f)['base_model_name_or_path'])
")
        echo "  Base model   : $MODEL_ID (auto-detected)"
    fi

    LORA_MODULES_ARG="--enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules blue=$ADAPTER_PATH"
    LORA_NAME_ARG="--lora-name blue"
fi

if [[ -z "$MODEL_ID" ]]; then
    echo "ERROR: --model-id is required (or auto-detected from --blueteam-dir)."
    exit 1
fi

OUTPUT_DIR="data/human_eval/${TAG}"
LOG_DIR="/tmp/vllm_logs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Human-Dataset Blueteam Evaluation"
echo "========================================"
echo "  Model         : $MODEL_ID"
echo "  LoRA          : ${ADAPTER_PATH:-none}"
echo "  System prompt : ${SYSTEM_PROMPT_FILE:-built-in sql_system_prompt (manually protected)}"
echo "  Port          : $PORT  (GPU $GPU)"
echo "  Seeds         : $NUM_SEEDS"
echo "  Attack file   : $ATTACK_FILE"
echo "  Output dir    : $OUTPUT_DIR"
echo ""

# ── Cleanup trap ───────────────────────────────────────────────────────────────
VLLM_PID=""

cleanup_vllm() {
    echo ""
    echo "[cleanup] Stopping vLLM server..."
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 10
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[cleanup] WARNING: vLLM still alive, sending SIGKILL."
            kill -9 "$VLLM_PID" 2>/dev/null || true
        fi
    fi
}
trap cleanup_vllm EXIT

# ── 1. Start Postgres + MCP ────────────────────────────────────────────────────
echo "[1/3] Starting Postgres and MCP..."
./script/init.sh
echo "Postgres and MCP ready."

# ── 2. Start vLLM ─────────────────────────────────────────────────────────────
echo ""
echo "[2/3] Starting vLLM server (GPU $GPU, port $PORT)..."
VLLM_LOG="$LOG_DIR/human_eval_${TAG}.log"

CUDA_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --disable-log-requests \
    --dtype auto \
    $LORA_MODULES_ARG \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID  log: $VLLM_LOG"

echo "  Waiting for vLLM to be ready (timeout=600s)..."
TIMEOUT=600
START=$(date +%s)
while true; do
    ELAPSED=$(( $(date +%s) - START ))
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        echo "ERROR: vLLM did not become ready within ${TIMEOUT}s. Check $VLLM_LOG"
        exit 1
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM process died. Check $VLLM_LOG"
        exit 1
    fi
    if curl -s "http://localhost:${PORT}/v1/models" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('data') else 1)" 2>/dev/null; then
        echo "  vLLM ready (${ELAPSED}s)"
        break
    fi
    echo "  Waiting... (${ELAPSED}/${TIMEOUT}s)"
    sleep 5
done

# ── 3. Run evaluation ──────────────────────────────────────────────────────────
echo ""
echo "[3/3] Running evaluation..."
SYS_PROMPT_ARG=""
[[ -n "$SYSTEM_PROMPT_FILE" ]] && SYS_PROMPT_ARG="--system-prompt-file $SYSTEM_PROMPT_FILE"

ADAPTER_PATH_ARG=""
[[ -n "$ADAPTER_PATH" ]] && ADAPTER_PATH_ARG="--adapter-path $ADAPTER_PATH"

python3 util/eval_human_vs_blueteam.py \
    --model-id "$MODEL_ID" \
    $LORA_NAME_ARG \
    $SYS_PROMPT_ARG \
    $ADAPTER_PATH_ARG \
    --port "$PORT" \
    --attack-file "$ATTACK_FILE" \
    --num-seeds "$NUM_SEEDS" \
    --output-dir "$OUTPUT_DIR" \
    --max-concurrent "$MAX_CONCURRENT" \
    --seed-base "$SEED_BASE" \
    $RESUME

echo ""
echo "========================================"
echo "Evaluation complete. Results in $OUTPUT_DIR"
echo "========================================"
