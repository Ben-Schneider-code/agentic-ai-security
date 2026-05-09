#!/bin/bash
# Run the full 3-configuration human baseline comparison in one shot.
#
# Produces a paper-ready LaTeX table at data/human_eval/comparison.tex showing
# PVR_turn, PVR_conv, BRR, and WF for three defender configurations:
#   1. Unprotected  — minimal system prompt (schema only, no policy)
#   2. Manually protected — full sql_system_prompt, no RL fine-tuning
#   3. RL protected — full sql_system_prompt + blueteam LoRA from iter_1
#
# Usage:
#   ./run_human_baseline_comparison.sh \
#       --canonical-run results-20260408-1726-t9s16 \
#       --port 8002 --gpu 0
#
# Optional flags:
#   --resume          Resume each config from any prior partial run
#   --num-seeds N     Seeds per attack prompt (default 10)
#   --max-concurrent N  vLLM request concurrency (default 8)
#   --skip-config TAG   Skip a config by tag; repeat to skip multiple
#                       (e.g., --skip-config human_unprotected --skip-config human_manual)
#   --reuse-existing    Symlink existing arctic_0 → human_manual instead of
#                       re-running it (only safe if arctic_0 was run with the
#                       same settings; disabled by default)
#   --attack-file PATH  Custom attack file (default new_jailbreaks.txt)
#   --gpu-mem FRAC      vLLM GPU memory utilization (default 0.90)
#   --max-model-len N   vLLM max model length (default 16384)

set -e
set -o pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CANONICAL_RUN=""
PORT=8002
GPU=0
GPU_MEM=0.90
MAX_MODEL_LEN=16384
NUM_SEEDS=10
MAX_CONCURRENT=8
ATTACK_FILE="new_jailbreaks.txt"
RESUME=""
REUSE_EXISTING=""
SKIP_CONFIGS=()

MODEL_ID="Snowflake/Arctic-Text2SQL-R1-7B"
UNPROTECTED_PROMPT="prompts/unprotected_system_prompt.txt"

TAG_UNPROTECTED="human_unprotected"
TAG_MANUAL="human_manual"
TAG_RL="human_rl_iter1"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --canonical-run)   CANONICAL_RUN="$2";  shift ;;
        --port)            PORT="$2";            shift ;;
        --gpu)             GPU="$2";             shift ;;
        --gpu-mem)         GPU_MEM="$2";         shift ;;
        --max-model-len)   MAX_MODEL_LEN="$2";   shift ;;
        --num-seeds)       NUM_SEEDS="$2";       shift ;;
        --max-concurrent)  MAX_CONCURRENT="$2";  shift ;;
        --attack-file)     ATTACK_FILE="$2";     shift ;;
        --resume)          RESUME="--resume" ;;
        --reuse-existing)  REUSE_EXISTING=1 ;;
        --skip-config)     SKIP_CONFIGS+=("$2"); shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# ── Validate ───────────────────────────────────────────────────────────────────
if [[ -z "$CANONICAL_RUN" ]]; then
    echo "ERROR: --canonical-run is required (e.g. results-20260408-1726-t9s16)"
    exit 1
fi
if [[ ! -d "$CANONICAL_RUN" ]]; then
    echo "ERROR: Canonical run directory not found: $CANONICAL_RUN"
    exit 1
fi
ITER1_BLUETEAM="$CANONICAL_RUN/iter_1/blueteam"
if [[ ! -d "$ITER1_BLUETEAM" ]]; then
    echo "ERROR: iter_1/blueteam not found inside $CANONICAL_RUN"
    exit 1
fi
if [[ ! -f "$UNPROTECTED_PROMPT" ]]; then
    echo "ERROR: Unprotected system prompt not found: $UNPROTECTED_PROMPT"
    exit 1
fi

# Helper: check if a config tag should be skipped
should_skip() {
    local tag="$1"
    for s in "${SKIP_CONFIGS[@]}"; do
        [[ "$s" == "$tag" ]] && return 0
    done
    return 1
}

# Helper: poll until a bind() on the port succeeds (max 90s).
# ss -tlnp only catches LISTEN sockets; a Python bind() attempt is the ground truth.
wait_port_free() {
    local port="$1"
    local timeout=90
    local start
    start=$(date +%s)
    while ! python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(('', $port))
    s.close()
    sys.exit(0)
except OSError:
    sys.exit(1)
" 2>/dev/null; do
        elapsed=$(( $(date +%s) - start ))
        if [[ $elapsed -ge $timeout ]]; then
            echo "  WARNING: port $port still in use after ${timeout}s; proceeding anyway"
            return 0
        fi
        echo "  Waiting for port $port to be released... (${elapsed}s)"
        sleep 5
    done
    echo "  Port $port free."
}

# Helper: true if a config has already produced a complete summary.json
is_complete() {
    local tag="$1"
    [[ -f "data/human_eval/${tag}/summary.json" ]]
}

# Common flags forwarded to run_human_eval.sh
COMMON_FLAGS=(
    --model-id "$MODEL_ID"
    --port "$PORT"
    --gpu "$GPU"
    --gpu-mem "$GPU_MEM"
    --max-model-len "$MAX_MODEL_LEN"
    --num-seeds "$NUM_SEEDS"
    --max-concurrent "$MAX_CONCURRENT"
    --attack-file "$ATTACK_FILE"
)
[[ -n "$RESUME" ]] && COMMON_FLAGS+=(--resume)

echo ""
echo "########################################"
echo "Human Baseline Comparison (3 configs)"
echo "########################################"
echo "  Canonical run : $CANONICAL_RUN"
echo "  Model         : $MODEL_ID"
echo "  Port          : $PORT  (GPU $GPU)"
echo "  Seeds         : $NUM_SEEDS"
echo ""

# ── Config 1: Unprotected ─────────────────────────────────────────────────────
if should_skip "$TAG_UNPROTECTED"; then
    echo "[skip] $TAG_UNPROTECTED (--skip-config)"
elif is_complete "$TAG_UNPROTECTED"; then
    echo "[done] $TAG_UNPROTECTED — summary.json exists, skipping"
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[1/3] Running: $TAG_UNPROTECTED"
    echo "      System prompt: $UNPROTECTED_PROMPT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ./run_human_eval.sh \
        "${COMMON_FLAGS[@]}" \
        --tag "$TAG_UNPROTECTED" \
        --system-prompt-file "$UNPROTECTED_PROMPT"
    wait_port_free "$PORT"
    echo ""
fi

# ── Config 2: Manually protected ──────────────────────────────────────────────
if should_skip "$TAG_MANUAL"; then
    echo "[skip] $TAG_MANUAL (--skip-config)"
elif is_complete "$TAG_MANUAL"; then
    echo "[done] $TAG_MANUAL — summary.json exists, skipping"
elif [[ -n "$REUSE_EXISTING" && -f "data/human_eval/arctic_0/summary.json" ]]; then
    echo "[reuse] Copying data/human_eval/arctic_0 → data/human_eval/$TAG_MANUAL"
    mkdir -p "data/human_eval/$TAG_MANUAL"
    cp "data/human_eval/arctic_0/"* "data/human_eval/$TAG_MANUAL/"
    echo "        (adapter_path/system_prompt_path fields may be null — arctic_0 predates those fields)"
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[2/3] Running: $TAG_MANUAL"
    echo "      System prompt: built-in sql_system_prompt"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ./run_human_eval.sh \
        "${COMMON_FLAGS[@]}" \
        --tag "$TAG_MANUAL"
    wait_port_free "$PORT"
    echo ""
fi

# ── Config 3: RL protected ────────────────────────────────────────────────────
if should_skip "$TAG_RL"; then
    echo "[skip] $TAG_RL (--skip-config)"
elif is_complete "$TAG_RL"; then
    echo "[done] $TAG_RL — summary.json exists, skipping"
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[3/3] Running: $TAG_RL"
    echo "      LoRA: $ITER1_BLUETEAM"
    echo "      System prompt: built-in sql_system_prompt"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ./run_human_eval.sh \
        "${COMMON_FLAGS[@]}" \
        --tag "$TAG_RL" \
        --blueteam-dir "$ITER1_BLUETEAM"
    wait_port_free "$PORT"
    echo ""
fi

# ── Aggregate ─────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Aggregating results..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 util/aggregate_human_baseline.py \
    --tags "$TAG_UNPROTECTED" "$TAG_MANUAL" "$TAG_RL" \
    --labels "Unprotected" "Manually Protected" "RL Protected" \
    --human-eval-dir data/human_eval \
    --output-dir data/human_eval

echo ""
echo "########################################"
echo "Done. LaTeX table: data/human_eval/comparison.tex"
echo "########################################"
