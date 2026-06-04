#!/bin/bash
# Cross-Evaluation: All n² (red_i, blue_j) pairings from a self-play run.
#
# Usage (dual-GPU, default):
#   ./run_cross_eval.sh --selfplay-dir results-20260322-1641-m92p4 \
#       [--episodes 100] [--base-model <model>] \
#       [--red-gpu 0] [--blue-gpu 1] [--horizon 5] [--seed 42] \
#       [--pairing-subset {full|diagonal|diag_plus_base|custom}] \
#       [--pairing-list r1:b1,r2:b2,...] \
#       [--plot-only]
#
# Usage (single-GPU, shared-server-friendly):
#   ./run_cross_eval.sh --selfplay-dir <dir> --single-gpu \
#       [--gpu 3] [--gpu-mem 0.50] [--max-model-len 4096] [--concurrency 8]
#
# Dual-GPU mode requires 2 fully-free A100s (80GB each). Single-GPU mode
# merges both red and blue LoRAs onto one vLLM server (same base model) —
# fits in ~40GB and is suitable when no GPU is fully free.
#
# After the paired evaluation + benign-only pass (both run by
# util/cross_evaluate.py), this script also invokes util/mcnemar_cross_eval.py
# to produce significance_tests.csv + significance_matrix.json, so that
# util/plot_cross_eval.py can annotate non-significant cells (p > 0.05) in the
# cross-eval heatmaps per methodology §sec:episode-protocol.

set -e
set -o pipefail

# --- Progress colors (suppressed when stdout is not a TTY) ---
if [[ -t 1 ]]; then
    _B='\033[1m'; _0='\033[0m'
    _CYN='\033[1;36m'; _GRN='\033[1;32m'; _YLW='\033[1;33m'; _RED='\033[1;31m'
else
    _B=''; _0=''; _CYN=''; _GRN=''; _YLW=''; _RED=''
fi
_step()   { printf "${_CYN}${_B}▶  %s${_0}\n"            "$*"; }
_ok()     { printf "${_GRN}${_B}✓  %s${_0}\n"            "$*"; }
_warn()   { printf "${_YLW}${_B}⚠  WARNING: %s${_0}\n"   "$*"; }
_err()    { printf "${_RED}${_B}✗  ERROR: %s${_0}\n"     "$*"; }
_banner() { printf "${_B}%s${_0}\n"                       "$*"; }

_banner "========================================"
_banner "Agentic AI Security: Cross-Evaluation"
_banner "========================================"

# --- Defaults ---
SELFPLAY_DIR=""
EPISODES=800
BASE_MODEL=""
RED_GPU=0
BLUE_GPU=1
HORIZON=5
SEED=42
PLOT_ONLY=false
QUICK=false
DIAGONAL_ONLY=false
PAIRING_SUBSET=""        # full | diagonal | diag_plus_base | diag_plus_adjacent | custom; default depends on mode flags
PAIRING_LIST=""          # only used when PAIRING_SUBSET=custom
QUICK_EPISODES=40
QUICK_HORIZON=3
SINGLE_GPU=false
GPU_ID=0                 # GPU index for single-GPU mode
RED_PORT=""              # vLLM port for red server; auto-allocated (free port) if unset
BLUE_PORT=""             # vLLM port for blue server; auto-allocated (free port) if unset
SKIP_INIT=false          # back-compat flag; DB reuse is inferred from $AAS_PG_CONTAINER
GPU_MEM=""               # gpu-memory-utilization (default varies by mode)
MAX_MODEL_LEN=""         # max-model-len (default varies by mode)
CONCURRENCY=""           # passed to cross_evaluate.py (default varies by mode)
INCLUDE_BASE=true        # forward --no-include-base to cross_evaluate.py (default: base included)
RESUME=false             # skip pairings whose summary.json already exists
BLUETEAM_SYSTEM_PROMPT_FILE=""  # override blue system prompt (P2 frozen-baseline experiments)
OUTPUT_DIR_OVERRIDE=""          # override default output dir (cross_eval, diagonal_eval, etc.)
EPISODES_EXPLICIT=false
PAIRING_SUBSET_EXPLICIT=false
MATCH_TRAIN_SEEDS=false  # when true, cross-eval reads red_seed from summary.json instead of --seed

# --- Parse arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --selfplay-dir) SELFPLAY_DIR="$2"; shift ;;
        --episodes) EPISODES="$2"; EPISODES_EXPLICIT=true; shift ;;
        --base-model) BASE_MODEL="$2"; shift ;;
        --red-gpu) RED_GPU="$2"; shift ;;
        --blue-gpu) BLUE_GPU="$2"; shift ;;
        --horizon) HORIZON="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --plot-only) PLOT_ONLY=true ;;
        --quick) QUICK=true ;;
        --diagonal-only) DIAGONAL_ONLY=true ;;
        --pairing-subset) PAIRING_SUBSET="$2"; PAIRING_SUBSET_EXPLICIT=true; shift ;;
        --pairing-list) PAIRING_LIST="$2"; shift ;;
        --quick-episodes) QUICK_EPISODES="$2"; shift ;;
        --quick-horizon) QUICK_HORIZON="$2"; shift ;;
        --single-gpu) SINGLE_GPU=true ;;
        --gpu) GPU_ID="$2"; shift ;;
        --gpu-mem) GPU_MEM="$2"; shift ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift ;;
        --concurrency) CONCURRENCY="$2"; shift ;;
        --no-include-base) INCLUDE_BASE=false ;;
        --resume) RESUME=true ;;
        --blueteam-system-prompt-file) BLUETEAM_SYSTEM_PROMPT_FILE="$2"; shift ;;
        --output-dir) OUTPUT_DIR_OVERRIDE="$2"; shift ;;
        --red-port) RED_PORT="$2"; shift ;;
        --blue-port) BLUE_PORT="$2"; shift ;;
        --skip-init) SKIP_INIT=true ;;
        --match-train-seeds) MATCH_TRAIN_SEEDS=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# --- Apply --quick mode defaults ---
if [[ "$QUICK" == "true" ]]; then
    [[ "$EPISODES_EXPLICIT"       == "false" ]] && EPISODES="$QUICK_EPISODES"
    HORIZON="$QUICK_HORIZON"
    [[ "$PAIRING_SUBSET_EXPLICIT" == "false" && -z "$PAIRING_SUBSET" ]] && PAIRING_SUBSET="diag_plus_base"
fi

# --- Apply --diagonal-only mode defaults (takes precedence over --quick for episodes/subset) ---
if [[ "$DIAGONAL_ONLY" == "true" ]]; then
    [[ "$EPISODES_EXPLICIT"       == "false" ]] && EPISODES=800
    [[ "$PAIRING_SUBSET_EXPLICIT" == "false" ]] && PAIRING_SUBSET="diag_plus_adjacent"
fi

if [[ -z "$PAIRING_SUBSET" ]]; then
    PAIRING_SUBSET="full"
fi

# --- Apply GPU-mode-dependent defaults ---
if [[ "$SINGLE_GPU" == "true" ]]; then
    GPU_MEM="${GPU_MEM:-0.50}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
    CONCURRENCY="${CONCURRENCY:-8}"
    # Cap at 0.90 in single-GPU mode: the A100 is shared, and anything
    # higher risks OOM'ing co-tenants / the OS and bricking the GPU.
    if awk -v g="$GPU_MEM" 'BEGIN{exit !(g > 0.90)}'; then
        _warn "--gpu-mem=$GPU_MEM exceeds 0.90 cap for single-GPU mode; clamping to 0.90."
        GPU_MEM="0.90"
    fi
else
    GPU_MEM="${GPU_MEM:-0.90}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
    CONCURRENCY="${CONCURRENCY:-12}"
fi

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

# --- Inherit honeypot arm from the self-play run (fail-fast) ----------------
# redteam_sql_env reads HONEYPOT_TYPE at import time and crashes if it's unset,
# so we must export it BEFORE launching any python3 process below. summary.json
# is the source of truth — re-deriving the arm any other way would risk drift.
SUMMARY_JSON="${SELFPLAY_DIR}/summary.json"
if [[ ! -f "$SUMMARY_JSON" ]]; then
    _err "summary.json not found at $SUMMARY_JSON — cannot determine honeypot arm."
    exit 1
fi
HONEYPOT_TYPE=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); v=d.get('honeypot_type'); assert v in ('rowcol','row','col'), f'bad honeypot_type={v!r}'; print(v)" "$SUMMARY_JSON") || {
    _err "could not parse honeypot_type from $SUMMARY_JSON"; exit 1;
}
export HONEYPOT_TYPE
_step "Honeypot arm (from $SUMMARY_JSON): $HONEYPOT_TYPE"

if [[ "$DIAGONAL_ONLY" == "true" ]]; then
    OUTPUT_DIR="${SELFPLAY_DIR}/diagonal_eval"
elif [[ "$QUICK" == "true" ]]; then
    OUTPUT_DIR="${SELFPLAY_DIR}/cross_eval_quick"
else
    OUTPUT_DIR="${SELFPLAY_DIR}/cross_eval"
fi
# --output-dir override (e.g. for P2 baseline experiment → cross_eval_baseline/)
[[ -n "$OUTPUT_DIR_OVERRIDE" ]] && OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
mkdir -p "$OUTPUT_DIR"

# --- Per-run namespace + ephemeral infra helpers ---
ROOT_DIR="$(pwd)"
source "${ROOT_DIR}/script/pg_ephemeral.sh"
if [[ -z "$AAS_RUN_ID" ]]; then
    AAS_RUN_ID="crosseval-$(date +%Y%m%d-%H%M%S)-$$"
fi
if [[ -z "$AAS_RUN_DIR" ]]; then
    AAS_RUN_DIR="${ROOT_DIR}/.runtime/${AAS_RUN_ID}"
fi
export AAS_RUN_ID AAS_RUN_DIR
export VLLM_LOG_DIR="${AAS_RUN_DIR}/vllm_logs"
mkdir -p "$VLLM_LOG_DIR"
OWNS_DB=0

# Allocate vLLM ports dynamically unless explicitly overridden, so concurrent
# cross-eval runs never collide on a fixed 8001/8003.
[[ -z "$RED_PORT"  ]] && RED_PORT=$(alloc_free_port)
[[ -z "$BLUE_PORT" ]] && BLUE_PORT=$(alloc_free_port)
while [[ "$BLUE_PORT" == "$RED_PORT" ]]; do BLUE_PORT=$(alloc_free_port); done

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
# -L: follow symlinks. Required for ablation eval_view/ trees where the
# steps_NNNN dir itself is a symlink to the actual training output.
find_latest_checkpoint() {
    local dir="$1"
    find -L "$dir" -name "sql_agent" -type d 2>/dev/null | sort -V | tail -n 1
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
            _warn "No red checkpoint found for iter_${iter}, skipping red_${iter}"
        fi
    fi

    if [[ -d "$blue_dir" ]]; then
        ckpt=$(find_latest_checkpoint "$blue_dir")
        if [[ -n "$ckpt" ]]; then
            BLUE_LORAS[$iter]=$(realpath "$ckpt")
            echo "  Blue iter_${iter}: ${BLUE_LORAS[$iter]}"
        else
            _warn "No blue checkpoint found for iter_${iter}, skipping blue_${iter}"
        fi
    fi
done

N_RED=${#RED_LORAS[@]}
N_BLUE=${#BLUE_LORAS[@]}
# Size --max-loras to fit all LoRAs in vLLM memory (avoids per-pairing swap stalls).
# Floor of 2 matches the prior default.
MAX_RED_LORAS=$(( N_RED > 2 ? N_RED : 2 ))
MAX_BLUE_LORAS=$(( N_BLUE > 2 ? N_BLUE : 2 ))
# Single-GPU: all LoRAs on one server
COMBINED_LORAS=$(( N_RED + N_BLUE ))
MAX_ALL_LORAS=$(( COMBINED_LORAS > 2 ? COMBINED_LORAS : 2 ))
# +1 for base model (iter_0)
TOTAL_PAIRINGS=$(( (N_RED + 1) * (N_BLUE + 1) ))
echo ""
echo "LoRAs found: ${N_RED} red, ${N_BLUE} blue"
echo "Total pairings (including base): ${TOTAL_PAIRINGS}"
echo "Results directory: $OUTPUT_DIR"
if [[ "$DIAGONAL_ONLY" == "true" ]]; then
    _step "Mode: DIAGONAL-ONLY (episodes=$EPISODES, horizon=$HORIZON, subset=$PAIRING_SUBSET)"
elif [[ "$QUICK" == "true" ]]; then
    _step "Mode: QUICK (episodes=$EPISODES, horizon=$HORIZON, subset=$PAIRING_SUBSET)"
else
    _step "Mode: FULL (episodes=$EPISODES, horizon=$HORIZON, subset=$PAIRING_SUBSET)"
fi
if [[ "$SINGLE_GPU" == "true" ]]; then
    _step "GPU mode: SINGLE-GPU (GPU $GPU_ID, mem=$GPU_MEM, max-model-len=$MAX_MODEL_LEN, concurrency=$CONCURRENCY)"
else
    _step "GPU mode: DUAL-GPU (red=GPU $RED_GPU, blue=GPU $BLUE_GPU, mem=$GPU_MEM, max-model-len=$MAX_MODEL_LEN, concurrency=$CONCURRENCY)"
fi

# --- Plot-only mode: skip infrastructure, just re-plot ---
if [[ "$PLOT_ONLY" == "true" ]]; then
    echo ""
    _step "[plot-only] Aggregating results..."
    python3 util/cross_evaluate.py \
        --selfplay-dir "$SELFPLAY_DIR" \
        --base-model "$BASE_MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --aggregate-only

    _step "[plot-only] Computing pairwise significance (two-proportion / Fisher + McNemar)..."
    python3 util/mcnemar_cross_eval.py "$OUTPUT_DIR" || \
        _warn "mcnemar_cross_eval.py failed; heatmaps will render without significance annotations."

    _step "[plot-only] Generating plots..."
    python3 util/plot_cross_eval.py "$OUTPUT_DIR"

    echo ""
    _banner "========================================"
    _ok  "Cross-Evaluation Plots Complete"
    _banner "========================================"
    echo "Results: $OUTPUT_DIR"
    exit 0
fi

# ============================================
# Cleanup function — SIGTERM first, SIGKILL only as last resort
# Mirrors run_selfplay.sh pattern to avoid bricking GPUs
# ============================================
RED_VLLM_PID=""
BLUE_VLLM_PID=""
SINGLE_VLLM_PID=""

cleanup_vllm() {
    echo ""
    echo "[cleanup] Stopping vLLM servers..."
    # Each server is launched under setsid, so its PID is also its PGID and
    # `kill -- -PID` reaps the api_server plus every vLLM worker child.
    for pid_var in RED_VLLM_PID BLUE_VLLM_PID SINGLE_VLLM_PID; do
        pid=${!pid_var}
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "[cleanup] Sending SIGTERM to $pid_var (PGID=$pid)..."
            kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    # Give vLLM time for graceful CUDA context teardown
    sleep 10
    for pid_var in RED_VLLM_PID BLUE_VLLM_PID SINGLE_VLLM_PID; do
        pid=${!pid_var}
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "[cleanup] WARNING: $pid_var still alive. Sending SIGKILL — may brick GPU."
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
    done
}
_cross_cleanup() {
    # Preserve the real exit status — bash exits with the EXIT trap's last
    # command status, so a trailing false `[[...]] &&` would mask success as 1.
    local _rc=$?
    cleanup_vllm
    [[ "$OWNS_DB" == "1" ]] && pg_ephemeral_stop
    return $_rc
}
trap _cross_cleanup EXIT
trap '_cross_cleanup; exit 130' INT
trap '_cross_cleanup; exit 143' TERM

# ============================================
# 1. Start Postgres + MCP
# ============================================
echo ""
if [[ -n "$AAS_PG_CONTAINER" ]]; then
    pg_ephemeral_require
    _step "[1/3] Reusing ephemeral Postgres from parent: ${AAS_PG_CONTAINER}"
elif [[ "$SKIP_INIT" == "true" ]]; then
    _err "--skip-init given but no AAS_PG_CONTAINER in environment — no database to use."
    exit 1
else
    _step "[1/3] Starting ephemeral Postgres..."
    OWNS_DB=1   # mark before bring-up so a mid-startup signal still tears down
    pg_ephemeral_start
    _ok "Ephemeral Postgres ready."
fi

# ============================================
# 2. Start vLLM server(s) with all LoRAs
# ============================================
echo ""
_step "[2/3] Starting vLLM server(s)..."

# Pre-flight: reap leftover vLLM workers ONLY on the ports we're about to use.
# Avoids killing parallel sibling cross-eval runs that are using different ports.
# Selfplay uses 8001/8002; cross_eval defaults to 8001/8003 but can be overridden
# via --red-port / --blue-port for parallel multi-rep dispatch.
echo "[cross_eval] Pre-flight: reaping leftover vLLM workers on ports $RED_PORT and $BLUE_PORT..."
for p in $RED_PORT $BLUE_PORT; do
    if command -v lsof >/dev/null 2>&1; then
        pids=$(lsof -ti :$p 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            echo "[cross_eval] Killing PIDs holding port $p: $pids"
            kill -9 $pids 2>/dev/null || true
        fi
    fi
done
sleep 3
for p in $RED_PORT $BLUE_PORT; do
    if command -v lsof >/dev/null 2>&1 && lsof -ti :$p >/dev/null 2>&1; then
        echo "[cross_eval] FATAL: port $p still in use after cleanup:" >&2
        lsof -i :$p >&2
        exit 1
    fi
done

LOG_DIR="$VLLM_LOG_DIR"
mkdir -p "$LOG_DIR"

if [[ "$SINGLE_GPU" == "true" ]]; then
    # --- SINGLE-GPU MODE: one vLLM server with all red + blue LoRAs ---
    echo "  Single server: GPU $GPU_ID, port $RED_PORT"
    echo "  All LoRA modules (${N_RED} red + ${N_BLUE} blue) on one server"

    ALL_LORA_MODULES=""
    for iter in "${!RED_LORAS[@]}"; do
        ALL_LORA_MODULES="${ALL_LORA_MODULES} red_${iter}=${RED_LORAS[$iter]}"
    done
    for iter in "${!BLUE_LORAS[@]}"; do
        ALL_LORA_MODULES="${ALL_LORA_MODULES} blue_${iter}=${BLUE_LORAS[$iter]}"
    done

    if [[ -z "${ALL_LORA_MODULES// }" ]]; then
        echo "ERROR: No LoRA checkpoints found. Cannot start vLLM server."
        exit 1
    fi

    echo "  LoRA modules:$ALL_LORA_MODULES"

    echo "Starting single vLLM server..."
    setsid env CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
        --model "$BASE_MODEL" \
        --port $RED_PORT \
        --host 0.0.0.0 \
        --gpu-memory-utilization $GPU_MEM \
        --max-model-len $MAX_MODEL_LEN \
        --tensor-parallel-size 1 \
        --trust-remote-code \
        --disable-log-requests \
        --dtype auto \
        --enable-lora \
        --max-lora-rank 64 \
        --max-loras $MAX_ALL_LORAS \
        --lora-modules $ALL_LORA_MODULES \
        > "$LOG_DIR/crosseval_single_${RED_PORT}.log" 2>&1 &
    SINGLE_VLLM_PID=$!
    echo "  vLLM PID: $SINGLE_VLLM_PID"

    echo "Waiting for vLLM server to be ready..."
    TIMEOUT=600
    START_TIME=$(date +%s)
    SERVER_READY=false

    while [[ "$SERVER_READY" == "false" ]]; do
        ELAPSED=$(($(date +%s) - START_TIME))
        if [[ $ELAPSED -ge $TIMEOUT ]]; then
            echo "ERROR: vLLM server did not become ready within ${TIMEOUT}s"
            echo "Check log: $LOG_DIR/crosseval_single_${RED_PORT}.log"
            exit 1
        fi

        if ! kill -0 $SINGLE_VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM server died. Check $LOG_DIR/crosseval_single_${RED_PORT}.log"
            exit 1
        fi

        if curl -s http://localhost:$RED_PORT/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('data') else 1)" 2>/dev/null; then
            SERVER_READY=true
            _ok "vLLM ready (${ELAPSED}s)"
        fi

        if [[ "$SERVER_READY" == "false" ]]; then
            echo "  Waiting... (${ELAPSED}/${TIMEOUT}s)"
            sleep 5
        fi
    done
    _ok "vLLM server ready."

    RED_VLLM_URL="http://localhost:$RED_PORT/v1"
    BLUE_VLLM_URL="http://localhost:$RED_PORT/v1"

else
    # --- DUAL-GPU MODE: separate red and blue vLLM servers ---
    echo "  Red server:  GPU $RED_GPU, port $RED_PORT"
    echo "  Blue server: GPU $BLUE_GPU, port $BLUE_PORT"

    RED_LORA_MODULES=""
    for iter in "${!RED_LORAS[@]}"; do
        RED_LORA_MODULES="${RED_LORA_MODULES} red_${iter}=${RED_LORAS[$iter]}"
    done

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

    echo "Starting red team vLLM server..."
    setsid env CUDA_VISIBLE_DEVICES=$RED_GPU python3 -m vllm.entrypoints.openai.api_server \
        --model "$BASE_MODEL" \
        --port $RED_PORT \
        --host 0.0.0.0 \
        --gpu-memory-utilization $GPU_MEM \
        --max-model-len $MAX_MODEL_LEN \
        --tensor-parallel-size 1 \
        --trust-remote-code \
        --disable-log-requests \
        --dtype auto \
        --enable-lora \
        --max-lora-rank 64 \
        --max-loras $MAX_RED_LORAS \
        --lora-modules $RED_LORA_MODULES \
        > "$LOG_DIR/crosseval_red_${RED_PORT}.log" 2>&1 &
    RED_VLLM_PID=$!
    echo "  Red vLLM PID: $RED_VLLM_PID"

    echo "Starting blue team vLLM server..."
    setsid env CUDA_VISIBLE_DEVICES=$BLUE_GPU python3 -m vllm.entrypoints.openai.api_server \
        --model "$BASE_MODEL" \
        --port $BLUE_PORT \
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
        > "$LOG_DIR/crosseval_blue_${BLUE_PORT}.log" 2>&1 &
    BLUE_VLLM_PID=$!
    echo "  Blue vLLM PID: $BLUE_VLLM_PID"

    echo "Waiting for vLLM servers to be ready..."
    TIMEOUT=600
    START_TIME=$(date +%s)
    RED_READY=false
    BLUE_READY=false

    while [[ "$RED_READY" == "false" || "$BLUE_READY" == "false" ]]; do
        ELAPSED=$(($(date +%s) - START_TIME))
        if [[ $ELAPSED -ge $TIMEOUT ]]; then
            echo "ERROR: vLLM servers did not become ready within ${TIMEOUT}s"
            echo "Check logs: $LOG_DIR/crosseval_red_${RED_PORT}.log and crosseval_blue_${BLUE_PORT}.log"
            exit 1
        fi

        if ! kill -0 $RED_VLLM_PID 2>/dev/null; then
            echo "ERROR: Red vLLM server died. Check $LOG_DIR/crosseval_red_${RED_PORT}.log"
            exit 1
        fi
        if ! kill -0 $BLUE_VLLM_PID 2>/dev/null; then
            echo "ERROR: Blue vLLM server died. Check $LOG_DIR/crosseval_blue_${BLUE_PORT}.log"
            exit 1
        fi

        if [[ "$RED_READY" == "false" ]]; then
            if curl -s http://localhost:$RED_PORT/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('data') else 1)" 2>/dev/null; then
                RED_READY=true
                _ok "Red vLLM ready (${ELAPSED}s)"
            fi
        fi
        if [[ "$BLUE_READY" == "false" ]]; then
            if curl -s http://localhost:$BLUE_PORT/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('data') else 1)" 2>/dev/null; then
                BLUE_READY=true
                _ok "Blue vLLM ready (${ELAPSED}s)"
            fi
        fi

        if [[ "$RED_READY" == "false" || "$BLUE_READY" == "false" ]]; then
            echo "  Waiting... (${ELAPSED}/${TIMEOUT}s)"
            sleep 5
        fi
    done
    _ok "Both vLLM servers ready."

    RED_VLLM_URL="http://localhost:$RED_PORT/v1"
    BLUE_VLLM_URL="http://localhost:$BLUE_PORT/v1"
fi

# ============================================
# 3. Run cross-evaluation
# ============================================
echo ""
_step "[3/3] Running cross-evaluation..."
SUBSET_ARGS=(--pairing-subset "$PAIRING_SUBSET")
if [[ -n "$PAIRING_LIST" ]]; then
    SUBSET_ARGS+=(--pairing-list "$PAIRING_LIST")
fi
EXTRA_CROSS_ARGS=()
[[ "$INCLUDE_BASE" == "false" ]] && EXTRA_CROSS_ARGS+=(--no-include-base)
[[ "$RESUME"       == "true"  ]] && EXTRA_CROSS_ARGS+=(--resume)
[[ "$MATCH_TRAIN_SEEDS" == "true" ]] && EXTRA_CROSS_ARGS+=(--match-train-seeds)
[[ -n "$BLUETEAM_SYSTEM_PROMPT_FILE" ]] && EXTRA_CROSS_ARGS+=(--blueteam-system-prompt-file "$BLUETEAM_SYSTEM_PROMPT_FILE")
python3 util/cross_evaluate.py \
    --selfplay-dir "$SELFPLAY_DIR" \
    --base-model "$BASE_MODEL" \
    --episodes "$EPISODES" \
    --horizon "$HORIZON" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR" \
    --red-vllm-url "$RED_VLLM_URL" \
    --blue-vllm-url "$BLUE_VLLM_URL" \
    --concurrency "$CONCURRENCY" \
    "${SUBSET_ARGS[@]}" \
    "${EXTRA_CROSS_ARGS[@]}"

echo ""
_step "Aggregating results..."
python3 util/cross_evaluate.py \
    --selfplay-dir "$SELFPLAY_DIR" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --aggregate-only

echo ""
_step "Computing pairwise significance (two-proportion / Fisher + McNemar)..."
python3 util/mcnemar_cross_eval.py "$OUTPUT_DIR" || \
    _warn "mcnemar_cross_eval.py failed; heatmaps will render without significance annotations."

echo ""
_step "Generating plots..."
python3 util/plot_cross_eval.py "$OUTPUT_DIR"

echo ""
_banner "========================================"
_ok "Cross-Evaluation Complete"
_banner "========================================"
echo "Results: $OUTPUT_DIR"
echo "Figures: $OUTPUT_DIR/figures/"

# Cleanup handled by EXIT trap
