#!/bin/bash
# Utility ablations — one driver for the two variants we need for the CCS paper.
#
# Variants (--variant):
#   none        (was: run_no_utility_ablation.sh)
#     BLUETEAM_FIXED_ATTACK_PROB=1.0 — every training episode is an attack; blue
#     sees zero benign turns. Demonstrates TPR collapse when utility signal is
#     removed entirely.
#
#   plain-only
#     BLUETEAM_PLAIN_ONLY=1 — blue trains with only "plain" benign queries
#     (adversarial-framed and multi-turn benigns stripped out of the training
#     pool). Demonstrates TPR collapse on adversarially-framed legit requests
#     when the utility distribution is too narrow. Attack curriculum unchanged.
#
# Both variants also skip blueteam early-stopping (BLUETEAM_DISABLE_EARLY_STOP=1)
# so compute budget is held constant — the only thing changing is the benign
# distribution.
#
# Post-training evaluation, both variants (run sequentially on the same 2 GPUs):
#   (1) cross_evaluate.py: 100 attack episodes + 100 benign episodes (full pool)
#   (2) benign_eval per style slice: plain / adversarial / multi_turn
#       so we can report TPR_plain vs TPR_adversarial for the ablated blue.
#
# Usage:
#   # Fresh run (variant = none OR plain-only)
#   ./run_utility_ablation.sh \
#       --variant {none|plain-only} \
#       --redteam-lora <path-to-.../steps_NNNN/sql_agent> \
#       [--base-model <hf-id>] [--num-eval-episodes 100] [--seed 42] \
#       [--gpu-0 N] [--gpu-1 N]
#
#   # Eval-only (skip training, use a prior result dir)
#   ./run_utility_ablation.sh \
#       --result-dir ablations/<variant>/<tag>/ \
#       [--num-eval-episodes 100] [--seed 42] \
#       [--gpu-0 N] [--gpu-1 N]
#
# GPU modes (orthogonal to fresh/eval-only):
#   Default (dual-GPU): two A100s, --gpu-0 and --gpu-1.
#   Single-GPU       : one A100 (~60GB peak), --single-gpu --gpu N.
#                      Eval-only path: cross_eval co-locates red+blue on the same
#                      server (~40GB), then a blue-only server (~25GB) is spun up
#                      for the per-style benign loop. Training in single-GPU
#                      mode is not supported (run_training.sh needs separate
#                      actor/training GPUs).

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
_err()    { printf "${_RED}${_B}✗  ERROR: %s${_0}\n"   "$*" >&2; }
_banner() { printf "${_B}%s${_0}\n"                     "$*"; }

# ── Defaults ───────────────────────────────────────────────────────────────────
VARIANT=""
REDTEAM_LORA=""
RESULT_DIR=""
BASE_MODEL="Snowflake/Arctic-Text2SQL-R1-7B"
NUM_EVAL_EPISODES=100
SEED=42
GPU0=0
GPU1=1
SINGLE_GPU=false
GPU=0
BENIGN_VLLM_PORT=8002
BENIGN_VLLM_GPU_MEM=0.50
BENIGN_VLLM_MAX_LEN=4096
BENIGN_VLLM_CONCURRENCY=8

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --variant)            VARIANT="$2";            shift ;;
        --redteam-lora)       REDTEAM_LORA="$2";       shift ;;
        --result-dir)         RESULT_DIR="$2";         shift ;;
        --base-model)         BASE_MODEL="$2";         shift ;;
        --num-eval-episodes)  NUM_EVAL_EPISODES="$2";  shift ;;
        --seed)               SEED="$2";               shift ;;
        --gpu-0)              GPU0="$2";               shift ;;
        --gpu-1)              GPU1="$2";               shift ;;
        --single-gpu)         SINGLE_GPU=true ;;
        --gpu)                GPU="$2";                shift ;;
        --benign-port)        BENIGN_VLLM_PORT="$2";   shift ;;
        --benign-gpu-mem)     BENIGN_VLLM_GPU_MEM="$2"; shift ;;
        --benign-max-model-len) BENIGN_VLLM_MAX_LEN="$2"; shift ;;
        *) _err "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ "$SINGLE_GPU" == "true" && -z "$RESULT_DIR" ]]; then
    _err "--single-gpu requires --result-dir (eval-only). Training needs two GPUs."
    exit 1
fi

if [[ -z "$VARIANT" && -z "$RESULT_DIR" ]]; then
    _err "--variant {none|plain-only} is required for fresh runs"
    exit 1
fi

if [[ -n "$VARIANT" && "$VARIANT" != "none" && "$VARIANT" != "plain-only" ]]; then
    _err "--variant must be one of: none, plain-only (got '$VARIANT')"
    exit 1
fi

_banner "========================================"
_banner "Agentic AI Security: Utility Ablation ($VARIANT)"
_banner "========================================"
if [[ "$SINGLE_GPU" == "true" ]]; then
    _step "GPU mode: SINGLE-GPU (gpu=$GPU, benign-port=$BENIGN_VLLM_PORT, mem=$BENIGN_VLLM_GPU_MEM)"
else
    _step "GPU mode: DUAL-GPU (gpu-0=$GPU0, gpu-1=$GPU1)"
fi

# ── LoRA validation ────────────────────────────────────────────────────────────
validate_lora() {
    local p="$1" tag="$2"
    [[ -d "$p" ]] || { _err "${tag} LoRA directory not found: $p"; exit 1; }
    [[ -f "$p/adapter_config.json" ]] || {
        _err "${tag} LoRA missing adapter_config.json: $p"; exit 1; }
    [[ -f "$p/adapter_model.safetensors" ]] || {
        _err "${tag} LoRA missing adapter_model.safetensors: $p"; exit 1; }
}

# Recorded LoRA paths may be container-absolute (/app/...) from a training run
# that happened inside Docker. When the same eval is re-run from the host,
# /app doesn't exist. Translate /app/<rest> → ${PWD}/<rest> as a fallback.
resolve_recorded_path() {
    local recorded="$1" tag="$2"
    if [[ -d "$recorded" ]]; then
        printf '%s\n' "$recorded"
        return 0
    fi
    if [[ "$recorded" == /app/* ]]; then
        local host="${PWD}/${recorded#/app/}"
        if [[ -d "$host" ]]; then
            _warn "${tag} path was container-absolute; remapped /app → ${PWD}" >&2
            printf '%s\n' "$host"
            return 0
        fi
    fi
    _err "${tag} path is not reachable from this host: $recorded"
    exit 1
}

# ── Mode dispatch ──────────────────────────────────────────────────────────────
if [[ -z "$RESULT_DIR" ]]; then
    [[ -n "$REDTEAM_LORA" ]] || {
        _err "--redteam-lora is required unless --result-dir is set"; exit 1; }

    REDTEAM_LORA="$(realpath "$REDTEAM_LORA")"
    validate_lora "$REDTEAM_LORA" "redteam"

    REDTEAM_RUN_DIR="$(dirname "$(dirname "$(dirname "$REDTEAM_LORA")")")"
    REDTEAM_ARGS="$REDTEAM_RUN_DIR/args.yaml"
    [[ -f "$REDTEAM_ARGS" ]] || {
        _err "Cannot find args.yaml at: $REDTEAM_ARGS"; exit 1; }
    HORIZON=$(python3 -c "import yaml; print(yaml.safe_load(open('$REDTEAM_ARGS'))['horizon'])")
    echo "Redteam run dir   : $REDTEAM_RUN_DIR"
    echo "Horizon (mirrored): $HORIZON"

    TAG="${VARIANT}-$(date +%Y%m%d-%H%M)-$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c 5 || true)"
    RESULT_DIR="ablations/${VARIANT}/${TAG}"
    mkdir -p "$RESULT_DIR"
    RESULTS_ID="${TAG}"

    # 30-episode cap: enough post-warmup convergence to expose TPR collapse on
    # the omitted benign distribution; ~2 days back-to-back vs ~5–6 days at the
    # canonical 100-ep cap. Disclosed in evaluation.md Pillar 1 footnote.
    ABLATION_MAX_EPS=30
    case "$VARIANT" in
        none)        ABLATION_ENV=(BLUETEAM_FIXED_ATTACK_PROB=1.0 BLUETEAM_DISABLE_EARLY_STOP=1 BLUETEAM_MAX_TRAINING_EPISODES="$ABLATION_MAX_EPS") ;;
        plain-only)  ABLATION_ENV=(BLUETEAM_PLAIN_ONLY=1          BLUETEAM_DISABLE_EARLY_STOP=1 BLUETEAM_MAX_TRAINING_EPISODES="$ABLATION_MAX_EPS") ;;
    esac
    echo "Result dir: $RESULT_DIR"
    echo "Ablation env: ${ABLATION_ENV[@]}"

    _step "[1/3] Training blueteam (variant=$VARIANT)..."
    env "${ABLATION_ENV[@]}" \
        ./run_training.sh \
            --target      blueteam \
            --opponent-lora "$REDTEAM_LORA" \
            --horizon     "$HORIZON" \
            --base-model  "$BASE_MODEL" \
            --results-id  "$RESULTS_ID" \
            --actor-gpu   "$GPU0" \
            --training-gpu "$GPU1" \
        2>&1 | tee "$RESULT_DIR/train.log"

    BLUE_TEAM_DIR="${PWD}/results-${RESULTS_ID}/blueteam"
    BLUE_LORA="$(find "$BLUE_TEAM_DIR" -name sql_agent -type d 2>/dev/null | sort -V | tail -n 1)"
    [[ -n "$BLUE_LORA" ]] || { _err "No sql_agent checkpoint under $BLUE_TEAM_DIR"; exit 1; }
    BLUE_LORA="$(realpath "$BLUE_LORA")"
    validate_lora "$BLUE_LORA" "trained blueteam"

    echo "$BLUE_LORA"    > "$RESULT_DIR/blueteam_lora_path.txt"
    echo "$REDTEAM_LORA" > "$RESULT_DIR/redteam_lora_path.txt"
    echo "$HORIZON"      > "$RESULT_DIR/horizon.txt"
    echo "$BASE_MODEL"   > "$RESULT_DIR/base_model.txt"
    echo "$VARIANT"      > "$RESULT_DIR/variant.txt"
    ln -sfn "../../results-${RESULTS_ID}" "$RESULT_DIR/training"
    _ok "Blueteam LoRA recorded → $RESULT_DIR/blueteam_lora_path.txt"

else
    [[ -d "$RESULT_DIR" ]] || { _err "--result-dir does not exist: $RESULT_DIR"; exit 1; }
    RESULT_DIR="$(realpath "$RESULT_DIR")"

    for f in blueteam_lora_path.txt redteam_lora_path.txt horizon.txt base_model.txt; do
        [[ -f "$RESULT_DIR/$f" ]] || {
            _err "$RESULT_DIR/$f is missing — cannot resume"; exit 1; }
    done

    BLUE_LORA_RAW="$(cat "$RESULT_DIR/blueteam_lora_path.txt")"
    REDTEAM_LORA_RAW="$(cat "$RESULT_DIR/redteam_lora_path.txt")"
    HORIZON="$(cat "$RESULT_DIR/horizon.txt")"
    BASE_MODEL="$(cat "$RESULT_DIR/base_model.txt")"
    VARIANT="$(cat "$RESULT_DIR/variant.txt" 2>/dev/null || echo '(legacy)')"
    BLUE_LORA="$(resolve_recorded_path "$BLUE_LORA_RAW" "recorded blueteam")"
    REDTEAM_LORA="$(resolve_recorded_path "$REDTEAM_LORA_RAW" "recorded redteam")"
    validate_lora "$BLUE_LORA"    "recorded blueteam"
    validate_lora "$REDTEAM_LORA" "recorded redteam"
    # Persist host-resolved paths so subsequent runs (and downstream tools that
    # read these files directly) don't have to re-do the /app translation.
    if [[ "$BLUE_LORA" != "$BLUE_LORA_RAW" ]]; then
        echo "$BLUE_LORA" > "$RESULT_DIR/blueteam_lora_path.txt"
    fi
    if [[ "$REDTEAM_LORA" != "$REDTEAM_LORA_RAW" ]]; then
        echo "$REDTEAM_LORA" > "$RESULT_DIR/redteam_lora_path.txt"
    fi
    _ok "Eval-only mode (variant=$VARIANT) — resuming from: $RESULT_DIR"
fi

# ── Evaluation ─────────────────────────────────────────────────────────────────
_step "[2/3] Attack + full-pool benign eval via cross_evaluate.py..."
EVAL_VIEW="$RESULT_DIR/eval_view"
# Idempotent symlink rebuild — preserves any prior cross_eval/ and benign_only/
# output so cross_evaluate.py --resume can skip already-completed pairings.
mkdir -p "$EVAL_VIEW/iter_1/redteam/checkpoints"
mkdir -p "$EVAL_VIEW/iter_1/blueteam/checkpoints"
RED_STEPS="$(dirname "$(realpath "$REDTEAM_LORA")")"
BLUE_STEPS="$(dirname "$(realpath "$BLUE_LORA")")"
ln -sfn "$RED_STEPS"  "$EVAL_VIEW/iter_1/redteam/checkpoints/$(basename "$RED_STEPS")"
ln -sfn "$BLUE_STEPS" "$EVAL_VIEW/iter_1/blueteam/checkpoints/$(basename "$BLUE_STEPS")"

CROSS_EVAL_GPU_ARGS=()
if [[ "$SINGLE_GPU" == "true" ]]; then
    CROSS_EVAL_GPU_ARGS+=(--single-gpu --gpu "$GPU")
else
    CROSS_EVAL_GPU_ARGS+=(--red-gpu "$GPU0" --blue-gpu "$GPU1")
fi

./run_cross_eval.sh \
    --selfplay-dir    "$EVAL_VIEW" \
    --base-model      "$BASE_MODEL" \
    --episodes        "$NUM_EVAL_EPISODES" \
    --horizon         "$HORIZON" \
    --seed            "$SEED" \
    "${CROSS_EVAL_GPU_ARGS[@]}" \
    --pairing-subset  custom \
    --pairing-list    "red_1:blue_1" \
    --no-include-base \
    --resume \
    2>&1 | tee "$RESULT_DIR/cross_eval.log"

_step "[3/3] Per-style benign eval (plain / adversarial / multi_turn)..."
# run_cross_eval.sh's EXIT trap kills its vLLM server, so we manage our own
# blue-only vLLM here (mirrors util/run_benign_eval.sh). Each style-filtered
# run writes into its own output dir so results are disjoint.

# Skip the vLLM spin-up entirely if all three styles already completed (resume).
BENIGN_ALL_DONE=true
for STYLE in plain adversarial multi_turn; do
    SUMMARY="$RESULT_DIR/benign_style_${STYLE}/benign_only/blue_1/summary.json"
    if [[ ! -f "$SUMMARY" ]]; then
        BENIGN_ALL_DONE=false
        break
    fi
done

if [[ "$BENIGN_ALL_DONE" == "true" ]]; then
    _ok "All three style summaries already present — skipping benign vLLM."
else
    BENIGN_VLLM_GPU="$GPU1"
    [[ "$SINGLE_GPU" == "true" ]] && BENIGN_VLLM_GPU="$GPU"

    # Clamp gpu-mem at 0.90 — same guard used by run_cross_eval.sh / run_benign_eval.sh.
    if awk -v g="$BENIGN_VLLM_GPU_MEM" 'BEGIN{exit !(g > 0.90)}'; then
        _warn "--benign-gpu-mem=$BENIGN_VLLM_GPU_MEM exceeds 0.90 cap; clamping."
        BENIGN_VLLM_GPU_MEM=0.90
    fi

    BENIGN_VLLM_PID=""
    cleanup_benign_vllm() {
        if [[ -n "$BENIGN_VLLM_PID" ]] && kill -0 "$BENIGN_VLLM_PID" 2>/dev/null; then
            echo ""
            echo "[cleanup] Stopping benign vLLM (PID=$BENIGN_VLLM_PID)..."
            kill "$BENIGN_VLLM_PID" 2>/dev/null || true
            for _ in $(seq 1 10); do
                kill -0 "$BENIGN_VLLM_PID" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "$BENIGN_VLLM_PID" 2>/dev/null; then
                echo "[cleanup] WARNING: vLLM still alive after 10s. SIGKILL — may brick GPU."
                kill -9 "$BENIGN_VLLM_PID" 2>/dev/null || true
            fi
        fi
    }
    trap cleanup_benign_vllm EXIT

    # Postgres / MCP must be up for env reset between episodes. run_cross_eval.sh
    # already started them; this is idempotent so it's safe to re-run if not.
    ./script/init.sh > /dev/null 2>&1 || _warn "script/init.sh non-zero — Postgres/MCP may already be running."

    # Refuse to start if the port is occupied — avoids silently hitting a stale server.
    if curl -s "http://localhost:${BENIGN_VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        _err "Port ${BENIGN_VLLM_PORT} is already serving — refusing to start a second vLLM. Free it first."
        exit 1
    fi

    LOG_DIR=/tmp/vllm_logs
    mkdir -p "$LOG_DIR"
    BENIGN_LOG="$LOG_DIR/ablation_benign_$(basename "$RESULT_DIR").log"

    _step "  Starting blue-only vLLM (GPU $BENIGN_VLLM_GPU, port $BENIGN_VLLM_PORT, mem=$BENIGN_VLLM_GPU_MEM, max-len=$BENIGN_VLLM_MAX_LEN)"
    CUDA_VISIBLE_DEVICES=$BENIGN_VLLM_GPU python3 -m vllm.entrypoints.openai.api_server \
        --model "$BASE_MODEL" \
        --port "$BENIGN_VLLM_PORT" \
        --host 0.0.0.0 \
        --gpu-memory-utilization "$BENIGN_VLLM_GPU_MEM" \
        --max-model-len "$BENIGN_VLLM_MAX_LEN" \
        --tensor-parallel-size 1 \
        --trust-remote-code \
        --disable-log-requests \
        --dtype auto \
        --enable-lora \
        --max-lora-rank 64 \
        --max-loras 2 \
        --lora-modules "blue_1=${BLUE_LORA}" \
        > "$BENIGN_LOG" 2>&1 &
    BENIGN_VLLM_PID=$!
    echo "  PID: $BENIGN_VLLM_PID  log: $BENIGN_LOG"

    TIMEOUT=600
    START_TIME=$(date +%s)
    SERVER_READY=false
    while [[ "$SERVER_READY" == "false" ]]; do
        ELAPSED=$(($(date +%s) - START_TIME))
        if [[ $ELAPSED -ge $TIMEOUT ]]; then
            _err "Benign vLLM did not become ready within ${TIMEOUT}s. Check $BENIGN_LOG."
            exit 1
        fi
        if ! kill -0 "$BENIGN_VLLM_PID" 2>/dev/null; then
            _err "Benign vLLM died. Check $BENIGN_LOG."
            exit 1
        fi
        if curl -s "http://localhost:${BENIGN_VLLM_PORT}/v1/models" \
            | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('data') else 1)" 2>/dev/null; then
            SERVER_READY=true
            _ok "Benign vLLM ready (${ELAPSED}s)"
        else
            sleep 5
        fi
    done

    BENIGN_VLLM_URL="http://localhost:${BENIGN_VLLM_PORT}/v1"

    for STYLE in plain adversarial multi_turn; do
        OUT="$RESULT_DIR/benign_style_${STYLE}"
        SUMMARY="$OUT/benign_only/blue_1/summary.json"
        if [[ -f "$SUMMARY" ]]; then
            _ok "  style=$STYLE already complete — skipping."
            continue
        fi
        _step "  style=$STYLE  →  $OUT"
        python3 util/run_benign_eval.py \
            --selfplay-dir    "$EVAL_VIEW" \
            --base-model      "$BASE_MODEL" \
            --output-dir      "$OUT" \
            --episodes        "$NUM_EVAL_EPISODES" \
            --seed            "$SEED" \
            --style-filter    "$STYLE" \
            --blue-vllm-url   "$BENIGN_VLLM_URL" \
            --concurrency     "$BENIGN_VLLM_CONCURRENCY" \
            --no-include-base \
            --resume \
            2>&1 | tee "$RESULT_DIR/benign_style_${STYLE}.log"
    done

    cleanup_benign_vllm
    BENIGN_VLLM_PID=""
    trap - EXIT
fi

_banner "========================================"
_ok  "Utility Ablation ($VARIANT) Complete"
_banner "========================================"
echo "Result dir  : $RESULT_DIR"
echo "Blueteam LoRA: $BLUE_LORA"
echo ""
echo "Summaries:"
echo "  Attack eval  : $EVAL_VIEW/cross_eval/pairings/red_1_blue_1/summary.json"
echo "  Benign full  : $EVAL_VIEW/cross_eval/benign_only/blue_1/summary.json"
for STYLE in plain adversarial multi_turn; do
    echo "  Benign $STYLE : $RESULT_DIR/benign_style_${STYLE}/benign_only/blue_1/summary.json"
done
