#!/bin/bash
# Agent-vs-Human baseline driver.
#
# For each cell in the manifest (base_model × variant), starts a vLLM
# server once per base_model and calls util/jailbreak_baseline.py with
# --cell-index for every variant that shares the model. Once every cell
# has been replayed N times, emits the four \input-ready LaTeX tables
# (Q1a, Q1b, Q2, non-determinism) plus the four per-cell outcome panels
# referenced from draft.tex §sec:eval-human.
#
# The RL-vs-manual comparison table (§sec:eval-rl-vs-human) is emitted
# separately by util/eval_rl_vs_human.py and assumes a cross_evaluation
# run already exists for each base. Supply --rl-config to wire that in;
# omit it to skip.
#
# Usage:
#   ./run_human_baseline.sh \
#       --manifest configs/human_baseline_manifest.json \
#       --num-seeds 5 \
#       --gpu 0 \
#       [--tables-config configs/human_baseline_tables.json] \
#       [--rl-config configs/rl_vs_human.json] \
#       [--plot-only]
#
# Manifest cells must specify "model", "tag", "variant", "port". Provide
# "system_prompt_file" only for the "unprotected" variant; omit it for
# the manually-protected variant (the default sql_system_prompt applies).

set -e
set -o pipefail

MANIFEST=""
NUM_SEEDS=5
GPU_ID=0
GPU_MEM="0.90"
MAX_MODEL_LEN="16384"
TABLES_CONFIG=""
RL_CONFIG=""
AUDIT_DIR="data/human_attack_audit"
OUT_JSON_DIR="data/rl_vs_human"
OUT_TABLES_DIR="figures/human_baseline"
OUT_OUTCOMES_DIR="figures/jailbreak_outcomes"
OUT_RL_TEX="figures/rl_vs_human/rl_vs_human_table.tex"
DATASET="new_jailbreaks.txt"
PLOT_ONLY=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --manifest) MANIFEST="$2"; shift ;;
        --num-seeds) NUM_SEEDS="$2"; shift ;;
        --gpu) GPU_ID="$2"; shift ;;
        --gpu-mem) GPU_MEM="$2"; shift ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift ;;
        --tables-config) TABLES_CONFIG="$2"; shift ;;
        --rl-config) RL_CONFIG="$2"; shift ;;
        --audit-dir) AUDIT_DIR="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --plot-only) PLOT_ONLY=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$MANIFEST" ]]; then
    echo "ERROR: --manifest is required."
    exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found: $MANIFEST"
    exit 1
fi

mkdir -p "$AUDIT_DIR" "$OUT_TABLES_DIR" "$OUT_OUTCOMES_DIR"

# ---- Plot / table emission only ---------------------------------------------
emit_tables_and_plots() {
    echo ""
    echo "[plot] util/plot_jailbreak_outcomes.py"
    python3 util/plot_jailbreak_outcomes.py \
        --audit-dir "$AUDIT_DIR" \
        --out "$OUT_OUTCOMES_DIR"

    if [[ -n "$TABLES_CONFIG" ]]; then
        echo "[plot] util/human_baseline_tables.py"
        python3 util/human_baseline_tables.py \
            --audit-dir "$AUDIT_DIR" \
            --dataset "$DATASET" \
            --config "$TABLES_CONFIG" \
            --out-dir "$OUT_TABLES_DIR"
    else
        echo "[skip] human_baseline_tables (no --tables-config)"
    fi

    echo "[plot] util/nondeterminism_audit.py"
    python3 util/nondeterminism_audit.py \
        --audit-dir "$AUDIT_DIR" \
        --out "$OUT_TABLES_DIR/nondeterminism_audit.tex"

    if [[ -n "$RL_CONFIG" ]]; then
        echo "[plot] util/eval_rl_vs_human.py"
        python3 util/eval_rl_vs_human.py \
            --config "$RL_CONFIG" \
            --audit-dir "$AUDIT_DIR" \
            --out-json-dir "$OUT_JSON_DIR" \
            --out-tex "$OUT_RL_TEX"
    else
        echo "[skip] eval_rl_vs_human (no --rl-config)"
    fi
}

if [[ "$PLOT_ONLY" == "true" ]]; then
    emit_tables_and_plots
    exit 0
fi

# ---- Sanity checks ---------------------------------------------------------
for var in system_prompt_file; do :; done
UNPROTECTED_PROMPT=$(python3 -c '
import json, sys
m = json.load(open("'"$MANIFEST"'"))
for c in m["cells"]:
    p = c.get("system_prompt_file")
    if p:
        print(p); break
')
if [[ -n "$UNPROTECTED_PROMPT" ]] && grep -q "TODO: PASTE OLD" "$UNPROTECTED_PROMPT" 2>/dev/null; then
    echo "ERROR: $UNPROTECTED_PROMPT still contains the TODO placeholder."
    echo "Paste the pre-patch blue-team system prompt into that file before running."
    exit 1
fi

# ---- Cleanup trap ----------------------------------------------------------
VLLM_PID=""
cleanup_vllm() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[cleanup] SIGTERM to vLLM PID=$VLLM_PID"
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 10
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[cleanup] WARNING: vLLM still alive; SIGKILL (may brick GPU)."
            kill -9 "$VLLM_PID" 2>/dev/null || true
        fi
    fi
}
trap cleanup_vllm EXIT

# ---- Postgres / MCP --------------------------------------------------------
echo "[init] Starting Postgres + MCP..."
./script/init.sh

# ---- Iterate cells grouped by model ---------------------------------------
# Build a newline-separated list: "<model>|<port>|<cell_index>" per cell.
CELL_INFO=$(python3 -c '
import json
m = json.load(open("'"$MANIFEST"'"))
for i, c in enumerate(m["cells"]):
    print(f"{c[\"model\"]}|{c.get(\"port\", 8001)}|{i}")
')

LOG_DIR="/tmp/vllm_logs"
mkdir -p "$LOG_DIR"

CURRENT_MODEL=""
CURRENT_PORT=""

start_vllm() {
    local model="$1"
    local port="$2"
    local log="$LOG_DIR/human_baseline_$(echo "$model" | tr '/' '_').log"
    echo "[vLLM] Starting $model on GPU $GPU_ID port $port (log: $log)"
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --port "$port" \
        --host 0.0.0.0 \
        --gpu-memory-utilization "$GPU_MEM" \
        --max-model-len "$MAX_MODEL_LEN" \
        --tensor-parallel-size 1 \
        --trust-remote-code \
        --disable-log-requests \
        --dtype auto \
        > "$log" 2>&1 &
    VLLM_PID=$!
    echo "[vLLM]   PID=$VLLM_PID"

    local timeout=900
    local start=$(date +%s)
    while true; do
        local elapsed=$(( $(date +%s) - start ))
        if [[ $elapsed -ge $timeout ]]; then
            echo "[vLLM] ERROR: did not become ready within ${timeout}s (log: $log)"
            return 1
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[vLLM] ERROR: process died (log: $log)"
            return 1
        fi
        if curl -s "http://localhost:$port/v1/models" | \
            python3 -c 'import sys,json; d=json.load(sys.stdin); exit(0 if d.get("data") else 1)' 2>/dev/null; then
            echo "[vLLM] ready (${elapsed}s)"
            return 0
        fi
        sleep 5
    done
}

stop_vllm() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[vLLM] Stopping PID=$VLLM_PID"
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 10
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[vLLM] SIGKILL (may brick GPU)"
            kill -9 "$VLLM_PID" 2>/dev/null || true
        fi
        VLLM_PID=""
    fi
}

while IFS='|' read -r MODEL PORT CELL_IDX; do
    [[ -z "$MODEL" ]] && continue
    if [[ "$MODEL|$PORT" != "$CURRENT_MODEL|$CURRENT_PORT" ]]; then
        stop_vllm
        start_vllm "$MODEL" "$PORT"
        CURRENT_MODEL="$MODEL"
        CURRENT_PORT="$PORT"
    fi
    echo "[cell] index=$CELL_IDX model=$MODEL port=$PORT"
    python3 util/jailbreak_baseline.py \
        --manifest "$MANIFEST" \
        --cell-index "$CELL_IDX" \
        --num-seeds "$NUM_SEEDS"
done <<< "$CELL_INFO"

stop_vllm

# ---- Emit plots + tables ---------------------------------------------------
emit_tables_and_plots

echo ""
echo "========================================"
echo "Agent-vs-Human baseline complete"
echo "========================================"
echo "Audit:          $AUDIT_DIR"
echo "Outcome panels: $OUT_OUTCOMES_DIR"
echo "Tables:         $OUT_TABLES_DIR"
if [[ -n "$RL_CONFIG" ]]; then
    echo "RL vs manual:   $OUT_RL_TEX"
fi
