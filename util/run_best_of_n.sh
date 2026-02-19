#!/usr/bin/env bash

# ──────────────────────────────────────────────────────────────────────────────
# run_best_of_n.sh — End-to-end Best-of-N sampling experiment
#
# Starts the vLLM fleet (attacker + victim), waits for readiness, runs the
# best_of_n_sampling.py script, and cleans up on exit.
#
# THIS SCRIPT IS DESIGNED TO RUN INSIDE THE DOCKER CONTAINER.
#
# Usage:
#   # Basic run (uses defaults: experiments/best_of_n.json, N=16)
#   bash util/run_best_of_n.sh
#
#   # Custom config and N
#   bash util/run_best_of_n.sh --config experiments/my_config.json -N 32
#
#   # With seed file and specific categories
#   bash util/run_best_of_n.sh -N 64 --seed-file new_jailbreaks.txt --categories customer_column
#
#   # Skip server startup (if vLLM servers are already running)
#   bash util/run_best_of_n.sh --no-server -N 16
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
VLLM_CONFIG="${PROJECT_ROOT}/experiments/best_of_n.json"
REGISTRY_PATH="/tmp/vllm_registry.json"
N=16
CONCURRENCY=4
TEMPERATURE=1.0
SEED_FILE=""
OUTPUT_FILE=""
MCP_SERVER_PATH="/app/mcp/postgres.py"
CATEGORIES=""
TARGETS=""
NO_SERVER=false
TIMEOUT=600

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)        VLLM_CONFIG="$2";       shift 2;;
        --registry)      REGISTRY_PATH="$2";     shift 2;;
        -N|--n)          N="$2";                 shift 2;;
        --concurrency)   CONCURRENCY="$2";       shift 2;;
        --temperature)   TEMPERATURE="$2";       shift 2;;
        --seed-file)     SEED_FILE="$2";         shift 2;;
        --output-file)   OUTPUT_FILE="$2";       shift 2;;
        --mcp-server)    MCP_SERVER_PATH="$2";   shift 2;;
        --categories)    CATEGORIES="$2";        shift 2;;
        --targets)       TARGETS="$2";           shift 2;;
        --timeout)       TIMEOUT="$2";           shift 2;;
        --no-server)     NO_SERVER=true;         shift;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Orchestrates Best-of-N sampling: start vLLM servers, run experiment, clean up."
            echo ""
            echo "Options:"
            echo "  --config PATH        vLLM fleet config JSON (default: experiments/best_of_n.json)"
            echo "  --registry PATH      Registry file path (default: /tmp/vllm_registry.json)"
            echo "  -N, --n NUM          Attacks per honeypot target (default: 16)"
            echo "  --concurrency NUM    Max concurrent attack tests (default: 4)"
            echo "  --temperature FLOAT  Attacker sampling temperature (default: 1.0)"
            echo "  --seed-file PATH     File with seed attack prompts"
            echo "  --output-file PATH   Output report path"
            echo "  --mcp-server PATH    MCP server script path (default: /app/mcp/postgres.py)"
            echo "  --categories CATS    Honeypot categories to test (space-separated)"
            echo "  --targets IDS        Specific honeypot identifiers to test"
            echo "  --timeout SECS       Server startup timeout (default: 600)"
            echo "  --no-server          Skip vLLM server startup (use existing servers)"
            echo "  -h, --help           Show this help"
            exit 0;;
        *)
            echo "Unknown argument: $1"
            exit 1;;
    esac
done

# ── Trap for cleanup ──────────────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
    echo ""
    echo "━━━ Cleaning up ━━━"
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM fleet (PID=$VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        echo "vLLM fleet stopped."
    fi
    # Registry cleanup is handled by start_vllm.py's shutdown handler
    echo "Done."
}
trap cleanup EXIT INT TERM

# ── Print experiment config ───────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Best-of-N Sampling Experiment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Config:       $VLLM_CONFIG"
echo "N per target: $N"
echo "Concurrency:  $CONCURRENCY"
echo "Temperature:  $TEMPERATURE"
echo "Registry:     $REGISTRY_PATH"
echo "Seed file:    ${SEED_FILE:-'(none)'}"
echo "Categories:   ${CATEGORIES:-'(all)'}"
echo "Targets:      ${TARGETS:-'(all)'}"
echo "No server:    $NO_SERVER"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Step 1: Start vLLM fleet ──────────────────────────────────────────────────
if [[ "$NO_SERVER" == false ]]; then
    if [[ ! -f "$VLLM_CONFIG" ]]; then
        echo "ERROR: vLLM config not found: $VLLM_CONFIG"
        echo "Create it or use --config to specify a different path."
        echo "See experiments/best_of_n.json for an example."
        exit 1
    fi

    echo "━━━ Step 1: Starting vLLM server fleet ━━━"
    echo "Config: $VLLM_CONFIG"
    echo ""

    # Remove stale registry and logs
    rm -f "$REGISTRY_PATH"
    rm -rf /tmp/vllm_logs

    # Start vLLM fleet in background (wait-only mode — no auto-restart, we handle cleanup)
    export VLLM_REGISTRY="$REGISTRY_PATH"
    python3 "$PROJECT_ROOT/start_vllm.py" \
        --config "$VLLM_CONFIG" \
        --timeout "$TIMEOUT" \
        --wait-only &
    VLLM_PID=$!
    echo "vLLM fleet started (PID=$VLLM_PID)"
    echo ""

    # Wait for registry to appear (start_vllm.py writes it after all servers are ready)
    echo "Waiting for registry at $REGISTRY_PATH..."
    elapsed=0
    while [[ ! -f "$REGISTRY_PATH" ]] && [[ $elapsed -lt $TIMEOUT ]]; do
        sleep 5
        elapsed=$((elapsed + 5))
        # Check if start_vllm.py died
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM fleet process died during startup."
            exit 1
        fi
        
        # Display live status from logs
        echo "  --- Status (${elapsed}s / ${TIMEOUT}s) ---"
        if ls /tmp/vllm_logs/*.log 1> /dev/null 2>&1; then
            for log in /tmp/vllm_logs/*.log; do
                sid=$(basename "$log" .log)
                # Show last line to indicate progress (loading bars, etc.)
                # Use tr to handle potential carriage returns from progress bars
                last_line=$(tail -n 1 "$log" | tr -d '\r' | cut -c 1-120)
                echo "    [$sid] $last_line"
            done
        else
            echo "    (Initializing...)"
        fi
        echo ""
    done

    if [[ ! -f "$REGISTRY_PATH" ]]; then
        echo "ERROR: Registry not found after ${TIMEOUT}s. Something went wrong."
        exit 1
    fi
    echo "Registry found!"
    echo ""
    cat "$REGISTRY_PATH"
    echo ""
else
    echo "━━━ Step 1: Skipped (--no-server) ━━━"
    if [[ ! -f "$REGISTRY_PATH" ]]; then
        echo "WARNING: Registry not found at $REGISTRY_PATH."
        echo "Make sure vLLM servers are running and the registry exists."
    else
        echo "Using existing registry:"
        cat "$REGISTRY_PATH"
    fi
    echo ""
fi

# ── Step 2: Run Best-of-N sampling ───────────────────────────────────────────
echo "━━━ Step 2: Running Best-of-N sampling ━━━"

# Build the python command
PYTHON_CMD="python3 ${PROJECT_ROOT}/util/best_of_n_sampling.py"
PYTHON_CMD+=" --registry $REGISTRY_PATH"
PYTHON_CMD+=" -N $N"
PYTHON_CMD+=" --concurrency $CONCURRENCY"
PYTHON_CMD+=" --temperature $TEMPERATURE"
PYTHON_CMD+=" --mcp-server-path $MCP_SERVER_PATH"
PYTHON_CMD+=" --timeout $TIMEOUT"  # use script timeout

if [[ -n "$SEED_FILE" ]]; then
    PYTHON_CMD+=" --seed-file $SEED_FILE"
fi
if [[ -n "$OUTPUT_FILE" ]]; then
    PYTHON_CMD+=" --output-file $OUTPUT_FILE"
fi
if [[ -n "$CATEGORIES" ]]; then
    PYTHON_CMD+=" --categories $CATEGORIES"
fi
if [[ -n "$TARGETS" ]]; then
    PYTHON_CMD+=" --targets $TARGETS"
fi

echo "Command: $PYTHON_CMD"
echo ""

eval "$PYTHON_CMD"

echo ""
echo "━━━ Experiment complete ━━━"
