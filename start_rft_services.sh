#!/bin/bash
# Start services for RFT training
# - Starts PostgreSQL and MCP server
# - Starts one vLLM instance using start_vllm.py with experiments/sql_training.json:
#   - GPU 0, Port 8000: Coach model (DeepSeek-R1-Distill-Qwen-32B) for RFT generation
#   - GPU 1: Reserved for training process (hardcoded in MARFT/marft/mas/mas.py)

set -e  # Exit on error
set -o pipefail

echo "================================"
echo "Starting RFT Training Services"
echo "================================"

# Set HuggingFace token for model access
echo "HF_TOKEN: $HF_TOKEN"

# Prevent CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================
# 1. Start PostgreSQL and MCP Server
# ============================================
echo ""
echo "[1/3] Initializing PostgreSQL and MCP server..."
./script/init.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Database/MCP initialization failed"
    exit 1
fi
echo "✓ PostgreSQL and MCP server ready"

# ============================================
# 2. Start vLLM Servers using start_vllm.py
# ============================================
echo ""
echo "[2/2] Starting vLLM server (coach only)..."
echo "      Config: experiments/sql_training.json"
echo "      GPU 0: Coach model (DeepSeek-R1-Distill-Qwen-32B) on port 8000"
echo "      Logs: /tmp/vllm_logs/"

# Start vLLM fleet in background
python3 start_vllm.py --config experiments/sql_training.json --timeout 600 --wait-only &
VLLM_FLEET_PID=$!

# Wait for registry to be written
echo "Waiting for vLLM servers to initialize..."
TIMEOUT=660
START_TIME=$(date +%s)
REGISTRY_PATH="/tmp/vllm_coach_registry.json"

while true; do
    if ! kill -0 $VLLM_FLEET_PID 2>/dev/null; then
        echo "ERROR: vLLM fleet process died unexpectedly"
        echo "Check logs in /tmp/vllm_logs/"
        exit 1
    fi

    if [ -f "$REGISTRY_PATH" ]; then
        # Verify registry has content
        if python3 -c "import json; data = json.load(open('$REGISTRY_PATH')); exit(0 if data else 1)" 2>/dev/null; then
            echo "✓ vLLM servers ready"
            break
        fi
    fi

    ELAPSED=$(($(date +%s) - START_TIME))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: vLLM servers timed out after $TIMEOUT seconds"
        echo "Check logs in /tmp/vllm_logs/"
        kill $VLLM_FLEET_PID 2>/dev/null
        exit 1
    fi

    echo "  Waiting... ($ELAPSED / $TIMEOUT s)"
    sleep 5
done

# ============================================
# Export environment variables from registry
# ============================================
echo ""
echo "Reading vLLM registry..."

# Extract URLs from registry
COACH_VLLM_URL=$(python3 -c "import json; reg = json.load(open('$REGISTRY_PATH')); print(reg['coach']['url'])")/v1

export COACH_VLLM_URL
export VLLM_COACH_FLEET_PID=$VLLM_FLEET_PID

echo ""
echo "================================"
echo "✓ All services ready!"
echo "================================"
echo "Coach vLLM:   $COACH_VLLM_URL (GPU 0)"
echo "vLLM Coach PID: $VLLM_COACH_FLEET_PID"
echo "Logs: /tmp/vllm_logs/coach.log"
echo "Registry: $REGISTRY_PATH"
echo ""
echo "To use in Python scripts:"
echo "  import os"
echo "  coach_url = os.environ.get('COACH_VLLM_URL', 'http://localhost:8000/v1')"
echo ""
