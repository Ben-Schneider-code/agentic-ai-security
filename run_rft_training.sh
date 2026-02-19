#!/bin/bash
# Wrapper script to run RFT training with proper service initialization
# Usage: ./run_rft_training.sh [additional args for train_redteam_rft.py]

set -e

echo "========================================"
echo "RFT Training Runner"
echo "========================================"

# Check if services are already running
COACH_RUNNING=$(pgrep -f "vllm.entrypoints.openai.api_server.*8000" || echo "")
STUDENT_RUNNING=$(pgrep -f "vllm.entrypoints.openai.api_server.*8001" || echo "")

if [ -z "$COACH_RUNNING" ] || [ -z "$STUDENT_RUNNING" ]; then
    echo "Starting services..."
    source /app/start_rft_services.sh
else
    echo "Services already running, using existing instances"
    export COACH_VLLM_URL="http://localhost:8000/v1"
    export STUDENT_VLLM_URL="http://localhost:8001/v1"
fi

# Run RFT training
echo ""
echo "========================================"
echo "Starting RFT Training"
echo "========================================"
echo "Coach URL: $COACH_VLLM_URL"
echo "Student URL: $STUDENT_VLLM_URL"
echo ""

cd /app/MARFT
python3 marft/scripts/train_redteam_rft.py \
    --coach_vllm_url "$COACH_VLLM_URL" \
    --student_vllm_url "$STUDENT_VLLM_URL" \
    "$@"

echo ""
echo "========================================"
echo "RFT Training Complete"
echo "========================================"
