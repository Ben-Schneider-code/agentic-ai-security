#!/bin/bash
set -e

echo "========================================"
echo "Agentic AI Security: Self-Play Orchestrator"
echo "========================================"

# Make sure environments are clean
pkill -f start_vllm.py || true
pkill -9 -f vllm.entrypoints || true
sleep 5

echo "[1/3] Phase 1: Training Red Team..."
if ! ./run_training.sh --target redteam; then
    echo "Red Team training failed!"
    exit 1
fi

echo "[2/3] Locating trained Red Team LoRA adapter..."
LATEST_CKPT=$(find MARFT/marft/scripts/results/redteam_sql_experiment -name "sql_agent" -type d | sort -r | head -n 1)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: Could not find any trained Red Team LoRA checkpoint!"
    exit 1
fi

LATEST_CKPT=$(realpath "$LATEST_CKPT")
echo "Found latest Red Team LoRA: $LATEST_CKPT"

echo "[3/3] Phase 2: Training Blue Team against Red Team LoRA..."
if ! ./run_training.sh --target blueteam --opponent-lora "$LATEST_CKPT"; then
    echo "Blue Team training failed!"
    exit 1
fi

echo "========================================"
echo "Self-Play Alignment Complete."
echo "========================================"
