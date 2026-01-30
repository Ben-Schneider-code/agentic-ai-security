#!/bin/bash
# Set HuggingFace token for model access
echo HF_TOKEN:
echo $HF_TOKEN

# Propagate exit code from script
set -o pipefail

# Start common services (vLLM, Postgres)
source /app/start_common_services.sh

# Run the redteam training
echo "Starting redteam training..."
cd /app/MARFT/marft/scripts
# Run the redteam training
./sample_redteam_script.sh 2>&1 | tee $REDTEAM_OUTPUT_LOG

if [ $? -ne 0 ]; then
    echo ""
    echo "========================================================"
    echo "FATAL ERROR: Redteam training script failed!"
    echo "========================================================"
    echo "Tail of output log:"
    tail -n 50 $REDTEAM_OUTPUT_LOG
fi

# Keep container alive
wait $MODEL_PID
