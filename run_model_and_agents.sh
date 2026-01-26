#!/bin/bash
# Set HuggingFace token for model access
echo HF_TOKEN:
echo $HF_TOKEN

# Propagate exit code from script
set -o pipefail

# Prevent CUDA memory fragmentation (essential for long training runs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_SERVER_LOG=/app/model_server.log
REDTEAM_OUTPUT_LOG=/app/redteam_output.log
touch $MODEL_SERVER_LOG
touch $REDTEAM_OUTPUT_LOG

# Start the model server in the background
echo "Starting vLLM model server..."
python3 /app/host_models.py > $MODEL_SERVER_LOG 2>&1 &
MODEL_PID=$!

# Wait for model server to start and load model (60 seconds)
echo "Waiting for model server to initialize..."
sleep 60

# Check if model server is still running
if ! kill -0 $MODEL_PID 2>/dev/null; then
    echo "ERROR: Model server failed to start. Check $MODEL_SERVER_LOG"
    cat $MODEL_SERVER_LOG
    exit 1
fi

echo "Model server started successfully"

# Start PostgreSQL and initialize database
echo "Initializing database..."
/app/script/init.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Database initialization failed"
    exit 1
fi

echo "Database initialized successfully"

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
