#!/bin/bash
# Start common services for all environments
# - Exports environment variables
# - Starts vLLM model server
# - Starts PostgreSQL
# - Exports MODEL_PID, MODEL_SERVER_LOG

# Set HuggingFace token for model access
echo HF_TOKEN:
echo $HF_TOKEN

# Propagate exit code from script
set -o pipefail

# Prevent CUDA memory fragmentation (essential for long training runs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_SERVER_LOG=/app/model_server.log
touch $MODEL_SERVER_LOG

# Export variables for use in calling scripts
export MODEL_SERVER_LOG

# Start the model server in the background
echo "Starting vLLM model server..."
python3 /app/host_models.py > $MODEL_SERVER_LOG 2>&1 &
MODEL_PID=$!
export MODEL_PID

# Wait for model server to start and load model
echo "Waiting for model server to initialize..."
TIMEOUT=120
START_TIME=$(date +%s)

while true; do
    # Check if process died
    if ! kill -0 $MODEL_PID 2>/dev/null; then
        echo "ERROR: Model server process died unexpectedly. Check $MODEL_SERVER_LOG"
        cat $MODEL_SERVER_LOG
        exit 1
    fi

    # Check if endpoint is up using python (curl might not be installed)
    if python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/models')" >/dev/null 2>&1; then
        echo "Model server is up and ready!"
        break
    fi

    CURRENT_TIME=$(date +%s)
    ELAPSED_TIME=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED_TIME -ge $TIMEOUT ]; then
        echo "ERROR: Model server timed out after $TIMEOUT seconds. Check $MODEL_SERVER_LOG"
        cat $MODEL_SERVER_LOG
        kill $MODEL_PID 2>/dev/null
        exit 1
    fi

    echo "Waiting for model server... ($ELAPSED_TIME / $TIMEOUT s)"
    sleep 3
done

echo "Model server started successfully"

# Start PostgreSQL and initialize database
echo "Initializing database..."
/app/script/init.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Database initialization failed"
    exit 1
fi

echo "Database initialized successfully"
