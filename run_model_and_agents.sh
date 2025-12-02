#!/bin/bash
# Set HuggingFace token for model access
export HF_TOKEN=""

# Start the model server in the background
echo "Starting vLLM model server..."
python3 /app/host_models.py > /app/model_server.log 2>&1 &
MODEL_PID=$!

# Wait for model server to start and load model (60 seconds)
echo "Waiting for model server to initialize..."
sleep 60

# Check if model server is still running
if ! kill -0 $MODEL_PID 2>/dev/null; then
    echo "ERROR: Model server failed to start. Check /app/model_server.log"
    cat /app/model_server.log
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
./sample_redteam_script.sh 2>&1 | tee /app/redteam_output.log

# Keep container alive
wait $MODEL_PID