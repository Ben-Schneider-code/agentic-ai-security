#!/bin/bash
# Interactive mode script - starts services but keeps container alive for manual interaction

# Export HuggingFace token for model access (ensure it's available to subprocesses)
export HF_TOKEN
# Also set HUGGING_FACE_HUB_TOKEN as some libraries use this name
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Show token status (first 10 chars for verification)
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN: ${HF_TOKEN:0:10}... (set)"
else
    echo "WARNING: HF_TOKEN is not set. Model server may fail to load gated models."
fi

# Start the model server in the background with explicit environment
echo "Starting vLLM model server..."
env HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" python3 /app/host_models.py > /app/model_server.log 2>&1 &
MODEL_PID=$!

# Wait for model server to start and load model (up to 120 seconds)
echo "Waiting for model server to initialize..."
TIMEOUT=120
START_TIME=$(date +%s)

while true; do
    # Check if process died
    if ! kill -0 $MODEL_PID 2>/dev/null; then
        echo "ERROR: Model server process died unexpectedly. Check /app/model_server.log"
        cat /app/model_server.log
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
        echo "ERROR: Model server timed out after $TIMEOUT seconds. Check /app/model_server.log"
        cat /app/model_server.log
        kill $MODEL_PID 2>/dev/null
        exit 1
    fi

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

# Print instructions
echo ""
echo "=========================================="
echo "Interactive SQLEnv is ready!"
echo "=========================================="
echo ""
echo "To start the interactive Python REPL, run:"
echo "  python3 -u /app/interactive_sql_env.py"
echo ""
echo "Note: Initialization may take 1-2 minutes as it connects to services."
echo "      Use -u flag for unbuffered output to see progress."
echo ""
echo "Or start an interactive shell:"
echo "  /bin/bash"
echo ""
echo "Services running:"
echo "  - vLLM server (PID: $MODEL_PID)"
echo "  - PostgreSQL database"
echo "  - MCP server"
echo ""
echo "Press Ctrl+C to stop all services and exit"
echo "=========================================="
echo ""

# Drop into an interactive shell
# Background processes (model server, postgres) will continue running
exec /bin/bash

