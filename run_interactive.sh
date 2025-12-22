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

