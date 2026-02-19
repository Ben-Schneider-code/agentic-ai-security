#!/bin/bash
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

# Start PostgreSQL and initialize database
echo "Initializing database..."
/app/script/init.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Database initialization failed"
    exit 1
fi

echo "Database initialized successfully"

echo "========================================================"
echo "Environment ready for testing (No vLLM)."
echo "You can run the conversation script with:"
echo "  python3 /app/util/run_conversations.py --input /path/to/file.txt"
echo "========================================================"

# Drop to shell
exec /bin/bash
