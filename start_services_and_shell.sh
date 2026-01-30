#!/bin/bash
# Set HuggingFace token for model access
echo HF_TOKEN:
echo $HF_TOKEN

# Propagate exit code from script
set -o pipefail

# Prevent CUDA memory fragmentation
# Start common services (vLLM, Postgres)
source /app/start_common_services.sh

echo "========================================================"
echo "Environment ready for testing."
echo "You can run the conversation script with:"
echo "  python3 /app/util/run_conversations.py --input /path/to/file.txt"
echo "========================================================"

# Drop to shell
exec /bin/bash
