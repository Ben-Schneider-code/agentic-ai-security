#!/bin/bash
# Interactive mode script - starts services but keeps container alive for manual interaction

# Export HuggingFace token for model access (ensure it's available to subprocesses)
# Start common services (vLLM with health check, Postgres)
# Note: start_common_services.sh handles env setup, model server startup with health checks, and DB init.
source /app/start_common_services.sh

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

