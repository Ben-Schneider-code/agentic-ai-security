#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.dev.yml"

VERBOSE="${VERBOSE:-0}"
POSTGRES_TIMEOUT="${POSTGRES_TIMEOUT:-60}"
VLLM_TIMEOUT="${VLLM_TIMEOUT:-300}"

DC="docker compose -f $COMPOSE_FILE --project-directory $PROJECT_DIR"

echo "Starting dev dependencies (Postgres + vLLM)..."
$DC up -d

echo ""
echo "Waiting for Postgres to be healthy (timeout: ${POSTGRES_TIMEOUT}s)..."
elapsed=0
until $DC exec -T postgres pg_isready -U julia -d msft_customers > /dev/null 2>&1; do
    if [ "$elapsed" -ge "$POSTGRES_TIMEOUT" ]; then
        echo "  ERROR: Postgres failed to become healthy after ${POSTGRES_TIMEOUT}s"
        echo "  Container status:"
        $DC ps postgres
        echo "  Recent logs:"
        $DC logs --tail=20 postgres
        exit 1
    fi
    if [ "$VERBOSE" = "1" ]; then
        echo "  [${elapsed}s] Postgres not ready yet... (container: $($DC ps --format '{{.Status}}' postgres 2>/dev/null || echo 'not found'))"
    elif [ $((elapsed % 10)) -eq 0 ] && [ "$elapsed" -gt 0 ]; then
        echo "  Still waiting... (${elapsed}s elapsed)"
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done
echo "  Postgres is ready. (${elapsed}s)"

echo ""
echo "Waiting for vLLM to be ready (timeout: ${VLLM_TIMEOUT}s, this can take a few minutes on first start)..."
elapsed=0
until curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; do
    if [ "$elapsed" -ge "$VLLM_TIMEOUT" ]; then
        echo "  ERROR: vLLM failed to become ready after ${VLLM_TIMEOUT}s"
        echo "  Container status:"
        $DC ps vllm
        echo "  Recent logs:"
        $DC logs --tail=20 vllm
        exit 1
    fi
    if [ "$VERBOSE" = "1" ]; then
        echo "  [${elapsed}s] vLLM not ready yet... (container: $($DC ps --format '{{.Status}}' vllm 2>/dev/null || echo 'not found'))"
    elif [ $((elapsed % 30)) -eq 0 ] && [ "$elapsed" -gt 0 ]; then
        echo "  Still waiting... (${elapsed}s elapsed)"
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done
echo "  vLLM is ready. (${elapsed}s)"

echo ""
echo "============================================"
echo "  Dev services are up!"
echo ""
echo "  Postgres: localhost:5432  (user=julia, db=msft_customers)"
echo "  vLLM:     localhost:8000  (model=${MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct})"
echo ""
echo "  Run your code with:"
echo "    uv run python util/run_conversations.py --input ..."
echo ""
echo "  Stop services with:"
echo "    docker compose -f docker-compose.dev.yml down"
echo "============================================"
