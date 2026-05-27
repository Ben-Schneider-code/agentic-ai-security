#!/bin/bash
# Ephemeral PostgreSQL helper — spins a per-run postgres:15 Docker container.
#
# SOURCE this file (do not execute it): the functions export variables and
# must run in the caller's shell so the AAS_DB_* contract and any exit/trap
# behaviour propagate.
#
#   source script/pg_ephemeral.sh
#   pg_ephemeral_start          # needs $AAS_RUN_ID; exports AAS_DB_* + AAS_PG_CONTAINER
#   ...
#   pg_ephemeral_stop           # idempotent teardown (call from your EXIT trap)
#
# Design goals (see plans/modify-all-the-code-nested-moonbeam.md):
#   * No sudo, no host postgres — the DB is a throwaway Docker container.
#   * Docker assigns the host port (-p 127.0.0.1::5432) so concurrent runs
#     never collide and there is no allocate-then-bind race.
#   * One container per top-level run; unique name aas_pg_${AAS_RUN_ID}.
#   * Fail-fast: any bring-up error tears down the partial container and
#     aborts the whole run rather than proceeding against a half-built DB.

# Repo root resolved from this file's location (script/ -> repo root).
_PG_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fixed bootstrap credentials for the throwaway container. `julia` is the
# image superuser used only for the one-time load; the agents connect as the
# restricted `agent_user` created by row_security_setup.sql.
_PG_BOOT_USER="julia"
_PG_BOOT_PASSWORD="123"
_PG_DB_NAME="msft_customers"
_PG_AGENT_USER="agent_user"
_PG_AGENT_PASSWORD="db_agent_password"

_pg_die() {
    echo "[pg_ephemeral] ERROR: $*" >&2
    # Best-effort teardown of whatever this run created, then abort.
    pg_ephemeral_stop
    exit 1
}

# Print an unused TCP port on the loopback interface.
alloc_free_port() {
    python3 -c "import socket
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()"
}

# Abort the run if <port> is already bound. Used as a pre-flight before
# launching an expensive vLLM server so collisions fail fast and loud.
assert_port_free() {
    local port="$1"
    [[ -z "$port" ]] && { echo "[pg_ephemeral] ERROR: assert_port_free needs a port" >&2; exit 1; }
    if ! python3 -c "import socket,sys
s = socket.socket()
try:
    s.bind(('127.0.0.1', int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()" "$port" 2>/dev/null; then
        echo "[pg_ephemeral] ERROR: TCP port $port is already in use." >&2
        exit 1
    fi
}

# Validate that a parent run already provisioned the DB and exported the full
# AAS_DB_* contract. A *partial* contract is a hard error — never half-trust it.
pg_ephemeral_require() {
    local missing=()
    for v in AAS_DB_HOST AAS_DB_PORT AAS_DB_NAME AAS_DB_AGENT_USER AAS_DB_AGENT_PASSWORD AAS_PG_CONTAINER; do
        [[ -z "${!v}" ]] && missing+=("$v")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "[pg_ephemeral] ERROR: incomplete DB contract — unset: ${missing[*]}" >&2
        echo "[pg_ephemeral]        a parent run set AAS_PG_CONTAINER but not the rest." >&2
        exit 1
    fi
}

# Tear down this run's container. Idempotent; safe to call from an EXIT trap
# and safe to call when nothing was ever started.
pg_ephemeral_stop() {
    if [[ -n "$AAS_PG_CONTAINER" ]]; then
        echo "[pg_ephemeral] Removing container ${AAS_PG_CONTAINER}..."
        docker rm -f "$AAS_PG_CONTAINER" >/dev/null 2>&1 || true
    fi
}

# Spin up the ephemeral postgres container and initialize the database.
# Requires $AAS_RUN_ID. Exports AAS_DB_* and AAS_PG_CONTAINER on success.
pg_ephemeral_start() {
    [[ -z "$AAS_RUN_ID" ]] && { echo "[pg_ephemeral] ERROR: AAS_RUN_ID must be set before pg_ephemeral_start" >&2; exit 1; }
    command -v docker >/dev/null 2>&1 || { echo "[pg_ephemeral] ERROR: docker not found on PATH" >&2; exit 1; }

    local name="aas_pg_${AAS_RUN_ID}"
    export AAS_PG_CONTAINER="$name"

    # Belt-and-suspenders: drop any stale container reusing this exact name
    # (e.g. a prior crashed run with the same id).
    docker rm -f "$name" >/dev/null 2>&1 || true

    echo "[pg_ephemeral] Starting container ${name} (postgres:15)..."
    if ! docker run -d \
        --name "$name" \
        --label "aas_run_id=${AAS_RUN_ID}" \
        -e POSTGRES_USER="$_PG_BOOT_USER" \
        -e POSTGRES_PASSWORD="$_PG_BOOT_PASSWORD" \
        -e POSTGRES_DB="$_PG_DB_NAME" \
        -p 127.0.0.1::5432 \
        --health-cmd="pg_isready -U $_PG_BOOT_USER -d $_PG_DB_NAME" \
        --health-interval=3s \
        --health-timeout=3s \
        --health-start-period=2s \
        --health-retries=20 \
        postgres:15 >/dev/null; then
        _pg_die "docker run failed for ${name}"
    fi

    # Resolve the Docker-assigned host port (no allocate-then-bind race).
    local hostport
    hostport="$(docker port "$name" 5432/tcp 2>/dev/null | head -n1 | sed 's/.*://')"
    [[ -z "$hostport" ]] && _pg_die "could not resolve host port for ${name}"

    # Wait for the healthcheck to report healthy.
    echo "[pg_ephemeral] Waiting for postgres to become healthy..."
    local waited=0 status=""
    while true; do
        status="$(docker inspect --format '{{.State.Health.Status}}' "$name" 2>/dev/null || echo "")"
        [[ "$status" == "healthy" ]] && break
        if [[ "$(docker inspect --format '{{.State.Running}}' "$name" 2>/dev/null || echo false)" != "true" ]]; then
            echo "[pg_ephemeral] container is no longer running — logs follow:" >&2
            docker logs "$name" 2>&1 | tail -n 40 >&2
            _pg_die "postgres container ${name} died during startup"
        fi
        if [[ $waited -ge 90 ]]; then
            docker logs "$name" 2>&1 | tail -n 40 >&2
            _pg_die "postgres container ${name} not healthy after ${waited}s"
        fi
        sleep 2
        waited=$((waited + 2))
    done
    echo "[pg_ephemeral] postgres healthy on 127.0.0.1:${hostport} (${waited}s)"

    # --- Initialize the database (schema -> CSV import -> row security) ---
    echo "[pg_ephemeral] Loading schema.sql..."
    docker exec -i "$name" psql -v ON_ERROR_STOP=1 -U "$_PG_BOOT_USER" -d "$_PG_DB_NAME" \
        < "${_PG_REPO_ROOT}/schema.sql" >/dev/null \
        || _pg_die "schema.sql load failed"

    echo "[pg_ephemeral] Importing CSV data..."
    docker exec "$name" mkdir -p /tmp/aas_data \
        || _pg_die "could not create /tmp/aas_data in container"
    docker cp "${_PG_REPO_ROOT}/data/." "${name}:/tmp/aas_data/" >/dev/null \
        || _pg_die "docker cp of data/ failed"
    docker cp "${_PG_REPO_ROOT}/script/import_csvs.sh" "${name}:/tmp/import_csvs.sh" >/dev/null \
        || _pg_die "docker cp of import_csvs.sh failed"
    docker exec \
        -e PGUSER="$_PG_BOOT_USER" \
        -e PGDATABASE="$_PG_DB_NAME" \
        -e AAS_DATA_DIR=/tmp/aas_data \
        "$name" bash /tmp/import_csvs.sh >/dev/null \
        || _pg_die "CSV import failed"

    echo "[pg_ephemeral] Applying row-security / honeypot setup..."
    docker exec -i "$name" psql -v ON_ERROR_STOP=1 -U "$_PG_BOOT_USER" -d "$_PG_DB_NAME" \
        < "${_PG_REPO_ROOT}/access_rules/row_security_setup.sql" >/dev/null \
        || _pg_die "row_security_setup.sql failed"

    # --- Publish the connection contract ---
    export AAS_DB_HOST="127.0.0.1"
    export AAS_DB_PORT="$hostport"
    export AAS_DB_NAME="$_PG_DB_NAME"
    export AAS_DB_AGENT_USER="$_PG_AGENT_USER"
    export AAS_DB_AGENT_PASSWORD="$_PG_AGENT_PASSWORD"

    echo "[pg_ephemeral] Database ready: ${AAS_DB_AGENT_USER}@${AAS_DB_HOST}:${AAS_DB_PORT}/${AAS_DB_NAME}"
}
