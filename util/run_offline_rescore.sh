#!/bin/bash
# Offline rescorer wrapper: brings up ONE ephemeral Postgres (the AAS_DB_*
# contract) so util/offline_rescore.py Stage B can replay logged SQL, then tears
# it down. Stage A needs no DB, but running through here is harmless. The DB
# schema+data are identical across runs, so one container serves all of them.
#
# Usage:
#   HONEYPOT_TYPE=col ./util/run_offline_rescore.sh --stage all --resume \
#       --runs results-20260603-2040-ofpbb results-20260605-0204-9m6bf ...
set -e
set -o pipefail

ROOT_DIR="$(pwd)"
source "${ROOT_DIR}/.venv/bin/activate"

export AAS_RUN_ID="rescore-$(date +%Y%m%d-%H%M%S)-$$"
export AAS_RUN_DIR="${ROOT_DIR}/.runtime/${AAS_RUN_ID}"
mkdir -p "$AAS_RUN_DIR"

source "${ROOT_DIR}/script/pg_ephemeral.sh"
_cleanup() { local rc=$?; pg_ephemeral_stop || true; return $rc; }
trap _cleanup EXIT
trap '_cleanup; exit 130' INT
trap '_cleanup; exit 143' TERM

pg_ephemeral_start
echo "[run_offline_rescore] ephemeral PG up (${AAS_DB_HOST}:${AAS_DB_PORT}/${AAS_DB_NAME}); arm=${HONEYPOT_TYPE:-col}"

HONEYPOT_TYPE="${HONEYPOT_TYPE:-col}" python3 util/offline_rescore.py "$@"
