#!/bin/bash
# Run one full replicate end-to-end on a single 2-GPU pair (Docker-friendly).
#
#   1. Self-play   (./run_selfplay.sh, forwarding all flags verbatim)
#   2. Cross-eval  (./run_cross_eval.sh on the resulting run dir)
#   3. Benign-eval (./util/run_benign_eval.sh on the resulting run dir)
#
# Designed for set-and-forget execution inside one Docker container. Drop the
# command into the container, walk away, return to a results-<ID>/ tree that
# contains cross_eval/ and benign_eval/ subdirs ready for the plotting step.
#
# All flags are forwarded to run_selfplay.sh. The wrapper sniffs --base-model,
# --redteam-gpu, and --blueteam-gpu so it can wire the downstream evals to the
# same GPU pair and model.
#
# Usage:
#   ./run_replicate.sh \
#       --replicate-seed 0 \
#       --num-env-steps 200 --num-iterations 4 --horizon 5 \
#       --vanilla-size 120 --bordercase-size 20 --honeypot-type rowcol \
#       --redteam-gpu 0 --blueteam-gpu 1

set -e
set -o pipefail

BASE_MODEL="Snowflake/Arctic-Text2SQL-R1-7B"
RED_GPU=""
BLUE_GPU=""
REPLICATE_SEED=0
RESULTS_ID_OVERRIDE=""
RESUME=false
HONEYPOT_TYPE_SNIFFED=""

source .venv/bin/activate

# Sniff (don't consume) base-model / GPU / seed / honeypot-type args for the
# downstream eval calls. --honeypot-type is sniffed so we can export it for
# Phases 2 and 3: their wrappers also read summary.json after Phase 1 writes
# it, but exporting here is defense-in-depth (catches a malformed summary.json
# or a future code path that imports redteam_sql_env before summary.json
# exists). --resume IS consumed here (filtered out of SELFPLAY_ARGS below)
# because run_selfplay.sh doesn't accept it — at the replicate level it means
# "skip Phase 1 if the cell is already selfplay-complete."
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
        --base-model)     BASE_MODEL="${args[$((i+1))]}" ;;
        --redteam-gpu)    RED_GPU="${args[$((i+1))]}" ;;
        --blueteam-gpu)   BLUE_GPU="${args[$((i+1))]}" ;;
        --replicate-seed) REPLICATE_SEED="${args[$((i+1))]}" ;;
        --results-id)     RESULTS_ID_OVERRIDE="${args[$((i+1))]}" ;;
        --honeypot-type)  HONEYPOT_TYPE_SNIFFED="${args[$((i+1))]}" ;;
        --resume)         RESUME=true ;;
    esac
done

if [[ "$RESUME" == "true" && -z "$RESULTS_ID_OVERRIDE" ]]; then
    echo "ERROR: --resume requires --results-id <id> (the cell to resume into)." >&2
    exit 1
fi

# --- Fail fast on GPU locators: never silently fall back to GPUs 0/1 --------
# On a shared server an unset GPU index would silently allocate on another
# user's card. GPU indices are required resource locators — no defaults.
# This wrapper forwards them verbatim to run_selfplay.sh and wires the same
# pair into cross-eval / benign-eval, so the whole replicate is pinned here.
if [[ -z "$RED_GPU" || -z "$BLUE_GPU" ]]; then
    echo "ERROR: --redteam-gpu and --blueteam-gpu are both required." >&2
    echo "       e.g. ./run_replicate.sh ... --redteam-gpu 4 --blueteam-gpu 5" >&2
    exit 1
fi
if ! [[ "$RED_GPU" =~ ^[0-9]+$ && "$BLUE_GPU" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --redteam-gpu / --blueteam-gpu must be non-negative integers (got $RED_GPU / $BLUE_GPU)." >&2
    exit 1
fi
if [[ "$RED_GPU" == "$BLUE_GPU" ]]; then
    echo "ERROR: --redteam-gpu and --blueteam-gpu must differ (got $RED_GPU / $BLUE_GPU)." >&2
    exit 1
fi

echo "========================================"
echo "Agentic AI Security: End-to-End Replicate"
echo "========================================"
echo "Base model:  $BASE_MODEL"
echo "Red GPU:     $RED_GPU"
echo "Blue GPU:    $BLUE_GPU"
echo "Forwarded:   $*"
echo "========================================"

# --- Per-run namespace + ONE ephemeral Postgres for the whole replicate -----
# Self-play, cross-eval and benign-eval all share this container; it is torn
# down when this script exits (success, failure, or Ctrl-C).
ROOT_DIR="$(pwd)"
if [[ -n "$RESULTS_ID_OVERRIDE" ]]; then
    AAS_RUN_ID="$RESULTS_ID_OVERRIDE"
else
    AAS_RUN_ID="$(date +%Y%m%d-%H%M)-$(LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 5 || true)"
fi
export AAS_RUN_ID
export AAS_RUN_DIR="${ROOT_DIR}/.runtime/${AAS_RUN_ID}"
mkdir -p "$AAS_RUN_DIR"

source "${ROOT_DIR}/script/pg_ephemeral.sh"
_replicate_cleanup() { local _rc=$?; pg_ephemeral_stop; return $_rc; }
trap _replicate_cleanup EXIT
trap '_replicate_cleanup; exit 130' INT
trap '_replicate_cleanup; exit 143' TERM
pg_ephemeral_start

# If --resume is set and the selfplay cell already has all .success markers,
# skip Phase 1 entirely. Partial cells are not auto-resumed here: run_selfplay.sh
# requires --continue/--continue-iteration/--continue-round, which the user must
# choose explicitly. Fail fast in that case rather than guessing.
SELFPLAY_ID=""
RESULTS_DIR=""
if [[ "$RESUME" == "true" ]]; then
    CANDIDATE_DIR="results-${RESULTS_ID_OVERRIDE}"
    if [[ ! -d "$CANDIDATE_DIR" ]]; then
        echo "[replicate] FAIL: --resume given but $CANDIDATE_DIR does not exist." >&2; exit 1
    fi
    if [[ ! -f "$CANDIDATE_DIR/summary.json" ]]; then
        echo "[replicate] FAIL: --resume given but $CANDIDATE_DIR/summary.json missing — cannot determine num_iterations." >&2; exit 1
    fi
    N_ITERS=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['num_iterations'])" "$CANDIDATE_DIR/summary.json")
    SELFPLAY_COMPLETE=true
    MISSING=""
    for i in $(seq 1 "$N_ITERS"); do
        for team in redteam blueteam; do
            if [[ ! -f "$CANDIDATE_DIR/iter_${i}/${team}/.success" ]]; then
                SELFPLAY_COMPLETE=false
                MISSING="iter_${i}/${team}"
                break 2
            fi
        done
    done
    if [[ "$SELFPLAY_COMPLETE" == "true" ]]; then
        SELFPLAY_ID="$RESULTS_ID_OVERRIDE"
        RESULTS_DIR="$CANDIDATE_DIR"
        echo "[replicate] --resume: selfplay cell already complete (${N_ITERS} iters). Skipping Phase 1."
    else
        echo "[replicate] FAIL: --resume cannot skip Phase 1: $MISSING is incomplete." >&2
        echo "       Continue self-play manually first, e.g.:" >&2
        echo "         ./run_selfplay.sh --continue $CANDIDATE_DIR --continue-iteration <N> --continue-round <red|blue> ..." >&2
        exit 1
    fi
fi

if [[ -z "$RESULTS_DIR" ]]; then
    LOG="replicate_$$.selfplay.log"

    # Phase 1 — self-play. Pin its run id to ours so all phases share one DB/run dir.
    # --resume is consumed by this wrapper, not run_selfplay.sh, so filter it out.
    echo "[replicate] Phase 1/3: self-play (log mirrored to $LOG)"
    SELFPLAY_ARGS=()
    for ((i=0; i<${#args[@]}; i++)); do
        [[ "${args[$i]}" == "--resume" ]] && continue
        SELFPLAY_ARGS+=("${args[$i]}")
    done
    [[ -z "$RESULTS_ID_OVERRIDE" ]] && SELFPLAY_ARGS+=(--results-id "$AAS_RUN_ID")
    ./run_selfplay.sh "${SELFPLAY_ARGS[@]}" 2>&1 | tee "$LOG"

    # Parse SELFPLAY_ID from the standard banner line printed at run_selfplay.sh:171.
    SELFPLAY_ID=$(grep -m1 "^Selfplay Run ID:" "$LOG" | awk '{print $NF}')
    [[ -z "$SELFPLAY_ID" ]] && { echo "[replicate] FAIL: could not parse SELFPLAY_ID from $LOG"; exit 1; }
    RESULTS_DIR="results-${SELFPLAY_ID}"
    [[ ! -d "$RESULTS_DIR" ]] && { echo "[replicate] FAIL: $RESULTS_DIR missing on disk"; exit 1; }
    echo "[replicate] Self-play complete. Run dir: $RESULTS_DIR"
fi

# --- Pin honeypot arm for downstream phases --------------------------------
# Phase 1 (self-play) writes results-<id>/summary.json. Phases 2 and 3 each
# also re-read summary.json themselves, but we export HONEYPOT_TYPE here so
# that any python process they launch inherits the correct arm even if a
# future code path imports redteam_sql_env before reading summary.json.
# Fail-fast: refuse to proceed if summary.json's honeypot_type contradicts
# what the user passed on the CLI.
SUMMARY_HP=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); v=d.get('honeypot_type'); assert v in ('rowcol','row','col'), f'bad honeypot_type={v!r}'; print(v)" "${RESULTS_DIR}/summary.json") || {
    echo "[replicate] FAIL: could not parse honeypot_type from ${RESULTS_DIR}/summary.json"; exit 1;
}
if [[ -n "$HONEYPOT_TYPE_SNIFFED" && "$HONEYPOT_TYPE_SNIFFED" != "$SUMMARY_HP" ]]; then
    echo "[replicate] FAIL: --honeypot-type=$HONEYPOT_TYPE_SNIFFED conflicts with summary.json honeypot_type=$SUMMARY_HP" >&2
    exit 1
fi
export HONEYPOT_TYPE="$SUMMARY_HP"
echo "[replicate] Honeypot arm (pinned from summary.json): $HONEYPOT_TYPE"

# Phase 2 — cross-eval (dual-GPU; same pair, same base model)
echo "[replicate] Phase 2/3: cross-eval"
./run_cross_eval.sh \
    --selfplay-dir "$RESULTS_DIR" \
    --base-model   "$BASE_MODEL" \
    --red-gpu  "$RED_GPU" \
    --blue-gpu "$BLUE_GPU" \
    --episodes    160 \
    --concurrency 32 \
    --seed       "$REPLICATE_SEED" \
    --skip-init \
    --resume

# Phase 3 — benign-eval (single GPU; reuse the blue GPU)
echo "[replicate] Phase 3/3: benign-eval"
./util/run_benign_eval.sh \
    --results-dir "$RESULTS_DIR" \
    --base-model  "$BASE_MODEL" \
    --gpu         "$BLUE_GPU" \
    --episodes    320 \
    --seed        "$REPLICATE_SEED" \
    --resume

echo "========================================"
echo "[replicate] complete. Artifacts under $RESULTS_DIR/"
echo "  - iter_*/{red,blue}team/        (training)"
echo "  - cross_eval/cross_eval_results.json"
echo "  - benign_eval/benign_eval_results.json"
echo "========================================"
