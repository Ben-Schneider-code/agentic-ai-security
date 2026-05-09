#!/bin/bash
# Ablation-run orchestrator for the self-play methodology (§sec:ablations).
#
# Each ablation toggles a single factor via environment variables read by the
# training env (MARFT/marft/envs/{redteam,blueteam}_sql/*.py) and then invokes
# run_selfplay.sh + run_cross_eval.sh into a named subdirectory.
#
# Usage:
#   ./run_ablations.sh --ablation {A1|A2|A3|A4|A5} [--num-iterations N]
#                      [--base-model <hf_id>] [--redteam-gpu X] [--blueteam-gpu Y]
#                      [--coach-gpu Z] [--horizon H] [--a3-prob 0.5] [--a5-k 4]
#                      [--dry-run]
#
# Env-var semantics (also set manually if you want to inspect rollouts):
#   A1: REDTEAM_DISABLE_REWARD_DECAY=1   (flat intermediate red rewards)
#   A2: BLUETEAM_PLAIN_ONLY=1            (drop adversarial + multi-turn benign)
#   A3: BLUETEAM_FIXED_ATTACK_PROB=<p>   (pin curriculum flat; default 0.5)
#   A4: BLUETEAM_REVERSE_REWARDS=1       (swap neutral_sql ↔ attack_refusal)
#   A5: no env var — vary --num-iterations across K ∈ {1,2,4,8}.
#
# Results land under ablations/<ablation>/ and each cross-eval into its
# selfplay dir's cross_eval/ subtree.

set -e
set -o pipefail

ABLATION=""
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
NUM_ITERATIONS=8
REDTEAM_GPU=0
BLUETEAM_GPU=1
COACH_GPU=0
HORIZON=5
A3_PROB="0.5"
A5_KS="1 2 4 8"
DRY_RUN=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ablation)         ABLATION="$2"; shift ;;
        --base-model)       BASE_MODEL="$2"; shift ;;
        --num-iterations)   NUM_ITERATIONS="$2"; shift ;;
        --redteam-gpu)      REDTEAM_GPU="$2"; shift ;;
        --blueteam-gpu)     BLUETEAM_GPU="$2"; shift ;;
        --coach-gpu)        COACH_GPU="$2"; shift ;;
        --horizon)          HORIZON="$2"; shift ;;
        --a3-prob)          A3_PROB="$2"; shift ;;
        --a5-k)             A5_KS="$2"; shift ;;
        --dry-run)          DRY_RUN=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$ABLATION" ]]; then
    echo "ERROR: --ablation is required (A1|A2|A3|A4|A5)"
    exit 1
fi

mkdir -p ablations

COMMON_SELFPLAY_ARGS=(
    --base-model "$BASE_MODEL"
    --num-iterations "$NUM_ITERATIONS"
    --redteam-gpu "$REDTEAM_GPU"
    --blueteam-gpu "$BLUETEAM_GPU"
    --coach-gpu "$COACH_GPU"
    --horizon "$HORIZON"
)

run_one() {
    local tag="$1"       # e.g. A1, A5_K2
    local selfplay_dir="ablations/${tag}"
    shift
    local env_exports=("$@")

    echo ""
    echo "========================================"
    echo "Ablation ${tag}"
    echo "  env: ${env_exports[*]:-<none>}"
    echo "  selfplay dir: ${selfplay_dir}"
    echo "========================================"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[dry-run] would invoke: env ${env_exports[*]} ./run_selfplay.sh ${COMMON_SELFPLAY_ARGS[*]}"
        echo "[dry-run] would invoke: ./run_cross_eval.sh --selfplay-dir <resolved-dir>"
        return 0
    fi

    mkdir -p "$selfplay_dir"
    # run_selfplay.sh writes to results-<id>/; we capture that ID and symlink it.
    local log_file="${selfplay_dir}/selfplay.log"
    env "${env_exports[@]}" ./run_selfplay.sh "${COMMON_SELFPLAY_ARGS[@]}" 2>&1 | tee "$log_file"

    local sp_id
    sp_id=$(grep -oE 'Selfplay Run ID: [^ ]+' "$log_file" | tail -n1 | awk '{print $4}')
    if [[ -z "$sp_id" ]]; then
        echo "ERROR: could not extract selfplay ID from ${log_file}"
        return 1
    fi
    ln -sfn "../results-${sp_id}" "${selfplay_dir}/selfplay"
    ./run_cross_eval.sh --selfplay-dir "results-${sp_id}"
}

case "$ABLATION" in
    A1)
        run_one "A1_flat_decay"    "REDTEAM_DISABLE_REWARD_DECAY=1"
        run_one "A1_baseline"      ""
        ;;
    A2)
        run_one "A2_plain_only"    "BLUETEAM_PLAIN_ONLY=1"
        run_one "A2_baseline"      ""
        ;;
    A3)
        run_one "A3_fixed_${A3_PROB}" "BLUETEAM_FIXED_ATTACK_PROB=${A3_PROB}"
        run_one "A3_baseline"      ""
        ;;
    A4)
        run_one "A4_reversed"      "BLUETEAM_REVERSE_REWARDS=1"
        run_one "A4_baseline"      ""
        ;;
    A5)
        for K in $A5_KS; do
            COMMON_SELFPLAY_ARGS_OVERRIDE=(
                --base-model "$BASE_MODEL"
                --num-iterations "$K"
                --redteam-gpu "$REDTEAM_GPU"
                --blueteam-gpu "$BLUETEAM_GPU"
                --coach-gpu "$COACH_GPU"
                --horizon "$HORIZON"
            )
            COMMON_SELFPLAY_ARGS=("${COMMON_SELFPLAY_ARGS_OVERRIDE[@]}") run_one "A5_K${K}" ""
        done
        ;;
    *)
        echo "ERROR: unknown ablation '$ABLATION'. Choose from A1, A2, A3, A4, A5."
        exit 1
        ;;
esac

echo ""
echo "Ablation ${ABLATION} complete. Aggregate with: python util/plot_ablations.py ablations/"
