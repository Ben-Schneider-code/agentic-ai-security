#!/bin/bash
# No-Utility Ablation: train a fresh blueteam against a fixed redteam with ZERO
# benign conversation turns, then evaluate 100× adversarial + 100× benign.
#
# The two ablation env vars it sets on training:
#   BLUETEAM_FIXED_ATTACK_PROB=1.0   — every episode is an attack; no benign branch
#   BLUETEAM_DISABLE_EARLY_STOP=1    — disable decisive-win / plateau halts;
#                                      the hard 100-episode cap still applies
#
# Settings like horizon are mirrored from the given redteam checkpoint's args.yaml.
# After training the checkpoint path is saved to <result-dir>/blueteam_lora_path.txt.
# Evaluation runs cross_evaluate.py on exactly one (red, blue) pairing (100 episodes)
# and a benign-only pass (100 episodes); base-model baseline is excluded.
#
# Two GPUs are required.  Training and eval run sequentially so they reuse the
# same two physical GPUs:
#   GPU-0: vLLM actor server (training) / red-team vLLM server (eval)
#   GPU-1: PyTorch trainer             / blue-team vLLM server (eval)
#
# Usage:
#   # Train + eval (fresh run)
#   ./run_no_utility_ablation.sh \
#       --redteam-lora <path-to-.../steps_NNNN/sql_agent> \
#       [--base-model <hf-id>] [--num-eval-episodes 100] [--seed 42] \
#       [--gpu-0 N] [--gpu-1 N]
#
#   # Eval-only (skip training, use a prior result dir)
#   ./run_no_utility_ablation.sh \
#       --result-dir ablations/no_utility/<tag>/ \
#       [--num-eval-episodes 100] [--seed 42] \
#       [--gpu-0 N] [--gpu-1 N]

# TODO: I think this was superseded by `run_utility_ablation.sh`

set -e
set -o pipefail

# ── Progress colors ────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    _B='\033[1m'; _0='\033[0m'
    _CYN='\033[1;36m'; _GRN='\033[1;32m'; _YLW='\033[1;33m'; _RED='\033[1;31m'
else
    _B=''; _0=''; _CYN=''; _GRN=''; _YLW=''; _RED=''
fi
_step()   { printf "${_CYN}${_B}▶  %s${_0}\n"          "$*"; }
_ok()     { printf "${_GRN}${_B}✓  %s${_0}\n"          "$*"; }
_warn()   { printf "${_YLW}${_B}⚠  WARNING: %s${_0}\n" "$*"; }
_err()    { printf "${_RED}${_B}✗  ERROR: %s${_0}\n"   "$*" >&2; }
_banner() { printf "${_B}%s${_0}\n"                     "$*"; }

_banner "========================================"
_banner "Agentic AI Security: No-Utility Ablation"
_banner "========================================"

# ── Defaults ───────────────────────────────────────────────────────────────────
REDTEAM_LORA=""
RESULT_DIR=""
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
NUM_EVAL_EPISODES=100
SEED=42
GPU0=0   # vLLM actor (training) / red-team vLLM (eval)
GPU1=1   # PyTorch trainer       / blue-team vLLM (eval)

# ── Parse arguments ────────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --redteam-lora)       REDTEAM_LORA="$2";       shift ;;
        --result-dir)         RESULT_DIR="$2";          shift ;;
        --base-model)         BASE_MODEL="$2";          shift ;;
        --num-eval-episodes)  NUM_EVAL_EPISODES="$2";   shift ;;
        --seed)               SEED="$2";                shift ;;
        --gpu-0)              GPU0="$2";                shift ;;
        --gpu-1)              GPU1="$2";                shift ;;
        *) _err "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# ── Validate LoRA directory ────────────────────────────────────────────────────
validate_lora() {
    local p="$1" tag="$2"
    if [[ ! -d "$p" ]]; then
        _err "${tag} LoRA directory not found: $p"
        exit 1
    fi
    if [[ ! -f "$p/adapter_config.json" ]]; then
        _err "${tag} LoRA missing adapter_config.json: $p"
        exit 1
    fi
    if [[ ! -f "$p/adapter_model.safetensors" ]]; then
        _err "${tag} LoRA missing adapter_model.safetensors: $p"
        exit 1
    fi
}

# ── Mode dispatch ──────────────────────────────────────────────────────────────
if [[ -z "$RESULT_DIR" ]]; then
    # ════════════════════════════════════════════════════
    # MODE A: train then eval
    # ════════════════════════════════════════════════════
    if [[ -z "$REDTEAM_LORA" ]]; then
        _err "--redteam-lora is required unless --result-dir is set"
        echo "Usage: ./run_no_utility_ablation.sh --redteam-lora <path>" >&2
        exit 1
    fi

    REDTEAM_LORA="$(realpath "$REDTEAM_LORA")"
    validate_lora "$REDTEAM_LORA" "redteam"

    # Mirror horizon from redteam's args.yaml.
    # LoRA path layout:  …/run_N_agent#1_seedS/checkpoints/steps_NNNN/sql_agent
    #                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                    dirname × 3 = run dir where args.yaml lives
    REDTEAM_RUN_DIR="$(dirname "$(dirname "$(dirname "$REDTEAM_LORA")")")"
    REDTEAM_ARGS="$REDTEAM_RUN_DIR/args.yaml"
    if [[ ! -f "$REDTEAM_ARGS" ]]; then
        _err "Cannot find args.yaml at: $REDTEAM_ARGS"
        _err "Expected it 3 directory levels above the sql_agent dir."
        exit 1
    fi
    HORIZON=$(python3 -c "import yaml; print(yaml.safe_load(open('$REDTEAM_ARGS'))['horizon'])")
    echo "Redteam run dir:   $REDTEAM_RUN_DIR"
    echo "Horizon (mirrored): $HORIZON"

    TAG="no-utility-$(date +%Y%m%d-%H%M)-$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c 5 || true)"
    RESULT_DIR="ablations/no_utility/${TAG}"
    mkdir -p "$RESULT_DIR"
    RESULTS_ID="${TAG}"
    echo "Result dir: $RESULT_DIR"

    _step "[1/2] Training blueteam (no-utility: BLUETEAM_FIXED_ATTACK_PROB=1.0, BLUETEAM_DISABLE_EARLY_STOP=1)..."
    env \
        BLUETEAM_FIXED_ATTACK_PROB=1.0 \
        BLUETEAM_DISABLE_EARLY_STOP=1 \
        ./run_training.sh \
            --target      blueteam \
            --opponent-lora "$REDTEAM_LORA" \
            --horizon     "$HORIZON" \
            --base-model  "$BASE_MODEL" \
            --results-id  "$RESULTS_ID" \
            --actor-gpu   "$GPU0" \
            --training-gpu "$GPU1" \
        2>&1 | tee "$RESULT_DIR/train.log"

    # Locate the produced LoRA checkpoint
    BLUE_TEAM_DIR="${PWD}/results-${RESULTS_ID}/blueteam"
    BLUE_LORA="$(find "$BLUE_TEAM_DIR" -name sql_agent -type d 2>/dev/null | sort -V | tail -n 1)"
    if [[ -z "$BLUE_LORA" ]]; then
        _err "No sql_agent checkpoint found under $BLUE_TEAM_DIR after training"
        exit 1
    fi
    BLUE_LORA="$(realpath "$BLUE_LORA")"
    validate_lora "$BLUE_LORA" "trained blueteam"

    # Persist all paths needed for --result-dir eval-only re-runs
    echo "$BLUE_LORA"   > "$RESULT_DIR/blueteam_lora_path.txt"
    echo "$REDTEAM_LORA" > "$RESULT_DIR/redteam_lora_path.txt"
    echo "$HORIZON"      > "$RESULT_DIR/horizon.txt"
    echo "$BASE_MODEL"   > "$RESULT_DIR/base_model.txt"
    ln -sfn "../../results-${RESULTS_ID}" "$RESULT_DIR/training"
    _ok "Blueteam LoRA recorded → $RESULT_DIR/blueteam_lora_path.txt"

else
    # ════════════════════════════════════════════════════
    # MODE B: eval-only (--result-dir supplied)
    # ════════════════════════════════════════════════════
    if [[ ! -d "$RESULT_DIR" ]]; then
        _err "--result-dir does not exist: $RESULT_DIR"
        exit 1
    fi
    RESULT_DIR="$(realpath "$RESULT_DIR")"

    for f in blueteam_lora_path.txt redteam_lora_path.txt horizon.txt base_model.txt; do
        if [[ ! -f "$RESULT_DIR/$f" ]]; then
            _err "$RESULT_DIR/$f is missing — cannot run eval-only (was training completed?)"
            exit 1
        fi
    done

    BLUE_LORA="$(cat "$RESULT_DIR/blueteam_lora_path.txt")"
    REDTEAM_LORA="$(cat "$RESULT_DIR/redteam_lora_path.txt")"
    HORIZON="$(cat "$RESULT_DIR/horizon.txt")"
    BASE_MODEL="$(cat "$RESULT_DIR/base_model.txt")"

    validate_lora "$BLUE_LORA"    "recorded blueteam"
    validate_lora "$REDTEAM_LORA" "recorded redteam"

    _ok "Eval-only mode — skipping training, resuming from: $RESULT_DIR"
    echo "  Blueteam LoRA: $BLUE_LORA"
    echo "  Redteam  LoRA: $REDTEAM_LORA"
    echo "  Horizon:       $HORIZON"
    echo "  Base model:    $BASE_MODEL"
fi

# ── Evaluation (shared by both modes) ─────────────────────────────────────────
_step "[2/2] Evaluation — building faux selfplay view and running cross-eval..."

# Build a minimal 1-iter selfplay dir so run_cross_eval.sh's discovery finds
# exactly our pair.  We symlink the steps_NNNN dirs rather than copying them.
EVAL_VIEW="$RESULT_DIR/eval_view"
rm -rf "$EVAL_VIEW"
mkdir -p "$EVAL_VIEW/iter_1/redteam/checkpoints"
mkdir -p "$EVAL_VIEW/iter_1/blueteam/checkpoints"

RED_STEPS="$(dirname "$(realpath "$REDTEAM_LORA")")"   # …/steps_NNNN
BLUE_STEPS="$(dirname "$(realpath "$BLUE_LORA")")"
ln -sfn "$RED_STEPS"  "$EVAL_VIEW/iter_1/redteam/checkpoints/$(basename "$RED_STEPS")"
ln -sfn "$BLUE_STEPS" "$EVAL_VIEW/iter_1/blueteam/checkpoints/$(basename "$BLUE_STEPS")"

echo "Eval view dir: $EVAL_VIEW"
echo "  iter_1/redteam  → $RED_STEPS"
echo "  iter_1/blueteam → $BLUE_STEPS"

# run_cross_eval.sh handles:
#   • vLLM startup (red + blue servers)
#   • 100 attack episodes for (red_1, blue_1)
#   • 100 benign episodes for blue_1 (via cross_evaluate.py's benign_only stage)
#   • --no-include-base: exclude base model (iter_0) from benign eval
./run_cross_eval.sh \
    --selfplay-dir    "$EVAL_VIEW" \
    --base-model      "$BASE_MODEL" \
    --episodes        "$NUM_EVAL_EPISODES" \
    --horizon         "$HORIZON" \
    --seed            "$SEED" \
    --red-gpu         "$GPU0" \
    --blue-gpu        "$GPU1" \
    --pairing-subset  custom \
    --pairing-list    "red_1:blue_1" \
    --no-include-base \
    2>&1 | tee "$RESULT_DIR/cross_eval.log"

echo ""
_banner "========================================"
_ok  "No-Utility Ablation Complete"
_banner "========================================"
echo "Result dir:   $RESULT_DIR"
echo "Blueteam LoRA: $BLUE_LORA"
echo "Redteam  LoRA: $REDTEAM_LORA"
echo ""
echo "Eval results:"
echo "  Attack (100 ep): $EVAL_VIEW/cross_eval/pairings/red_1_blue_1/summary.json"
echo "  Benign (100 ep): $EVAL_VIEW/cross_eval/benign_only/blue_1/summary.json"
