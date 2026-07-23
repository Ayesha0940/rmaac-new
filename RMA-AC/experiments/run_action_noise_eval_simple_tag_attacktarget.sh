#!/usr/bin/env bash
set -euo pipefail

# Gaussian action-noise sweep, simple_tag only, where the noise/denoiser is
# applied to ONLY ONE role (predator or prey); the other role always plays
# its raw, unperturbed policy action. Captures the same full predator/prey
# benchmark metrics (reward split, captures, survival) as
# run_action_noise_eval_simple_tag_predprey.sh, minus the cross-play columns
# (which don't apply when only one role is ever attacked). See
# --attack-target in train.py's parse_args() and the act-std branch of
# __main__.
#
# Usage (run from experiments/):
#   bash run_action_noise_eval_simple_tag_attacktarget.sh [predator|prey] [SEED ...]
#
# Examples:
#   bash run_action_noise_eval_simple_tag_attacktarget.sh              # prey, seed 49 (defaults)
#   bash run_action_noise_eval_simple_tag_attacktarget.sh predator     # predator, seed 49
#   bash run_action_noise_eval_simple_tag_attacktarget.sh predator 49  # explicit seed list

ATTACK_TARGET="${1:-prey}"
if [[ "${ATTACK_TARGET}" != "predator" && "${ATTACK_TARGET}" != "prey" ]]; then
    echo "Usage: $0 [predator|prey] [SEED ...]" >&2
    exit 1
fi
if [[ $# -gt 0 ]]; then
    shift
fi

# ─── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"
MODEL_DIR="${MODEL_DIR:-$EXPERIMENTS_DIR/model}"
DIFFUSION_MODELS_DIR="${DIFFUSION_MODELS_DIR:-$EXPERIMENTS_DIR/diffusion_models}"
SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
    SEEDS=(49)
fi
NUM_TEST_EPISODES="${NUM_TEST_EPISODES:-800}"

declare -A SCENARIO_NUM_ADVERSARIES
SCENARIO_NUM_ADVERSARIES["simple_tag"]=3

export SUPPRESS_MA_PROMPT=1
# Pin to a single GPU (not empty/CPU-only — that hangs TF 1.10's cuInit on
# this host — and not all 8, since every testRobustnessAP() call tears down
# and rebuilds its TF session from scratch). Override by exporting
# CUDA_VISIBLE_DEVICES yourself before invoking this.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SCENARIOS=(
    simple_tag
)

# Each entry: "VARIANT_FLAG:EXP_SUFFIX"
VARIANTS=(
    # "maddpg-none:maddpg"
    # "maddpg-earnie:earnie"
    # "maddpg-act_adv:rmaac"
    "m3ddpg:m3ddpg"
)

# Biased Gaussian noise means to evaluate
NOISE_MU_LIST=(-1 0 1)

# ─── Helpers ──────────────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# ─── Main loop ────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Action-noise sweep, target=${ATTACK_TARGET} only (simple_tag only)"
echo "Scenarios    : ${SCENARIOS[*]}"
echo "Noise mu     : ${NOISE_MU_LIST[*]}"
echo "Noise sigma  : 0 1 2 3"
echo "t-start list : 20 40"
echo "Seeds        : ${SEEDS[*]}"
echo "Model dir    : $MODEL_DIR"
echo "Diffusion dir: $DIFFUSION_MODELS_DIR"
echo "============================================================"

for SEED in "${SEEDS[@]}"; do
echo ""
echo "############################################################"
echo "### Seed: ${SEED}"
echo "############################################################"

LOG_DIR="$EXPERIMENTS_DIR/logs/action_noise_eval_attacktarget/seed${SEED}"
OUT_DIR="$PROJECT_ROOT/../noise_sweeps/guassian_noise/seed${SEED}/predprey"
mkdir -p "$LOG_DIR" "$OUT_DIR"

for SCENARIO in "${SCENARIOS[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "Scenario: $SCENARIO"
    echo "════════════════════════════════════════════════════════════"

    DIFFUSION_MODEL="$DIFFUSION_MODELS_DIR/${SCENARIO}_m3ddpg.pt"
    if [[ ! -f "$DIFFUSION_MODEL" ]]; then
        echo "[$(timestamp)] [SKIP] $SCENARIO — diffusion model not found: $DIFFUSION_MODEL"
        continue
    fi
    echo "[$(timestamp)] Diffusion model found: $DIFFUSION_MODEL"

    NUM_ADVERSARIES="${SCENARIO_NUM_ADVERSARIES[$SCENARIO]}"

    for ENTRY in "${VARIANTS[@]}"; do
        VARIANT="${ENTRY%%:*}"
        SUFFIX="${ENTRY##*:}"
        EXP_NAME="${SCENARIO}__${SUFFIX}best"
        BASE_CKPT="$MODEL_DIR/${EXP_NAME}.index"

        if [[ ! -f "$BASE_CKPT" ]]; then
            echo "[$(timestamp)] [SKIP] $EXP_NAME — base policy checkpoint not found: $BASE_CKPT"
            continue
        fi
        echo "[$(timestamp)] Base policy found: $EXP_NAME"

        for MU in "${NOISE_MU_LIST[@]}"; do
            LOG_FILE="$LOG_DIR/${EXP_NAME}_mu${MU}.log"
            MU_FMT=$(printf '%.1f' "$MU")
            CSV_FILE="${EXP_NAME}_mu${MU_FMT}_actstd_tstart_sweep_${ATTACK_TARGET}only.csv"

            echo ""
            echo "────────────────────────────────────────────────────────────"
            echo "[$(timestamp)] [RUN] $EXP_NAME  noise_mu=$MU  attack_target=$ATTACK_TARGET  seed=$SEED"
            echo "────────────────────────────────────────────────────────────"

            (cd "$EXPERIMENTS_DIR" && python -u train.py \
                --scenario             "$SCENARIO" \
                --variant              "$VARIANT" \
                --mode                 test \
                --exp-name             "$EXP_NAME" \
                --save-dir             "$MODEL_DIR" \
                --num-adversaries      "$NUM_ADVERSARIES" \
                --num-test-episodes    "$NUM_TEST_EPISODES" \
                --noise-mu             "$MU" \
                --act-std-list         0 1 2 3 \
                --diffusion-model-path "$DIFFUSION_MODEL" \
                --diffusion-steps      100 \
                --t-start-list         20 40 \
                --benchmark \
                --attack-target        "$ATTACK_TARGET" \
                --seed                 "$SEED") 

            mv "$EXPERIMENTS_DIR/$CSV_FILE" "$OUT_DIR/"
            echo "[$(timestamp)] [DONE] $EXP_NAME  mu=$MU → $OUT_DIR/$CSV_FILE"
        done
    done
done

done # SEED

echo ""
echo "============================================================"
echo "Action-noise sweep, target=${ATTACK_TARGET} only, evaluation complete."
echo "Logs : $EXPERIMENTS_DIR/logs/action_noise_eval_attacktarget/seed{${SEEDS[*]}}/"
echo "CSVs : $PROJECT_ROOT/../noise_sweeps/guassian_noise/seed{${SEEDS[*]}}/predprey/*_actstd_tstart_sweep_${ATTACK_TARGET}only.csv"
echo "============================================================"
