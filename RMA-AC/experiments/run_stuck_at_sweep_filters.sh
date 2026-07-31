#!/usr/bin/env bash
# run_stuck_at_sweep_filters.sh — sweep stuck_at (actuator fault) attack across
# all 4 variants, evaluating EMA/median/Kalman action-recovery filters (see
# action_filters.py / filter_eval.py) instead of the DDPM denoiser.
#
# Usage (run from experiments/):
#   bash run_stuck_at_sweep_filters.sh [SCENARIO] [NUM_ADVERSARIES] [SEED]
#
# Examples:
#   bash run_stuck_at_sweep_filters.sh simple_adversary 0
#   bash run_stuck_at_sweep_filters.sh simple_tag 3 5

set -euo pipefail

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""

SCENARIO="${1:-simple_adversary}"
NUM_ADVERSARIES="${2:-0}"
SEED="${3:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/model"
OUT_DIR="${SCRIPT_DIR}/../../noise_sweeps/filters/actuation_fault/seed${SEED}"

NUM_TEST_EPISODES=800
SP_PROB_LIST="0.0 0.1 0.25 0.5 0.75 1.0"

declare -A VARIANT_FLAG
VARIANT_FLAG["maddpg"]="maddpg-none"
VARIANT_FLAG["earnie"]="maddpg-earnie"
VARIANT_FLAG["rmaac"]="maddpg-act_adv"
VARIANT_FLAG["m3ddpg"]="m3ddpg"

echo "=== [filters] stuck_at sweep | scenario=${SCENARIO} | num_adversaries=${NUM_ADVERSARIES} | seed=${SEED} ==="

cd "${SCRIPT_DIR}"
mkdir -p "${OUT_DIR}"

for SUFFIX in maddpg earnie rmaac m3ddpg; do
    VARIANT="${VARIANT_FLAG[$SUFFIX]}"
    EXP_NAME="${SCENARIO}__${SUFFIX}best"

    echo ""
    echo "--- Variant: ${VARIANT} (${EXP_NAME}) ---"

    python filter_eval.py \
        --scenario    "${SCENARIO}"          \
        --variant     "${VARIANT}"           \
        --mode        test                   \
        --exp-name    "${EXP_NAME}"          \
        --save-dir    "${MODEL_DIR}"         \
        --num-adversaries "${NUM_ADVERSARIES}" \
        --num-test-episodes "${NUM_TEST_EPISODES}" \
        --attack      stuck_at               \
        --sp-prob-list ${SP_PROB_LIST}       \
        --act-low     -1.0                   \
        --act-high     1.0                   \
        --seed        "${SEED}"

    mv "${EXP_NAME}_stuck_at_filter_sweep.csv" "${OUT_DIR}/"
    echo "  => Results: ${OUT_DIR}/${EXP_NAME}_stuck_at_filter_sweep.csv"
done

echo ""
echo "=== [filters] stuck_at sweep complete ==="
