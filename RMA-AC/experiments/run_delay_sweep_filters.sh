#!/usr/bin/env bash
# run_delay_sweep_filters.sh — sweep communication-delay attack across all 4
# variants, evaluating EMA/median/Kalman action-recovery filters (see
# action_filters.py / filter_eval.py) instead of the DDPM denoiser.
#
# Usage (run from experiments/):
#   bash run_delay_sweep_filters.sh [SCENARIO] [NUM_ADVERSARIES] [SEED]
#
# Examples:
#   bash run_delay_sweep_filters.sh simple_adversary 0
#   bash run_delay_sweep_filters.sh simple_tag 3 5

set -euo pipefail

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""

SCENARIO="${1:-simple_adversary}"
NUM_ADVERSARIES="${2:-0}"
SEED="${3:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/model"
OUT_DIR="${SCRIPT_DIR}/../../noise_sweeps/filters/delay/seed${SEED}"

NUM_TEST_EPISODES=800
DELAY_K_LIST="1 2 3 5 8"

declare -A VARIANT_FLAG
VARIANT_FLAG["maddpg"]="maddpg-none"
VARIANT_FLAG["earnie"]="maddpg-earnie"
VARIANT_FLAG["rmaac"]="maddpg-act_adv"
VARIANT_FLAG["m3ddpg"]="m3ddpg"

echo "=== [filters] Delay sweep | scenario=${SCENARIO} | num_adversaries=${NUM_ADVERSARIES} | seed=${SEED} ==="

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
        --attack      delay                  \
        --delay-k-list ${DELAY_K_LIST}       \
        --act-low     -1.0                   \
        --act-high     1.0                   \
        --seed        "${SEED}"

    mv "${EXP_NAME}_delay_filter_sweep.csv" "${OUT_DIR}/"
    echo "  => Results: ${OUT_DIR}/${EXP_NAME}_delay_filter_sweep.csv"
done

echo ""
echo "=== [filters] Delay sweep complete ==="
