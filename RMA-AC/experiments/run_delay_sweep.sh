#!/usr/bin/env bash
# run_delay_sweep.sh — sweep communication-delay attack across all 4 variants
#
# Usage (run from experiments/):
#   bash run_delay_sweep.sh [SCENARIO] [NUM_ADVERSARIES] [SEED]
#
# Examples:
#   bash run_delay_sweep.sh simple_adversary 0
#   bash run_delay_sweep.sh simple_spread 0 5
#   SCENARIO=simple_push NUM_ADVERSARIES=0 bash run_delay_sweep.sh

set -euo pipefail

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""

SCENARIO="${1:-simple_adversary}"
NUM_ADVERSARIES="${2:-0}"
SEED="${3:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/model"
DIFFUSION_MODEL="${SCRIPT_DIR}/diffusion_models/${SCENARIO}_m3ddpg.pt"
OUT_DIR="${SCRIPT_DIR}/../../noise_sweeps/delay_sweeps/seed${SEED}"

NUM_TEST_EPISODES=800
DIFFUSION_STEPS=100
T_START_LIST="20 40"
DELAY_K_LIST="1 2 3 5 8"

declare -A VARIANT_FLAG
VARIANT_FLAG["maddpg"]="maddpg-none"
VARIANT_FLAG["earnie"]="maddpg-earnie"
VARIANT_FLAG["rmaac"]="maddpg-act_adv"
VARIANT_FLAG["m3ddpg"]="m3ddpg"

echo "=== Delay sweep | scenario=${SCENARIO} | num_adversaries=${NUM_ADVERSARIES} | seed=${SEED} ==="
echo "=== Diffusion model: ${DIFFUSION_MODEL} ==="

cd "${SCRIPT_DIR}"
mkdir -p "${OUT_DIR}"

for SUFFIX in maddpg earnie rmaac m3ddpg; do
    VARIANT="${VARIANT_FLAG[$SUFFIX]}"
    EXP_NAME="${SCENARIO}__${SUFFIX}best"

    echo ""
    echo "--- Variant: ${VARIANT} (${EXP_NAME}) ---"

    python train.py \
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
        --diffusion-model-path "${DIFFUSION_MODEL}" \
        --diffusion-steps      "${DIFFUSION_STEPS}" \
        --t-start-list ${T_START_LIST}       \
        --seed        "${SEED}"

    mv "${EXP_NAME}_delay_sweep.csv" "${OUT_DIR}/"
    echo "  => Results: ${OUT_DIR}/${EXP_NAME}_delay_sweep.csv"
done

echo ""
echo "=== Delay sweep complete ==="
