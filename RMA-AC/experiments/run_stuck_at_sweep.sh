#!/usr/bin/env bash
# run_stuck_at_sweep.sh — sweep stuck_at attack across all 4 variants
#
# Usage (run from experiments/):
#   bash run_stuck_at_sweep.sh [SCENARIO] [NUM_ADVERSARIES]
#
# Examples:
#   bash run_stuck_at_sweep.sh simple_adversary 0
#   bash run_stuck_at_sweep.sh simple_tag 3

set -euo pipefail

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""

SCENARIO="${1:-simple_adversary}"
NUM_ADVERSARIES="${2:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/model"
DIFFUSION_MODEL="${SCRIPT_DIR}/diffusion_models/${SCENARIO}_m3ddpg.pt"

NUM_TEST_EPISODES=800
DIFFUSION_STEPS=100
T_START_LIST="20 40"
SP_PROB_LIST="0.0 0.1 0.25 0.5 0.75 1.0"

declare -A VARIANT_FLAG
VARIANT_FLAG["maddpg"]="maddpg-none"
VARIANT_FLAG["earnie"]="maddpg-earnie"
VARIANT_FLAG["rmaac"]="maddpg-act_adv"
VARIANT_FLAG["m3ddpg"]="m3ddpg"

echo "=== stuck_at sweep | scenario=${SCENARIO} | num_adversaries=${NUM_ADVERSARIES} ==="
echo "=== Diffusion model: ${DIFFUSION_MODEL} ==="

cd "${SCRIPT_DIR}"

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
        --attack      stuck_at               \
        --sp-prob-list ${SP_PROB_LIST}       \
        --act-low     -1.0                   \
        --act-high     1.0                   \
        --diffusion-model-path "${DIFFUSION_MODEL}" \
        --diffusion-steps      "${DIFFUSION_STEPS}" \
        --t-start-list ${T_START_LIST}

    echo "  => Results: ${EXP_NAME}_stuck_at_sweep.csv"
done

echo ""
echo "=== stuck_at sweep complete ==="
