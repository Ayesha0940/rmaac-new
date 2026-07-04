#!/usr/bin/env bash
# run_tv_gauss_sweep.sh — sweep sinusoidal time-varying noise across all 4 variants
#
# Usage (run from anywhere):
#   bash run_tv_gauss_sweep.sh [SCENARIO] [NUM_ADVERSARIES]
#
# Examples:
#   bash run_tv_gauss_sweep.sh simple_adversary 0
#   bash run_tv_gauss_sweep.sh simple_tag 3

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
T_START_LIST="20 40 60"
ATTACK_SIGMA_LIST="0.0 0.5 1.0 1.5 2.0 2.5 3.0"
TV_PERIOD=25   # = max_episode_len: one full sin cycle per episode

declare -A VARIANT_FLAG
VARIANT_FLAG["maddpg"]="maddpg-none"
VARIANT_FLAG["earnie"]="maddpg-earnie"
VARIANT_FLAG["rmaac"]="maddpg-act_adv"
VARIANT_FLAG["m3ddpg"]="m3ddpg"

echo "=== tv_gauss sweep | scenario=${SCENARIO} | num_adversaries=${NUM_ADVERSARIES} ==="
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
        --attack      timevarying_gauss      \
        --tv-profile  sin                    \
        --tv-period   "${TV_PERIOD}"         \
        --attack-sigma-list ${ATTACK_SIGMA_LIST} \
        --attack-mu   0.0                    \
        --act-low     -1.0                   \
        --act-high     1.0                   \
        --diffusion-model-path "${DIFFUSION_MODEL}" \
        --diffusion-steps      "${DIFFUSION_STEPS}" \
        --t-start-list ${T_START_LIST}

    echo "  => Results: ${EXP_NAME}_tv_gauss_sweep.csv"
done

echo ""
echo "=== tv_gauss sweep complete ==="
