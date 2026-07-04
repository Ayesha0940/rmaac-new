#!/usr/bin/env bash
# run_diffusion_policy_ablation.sh — 3-condition action-processing ablation
# on simple_spread / m3ddpg / gauss noise at mu=0:
#   1. no diffusion            (reward_noise_no_diffusion)
#   2. denoiser (t_start sweep)(best_reward_with_diffusion, best of t_start in T_START_LIST)
#   3. diffusion policy control(reward_diffusion_policy — samples from pure
#                                noise, ignoring the intended action)
#
# Usage (run from experiments/):
#   bash run_diffusion_policy_ablation.sh

set -euo pipefail

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""

SCENARIO="simple_spread"
NUM_ADVERSARIES=0
VARIANT="m3ddpg"
EXP_NAME="${SCENARIO}__m3ddpgbest"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/model"
DIFFUSION_MODEL="${SCRIPT_DIR}/diffusion_models/${SCENARIO}_m3ddpg.pt"

NUM_TEST_EPISODES=800
DIFFUSION_STEPS=100
T_START_LIST="20 40"
ACT_STD_LIST="0.0 1.0 2.0 2.5 3.0"

echo "=== diffusion-policy ablation | scenario=${SCENARIO} | variant=${VARIANT} ==="
echo "=== Diffusion model: ${DIFFUSION_MODEL} ==="

cd "${SCRIPT_DIR}"

python train.py \
    --scenario    "${SCENARIO}"          \
    --variant     "${VARIANT}"           \
    --mode        test                   \
    --exp-name    "${EXP_NAME}"          \
    --save-dir    "${MODEL_DIR}"         \
    --num-adversaries "${NUM_ADVERSARIES}" \
    --num-test-episodes "${NUM_TEST_EPISODES}" \
    --attack      gauss                  \
    --noise-mu    0.0                    \
    --act-std-list ${ACT_STD_LIST}       \
    --act-low     -1.0                   \
    --act-high     1.0                   \
    --diffusion-model-path "${DIFFUSION_MODEL}" \
    --diffusion-steps      "${DIFFUSION_STEPS}" \
    --t-start-list ${T_START_LIST}       \
    --eval-diffusion-policy

echo ""
echo "  => Results: ${EXP_NAME}_diffusion_policy_ablation_dpfull.csv"
echo "=== diffusion-policy ablation complete ==="
