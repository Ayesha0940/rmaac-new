#!/usr/bin/env bash
# run_diffusion_policy_ablation_t20.sh — same 3-condition action-processing
# ablation as run_diffusion_policy_ablation.sh, but the diffusion-policy
# control runs only a partial reverse chain (t_start=20) instead of the full
# chain (t_start=99). This matches one of the denoiser's own t_start values,
# so the only difference between denoiser@t_start=20 and this diffusion-policy
# control is the seed (corrupted action vs. pure noise), not the chain length.
#
# Usage (run from experiments/):
#   bash run_diffusion_policy_ablation_t20.sh [SCENARIO] [NUM_ADVERSARIES]
#
# Examples:
#   bash run_diffusion_policy_ablation_t20.sh
#   bash run_diffusion_policy_ablation_t20.sh simple_adversary 0

set -euo pipefail

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""

SCENARIO="${1:-simple_spread}"
NUM_ADVERSARIES="${2:-0}"
VARIANT="m3ddpg"
EXP_NAME="${SCENARIO}__m3ddpgbest"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/model"
DIFFUSION_MODEL="${SCRIPT_DIR}/diffusion_models/${SCENARIO}_m3ddpg.pt"

NUM_TEST_EPISODES=800
DIFFUSION_STEPS=100
T_START_LIST="20"
ACT_STD_LIST="0.0 1.0 2.0 2.5 3.0"
DIFFUSION_POLICY_T_START=20

echo "=== diffusion-policy ablation (t_start=${DIFFUSION_POLICY_T_START}) | scenario=${SCENARIO} | variant=${VARIANT} ==="
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
    --eval-diffusion-policy              \
    --diffusion-policy-t-start "${DIFFUSION_POLICY_T_START}"

echo ""
echo "  => Results: ${EXP_NAME}_diffusion_policy_ablation_dpt${DIFFUSION_POLICY_T_START}.csv"
echo "=== diffusion-policy ablation (t_start=${DIFFUSION_POLICY_T_START}) complete ==="
