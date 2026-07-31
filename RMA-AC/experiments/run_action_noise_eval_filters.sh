#!/usr/bin/env bash
# run_action_noise_eval_filters.sh — sweep biased-Gaussian action noise across
# all 4 variants, evaluating EMA/median/Kalman action-recovery filters (see
# action_filters.py / filter_eval.py) instead of the DDPM denoiser.
#
# Usage (run from experiments/):
#   bash run_action_noise_eval_filters.sh [SCENARIO] [NUM_ADVERSARIES] [SEED] [NOISE_MU]
#
# Examples:
#   bash run_action_noise_eval_filters.sh simple_adversary 0
#   bash run_action_noise_eval_filters.sh simple_tag 3 5 0

set -euo pipefail

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""

SCENARIO="${1:-simple_adversary}"
NUM_ADVERSARIES="${2:-0}"
SEED="${3:-0}"
NOISE_MU="${4:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/model"
OUT_DIR="${SCRIPT_DIR}/../../noise_sweeps/filters/noise/seed${SEED}"

NUM_TEST_EPISODES=800
ACT_STD_LIST="0.0 0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.0"

declare -A VARIANT_FLAG
VARIANT_FLAG["maddpg"]="maddpg-none"
VARIANT_FLAG["earnie"]="maddpg-earnie"
VARIANT_FLAG["rmaac"]="maddpg-act_adv"
VARIANT_FLAG["m3ddpg"]="m3ddpg"

echo "=== [filters] action-noise sweep | scenario=${SCENARIO} | num_adversaries=${NUM_ADVERSARIES} | seed=${SEED} | mu=${NOISE_MU} ==="

cd "${SCRIPT_DIR}"
mkdir -p "${OUT_DIR}"

# train.py/filter_eval.py parse --noise-mu as a Python float, so the CSV
# filename embeds it via "{}".format(float(...)) (e.g. "0.0", "-1.0").
MU_FMT=$(python -c "print(float(\"${NOISE_MU}\"))")

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
        --attack      gauss                  \
        --noise-mu    "${NOISE_MU}"           \
        --act-std-list ${ACT_STD_LIST}       \
        --act-low     -1.0                   \
        --act-high     1.0                   \
        --seed        "${SEED}"

    CSV_FILE="${EXP_NAME}_mu${MU_FMT}_actstd_filter_sweep.csv"
    mv "${CSV_FILE}" "${OUT_DIR}/"
    echo "  => Results: ${OUT_DIR}/${CSV_FILE}"
done

echo ""
echo "=== [filters] action-noise sweep complete ==="
