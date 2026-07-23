#!/usr/bin/env bash
# run_delay_sweep_simple_tag_attacktarget.sh — communication-delay attack
# sweep, simple_tag only, where the delay is applied to ONLY ONE role
# (predator or prey); the other role always plays its raw, unperturbed
# policy action. Captures the same full predator/prey benchmark metrics
# (reward split, captures, survival) as run_delay_sweep_simple_tag_predprey.sh,
# minus the cross-play columns (which don't apply when only one role is
# ever attacked). See --attack-target in train.py's parse_args() and the
# "delay" branch of __main__.
#
# Usage (run from experiments/):
#   bash run_delay_sweep_simple_tag_attacktarget.sh [predator|prey] [SEED ...]
#
# Examples:
#   bash run_delay_sweep_simple_tag_attacktarget.sh              # prey, seed 49 (defaults)
#   bash run_delay_sweep_simple_tag_attacktarget.sh predator     # predator, seed 49
#   bash run_delay_sweep_simple_tag_attacktarget.sh predator 49  # explicit seed list

set -euo pipefail

export SUPPRESS_MA_PROMPT=1
# Pin to a single GPU (not empty/CPU-only — that hangs TF 1.10's cuInit on
# this host — and not all 8, since every testRobustnessAP() call tears down
# and rebuilds its TF session from scratch). Override by exporting
# CUDA_VISIBLE_DEVICES yourself before invoking this.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

ATTACK_TARGET="${1:-prey}"
if [[ "${ATTACK_TARGET}" != "predator" && "${ATTACK_TARGET}" != "prey" ]]; then
    echo "Usage: $0 [predator|prey] [SEED ...]" >&2
    exit 1
fi
if [[ $# -gt 0 ]]; then
    shift
fi

SCENARIO="simple_tag"
NUM_ADVERSARIES=3
SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
    SEEDS=(49)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/model"
DIFFUSION_MODEL="${SCRIPT_DIR}/diffusion_models/${SCENARIO}_m3ddpg.pt"

NUM_TEST_EPISODES="${NUM_TEST_EPISODES:-800}"
DIFFUSION_STEPS=100
T_START_LIST="20 40"
DELAY_K_LIST="0 1 2 3 5 8"

declare -A VARIANT_FLAG
VARIANT_FLAG["maddpg"]="maddpg-none"
VARIANT_FLAG["earnie"]="maddpg-earnie"
VARIANT_FLAG["rmaac"]="maddpg-act_adv"
VARIANT_FLAG["m3ddpg"]="m3ddpg"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "============================================================"
echo "Delay-attack, target=${ATTACK_TARGET} only (simple_tag only)"
echo "Delay k list : ${DELAY_K_LIST}"
echo "t-start list : ${T_START_LIST}"
echo "Seeds        : ${SEEDS[*]}"
echo "Model dir    : ${MODEL_DIR}"
echo "Diffusion    : ${DIFFUSION_MODEL}"
echo "============================================================"

if [[ ! -f "${DIFFUSION_MODEL}" ]]; then
    echo "[$(timestamp)] [SKIP] ${SCENARIO} — diffusion model not found: ${DIFFUSION_MODEL}"
    exit 1
fi

cd "${SCRIPT_DIR}"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "############################################################"
    echo "### Seed: ${SEED}"
    echo "############################################################"

    LOG_DIR="${SCRIPT_DIR}/logs/delay_sweep_attacktarget/seed${SEED}"
    OUT_DIR="${SCRIPT_DIR}/../../noise_sweeps/delay_sweeps/seed${SEED}/predprey"
    mkdir -p "${LOG_DIR}" "${OUT_DIR}"

    for SUFFIX in maddpg earnie rmaac m3ddpg; do
        VARIANT="${VARIANT_FLAG[$SUFFIX]}"
        EXP_NAME="${SCENARIO}__${SUFFIX}best"
        BASE_CKPT="${MODEL_DIR}/${EXP_NAME}.index"

        if [[ ! -f "${BASE_CKPT}" ]]; then
            echo "[$(timestamp)] [SKIP] ${EXP_NAME} — base policy checkpoint not found: ${BASE_CKPT}"
            continue
        fi

        LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
        CSV_FILE="${EXP_NAME}_delay_sweep_${ATTACK_TARGET}only.csv"
        echo ""
        echo "────────────────────────────────────────────────────────────"
        echo "[$(timestamp)] [RUN] ${EXP_NAME}  attack_target=${ATTACK_TARGET}  seed=${SEED}"
        echo "────────────────────────────────────────────────────────────"

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
            --benchmark                          \
            --attack-target "${ATTACK_TARGET}"   \
            --seed        "${SEED}"

        mv "${EXP_NAME}_delay_sweep_${ATTACK_TARGET}only.csv" "${OUT_DIR}/"
        echo "[$(timestamp)] [DONE] ${EXP_NAME} → ${OUT_DIR}/${CSV_FILE}"
    done
done

echo ""
echo "============================================================"
echo "Delay-attack, target=${ATTACK_TARGET} only, evaluation complete."
echo "Logs : ${SCRIPT_DIR}/logs/delay_sweep_attacktarget/seed{${SEEDS[*]}}/"
echo "CSVs : ${SCRIPT_DIR}/../../noise_sweeps/delay_sweeps/seed{${SEEDS[*]}}/predprey/*_delay_sweep_${ATTACK_TARGET}only.csv"
echo "============================================================"
