#!/usr/bin/env bash
# run_stuck_at_sweep_simple_tag_predprey.sh — actuator stuck_at fault sweep,
# simple_tag only, with predator/prey benchmark metrics (captures, survival,
# predator/prey reward split, predator-only/prey-only denoise cross-play).
#
# Copy of run_stuck_at_sweep.sh, scoped to simple_tag and with --benchmark
# added (see the "stuck_at" branch of train.py's __main__ for what that
# unlocks).
#
# Usage (run from experiments/):
#   bash run_stuck_at_sweep_simple_tag_predprey.sh [SEED ...]
#
# Examples:
#   bash run_stuck_at_sweep_simple_tag_predprey.sh          # seeds 5 10 (default)
#   bash run_stuck_at_sweep_simple_tag_predprey.sh 0 15 42  # explicit seed list

set -euo pipefail

export SUPPRESS_MA_PROMPT=1
# Pin to a single GPU (not empty/CPU-only — that hangs TF 1.10's cuInit on
# this host — and not all 8, since every testRobustnessAP() call tears down
# and rebuilds its TF session from scratch, and doing that against all 8
# devices ~168 times per seed adds a lot of avoidable per-call overhead).
# Override by exporting CUDA_VISIBLE_DEVICES yourself before invoking this.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

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
SP_PROB_LIST="0.0 0.1 0.25 0.5 0.75 1.0"

declare -A VARIANT_FLAG
VARIANT_FLAG["maddpg"]="maddpg-none"
VARIANT_FLAG["earnie"]="maddpg-earnie"
VARIANT_FLAG["rmaac"]="maddpg-act_adv"
VARIANT_FLAG["m3ddpg"]="m3ddpg"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "============================================================"
echo "stuck_at-attack predator/prey benchmark evaluation (simple_tag only)"
echo "sp_prob list : ${SP_PROB_LIST}"
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

    LOG_DIR="${SCRIPT_DIR}/logs/stuck_at_sweep_predprey/seed${SEED}"
    OUT_DIR="${SCRIPT_DIR}/../../noise_sweeps/actuation_fault/seed${SEED}/predprey"
    mkdir -p "${LOG_DIR}" "${OUT_DIR}"

    for SUFFIX in maddpg earnie rmaac; do
        VARIANT="${VARIANT_FLAG[$SUFFIX]}"
        EXP_NAME="${SCENARIO}__${SUFFIX}best"
        BASE_CKPT="${MODEL_DIR}/${EXP_NAME}.index"

        if [[ ! -f "${BASE_CKPT}" ]]; then
            echo "[$(timestamp)] [SKIP] ${EXP_NAME} — base policy checkpoint not found: ${BASE_CKPT}"
            continue
        fi

        LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
        echo ""
        echo "────────────────────────────────────────────────────────────"
        echo "[$(timestamp)] [RUN] ${EXP_NAME}  seed=${SEED}"
        echo "────────────────────────────────────────────────────────────"

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
            --t-start-list ${T_START_LIST}       \
            --benchmark                          \
            --seed        "${SEED}"              

        mv "${EXP_NAME}_stuck_at_sweep.csv" "${OUT_DIR}/"
        echo "[$(timestamp)] [DONE] ${EXP_NAME} → ${OUT_DIR}/${EXP_NAME}_stuck_at_sweep.csv"
    done
done

echo ""
echo "============================================================"
echo "stuck_at-attack predator/prey evaluation complete."
echo "Logs : ${SCRIPT_DIR}/logs/stuck_at_sweep_predprey/seed{${SEEDS[*]}}/"
echo "CSVs : ${SCRIPT_DIR}/../../noise_sweeps/actuation_fault/seed{${SEEDS[*]}}/predprey/*_stuck_at_sweep.csv"
echo "============================================================"
