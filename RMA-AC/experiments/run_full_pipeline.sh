#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
# All env vars are forwarded to sub-scripts automatically.
# Override any of these to customise the run:
#
#   SCENARIO=simple_spread bash run_full_pipeline.sh
#
# Training sub-script vars: SCENARIO, NUM_EPISODES, SAVE_RATE, NUM_ADVERSARIES, MODEL_DIR
# Diffusion sub-script vars: DIFFUSION_COLLECT_EPISODES, DIFFUSION_HORIZON,
#                             DIFFUSION_STEPS, DIFFUSION_EPOCHS, DIFFUSION_LR,
#                             DIFFUSION_BATCH_SIZE, T_START_LIST, NUM_TEST_EPISODES

SCENARIO="${SCENARIO:-simple_speaker_listener}"

export SCENARIO
export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# ─── Stage 1: Train all 4 variants ───────────────────────────────────────────
echo ""
echo "============================================================"
echo "[$(timestamp)] STAGE 1 — Training all variants"
echo "  Scenario : ${SCENARIO}"
echo "============================================================"

bash "${HERE}/run_train_all_variants.sh"

# ─── Stage 2: Diffusion pipeline (collect → train → evaluate) ────────────────
echo ""
echo "============================================================"
echo "[$(timestamp)] STAGE 2 — Diffusion pipeline"
echo "  Scenario : ${SCENARIO}"
echo "============================================================"

bash "${HERE}/run_diffusion_pipeline.sh"

echo ""
echo "============================================================"
echo "[$(timestamp)] Full pipeline complete for scenario: ${SCENARIO}"
echo "============================================================"
