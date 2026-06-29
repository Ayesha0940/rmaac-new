#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"
MODEL_DIR="${MODEL_DIR:-$EXPERIMENTS_DIR/model}"
DIFFUSION_MODELS_DIR="${DIFFUSION_MODELS_DIR:-$EXPERIMENTS_DIR/diffusion_models}"
LOG_DIR="$EXPERIMENTS_DIR/logs/action_noise_eval"
NUM_ADVERSARIES="${NUM_ADVERSARIES:-0}"
NUM_TEST_EPISODES="${NUM_TEST_EPISODES:-800}"

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SCENARIOS=(
    simple_adversary
    simple_crypto
    simple_push
    simple_speaker_listener
    simple_spread
    simple_tag
)

# Each entry: "VARIANT_FLAG:EXP_SUFFIX"
VARIANTS=(
    "maddpg-none:maddpg"
    "maddpg-earnie:earnie"
    "maddpg-act_adv:rmaac"
    "m3ddpg:m3ddpg"
)

# Biased Gaussian noise means to evaluate
NOISE_MU_LIST=(-1 1)

mkdir -p "$LOG_DIR"

# ─── Helpers ──────────────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# ─── Main loop ────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Biased action-noise evaluation"
echo "Scenarios    : ${SCENARIOS[*]}"
echo "Noise mu     : ${NOISE_MU_LIST[*]}"
echo "Noise sigma  : 0 1 2 3"
echo "t-start list : 20 40 60"
echo "Model dir    : $MODEL_DIR"
echo "Diffusion dir: $DIFFUSION_MODELS_DIR"
echo "============================================================"

for SCENARIO in "${SCENARIOS[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "Scenario: $SCENARIO"
    echo "════════════════════════════════════════════════════════════"

    DIFFUSION_MODEL="$DIFFUSION_MODELS_DIR/${SCENARIO}_m3ddpg.pt"
    if [[ ! -f "$DIFFUSION_MODEL" ]]; then
        echo "[$(timestamp)] [SKIP] $SCENARIO — diffusion model not found: $DIFFUSION_MODEL"
        continue
    fi
    echo "[$(timestamp)] Diffusion model found: $DIFFUSION_MODEL"

    for ENTRY in "${VARIANTS[@]}"; do
        VARIANT="${ENTRY%%:*}"
        SUFFIX="${ENTRY##*:}"
        EXP_NAME="${SCENARIO}__${SUFFIX}best"
        BASE_CKPT="$MODEL_DIR/${EXP_NAME}.index"

        if [[ ! -f "$BASE_CKPT" ]]; then
            echo "[$(timestamp)] [SKIP] $EXP_NAME — base policy checkpoint not found: $BASE_CKPT"
            continue
        fi
        echo "[$(timestamp)] Base policy found: $EXP_NAME"

        for MU in "${NOISE_MU_LIST[@]}"; do
            LOG_FILE="$LOG_DIR/${EXP_NAME}_mu${MU}.log"
            CSV_OUT="${EXPERIMENTS_DIR}/${EXP_NAME}_mu${MU}_actstd_tstart_sweep.csv"

            echo ""
            echo "────────────────────────────────────────────────────────────"
            echo "[$(timestamp)] [RUN] $EXP_NAME  noise_mu=$MU"
            echo "────────────────────────────────────────────────────────────"

            python -u "$EXPERIMENTS_DIR/train.py" \
                --scenario             "$SCENARIO" \
                --variant              "$VARIANT" \
                --mode                 test \
                --exp-name             "$EXP_NAME" \
                --save-dir             "$MODEL_DIR" \
                --num-adversaries      "$NUM_ADVERSARIES" \
                --num-test-episodes    "$NUM_TEST_EPISODES" \
                --noise-mu             "$MU" \
                --act-std-list         0 1 2 3 \
                --diffusion-model-path "$DIFFUSION_MODEL" \
                --diffusion-steps      100 \
                --t-start-list         20 40 60 \
            2>&1 | tee "$LOG_FILE"

            echo "[$(timestamp)] [DONE] $EXP_NAME  mu=$MU → $CSV_OUT"
        done
    done
done

echo ""
echo "============================================================"
echo "All evaluations complete."
echo "Logs : $LOG_DIR/"
echo "CSVs : $EXPERIMENTS_DIR/*_actstd_tstart_sweep.csv"
echo "============================================================"
