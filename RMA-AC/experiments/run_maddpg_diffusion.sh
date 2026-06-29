#!/usr/bin/env bash
# Collect trajectories from a MADDPG checkpoint, train a diffusion denoiser,
# then evaluate MADDPG + diffusion under action noise.
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
SCENARIO="${SCENARIO:-simple_crypto}"
NUM_ADVERSARIES="${NUM_ADVERSARIES:-0}"

# Collection
NUM_COLLECT_EPISODES="${NUM_COLLECT_EPISODES:-2000}"

# Diffusion training
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
DIFFUSION_EPOCHS="${DIFFUSION_EPOCHS:-500}"
DIFFUSION_HORIZON="${DIFFUSION_HORIZON:-25}"
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-64}"
DIFFUSION_LR="${DIFFUSION_LR:-1e-4}"

# Evaluation
NUM_TEST_EPISODES="${NUM_TEST_EPISODES:-800}"
ACT_STD_LIST="${ACT_STD_LIST:-0 0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.0}"
T_START_LIST="${T_START_LIST:-20 40 60}"
NOISE_MU="${NOISE_MU:-0}"

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"
MODEL_DIR="${MODEL_DIR:-$EXPERIMENTS_DIR/model}"
DIFFUSION_DATA_DIR="${DIFFUSION_DATA_DIR:-$EXPERIMENTS_DIR/diffusion_data}"
DIFFUSION_MODELS_DIR="${DIFFUSION_MODELS_DIR:-$EXPERIMENTS_DIR/diffusion_models}"
LOG_DIR="$EXPERIMENTS_DIR/logs/maddpg_diffusion"

EXP_NAME="${SCENARIO}__maddpgbest"
DIFFUSION_DATA_PATH="$DIFFUSION_DATA_DIR/${SCENARIO}_maddpg.npz"
DIFFUSION_MODEL_PATH="$DIFFUSION_MODELS_DIR/${SCENARIO}_maddpg.pt"

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "$LOG_DIR" "$DIFFUSION_DATA_DIR" "$DIFFUSION_MODELS_DIR"

# ─── Helpers ──────────────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_phase() {
    local phase="$1"
    local log_file="$2"
    shift 2
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "[$(timestamp)] [${phase}] START"
    echo "[$(timestamp)] [${phase}] LOG: $log_file"
    echo "────────────────────────────────────────────────────────────"
    set +e
    { echo "COMMAND: $*"; "$@"; } 2>&1 | tee "$log_file"
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -ne 0 ]]; then
        echo "[$(timestamp)] [${phase}] FAILED (exit=$rc) — see $log_file"
        exit "$rc"
    fi
    echo "[$(timestamp)] [${phase}] DONE"
}

# ─── Preflight ────────────────────────────────────────────────────────────────
echo "============================================================"
echo "MADDPG Diffusion Pipeline"
echo "Scenario         : $SCENARIO"
echo "Checkpoint       : $EXP_NAME"
echo "Collect episodes : $NUM_COLLECT_EPISODES"
echo "Diffusion steps  : $DIFFUSION_STEPS  epochs=$DIFFUSION_EPOCHS  horizon=$DIFFUSION_HORIZON"
echo "Data path        : $DIFFUSION_DATA_PATH"
echo "Model path       : $DIFFUSION_MODEL_PATH"
echo "Test episodes    : $NUM_TEST_EPISODES  noise_mu=$NOISE_MU"
echo "act_std sweep    : $ACT_STD_LIST"
echo "t_start sweep    : $T_START_LIST"
echo "============================================================"

if [[ ! -f "$MODEL_DIR/${EXP_NAME}.index" ]]; then
    echo "ERROR: MADDPG checkpoint not found: $MODEL_DIR/${EXP_NAME}.index"
    echo "Train first with run_train_all_variants.sh or set MODEL_DIR."
    exit 1
fi

# ─── Phase 1: Collect trajectories ────────────────────────────────────────────
if [[ -f "$DIFFUSION_DATA_PATH" ]]; then
    echo "[$(timestamp)] [collect] Data already exists at $DIFFUSION_DATA_PATH — skipping collection."
else
    run_phase "collect" "$LOG_DIR/${SCENARIO}_maddpg_collect.log" \
        python -u "$EXPERIMENTS_DIR/train.py" \
            --scenario             "$SCENARIO" \
            --variant              "maddpg-none" \
            --mode                 collect_diffusion \
            --exp-name             "$EXP_NAME" \
            --save-dir             "$MODEL_DIR" \
            --num-adversaries      "$NUM_ADVERSARIES" \
            --num-episodes         "$NUM_COLLECT_EPISODES" \
            --diffusion-horizon    "$DIFFUSION_HORIZON" \
            --diffusion-data-path  "$DIFFUSION_DATA_PATH"
fi

# ─── Phase 2: Train diffusion model ───────────────────────────────────────────
if [[ -f "$DIFFUSION_MODEL_PATH" ]]; then
    echo "[$(timestamp)] [train_diffusion] Model already exists at $DIFFUSION_MODEL_PATH — skipping training."
else
    run_phase "train_diffusion" "$LOG_DIR/${SCENARIO}_maddpg_traindiff.log" \
        python -u "$EXPERIMENTS_DIR/train.py" \
            --scenario              "$SCENARIO" \
            --variant               "maddpg-none" \
            --mode                  train_diffusion \
            --diffusion-data-path   "$DIFFUSION_DATA_PATH" \
            --diffusion-model-path  "$DIFFUSION_MODEL_PATH" \
            --diffusion-steps       "$DIFFUSION_STEPS" \
            --diffusion-epochs      "$DIFFUSION_EPOCHS" \
            --diffusion-horizon     "$DIFFUSION_HORIZON" \
            --diffusion-batch-size  "$DIFFUSION_BATCH_SIZE" \
            --diffusion-lr          "$DIFFUSION_LR"
fi

# ─── Phase 3: Evaluate MADDPG + diffusion under action noise ──────────────────
run_phase "eval" "$LOG_DIR/${SCENARIO}_maddpg_eval.log" \
    python -u "$EXPERIMENTS_DIR/train.py" \
        --scenario             "$SCENARIO" \
        --variant              "maddpg-none" \
        --mode                 test \
        --exp-name             "$EXP_NAME" \
        --save-dir             "$MODEL_DIR" \
        --num-adversaries      "$NUM_ADVERSARIES" \
        --num-test-episodes    "$NUM_TEST_EPISODES" \
        --noise-mu             "$NOISE_MU" \
        --act-std-list         $ACT_STD_LIST \
        --diffusion-model-path "$DIFFUSION_MODEL_PATH" \
        --diffusion-steps      "$DIFFUSION_STEPS" \
        --t-start-list         $T_START_LIST

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Pipeline complete for $SCENARIO / maddpg"
echo "Data    : $DIFFUSION_DATA_PATH"
echo "Model   : $DIFFUSION_MODEL_PATH"
echo "Logs    : $LOG_DIR/"
CSV_GLOB="$EXPERIMENTS_DIR/${EXP_NAME}_mu${NOISE_MU}_actstd_tstart_sweep.csv"
echo "CSV     : $CSV_GLOB"
echo "============================================================"
