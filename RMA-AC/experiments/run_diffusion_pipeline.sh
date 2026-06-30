#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
SCENARIO="${SCENARIO:-simple_speaker_listener}"
NUM_ADVERSARIES="${NUM_ADVERSARIES:-0}"

# Diffusion hyperparameters
DIFFUSION_COLLECT_EPISODES="${DIFFUSION_COLLECT_EPISODES:-2000}"
DIFFUSION_HORIZON="${DIFFUSION_HORIZON:-25}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
DIFFUSION_EPOCHS="${DIFFUSION_EPOCHS:-500}"
DIFFUSION_LR="${DIFFUSION_LR:-1e-4}"
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-64}"
T_START_LIST="${T_START_LIST:-20 40 60}"

# Evaluation
NUM_TEST_EPISODES="${NUM_TEST_EPISODES:-800}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/experiments/model}"
LOG_DIR="$PROJECT_ROOT/experiments/logs/diffusion_pipeline"
DIFFUSION_DATA="${PROJECT_ROOT}/experiments/diffusion_data/${SCENARIO}_m3ddpg.npz"
DIFFUSION_MODEL="${PROJECT_ROOT}/experiments/diffusion_models/${SCENARIO}_m3ddpg.pt"

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "$LOG_DIR" "$(dirname "$DIFFUSION_DATA")" "$(dirname "$DIFFUSION_MODEL")"

# ─── Helpers ──────────────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_phase() {
    local phase="$1"; local log_file="$2"; shift 2
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "[$(timestamp)] [${phase}] START"
    echo "  CMD: $*"
    echo "────────────────────────────────────────────────────────────"
    set +e
    { "$@"; } 2>&1 | tee "$log_file"
    local rc=${PIPESTATUS[0]}; set -e
    if [[ $rc -ne 0 ]]; then
        echo "[$(timestamp)] [${phase}] FAILED (exit code: ${rc})"
        tail -n 40 "$log_file"
        exit "$rc"
    fi
    echo "[$(timestamp)] [${phase}] DONE"
}

# ─── Print config ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "Scenario         : ${SCENARIO}"
echo "Collect episodes : ${DIFFUSION_COLLECT_EPISODES}"
echo "Diffusion horizon: ${DIFFUSION_HORIZON}"
echo "Diffusion steps  : ${DIFFUSION_STEPS}"
echo "Diffusion epochs : ${DIFFUSION_EPOCHS}"
echo "Data path        : ${DIFFUSION_DATA}"
echo "Model path       : ${DIFFUSION_MODEL}"
echo "Model dir        : ${MODEL_DIR}"
echo "============================================================"

# ─── Checkpoint used for data collection ─────────────────────────────────────
# m3ddpg best checkpoint: MODEL_DIR/<SCENARIO>__m3ddpgbest/
COLLECT_EXP="${SCENARIO}__m3ddpgbest"

# ─── Step 1: Collect trajectories from m3ddpg best checkpoint ────────────────
run_phase "collect_diffusion" "${LOG_DIR}/collect.log" \
    env SUPPRESS_MA_PROMPT=1 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 \
    python -u "${PROJECT_ROOT}/experiments/train.py" \
        --scenario "$SCENARIO" \
        --variant  m3ddpg \
        --mode     collect_diffusion \
        --num-adversaries "$NUM_ADVERSARIES" \
        --save-dir "$MODEL_DIR" \
        --exp-name "$COLLECT_EXP" \
        --num-episodes "$DIFFUSION_COLLECT_EPISODES" \
        --diffusion-data-path "$DIFFUSION_DATA" \
        --diffusion-horizon   "$DIFFUSION_HORIZON"

# ─── Step 2: Train the DDPM denoiser ─────────────────────────────────────────
run_phase "train_diffusion" "${LOG_DIR}/train_diffusion.log" \
    env PYTHONUNBUFFERED=1 \
    python -u "${PROJECT_ROOT}/experiments/train.py" \
        --scenario             "$SCENARIO" \
        --mode                 train_diffusion \
        --diffusion-data-path  "$DIFFUSION_DATA" \
        --diffusion-model-path "$DIFFUSION_MODEL" \
        --diffusion-horizon    "$DIFFUSION_HORIZON" \
        --diffusion-steps      "$DIFFUSION_STEPS" \
        --diffusion-epochs     "$DIFFUSION_EPOCHS" \
        --diffusion-lr         "$DIFFUSION_LR" \
        --diffusion-batch-size "$DIFFUSION_BATCH_SIZE"

# Verify the model was saved
if [[ ! -f "$DIFFUSION_MODEL" ]]; then
    echo "ERROR: diffusion model not found at ${DIFFUSION_MODEL}"
    exit 1
fi
echo "[smoke] Diffusion model saved → PASS (${DIFFUSION_MODEL})"

# ─── Step 3: Evaluate all 4 best checkpoints with the denoiser ───────────────
# Variant → (variant-flag, exp-suffix)
declare -A VARIANT_FLAG
VARIANT_FLAG["maddpg"]="maddpg-none"
VARIANT_FLAG["earnie"]="maddpg-earnie"
VARIANT_FLAG["rmaac"]="maddpg-act_adv"
VARIANT_FLAG["m3ddpg"]="m3ddpg"

echo ""
echo "============================================================"
echo "Evaluating 4 best checkpoints with diffusion denoiser"
echo "t_start sweep: ${T_START_LIST}"
echo "============================================================"

pids=()
for algo in maddpg earnie rmaac m3ddpg; do
    vflag="${VARIANT_FLAG[$algo]}"
    exp_name="${SCENARIO}__${algo}best"

    (
        run_phase "eval:${algo}" "${LOG_DIR}/eval_${algo}.log" \
            env SUPPRESS_MA_PROMPT=1 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 \
                OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
            python -u "${PROJECT_ROOT}/experiments/train.py" \
                --scenario             "$SCENARIO" \
                --variant              "$vflag" \
                --mode                 test \
                --num-adversaries      "$NUM_ADVERSARIES" \
                --save-dir             "$MODEL_DIR" \
                --exp-name             "$exp_name" \
                --num-test-episodes    "$NUM_TEST_EPISODES" \
                --diffusion-model-path "$DIFFUSION_MODEL" \
                --diffusion-steps      "$DIFFUSION_STEPS" \
                --t-start-list         $T_START_LIST
    ) &
    pids+=($!)
    echo "[$(timestamp)] Launched eval:${algo} (PID $!)"
done

any_failed=false
for pid in "${pids[@]}"; do
    wait "$pid" || any_failed=true
done
if [[ "$any_failed" == "true" ]]; then
    echo "[$(timestamp)] One or more eval jobs failed — check logs in ${LOG_DIR}/"
    exit 1
fi

echo ""
echo "============================================================"
echo "All done."
echo "Logs    : ${LOG_DIR}/"
echo "Results : one CSV per algorithm in the experiments/ dir:"
for algo in maddpg earnie rmaac m3ddpg; do
    echo "  ${SCENARIO}__${algo}best_actstd_tstart_sweep.csv"
done
echo "============================================================"
