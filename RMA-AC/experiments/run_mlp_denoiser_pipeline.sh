#!/usr/bin/env bash
set -euo pipefail

# Trains one MLP denoiser per scenario on the existing m3ddpg rollouts, then
# evaluates them on the maddpg checkpoints. Existing models and CSVs are kept.
# Run inside the rmaac_env conda environment.

SCENARIOS="${SCENARIOS:-simple_adversary simple_push simple_speaker_listener simple_spread}"
SEED="${SEED:-0}"
NUM_TEST_EPISODES="${NUM_TEST_EPISODES:-800}"
EPOCHS="${EPOCHS:-200}"
COND_MODE="${COND_MODE:-current}"  # initial: same state pairing as the DDPM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/model}"
SUFFIX="$([[ "$COND_MODE" == "current" ]] && echo "" || echo "_${COND_MODE}")"
MLP_MODEL_DIR="${MLP_MODEL_DIR:-$SCRIPT_DIR/mlp_denoiser_models${SUFFIX}}"
MLP_OUT_ROOT="${MLP_OUT_ROOT:-$PROJECT_ROOT/../noise_sweeps/mlp_denoiser${SUFFIX}}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/mlp_denoiser${SUFFIX}}"

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

mkdir -p "$LOG_DIR" "$MLP_MODEL_DIR"
cd "$SCRIPT_DIR"

for SCENARIO in $SCENARIOS; do
    python -u mlp_denoiser.py \
        --data-path  "diffusion_data/${SCENARIO}_m3ddpg.npz" \
        --model-path "${MLP_MODEL_DIR}/${SCENARIO}_m3ddpg.pt" \
        --cond-mode  "$COND_MODE" \
        --epochs     "$EPOCHS" \
        --seed       "$SEED" \
        2>&1 | tee -a "${LOG_DIR}/train_${SCENARIO}.log"
done

pids=()
for SCENARIO in $SCENARIOS; do
    python -u run_mlp_denoiser_sweep.py \
        --mlp-scenarios "$SCENARIO" \
        --mlp-model-dir "$MLP_MODEL_DIR" \
        --mlp-out-root  "$MLP_OUT_ROOT" \
        --save-dir      "$MODEL_DIR" \
        --num-test-episodes "$NUM_TEST_EPISODES" \
        --seed          "$SEED" \
        > "${LOG_DIR}/eval_${SCENARIO}.log" 2>&1 &
    pids+=($!)
    echo "Launched eval:${SCENARIO} (PID $!)"
done

any_failed=false
for pid in "${pids[@]}"; do
    wait "$pid" || any_failed=true
done
if [[ "$any_failed" == "true" ]]; then
    echo "One or more eval jobs failed -- check logs in ${LOG_DIR}/"
    exit 1
fi
echo "All done. CSVs in ${MLP_OUT_ROOT}/seed${SEED}/"
