#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
DIFFUSION_EPOCHS="${DIFFUSION_EPOCHS:-500}"
DIFFUSION_LR="${DIFFUSION_LR:-1e-4}"
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-64}"
T_START_LIST="${T_START_LIST:-20 40}"
ACT_STD_LIST="${ACT_STD_LIST:-0.0 1.0 2.0 3.0}"
NUM_TEST_EPISODES="${NUM_TEST_EPISODES:-800}"

# Restrict to a subset of scenarios for smoke-testing, e.g.:
#   SCENARIOS="simple_push" ./run_diffusion_pipeline_h1.sh
SCENARIOS="${SCENARIOS:-simple_adversary simple_push simple_speaker_listener simple_spread simple_tag}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"
MODEL_DIR="${MODEL_DIR:-$EXPERIMENTS_DIR/model}"
DATA_DIR="$EXPERIMENTS_DIR/diffusion_data"
MODELS_DIR="$EXPERIMENTS_DIR/diffusion_models"
RESULTS_ROOT="$EXPERIMENTS_DIR/results/diffusion_h1"
LOG_DIR="$EXPERIMENTS_DIR/logs/diffusion_pipeline_h1"

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "$LOG_DIR" "$RESULTS_ROOT"

declare -A NUM_ADV=(
    [simple_adversary]=0
    [simple_push]=0
    [simple_speaker_listener]=0
    [simple_spread]=0
    [simple_tag]=3
)

declare -A VARIANT_FLAG
VARIANT_FLAG["maddpg"]="maddpg-none"
VARIANT_FLAG["earnie"]="maddpg-earnie"
VARIANT_FLAG["rmaac"]="maddpg-act_adv"
VARIANT_FLAG["m3ddpg"]="m3ddpg"

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

for SCENARIO in $SCENARIOS; do
    NUM_ADVERSARIES="${NUM_ADV[$SCENARIO]}"
    SRC_DATA="$DATA_DIR/${SCENARIO}_m3ddpg.npz"
    H1_DATA="$DATA_DIR/${SCENARIO}_m3ddpg_H1.npz"
    H1_MODEL="$MODELS_DIR/${SCENARIO}_m3ddpg_H1.pt"
    RESULTS_DIR="$RESULTS_ROOT/${SCENARIO}"
    mkdir -p "$RESULTS_DIR"

    echo ""
    echo "============================================================"
    echo "Scenario: ${SCENARIO}  (num_adversaries=${NUM_ADVERSARIES})"
    echo "Source H=25 data : ${SRC_DATA}"
    echo "New H=1 data     : ${H1_DATA}"
    echo "New H=1 model    : ${H1_MODEL}"
    echo "Results dir      : ${RESULTS_DIR}"
    echo "============================================================"

    if [[ ! -f "$SRC_DATA" ]]; then
        echo "ERROR: expected existing H=25 data not found: $SRC_DATA"
        exit 1
    fi

    # ─── Step 1: derive H=1 data by slicing existing H=25 rollouts ───────────
    if [[ -f "$H1_DATA" ]]; then
        echo "[$(timestamp)] [make_h1_data] $H1_DATA already exists — skipping."
    else
        run_phase "make_h1_data:${SCENARIO}" "${LOG_DIR}/${SCENARIO}_make_h1_data.log" \
            python -u "${EXPERIMENTS_DIR}/make_h1_diffusion_data.py" \
                --src "$SRC_DATA" --dst "$H1_DATA"
    fi

    # ─── Step 2: train the H=1 denoiser ───────────────────────────────────────
    if [[ -f "$H1_MODEL" ]]; then
        echo "[$(timestamp)] [train_diffusion] $H1_MODEL already exists — skipping."
    else
        run_phase "train_diffusion:${SCENARIO}" "${LOG_DIR}/${SCENARIO}_train_diffusion.log" \
            env PYTHONUNBUFFERED=1 \
            python -u "${EXPERIMENTS_DIR}/train.py" \
                --scenario             "$SCENARIO" \
                --mode                 train_diffusion \
                --diffusion-data-path  "$H1_DATA" \
                --diffusion-model-path "$H1_MODEL" \
                --diffusion-horizon    1 \
                --diffusion-steps      "$DIFFUSION_STEPS" \
                --diffusion-epochs     "$DIFFUSION_EPOCHS" \
                --diffusion-lr         "$DIFFUSION_LR" \
                --diffusion-batch-size "$DIFFUSION_BATCH_SIZE"
    fi

    if [[ ! -f "$H1_MODEL" ]]; then
        echo "ERROR: H=1 diffusion model not found at ${H1_MODEL}"
        exit 1
    fi

    # ─── Step 3: evaluate all 4 best checkpoints with the H=1 denoiser ───────
    echo ""
    echo "[$(timestamp)] Evaluating 4 variants for ${SCENARIO} with H=1 denoiser"
    pids=()
    for algo in maddpg earnie rmaac; do
        vflag="${VARIANT_FLAG[$algo]}"
        exp_name="${SCENARIO}__${algo}best"

        (
            cd "$RESULTS_DIR" && \
            run_phase "eval:${SCENARIO}:${algo}" "${LOG_DIR}/${SCENARIO}_eval_${algo}.log" \
                env SUPPRESS_MA_PROMPT=1 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 \
                    OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
                python -u "${EXPERIMENTS_DIR}/train.py" \
                    --scenario             "$SCENARIO" \
                    --variant              "$vflag" \
                    --mode                 test \
                    --num-adversaries      "$NUM_ADVERSARIES" \
                    --save-dir             "$MODEL_DIR" \
                    --exp-name             "$exp_name" \
                    --num-test-episodes    "$NUM_TEST_EPISODES" \
                    --diffusion-model-path "$H1_MODEL" \
                    --diffusion-steps      "$DIFFUSION_STEPS" \
                    --act-std-list         $ACT_STD_LIST \
                    --t-start-list         $T_START_LIST
        ) &
        pids+=($!)
        echo "[$(timestamp)] Launched eval:${SCENARIO}:${algo} (PID $!)"
    done

    any_failed=false
    for pid in "${pids[@]}"; do
        wait "$pid" || any_failed=true
    done
    if [[ "$any_failed" == "true" ]]; then
        echo "[$(timestamp)] One or more eval jobs failed for ${SCENARIO} — check logs in ${LOG_DIR}/"
        exit 1
    fi

    echo "[$(timestamp)] Scenario ${SCENARIO} complete. CSVs in ${RESULTS_DIR}/"
done

echo ""
echo "============================================================"
echo "All scenarios complete."
echo "Data    : ${DATA_DIR}/*_m3ddpg_H1.npz"
echo "Models  : ${MODELS_DIR}/*_m3ddpg_H1.pt"
echo "Results : ${RESULTS_ROOT}/<scenario>/*_mu0.0_actstd_tstart_sweep.csv"
echo "Logs    : ${LOG_DIR}/"
echo "============================================================"
