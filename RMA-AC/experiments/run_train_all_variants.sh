#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
# Change SCENARIO here (or pass as env var) to train on a different MPE env.
SCENARIO="${SCENARIO:-simple_speaker_listener}"
NUM_EPISODES="${NUM_EPISODES:-60000}"
SAVE_RATE="${SAVE_RATE:-500}"
NUM_ADVERSARIES="${NUM_ADVERSARIES:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/experiments/model}"
LOG_DIR="$PROJECT_ROOT/experiments/logs/train_all_variants"

export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "$LOG_DIR"

# ─── Helpers ──────────────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_phase() {
    local phase="$1"
    local log_file="$2"
    shift 2

    echo "[$(timestamp)] [${phase}] START"
    set +e
    {
        echo "[$(timestamp)] [${phase}] COMMAND: $*"
        "$@"
    } 2>&1 | tee "$log_file"
    local rc=${PIPESTATUS[0]}
    set -e

    if [[ $rc -ne 0 ]]; then
        echo "[$(timestamp)] [${phase}] FAILED (exit code: ${rc})"
        tail -n 40 "$log_file"
        exit "$rc"
    fi
    echo "[$(timestamp)] [${phase}] DONE"
}

smoke_test() {
    local exp_name="$1"
    local best_dir="${MODEL_DIR}/${exp_name}best"
    if compgen -G "${best_dir}/*.index" > /dev/null 2>&1; then
        printf "[smoke] %-45s → PASS (checkpoint found)\n" "${exp_name}best"
    else
        printf "[smoke] %-45s → FAIL (no checkpoint in %s)\n" "${exp_name}best" "$best_dir"
        return 1
    fi
}

# ─── Variant definitions ───────────────────────────────────────────────────────
# Each entry: "VARIANT_FLAG:EXP_SUFFIX"
# Checkpoint dirs: MODEL_DIR/<SCENARIO>__<EXP_SUFFIX>/  (final)
#                  MODEL_DIR/<SCENARIO>__<EXP_SUFFIX>best/  (best, auto-saved)
VARIANTS=(
    "maddpg-none:maddpg"
    "maddpg-earnie:earnie"
    "maddpg-act_adv:rmaac"
    "m3ddpg:m3ddpg"
)

# ─── Training loop ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "Scenario      : ${SCENARIO}"
echo "Num episodes  : ${NUM_EPISODES}"
echo "Save rate     : ${SAVE_RATE}"
echo "Num adversaries: ${NUM_ADVERSARIES}"
echo "Model dir     : ${MODEL_DIR}"
echo "============================================================"

for entry in "${VARIANTS[@]}"; do
    VARIANT="${entry%%:*}"
    SUFFIX="${entry##*:}"
    EXP_NAME="${SCENARIO}__${SUFFIX}"

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "Variant : ${VARIANT}  →  exp-name: ${EXP_NAME}"
    echo "────────────────────────────────────────────────────────────"

    run_phase "${EXP_NAME}:train" "${LOG_DIR}/${EXP_NAME}.train.log" \
        env SUPPRESS_MA_PROMPT=1 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 \
        python -u "${PROJECT_ROOT}/experiments/train.py" \
            --scenario        "$SCENARIO" \
            --variant         "$VARIANT" \
            --num-episodes    "$NUM_EPISODES" \
            --save-rate       "$SAVE_RATE" \
            --num-adversaries "$NUM_ADVERSARIES" \
            --save-dir        "$MODEL_DIR" \
            --exp-name        "$EXP_NAME"
done

# ─── Smoke tests ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Smoke tests — verifying best checkpoints"
echo "============================================================"

all_pass=true
for entry in "${VARIANTS[@]}"; do
    SUFFIX="${entry##*:}"
    EXP_NAME="${SCENARIO}__${SUFFIX}"
    smoke_test "$EXP_NAME" || all_pass=false
done

if [[ "$all_pass" == "false" ]]; then
    echo ""
    echo "ERROR: one or more best checkpoints are missing — check logs in ${LOG_DIR}/"
    exit 1
fi

echo ""
echo "All variants trained successfully."
echo "Logs    : ${LOG_DIR}/"
echo "Models  : ${MODEL_DIR}/"
echo ""
echo "Eval examples:"
echo "  python experiments/train.py --variant maddpg-none     --mode test --save-dir ${MODEL_DIR} --exp-name ${SCENARIO}__maddpgbest"
echo "  python experiments/train.py --variant maddpg-earnie   --mode test --save-dir ${MODEL_DIR} --exp-name ${SCENARIO}__earniebest"
echo "  python experiments/train.py --variant maddpg-act_adv  --mode test --save-dir ${MODEL_DIR} --exp-name ${SCENARIO}__rmaacbest"
echo "  python experiments/train.py --variant m3ddpg          --mode test --save-dir ${MODEL_DIR} --exp-name ${SCENARIO}__m3ddpgbest"
