#!/usr/bin/env bash
# State-only BC baseline: train one BC per scenario on the H=25 dataset, then
# roll out BC + the 4 clean base policies for each seed.
#
# Usage:
#   bash run_bc_pipeline.sh                    # all 5 scenarios, seeds 47 48 49
#   SCENARIOS="simple_spread" SEEDS="47" NUM_TEST_EPISODES=5 bash run_bc_pipeline.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""      # CPU, as in the other sweep scripts
export SUPPRESS_MA_PROMPT=1         # multiagent package otherwise blocks on input()
export PYTHONUNBUFFERED=1

PY="${PY:-python}"
SCENARIOS="${SCENARIOS:-simple_adversary simple_push simple_speaker_listener simple_spread simple_tag}"
SEEDS="${SEEDS:-47 48 49}"
NUM_TEST_EPISODES="${NUM_TEST_EPISODES:-800}"
BC_EPOCHS="${BC_EPOCHS:-500}"
LOG_DIR="${SCRIPT_DIR}/logs/bc"
mkdir -p "$LOG_DIR"

for SCENARIO in $SCENARIOS; do
    MODEL="${SCRIPT_DIR}/bc_models/${SCENARIO}_m3ddpg_bc.pt"
    if [[ -f "$MODEL" ]]; then
        echo "[train_bc] $MODEL exists, skipping"
    else
        $PY -u train_bc.py --scenario "$SCENARIO" --epochs "$BC_EPOCHS" 2>&1 | tee "$LOG_DIR/${SCENARIO}_train_bc.log"
    fi
    for SEED in $SEEDS; do
        $PY -u run_bc_sweep.py --scenario "$SCENARIO" --num-test-episodes "$NUM_TEST_EPISODES" \
            --seed "$SEED" --save-dir ./model 2>&1 | tee "$LOG_DIR/${SCENARIO}_seed${SEED}.log"
    done
done
