#!/usr/bin/env bash
# Run stuck_at sweep for each scenario sequentially.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -A NUM_ADV
NUM_ADV["simple_adversary"]=0
NUM_ADV["simple_push"]=0
NUM_ADV["simple_speaker_listener"]=0
NUM_ADV["simple_spread"]=0
NUM_ADV["simple_tag"]=3

SCENARIOS=(simple_adversary simple_push simple_speaker_listener simple_spread simple_tag)
SEEDS=(5 10)

for SEED in "${SEEDS[@]}"; do
    for SCENARIO in "${SCENARIOS[@]}"; do
        echo ""
        echo "========================================"
        echo "=== Scenario: ${SCENARIO} (adv=${NUM_ADV[$SCENARIO]}, seed=${SEED}) ==="
        echo "========================================"
        bash "${SCRIPT_DIR}/run_stuck_at_sweep.sh" "${SCENARIO}" "${NUM_ADV[$SCENARIO]}" "${SEED}"
    done
done

echo ""
echo "=== All scenarios (stuck_at) complete ==="
