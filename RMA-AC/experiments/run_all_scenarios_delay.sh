#!/usr/bin/env bash
# Run delay sweep for each scenario sequentially.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# scenario -> num_adversaries
declare -A NUM_ADV
NUM_ADV["simple_adversary"]=0
NUM_ADV["simple_push"]=0
NUM_ADV["simple_speaker_listener"]=0
NUM_ADV["simple_spread"]=0
NUM_ADV["simple_tag"]=3

SCENARIOS=(simple_adversary simple_push simple_speaker_listener simple_spread simple_tag)

for SCENARIO in "${SCENARIOS[@]}"; do
    echo ""
    echo "========================================"
    echo "=== Scenario: ${SCENARIO} (adv=${NUM_ADV[$SCENARIO]}) ==="
    echo "========================================"
    bash "${SCRIPT_DIR}/run_delay_sweep.sh" "${SCENARIO}" "${NUM_ADV[$SCENARIO]}"
done

echo ""
echo "=== All scenarios complete ==="
