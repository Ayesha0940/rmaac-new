#!/usr/bin/env bash
# information-level sweep: train 35 models, then 168 evaluation jobs (detached driver)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
JOBS=35 bash run_recon.sh train4 > logs/phase_train4.log 2>&1
n=$(( $(ls models | grep -c "_L") + $(ls models | grep -c "^bc_simple_adversary_full_good") ))
echo "train4 done: new models $n (expect 35)" > logs/phase_il_driver.log
if [[ "$n" -eq 35 ]]; then JOBS=120 bash run_recon.sh il > logs/phase_il.log 2>&1; echo "il done" >> logs/phase_il_driver.log; else echo "ABORT: model count $n" >> logs/phase_il_driver.log; fi
