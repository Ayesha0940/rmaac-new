#!/usr/bin/env bash
# Step 2: train 49 models (C1 seeds 0-2, denoiser/BC seeds 1-2 for the 7 candidate levels), then 169 evaluation jobs
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
JOBS=49 bash run_recon.sh train5 > logs/phase_train5.log 2>&1
n=$(( $(ls models | grep -cE "^c1_s1_L") + $(ls models | grep -cE "^(den_s1|bc)_L.*_s[12]\.pt$") ))
echo "train5 done: new models $n (expect 49)" > logs/phase_step2_driver.log
if [[ "$n" -eq 49 ]]; then JOBS=120 bash run_recon.sh step2 > logs/phase_step2.log 2>&1; echo "step2 done" >> logs/phase_step2_driver.log; else echo "ABORT: model count $n" >> logs/phase_step2_driver.log; fi
