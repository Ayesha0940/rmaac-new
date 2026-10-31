#!/usr/bin/env bash
# per-agent denoiser stage: train the 7 new models, then evaluate the 24 cells (detached driver)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
bash run_recon.sh train3 > logs/phase_train3.log 2>&1
n=$(ls models | grep -E "^den_s1_|^den_simple_adversary" | wc -l)
echo "train3 done, new models: $n (expect 7)" > logs/phase_pa_driver.log
if [[ "$n" -eq 7 ]]; then bash run_recon.sh pa > logs/phase_pa.log 2>&1; echo "pa done" >> logs/phase_pa_driver.log; else echo "ABORT: model count $n" >> logs/phase_pa_driver.log; fi
