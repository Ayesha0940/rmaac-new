#!/usr/bin/env bash
# MPE Config-B confirmation: train 66 models, then 248 evaluation jobs, then the report (detached driver)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=/home/axs0940/miniconda3/envs/rmaac_env/bin/python
JOBS=66 bash run_recon.sh train6 > logs/phase_train6.log 2>&1
n1=$(ls models | grep -cE "^c1_L(removed|bn1|bn2|bn4)_.*_s[0-4]\.pt$")
n2=$(ls models | grep -cE "^(den_s1|bc|c1_s1)_L(removed|bn1|bn2|bn4)_.*_s[34]\.pt$")
echo "train6 done: C1-linear $n1 (expect 30), seeds 3-4 den/bc/c1 $n2 (expect 36)" > logs/phase_confirm_driver.log
if [[ "$n1" -ne 30 || "$n2" -ne 36 ]] || grep -lq Traceback logs/train6_*.log; then echo "ABORT: training incomplete or errors" >> logs/phase_confirm_driver.log; exit 1; fi
JOBS=120 bash run_recon.sh confirm > logs/phase_confirm.log 2>&1
np=$(ls results/confirm/parts | wc -l)
echo "confirm done: $np parts (expect 248)" >> logs/phase_confirm_driver.log
if [[ "$np" -ne 248 ]] || grep -lq Traceback logs/eval_confirm_*.log; then echo "ABORT: evaluation incomplete or errors" >> logs/phase_confirm_driver.log; exit 1; fi
$PY report_confirm.py > logs/report_confirm.log 2>&1 && echo "report done" >> logs/phase_confirm_driver.log || echo "ABORT: report failed (see logs/report_confirm.log)" >> logs/phase_confirm_driver.log
