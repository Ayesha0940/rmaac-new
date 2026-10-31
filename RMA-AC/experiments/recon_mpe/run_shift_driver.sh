#!/usr/bin/env bash
# MPE corruption-shift: train 10 C1-gauss models, then 80 evaluation jobs, then the report (detached driver)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=/home/axs0940/miniconda3/envs/rmaac_env/bin/python
L=logs/phase_shift_driver.log
JOBS=10 bash run_recon.sh train7 > logs/phase_train7.log 2>&1
n=$(ls models | grep -cE "^c2_L(removed|bn2)_.*_s[0-4]\.pt$")
echo "train7 done: C1-gauss models $n (expect 10)" > $L
if [[ "$n" -ne 10 ]] || grep -lq Traceback logs/train6_*_c2_linear_*.log; then echo "ABORT: training incomplete or errors" >> $L; exit 1; fi
sha256sum models/c2_L*_s[0-4].pt > logs/shift_ckpt_sha_before.txt
JOBS=80 bash run_recon.sh shift > logs/phase_shift.log 2>&1
np=$(ls results/shift/parts | wc -l)
echo "shift done: $np parts (expect 80)" >> $L
if [[ "$np" -ne 80 ]] || grep -lq Traceback logs/eval_shift_*.log; then echo "ABORT: evaluation incomplete or errors" >> $L; exit 1; fi
sha256sum models/c2_L*_s[0-4].pt > logs/shift_ckpt_sha_after.txt
if ! cmp -s logs/shift_ckpt_sha_before.txt logs/shift_ckpt_sha_after.txt; then echo "ABORT: C1-gauss checkpoints changed during evaluation" >> $L; exit 1; fi
echo "C1-gauss checkpoints unchanged during evaluation (sha256 before == after)" >> $L
$PY report_shift.py > logs/report_shift.log 2>&1 && echo "report done" >> $L || echo "ABORT: report failed (see logs/report_shift.log)" >> $L
