#!/usr/bin/env bash
# Speaker-Listener hardening: train 5 C1-gauss models (speaker 'removed'), evaluate 40 cells (C1-gauss only), report.
# Env: MODEL_DIR (default models), RESULT_DIR (default results/speaker_hardening), EPOCHS (200), EPISODES (800), SUMMARY.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="" SUPPRESS_MA_PROMPT=1 PYTHONUNBUFFERED=1
PY=/home/axs0940/miniconda3/envs/rmaac_env/bin/python
MODEL_DIR="${MODEL_DIR:-$PWD/models}"; RD="${RESULT_DIR:-$PWD/results/speaker_hardening}"
EPOCHS="${EPOCHS:-200}"; EPISODES="${EPISODES:-800}"; SUMMARY="${SUMMARY:-$PWD/results/speaker_hardening_summary.txt}"
L=logs/phase_speaker_hardening.log; mkdir -p "$RD/parts" logs
for s in 0 1 2 3 4; do
  echo "$PY -u train_recon.py --scenario simple_speaker_listener --info full --role speaker --method c2 --schedule linear --level removed --seed $s --epochs $EPOCHS --model-dir $MODEL_DIR > logs/train_hard_c2_s$s.log 2>&1"
done | xargs -d '\n' -P 5 -I{} bash -c '{}'
n=$(ls "$MODEL_DIR" | grep -cE "^c2_Lremoved_simple_speaker_listener_full_speaker_s[0-4]\.pt$")
echo "train done: $n C1-gauss models (expect 5)" > $L
if [[ "$n" -ne 5 ]] || grep -lq Traceback logs/train_hard_c2_s*.log; then echo "ABORT: training" >> $L; exit 1; fi
sha256sum "$MODEL_DIR"/c2_Lremoved_simple_speaker_listener_full_speaker_s[0-4].pt > logs/hard_sha_before.txt
for cell in "gauss 0" "gauss 1" "gauss 3" "biased 1" "delay 3" "delay 8" "stuck 0.5" "stuck 1.0"; do
  set -- $cell
  for s in 0 1 2 3 4; do
    echo "$PY -u eval_il.py --phase hardening --scenario simple_speaker_listener --varied speaker --info-level removed --fault $1 --flevel $2 --seed $s --episodes $EPISODES --model-dir $MODEL_DIR --shift --only c1g --out $RD/parts/c1g_$1_$2_s$s.csv > logs/eval_hard_$1_$2_s$s.log 2>&1"
  done
done | xargs -d '\n' -P 40 -I{} bash -c '{}'
np=$(ls "$RD/parts" | wc -l); echo "eval done: $np parts (expect 40)" >> $L
if [[ "$np" -ne 40 ]] || grep -lq Traceback logs/eval_hard_*.log; then echo "ABORT: evaluation" >> $L; exit 1; fi
sha256sum "$MODEL_DIR"/c2_Lremoved_simple_speaker_listener_full_speaker_s[0-4].pt > logs/hard_sha_after.txt
cmp -s logs/hard_sha_before.txt logs/hard_sha_after.txt || { echo "ABORT: checkpoints changed during evaluation" >> $L; exit 1; }
echo "checkpoints unchanged during evaluation" >> $L
$PY report_speaker_hardening.py --hard-dir "$RD" --out "$SUMMARY" --model-dir "$MODEL_DIR" > logs/report_hardening.log 2>&1 && echo "report done" >> $L || echo "ABORT: report failed" >> $L
