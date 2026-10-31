#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="" SUPPRESS_MA_PROMPT=1 PYTHONUNBUFFERED=1
PY=/home/axs0940/miniconda3/envs/rmaac_env/bin/python
OUT=/home/axs0940/rmaac-new/noise_sweeps/decentralized/state_only/seed0
for sc in simple_adversary simple_push simple_speaker_listener simple_spread; do for m in full reduced; do
  echo "$PY -u eval_state_only.py --scenario $sc --mode $m --episodes 800 --out $OUT/${sc}__state_only_$m.csv > logs/state_only_${sc}_$m.log 2>&1"
done; done | xargs -d '\n' -P 8 -I{} bash -c '{}'
n=$(ls $OUT | grep -c state_only); echo "files: $n (expect 8)" > logs/phase_state_only.log
if [[ "$n" -ne 8 ]] || grep -lq Traceback logs/state_only_*.log; then echo "ABORT" >> logs/phase_state_only.log; else echo "done" >> logs/phase_state_only.log; fi
