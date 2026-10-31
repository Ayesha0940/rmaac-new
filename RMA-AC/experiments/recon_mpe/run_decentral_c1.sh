#!/usr/bin/env bash
# C1 (supervised reconstructor, unscaled) under the decentralized protocol: train 11 models -> 80 eval jobs -> assemble.
# Env: MODEL_DIR (models/c1_decentral), PARTS (results/decentral_c1/parts), EPOCHS (200), EPISODES (800), JOBS (80)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="" SUPPRESS_MA_PROMPT=1 PYTHONUNBUFFERED=1
PY=/home/axs0940/miniconda3/envs/rmaac_env/bin/python
MODEL_DIR="${MODEL_DIR:-$PWD/models/c1_decentral}"; PARTS="${PARTS:-$PWD/results/decentral_c1/parts}"
EPOCHS="${EPOCHS:-200}"; EPISODES="${EPISODES:-800}"; JOBS="${JOBS:-80}"
L=logs/phase_decentral_c1.log; mkdir -p "$MODEL_DIR" "$PARTS" logs
TR="$PY -u train_recon.py --method c1 --schedule cosine --seed 0 --epochs $EPOCHS --model-dir $MODEL_DIR"
{
  for sc in simple_speaker_listener simple_adversary simple_push simple_spread; do echo "$TR --scenario $sc --info full > logs/train_c1d_${sc}_full.log 2>&1"; done
  echo "$TR --scenario simple_speaker_listener --info full --role speaker --level removed > logs/train_c1d_sl_red.log 2>&1"
  echo "$TR --scenario simple_adversary --info full --role good --level removed > logs/train_c1d_adv_red.log 2>&1"
  echo "$TR --scenario simple_push --info full --role good --level goal-hidden > logs/train_c1d_push_red.log 2>&1"
  echo "$TR --scenario simple_spread --info full --role shared --level others-removed > logs/train_c1d_spread_red.log 2>&1"
} | xargs -d '\n' -P 8 -I{} bash -c '{}'
n=$(ls "$MODEL_DIR" | grep -c "^c1_s1_.*_s0\.pt$"); echo "models: $n (expect 11)" > $L
if [[ "$n" -ne 11 ]] || grep -lq Traceback logs/train_c1d_*.log; then echo "ABORT: training" >> $L; exit 1; fi
{
for sc in simple_adversary simple_push simple_speaker_listener simple_spread; do
  for k in 0 1 2 4; do echo "$sc delay $k 0"; done
  for p in 0 0.1 0.2 0.4; do echo "$sc stuck $p 0"; done
  for sd in 0 1 2 3; do echo "$sc gauss $sd 0"; echo "$sc biased $sd -1"; echo "$sc biased $sd 1"; done
done
} | while read -r sc f fl mu; do
  echo "$PY -u eval_decentral_c1.py --scenario $sc --fault $f --flevel $fl --mu $mu --seed 0 --episodes $EPISODES --model-dir $MODEL_DIR --out $PARTS/${sc}_${f}_${fl}_mu${mu}.csv > logs/eval_c1d_${sc}_${f}_${fl}_mu${mu}.log 2>&1"
done | xargs -d '\n' -P "$JOBS" -I{} bash -c '{}'
np=$(ls "$PARTS" | wc -l); echo "jobs done: $np parts (expect 80)" >> $L
if [[ "$np" -ne 80 ]] || grep -lq Traceback logs/eval_c1d_*.log; then echo "ABORT: evaluation" >> $L; exit 1; fi
$PY assemble_decentral_c1.py --parts "$PARTS" > logs/assemble_c1d.log 2>&1 && echo "done" >> $L || echo "ABORT: assemble failed" >> $L
