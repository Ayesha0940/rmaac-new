#!/usr/bin/env bash
# Reduced-observation screen (decentralized protocol): train 4 new models, run 5 levels x 20 cells, report.
# Env: MODEL_DIR (models), PARTS (results/decentral_reduced/parts), OUT_ROOT (noise_sweeps/decentralized_reduced), EPOCHS (200), EPISODES (800), JOBS (100)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="" SUPPRESS_MA_PROMPT=1 PYTHONUNBUFFERED=1
PY=/home/axs0940/miniconda3/envs/rmaac_env/bin/python
MODEL_DIR="${MODEL_DIR:-$PWD/models}"; PARTS="${PARTS:-$PWD/results/decentral_reduced/parts}"
OUT_ROOT="${OUT_ROOT:-$(cd ../../.. && pwd)/noise_sweeps/decentralized_reduced}"
EPOCHS="${EPOCHS:-200}"; EPISODES="${EPISODES:-800}"; JOBS="${JOBS:-100}"
L=logs/phase_decentral_reduced.log; mkdir -p "$PARTS" logs
{
  echo "$PY -u train_recon.py --scenario simple_push --info full --role good --method den --schedule cosine --level goal-hidden --seed 0 --epochs $EPOCHS --model-dir $MODEL_DIR > logs/train_red_push_den.log 2>&1"
  echo "$PY -u train_recon.py --scenario simple_push --info full --role good --method bc --schedule linear --level goal-hidden --seed 0 --epochs $EPOCHS --model-dir $MODEL_DIR > logs/train_red_push_bc.log 2>&1"
  echo "$PY -u train_recon.py --scenario simple_spread --info full --role shared --method den --schedule cosine --level others-removed --seed 0 --epochs $EPOCHS --model-dir $MODEL_DIR > logs/train_red_spread_den.log 2>&1"
  echo "$PY -u train_recon.py --scenario simple_spread --info full --role shared --method bc --schedule linear --level others-removed --seed 0 --epochs $EPOCHS --model-dir $MODEL_DIR > logs/train_red_spread_bc.log 2>&1"
} | xargs -d '\n' -P 4 -I{} bash -c '{}'
n=$(ls "$MODEL_DIR" | grep -cE "^(den_s1|bc)_(Lgh_simple_push_full_good|Lor_simple_spread_full_shared)_s0\.pt$")
echo "new models: $n (expect 4)" > $L
if [[ "$n" -ne 4 ]] || grep -lq Traceback logs/train_red_*.log; then echo "ABORT: training" >> $L; exit 1; fi
{
for spec in "simple_speaker_listener speaker removed" "simple_speaker_listener listener removed" "simple_adversary good removed" "simple_push good goal-hidden" "simple_spread shared others-removed"; do
  set -- $spec
  for k in 0 1 2 4; do echo "$1 $2 $3 delay $k 0"; done
  for p in 0 0.1 0.2 0.4; do echo "$1 $2 $3 stuck $p 0"; done
  for sd in 0 1 2 3; do echo "$1 $2 $3 gauss $sd 0"; echo "$1 $2 $3 biased $sd -1"; echo "$1 $2 $3 biased $sd 1"; done
done
} | while read -r sc ro lv f fl mu; do
  echo "$PY -u eval_decentral_reduced.py --scenario $sc --role $ro --info-level $lv --fault $f --flevel $fl --mu $mu --seed 0 --episodes $EPISODES --model-dir $MODEL_DIR --out $PARTS/${sc}_${ro}_${lv}_${f}_${fl}_mu${mu}.csv > logs/eval_red_${sc}_${ro}_${lv}_${f}_${fl}_mu${mu}.log 2>&1"
done | xargs -d '\n' -P "$JOBS" -I{} bash -c '{}'
np=$(ls "$PARTS" | wc -l); echo "jobs done: $np parts (expect 100)" >> $L
if [[ "$np" -ne 100 ]] || grep -lq Traceback logs/eval_red_*.log; then echo "ABORT: evaluation" >> $L; exit 1; fi
$PY report_reduced.py --parts "$PARTS" --out-root "$OUT_ROOT" > logs/report_reduced.log 2>&1 && echo "report done" >> $L || echo "ABORT: report failed" >> $L
