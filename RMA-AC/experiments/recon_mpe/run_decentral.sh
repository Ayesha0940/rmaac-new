#!/usr/bin/env bash
# Decentralized (per-agent) denoiser sweep: train simple_push per-agent S1 models, run 84 jobs, assemble CSVs.
# Env: MODEL_DIR (models), PARTS (results/decentral/parts), OUT_ROOT (noise_sweeps/decentralized), EPOCHS (200), EPISODES (800), JOBS (90)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="" SUPPRESS_MA_PROMPT=1 PYTHONUNBUFFERED=1
PY=/home/axs0940/miniconda3/envs/rmaac_env/bin/python
MODEL_DIR="${MODEL_DIR:-$PWD/models}"; PARTS="${PARTS:-$PWD/results/decentral/parts}"
OUT_ROOT="${OUT_ROOT:-$(cd ../../.. && pwd)/noise_sweeps/decentralized}"
EPOCHS="${EPOCHS:-200}"; EPISODES="${EPISODES:-800}"; JOBS="${JOBS:-90}"
L=logs/phase_decentral.log; mkdir -p "$PARTS" logs
$PY -u train_recon.py --scenario simple_push --method den --schedule cosine --seed 0 --epochs "$EPOCHS" --model-dir "$MODEL_DIR" > logs/train_decentral_push.log 2>&1
n=$(ls "$MODEL_DIR" | grep -cE "^den_s1_simple_push_full_(adv|good)_s0\.pt$")
echo "push models: $n (expect 2)" > $L
if [[ "$n" -ne 2 ]] || grep -q Traceback logs/train_decentral_push.log; then echo "ABORT: push training" >> $L; exit 1; fi
{
for sc in simple_adversary simple_push simple_speaker_listener simple_spread; do
  echo "clean 0 0 $sc"
  for k in 0 1 2 4; do echo "delay $k 0 $sc"; done
  for p in 0 0.1 0.2 0.4; do echo "stuck $p 0 $sc"; done
  for sd in 0 1 2 3; do echo "gauss $sd 0 $sc"; echo "biased $sd -1 $sc"; echo "biased $sd 1 $sc"; done
done
} | while read -r f lv mu sc; do
  echo "$PY -u eval_decentral.py --scenario $sc --fault $f --flevel $lv --mu $mu --seed 0 --episodes $EPISODES --model-dir $MODEL_DIR --out $PARTS/${sc}_${f}_${lv}_mu${mu}.csv > logs/eval_decentral_${sc}_${f}_${lv}_mu${mu}.log 2>&1"
done | xargs -d '\n' -P "$JOBS" -I{} bash -c '{}'
np=$(ls "$PARTS" | wc -l); echo "jobs done: $np parts (expect 84)" >> $L
if [[ "$np" -ne 84 ]] || grep -lq Traceback logs/eval_decentral_*.log; then echo "ABORT: evaluation" >> $L; exit 1; fi
$PY assemble_decentral.py --parts "$PARTS" --out-root "$OUT_ROOT" > logs/assemble_decentral.log 2>&1 && echo "assemble done" >> $L || echo "ABORT: assemble failed" >> $L
