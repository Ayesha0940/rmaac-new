#!/usr/bin/env bash
# Per-agent reconstructor experiment on MPE. Phases run separately:
#   bash run_recon.sh train | screen | headline | scope | merge
# Env: PY (python with torch+tensorflow), JOBS (parallel processes, default 24), EPISODES (default 800).
# Models -> models/, per-job CSV parts -> results/parts/<phase>/, merged -> results/<phase>.csv, logs -> logs/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export PYTHONPATH="${HERE}/../..:${HERE}/..:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="" SUPPRESS_MA_PROMPT=1 PYTHONUNBUFFERED=1
PY="${PY:-/home/axs0940/miniconda3/envs/rmaac_env/bin/python}"
JOBS="${JOBS:-24}"
EPISODES="${EPISODES:-800}"
MODEL_DIR="${MODEL_DIR:-${HERE}/models}"
RESULT_DIR="${RESULT_DIR:-${HERE}/results}"
LOG_DIR="${HERE}/logs"
EPOCHS="${EPOCHS:-200}"
mkdir -p "$MODEL_DIR" "$RESULT_DIR" "$LOG_DIR"

SL=simple_speaker_listener
SP=simple_spread
METHODS="bc c1 c2 den"
export PY HERE MODEL_DIR RESULT_DIR LOG_DIR EPISODES EPOCHS

train_job() {  # scenario infos method seed   (infos: "hidden full" or "hidden" or "full")
    local scen=$1 infos=$2 m=$3 seed=$4
    for info in $infos; do
        $PY -u train_recon.py --scenario "$scen" --info "$info" --method "$m" --seed "$seed" \
            --epochs "$EPOCHS" --model-dir "$MODEL_DIR" > "$LOG_DIR/train_${scen}_${info}_${m}_s${seed}.log" 2>&1
    done
}
eval_job() {   # phase scenario info seed fault level
    local phase=$1 scen=$2 info=$3 seed=$4 fault=$5 level=$6
    local out="$RESULT_DIR/parts/${phase}/${scen}_${info}_${fault}_${level}_s${seed}.csv"
    mkdir -p "$(dirname "$out")"
    rm -f "$out"      # a part file is written by exactly one job; rerun replaces only its own part
    $PY -u eval_recon.py --phase "$phase" --scenario "$scen" --info "$info" --seed "$seed" \
        --fault "$fault" --levels "$level" --episodes "$EPISODES" --model-dir "$MODEL_DIR" --out "$out" \
        > "$LOG_DIR/eval_${phase}_${scen}_${info}_${fault}_${level}_s${seed}.log" 2>&1
}
eval2_job() {  # scenario info seed level   (screen2: S1 denoiser variants + speaker-history variants)
    local scen=$1 info=$2 seed=$3 level=$4
    local out="$RESULT_DIR/parts/screen2/${scen}_${info}_noise_${level}_s${seed}.csv"
    mkdir -p "$(dirname "$out")"
    rm -f "$out"
    $PY -u eval_recon_s1.py --phase screen2 --scenario "$scen" --info "$info" --seed "$seed" \
        --levels "$level" --episodes "$EPISODES" --model-dir "$MODEL_DIR" --out "$out" \
        > "$LOG_DIR/eval_screen2_${scen}_${info}_noise_${level}_s${seed}.log" 2>&1
}
train2_job() {  # scenario info method schedule hist(0/1) seed
    local scen=$1 info=$2 m=$3 sch=$4 hist=$5 seed=$6
    local extra=""; [[ "$hist" == "1" ]] && extra="--hist"
    $PY -u train_recon.py --scenario "$scen" --info "$info" --method "$m" --schedule "$sch" $extra --seed "$seed" \
        --epochs "$EPOCHS" --model-dir "$MODEL_DIR" > "$LOG_DIR/train2_${scen}_${info}_${m}_${sch}_h${hist}_s${seed}.log" 2>&1
}
pa_job() {  # scenario fault level      (per-agent denoiser stage; seed 0)
    local scen=$1 fault=$2 level=$3
    local out="$RESULT_DIR/parts/pa/${scen}_${fault}_${level}_s0.csv"
    mkdir -p "$(dirname "$out")"
    rm -f "$out"
    $PY -u eval_pa.py --phase pa --scenario "$scen" --fault "$fault" --level "$level" --seed 0 \
        --episodes "$EPISODES" --model-dir "$MODEL_DIR" --out "$out" \
        > "$LOG_DIR/eval_pa_${scen}_${fault}_${level}_s0.log" 2>&1
}
train4_job() {  # scenario role method level      (information-level sweep; S1 cosine for the denoiser)
    local scen=$1 role=$2 m=$3 lv=$4
    local sch=linear; [[ "$m" == "den" ]] && sch=cosine
    $PY -u train_recon.py --scenario "$scen" --info full --role "$role" --method "$m" --schedule "$sch" --level "$lv" \
        --seed 0 --epochs "$EPOCHS" --model-dir "$MODEL_DIR" > "$LOG_DIR/train4_${scen}_${role}_${m}_${lv}.log" 2>&1
}
il_job() {  # scenario varied level fault flevel [--with-bc]
    local scen=$1 varied=$2 lv=$3 fault=$4 fl=$5 extra=${6:-}
    local out="$RESULT_DIR/il/parts/${scen}_${varied}_${lv}_${fault}_${fl}.csv"
    mkdir -p "$(dirname "$out")"
    rm -f "$out"
    $PY -u eval_il.py --phase il --scenario "$scen" --varied "$varied" --info-level "$lv" --fault "$fault" --flevel "$fl" \
        --seed 0 --episodes "$EPISODES" --model-dir "$MODEL_DIR" --out "$out" $extra \
        > "$LOG_DIR/eval_il_${scen}_${varied}_${lv}_${fault}_${fl}.log" 2>&1
}
train5_job() {  # scenario role method level seed      (Step 2: C1 seeds 0-2, denoiser / BC seeds 1-2)
    local scen=$1 role=$2 m=$3 lv=$4 seed=$5
    local sch=cosine; [[ "$m" == "bc" ]] && sch=linear
    $PY -u train_recon.py --scenario "$scen" --info full --role "$role" --method "$m" --schedule "$sch" --level "$lv" \
        --seed "$seed" --epochs "$EPOCHS" --model-dir "$MODEL_DIR" > "$LOG_DIR/train5_${scen}_${role}_${m}_${lv}_s${seed}.log" 2>&1
}
step2_job() {  # scenario varied level fault flevel seed only...
    local scen=$1 varied=$2 lv=$3 fault=$4 fl=$5 seed=$6; shift 6
    local tag=""; [[ $# -gt 0 ]] && tag="_only-$(echo "$*" | tr ' ' '-')"
    local out="$RESULT_DIR/step2/parts/${scen}_${varied}_${lv}_${fault}_${fl}_s${seed}${tag}.csv"
    mkdir -p "$(dirname "$out")"
    rm -f "$out"
    local extra="--step2"; [[ "$fault" == "gauss" && "$fl" == "0" ]] && extra="$extra --with-bc"
    local only=""; [[ $# -gt 0 ]] && only="--only $*"
    $PY -u eval_il.py --phase step2 --scenario "$scen" --varied "$varied" --info-level "$lv" --fault "$fault" --flevel "$fl" \
        --seed "$seed" --episodes "$EPISODES" --model-dir "$MODEL_DIR" --out "$out" $extra $only \
        > "$LOG_DIR/eval_step2_${scen}_${varied}_${lv}_${fault}_${fl}_s${seed}${tag}.log" 2>&1
}
train6_job() {  # scenario role method schedule level seed      (confirmation run)
    local scen=$1 role=$2 m=$3 sch=$4 lv=$5 seed=$6
    $PY -u train_recon.py --scenario "$scen" --info full --role "$role" --method "$m" --schedule "$sch" --level "$lv" \
        --seed "$seed" --epochs "$EPOCHS" --model-dir "$MODEL_DIR" > "$LOG_DIR/train6_${scen}_${role}_${m}_${sch}_${lv}_s${seed}.log" 2>&1
}
confirm_job() {  # kind scenario varied level fault flevel seed      kind: full | c1lin | random
    local kind=$1 scen=$2 varied=$3 lv=$4 fault=$5 fl=$6 seed=$7
    local out="$RESULT_DIR/confirm/parts/${kind}_${scen}_${varied}_${lv}_${fault}_${fl}_s${seed}.csv"
    mkdir -p "$(dirname "$out")"
    rm -f "$out"
    local extra=""
    case "$kind" in
        full) extra="--step2 --confirm"; [[ "$fault" == "gauss" && "$fl" == "0" ]] && extra="$extra --with-bc" ;;
        c1lin) extra="--confirm --only c1lin" ;;
        random) extra="--only clean random" ;;
    esac
    $PY -u eval_il.py --phase confirm --scenario "$scen" --varied "$varied" --info-level "$lv" --fault "$fault" --flevel "$fl" \
        --seed "$seed" --episodes "$EPISODES" --model-dir "$MODEL_DIR" --out "$out" $extra \
        > "$LOG_DIR/eval_confirm_${kind}_${scen}_${varied}_${lv}_${fault}_${fl}_s${seed}.log" 2>&1
}
shift_job() {  # scenario varied level fault flevel seed      (C1-gauss only)
    local scen=$1 varied=$2 lv=$3 fault=$4 fl=$5 seed=$6
    local out="$RESULT_DIR/shift/parts/c1g_${scen}_${varied}_${lv}_${fault}_${fl}_s${seed}.csv"
    mkdir -p "$(dirname "$out")"
    rm -f "$out"
    $PY -u eval_il.py --phase shift --scenario "$scen" --varied "$varied" --info-level "$lv" --fault "$fault" --flevel "$fl" \
        --seed "$seed" --episodes "$EPISODES" --model-dir "$MODEL_DIR" --out "$out" --shift --only c1g \
        > "$LOG_DIR/eval_shift_${scen}_${varied}_${lv}_${fault}_${fl}_s${seed}.log" 2>&1
}
export -f train_job eval_job eval2_job train2_job pa_job train4_job il_job train5_job step2_job train6_job confirm_job shift_job

run_parallel() { xargs -d '\n' -P "$JOBS" -I{} bash -c '{}'; }

phase_train() {
    {
        for m in $METHODS; do
            echo "train_job $SL 'hidden full' $m 0"      # hidden speaker, then full speaker (listener shared)
            echo "train_job $SP full $m 0"
            for s in 1 2; do
                echo "train_job $SL hidden $m $s"
                echo "train_job $SP full $m $s"
            done
        done
    } | run_parallel
}
phase_screen() {   # seed 0, noise std 0 0.4 1 2 3
    {
        for lv in 0 0.4 1 2 3; do
            echo "eval_job screen $SL hidden 0 noise $lv"
            echo "eval_job screen $SL full 0 noise $lv"
            echo "eval_job screen $SP full 0 noise $lv"
        done
    } | run_parallel
}
phase_headline() { # seeds 1, 2; std 0.4 1 2; speaker_listener hidden + spread full
    {
        for s in 1 2; do for lv in 0.4 1 2; do
            echo "eval_job headline $SL hidden $s noise $lv"
            echo "eval_job headline $SP full $s noise $lv"
        done; done
    } | run_parallel
}
phase_scope() {    # seed 0; delay k 3 8, stuck p 0.5 1.0
    {
        for lv in 3 8; do
            echo "eval_job scope $SL hidden 0 delay $lv"
            echo "eval_job scope $SP full 0 delay $lv"
        done
        for lv in 0.5 1.0; do
            echo "eval_job scope $SL hidden 0 stuck $lv"
            echo "eval_job scope $SP full 0 stuck $lv"
        done
    } | run_parallel
}
phase_train2() {   # seed 0: S1 (cosine) denoiser + C1; speaker-history BC/C1/C2/denoiser (speaker_listener hidden)
    {
        for m in den c1; do
            echo "train2_job $SL hidden $m cosine 0 0"
            echo "train2_job $SP full $m cosine 0 0"
        done
        echo "train2_job $SL hidden bc linear 1 0"
        echo "train2_job $SL hidden c1 cosine 1 0"
        echo "train2_job $SL hidden c2 linear 1 0"
        echo "train2_job $SL hidden den cosine 1 0"
    } | run_parallel
}
phase_screen2() {  # seed 0, noise std 0 0.4 1 2 3; speaker_listener hidden + spread full
    {
        for lv in 0 0.4 1 2 3; do
            echo "eval2_job $SL hidden 0 $lv"
            echo "eval2_job $SP full 0 $lv"
        done
    } | run_parallel
}
phase_train3() {   # per-agent denoiser stage: S0/S1 for adversary, S1 for speaker_listener (full) and spread (S0 reused)
    {
        echo "train2_job simple_adversary full den linear 0 0"
        echo "train2_job simple_adversary full den cosine 0 0"
        echo "train2_job $SL full den cosine 0 0"
        echo "train2_job $SP full den cosine 0 0"
    } | run_parallel
}
phase_pa() {
    {
        for scen in $SL simple_adversary $SP; do
            for cell in "gauss 0" "gauss 1" "gauss 3" "biased 1" "delay 3" "delay 8" "stuck 0.5" "stuck 1.0"; do
                echo "pa_job $scen $cell"
            done
        done
    } | run_parallel
}
phase_train4() {   # 17 denoisers + 18 BC for the information-level sweep (full-level models are reused)
    {
        for m in den bc; do
            for lv in removed bottleneck-1 bottleneck-2; do echo "train4_job $SL speaker $m $lv"; done
            for lv in removed bottleneck-1 bottleneck-2 bottleneck-4; do echo "train4_job $SL listener $m $lv"; done
            for lv in removed bottleneck-1 bottleneck-2 bottleneck-4; do echo "train4_job simple_adversary good $m $lv"; done
            for lv in coarse-8 coarse-4 removed bottleneck-1 bottleneck-2 bottleneck-4; do echo "train4_job $SP shared $m $lv"; done
        done
        echo "train4_job simple_adversary good bc full"          # full-level BC for the good agents does not exist yet
    } | run_parallel
}
phase_il() {       # 21 (task, varied, level) combinations x 8 fault cells
    {
        while read -r scen varied levels; do
            for lv in $levels; do
                for cell in "gauss 0" "gauss 1" "gauss 3" "biased 1" "delay 3" "delay 8" "stuck 0.5" "stuck 1.0"; do
                    extra=""; [[ "$cell" == "gauss 0" ]] && extra="--with-bc"
                    echo "il_job $scen $varied $lv $cell $extra"
                done
            done
        done <<LIST
$SL speaker full removed bottleneck-1 bottleneck-2
$SL listener full removed bottleneck-1 bottleneck-2 bottleneck-4
simple_adversary good full removed bottleneck-1 bottleneck-2 bottleneck-4
$SP shared full coarse-8 coarse-4 removed bottleneck-1 bottleneck-2 bottleneck-4
LIST
    } | run_parallel
}
STEP2_CANDIDATES="$SL speaker removed
$SL listener bottleneck-1
simple_adversary good removed
simple_adversary good bottleneck-1
$SP shared coarse-4
$SP shared bottleneck-2
$SP shared bottleneck-4"
phase_train5() {   # 49 models: C1 seeds 0-2, denoiser and BC seeds 1-2, for the 7 candidate levels
    {
        while read -r scen role lv; do
            for s in 0 1 2; do echo "train5_job $scen $role c1 $lv $s"; done
            for s in 1 2; do echo "train5_job $scen $role den $lv $s"; echo "train5_job $scen $role bc $lv $s"; done
        done <<< "$STEP2_CANDIDATES"
    } | run_parallel
}
phase_step2() {    # 7 candidates x 8 faults x 3 seeds; seed 0 only adds bmean + c1 (den/bc/clean/noisy reused from il.csv)
    {
        while read -r scen varied lv; do
            for cell in "gauss 0" "gauss 1" "gauss 3" "biased 1" "delay 3" "delay 8" "stuck 0.5" "stuck 1.0"; do
                echo "step2_job $scen $varied $lv $cell 0 bmean c1"
                for s in 1 2; do echo "step2_job $scen $varied $lv $cell $s"; done
            done
        done <<< "$STEP2_CANDIDATES"
        # regression: one seed-0 cell rerun in full, must equal il.csv exactly
        echo "step2_job $SL speaker removed gauss 1 0 clean noisy den"
    } | run_parallel
}
CONFIRM_LEVELS="$SL speaker removed
$SL listener bottleneck-1
simple_adversary good removed
simple_adversary good bottleneck-1
$SP shared bottleneck-2
$SP shared bottleneck-4"
phase_train6() {   # 66 models: seeds 3-4 for den (S1) / bc / c1 (S1); C1-linear (S0) seeds 0-4
    {
        while read -r scen role lv; do
            for s in 3 4; do
                echo "train6_job $scen $role den cosine $lv $s"
                echo "train6_job $scen $role bc linear $lv $s"
                echo "train6_job $scen $role c1 cosine $lv $s"
            done
            for s in 0 1 2 3 4; do echo "train6_job $scen $role c1 linear $lv $s"; done
        done <<< "$CONFIRM_LEVELS"
    } | run_parallel
}
phase_confirm() {  # seeds 3-4 full method set; seeds 0-2 C1-linear only; random policy seeds 3-4
    {
        while read -r scen varied lv; do
            for cell in "gauss 0" "gauss 1" "gauss 3" "biased 1" "delay 3" "delay 8" "stuck 0.5" "stuck 1.0"; do
                for s in 3 4; do echo "confirm_job full $scen $varied $lv $cell $s"; done
                for s in 0 1 2; do echo "confirm_job c1lin $scen $varied $lv $cell $s"; done
            done
        done <<< "$CONFIRM_LEVELS"
        for spec in "$SL speaker removed" "$SL listener removed" "simple_adversary good removed" "$SP shared removed"; do
            for s in 3 4; do echo "confirm_job random $spec gauss 0 $s"; done
        done
    } | run_parallel
}
SHIFT_LEVELS="simple_adversary good removed
$SP shared bottleneck-2"
phase_train7() {   # C1-gauss (method c2 = Gaussian fault model only), seeds 0-4, 2 levels
    {
        while read -r scen role lv; do
            for s in 0 1 2 3 4; do echo "train6_job $scen $role c2 linear $lv $s"; done
        done <<< "$SHIFT_LEVELS"
    } | run_parallel
}
phase_shift() {
    {
        while read -r scen varied lv; do
            for cell in "gauss 0" "gauss 1" "gauss 3" "biased 1" "delay 3" "delay 8" "stuck 0.5" "stuck 1.0"; do
                for s in 0 1 2 3 4; do echo "shift_job $scen $varied $lv $cell $s"; done
            done
        done <<< "$SHIFT_LEVELS"
    } | run_parallel
}
phase_merge() {
    for ph in screen headline scope screen2 pa; do
        [[ -d "$RESULT_DIR/parts/$ph" ]] || continue
        $PY - "$RESULT_DIR" "$ph" <<'E'
import glob, sys, pandas as pd
rd, ph = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob("{}/parts/{}/*.csv".format(rd, ph)))
pd.concat([pd.read_csv(f) for f in fs]).to_csv("{}/{}.csv".format(rd, ph), index=False)
print(ph, len(fs), "parts merged")
E
    done
}

case "${1:-}" in
    train) phase_train ;;
    screen) phase_screen; phase_merge ;;
    train2) phase_train2 ;;
    screen2) phase_screen2; phase_merge ;;
    train3) phase_train3 ;;
    pa) phase_pa; phase_merge ;;
    train4) phase_train4 ;;
    il) phase_il; $PY merge_il.py "$RESULT_DIR" ;;
    train5) phase_train5 ;;
    step2) phase_step2 ;;
    train6) phase_train6 ;;
    confirm) phase_confirm ;;
    train7) phase_train7 ;;
    shift) phase_shift ;;
    headline) phase_headline; phase_merge ;;
    scope) phase_scope; phase_merge ;;
    merge) phase_merge ;;
    *) echo "usage: $0 train|screen|headline|scope|merge"; exit 1 ;;
esac
