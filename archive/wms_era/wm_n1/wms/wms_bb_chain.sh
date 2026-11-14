#!/bin/bash
# C1 (leave-one-controller-out) gpuA chain: train BOTH WM-S variants on the P5
# no-PI windows, smoke400 -> check400 gate -> full 10/30/60/120 s curriculum.
#   structured : existing physics + learned HDV residual, CAVs teacher-forced
#   black-box  : CAVs learned like HDVs + controller-id embedding (no controller code)
# The only training variable is PI removal (--window-list windows_train_noPI.json);
# everything else (packs, losses, curriculum, truncation, check400) is identical.
# Usage: bash wms_bb_chain.sh   (expects $W/code = extracted code tar)
#   LPO=/path/with/windows_train_noPI.json+windows_val.json+pi_excluded_runs.json
set -u
W=${W:-/root/autodl-tmp/wms_bb_20261006/work}
LPO=${LPO:-$W/lpo}
SC_SRC=/root/autodl-tmp/wmn1_a11_20261004T172721Z/work/sidecars
D9=/root/autodl-tmp/wmn1_a9_data
D10=/root/autodl-tmp/wmn1_a10_data
BASE=/root/autodl-tmp/wmn1_m1_45c9032_20261003_A1/work/pack
PY=/root/autodl-tmp/gpu1/env/rd_gpu/bin/python
BUDGET_S=${BUDGET_S:-19800}           # 5.5 h of full-run compute PER model
STAGES=${STAGES:-10,30,60,120}        # Run 2/B2: e.g. 10,30,60,120,300
EXTRA_FLAGS=${EXTRA_FLAGS:-}          # e.g. "--traffic-loss --state-noise-std-m 0.5"
export PATH=/root/autodl-tmp/gpu1/env/rd_gpu/bin:$PATH
mkdir -p $W/sidecars
log(){ echo "$(date '+%F %T') $*" >> $W/chain.log; }
if [ ! -d $W/code ] && [ -f $W/code_wms.tar ]; then mkdir -p $W/code && tar -xf $W/code_wms.tar -C $W/code; fi
C=$W/code
[ -d $C/output/wd01/wm_n1/wms ] || { log "no code at $C - abort"; exit 2; }
for d in labels_cov_20261004T072826Z identities identities_dspeed; do
  [ -d $W/sidecars/$d ] || cp -r $SC_SRC/$d $W/sidecars/
done
for f in windows_train_noPI.json windows_val.json pi_excluded_runs.json; do
  [ -f "$LPO/$f" ] || { log "missing LPO window list $LPO/$f - abort"; exit 2; }
done
export WM_N1_TRAIN_OUTCOMES=$W/sidecars/labels_cov_20261004T072826Z
export WM_N1_TRAIN_IDENTITIES=$W/sidecars/identities
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $C/output/wd01/wm_n1
run(){  # $1 = out dir; $2 = model tag; rest = extra flags
  out=$1; tag=$2; shift 2
  extra=""
  [ "$tag" = "bb" ] && extra="--black-box"
  $PY -m wms.cli train \
    --pack $BASE --identities $W/sidecars/identities \
    --pack $D9/pack_dspeed --identities $W/sidecars/identities_dspeed --overlay $D9/labels_overlay_dspeed \
    --pack $D10/pack_damping --identities $W/sidecars/identities_damping --overlay $D10/labels_overlay_damping \
    --window-list $LPO/windows_train_noPI.json --exclude-window-list $LPO/pi_excluded_runs.json \
    --out "$out" --device cuda --threads 4 --num-workers 10 --batch-size 8 --sumo-fidelity \
    --start-shift-prob 0.75 --truncation-steps 20 $extra "$@"
}
TOTAL_EST=0
for tag in struct bb; do
  log "$tag smoke400 start (code $C)"
  run $W/smoke400_$tag $tag --unroll-s 10 --steps 400 > $W/smoke400_$tag.log 2>&1; RC=$?
  $PY -m wms.check400 $W/smoke400_$tag/train_log.jsonl --expected 400 > $W/smoke400_${tag}_check.txt 2>&1; GC=$?
  log "$tag smoke400 rc=$RC gate_rc=$GC; $(tr '\n' ' ' < $W/smoke400_${tag}_check.txt | cut -c1-300)"
  if [ $RC -ne 0 ] || [ $GC -ne 0 ] || ! grep -q "records 400" $W/smoke400_${tag}_check.txt || ! grep -q "nonfinite 0 missing 0" $W/smoke400_${tag}_check.txt; then
    log "$tag GATE FAILED - full run NOT started"; continue; fi
  UNROLL_FLAGS=""
  IFS=',' read -ra ST_ARR <<< "$STAGES"
  for s in "${ST_ARR[@]}"; do UNROLL_FLAGS="$UNROLL_FLAGS --unroll-s $s"; done
  SS=$($PY -m wms.plan_steps $W/smoke400_$tag/train_log.jsonl --budget-s $BUDGET_S --stages "$STAGES")
  EST=$($PY -c "import json;print(json.load(open('$W/smoke400_$tag/plan_steps.json'))['estimated_hours'])")
  log "$tag GATE PASSED; stages $STAGES stage steps $SS; est $EST h"
  if $PY -c "import sys;sys.exit(0 if float('$EST')<=8.0 else 1)"; then :; else
    log "$tag est $EST h > 8 h at minimum steps - full run NOT started (owner decision)"; continue; fi
  TOTAL_EST=$($PY -c "print(round($TOTAL_EST + $EST, 2))")
  run $W/full_$tag $tag $UNROLL_FLAGS \
    --val-window-list $LPO/windows_val.json --stage-steps $SS $EXTRA_FLAGS > $W/full_${tag}_run.log 2>&1
  log "$tag full finished rc=$?: $(tail -1 $W/full_${tag}_run.log | cut -c1-200)"
done
log "C1 chain done; estimated total GPU hours for both models: $TOTAL_EST"
