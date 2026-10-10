#!/bin/bash
# WM-S gpuA chain: smoke400 -> gate -> full curriculum run (10 -> 30 -> 60 -> 120 s).
# Usage: bash wms_chain.sh   (expects $W/code = extracted code_wms_<sha>.tar, or $W/code_wms.tar to extract)
set -u
W=${W:-/root/autodl-tmp/wms_20261005/work}
SC_SRC=/root/autodl-tmp/wmn1_a11_20261004T172721Z/work/sidecars
D9=/root/autodl-tmp/wmn1_a9_data
D10=/root/autodl-tmp/wmn1_a10_data
BASE=/root/autodl-tmp/wmn1_m1_45c9032_20261003_A1/work/pack
PY=/root/autodl-tmp/gpu1/env/rd_gpu/bin/python
BUDGET_S=${BUDGET_S:-19800}           # 5.5 h of full-run compute
LPO=${LPO:-$W/lpo}                    # pi_excluded_runs.json + windows_val.json (upload)
STAGES=${STAGES:-10,30,60,120}        # Run 2/B2: e.g. 10,30,60,120,300
EXTRA_FLAGS=${EXTRA_FLAGS:-}          # e.g. "--traffic-loss --state-noise-std-m 0.5"
export PATH=/root/autodl-tmp/gpu1/env/rd_gpu/bin:$PATH
mkdir -p $W/sidecars
log(){ echo "$(date '+%F %T') $*" >> $W/chain.log; }
[ -f $LPO/pi_excluded_runs.json ] && [ -f $LPO/windows_val.json ] || { log "missing LPO lists at $LPO - abort"; exit 2; }
if [ ! -d $W/code ] && [ -f $W/code_wms.tar ]; then mkdir -p $W/code && tar -xf $W/code_wms.tar -C $W/code; fi
C=$W/code
[ -d $C/output/wd01/wm_n1/wms ] || { log "no code at $C - abort"; exit 2; }
for d in labels_cov_20261004T072826Z identities identities_dspeed; do
  [ -d $W/sidecars/$d ] || cp -r $SC_SRC/$d $W/sidecars/
done
export WM_N1_TRAIN_OUTCOMES=$W/sidecars/labels_cov_20261004T072826Z
export WM_N1_TRAIN_IDENTITIES=$W/sidecars/identities
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $C/output/wd01/wm_n1
run(){  # $1 = out dir; rest = extra flags
  out=$1; shift
  $PY -m wms.cli train \
    --pack $BASE --identities $W/sidecars/identities \
    --pack $D9/pack_dspeed --identities $W/sidecars/identities_dspeed --overlay $D9/labels_overlay_dspeed \
    --pack $D10/pack_damping --identities $D10/identities_damping --overlay $D10/labels_overlay_damping \
    --out "$out" --device cuda --threads 4 --num-workers 10 --batch-size 8 --sumo-fidelity \
    --start-shift-prob 0.75 --truncation-steps 20 \
    --exclude-window-list $LPO/pi_excluded_runs.json --window-list $LPO/windows_val.json "$@"
}
log "smoke400 start (code $C)"
run $W/smoke400 --unroll-s 10 --steps 400 > $W/smoke400.log 2>&1; RC=$?
$PY -m wms.check400 $W/smoke400/train_log.jsonl --expected 400 > $W/smoke400_check.txt 2>&1; GC=$?
log "smoke400 rc=$RC gate_rc=$GC; $(tr '\n' ' ' < $W/smoke400_check.txt | cut -c1-500)"
if [ $RC -ne 0 ] || [ $GC -ne 0 ] || ! grep -q "records 400" $W/smoke400_check.txt || ! grep -q "nonfinite 0 missing 0" $W/smoke400_check.txt; then log "GATE FAILED - full run NOT started"; exit 1; fi
UNROLL_FLAGS=""
IFS=',' read -ra ST_ARR <<< "$STAGES"
for s in "${ST_ARR[@]}"; do UNROLL_FLAGS="$UNROLL_FLAGS --unroll-s $s"; done
SS=$($PY -m wms.plan_steps $W/smoke400/train_log.jsonl --budget-s $BUDGET_S --stages "$STAGES")
EST=$($PY -c "import json;print(json.load(open('$W/smoke400/plan_steps.json'))['estimated_hours'])")
if $PY -c "import sys;sys.exit(0 if float('$EST')<=8.0 else 1)"; then :; else log "GATE PASSED but estimated full run $EST h > 8 h at minimum steps - full run NOT started (owner decision)"; exit 3; fi
log "GATE PASSED; stages $STAGES stage steps $SS ($(tr '\n' ' ' < $W/smoke400/plan_steps.json))"
run $W/full $UNROLL_FLAGS --stage-steps $SS $EXTRA_FLAGS > $W/full_run.log 2>&1
log "full run finished rc=$?: $(tail -1 $W/full_run.log | cut -c1-200)"
