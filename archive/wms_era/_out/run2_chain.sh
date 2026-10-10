#!/bin/bash
# WM-S RUN 2 gpuA chain: smoke400 gate -> full 10/30/60/120/300 s curriculum.
#
# Run-2 is a single structured WM-S training: run-1 defaults + junction
# carry-through (active) + calibrated merge_cal ON + observed-history entries
# (trainer --entries-v2) + traffic-level losses + 300 s curriculum stage + the
# wms/stab long-horizon stability features (feasibility projection, capped /
# contractive residual, conservation metric).  State noise injection is OFF.
# CAL/HOLDOUT/sealed data is never read (TRAIN only).
#
# Modelled on _out/c1_chain.sh (wms/integrate) and E:/rd-wt/ws-p2/_out/wms_chain.sh.
# gpuA, one process, one 48 GB GPU; path layout matches the frozen run-1/C1 data.
#
# Usage: bash run2_chain.sh   (expects $W/code = extracted _out/code_run2_<sha>.tar,
#        or $W/code_run2.tar to extract, plus $W/lpo window lists and sidecars)
set -u
W=${W:-/root/autodl-tmp/wms_run2_20261006/work}
LPO=${LPO:-$W/lpo}
SC_SRC=/root/autodl-tmp/wmn1_a11_20261004T172721Z/work/sidecars
D9=/root/autodl-tmp/wmn1_a9_data
D10=/root/autodl-tmp/wmn1_a10_data
BASE=/root/autodl-tmp/wmn1_m1_45c9032_20261003_A1/work/pack
PY=${PY:-/root/autodl-tmp/gpu1/env/rd_gpu/bin/python}
BUDGET_S=${BUDGET_S:-18000}          # ~5 h of full-run compute (owner plan)
HARD_CAP_S=${HARD_CAP_S:-28800}      # 8 h absolute cap: refuse to start above
OMP=${OMP:-4}
NUM_WORKERS=${NUM_WORKERS:-10}
STAGES="10 30 60 120 300"
STAGE_S_CSV="10,30,60,120,300"
# Run-2 feature flags (all default OFF in code; the full run turns them ON):
ENTRY_FLAGS=${ENTRY_FLAGS:---entries-v2 --entry-pad 8}
MERGE_FLAGS=${MERGE_FLAGS:---merge-cal}
TRAFFIC_FLAGS=${TRAFFIC_FLAGS:---traffic-loss}   # trafloss branch CLI (B1)
NOISE_FLAGS=${NOISE_FLAGS:-}                       # B4 noise injection OFF (default 0.0)
# Run-2 stability flags (stab branch; all default OFF in code):
PROJECTION_FLAGS=${PROJECTION_FLAGS:---projection --projection-min-gap-m 2.0}
RESIDUAL_FLAGS=${RESIDUAL_FLAGS:---residual-cap-mps2 1.5 --lane-logit-cap 3.0}
CONTRACTIVE_FLAGS=${CONTRACTIVE_FLAGS:---stability-weight 0.01 --stability-margin 1.0 --stability-probes 1}
# Conservation metric ON, loss term OFF (stab branch does not recommend the loss).
CONSERVATION_FLAGS=${CONSERVATION_FLAGS:---conservation-metric}
STABILITY_FLAGS="$PROJECTION_FLAGS $RESIDUAL_FLAGS $CONTRACTIVE_FLAGS $CONSERVATION_FLAGS"
export PATH=/root/autodl-tmp/gpu1/env/rd_gpu/bin:$PATH
mkdir -p $W/sidecars
log(){ echo "$(date '+%F %T') $*" >> $W/chain.log; }
if [ ! -d $W/code ] && [ -f $W/code_run2.tar ]; then mkdir -p $W/code && tar -xf $W/code_run2.tar -C $W/code; fi
C=$W/code
[ -d $C/output/wd01/wm_n1/wms ] || { log "no code at $C - abort"; exit 2; }
for d in labels_cov_20261004T072826Z identities identities_dspeed; do
  [ -d $W/sidecars/$d ] || cp -r $SC_SRC/$d $W/sidecars/
done
for f in windows_train_noPI.json windows_val.json pi_excluded_runs.json; do
  [ -f "$LPO/$f" ] || { log "missing LPO window list $LPO/$f - abort"; exit 2; }
done
[ -f $W/merge_params.json ] || cp $C/_out/merge_params.json $W/merge_params.json \
  || { log "missing merge_params.json - abort"; exit 2; }
export WM_N1_TRAIN_OUTCOMES=$W/sidecars/labels_cov_20261004T072826Z
export WM_N1_TRAIN_IDENTITIES=$W/sidecars/identities
export OMP_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP
cd $C/output/wd01/wm_n1

# ---- Flag-presence guard: fail closed if the merged branch lacks a run-2 flag -
HELP=$($PY -m wms.cli train --help 2>&1)
for f in --merge-cal --entries-v2 --merge-cal-params --projection \
         --residual-cap-mps2 --stability-weight --conservation-metric; do
  echo "$HELP" | grep -q -- "$f" || { log "run-2 flag $f absent from merged CLI - abort"; echo "MISSING FLAG $f" >&2; exit 4; }
done
for f in $TRAFFIC_FLAGS; do
  case "$f" in --*) echo "$HELP" | grep -q -- "$f" || { log "traffic flag $f absent - abort"; echo "MISSING FLAG $f" >&2; exit 4; };; esac
done

run(){  # $1 = out dir; rest = extra flags
  out=$1; shift
  $PY -m wms.cli train \
    --pack $BASE --identities $W/sidecars/identities \
    --pack $D9/pack_dspeed --identities $W/sidecars/identities_dspeed --overlay $D9/labels_overlay_dspeed \
    --pack $D10/pack_damping --identities $D10/identities_damping --overlay $D10/labels_overlay_damping \
    --window-list $LPO/windows_train_noPI.json \
    --exclude-window-list $LPO/pi_excluded_runs.json \
    --out "$out" --device cuda --threads $OMP --num-workers $NUM_WORKERS --batch-size 8 \
    --sumo-fidelity --start-shift-prob 0.75 --truncation-steps 20 \
    $ENTRY_FLAGS $MERGE_FLAGS --merge-cal-params $W/merge_params.json $TRAFFIC_FLAGS $NOISE_FLAGS \
    $STABILITY_FLAGS "$@"
}

# ---- Stage 1: smoke400 gate (traffic-loss finite + entrants > 0 + stability) --
log "smoke400 start (code $C)"
run $W/smoke400 --unroll-s 10 --steps 400 > $W/smoke400.log 2>&1; RC=$?
$PY -m wms.gate_run2 $W/smoke400/train_log.jsonl --expected 400 \
    --require-projection --require-conservation > $W/smoke400_check.txt 2>&1; GC=$?
log "smoke400 rc=$RC gate_rc=$GC; $(tr '\n' ' ' < $W/smoke400_check.txt | cut -c1-400)"
if [ $RC -ne 0 ] || [ $GC -ne 0 ] || ! grep -q "records 400" $W/smoke400_check.txt \
   || ! grep -q "nonfinite 0 missing 0" $W/smoke400_check.txt \
   || ! grep -qE "projection terms .+" $W/smoke400_check.txt \
   || ! grep -qE "conservation terms .+" $W/smoke400_check.txt \
   || ! grep -q "run2_gate" $W/smoke400_check.txt; then
  log "GATE FAILED - full run NOT started"; echo "GATE FAILED" >&2; exit 1
fi

# ---- Step budget for the 5-stage curriculum, hard 8 h cap ---------------------
SS=$($PY -m wms.plan_steps $W/smoke400/train_log.jsonl --budget-s $BUDGET_S --stages $STAGE_S_CSV)
EST=$($PY -c "import json;print(json.load(open('$W/smoke400/plan_steps.json'))['estimated_hours'])")
CAP_H=$($PY -c "print(round($HARD_CAP_S/3600.0,2))")
if ! $PY -c "import sys;sys.exit(0 if float('$EST') <= $HARD_CAP_S/3600.0 else 1)"; then
  log "GATE PASSED but estimated full run $EST h > ${CAP_H} h hard cap - full run NOT started"; exit 3
fi
log "GATE PASSED; stage steps $SS (est ${EST} h, hard cap ${CAP_H} h)"

# ---- Stage 2: full run -------------------------------------------------------
FULL_ARGS=""
for s in $STAGES; do FULL_ARGS="$FULL_ARGS --unroll-s $s"; done
run $W/full $FULL_ARGS --val-window-list $LPO/windows_val.json --stage-steps $SS \
  > $W/full_run.log 2>&1
log "full run finished rc=$?: $(tail -1 $W/full_run.log | cut -c1-200)"
