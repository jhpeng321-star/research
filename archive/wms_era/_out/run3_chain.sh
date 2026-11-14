#!/bin/bash
# WM-S RUN 3 gpuA chain: MULTI-STAGE smoke gate -> full 10/30/60/120/300 s curriculum.
#
# Run 3 = run 2 (branch wms/run2b, commit 6ee1d8a) + the reviewed traffic-loss
# fix (branches wms/trafloss-fix 9e130ec and this branch).  Differences from
# _out/run2_chain.sh:
#
#  * The single stage-10 smoke400 gate is REPLACED by a short MULTI-STAGE smoke
#    that passes through EVERY curriculum stage with every loss term active
#    (position, speed, lane, traffic, stability, conservation), checked by
#    ``wms.gate_stages``: fails closed on any non-finite/missing term, on a term
#    >100x the position term, or on a >10x growth between consecutive stages.
#    It logs per-term magnitudes and, via ``--loss-share-logging``, each term's
#    share of the total gradient norm per stage (``loss_share.jsonl``).
#  * ``--conservation-weight`` is ON so the conservation LOSS term is exercised
#    (the gate requires all six terms); the conservation metric stays ON too.
#
# TRAIN only: no CAL / CAL_SPLIT_A4 / HOLDOUT / chunk-5 / J4B / INC1 / sealed
# bytes are read.  No state-noise injection.  Modelled on _out/run2_chain.sh.
#
# Usage: bash run3_chain.sh   (expects $W/code = extracted _out/code_run3_<sha>.tar,
#        or $W/code_run3.tar to extract, plus $W/lpo window lists and sidecars)
set -u
W=${W:-/root/autodl-tmp/wms_run3_20261007/work}
LPO=${LPO:-$W/lpo}
SC_SRC=/root/autodl-tmp/wmn1_a11_20261004T172721Z/work/sidecars
D9=/root/autodl-tmp/wmn1_a9_data
D10=/root/autodl-tmp/wmn1_a10_data
BASE=/root/autodl-tmp/wmn1_m1_45c9032_20261003_A1/work/pack
PY=${PY:-/root/autodl-tmp/gpu1/env/rd_gpu/bin/python}
OMP=${OMP:-4}
NUM_WORKERS=${NUM_WORKERS:-10}
STAGES="10 30 60 120 300"
STAGE_S_CSV="10,30,60,120,300"
# Explicit optimizer-step budget per stage.  The 10/30/60/120 s stages MATCH C1
# (2500,2500,1875,1250) exactly; the new 300 s stage gets a modest 400.  The
# budget no longer depends on a smoke timed under CPU contention - the smoke is
# only used to PROJECT the wall-clock hours, and the chain refuses to start if
# the projection exceeds MAX_HOURS (it never silently shrinks the steps).
STAGE_STEPS=${STAGE_STEPS:-2500,2500,1875,1250,400}
MAX_HOURS=${MAX_HOURS:-10}           # hard projected-hours cap: refuse above this
# Multi-stage smoke: few optimizer steps per stage, every stage covered.
SMOKE_STAGE_STEPS=${SMOKE_STAGE_STEPS:-6,6,6,6,6}
# Run-2 feature flags (all default OFF in code; the chain turns them ON):
ENTRY_FLAGS=${ENTRY_FLAGS:---entries-v2 --entry-pad 8}
MERGE_FLAGS=${MERGE_FLAGS:---merge-cal}
TRAFFIC_FLAGS=${TRAFFIC_FLAGS:---traffic-loss}   # trafloss-fix branch CLI (B1)
NOISE_FLAGS=${NOISE_FLAGS:-}                       # B4 noise injection OFF (default 0.0)
PROJECTION_FLAGS=${PROJECTION_FLAGS:---projection --projection-min-gap-m 2.0}
RESIDUAL_FLAGS=${RESIDUAL_FLAGS:---residual-cap-mps2 1.5 --lane-logit-cap 3.0}
CONTRACTIVE_FLAGS=${CONTRACTIVE_FLAGS:---stability-weight 0.01 --stability-margin 1.0 --stability-probes 1}
# Conservation metric AND loss ON (run 3 exercises every loss term).
CONSERVATION_FLAGS=${CONSERVATION_FLAGS:---conservation-metric --conservation-weight 0.01}
# Run 3: per-stage gradient-norm share logging (diagnostic-only forward).
SHARE_FLAGS=${SHARE_FLAGS:---loss-share-logging}
STABILITY_FLAGS="$PROJECTION_FLAGS $RESIDUAL_FLAGS $CONTRACTIVE_FLAGS $CONSERVATION_FLAGS"
export PATH=/root/autodl-tmp/gpu1/env/rd_gpu/bin:$PATH
mkdir -p $W/sidecars
log(){ echo "$(date '+%F %T') $*" >> $W/chain.log; }
if [ ! -d $W/code ] && [ -f $W/code_run3.tar ]; then mkdir -p $W/code && tar -xf $W/code_run3.tar -C $W/code; fi
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

# ---- Flag-presence guard: fail closed if the merged branch lacks a run-3 flag -
HELP=$($PY -m wms.cli train --help 2>&1)
for f in --merge-cal --entries-v2 --merge-cal-params --projection \
         --residual-cap-mps2 --stability-weight --conservation-metric \
         --conservation-weight --loss-share-logging; do
  echo "$HELP" | grep -q -- "$f" || { log "run-3 flag $f absent from merged CLI - abort"; echo "MISSING FLAG $f" >&2; exit 4; }
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

# ---- Stage 1: MULTI-STAGE smoke through every curriculum stage --------------
log "multi-stage smoke start (code $C, stage_steps $SMOKE_STAGE_STEPS)"
SMOKE_UNROLL=""
for s in $STAGES; do SMOKE_UNROLL="$SMOKE_UNROLL --unroll-s $s"; done
run $W/smoke_stages $SMOKE_UNROLL --stage-steps $SMOKE_STAGE_STEPS $SHARE_FLAGS \
    > $W/smoke_stages.log 2>&1; RC=$?
$PY -m wms.gate_stages $W/smoke_stages/train_log.jsonl --stages $STAGE_S_CSV \
    --require-terms position,speed,lane,traffic,stability,conservation \
    > $W/smoke_stages_check.txt 2>&1; GC=$?
log "smoke rc=$RC gate_rc=$GC; $(tr '\n' ' ' < $W/smoke_stages_check.txt | cut -c1-600)"
SHARES=$( [ -f $W/smoke_stages/loss_share.jsonl ] && wc -l < $W/smoke_stages/loss_share.jsonl | tr -d ' ' || echo 0 )
log "smoke loss_share stages=$SHARES"
if [ $RC -ne 0 ] || [ $GC -ne 0 ] || ! grep -q '"status": "passed"' $W/smoke_stages_check.txt \
   || [ "$SHARES" -lt 5 ]; then
  log "MULTI-STAGE GATE FAILED - full run NOT started"; echo "GATE FAILED" >&2; exit 1
fi

# ---- Projected hours for the EXPLICIT stage budget, hard MAX_HOURS cap --------
# The smoke measures the real per-stage step time; ``project`` multiplies it by
# the explicit STAGE_STEPS.  We refuse to start above the cap rather than shrink.
$PY -m wms.plan_steps $W/smoke_stages/train_log.jsonl \
    --stages $STAGE_S_CSV --stage-steps $STAGE_STEPS --max-hours $MAX_HOURS \
    > $W/projected_hours.txt 2> $W/projected_hours.err
PRC=$?
EST=$($PY -c "import json;print(round(json.load(open('$W/smoke_stages/plan_steps.json'))['projected_hours'],3))" 2>/dev/null || echo unknown)
if [ $PRC -ne 0 ]; then
  log "GATE PASSED but projected full run ${EST} h exceeds ${MAX_HOURS} h cap (rc=$PRC) - full run NOT started"
  cat $W/projected_hours.err >> $W/chain.log
  echo "PROJECTION OVER CAP" >&2; exit 3
fi
log "GATE PASSED; explicit stage steps $STAGE_STEPS (projected ${EST} h, cap ${MAX_HOURS} h)"

# ---- Stage 2: full run -------------------------------------------------------
FULL_ARGS=""
for s in $STAGES; do FULL_ARGS="$FULL_ARGS --unroll-s $s"; done
run $W/full $FULL_ARGS --val-window-list $LPO/windows_val.json --stage-steps $STAGE_STEPS \
  > $W/full_run.log 2>&1
log "full run finished rc=$?: $(tail -1 $W/full_run.log | cut -c1-200)"
