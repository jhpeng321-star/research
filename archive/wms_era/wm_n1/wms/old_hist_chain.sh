#!/bin/bash
# old_hist gpuA chain: C1-structured data/flags PLUS --rollout-mode driver_params.
#
# One history-conditioned WM-S member is trained on the SAME data and windows as
# the C1 structured model (pack_v3 + pack_dspeed + pack_damping; TRAIN windows =
# windows_train_noPI.json minus pi_excluded_runs.json; monitoring =
# windows_val.json) with the ONLY model-flag change being
#   --rollout-mode driver_params   (per-HDV IDM/MOBIL params from pre-t0 history)
# Nothing from run 2 is enabled (no traffic loss, entries, projection, stability),
# so a C1 vs old_hist comparison isolates the history/parameter-encoder effect.
#
# The driver parameters are learned ONLY through the trajectory loss: the old
# data has no per-driver truth sidecar and none is passed or required.
#
# Stage 1a is a MULTI-STAGE smoke gate: every curriculum stage runs with the
# driver-parameter encoder active and `wms.gate_s2 --no-required-terms` fails
# closed unless every stage produced its steps and every recovered parameter
# stays finite and inside its hard bounds.  Stage 1b is the C1 400-step 10 s
# smoke (`wms.check400 --driver-params`) that supplies the per-sim-step timing.
# The full run uses C1's FIXED budget (2500/2500/1875/1250 = 8125 steps); if the
# smoke timing implies more than MAX_HOURS the full run is NOT started.
#
# TRAIN-only trainer; VAL/TEST are offline evaluation inputs only (scored later
# by wms_eval / driver recovery, never here).  No SUMO, no acceptance.
#
# Usage: bash old_hist_chain.sh   (expects $W/code_oldhist.tar -> $W/code)
set -u
W=${W:-/root/autodl-tmp/wms_oldhist_20261007/work}
LPO=${LPO:-$W/lpo}                                    # P5 window lists (upload)
SC_SRC=/root/autodl-tmp/wmn1_a11_20261004T172721Z/work/sidecars
D9=/root/autodl-tmp/wmn1_a9_data
D10=/root/autodl-tmp/wmn1_a10_data
BASE=/root/autodl-tmp/wmn1_m1_45c9032_20261003_A1/work/pack
PY=/root/autodl-tmp/gpu1/env/rd_gpu/bin/python
NUM_WORKERS=${NUM_WORKERS:-5}
OMP=${OMP:-2}
MAX_HOURS=${MAX_HOURS:-14}                            # refuse above this
LANE_HEAD=${LANE_HEAD:-off}                            # OFF preserves old hist chain
STAGES=${STAGES:-10,30,60,120}
STAGE_STEPS=${STAGE_STEPS:-2500,2500,1875,1250}       # C1's exact step budget
SMOKE_STEPS=${SMOKE_STEPS:-4}                         # steps per curriculum stage
export PATH=/root/autodl-tmp/gpu1/env/rd_gpu/bin:$PATH
mkdir -p $W/sidecars
log(){ echo "$(date '+%F %T') $*" >> $W/chain.log; }
if [ ! -d $W/code ] && [ -f $W/code_oldhist.tar ]; then mkdir -p $W/code && tar -xf $W/code_oldhist.tar -C $W/code; fi
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
export OMP_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP
cd $C/output/wd01/wm_n1

run(){  # $1 = out dir, rest = extra flags
  out=$1; shift
  LANE_HEAD_ARGS=()
  if [ "$LANE_HEAD" = history ]; then LANE_HEAD_ARGS=(--lane-head history); fi
  $PY -m wms.cli train \
    --pack $BASE --identities $W/sidecars/identities \
    --pack $D9/pack_dspeed --identities $W/sidecars/identities_dspeed --overlay $D9/labels_overlay_dspeed \
    --pack $D10/pack_damping --identities $D10/identities_damping --overlay $D10/labels_overlay_damping \
    --window-list $LPO/windows_train_noPI.json \
    --exclude-window-list $LPO/pi_excluded_runs.json \
    --out "$out" --device cuda --threads $OMP --num-workers $NUM_WORKERS --batch-size 8 \
    --sumo-fidelity --start-shift-prob 0.75 --truncation-steps 20 --lr 1e-3 \
    --rollout-mode driver_params "${LANE_HEAD_ARGS[@]}" "$@"
}

IFS=',' read -ra ST_ARR <<< "$STAGES"
UNROLL_FLAGS=""
for s in "${ST_ARR[@]}"; do UNROLL_FLAGS="$UNROLL_FLAGS --unroll-s $s"; done
SS_SMOKE=$(printf "%0.s${SMOKE_STEPS}," $(seq 1 "${#ST_ARR[@]}")); SS_SMOKE=${SS_SMOKE%,}

# ---- Stage 1a: multi-stage smoke gate (every stage, driver params gated) -----
log "multi-stage smoke start (code $C; stages $STAGES; $SMOKE_STEPS steps/stage)"
run $W/smoke_stages $UNROLL_FLAGS --stage-steps "$SS_SMOKE" > $W/smoke_stages.log 2>&1; RC=$?
$PY -m wms.gate_s2 $W/smoke_stages/train_log.jsonl --stages "$STAGES" \
    --steps-per-stage $SMOKE_STEPS --no-required-terms > $W/smoke_stages_check.txt 2>&1; GC=$?
log "smoke_stages rc=$RC gate_rc=$GC; $(tr '\n' ' ' < $W/smoke_stages_check.txt | cut -c1-500)"
if [ $RC -ne 0 ] || [ $GC -ne 0 ]; then
  log "MULTI-STAGE GATE FAILED - full run NOT started"
  exit 1
fi

# ---- Stage 1b: C1 400-step 10 s smoke (timing + driver-param bounds) ---------
log "smoke400 start (code $C)"
run $W/smoke400 --unroll-s 10 --steps 400 > $W/smoke400.log 2>&1; RC=$?
$PY -m wms.check400 $W/smoke400/train_log.jsonl --expected 400 --driver-params \
    > $W/smoke400_check.txt 2>&1; GC=$?
log "smoke400 rc=$RC gate_rc=$GC; $(tr '\n' ' ' < $W/smoke400_check.txt | cut -c1-500)"
if [ $RC -ne 0 ] || [ $GC -ne 0 ] \
   || ! grep -q "records 400" $W/smoke400_check.txt \
   || ! grep -q "nonfinite 0 missing 0" $W/smoke400_check.txt; then
  log "GATE FAILED (or encoder out of bounds) - full run NOT started"
  exit 1
fi

# ---- Estimate C1's fixed budget from the smoke timing; refuse > MAX_HOURS ----
$PY -m wms.plan_steps $W/smoke400/train_log.jsonl --budget-s 1 --stages "$STAGES" \
    > $W/plan_steps.txt
EST=$($PY -c "import json;print(round(json.load(open('$W/smoke400/plan_steps.json'))['s_per_sim_step']*290*2500/3600.0,2))")
log "smoke timing s_per_sim_step=$(cat $W/plan_steps.txt); fixed-budget estimate ${EST} h (max ${MAX_HOURS})"
if $PY -c "import sys;sys.exit(0 if float('$EST')<=float('${MAX_HOURS}') else 1)"; then :; else
  log "GATE PASSED but estimated full run $EST h > ${MAX_HOURS} h - NOT started"
  exit 3
fi

# ---- Stage 2: one full run at C1's exact budget ------------------------------
log "full run start (stage-steps $STAGE_STEPS; est ${EST} h)"
run $W/full $UNROLL_FLAGS --stage-steps "$STAGE_STEPS" \
    --val-window-list $LPO/windows_val.json > $W/full_run.log 2>&1
log "full run finished rc=$?: $(tail -1 $W/full_run.log | cut -c1-200)"
