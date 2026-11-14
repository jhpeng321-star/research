#!/bin/bash
# WM-S2 gpuA chain (heterogeneous-driver data): multi-stage smoke gate ->
# EXPLICIT-BUDGET curriculum 10/30/60/120/300 s.
#
# One WM-S2 model trains on the WM-S2-HETDATA TRAIN pack with the integrated
# WM-S2 feature set ON:
#   --rollout-mode driver_params   (per-HDV IDM/MOBIL parameter encoder)
#   --traffic-loss                 (traffloss section x 30 s-bin traffic loss)
#   --entries-v2                   (observed-history entrant padding slots)
#   --projection-enabled           (feasibility projection, wms/stab)
#   --residual-cap-mps2/--lane-logit-cap (bounded/contractive residual)
#   --stability-weight/--conservation-weight (regularisers)
#   --loss-share-logging           (run-3 per-term gradient-norm shares)
#
# Stage 1 is a MULTI-STAGE smoke gate over EVERY curriculum stage with every loss
# term active.  It is checked by BOTH:
#   * `wms.gate_stages` (run-3): all six terms present, finite, no term >100x the
#     position term, no >10x growth between stages, plus `loss_share.jsonl` per
#     stage; and
#   * `wms.gate_s2` (WM-S2): every stage logged exactly SMOKE_STEPS records and
#     the driver-parameter encoder outputs stay finite and inside their bounds.
# Either failure refuses to start the full run.
#
# The full run uses an EXPLICIT optimizer-step budget (`--stage-steps`) whose
# 10/30/60/120 s stages MATCH C1 (2500/2500/1875/1250) exactly.  The smoke is
# used only to PROJECT the wall-clock hours (`wms.plan_steps --stage-steps
# ... --max-hours`); the chain REFUSES to start above MAX_HOURS (default 14 h,
# because it shares the GPU with other trainings) and never silently shrinks the
# steps.
#
# The WM-S trainer is TRAIN-only (it refuses any pack whose manifest split is not
# TRAIN).  The VAL/TEST packs are therefore HELD-OUT EVALUATION inputs only; this
# chain only asserts their artefacts exist and prints their paths (scoring is done
# offline by wms_eval / driver recovery, never here).  No SUMO, no acceptance.
#
# Usage: bash s2_chain.sh   (expects $W/code_s2.tar -> extracted $W/code)
set -u
W=${W:-/root/autodl-tmp/wms_s2_20261006/work}
HP3=${HP3:-/root/autodl-tmp}                          # parent of hetpack3_*
HP3T=${HP3T:-$HP3/hetpack3_TRAIN}
HP3V=${HP3V:-$HP3/hetpack3_VAL}
HP3X=${HP3X:-$HP3/hetpack3_TEST}
TRAIN_PACK=${TRAIN_PACK:-$HP3T/pack_hetdata_train}
TRAIN_ID=${TRAIN_ID:-$HP3T/identities_hetdata_train}
TRAIN_WIN=${TRAIN_WIN:-$HP3T/windows_train_hetdata.json}
DRV_TRAIN=${DRV_TRAIN:-$HP3T/driver_params_hetdata_train}
VAL_PACK=${VAL_PACK:-$HP3V/pack_hetdata_val}
VAL_WIN=${VAL_WIN:-$HP3V/windows_val_hetdata.json}
DRV_VAL=${DRV_VAL:-$HP3V/driver_params_hetdata_val}
TEST_PACK=${TEST_PACK:-$HP3X/pack_hetdata_test}
TEST_WIN=${TEST_WIN:-$HP3X/windows_test_hetdata.json}
DRV_TEST=${DRV_TEST:-$HP3X/driver_params_hetdata_test}
PY=${PY:-/root/autodl-tmp/gpu1/env/rd_gpu/bin/python}
NUM_WORKERS=${NUM_WORKERS:-5}
OMP=${OMP:-2}
STAGES=${STAGES:-10,30,60,120,300}
# Explicit optimizer-step budget.  The 10/30/60/120 s stages MATCH C1
# (2500,2500,1875,1250) exactly; the new 300 s stage gets a modest 400.
STAGE_STEPS=${STAGE_STEPS:-2500,2500,1875,1250,400}
MAX_HOURS=${MAX_HOURS:-14}      # projected-hours refusal cap (GPU is shared)
# Multi-stage smoke: a few optimizer steps per stage, every stage covered.
SMOKE_STEPS=${SMOKE_STEPS:-6}
SIDECAR=${SIDECAR:-}            # optional driver-parameter truth sidecar (eval only)
VAL_HOLDOUT_FRAC=${VAL_HOLDOUT_FRAC:-0.1}   # ~10% of TRAIN worlds held out (monitoring)
VAL_HOLDOUT_SEED=${VAL_HOLDOUT_SEED:-0}
if [ "${VAL_HOLDOUT_FRAC}" != "0" ]; then
  VAL_HOLDOUT_FLAGS="--val-holdout-frac $VAL_HOLDOUT_FRAC --val-holdout-seed $VAL_HOLDOUT_SEED"
else
  VAL_HOLDOUT_FLAGS=""
fi
S2_FLAGS="--rollout-mode driver_params --traffic-loss --traffic-weight 0.1 \
--traffic-bin-s 30 --traffic-ramp-start-s 30 --traffic-ramp-end-s 120 \
--entries-v2 --entry-pad 8 \
--projection-enabled --projection-min-gap-m 2.0 \
--residual-cap-mps2 1.5 --lane-logit-cap 1.0 \
--stability-weight 0.1 --stability-margin 1.0 --stability-probes 1 \
--conservation-weight 0.1 --loss-share-logging"
export PATH=/root/autodl-tmp/gpu1/env/rd_gpu/bin:$PATH
log(){ echo "$(date '+%F %T') $*" >> $W/chain.log; }
if [ ! -d $W/code ] && [ -f $W/code_s2.tar ]; then mkdir -p $W/code && tar -xf $W/code_s2.tar -C $W/code; fi
C=$W/code
[ -d $C/output/wd01/wm_n1/wms ] || { log "no code at $C - abort"; exit 2; }

# ---- inputs: WM-S2-HETDATA hetpack3 artefacts --------------------------------
for p in "$TRAIN_PACK" "$TRAIN_ID" "$TRAIN_WIN" "$TRAIN_ID/TRAIN_IDENTITY_RECEIPT.json"; do
  [ -e "$p" ] || { log "missing TRAIN input $p - abort"; exit 2; }
done
# Held-out inputs are read by the offline evaluator, not the trainer.
for p in "$DRV_TRAIN" "$VAL_PACK" "$VAL_WIN" "$DRV_VAL" "$TEST_PACK" "$TEST_WIN" "$DRV_TEST"; do
  [ -e "$p" ] || log "WARN held-out eval input absent: $p"
done
export OMP_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP
cd $C/output/wd01/wm_n1

run(){  # $1 = out dir, rest = extra flags
  out=$1; shift
  $PY -m wms.cli train \
    --pack "$TRAIN_PACK" --identities "$TRAIN_ID" \
    --window-list "$TRAIN_WIN" \
    --out "$out" --device cuda --threads $OMP --num-workers $NUM_WORKERS --batch-size 8 \
    --sumo-fidelity --start-shift-prob 0.75 --truncation-steps 20 \
    ${SIDECAR:+--driver-sidecar $SIDECAR} $VAL_HOLDOUT_FLAGS $S2_FLAGS "$@"
}

IFS=',' read -ra ST_ARR <<< "$STAGES"
UNROLL_FLAGS=""
for s in "${ST_ARR[@]}"; do UNROLL_FLAGS="$UNROLL_FLAGS --unroll-s $s"; done
SS_SMOKE=$(printf "%0.s${SMOKE_STEPS}," $(seq 1 "${#ST_ARR[@]}")); SS_SMOKE=${SS_SMOKE%,}

# ---- Stage 1: multi-stage smoke gate (every stage, every loss term active) ----
log "multi-stage smoke start (code $C; stages $STAGES; $SMOKE_STEPS steps/stage)"
run $W/smoke_stages $UNROLL_FLAGS --stage-steps "$SS_SMOKE" > $W/smoke_stages.log 2>&1; RC=$?
$PY -m wms.gate_stages $W/smoke_stages/train_log.jsonl --stages "$STAGES" \
    --require-terms position,speed,lane,traffic,stability,conservation \
    > $W/smoke_stages_check.txt 2>&1; GC=$?
$PY -m wms.gate_s2 $W/smoke_stages/train_log.jsonl --stages "$STAGES" \
    --steps-per-stage $SMOKE_STEPS > $W/smoke_stages_s2.txt 2>&1; SC=$?
SHARES=$( [ -f $W/smoke_stages/loss_share.jsonl ] && wc -l < $W/smoke_stages/loss_share.jsonl | tr -d ' ' || echo 0 )
log "smoke rc=$RC gate_stages_rc=$GC gate_s2_rc=$SC loss_share_stages=$SHARES"
if [ $RC -ne 0 ] || [ $GC -ne 0 ] || [ $SC -ne 0 ] \
   || ! grep -q '"status": "passed"' $W/smoke_stages_check.txt \
   || [ "$SHARES" -lt "${#ST_ARR[@]}" ]; then
  log "MULTI-STAGE GATE FAILED - full run NOT started"
  exit 1
fi

# ---- Project hours for the EXPLICIT budget; REFUSE above MAX_HOURS -----------
# The smoke measures the real per-stage step time; `project` multiplies it by the
# explicit STAGE_STEPS.  We refuse to start above the cap rather than shrink.
$PY -m wms.plan_steps $W/smoke_stages/train_log.jsonl \
    --stages "$STAGES" --stage-steps "$STAGE_STEPS" --max-hours "$MAX_HOURS" \
    > $W/projected_hours.txt 2> $W/projected_hours.err
PRC=$?
EST=$($PY -c "import json;print(round(json.load(open('$W/smoke_stages/plan_steps.json'))['projected_hours'],3))" 2>/dev/null || echo unknown)
if [ $PRC -ne 0 ]; then
  log "GATE PASSED but projected full run ${EST} h exceeds ${MAX_HOURS} h cap (rc=$PRC) - full run NOT started"
  cat $W/projected_hours.err >> $W/chain.log
  exit 3
fi
log "GATE PASSED; explicit stage steps $STAGE_STEPS (projected ${EST} h, cap ${MAX_HOURS} h)"

# ---- Stage 2: single full curriculum run -------------------------------------
run $W/full $UNROLL_FLAGS --stage-steps "$STAGE_STEPS" \
    > $W/full_run.log 2>&1
log "full run finished rc=$?: $(tail -1 $W/full_run.log | cut -c1-200)"
