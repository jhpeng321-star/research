#!/usr/bin/env bash
# Registered round-12 MID-RUN check (R12_REVIEW C2; R12_REVIEW_PACKET section 7), gpuR CPU, laptop independent.
# Waits for runs/c4_wm_v10_round12/wm_s_stage120.pt (end of the 120 s stage, step 40), then replays about 24 TRAIN
# windows one-step with the FILLED presence for round 9 (legacy encoder) and round 12 stage-120
# (tools/r10b_checks/run_onestep.py --base-cols, <= 8 workers, nice 15) and applies tools/r12_checks/midrun_check.py:
#   STOP if round-12 free-road bias < -0.6 OR round-12 braking bias > round-9 braking bias + 0.3 (signed means).
# On STOP it sends SIGTERM to the round-12 training PID only (gate_r12.pid, verified by its command line), writes
# $CHK/MIDRUN_STOP and does NOT relaunch (the coordinator relaunches a corrected round).  On CONTINUE/UNDECIDED it
# writes $CHK/MIDRUN_CONTINUE.  Verdict files: $CHK/results/MIDRUN_R12.{json,md}.
# Launch: cd $REPO && setsid nohup bash tools/r12_checks/midrun_arm.sh $REPO > $L/chk_r12/midrun_arm.log 2>&1 < /dev/null &
set -u -o pipefail
L=${R12_ROOT:-/root/autodl-tmp/c4/v5}
REPO=${1:?usage: midrun_arm.sh REPO_DIR}
PY=${R12_PYTHON:-/root/miniconda3/bin/python}
CHK=${R12_CHK:-$L/chk_r12}
RUN=${R12_RUN:-$L/runs/c4_wm_v10_round12}
REF_CK=${R12_REF_CK:-$L/runs/c4_wm_v10_round9/wm_s_stage300.pt}
[ -f "$REF_CK" ] || REF_CK=$L/runs/c4_wm_v10_round9/wm_s_best.pt
WORKERS=${WORKERS:-8}
[ "$WORKERS" -gt 8 ] && WORKERS=8
OUT=${MIDRUN_OUT:-/dev/shm/chk_r12_midrun}
mkdir -p "$CHK/results"
echo $$ > "$CHK/midrun_arm.pid"
echo "midrun_arm $$ started $(date '+%F %T'); waiting for $RUN/wm_s_stage120.pt"
while [ ! -f "$RUN/wm_s_stage120.pt" ]; do
  if [ -f "$RUN/RUN_SUMMARY.json" ]; then echo "run finished without a stage-120 checkpoint"; exit 3; fi
  sleep 120
done
sleep 120   # the checkpoint is saved at the stage end; let the write and the stage-end validation start settle
CK=$RUN/wm_s_stage120.pt
echo "stage-120 checkpoint seen $(date '+%F %T') sha $(sha256sum "$CK" | cut -c1-16)"
SNAP=$CHK/midrun_stage120.pt
cp "$CK" "$SNAP" || exit 3
rm -rf "$OUT" && mkdir -p "$OUT"
export PYTHONPATH=$REPO/src:$REPO:$REPO/tools:$REPO/tools/r10b_checks OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO" || exit 2
nice -n 15 $PY tools/r10b_checks/run_onestep.py --repo "$REPO" --config c4_wm_v10round12.toml \
  --var "r9=$REF_CK" --var "r10b=$SNAP" \
  --arm-set r9=model.driver_param_mode=legacy --arm-set r9=model.driver_param_init=checkpoint \
  --arm-set r9=model.residual_accel_reinit=false \
  --replay-present filled --base-cols --roots "$L/data/v5_native" "$L/data/genrand_plain" \
  --workers "$WORKERS" --limit 24 --stride-eps 10 --out "$OUT/os" 2>&1 | grep -v -i warn | tail -5
nice -n 15 $PY tools/r12_checks/midrun_check.py --os-dir "$OUT/os" --out-json "$CHK/results/MIDRUN_R12.json" \
  --out-md "$CHK/results/MIDRUN_R12.md" 2>&1 | grep -v -i warn | tail -3
DECISION=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['decision'])" "$CHK/results/MIDRUN_R12.json" 2>/dev/null || echo MISSING)
echo "mid-run decision: $DECISION $(date '+%F %T')"
if [ "$DECISION" = STOP ]; then
  TPID=$(cat "$L/gate_r12.pid" 2>/dev/null || echo "")
  if [ -n "$TPID" ] && tr '\0' ' ' < "/proc/$TPID/cmdline" 2>/dev/null | grep -q "c4_wm_v10_round12"; then
    kill -TERM "$TPID" && echo "SIGTERM sent to round-12 training PID $TPID $(date '+%F %T')"
  else
    echo "round-12 training PID '$TPID' not found or not the round-12 run; nothing killed"
  fi
  date '+%F %T' > "$CHK/MIDRUN_STOP"
else
  date '+%F %T' > "$CHK/MIDRUN_CONTINUE"
fi
echo "midrun_arm done $(date '+%F %T')"
