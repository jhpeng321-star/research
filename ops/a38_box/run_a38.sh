#!/bin/bash
# A38 box chain (detached via launch_a38.sh; never inside a retry loop). Stages, each stops the chain on failure:
#  0 tests (cpu)  1 plant identity gate on wz3to2 (U0 x2 + LT x2, sealed p20c J)  2 U0 calibration on wz2to1
#  3 calsummary + registered demand pick  4 evaluation (ARMS on the picked panel)  5 reports + analyse.
# Phase caps are parameters (MAIN registration); STOP_A38 in $ROOT stops between chunks. nice 19 (plant requires it).
source /root/a38/env_a38.sh
W=${WORKERS:-44}; CH=${CHUNK:-88}
ARMS=${ARMS:-U0,HOLD_R2}
CAP_CAL=${CAP_CAL:-31680}; CAP_EVAL=${CAP_EVAL:-258720}
SETTING=${SETTING:-}              # empty = registered pick rule; else the MAIN-registered setting
L=$BASE/chain.out
say() { echo "$(date '+%F %T') $*" >> "$L"; }
say "chain start pid $$ arms=$ARMS workers=$W cap_cal=$CAP_CAL cap_eval=$CAP_EVAL"
nice -n 19 $PY -m pytest -q -p no:cacheprovider ops/test_nm_finaleval.py pytools/tools/coopmwm/test_a38_road2_cpu.py \
  pytools/tools/coopmwm/test_a35_transfer_cpu.py pytools/tools/coopmwm/test_a32_arms_cpu.py > $BASE/tests.out 2>&1 \
  || { say "TESTS FAILED"; exit 1; }
say "tests: $(tail -1 $BASE/tests.out)"
if [ ! -f $BASE/IDENTITY_GATE.json ]; then
  COOPMWM_MAP_SPEC=$MAP3 NM_SETTING=a35_D2400_p20 nice -n 19 $PY ops/a38_road2.py identity --final-root $ROOT --workers 4 \
    --cap $CAP_EVAL > $BASE/identity.out 2>&1 || { say "IDENTITY RUN FAILED"; exit 2; }
  COOPMWM_MAP_SPEC=$MAP3 nice -n 19 $PY ops/a38_road2.py identity-compare --final-root $ROOT --bank-root $BASE/bundle/p20c_ref \
    > $BASE/IDENTITY_GATE.json 2>> $L || { say "IDENTITY COMPARE ERROR"; exit 2; }
fi
grep -q '"verdict": "PASS"' $BASE/IDENTITY_GATE.json || { say "IDENTITY GATE FAILED (no wz2to1 physics)"; exit 2; }
say "identity PASS"
export COOPMWM_MAP_SPEC=$MAP2
[ -f $ROOT/STOP_A38 ] && { say "STOP_A38 before calibration"; exit 3; }
nice -n 19 $PY ops/a38_road2.py calibrate --final-root $ROOT --workers $W --chunk $CH --cap $CAP_CAL >> $BASE/cal.out 2>&1 \
  || { say "CALIBRATION FAILED"; exit 4; }
nice -n 19 $PY ops/a38_road2.py calsummary --final-root $ROOT --out $BASE/cal > $BASE/CALSUMMARY.json 2>> $L \
  || { say "CALSUMMARY FAILED"; exit 4; }
if [ -z "$SETTING" ]; then
  SETTING=$($PY $BASE/bundle/scripts/pick_demand.py $BASE/CALSUMMARY.json) || { say "CAL: NO_CELL - MAIN decides"; exit 5; }
fi
say "eval setting $SETTING"
[ -f $ROOT/STOP_A38 ] && { say "STOP_A38 before evaluation"; exit 3; }
nice -n 19 $PY ops/a38_road2.py run --setting $SETTING --arms $ARMS --final-root $ROOT --workers $W --chunk $CH \
  --cap $CAP_EVAL >> $BASE/run.out 2>&1 || { say "RUN FAILED"; exit 6; }
nice -n 19 $PY ops/a38_road2.py reports --setting $SETTING --final-root $ROOT >> $BASE/reports.out 2>&1 || say "reports FAILED"
nice -n 19 $PY ops/a38_road2.py analyse --setting $SETTING --arms $ARMS --final-root $ROOT --out $BASE/analysis \
  >> $BASE/analyse.out 2>&1 || say "analyse FAILED"
say "A38_ALL_DONE"
