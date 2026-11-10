#!/bin/bash
# AMENDMENT 40 host chain (detached; survives SSH drops; never inside a retry loop). Own root /root/a40gv; never touches
# /root/a37ct. Stages (each stops the chain on failure):
#   0  WAIT until the A37 chain is finished (/root/a37ct/chain.out contains A37_ALL_DONE and its chain pid is gone) OR
#      MAIN touched /root/a40gv/A40_GO (explicit launch decision); polls every 5 min, does nothing else meanwhile
#   1  plant identity gate on the sealed wz3to2 p20c worlds 993100/993101 (U0 x2 + LEARNED_TRIGGER x2, a35 path; 4,800
#      sim-s, charged to geomvar_a40_eval); FAIL -> stop (MAIN decides)
#   2  U0 demand calibration on wz3to2v80 (3 demands x 8 TRAIN-type worlds, phase geomvar_a40_cal, cap 31,680)
#   3  calsummary + the registered pick rule (or SETTING=<a40_D..._p20> from MAIN) ; NO_CELL -> stop
#   4  evaluation: U0, LEARNED_TRIGGER, DUO_HOLD on the 96-world panel (phase geomvar_a40_eval, cap 385,440)
#   5  vehicle-log reports + registered analysis -> $BASE/analysis
# run_waves dispatches every wave canary-first. Stop between dispatches: touch /root/a40gv/root/STOP_A40.
source /root/a40gv/env_a40.sh
R=$BASE/root
D=ops/a40_geomvar.py
C=$BASE/chain.out
ARMS=${ARMS:-U0,LEARNED_TRIGGER,DUO_HOLD}
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" $D "$@" --final-root "$R"; }
stopped() { [ -f "$R/STOP_A40" ] && { log "STOP_A40 seen before $1"; exit 3; }; }
[ -f "$BASE/SETUP_OK" ] || { log "no SETUP_OK"; exit 1; }
echo $$ > "$BASE/chain.pid"
log "chain start pid $$ ($(nproc) cores)"

# 0. wait for A37 (or MAIN's explicit go)
a37_done() {
  grep -q A37_ALL_DONE /root/a37ct/chain.out 2>/dev/null || return 1
  p=$(cat /root/a37ct/chain.pid 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && return 1
  return 0
}
until a37_done || [ -f "$BASE/A40_GO" ]; do
  [ -f "$R/STOP_A40" ] && { log "STOP_A40 while waiting"; exit 3; }
  sleep 300
done
if a37_done; then log "A37 finished: $(tail -1 /root/a37ct/chain.out)"; else log "A40_GO present (MAIN launch decision); A37 state: $(tail -1 /root/a37ct/chain.out 2>/dev/null)"; fi
MEM_GB=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
W=$(( $(nproc --all) - 4 )); MW=$(( MEM_GB * 10 / 15 )); [ "$MW" -lt "$W" ] && W=$MW
[ "$W" -ge 4 ] || { log "too few workers ($W; mem_avail_gb $MEM_GB)"; exit 1; }
echo "$W" > "$BASE/WORKERS"
log "workers $W (mem_avail_gb $MEM_GB)"

# 1. plant identity gate (wz3to2 process)
if [ ! -f "$BASE/IDENTITY_GATE.json" ]; then
  COOPMWM_MAP_SPEC=$MAP3 run identity --workers 4 --chunk 4 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 2; }
  COOPMWM_MAP_SPEC=$MAP3 "$PY" $D identity-compare --final-root "$R" --bank-root "$BASE/p20c_ref" > "$BASE/IDENTITY_GATE.json.tmp" 2>> "$C" \
    && mv "$BASE/IDENTITY_GATE.json.tmp" "$BASE/IDENTITY_GATE.json" || { log "IDENTITY COMPARE ERROR"; exit 2; }
fi
grep -q '"verdict": "PASS"' "$BASE/IDENTITY_GATE.json" || { log "IDENTITY GATE FAILED: no wz3to2v80 physics (MAIN decides)"; exit 2; }
log "identity PASS"

# 2. calibration (U0 only, TRAIN-type)
stopped calibration
export COOPMWM_MAP_SPEC=$MAPV
run calibrate --workers "$W" --chunk $((W+1)) >> "$BASE/cal.out" 2>&1 || { log "CALIBRATION FAILED"; exit 4; }
run calsummary --out "$BASE/cal" >> "$BASE/calsummary.out" 2>&1 || { log "CALSUMMARY FAILED"; exit 4; }
log "calsummary $(tr -d '\n ' < "$BASE/cal/A40_CALSUMMARY.json" | grep -o '"summary_sha256":"[0-9a-f]*"')"

# 3. demand pick (registered rule) unless MAIN fixed SETTING
if [ -z "${SETTING:-}" ]; then
  SETTING=$("$PY" $D pick --final-root "$R" --summary "$BASE/cal/A40_CALSUMMARY.json") || { log "CAL: NO_CELL - MAIN decides"; exit 5; }
fi
echo "$SETTING" > "$BASE/EVAL_SETTING"
log "eval setting $SETTING"

# 4. evaluation
stopped evaluation
run run --setting "$SETTING" --arms "$ARMS" --workers "$W" --chunk $((2*W+1)) >> "$BASE/run.out" 2>&1 || { log "RUN FAILED"; exit 6; }

# 5. reports + analysis
run reports --setting "$SETTING" --arms "$ARMS" >> "$BASE/reports.out" 2>&1 || log "reports FAILED"
run analyse --setting "$SETTING" --arms "$ARMS" --out "$BASE/analysis" >> "$BASE/analyse.out" 2>&1 || log "analyse FAILED"
log "charged $("$PY" -c "import json;r=[json.loads(l) for l in open('$R/CHARGES.jsonl')];import collections;c=collections.Counter();[c.update({x.get('phase'):x['sim_s']}) for x in r];print(dict(c))" 2>/dev/null)"
log "A40_ALL_DONE"
