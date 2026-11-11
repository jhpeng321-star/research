#!/bin/bash
# AMENDMENT 62b host chain (detached, launched ONCE; never inside a retry loop; no automatic replay).
#   1. identity: plan p (no filter) as a one-row CELL_BANK on one stored A42 admission world vs the stored A37_PLAN
#      record; FAILED -> stop   2. smoke: world index 0 of the 12 part waves + outcome-blind check; FAILED -> stop
#   3. run: 12 x 24 (smoke chunk reused)   4. analyse (zero physics)
# Box 1 is shared (A60 / A61 / A54): spare cores only. Only /root/a62 is written. Stop: touch /root/a62/root/STOP_A62.
BASE=/root/a62
source "$BASE/env_a62.sh"
R=$BASE/root
C=$BASE/chain.out
D=ops/a62_coopperplan.py
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" $D "$@" --final-root "$R"; }
[ -f "$C" ] && { echo "chain.out exists: not relaunching"; exit 1; }
echo $$ > "$BASE/chain.pid"
L=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
W=12
[ "$L" -ge 24 ] && W=8
log "chain start pid $$; load $(cut -d' ' -f1-3 /proc/loadavg); workers $W; a54 $(pgrep -fc '[a]54'); a60 $(pgrep -fc '[a]60'); a61 $(pgrep -fc '[a]61')"
[ -f "$R/STOP_A62" ] && { log "STOP_A62 before identity"; exit 3; }
run identity --workers 1 --chunk 1 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 4; }
run identity-compare >> "$BASE/identity.out" 2>&1 || { log "IDENTITY COMPARE ERROR"; exit 4; }
log "identity $(tail -1 "$BASE/identity.out" | cut -c1-400)"
grep -q '"verdict": "PASS"' "$R/IDENTITY_A62.json" || { log "IDENTITY FAILED: STOP"; exit 5; }
[ -f "$R/STOP_A62" ] && { log "STOP_A62 before smoke"; exit 3; }
run smoke --workers $W --chunk 12 >> "$BASE/smoke.out" 2>&1 || { log "SMOKE RUN FAILED"; exit 6; }
run smoke-check >> "$BASE/smoke.out" 2>&1 || { log "SMOKE CHECK ERROR"; exit 6; }
log "smoke $(tail -1 "$BASE/smoke.out" | cut -c1-700)"
grep -q '"verdict": "PASS"' "$R/SMOKE_A62.json" || { log "SMOKE FAILED: STOP"; exit 7; }
[ -f "$R/STOP_A62" ] && { log "STOP_A62 before run"; exit 3; }
log "run start (workers $W; load $(cut -d' ' -f1-3 /proc/loadavg))"
run run --workers $W --chunk $W >> "$BASE/run.out" 2>&1 || { log "RUN FAILED"; exit 8; }
log "run done; charged $(python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$R/CHARGES.jsonl')))")"
run analyse --out "$R/analysis" >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
log "done: $(tail -1 "$BASE/analyse.out" | cut -c1-400)"
log "A62_ALL_DONE"
