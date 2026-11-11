#!/bin/bash
# AMENDMENT 63 host chain (detached, launched ONCE; never inside a retry loop; no automatic replay).
#   1. identity: BIG_BANK filter OFF on one stored A42 world (1 episode) vs the stored A42 record; FAILED -> stop
#   2. smoke: world index 0 of each of the four arms (4 episodes) + outcome-blind receipt check; FAILED -> stop
#   3. run: the 4 x 96 waves (smoke chunk reused)
#   4. analyse (zero physics): -> /root/a63/root/analysis/STRICTBANK.json + STRICTBANK_PER_WORLD.csv
# Only /root/a63 is written (the A42 root /root/a42v/root is read). Box shared: spare cores only, nice 19, workers
# chosen from the load at launch (total load kept <= ~44). Stop between stages: touch /root/a63/root/STOP_A63.
BASE=/root/a63
source "$BASE/env_a63.sh"
R=$BASE/root
C=$BASE/chain.out
D=ops/a63_strictbank.py
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" $D "$@" --final-root "$R"; }
[ -f "$C" ] && { echo "chain.out exists: not relaunching"; exit 1; }
echo $$ > "$BASE/chain.pid"
log "chain start pid $$"
L=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
W=$((44 - L - 12))
[ "$W" -gt 20 ] && W=20
[ "$W" -lt 8 ] && W=8
log "workers $W; load $(cut -d' ' -f1-3 /proc/loadavg)"
[ -f "$R/STOP_A63" ] && { log "STOP_A63 before identity"; exit 3; }
log "identity start"
run identity --workers 1 --chunk 1 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 4; }
run identity-compare >> "$BASE/identity.out" 2>&1 || { log "IDENTITY COMPARE ERROR"; exit 4; }
log "identity $(tail -1 "$BASE/identity.out" | cut -c1-400)"
grep -q '"verdict": "PASS"' "$R/IDENTITY_A63.json" || { log "IDENTITY FAILED: STOP (A63 arms not run)"; exit 5; }
[ -f "$R/STOP_A63" ] && { log "STOP_A63 before smoke"; exit 3; }
log "smoke start"
run smoke --workers 4 --chunk 4 >> "$BASE/smoke.out" 2>&1 || { log "SMOKE RUN FAILED"; exit 6; }
run smoke-check >> "$BASE/smoke.out" 2>&1 || { log "SMOKE CHECK ERROR"; exit 6; }
log "smoke $(tail -1 "$BASE/smoke.out" | cut -c1-1200)"
grep -q '"verdict": "PASS"' "$R/SMOKE_A63.json" || { log "SMOKE FAILED: STOP"; exit 7; }
[ -f "$R/STOP_A63" ] && { log "STOP_A63 before run"; exit 3; }
log "run start (workers $W; load $(cut -d' ' -f1-3 /proc/loadavg))"
run run --workers $W --chunk $W >> "$BASE/run.out" 2>&1 || { log "RUN FAILED"; exit 8; }
log "run done; charged $(python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$R/CHARGES.jsonl')))")"
run analyse --out "$R/analysis" >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
log "done: $(tail -1 "$BASE/analyse.out" | cut -c1-400)"
log "A63_ALL_DONE"
