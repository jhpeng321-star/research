#!/bin/bash
# AMENDMENT 61 host chain (detached, launched ONCE; never inside a retry loop; no automatic replay).
#   1. identity: A57 PART_L2 (no pick) on world index 0 with this build vs the stored A57 record; FAILED -> stop
#   2. smoke: world index 0 of PART_L2A / PART_L2B + outcome-blind receipt check; FAILED -> stop
#   3. run: the 2 x 96 waves (smoke chunk reused)   4. analyse (zero physics)
# Box 1 is shared (A59 / A60 / A54): spare cores only. Only /root/a61 is written. Stop: touch /root/a61/root/STOP_A61.
BASE=/root/a61
source "$BASE/env_a61.sh"
R=$BASE/root
C=$BASE/chain.out
D=ops/a61_coopl2split.py
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" $D "$@" --final-root "$R"; }
[ -f "$C" ] && { echo "chain.out exists: not relaunching"; exit 1; }
echo $$ > "$BASE/chain.pid"
L=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
W=16
[ "$L" -ge 24 ] && W=8
log "chain start pid $$; load $(cut -d' ' -f1-3 /proc/loadavg); workers $W; a54 procs $(pgrep -fc '[a]54'); a59 procs $(pgrep -fc '[a]59'); a60 procs $(pgrep -fc '[a]60')"
[ -f "$R/STOP_A61" ] && { log "STOP_A61 before identity"; exit 3; }
run identity --workers 1 --chunk 1 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 4; }
run identity-compare >> "$BASE/identity.out" 2>&1 || { log "IDENTITY COMPARE ERROR"; exit 4; }
log "identity $(tail -1 "$BASE/identity.out" | cut -c1-400)"
grep -q '"verdict": "PASS"' "$R/IDENTITY_A61.json" || { log "IDENTITY FAILED: STOP"; exit 5; }
[ -f "$R/STOP_A61" ] && { log "STOP_A61 before smoke"; exit 3; }
run smoke --workers 2 --chunk 2 >> "$BASE/smoke.out" 2>&1 || { log "SMOKE RUN FAILED"; exit 6; }
run smoke-check >> "$BASE/smoke.out" 2>&1 || { log "SMOKE CHECK ERROR"; exit 6; }
log "smoke $(tail -1 "$BASE/smoke.out" | cut -c1-600)"
grep -q '"verdict": "PASS"' "$R/SMOKE_A61.json" || { log "SMOKE FAILED: STOP"; exit 7; }
[ -f "$R/STOP_A61" ] && { log "STOP_A61 before run"; exit 3; }
log "run start (workers $W; load $(cut -d' ' -f1-3 /proc/loadavg))"
run run --workers $W --chunk $W >> "$BASE/run.out" 2>&1 || { log "RUN FAILED"; exit 8; }
log "run done; charged $(python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$R/CHARGES.jsonl')))")"
run analyse --out "$R/analysis" >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
log "done: $(tail -1 "$BASE/analyse.out" | cut -c1-300)"
log "A61_ALL_DONE"
