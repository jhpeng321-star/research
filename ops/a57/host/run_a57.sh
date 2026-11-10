#!/bin/bash
# AMENDMENT 57 host chain (detached, launched ONCE; never inside a retry loop; no automatic replay).
#   0. wait until the A56 chain (/root/a56/chain.pid, 32 workers) has ended (registered start condition)
#   1. identity: BIG_BANK filter OFF on one stored A42 world (1 episode) vs the stored A42 record; FAILED -> stop
#   2. smoke: world index 0 of PART_L0/1/2 (3 episodes) + outcome-blind receipt check; FAILED -> stop
#   3. run: the 3 x 96 PART waves (smoke chunk reused)
#   4. analyse (zero physics): -> /root/a57/root/analysis/COOPSYN.json + COOPSYN_PER_WORLD.csv
# Only /root/a57 is written (the A42 root /root/a42v/root is read). Stop between stages: touch /root/a57/root/STOP_A57.
BASE=/root/a57
source "$BASE/env_a57.sh"
R=$BASE/root
C=$BASE/chain.out
D=ops/a57_coopsyn.py
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" $D "$@" --final-root "$R"; }
[ -f "$C" ] && { echo "chain.out exists: not relaunching"; exit 1; }
echo $$ > "$BASE/chain.pid"
log "chain start pid $$"
A56PID=$(cat /root/a56/chain.pid 2>/dev/null)
if [ -n "$A56PID" ]; then
  log "waiting for the A56 chain pid $A56PID"
  while kill -0 "$A56PID" 2>/dev/null; do
    [ -f "$R/STOP_A57" ] && { log "STOP_A57 while waiting"; exit 3; }
    sleep 60
  done
  log "A56 chain ended: $(tail -1 /root/a56/chain.out | cut -c1-200)"
fi
W=24
if pgrep -f "[a]54" > /dev/null; then W=16; fi
log "workers $W; load $(cut -d' ' -f1-3 /proc/loadavg); a54 procs: $(pgrep -fc '[a]54')"
[ -f "$R/STOP_A57" ] && { log "STOP_A57 before identity"; exit 3; }
log "identity start"
run identity --workers 1 --chunk 1 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 4; }
run identity-compare >> "$BASE/identity.out" 2>&1 || { log "IDENTITY COMPARE ERROR"; exit 4; }
log "identity $(tail -1 "$BASE/identity.out" | cut -c1-400)"
grep -q '"verdict": "PASS"' "$R/IDENTITY_A57.json" || { log "IDENTITY FAILED: STOP (PART arms not run)"; exit 5; }
[ -f "$R/STOP_A57" ] && { log "STOP_A57 before smoke"; exit 3; }
log "smoke start"
run smoke --workers 3 --chunk 3 >> "$BASE/smoke.out" 2>&1 || { log "SMOKE RUN FAILED"; exit 6; }
run smoke-check >> "$BASE/smoke.out" 2>&1 || { log "SMOKE CHECK ERROR"; exit 6; }
log "smoke $(tail -1 "$BASE/smoke.out" | cut -c1-600)"
grep -q '"verdict": "PASS"' "$R/SMOKE_A57.json" || { log "SMOKE FAILED: STOP"; exit 7; }
[ -f "$R/STOP_A57" ] && { log "STOP_A57 before run"; exit 3; }
log "run start (workers $W)"
run run --workers $W --chunk $W >> "$BASE/run.out" 2>&1 || { log "RUN FAILED"; exit 8; }
log "run done; charged $(python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$R/CHARGES.jsonl')))")"
run analyse --out "$R/analysis" >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
log "done: $(tail -1 "$BASE/analyse.out" | cut -c1-300)"
log "A57_ALL_DONE"
