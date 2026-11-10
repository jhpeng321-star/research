#!/bin/bash
# AMENDMENT 56 host chain (detached, launched ONCE; never inside a retry loop; no automatic replay).
#   1. identity: U0 + rand_3a452b1348b01083 on 992000-992003 (8 episodes) vs the stored A42 records; FAILED -> stop
#   2. verify: the 12 random plans x 24 worlds (one dispatch, 32 workers)
#   3. reduce (zero physics): per-world records + UNC admission -> /root/a56/root/A56_REDUCE_D2800_p60.json
# Only /root/a56 is written (the A42 root /root/a42v/root is read). Stop between stages: touch /root/a56/root/STOP_A56.
BASE=/root/a56
source "$BASE/env_a56.sh"
R=$BASE/root
W=32
C=$BASE/chain.out
D=ops/a56_randctrl.py
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" $D "$@" --final-root "$R"; }
[ -f "$C" ] && { echo "chain.out exists: not relaunching"; exit 1; }
log "chain start pid $$ ($(nproc) cores, workers $W)"
echo $$ > "$BASE/chain.pid"
[ -f "$R/STOP_A56" ] && { log "STOP_A56 before identity"; exit 3; }
log "identity start"
run identity --workers 8 --chunk 8 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 4; }
run identity-compare >> "$BASE/identity.out" 2>&1 || { log "IDENTITY COMPARE ERROR"; exit 4; }
log "identity $(tail -1 "$BASE/identity.out")"
grep -q '"verdict": "PASS"' "$R/IDENTITY_A56.json" || { log "IDENTITY FAILED: STOP (random plans not run)"; exit 5; }
[ -f "$R/STOP_A56" ] && { log "STOP_A56 before verify"; exit 3; }
log "verify start"
run verify --workers $W --chunk $((W+1)) >> "$BASE/verify.out" 2>&1 || { log "VERIFY FAILED"; exit 6; }
log "verify done; charged $(python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$R/CHARGES.jsonl')))")"
run reduce >> "$BASE/reduce.out" 2>&1 || { log "REDUCE FAILED"; exit 7; }
log "done: $(tail -1 "$BASE/reduce.out" | cut -c1-300)"
log "A56_ALL_DONE"
