#!/bin/bash
# AMENDMENT 60 host chain (detached, launched ONCE; never inside a retry loop; no automatic replay). 60C then 60A:
#   60C: identity (U0 + BIG_BANK on 998600-998603 vs stored A42) -> PASS required -> one-world smoke (RAND_BANK on
#        998600) -> full RAND_BANK on 998600-998695 -> reduce-c
#   60A: identity (U0 + rand_3a452b13 on 992000-992003 vs stored A42) -> PASS required -> one-world smoke (U0 + 18
#        plans on 995200) -> full 19 x 96 -> reduce-a
# Workers per dispatch round = min(--workers-max, A60_LOAD_TARGET - cores busy with other jobs) (execution only).
# Stop between stages: touch /root/a60/STOP_CHAIN. Stop inside a part: touch <root>/STOP_A60A or STOP_A60C.
BASE=/root/a60
source "$BASE/env_a60.sh"
C=$BASE/chain.out
D=ops/a60_wmvalue.py
WMAX=${A60_WORKERS_MAX:-24}
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { local cmd=$1 root=$2; shift 2; nice -n 19 "$PY" $D "$cmd" --final-root "$root" "$@"; }
stopchk() { [ -f "$BASE/STOP_CHAIN" ] && { log "STOP_CHAIN before $1"; exit 3; }; }
[ -f "$C" ] && { echo "chain.out exists: not relaunching"; exit 1; }
log "chain start pid $$ ($(nproc) cores, workers max $WMAX, load target ${A60_LOAD_TARGET:-44})"
echo $$ > "$BASE/chain.pid"
RC=$BASE/rootC; RA=$BASE/rootA
# ---------------- 60C
stopchk "60C identity"
log "60C identity start"
run identity-c "$RC" --workers-max 8 >> "$BASE/c_identity.out" 2>&1 || { log "60C IDENTITY RUN FAILED"; exit 4; }
run compare-c "$RC" >> "$BASE/c_identity.out" 2>&1 || { log "60C IDENTITY COMPARE ERROR"; exit 4; }
log "60C identity $(tail -1 "$BASE/c_identity.out" | cut -c1-200)"
grep -q '"verdict": "PASS"' "$RC/IDENTITY_A60C.json" || { log "60C IDENTITY FAILED: STOP"; exit 5; }
stopchk "60C smoke"
log "60C smoke start"
run run-c "$RC" --workers-max 2 --limit 1 >> "$BASE/c_smoke.out" 2>&1 || { log "60C SMOKE FAILED"; exit 6; }
log "60C smoke done"
stopchk "60C run"
log "60C run start"
run run-c "$RC" --workers-max "$WMAX" >> "$BASE/c_run.out" 2>&1 || { log "60C RUN FAILED"; exit 6; }
log "60C run done; charged $(python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$RC/CHARGES.jsonl')))")"
run reduce-c "$RC" >> "$BASE/c_reduce.out" 2>&1 || { log "60C REDUCE FAILED"; exit 7; }
log "60C done: $(tail -1 "$BASE/c_reduce.out" | cut -c1-200)"
log "A60C_DONE"
# ---------------- 60A
stopchk "60A identity"
log "60A identity start"
run identity-a "$RA" --workers-max 8 >> "$BASE/a_identity.out" 2>&1 || { log "60A IDENTITY RUN FAILED"; exit 14; }
run compare-a "$RA" >> "$BASE/a_identity.out" 2>&1 || { log "60A IDENTITY COMPARE ERROR"; exit 14; }
log "60A identity $(tail -1 "$BASE/a_identity.out" | cut -c1-200)"
grep -q '"verdict": "PASS"' "$RA/IDENTITY_A60A.json" || { log "60A IDENTITY FAILED: STOP"; exit 15; }
stopchk "60A smoke"
log "60A smoke start"
run run-a "$RA" --workers-max 19 --limit 1 >> "$BASE/a_smoke.out" 2>&1 || { log "60A SMOKE FAILED"; exit 16; }
log "60A smoke done"
stopchk "60A run"
log "60A run start"
run run-a "$RA" --workers-max "$WMAX" >> "$BASE/a_run.out" 2>&1 || { log "60A RUN FAILED"; exit 16; }
log "60A run done; charged $(python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$RA/CHARGES.jsonl')))")"
run reduce-a "$RA" >> "$BASE/a_reduce.out" 2>&1 || { log "60A REDUCE FAILED"; exit 17; }
log "60A done: $(tail -1 "$BASE/a_reduce.out" | cut -c1-200)"
log "A60_ALL_DONE"
