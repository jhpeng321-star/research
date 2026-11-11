#!/bin/bash
# AMENDMENT 64 host chain (detached, launched ONCE; never inside a retry loop; no automatic replay).
#   0. ALLOCATIONS.jsonl written at chain start (before any episode is charged)
#   1. identity: unchanged A59 HAND3 bytes as one-row CELL_BANK, filter OFF, panel world index 0 (1 episode) vs the
#      stored A59 HAND3 record; FAILED -> stop
#   2. smoke: world index 0 of each of the four arms (4 episodes) + outcome-blind receipt / implied-speed check;
#      FAILED -> stop
#   3. run: the 4 x 96 waves (smoke chunk reused)
#   4. analyse (zero physics): -> /root/a64/root/analysis/HAND3FAST.json + HAND3FAST_PER_WORLD.csv
# Only /root/a64 is written (A42 /root/a42v/root and A59 /root/a59/root are read). Box shared with A60 (priority):
# nice 19, workers = 44 - load - 4 clamped to [8, 16]. Stop between stages: touch /root/a64/root/STOP_A64.
BASE=/root/a64
source "$BASE/env_a64.sh"
R=$BASE/root
C=$BASE/chain.out
D=ops/a64_hand3fast.py
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" $D "$@" --final-root "$R"; }
[ -f "$C" ] && { echo "chain.out exists: not relaunching"; exit 1; }
echo $$ > "$BASE/chain.pid"
log "chain start pid $$"
L=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
W=$((44 - L - 4))
[ "$W" -gt 16 ] && W=16
[ "$W" -lt 8 ] && W=8
log "workers $W; load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "{\"id\": \"A64_CHAIN\", \"amendment\": \"64\", \"phase\": \"hand3fast_a64\", \"started\": \"$(date '+%F %T %z')\", \"host\": \"Box 1 /root/a64\", \"stages\": {\"identity\": 1200, \"smoke_inside_run\": 4800, \"run\": 460800}, \"planned_sim_s\": 462000, \"cap_sim_s\": 480000, \"code\": \"$(cat $BASE/BASE_CODE)\", \"workers\": $W, \"plan_sha256\": {\"F23\": \"69d9f9cce304d472ebedc9ec60999501a3c91e03ee9c8bd4a00b5e586de8842b\", \"F26\": \"f17311747fdc3f81db410a7e6bd756b7d0e797ddf3b8a17c583abb1f2b4695bb\", \"HAND3_source\": \"498a520629fc5305d63f4e9696a4119afab04ab345fba8fa015e33797fa4e90c\"}, \"note\": \"one launch, no replay; charges per episode in host CHARGES.jsonl (written as each episode is charged)\"}" >> "$R/ALLOCATIONS.jsonl"
[ -f "$R/STOP_A64" ] && { log "STOP_A64 before identity"; exit 3; }
log "identity start"
run identity --workers 1 --chunk 1 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 4; }
run identity-compare >> "$BASE/identity.out" 2>&1 || { log "IDENTITY COMPARE ERROR"; exit 4; }
log "identity $(tail -1 "$BASE/identity.out" | cut -c1-500)"
grep -q '"verdict": "PASS"' "$R/IDENTITY_A64.json" || { log "IDENTITY FAILED: STOP (A64 arms not run)"; exit 5; }
[ -f "$R/STOP_A64" ] && { log "STOP_A64 before smoke"; exit 3; }
log "smoke start"
run smoke --workers 4 --chunk 4 >> "$BASE/smoke.out" 2>&1 || { log "SMOKE RUN FAILED"; exit 6; }
run smoke-check >> "$BASE/smoke.out" 2>&1 || { log "SMOKE CHECK ERROR"; exit 6; }
log "smoke $(tail -1 "$BASE/smoke.out" | cut -c1-1500)"
grep -q '"verdict": "PASS"' "$R/SMOKE_A64.json" || { log "SMOKE FAILED: STOP"; exit 7; }
[ -f "$R/STOP_A64" ] && { log "STOP_A64 before run"; exit 3; }
log "run start (workers $W; load $(cut -d' ' -f1-3 /proc/loadavg))"
run run --workers $W --chunk $W >> "$BASE/run.out" 2>&1 || { log "RUN FAILED"; exit 8; }
log "run done; charged $(python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$R/CHARGES.jsonl')))")"
run analyse --out "$R/analysis" >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
log "done: $(tail -1 "$BASE/analyse.out" | cut -c1-600)"
log "A64_ALL_DONE"
