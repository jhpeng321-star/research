#!/bin/bash
# AMENDMENT 41 host chain (detached; survives SSH drops): [wait for a live A37 chain on this box to end] ->
# identity gate (FAIL -> stop, MAIN decides) -> stage 1 SELECTION run (U0 + S1..S5 + N2..N4, 48 worlds; no DUO_HOLD)
# -> reports -> sealed SELECTION_A41.json (STOP -> chain ends, no stage 2) -> stage 2 CONFIRMATION run (U0, DUO_HOLD,
# LEARNED_TRIGGER, SEL; 192 worlds; refused without the receipt) -> reports -> analyse.
# Ledger phase d3200bound_a41 (own root, cap 1,560,000) enforced by ops/a41_d3200bound.py.
# Stop between dispatches: touch /root/a41db/root/STOP_A41. Never relaunch inside a retry loop.
source /root/a41db/env_a41.sh
R=$BASE/root
W=$(cat "$BASE/WORKERS")
D=ops/a41_d3200bound.py
C=$BASE/chain.out
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" $D "$@" --final-root "$R"; }
stopped() { [ -f "$R/STOP_A41" ] && { log "STOP_A41 seen before $1"; exit 3; }; }
[ -f "$BASE/SETUP_OK" ] || { log "no SETUP_OK"; exit 1; }
echo $$ > "$BASE/chain.pid"
log "chain start pid $$ workers $W ($(nproc --all) cores)"

# 0. never share the CPUs with a live A37 chain (queue behind it)
while p=$(cat /root/a37ct/chain.pid 2>/dev/null) && [ -n "$p" ] && kill -0 "$p" 2>/dev/null; do
  log "waiting: A37 chain pid $p alive"; sleep 300; stopped "start"
done

# 1. plant identity gate (U0 on p20c 993100/1 and a35 D3200 998100/1; 4,800 sim-s)
if [ ! -f "$R/IDENTITY_GATE.json" ]; then
  run identity --workers 4 --chunk 4 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 1; }
  "$PY" $D identity-compare --final-root "$R" >> "$BASE/identity.out" 2>&1
fi
if ! grep -q '"verdict": "PASS"' "$R/IDENTITY_GATE.json"; then log "IDENTITY GATE FAILED: stop, MAIN decides"; exit 2; fi
log "identity PASS"

# 2. stage 1 SELECTION (U0 + 8 candidates x 48 worlds = 518,400 sim-s)
stopped "stage 1"
if [ ! -f "$R/SELECTION_A41.json" ]; then
  run select-run --workers "$W" --chunk $((2*W+1)) >> "$BASE/stage1.out" 2>&1 || { log "STAGE 1 FAILED"; exit 4; }
  run reports --stage 1 >> "$BASE/reports.out" 2>&1 || { log "reports stage 1 FAILED"; exit 4; }
  run select >> "$BASE/select.out" 2>&1 || { log "SELECT FAILED"; exit 5; }
fi
log "selection: $(tail -1 "$BASE/select.out")"
if grep -q '"decision": "STOP"' "$R/SELECTION_A41.json"; then log "A41_STOP_NO_BANK_INSIDE_THE_BOUND (no stage 2)"; exit 0; fi

# 3. stage 2 CONFIRMATION (U0, DUO_HOLD, LEARNED_TRIGGER, SEL x 192 worlds = 921,600 sim-s)
stopped "stage 2"
run confirm-run --workers "$W" --chunk $((2*W+1)) >> "$BASE/stage2.out" 2>&1 || { log "STAGE 2 FAILED"; exit 6; }
run reports --stage 2 >> "$BASE/reports.out" 2>&1 || log "reports stage 2 FAILED"
run analyse --out "$BASE/analysis" >> "$BASE/analyse.out" 2>&1 || log "analyse FAILED"
log "analysis: $(tail -1 "$BASE/analyse.out")"
log "charged $(python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$R/CHARGES.jsonl')))" 2>/dev/null)"
log "A41_ALL_DONE"
