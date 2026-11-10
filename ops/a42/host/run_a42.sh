#!/bin/bash
# AMENDMENT 42 host chain (detached; survives SSH drops): identity gate -> verification (D2800_p60, U0 + 6 candidates,
# 24 worlds) -> admission (A37 UNC rule) -> [none admitted: BIG_BANK := U0, stop] -> confirmation (U0, DUO_HOLD,
# LEARNED_TRIGGER, BIG_BANK; 96 fresh worlds) -> reports -> analysis. Ledger phase bigp60_a42 (cap 720,000) enforced by
# ops/a42_bigp60.py. Stop between dispatches: touch $BASE/root/STOP_A42. Never relaunch inside a retry loop.
BASE=${A42_HOST_BASE:-/root/a42v}
source "$BASE/env_a42.sh"
R=$BASE/root
W=$(cat "$BASE/WORKERS")
D=ops/a42_bigp60.py
C=$BASE/chain.out
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" $D "$@" --final-root "$R"; }
stopped() { [ -f "$R/STOP_A42" ] && { log "STOP_A42 seen before $1"; exit 3; }; }
[ -f "$BASE/SETUP_OK" ] || { log "no SETUP_OK"; exit 1; }
log "chain start pid $$ workers $W ($(nproc --all) cores)"
echo $$ > "$BASE/chain.pid"

# 1. plant identity gate (U0 on 2 worlds each of p20c / a35-D3200 vs IDENTITY_REFS f049f546; 4,800 sim-s). As A37:
#    the chain continues on FAILED (every A42 arm runs on this host and is analysed only among themselves).
if [ ! -f "$R/IDENTITY_GATE.json" ]; then
  run identity --setting a42v_D2800_p60 --workers 4 --chunk 4 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 1; }
  "$PY" $D identity-compare --setting a42v_D2800_p60 --final-root "$R" >> "$BASE/identity.out" 2>&1 \
    || { log "IDENTITY COMPARE ERROR"; exit 1; }
fi
log "identity $(grep -o '"verdict": "[A-Z]*"' "$R/IDENTITY_GATE.json") $(tr -d '\n ' < "$R/IDENTITY_GATE.json" | grep -o '"per_panel":{[^}]*}')"

# 2. verification (UNC rule, 24 TRAIN-type worlds, U0 + 6 candidates) and 3. admission
stopped "verify"
log "verify D2800_p60"
run verify --setting a42v_D2800_p60 --workers "$W" --chunk $((W+1)) >> "$BASE/verify.out" 2>&1 || { log "VERIFY FAILED"; exit 4; }
if [ ! -f "$R/ADMISSION_D2800_p60.json" ]; then
  run admit --setting a42v_D2800_p60 >> "$BASE/admit.out" 2>&1 || { log "ADMIT FAILED"; exit 5; }
fi
log "admitted: $(tail -1 "$BASE/admit.out")"
run reports --setting a42v_D2800_p60 >> "$BASE/reports.out" 2>&1 || log "reports verify FAILED"

# 4. confirmation on the fresh 96-world panel (skipped by the driver when BIG_BANK := U0)
stopped "confirm"
log "confirm D2800_p60"
run confirm --setting a42c_D2800_p60 --workers "$W" --chunk $((2*W+1)) >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED"; exit 6; }
run reports --setting a42c_D2800_p60 >> "$BASE/reports.out" 2>&1 || log "reports confirm FAILED"
run analyse --setting a42c_D2800_p60 --out "$BASE/analysis" >> "$BASE/analyse.out" 2>&1 || log "analyse FAILED"
log "done: $(tail -1 "$BASE/analyse.out")"
"$PY" $D status --setting a42c_D2800_p60 --final-root "$R" >> "$BASE/status.out" 2>&1
log "charged $(python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$R/CHARGES.jsonl')))" 2>/dev/null)"
log "A42_ALL_DONE"

# AMENDMENT 45 (same chain, after every A42 stage): diverse verification + the DIVERSE_BANK confirmation arm.
# Search path: $BASE/A45_SEARCH (written by launch_a42.sh) else the A47 host search output.
A45=$SRC/ops/a45/host/run_a45.sh
if [ -f "$A45" ]; then
  S45=$(cat "$BASE/A45_SEARCH" 2>/dev/null || echo /root/a47/search/STRATEGY_SEARCH_D2800_p60.json)
  bash "$A45" "$S45" || log "A45 chain exit $?"
fi
