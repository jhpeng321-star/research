#!/bin/bash
# AMENDMENT 37 host chain (detached; survives SSH drops): identity gate -> verification (D3200_p20, D2400_p20) ->
# admission -> evaluation (D3200_p20, D2400_p20: U0, LEARNED_TRIGGER, DUO_HOLD, CELL_BANK) -> 37b NATIVE (a33 panel).
# Ledger phase celltailor_a37 (cap 1,900,000; verification <= 600,000) enforced by ops/a37_celltailor.py.
# Stop between dispatches: touch /root/a37ct/root/STOP_A37. Never relaunch inside a retry loop.
source /root/a37ct/env_a37.sh
R=$BASE/root
W=$(cat "$BASE/WORKERS")
D=ops/a37_celltailor.py
C=$BASE/chain.out
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" $D "$@" --final-root "$R"; }
stopped() { [ -f "$R/STOP_A37" ] && { log "STOP_A37 seen before $1"; exit 3; }; }
[ -f "$BASE/SETUP_OK" ] || { log "no SETUP_OK"; exit 1; }
log "chain start pid $$ workers $W ($(nproc) cores)"
echo $$ > "$BASE/chain.pid"

# 1. plant identity gate (U0 on 2 worlds each of p20c / a33 / a35-D3200 / a35-D2400; 9,600 sim-s)
if [ ! -f "$R/IDENTITY_GATE.json" ]; then
  run identity --setting a37v_D3200_p20 --workers 8 --chunk 8 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 1; }
  "$PY" $D identity-compare --setting a37v_D3200_p20 --final-root "$R" >> "$BASE/identity.out" 2>&1 \
    || { log "IDENTITY COMPARE ERROR"; exit 1; }
fi
log "identity $(grep -o '"verdict": "[A-Z]*"' "$R/IDENTITY_GATE.json") $(tr -d '\n ' < "$R/IDENTITY_GATE.json" | grep -o '"per_panel":{[^}]*}')"

# 2. verification (UNC rule, 24 TRAIN worlds per cell, U0 + 9 candidates) and 3. admission
for c in D3200_p20 D2400_p20; do
  stopped "verify $c"
  log "verify $c"
  run verify --setting a37v_$c --workers "$W" --chunk $((W+1)) >> "$BASE/verify.out" 2>&1 || { log "VERIFY $c FAILED"; exit 4; }
  if [ ! -f "$R/ADMISSION_$c.json" ]; then
    run admit --setting a37v_$c >> "$BASE/admit.out" 2>&1 || { log "ADMIT $c FAILED"; exit 5; }
  fi
  log "admitted $c: $(tail -1 "$BASE/admit.out")"
done

# 4. evaluation on the amendment-35 panels
for c in D3200_p20 D2400_p20; do
  stopped "eval $c"
  log "eval $c"
  run eval --setting a35_$c --workers "$W" --chunk $((2*W+1)) >> "$BASE/eval.out" 2>&1 || { log "EVAL $c FAILED"; exit 6; }
  run reports --setting a35_$c >> "$BASE/reports.out" 2>&1 || log "reports $c FAILED"
  run analyse --setting a35_$c --out "$BASE/analysis" >> "$BASE/analyse.out" 2>&1 || log "analyse $c FAILED"
  log "done $c: $(tail -1 "$BASE/analyse.out")"
done

# 5. 37b NATIVE on the a33 fresh panel (context, no pass/fail)
stopped "native"
log "native"
run native --setting a33 --workers "$W" --chunk $((2*W+1)) >> "$BASE/native.out" 2>&1 || { log "NATIVE FAILED"; exit 7; }
run reports --setting a33 >> "$BASE/reports.out" 2>&1 || log "reports a33 FAILED"
run analyse-native --setting a33 --out "$BASE/analysis" >> "$BASE/analyse.out" 2>&1 || log "analyse-native on host skipped (needs the host U0 rerun; laptop pairs with the a33 records)"
for c in D3200_p20 D2400_p20; do run reports --setting a37v_$c >> "$BASE/reports.out" 2>&1 || log "reports v $c FAILED"; done
"$PY" $D status --setting a33 --final-root "$R" >> "$BASE/status.out" 2>&1
log "charged $(python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$R/CHARGES.jsonl')))" 2>/dev/null)"
log "A37_ALL_DONE"
