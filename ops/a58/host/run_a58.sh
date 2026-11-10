#!/bin/bash
# AMENDMENT 58 host chain (detached, launched ONCE; never inside a retry loop; no automatic replay).
#   1. identity: DUO_HOLD on 996800 vs the stored A50 record (refs/a50); FAILED -> stop (nothing else run)
#   2. illustration: U0 + BIG_BANK on the median-rule A42 world vs the stored A42 records (refs/a42c); FAILED -> stop
#   3. confirm: FAMA (A42 admitted bank, byte copy) on the 96 A50 confirmation worlds (one dispatch)
#   4. reports (zero physics): vehicles_report.json next to every FAMA episode log
# Only /root/a58 is written. Stop between stages: touch /root/a58/root/STOP_A58.
BASE=/root/a58
source "$BASE/env_a58.sh"
R=$BASE/root
REFS=$BASE/refs
W=${A58_WORKERS:-32}
C=$BASE/chain.out
D=ops/a58_famaconf.py
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" $D "$@" --final-root "$R"; }
charged() { python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$R/CHARGES.jsonl')))" 2>/dev/null; }
[ -f "$C" ] && { echo "chain.out exists: not relaunching"; exit 1; }
log "chain start pid $$ ($(nproc) cores, workers $W)"
echo $$ > "$BASE/chain.pid"

[ -f "$R/STOP_A58" ] && { log "STOP_A58 before identity"; exit 3; }
log "identity start"
run identity --workers 1 --chunk 1 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 4; }
run identity-compare --ref-root "$REFS" >> "$BASE/identity.out" 2>&1 || { log "IDENTITY COMPARE ERROR"; exit 4; }
log "identity $(tail -1 "$BASE/identity.out")"
grep -q '"verdict": "PASS"' "$R/IDENTITY_A58.json" || { log "IDENTITY FAILED: STOP (FAMA not run)"; exit 5; }

[ -f "$R/STOP_A58" ] && { log "STOP_A58 before illustration"; exit 3; }
log "illustration start"
run illustrate --ref-root "$REFS" --workers 2 --chunk 2 >> "$BASE/illustrate.out" 2>&1 || { log "ILLUSTRATION RUN FAILED"; exit 6; }
run illustrate-compare --ref-root "$REFS" >> "$BASE/illustrate.out" 2>&1 || { log "ILLUSTRATION COMPARE ERROR"; exit 6; }
log "illustration $(tail -1 "$BASE/illustrate.out")"
grep -q '"verdict": "PASS"' "$R/ILLUSTRATION_A58.json" || { log "ILLUSTRATION FAILED: STOP (FAMA not run)"; exit 7; }

[ -f "$R/STOP_A58" ] && { log "STOP_A58 before confirm"; exit 3; }
log "confirm FAMA start"
run confirm --workers $W --chunk 97 >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED"; exit 8; }
log "confirm done; charged $(charged)"
run reports >> "$BASE/reports.out" 2>&1 || { log "REPORTS FAILED"; exit 9; }
log "reports $(tail -1 "$BASE/reports.out")"
run status >> "$BASE/status.out" 2>&1
log "A58_ALL_DONE"
