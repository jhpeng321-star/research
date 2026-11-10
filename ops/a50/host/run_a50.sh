#!/bin/bash
# AMENDMENT 50 host chain (detached; survives SSH drops) = ops/a42/host/run_a42.sh followed by ops/a45/host/run_a45.sh,
# with the A50 names, in ONE final root $BASE/root (base /root/a50v; never /root/a42v):
#   0. search check (zero physics): the host search file must reproduce the pinned A50 candidates document (top 6 and
#      diverse); refused -> nothing is run
#   big-bank part (ops/a50_fullp60.py, ledger phase bigp60_a50 cap 720,000): identity gate -> verification (U0 + 6
#      candidates, 24 worlds 992800-992823) -> admission (A37 UNC rule) -> [none admitted: BIG_BANK := U0] ->
#      confirmation (U0, DUO_HOLD, LEARNED_TRIGGER, BIG_BANK; 96 worlds 996800-996895) -> reports -> analysis
#   diverse part (ops/a50_divp60.py, ledger phase divp60_a50 cap 480,000), only after every big-bank stage: prepare ->
#      verification of the diverse candidates (same 24 worlds; U0 = A50V_U0, never rerun) -> admission -> [none: no
#      DIVERSE_BANK] -> confirmation arm DIVERSE_BANK (same 96 worlds) -> reports -> analysis
# Stop between dispatches: touch $BASE/root/STOP_A50 (big-bank part) or $BASE/root/STOP_A50DIV (diverse part).
# Usage: bash run_a50.sh [SEARCH_JSON]   (default: the path in $BASE/A50_SEARCH written by launch_a50.sh)
BASE=${A50_HOST_BASE:-/root/a50v}
case "$BASE" in /root/a42v|/root/a42v/*) echo "refusing A50 base $BASE (A42 path)"; exit 1 ;; esac
source "$BASE/env_a50.sh"
R=$BASE/root
W=$(cat "$BASE/WORKERS")
D=ops/a50_fullp60.py
E=ops/a50_divp60.py
C=$BASE/chain.out
SEARCH=${1:-$(cat "$BASE/A50_SEARCH" 2>/dev/null)}
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" $D "$@" --final-root "$R"; }
rund() { nice -n 19 "$PY" $E "$@" --final-root "$R"; }
stopped() { [ -f "$R/$2" ] && { log "$2 seen before $1"; exit 3; }; }
[ -f "$BASE/SETUP_OK" ] || { log "no SETUP_OK"; exit 1; }
log "chain start pid $$ workers $W ($(nproc --all) cores) search $SEARCH"
echo $$ > "$BASE/chain.pid"

# 0. the candidate source (zero physics)
[ -n "$SEARCH" ] || { log "no search path (A50_SEARCH)"; exit 2; }
"$PY" $E check-search --search "$SEARCH" >> "$BASE/check_search.out" 2>&1 || { log "SEARCH CHECK FAILED"; exit 2; }
log "search check $(tail -1 "$BASE/check_search.out")"

# 1. plant identity gate (U0 on 2 worlds each of p20c / a35-D3200 vs IDENTITY_REFS f049f546; 4,800 sim-s). As A42:
#    the chain continues on FAILED (every A50 arm runs on this host and is analysed only among themselves).
if [ ! -f "$R/IDENTITY_GATE.json" ]; then
  run identity --setting a50v_D2800_p60 --workers 4 --chunk 4 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 1; }
  "$PY" $D identity-compare --setting a50v_D2800_p60 --final-root "$R" >> "$BASE/identity.out" 2>&1 \
    || { log "IDENTITY COMPARE ERROR"; exit 1; }
fi
log "identity $(grep -o '"verdict": "[A-Z]*"' "$R/IDENTITY_GATE.json") $(tr -d '\n ' < "$R/IDENTITY_GATE.json" | grep -o '"per_panel":{[^}]*}')"

# 2. verification (UNC rule, 24 TRAIN-type worlds, U0 + 6 candidates) and 3. admission
stopped "verify" STOP_A50
log "verify D2800_p60"
run verify --setting a50v_D2800_p60 --workers "$W" --chunk $((W+1)) >> "$BASE/verify.out" 2>&1 || { log "VERIFY FAILED"; exit 4; }
if [ ! -f "$R/ADMISSION_A50_D2800_p60.json" ]; then
  run admit --setting a50v_D2800_p60 >> "$BASE/admit.out" 2>&1 || { log "ADMIT FAILED"; exit 5; }
fi
log "admitted: $(tail -1 "$BASE/admit.out")"
run reports --setting a50v_D2800_p60 >> "$BASE/reports.out" 2>&1 || log "reports verify FAILED"

# 4. confirmation on the fresh 96-world panel (skipped by the driver when BIG_BANK := U0)
stopped "confirm" STOP_A50
log "confirm D2800_p60"
run confirm --setting a50c_D2800_p60 --workers "$W" --chunk $((2*W+1)) >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED"; exit 6; }
run reports --setting a50c_D2800_p60 >> "$BASE/reports.out" 2>&1 || log "reports confirm FAILED"
run analyse --setting a50c_D2800_p60 --out "$BASE/analysis" >> "$BASE/analyse.out" 2>&1 || log "analyse FAILED"
log "done: $(tail -1 "$BASE/analyse.out")"
"$PY" $D status --setting a50c_D2800_p60 --final-root "$R" >> "$BASE/status.out" 2>&1
log "charged $(python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$R/CHARGES.jsonl')))" 2>/dev/null)"
log "A50_BIG_DONE"

# 5. diverse part (the A45 chain on the A50 root), after every big-bank stage
[ -f "$R/ADMISSION_A50_D2800_p60.json" ] || { log "DIV no A50 ADMISSION: diverse part not started"; exit 1; }
log "DIV start"
rund prepare --setting a50v_D2800_p60 >> "$BASE/div_prepare.out" 2>&1 || { log "DIV PREPARE FAILED"; exit 2; }
"$PY" $E dry --setting a50v_D2800_p60 --final-root "$R" > "$BASE/div_dry_verify.json" 2>>"$BASE/div_prepare.out" || { log "DIV DRY FAILED"; exit 2; }
stopped "div verify" STOP_A50DIV
log "DIV verify diverse D2800_p60"
rund verify --setting a50v_D2800_p60 --workers "$W" --chunk $((W+1)) >> "$BASE/div_verify.out" 2>&1 || { log "DIV VERIFY FAILED"; exit 4; }
if [ ! -f "$R/ADMISSION_A50DIV_D2800_p60.json" ]; then
  rund admit --setting a50v_D2800_p60 >> "$BASE/div_admit.out" 2>&1 || { log "DIV ADMIT FAILED"; exit 5; }
fi
log "DIV admitted: $(tail -1 "$BASE/div_admit.out")"
rund reports --setting a50v_D2800_p60 >> "$BASE/div_reports.out" 2>&1 || log "DIV reports verify FAILED"
stopped "div confirm" STOP_A50DIV
log "DIV confirm DIVERSE_BANK D2800_p60"
rund confirm --setting a50c_D2800_p60 --workers "$W" --chunk $((W+1)) >> "$BASE/div_confirm.out" 2>&1 || { log "DIV CONFIRM FAILED"; exit 6; }
rund reports --setting a50c_D2800_p60 >> "$BASE/div_reports.out" 2>&1 || log "DIV reports confirm FAILED"
rund analyse --setting a50c_D2800_p60 --out "$BASE/analysis" >> "$BASE/div_analyse.out" 2>&1 || log "DIV analyse FAILED"
log "DIV done: $(tail -1 "$BASE/div_analyse.out")"
"$PY" $E status --setting a50c_D2800_p60 --final-root "$R" >> "$BASE/div_status.out" 2>&1
log "charged $(python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$R/CHARGES.jsonl')))" 2>/dev/null)"
log "A50_ALL_DONE"
