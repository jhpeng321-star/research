#!/bin/bash
# AMENDMENT 45 host chain, run AFTER the A42 chain in the same final root (called at the end of run_a42.sh; also
# runnable alone): search check (the host search file must reproduce the pinned candidates document) -> prepare ->
# verification of the diverse candidates (same 24 worlds; U0 = the A42 wave, never rerun) -> A45 admission (UNC rule)
# -> [no diverse plan admitted: DIVERSE_BANK not built, stop] -> confirmation arm DIVERSE_BANK (same 96 worlds) ->
# reports -> analysis. Ledger phase divp60_a45 (cap 480,000) enforced by ops/a45_divp60.py. Stop: touch $BASE/root/STOP_A45.
# Usage: bash run_a45.sh [SEARCH_JSON]   (default /root/a47/search/STRATEGY_SEARCH_D2800_p60.json)
BASE=${A42_HOST_BASE:-/root/a42v}
SEARCH=${1:-${A45_SEARCH:-/root/a47/search/STRATEGY_SEARCH_D2800_p60.json}}
source "$BASE/env_a42.sh"
R=$BASE/root
W=$(cat "$BASE/WORKERS")
E=ops/a45_divp60.py
C=$BASE/chain.out
log() { echo "$(date '+%F %T') A45 $*" >> "$C"; }
run() { nice -n 19 "$PY" $E "$@" --final-root "$R"; }
stopped() { [ -f "$R/STOP_A45" ] && { log "STOP_A45 seen before $1"; exit 3; }; }
[ -f "$R/ADMISSION_D2800_p60.json" ] || { log "no A42 ADMISSION: A45 not started"; exit 1; }
log "start pid $$ search $SEARCH"
"$PY" $E check-search --search "$SEARCH" >> "$BASE/a45_check.out" 2>&1 || { log "SEARCH CHECK FAILED"; exit 2; }
run prepare --setting a42v_D2800_p60 >> "$BASE/a45_prepare.out" 2>&1 || { log "PREPARE FAILED"; exit 2; }
"$PY" $E dry --setting a42v_D2800_p60 --final-root "$R" > "$BASE/a45_dry_verify.json" 2>>"$BASE/a45_prepare.out" || { log "DRY FAILED"; exit 2; }

stopped "verify"
log "verify diverse D2800_p60"
run verify --setting a42v_D2800_p60 --workers "$W" --chunk $((W+1)) >> "$BASE/a45_verify.out" 2>&1 || { log "VERIFY FAILED"; exit 4; }
if [ ! -f "$R/ADMISSION_A45_D2800_p60.json" ]; then
  run admit --setting a42v_D2800_p60 >> "$BASE/a45_admit.out" 2>&1 || { log "ADMIT FAILED"; exit 5; }
fi
log "admitted: $(tail -1 "$BASE/a45_admit.out")"
run reports --setting a42v_D2800_p60 >> "$BASE/a45_reports.out" 2>&1 || log "reports verify FAILED"

stopped "confirm"
log "confirm DIVERSE_BANK D2800_p60"
run confirm --setting a42c_D2800_p60 --workers "$W" --chunk $((W+1)) >> "$BASE/a45_confirm.out" 2>&1 || { log "CONFIRM FAILED"; exit 6; }
run reports --setting a42c_D2800_p60 >> "$BASE/a45_reports.out" 2>&1 || log "reports confirm FAILED"
run analyse --setting a42c_D2800_p60 --out "$BASE/analysis" >> "$BASE/a45_analyse.out" 2>&1 || log "analyse FAILED"
log "done: $(tail -1 "$BASE/a45_analyse.out")"
"$PY" $E status --setting a42c_D2800_p60 --final-root "$R" >> "$BASE/a45_status.out" 2>&1
log "A45_ALL_DONE"
