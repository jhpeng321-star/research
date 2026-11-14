#!/bin/bash
# AMENDMENT 69 / A44 mechanics host chain, ONE STAGE per launch (detached, launched ONCE; no retry loop):
#   run_a69.sh mech | ident | admit | confirm
#   mech    : A44 mechanics exactly as registered (driver ops/a44_lcmech.py): prepare, identity (4 ep),
#             identity-compare (+ decompressed vehicle-log check), lcfix (8 ep), lc-report
#   ident   : A69 identity (hold plan, A67 path vs variant, on MAPPO60C; 4 ep) -> identity-compare
#   admit   : prepare (candidate bank) -> verify (<= 6 x 24) -> admit
#   confirm : confirm (LC_BANK_60C, 96 worlds) -> analyse
# Only /root/a69 is written.
STAGE=${1:?stage}
source /root/a69/env_a69.sh
BASE=/root/a69
W=${A69_WORKERS:?set A69_WORKERS}
C=$BASE/chain_$STAGE.out
log() { echo "$(date '+%F %T') $*" >> "$C"; }
[ -f "$C" ] && { echo "$C exists: not relaunching"; exit 1; }
log "chain $STAGE start pid $$ workers $W"
echo $$ > "$BASE/chain_$STAGE.pid"
MR=$BASE/a44m/root
R=$BASE/root
mech() { nice -n 19 "$PY" ops/a44_lcmech.py "$@" --final-root "$MR" --seeds 992400-992407; }
run() { nice -n 19 "$PY" ops/a69_lcside.py "$@" --final-root "$R" --a68-root /root/a68/root --a66-root /root/a66/root; }
case "$STAGE" in
mech)
  O=$BASE/mech.out
  mech prepare >> "$O" 2>&1 || { log "MECH PREPARE FAILED"; exit 4; }
  mech identity --workers 4 --chunk 4 >> "$O" 2>&1 || { log "MECH IDENTITY RUN FAILED"; exit 4; }
  mech identity-compare >> "$O" 2>&1; log "mech identity-compare rc $? : $(tail -1 "$O" | cut -c1-700)"
  "$PY" ops/a69/vlcmp.py "$MR" a44m_ident_a37 a44m_ident_a44 2 > "$BASE/mech_vlcmp.json" 2>> "$O"; log "vlcmp rc $? $(cut -c1-600 "$BASE/mech_vlcmp.json")"
  mech lcfix --n-lc 8 --workers 8 --chunk 8 >> "$O" 2>&1 || { log "MECH LCFIX RUN FAILED"; exit 5; }
  mech lc-report --n-lc 8 >> "$O" 2>&1; log "mech lc-report rc $? : $(tail -1 "$O" | cut -c1-900)"
  log "A69_MECH_DONE" ;;
ident)
  run prepare-ident >> "$BASE/identity.out" 2>&1 || { log "PREPARE FAILED"; exit 4; }
  run identity --workers 4 --chunk 4 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 4; }
  run identity-compare >> "$BASE/identity.out" 2>&1 || { log "IDENTITY COMPARE ERROR"; exit 4; }
  log "identity $(tail -1 "$BASE/identity.out" | cut -c1-900)"
  log "A69_IDENT_DONE" ;;
admit)
  run prepare >> "$BASE/verify.out" 2>&1 || { log "PREPARE FAILED"; exit 6; }
  log "verify start"
  run verify --workers "$W" --chunk 96 >> "$BASE/verify.out" 2>&1 || { log "VERIFY FAILED"; exit 6; }
  log "verify done"
  run admit >> "$BASE/admit.out" 2>&1 || { log "ADMIT FAILED"; exit 7; }
  log "admit: $(tail -1 "$BASE/admit.out" | cut -c1-1500)"
  log "A69_ADMIT_DONE" ;;
confirm)
  log "confirm start"
  run confirm --workers "$W" --chunk 96 >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED"; exit 8; }
  log "confirm done"
  run analyse --out "$R" >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-900)"
  log "A69_CONFIRM_DONE" ;;
*) log "unknown stage $STAGE"; exit 2 ;;
esac
