#!/bin/bash
# AMENDMENT 73 (S-LCSIDE20) host chain, ONE STAGE per launch (physics stages detached, launched ONCE; no retry loop):
#   run_a73.sh dry | test | mech | admit | confirm
#   dry     : zero SUMO, no episodes: prepare-mech (two small bank files) + dry m; when pinned also prepare + dry v + dry c
#   test    : zero SUMO: the fake-runner tests, temporary files in /dev/shm only
#   mech    : identity (hold plan, A37 path vs lane-change variant, on U0; 4 ep) -> identity-compare (+ decompressed
#             vehicle-log check) -> lcfix (8 ep) -> lc-report            [MAIN reviews both verdicts before admit]
#   admit   : prepare (candidate bank) -> verify (U0 + <= 6 plans, 24 worlds) -> admit
#   confirm : confirm (U0, DUO_HOLD, LEARNED_TRIGGER, LC_BANK_20; 96 worlds) -> analyse
# Only /root/a73 (and /dev/shm for the tests) is written. Physics stages refuse without the pin file and free disk.
STAGE=${1:?stage}
source /root/a73/env_a73.sh
BASE=/root/a73
R=$BASE/root
run() { nice -n 19 "$PY" ops/a73_lcside20.py "$@" --final-root "$R"; }
if [ "$STAGE" = dry ]; then
  run prepare-mech > "$BASE/dry_prepare_mech.out" 2>&1 || { echo "PREPARE-MECH FAILED"; tail -3 "$BASE/dry_prepare_mech.out"; exit 3; }
  run dry --stage m || exit 3
  if [ -f ops/a73/PINS_A73.json ]; then
    run prepare > "$BASE/dry_prepare.out" 2>&1 || { echo "PREPARE FAILED"; tail -3 "$BASE/dry_prepare.out"; exit 3; }
    run dry --stage v || exit 3
    run dry --stage c || exit 3
  else echo "not pinned: dry v / c skipped"; fi
  exit 0
fi
if [ "$STAGE" = test ]; then
  A37_MIRROR=$R exec "$PY" -m pytest -q ops/test_a73_lcside20.py -p no:cacheprovider --basetemp=/dev/shm/a73_test_$$
fi
W=${A73_WORKERS:?set A73_WORKERS}
C=$BASE/chain_$STAGE.out
log() { echo "$(date "+%F %T") $*" >> "$C"; }
case "$STAGE" in mech) NEED=1 ;; admit) NEED=5 ;; confirm) NEED=10 ;; *) echo "unknown stage $STAGE"; exit 2 ;; esac
[ -f "$C" ] && { echo "$C exists: not relaunching"; exit 1; }
[ -f ops/a73/PINS_A73.json ] || { echo "ops/a73/PINS_A73.json missing: not launching"; exit 1; }
FREE=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge "$NEED" ] || { echo "free disk ${FREE}G < ${NEED}G needed by stage $STAGE: not launching"; exit 1; }
log "chain $STAGE start pid $$ workers $W free ${FREE}G"
echo $$ > "$BASE/chain_$STAGE.pid"
case "$STAGE" in
mech)
  O=$BASE/mech.out
  run prepare-mech >> "$O" 2>&1 || { log "PREPARE-MECH FAILED"; exit 4; }
  run identity --workers 4 --chunk 4 >> "$O" 2>&1 || { log "IDENTITY RUN FAILED"; exit 4; }
  run identity-compare >> "$O" 2>&1; log "identity-compare rc $? : $(tail -1 "$O" | cut -c1-900)"
  "$PY" ops/a69/vlcmp.py "$R" a73i_a37 a73i_a44 2 > "$BASE/mech_vlcmp.json" 2>> "$O"; log "vlcmp rc $? $(cut -c1-600 "$BASE/mech_vlcmp.json")"
  run lcfix --workers 8 --chunk 8 >> "$O" 2>&1 || { log "LCFIX RUN FAILED"; exit 5; }
  run lc-report >> "$O" 2>&1; log "lc-report rc $? : $(tail -1 "$O" | cut -c1-900)"
  log "A73_MECH_DONE" ;;
admit)
  run prepare >> "$BASE/verify.out" 2>&1 || { log "PREPARE FAILED"; exit 6; }
  log "verify start"
  run verify --workers "$W" --chunk 96 >> "$BASE/verify.out" 2>&1 || { log "VERIFY FAILED"; exit 6; }
  log "verify done"
  run admit >> "$BASE/admit.out" 2>&1 || { log "ADMIT FAILED"; exit 7; }
  log "admit: $(tail -1 "$BASE/admit.out" | cut -c1-1500)"
  log "A73_ADMIT_DONE" ;;
confirm)
  log "confirm start"
  run confirm --workers "$W" --chunk 96 >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED"; exit 8; }
  log "confirm done"
  run analyse >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-900)"
  log "A73_CONFIRM_DONE" ;;
esac
