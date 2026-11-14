#!/bin/bash
# AMENDMENT 81 (S-BIGCONFIRM-A81) host chain, ONE STAGE per launch, each physics stage runs ONCE (a stage whose chain
# file exists refuses); physics stages detached by the caller; no retry loop; no automatic replay.
#   run_a81.sh setup   : zero SUMO install (setup_a81.sh, from the ship dir)
#   run_a81.sh test    : zero SUMO unit tests (temporary files in /dev/shm only)
#   run_a81.sh dry     : zero SUMO: panel-c (write-once) + bank copy + dry i / c (job lists, inputs, stored refs)
#   run_a81.sh mech    : 3 arms x 2 A70 worlds (7,200 sim-s) -> mech-compare (A81_MECH.json PASS/FAIL)
#   run_a81.sh confirm : 3 arms x 384 fresh worlds (1,382,400 sim-s; raw archives dropped per round) -> analyse
#   run_a81.sh analyse : zero SUMO re-run of the registered analysis ONLY if confirm ran and no results file exists
# Only /root/a81big (and /dev/shm for the tests) is written. Stop between dispatch rounds: touch /root/a81big/root/STOP_A81.
# A81_KEEP_RAW=1 keeps the raw archives of confirm (needs ~21 GB per 1,000 episodes; default: dropped).
STAGE=${1:?stage: setup | test | dry | mech | confirm | analyse}
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a81big
if [ "$STAGE" = setup ]; then
  [ -f "$BASE/SETUP_OK" ] && { echo "setup already done: not rerunning"; exit 1; }
  exec bash "$HERE/setup_a81.sh"
fi
source $BASE/env_a81.sh
R=$BASE/root
A70R=/root/a75/a70ref/root
REF70=/root/a75/ref70
A75D2R=/root/a75d2c/root
REFS="--a66-root /root/a66/root --a70-root $A70R --ref70-reports $REF70 --a75d2-root $A75D2R"
cd "$SRC" || exit 2
run() { nice -n 19 "$PY" ops/a81_bigconfirm.py "$@" --final-root "$R" $REFS; }
charged() { python3 -c "import json,os;p='$R/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
[ -f "$BASE/SETUP_OK" ] || { echo "no SETUP_OK: run setup first"; exit 1; }
if [ "$STAGE" = test ]; then
  exec "$PY" -m pytest -q -rs ops/test_a81_bigconfirm.py -p no:cacheprovider --basetemp=/dev/shm/a81_test_$$
fi
if [ "$STAGE" = dry ]; then
  [ -f "$R/A81_PANEL_C.json" ] || run panel-c > "$BASE/dry_panel_c.out" 2>&1 || { echo "PANEL-C FAILED"; tail -3 "$BASE/dry_panel_c.out"; exit 3; }
  run prepare > "$BASE/dry_prepare.out" 2>&1 || { echo "PREPARE FAILED"; tail -3 "$BASE/dry_prepare.out"; exit 3; }
  run dry --stage c > "$BASE/dry_c.out" 2>&1 || { echo "DRY C FAILED"; tail -3 "$BASE/dry_c.out"; exit 3; }
  run dry --stage i > "$BASE/dry_i.out" 2>&1 || { echo "DRY I FAILED"; tail -3 "$BASE/dry_i.out"; exit 3; }
  exit 0
fi
W=${A81_WORKERS:?set A81_WORKERS}
C=$BASE/chain_$STAGE.out
log() { echo "$(date "+%F %T") $*" >> "$C"; }
case "$STAGE" in mech) NEED=2 ;; confirm) NEED=6 ;; analyse) NEED=1 ;; *) echo "unknown stage $STAGE"; exit 2 ;; esac
[ -f "$C" ] && { echo "$C exists: stage $STAGE already launched; not relaunching"; exit 1; }
[ -f "$R/STOP_A81" ] && { echo "STOP_A81 present: not launching"; exit 1; }
FREE=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge "$NEED" ] || { echo "free disk ${FREE}G < ${NEED}G needed by stage $STAGE: not launching"; exit 1; }
DROP="--drop-raw --drop-procs 16"
[ "${A81_KEEP_RAW:-0}" = 1 ] && DROP=""
if [ -z "$DROP" ] && [ "$STAGE" = confirm ] && [ "${FREE:-0}" -lt 30 ]; then echo "A81_KEEP_RAW=1 needs 30G free (have ${FREE}G): not launching"; exit 1; fi
mechpass() { grep -q '"verdict": "PASS"' "$R/A81_MECH.json" 2>/dev/null; }
case "$STAGE" in
mech) [ -f "$R/A81_PANEL_C.json" ] && [ -f "$R/a75d2_banks/A75D2_BANK_D2800_p60.json" ] || { echo "run dry first"; exit 5; } ;;
confirm) mechpass || { echo "mechanics is not PASS: nothing runs"; exit 5; }
         [ -n "${A81_PANEL_SHA256:-}" ] || { echo "A81_PANEL_SHA256 unset (the registered panel sha): not launching"; exit 5; } ;;
analyse) [ -f "$BASE/chain_confirm.out" ] && grep -q A81_CONFIRM_RUN_DONE "$BASE/chain_confirm.out" || { echo "confirm not done"; exit 5; }
         [ -f "$R/A81_RESULTS_D2800_p60.json" ] && { echo "results exist: not rerunning"; exit 5; } ;;
esac
log "chain $STAGE start pid $$ workers $W free ${FREE}G drop '${DROP}' panel ${A81_PANEL_SHA256:-unset}"
echo $$ > "$BASE/chain_$STAGE.pid"
case "$STAGE" in
mech)
  run mech --workers $(( W < 6 ? W : 6 )) --chunk 6 >> "$BASE/mech.out" 2>&1 || { log "MECH RUN FAILED charged $(charged)"; exit 4; }
  run mech-compare >> "$BASE/mech.out" 2>&1
  log "mech $(tail -1 "$BASE/mech.out" | cut -c1-1500) charged $(charged)"
  mechpass || { log "MECHANICS FAILED: STOP"; exit 5; }
  log "A81_MECH_DONE" ;;
confirm)
  log "confirm start"
  run confirm --workers "$W" --chunk 96 $DROP >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED charged $(charged) free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")"; exit 8; }
  if [ -n "$DROP" ]; then run drop-raw --drop-procs 16 >> "$BASE/confirm.out" 2>&1 || { log "DROP-RAW FAILED"; exit 8; }; fi
  log "confirm done; charged $(charged)"
  log "A81_CONFIRM_RUN_DONE free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")"
  run analyse >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-1500)"
  log "A81_CONFIRM_DONE" ;;
analyse)
  run analyse >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-1500)"
  log "A81_ANALYSE_DONE" ;;
esac
