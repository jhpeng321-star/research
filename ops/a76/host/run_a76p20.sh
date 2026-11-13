#!/bin/bash
# AMENDMENT 76 (A76-D2P20) host chain, ONE STAGE per launch, each physics stage runs ONCE (a stage whose chain file
# exists refuses); physics stages detached by the caller; no retry loop; no automatic replay.
#   run_a76p20.sh setup    : zero SUMO install (setup_a76p20.sh from the ship dir)
#   run_a76p20.sh bind     : zero SUMO, ONLY for a comparator other than the frozen U0: seal BOUND_A76_COMP.json
#                            (needs A76_COMP_POLICY, A76_COMP_SHA256, A76_COMP_AUTHORITY)
#   run_a76p20.sh test     : zero SUMO unit tests (temporary files in /dev/shm only)
#   run_a76p20.sh dry      : zero SUMO: panels m / v / c (write-once), banks, A76_PINS.json, dry m / v / c
#   run_a76p20.sh mech     : 4 identity + 16 form episodes (24,000 sim-s) -> mech-compare (A76_MECH.json PASS/FAIL)
#   run_a76p20.sh admit    : comparator + 36 plans x 24 worlds (1,065,600 sim-s; raw archives dropped per round)
#   run_a76p20.sh bank     : zero physics: admission rule + bank rule -> A76_ADMISSION_D2800_p20.json + bank file
#   run_a76p20.sh confirm  : comparator + A76_BANK on 96 worlds (230,400 sim-s) -> analyse
# Every stage needs A76_COMP_SHA256 (and A76_COMP_POLICY for a non-U0 comparator); the root refuses a comparator other
# than the one sealed in A76_COMPARATOR.json. Base dir A76_BASE (default /root/a76p20; one base per comparator).
# Only $A76_BASE (and /dev/shm for the tests) is written. Stop between dispatch rounds: touch $A76_BASE/root/STOP_A76.
# A76_KEEP_RAW=1 keeps the raw archives of admit / confirm (needs ~20 GB per 1,000 episodes; default: dropped).
STAGE=${1:?stage: setup | bind | test | dry | mech | admit | bank | confirm}
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${A76_BASE:-/root/a76p20}
case "$BASE" in /root/a76p20*) ;; *) echo "A76_BASE must be /root/a76p20*"; exit 2;; esac
export A76_BASE=$BASE
if [ "$STAGE" = setup ]; then
  [ -f "$BASE/SETUP_OK" ] && { echo "setup already done: not rerunning"; exit 1; }
  exec bash "$HERE/setup_a76p20.sh"
fi
[ -f "$BASE/SETUP_OK" ] || { echo "no SETUP_OK in $BASE: run setup first"; exit 1; }
source $BASE/env_a76.sh
R=$BASE/root
[ -n "${A76_COMP_SHA256:-}" ] || { echo "A76_COMP_SHA256 unset: the comparator is pinned at registration; nothing runs"; exit 1; }
cd "$SRC" || exit 2
run() { nice -n 19 "$PY" ops/a76_d2p20.py "$@" --final-root "$R"; }
charged() { python3 -c "import json,os;p='$R/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
if [ "$STAGE" = test ]; then
  exec nice -n 19 "$PY" -m pytest -q -rs ops/test_a76_d2p20.py -p no:cacheprovider --basetemp=/dev/shm/a76_test_$$
fi
if [ "$STAGE" = bind ]; then
  [ -n "${A76_COMP_POLICY:-}" ] && [ -n "${A76_COMP_AUTHORITY:-}" ] || { echo "bind needs A76_COMP_POLICY and A76_COMP_AUTHORITY"; exit 1; }
  run bind-comp --policy "$A76_COMP_POLICY" --sha "$A76_COMP_SHA256" --authority "$A76_COMP_AUTHORITY"; exit $?
fi
if [ "$STAGE" = dry ]; then
  for st in m v c; do [ -f "$R/A76_PANEL_${st^^}.json" ] || run panel --stage $st || exit 3; done
  for st in m v; do run prepare --stage $st > "$BASE/dry_prepare_$st.out" 2>&1 || { echo "PREPARE $st FAILED"; tail -3 "$BASE/dry_prepare_$st.out"; exit 3; }; done
  [ -f "$R/A76_PINS.json" ] || run pin || exit 3
  for st in m v c; do run dry --stage $st || exit 3; done
  exit 0
fi
W=${A76_WORKERS:?set A76_WORKERS}
C=$BASE/chain_$STAGE.out
log() { echo "$(date "+%F %T") $*" >> "$C"; }
case "$STAGE" in mech) NEED=2 ;; admit) NEED=4 ;; bank) NEED=1 ;; confirm) NEED=4 ;; *) echo "unknown stage $STAGE"; exit 2 ;; esac
[ -f "$C" ] && { echo "$C exists: stage $STAGE already launched; not relaunching"; exit 1; }
[ -f "$R/STOP_A76" ] && { echo "STOP_A76 present: not launching"; exit 1; }
[ -f "$R/A76_PINS.json" ] || { echo "A76_PINS.json missing (run dry): not launching"; exit 1; }
FREE=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge "$NEED" ] || { echo "free disk ${FREE}G < ${NEED}G needed by stage $STAGE: not launching"; exit 1; }
DROP="--drop-raw --drop-procs 16"
[ "${A76_KEEP_RAW:-0}" = 1 ] && DROP=""
if [ -z "$DROP" ] && [ "$STAGE" = admit ] && [ "${FREE:-0}" -lt 25 ]; then echo "A76_KEEP_RAW=1 needs 25G free (have ${FREE}G): not launching"; exit 1; fi
mechpass() { grep -q '"verdict": "PASS"' "$R/A76_MECH.json" 2>/dev/null; }
case "$STAGE" in
admit) mechpass || { echo "mechanics is not PASS: nothing runs"; exit 5; } ;;
bank) [ -f "$BASE/chain_admit.out" ] && grep -q A76_VERIFY_DONE "$BASE/chain_admit.out" || { echo "admit stage not done"; exit 5; } ;;
confirm) [ -f "$R/A76_ADMISSION_D2800_p20.json" ] || { echo "no admission receipt"; exit 7; } ;;
esac
log "chain $STAGE start pid $$ workers $W free ${FREE}G drop '${DROP}' comparator ${A76_COMP_SHA256:0:12}"
echo $$ > "$BASE/chain_$STAGE.pid"
case "$STAGE" in
mech)
  run mech --workers $(( W < 20 ? W : 20 )) --chunk 20 >> "$BASE/mech.out" 2>&1 || { log "MECH RUN FAILED charged $(charged)"; exit 4; }
  run mech-compare >> "$BASE/mech.out" 2>&1
  log "mech $(tail -1 "$BASE/mech.out" | cut -c1-1500) charged $(charged)"
  mechpass || { log "MECHANICS FAILED: STOP"; exit 5; }
  log "A76_MECH_DONE" ;;
admit)
  run prepare --stage v >> "$BASE/verify.out" 2>&1 || { log "PREPARE FAILED"; exit 6; }
  log "verify start"
  run verify --workers "$W" --chunk 96 $DROP >> "$BASE/verify.out" 2>&1 || { log "VERIFY FAILED charged $(charged) free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")"; exit 6; }
  if [ -n "$DROP" ]; then run drop-raw --stage v --drop-procs 16 >> "$BASE/verify.out" 2>&1 || { log "DROP-RAW FAILED"; exit 6; }; fi
  log "verify done; charged $(charged)"
  log "A76_VERIFY_DONE free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")" ;;
bank)
  run admit >> "$BASE/admit.out" 2>&1 || { log "ADMIT FAILED"; exit 7; }
  log "admit: $(tail -1 "$BASE/admit.out" | cut -c1-2500)"
  log "A76_BANK_DONE" ;;
confirm)
  log "confirm start"
  run confirm --workers "$W" --chunk 96 $DROP >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED charged $(charged)"; exit 8; }
  if [ -n "$DROP" ]; then run drop-raw --stage c --drop-procs 16 >> "$BASE/confirm.out" 2>&1 || { log "DROP-RAW FAILED"; exit 8; }; fi
  log "confirm done; charged $(charged)"
  run analyse >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-1500)"
  log "A76_CONFIRM_DONE" ;;
esac
