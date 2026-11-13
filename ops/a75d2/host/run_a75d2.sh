#!/bin/bash
# AMENDMENT 75 stage D2 (S-HOLDFORMS-D2) host chain, ONE STAGE per launch, each stage runs ONCE (a stage whose chain
# file exists refuses); physics stages detached; no retry loop; no automatic replay.
#   run_a75d2.sh setup [ref68_reports.tar] : zero SUMO install (setup_a75d2.sh; needs A75D2_MECH_SEED0)
#   run_a75d2.sh test     : zero SUMO unit tests (temporary files in /dev/shm only)
#   run_a75d2.sh dry      : zero SUMO: prepare banks, panel-m, dry i / m / v (incl. the stored-reference binding check)
#   run_a75d2.sh mech     : 4 identity + 16 form episodes (24,000 sim-s) -> mech-compare (A75D2_MECH.json PASS/FAIL)
#   run_a75d2.sh admit    : 36 plans x 24 worlds (1,036,800 sim-s; reference NOT re-run; raw archives dropped per round)
#   run_a75d2.sh bank     : zero physics: admission rule + bank rule -> A75D2_ADMISSION_D2800_p60.json + bank file
#   run_a75d2.sh confirm  : A75D2_BANK_60C on the 96 A70 worlds (115,200 sim-s) -> analyse
# Only /root/a75d2b (and /dev/shm for the tests) is written. Stop between dispatch rounds: touch /root/a75d2b/root/STOP_A75D2.
# A75D2_KEEP_RAW=1 keeps the raw archives of admit / confirm (needs ~20 GB per 1,000 episodes; default: dropped).
STAGE=${1:?stage: setup | test | dry | mech | admit | bank | confirm}
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a75d2b
if [ "$STAGE" = setup ]; then
  [ -f "$BASE/SETUP_OK" ] && { echo "setup already done: not rerunning"; exit 1; }
  exec bash "$HERE/setup_a75d2.sh" "${2:-}"
fi
source $BASE/env_a75d2.sh
R=$BASE/root
A70R=${A75D2_A70_ROOT:-/root/a75/a70ref/root}
REF70=${A75D2_REF70:-/root/a75/ref70}
REFS="--a68-root /root/a68/root --a66-root /root/a66/root --ref68-reports $BASE/ref68"
[ -d "$A70R" ] && [ -d "$REF70" ] && REFS="$REFS --a70-root $A70R --ref70-reports $REF70"
cd "$SRC" || exit 2
run() { nice -n 19 "$PY" ops/a75_d2forms.py "$@" --final-root "$R" $REFS; }
charged() { python3 -c "import json,os;p='$R/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
[ -f "$BASE/SETUP_OK" ] || { echo "no SETUP_OK: run setup first"; exit 1; }
[ -n "${A75D2_MECH_SEED0:-}" ] || { echo "A75D2_MECH_SEED0 unset: nothing runs"; exit 1; }
if [ "$STAGE" = test ]; then
  exec "$PY" -m pytest -q ops/test_a75_d2forms.py -p no:cacheprovider --basetemp=/dev/shm/a75d2_test_$$
fi
if [ "$STAGE" = dry ]; then
  for st in i m v; do run prepare --stage $st > "$BASE/dry_prepare_$st.out" 2>&1 || { echo "PREPARE $st FAILED"; tail -3 "$BASE/dry_prepare_$st.out"; exit 3; }; done
  [ -f "$R/A75D2_PANEL_M.json" ] || run panel-m || exit 3
  for st in i m v; do run dry --stage $st || exit 3; done
  exit 0
fi
W=${A75D2_WORKERS:?set A75D2_WORKERS}
C=$BASE/chain_$STAGE.out
log() { echo "$(date "+%F %T") $*" >> "$C"; }
case "$STAGE" in mech) NEED=2 ;; admit) NEED=4 ;; bank) NEED=1 ;; confirm) NEED=4 ;; *) echo "unknown stage $STAGE"; exit 2 ;; esac
[ -f "$C" ] && { echo "$C exists: stage $STAGE already launched; not relaunching"; exit 1; }
[ -f "$R/STOP_A75D2" ] && { echo "STOP_A75D2 present: not launching"; exit 1; }
FREE=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge "$NEED" ] || { echo "free disk ${FREE}G < ${NEED}G needed by stage $STAGE: not launching"; exit 1; }
DROP="--drop-raw --drop-procs 16"
[ "${A75D2_KEEP_RAW:-0}" = 1 ] && DROP=""
if [ -z "$DROP" ] && [ "$STAGE" = admit ] && [ "${FREE:-0}" -lt 25 ]; then echo "A75D2_KEEP_RAW=1 needs 25G free (have ${FREE}G): not launching"; exit 1; fi
mechpass() { grep -q '"verdict": "PASS"' "$R/A75D2_MECH.json" 2>/dev/null; }
case "$STAGE" in
admit) mechpass || { echo "mechanics is not PASS: nothing runs"; exit 5; } ;;
bank) [ -f "$BASE/chain_admit.out" ] && grep -q A75D2_VERIFY_DONE "$BASE/chain_admit.out" || { echo "admit stage not done"; exit 5; } ;;
confirm) [ -f "$R/A75D2_ADMISSION_D2800_p60.json" ] || { echo "no admission receipt"; exit 7; }
         [ -d "$A70R" ] && [ -d "$REF70" ] || { echo "A70 references missing ($A70R, $REF70): not launching"; exit 7; } ;;
esac
log "chain $STAGE start pid $$ workers $W free ${FREE}G drop '${DROP}' seed0 $A75D2_MECH_SEED0"
echo $$ > "$BASE/chain_$STAGE.pid"
case "$STAGE" in
mech)
  { run prepare --stage i && run prepare --stage m; } >> "$BASE/mech.out" 2>&1 || { log "PREPARE FAILED"; exit 4; }
  [ -f "$R/A75D2_PANEL_M.json" ] || run panel-m >> "$BASE/mech.out" 2>&1 || { log "PANEL-M FAILED"; exit 4; }
  run mech-ident --workers 4 --chunk 4 >> "$BASE/mech.out" 2>&1 || { log "MECH-IDENT RUN FAILED charged $(charged)"; exit 4; }
  run mech-forms --workers $(( W < 16 ? W : 16 )) --chunk 16 >> "$BASE/mech.out" 2>&1 || { log "MECH-FORMS RUN FAILED charged $(charged)"; exit 4; }
  run mech-compare >> "$BASE/mech.out" 2>&1
  log "mech $(tail -1 "$BASE/mech.out" | cut -c1-1500) charged $(charged)"
  mechpass || { log "MECHANICS FAILED: STOP"; exit 5; }
  log "A75D2_MECH_DONE" ;;
admit)
  run prepare --stage v >> "$BASE/verify.out" 2>&1 || { log "PREPARE FAILED"; exit 6; }
  log "verify start"
  run verify --workers "$W" --chunk 96 $DROP >> "$BASE/verify.out" 2>&1 || { log "VERIFY FAILED charged $(charged) free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")"; exit 6; }
  if [ -n "$DROP" ]; then run drop-raw --stage v --drop-procs 16 >> "$BASE/verify.out" 2>&1 || { log "DROP-RAW FAILED"; exit 6; }; fi
  log "verify done; charged $(charged)"
  log "A75D2_VERIFY_DONE free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")" ;;
bank)
  run admit >> "$BASE/admit.out" 2>&1 || { log "ADMIT FAILED"; exit 7; }
  log "admit: $(tail -1 "$BASE/admit.out" | cut -c1-2500)"
  log "A75D2_BANK_DONE" ;;
confirm)
  log "confirm start"
  run confirm --workers "$W" --chunk 96 $DROP >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED charged $(charged)"; exit 8; }
  if [ -n "$DROP" ]; then run drop-raw --stage c --drop-procs 16 >> "$BASE/confirm.out" 2>&1 || { log "DROP-RAW FAILED"; exit 8; }; fi
  log "confirm done; charged $(charged)"
  run analyse >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-1500)"
  log "A75D2_CONFIRM_DONE" ;;
esac
