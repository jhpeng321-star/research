#!/bin/bash
# AMENDMENT 80 (A80-S2B) host chain, ONE STAGE per launch, each physics / sealing stage runs ONCE (its chain file exists
# -> refuse); physics stages are meant to be detached (setsid nohup); no retry loop; no automatic replay.
#   run_a80.sh setup     : zero SUMO install (setup_a80.sh)
#   run_a80.sh test      : zero SUMO unit tests of the A80 driver (one process; temporary files in /dev/shm)
#   run_a80.sh synthdry  : zero SUMO, THROWAWAY root $BASE/synth: pin the synthetic a75ss-writer file, prepare, dry v / c
#   run_a80.sh pin       : zero SUMO, the real root: pin-top (needs A80_TOP_FILE + A80_TOP_SHA256), prepare, dry v / c
#   run_a80.sh mech      : uncovered-token plans x 2 admission worlds (<= 12 x 2,400 sim-s, inside the verify cap)
#                          -> mech-compare (A80_MECH.json PASS / FAIL); nothing else runs unless PASS
#   run_a80.sh admit     : the rest of the <= 12 x 24 verification episodes (<= 345,600 sim-s incl. mech; reference
#                          NOT re-run; raw archives dropped per round)
#   run_a80.sh bank      : zero physics: receipt invariants + admission rule + bank rule -> A80_ADMISSION + bank file
#   run_a80.sh confirm   : A80_BANK_60C on the 96 A70 worlds (115,200 sim-s) -> analyse
# Only $BASE (default /root/a78s2b) and /dev/shm (tests) are written. Stop between dispatch rounds: touch $BASE/root/STOP_A80.
STAGE=${1:?stage: setup | test | synthdry | pin | mech | admit | bank | confirm}
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${A80_BASE:-/root/a78s2b}
if [ "$STAGE" = setup ]; then
  [ -f "$BASE/SETUP_OK" ] && { echo "setup already done: not rerunning"; exit 1; }
  exec bash "$HERE/setup_a80.sh"
fi
source $BASE/env_a80.sh
R=$BASE/root
A70R=${A80_A70_ROOT:-/root/a75/a70ref/root}
REF70=${A80_REF70:-/root/a75/ref70}
REFS="--a68-root /root/a68/root --a66-root /root/a66/root --ref68-reports $BASE/ref68"
[ -d "$A70R" ] && [ -d "$REF70" ] && REFS="$REFS --a70-root $A70R --ref70-reports $REF70"
cd "$SRC" || exit 2
run() { nice -n 19 "$PY" ops/a80_s2b.py "$@" --final-root "$R" $REFS; }
charged() { python3 -c "import json,os;p='$R/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
[ -f "$BASE/SETUP_OK" ] || { echo "no SETUP_OK: run setup first"; exit 1; }
if [ "$STAGE" = test ]; then
  exec nice -n 19 "$PY" -m pytest -q -rs ops/test_a80_s2b.py -p no:cacheprovider --basetemp=/dev/shm/a80_test_$$
fi
if [ "$STAGE" = synthdry ]; then
  SR=$BASE/synth/root
  [ -f "$SR/A80_CANDIDATES.json" ] && { echo "synthetic root already pinned: not rerunning"; exit 1; }
  export A80_HOST_BASE=$BASE/synth A80_TOP_FILE=$SRC/ops/a80/SYNTH_TOP_OFFER_CONDITIONS_D2800_p60.json
  export A80_TOP_SHA256=$(sha256sum "$A80_TOP_FILE" | cut -c1-64)
  srun() { nice -n 19 "$PY" ops/a80_s2b.py "$@" --final-root "$SR" $REFS; }
  srun pin-top && srun prepare && srun dry --stage v > "$BASE/synth/dry_v.out" 2>&1 \
    || { echo "SYNTHDRY FAILED"; tail -5 "$BASE/synth/dry_v.out" 2>/dev/null; exit 3; }
  srun dry --stage c > "$BASE/synth/dry_c.out" 2>&1 || echo "WARNING: dry c failed (see $BASE/synth/dry_c.out)"
  [ -f "$SR/CHARGES.jsonl" ] && { echo "SYNTHDRY charged SUMO seconds: must not happen"; exit 3; }
  echo SYNTHDRY_OK; exit 0
fi
if [ "$STAGE" = pin ]; then
  [ -n "${A80_TOP_FILE:-}" ] && [ -n "${A80_TOP_SHA256:-}" ] || { echo "A80_TOP_FILE and A80_TOP_SHA256 required"; exit 1; }
  [ -f "$R/A80_CANDIDATES.json" ] && { echo "already pinned: not rerunning"; exit 1; }
  run pin-top > "$BASE/pin.out" 2>&1 || { echo "PIN-TOP REFUSED (see $BASE/pin.out, $R/A80_MAPPING_REFUSED.json): STOP"; tail -3 "$BASE/pin.out"; exit 3; }
  run prepare >> "$BASE/pin.out" 2>&1 || { echo "PREPARE FAILED"; tail -3 "$BASE/pin.out"; exit 3; }
  run dry --stage v > "$BASE/dry_v.out" 2>&1 || { echo "DRY v FAILED"; tail -3 "$BASE/dry_v.out"; exit 3; }
  run dry --stage c > "$BASE/dry_c.out" 2>&1 || echo "WARNING: dry c failed (see $BASE/dry_c.out)"
  cat "$BASE/pin.out"; echo PIN_OK; exit 0
fi
W=${A80_WORKERS:?set A80_WORKERS}
C=$BASE/chain_$STAGE.out
log() { echo "$(date "+%F %T") $*" >> "$C"; }
case "$STAGE" in mech) NEED=2 ;; admit) NEED=4 ;; bank) NEED=1 ;; confirm) NEED=4 ;; *) echo "unknown stage $STAGE"; exit 2 ;; esac
[ -f "$C" ] && { echo "$C exists: stage $STAGE already launched; not relaunching"; exit 1; }
[ -f "$R/STOP_A80" ] && { echo "STOP_A80 present: not launching"; exit 1; }
[ -f "$R/A80_CANDIDATES.json" ] || { echo "no pinned candidates (run pin): nothing runs"; exit 1; }
FREE=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge "$NEED" ] || { echo "free disk ${FREE}G < ${NEED}G needed by stage $STAGE: not launching"; exit 1; }
DROP="--drop-raw --drop-procs 16"
[ "${A80_KEEP_RAW:-0}" = 1 ] && DROP=""
mechpass() { grep -q '"verdict": "PASS"' "$R/A80_MECH.json" 2>/dev/null; }
case "$STAGE" in
admit) mechpass || { echo "mechanics is not PASS: nothing runs"; exit 5; } ;;
bank) [ -f "$BASE/chain_admit.out" ] && grep -q A80_VERIFY_DONE "$BASE/chain_admit.out" || { echo "admit stage not done"; exit 5; } ;;
confirm) [ -f "$R/A80_ADMISSION_D2800_p60.json" ] || { echo "no admission receipt"; exit 7; }
         [ -d "$A70R" ] && [ -d "$REF70" ] || { echo "A70 references missing ($A70R, $REF70): not launching"; exit 7; } ;;
esac
log "chain $STAGE start pid $$ workers $W free ${FREE}G drop '${DROP}'"
echo $$ > "$BASE/chain_$STAGE.pid"
case "$STAGE" in
mech)
  run mech --workers $(( W < 24 ? W : 24 )) --chunk 24 >> "$BASE/mech.out" 2>&1 || { log "MECH RUN FAILED charged $(charged)"; exit 4; }
  run mech-compare >> "$BASE/mech.out" 2>&1
  log "mech $(tail -1 "$BASE/mech.out" | cut -c1-2500) charged $(charged)"
  mechpass || { log "MECHANICS FAILED: STOP"; exit 5; }
  log "A80_MECH_DONE" ;;
admit)
  log "verify start"
  run verify --workers "$W" --chunk 96 $DROP >> "$BASE/verify.out" 2>&1 || { log "VERIFY FAILED charged $(charged) free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")"; exit 6; }
  if [ -n "$DROP" ]; then run drop-raw --stage v --drop-procs 16 >> "$BASE/verify.out" 2>&1 || { log "DROP-RAW FAILED"; exit 6; }; fi
  log "verify done; charged $(charged)"
  log "A80_VERIFY_DONE free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")" ;;
bank)
  run admit >> "$BASE/admit.out" 2>&1 || { log "ADMIT FAILED: $(tail -1 "$BASE/admit.out" | cut -c1-1500)"; exit 7; }
  log "admit: $(tail -1 "$BASE/admit.out" | cut -c1-2500)"
  log "A80_BANK_DONE" ;;
confirm)
  log "confirm start"
  run confirm --workers "$W" --chunk 96 $DROP >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED charged $(charged)"; exit 8; }
  if [ -n "$DROP" ]; then run drop-raw --stage c --drop-procs 16 >> "$BASE/confirm.out" 2>&1 || { log "DROP-RAW FAILED"; exit 8; }; fi
  log "confirm done; charged $(charged)"
  run analyse >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-1500)"
  log "A80_CONFIRM_DONE" ;;
esac
