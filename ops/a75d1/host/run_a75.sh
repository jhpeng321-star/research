#!/bin/bash
# AMENDMENT 75 stage D1 (S-DIRECTHOLD60C) host chain, ONE STAGE per launch (physics stages detached, launched ONCE; no
# retry loop; no automatic replay):   run_a75.sh dry | test | ident | admit | confirm
#   dry     : zero SUMO, no episodes: prepare (two small bank files) + dry i + dry v (incl. the stored-reference binding check); dry c when installed
#   test    : zero SUMO: the unit tests, temporary files in /dev/shm only
#   ident   : MAPPO60C + the first A68 candidate plan on the first 2 admission worlds (4 ep, 4,800 sim-s) ->
#             identity-compare vs the stored A68 records (FAILED: nothing else may run)
#   admit   : prepare (candidate bank) -> verify (36 plans x 24 worlds, 1,036,800 sim-s; reference NOT re-run; raw
#             archives dropped per round after report + receipt, see DROP_RULE) -> admit (bank, zero physics)
#   confirm : confirm (DIRECT_BANK_60C on the 96 A70 worlds, 115,200 sim-s) -> analyse   [needs the confirm install]
# Only /root/a75 (and /dev/shm for the tests) is written. Stop between dispatch rounds: touch /root/a75/root/STOP_A75.
# A75_KEEP_RAW=1 keeps the raw archives (needs ~20 GB per 1,000 episodes).
STAGE=${1:?stage: dry | test | ident | admit | confirm}
source /root/a75/env_a75.sh
BASE=/root/a75
R=$BASE/root
REFS="--a68-root /root/a68/root --a66-root /root/a66/root --ref68-reports $BASE/ref68"
[ -d "$BASE/a70ref/root" ] && REFS="$REFS --a70-root $BASE/a70ref/root --ref70-reports $BASE/ref70"
run() { nice -n 19 "$PY" ops/a75_directhold.py "$@" --final-root "$R" $REFS; }
charged() { python3 -c "import json,os;p='$R/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
if [ "$STAGE" = dry ]; then
  run prepare > "$BASE/dry_prepare.out" 2>&1 || { echo "PREPARE FAILED"; tail -3 "$BASE/dry_prepare.out"; exit 3; }
  run dry --stage i || exit 3
  run dry --stage v || exit 3
  if [ -f "$BASE/CONFIRM_SETUP_OK" ]; then run dry --stage c || exit 3; else echo "confirm pieces not installed: dry c skipped"; fi
  exit 0
fi
if [ "$STAGE" = test ]; then
  exec "$PY" -m pytest -q ops/test_a75_directhold.py -p no:cacheprovider --basetemp=/dev/shm/a75_test_$$
fi
W=${A75_WORKERS:?set A75_WORKERS}
C=$BASE/chain_$STAGE.out
log() { echo "$(date "+%F %T") $*" >> "$C"; }
case "$STAGE" in ident) NEED=1 ;; admit) NEED=6 ;; confirm) NEED=4 ;; *) echo "unknown stage $STAGE"; exit 2 ;; esac
[ -f "$C" ] && { echo "$C exists: not relaunching"; exit 1; }
[ -f "$BASE/SETUP_OK" ] || { echo "no SETUP_OK: not launching"; exit 1; }
[ -f ops/a75d1/CANDIDATES_D1_D2800_p60.sha256 ] || { echo "candidate pin missing: not launching"; exit 1; }
[ "$STAGE" = confirm ] && [ ! -f "$BASE/CONFIRM_SETUP_OK" ] && { echo "confirm pieces not installed: not launching"; exit 1; }
[ -f "$R/STOP_A75" ] && { echo "STOP_A75 present: not launching"; exit 1; }
FREE=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge "$NEED" ] || { echo "free disk ${FREE}G < ${NEED}G needed by stage $STAGE: not launching"; exit 1; }
DROP="--drop-raw --drop-procs 16"
[ "${A75_KEEP_RAW:-0}" = 1 ] && DROP=""
if [ -z "$DROP" ] && [ "$STAGE" = admit ] && [ "${FREE:-0}" -lt 25 ]; then echo "A75_KEEP_RAW=1 needs 25G free (have ${FREE}G): not launching"; exit 1; fi
log "chain $STAGE start pid $$ workers $W free ${FREE}G drop '${DROP}'"
echo $$ > "$BASE/chain_$STAGE.pid"
case "$STAGE" in
ident)
  run prepare >> "$BASE/identity.out" 2>&1 || { log "PREPARE FAILED"; exit 4; }
  run ident --workers 4 --chunk 4 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED charged $(charged)"; exit 4; }
  run identity-compare >> "$BASE/identity.out" 2>&1 || { log "IDENTITY COMPARE ERROR"; exit 4; }
  log "identity $(tail -1 "$BASE/identity.out" | cut -c1-1200) charged $(charged)"
  grep -q '"verdict": "PASS"' "$R/IDENTITY_A75.json" || { log "IDENTITY FAILED: STOP"; exit 5; }
  log "A75_IDENT_DONE" ;;
admit)
  grep -q '"verdict": "PASS"' "$R/IDENTITY_A75.json" 2>/dev/null || { log "identity is not PASS: nothing runs"; exit 5; }
  run prepare >> "$BASE/verify.out" 2>&1 || { log "PREPARE FAILED"; exit 6; }
  log "verify start"
  run verify --workers "$W" --chunk 96 $DROP >> "$BASE/verify.out" 2>&1 || { log "VERIFY FAILED charged $(charged) free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")"; exit 6; }
  log "verify done; charged $(charged)"
  if [ -n "$DROP" ]; then run drop-raw --stage v --drop-procs 16 >> "$BASE/verify.out" 2>&1 || { log "DROP-RAW FAILED"; exit 6; }; fi
  run admit >> "$BASE/admit.out" 2>&1 || { log "ADMIT FAILED"; exit 7; }
  log "admit: $(tail -1 "$BASE/admit.out" | cut -c1-2500)"
  log "A75_ADMIT_DONE free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")" ;;
confirm)
  [ -f "$R/A75_ADMISSION_D2800_p60.json" ] || { log "no admission receipt"; exit 7; }
  log "confirm start"
  run confirm --workers "$W" --chunk 96 $DROP >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED charged $(charged)"; exit 8; }
  log "confirm done; charged $(charged)"
  if [ -n "$DROP" ]; then run drop-raw --stage c --drop-procs 16 >> "$BASE/confirm.out" 2>&1 || { log "DROP-RAW FAILED"; exit 8; }; fi
  run analyse >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-1500)"
  log "A75_CONFIRM_DONE" ;;
esac
