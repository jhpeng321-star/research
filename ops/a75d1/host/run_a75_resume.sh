#!/bin/bash
# AMENDMENT 75 stage D1: RESUME of the admission stage after the 13:45 stop (drop_raw compared the RESULT.json pointer
# with the sha of the .gz FILE; the runner's pointer records the DECOMPRESSED content). One launch only.
#   run_a75_resume.sh dry          : zero SUMO: prepare (idempotent) + dry v (plan: published waves are not pending)
#   run_a75_resume.sh test         : zero SUMO: unit tests (/dev/shm)
#   run_a75_resume.sh admit_resume : prepare -> verify (ONLY the unpublished waves run and are charged; the published
#                                    round is reused as is; same 1,036,800 cap) -> drop-raw -> admit
# Refuses: second launch (chain_admit_resume.out exists), no stopped first launch, another driver running, a driver that
# is not the pinned fixed one, charges that are not exactly the 120,000 sim-s of ident + the first round.
STAGE=${1:?stage: dry | test | admit_resume}
source /root/a75/env_a75.sh
BASE=/root/a75
R=$BASE/root
REFS="--a68-root /root/a68/root --a66-root /root/a66/root --ref68-reports $BASE/ref68"
[ -d "$BASE/a70ref/root" ] && REFS="$REFS --a70-root $BASE/a70ref/root --ref70-reports $BASE/ref70"
run() { nice -n 19 "$PY" ops/a75_directhold.py "$@" --final-root "$R" $REFS; }
charged() { python3 -c "import json,os;p='$R/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
PIN=$(cut -c1-64 "$BASE/DRIVER_FIX.sha256" 2>/dev/null)
[ -n "$PIN" ] && [ "$(sha256sum ops/a75_directhold.py | cut -c1-64)" = "$PIN" ] || { echo "driver is not the pinned fixed driver: not running"; exit 1; }
if [ "$STAGE" = dry ]; then
  run prepare > "$BASE/dry_prepare_resume.out" 2>&1 || { echo "PREPARE FAILED"; tail -3 "$BASE/dry_prepare_resume.out"; exit 3; }
  run dry --stage v || exit 3
  echo "charged $(charged)"
  exit 0
fi
if [ "$STAGE" = test ]; then
  exec "$PY" -m pytest -q ops/test_a75_directhold.py -p no:cacheprovider --basetemp=/dev/shm/a75_test_$$
fi
[ "$STAGE" = admit_resume ] || { echo "unknown stage $STAGE"; exit 2; }
W=${A75_WORKERS:?set A75_WORKERS}
C=$BASE/chain_admit_resume.out
log() { echo "$(date "+%F %T") $*" >> "$C"; }
grep -q "VERIFY FAILED" "$BASE/chain_admit.out" 2>/dev/null || { echo "no stopped first admit launch: not resuming"; exit 1; }
grep -q "A75_ADMIT_DONE" "$BASE/chain_admit.out" && { echo "admit already done: not resuming"; exit 1; }
[ -f "$R/STOP_A75" ] && { echo "STOP_A75 present: not launching"; exit 1; }
pgrep -f "[a]75_directhold.py" > /dev/null && { echo "a driver is running: not launching"; exit 1; }
grep -q '"verdict": "PASS"' "$R/IDENTITY_A75.json" 2>/dev/null || { echo "identity is not PASS: nothing runs"; exit 5; }
CH=$(charged)
[ "$CH" = 120000 ] || { echo "charged $CH != 120000 (ident 4,800 + first round 115,200): not launching"; exit 1; }
FREE=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge 6 ] || { echo "free disk ${FREE}G < 6G: not launching"; exit 1; }
( set -C; echo "$(date "+%F %T") chain admit_resume start pid $$ workers $W free ${FREE}G charged $CH" > "$C" ) 2>/dev/null || { echo "$C exists: not relaunching"; exit 1; }
echo $$ > "$BASE/chain_admit_resume.pid"
V=$BASE/verify_resume.out
run prepare >> "$V" 2>&1 || { log "PREPARE FAILED"; exit 6; }
log "verify resume start"
run verify --workers "$W" --chunk 96 --drop-raw --drop-procs 16 >> "$V" 2>&1 || { log "VERIFY FAILED charged $(charged) free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")"; exit 6; }
log "verify done; charged $(charged)"
run drop-raw --stage v --drop-procs 16 >> "$V" 2>&1 || { log "DROP-RAW FAILED"; exit 6; }
run admit >> "$BASE/admit.out" 2>&1 || { log "ADMIT FAILED"; exit 7; }
log "admit: $(tail -1 "$BASE/admit.out" | cut -c1-2500)"
log "A75_ADMIT_DONE free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")"
