#!/bin/bash
# AMENDMENT 75 stage D1 confirmation, run from the confirm tree /root/a75/src_c (A70 panel tables; see
# setup_a75_confirm.sh). ONE launch (detached, launched ONCE; no retry loop; no automatic replay):
#   run_a75_confirm.sh dry | confirm
#   dry     : zero SUMO: the plan of the confirmation wave (needs the admission receipt with a non-empty bank)
#   confirm : DIRECT_BANK_60C on the 96 A70 worlds (115,200 sim-s; MAPPO60C / DUO_HOLD_60C NOT re-run: stored A70
#             waves, pinned) -> analyse (A75_RESULTS_D2800_p60.json incl. worst-10% delay)
# Only /root/a75 is written. Stop between dispatch rounds: touch /root/a75/root/STOP_A75. A75_KEEP_RAW=1 keeps raw.
STAGE=${1:?stage: dry | confirm}
source /root/a75/env_a75c.sh
BASE=/root/a75
R=$BASE/root
REFS="--a68-root /root/a68/root --a66-root /root/a66/root --ref68-reports $BASE/ref68 --a70-root $BASE/a70ref/root --ref70-reports $BASE/ref70"
run() { nice -n 19 "$PY" ops/a75_directhold.py "$@" --final-root "$R" $REFS; }
charged() { python3 -c "import json,os;p='$R/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
[ -f "$BASE/CONFIRM_TREE_OK" ] || { echo "no CONFIRM_TREE_OK: not launching"; exit 1; }
[ -f "$R/A75_ADMISSION_D2800_p60.json" ] || { echo "no admission receipt: nothing to confirm"; exit 1; }
if [ "$STAGE" = dry ]; then run dry --stage c; exit $?; fi
[ "$STAGE" = confirm ] || { echo "unknown stage $STAGE"; exit 2; }
W=${A75_WORKERS:?set A75_WORKERS}
C=$BASE/chain_confirm.out
log() { echo "$(date "+%F %T") $*" >> "$C"; }
[ -f "$C" ] && { echo "$C exists: not relaunching"; exit 1; }
[ -f "$R/STOP_A75" ] && { echo "STOP_A75 present: not launching"; exit 1; }
FREE=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge 4 ] || { echo "free disk ${FREE}G < 4G needed by stage confirm: not launching"; exit 1; }
DROP="--drop-raw --drop-procs 16"
[ "${A75_KEEP_RAW:-0}" = 1 ] && DROP=""
log "chain confirm start pid $$ workers $W free ${FREE}G drop '${DROP}'"
echo $$ > "$BASE/chain_confirm.pid"
log "confirm start"
run confirm --workers "$W" --chunk 96 $DROP >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED charged $(charged)"; exit 8; }
log "confirm done; charged $(charged)"
if [ -n "$DROP" ]; then run drop-raw --stage c --drop-procs 16 >> "$BASE/confirm.out" 2>&1 || { log "DROP-RAW FAILED"; exit 8; }; fi
run analyse >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-1500)"
log "A75_CONFIRM_DONE"
