#!/bin/bash
# AMENDMENT 71 host chain, ONE STAGE per launch (detached, launched ONCE per stage; never inside a retry loop; no
# automatic replay):   bash run_a71.sh admit | confirm
#   admit   : verify (every new plan alone, 24 worlds 992500-992523) -> admit (verdicts, per-size table, WIDE_BANK)
#   confirm : WIDE_BANK_60C on the A70 panel 996300-996395 -> analyse (not run when no plan is admitted)
# Only /root/a71 is written. Stop between dispatch rounds: touch /root/a71/root/STOP_A71.
STAGE=${1:?stage: admit | confirm}
source /root/a70/env_a70.sh
BASE=/root/a71
R=$BASE/root
W=${A71_WORKERS:?set A71_WORKERS}
C=$BASE/chain_$STAGE.out
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" ops/a71_wideadmit.py "$@" --final-root "$R" --a68-root /root/a68/root --a70-root /root/a70/root --a66-root /root/a66/root; }
charged() { python3 -c "import json,os;p='$R/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
[ -f "$C" ] && { echo "$C exists: not relaunching"; exit 1; }
log "chain $STAGE start pid $$ workers $W"
echo $$ > "$BASE/chain_$STAGE.pid"
[ -f "$R/STOP_A71" ] && { log "STOP_A71 present"; exit 3; }
grep -q '"verdict": "PASS"' /root/a70/root/IDENTITY_A70.json || { log "A70 tree identity is not PASS: nothing runs"; exit 5; }
case "$STAGE" in
admit)
  log "verify start"
  run verify --workers "$W" --chunk 96 >> "$BASE/verify.out" 2>&1 || { log "VERIFY FAILED charged $(charged)"; exit 6; }
  log "verify done; charged $(charged)"
  run admit >> "$BASE/admit.out" 2>&1 || { log "ADMIT FAILED"; exit 7; }
  log "admit: $(tail -1 "$BASE/admit.out" | cut -c1-2500)"
  log "A71_ADMIT_DONE" ;;
confirm)
  [ -f "$R/A71_ADMISSION_D2800_p60.json" ] || { log "no admission receipt"; exit 7; }
  [ -f "$R/a71_banks/A71_WIDE_BANK_D2800_p60.json" ] || { log "no plan admitted: no confirmation"; exit 0; }
  [ -f /root/a70/root/A70_RESULTS_D2800_p60.json ] || { log "A70 arms are not complete"; exit 7; }
  log "confirm start"
  run confirm --workers "$W" --chunk 96 >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED charged $(charged)"; exit 8; }
  log "confirm done; charged $(charged)"
  run analyse --out "$R" >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-900)"
  log "A71_CONFIRM_DONE" ;;
*) log "unknown stage $STAGE"; exit 2 ;;
esac
