#!/bin/bash
# AMENDMENT 70 host chain, ONE STAGE per launch (detached, launched ONCE per stage; never inside a retry loop; no
# automatic replay):   bash run_a70.sh ident | confirm
#   ident   : identity (4 episodes) -> identity-compare (FAILED: nothing else may run)
#   confirm : the four arms on the 96 worlds 996300-996395 -> analyse
# Only /root/a70 is written. Stop between dispatch rounds: touch /root/a70/root/STOP_A70.
STAGE=${1:?stage: ident | confirm}
source /root/a70/env_a70.sh
BASE=/root/a70
R=$BASE/root
W=${A70_WORKERS:?set A70_WORKERS}
C=$BASE/chain_$STAGE.out
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" ops/a70_smallbank.py "$@" --final-root "$R" --a68-root /root/a68/root --a66-root /root/a66/root --a67-root /root/a67/root --a42-root /root/a66/refs/a42/root; }
charged() { python3 -c "import json,os;p='$R/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
[ -f "$C" ] && { echo "$C exists: not relaunching"; exit 1; }
log "chain $STAGE start pid $$ workers $W"
echo $$ > "$BASE/chain_$STAGE.pid"
[ -f "$R/STOP_A70" ] && { log "STOP_A70 present"; exit 3; }
case "$STAGE" in
ident)
  run identity --workers 4 --chunk 4 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 4; }
  run identity-compare >> "$BASE/identity.out" 2>&1 || { log "IDENTITY COMPARE ERROR"; exit 4; }
  log "identity $(tail -1 "$BASE/identity.out" | cut -c1-900) charged $(charged)"
  grep -q '"verdict": "PASS"' "$R/IDENTITY_A70.json" || { log "IDENTITY FAILED: STOP"; exit 5; }
  log "A70_IDENT_DONE" ;;
confirm)
  grep -q '"verdict": "PASS"' "$R/IDENTITY_A70.json" || { log "identity is not PASS: nothing runs"; exit 5; }
  [ -f "$R/A70_SELECTION_D2800_p60.json" ] || { log "no selection record"; exit 7; }
  log "confirm start"
  run confirm --workers "$W" --chunk 96 >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED charged $(charged)"; exit 8; }
  log "confirm done; charged $(charged)"
  run analyse --out "$R" >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-900)"
  log "A70_CONFIRM_DONE" ;;
*) log "unknown stage $STAGE"; exit 2 ;;
esac
