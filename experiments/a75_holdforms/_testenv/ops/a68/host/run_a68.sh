#!/bin/bash
# AMENDMENT 68 host chain, ONE STAGE per launch (detached, launched ONCE per stage; never inside a retry loop; no
# automatic replay):   run_a68.sh ident | admit | confirm
#   ident   : identity (4 episodes) -> identity-compare (FAILED: nothing else may run)
#   admit   : prepare (candidates bank) -> verify (MAPPO60C + candidates, 24 worlds) -> admit (banks, zero physics)
#   confirm : confirm (arms per the admission receipt, 96 worlds) -> analyse
# Only /root/a68 is written. Stop between dispatch rounds: touch /root/a68/root/STOP_A68.
STAGE=${1:?stage: ident | admit | confirm}
source /root/a68/env_a68.sh
BASE=/root/a68
R=$BASE/root
W=${A68_WORKERS:?set A68_WORKERS}
C=$BASE/chain_$STAGE.out
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" ops/a68_research60c.py "$@" --final-root "$R" --a66-root /root/a66/root --a67-root /root/a67/root --a42-root /root/a66/refs/a42/root; }
charged() { python3 -c "import json,os;p='$R/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
[ -f "$C" ] && { echo "$C exists: not relaunching"; exit 1; }
log "chain $STAGE start pid $$ workers $W"
echo $$ > "$BASE/chain_$STAGE.pid"
[ -f "$R/STOP_A68" ] && { log "STOP_A68 present"; exit 3; }
case "$STAGE" in
ident)
  run identity --workers 4 --chunk 4 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 4; }
  run identity-compare >> "$BASE/identity.out" 2>&1 || { log "IDENTITY COMPARE ERROR"; exit 4; }
  log "identity $(tail -1 "$BASE/identity.out" | cut -c1-900) charged $(charged)"
  grep -q '"verdict": "PASS"' "$R/IDENTITY_A68.json" || { log "IDENTITY FAILED: STOP"; exit 5; }
  log "A68_IDENT_DONE" ;;
admit)
  grep -q '"verdict": "PASS"' "$R/IDENTITY_A68.json" || { log "identity is not PASS: nothing runs"; exit 5; }
  run prepare >> "$BASE/verify.out" 2>&1 || { log "PREPARE FAILED"; exit 6; }
  log "verify start"
  run verify --workers "$W" --chunk 96 >> "$BASE/verify.out" 2>&1 || { log "VERIFY FAILED charged $(charged)"; exit 6; }
  log "verify done; charged $(charged)"
  run admit >> "$BASE/admit.out" 2>&1 || { log "ADMIT FAILED"; exit 7; }
  log "admit: $(tail -1 "$BASE/admit.out" | cut -c1-1500)"
  log "A68_ADMIT_DONE" ;;
confirm)
  [ -f "$R/A68_ADMISSION_D2800_p60.json" ] || { log "no admission receipt"; exit 7; }
  log "confirm start"
  run confirm --workers "$W" --chunk 96 >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED charged $(charged)"; exit 8; }
  log "confirm done; charged $(charged)"
  run analyse --out "$R" >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-900)"
  log "A68_CONFIRM_DONE" ;;
*) log "unknown stage $STAGE"; exit 2 ;;
esac
