#!/bin/bash
# AMENDMENT 67 host chain (detached, launched ONCE; never inside a retry loop; no automatic replay).
#   identity (4 episodes, empty plan set on MAPPO60C) -> identity-compare (FAILED: the chain STOPS, nothing else runs)
#   -> confirm (BIG_BANK_60C, DUO_HOLD_60C on the 96 panel-1 worlds) -> analysis against the read-only stored roots.
# Only /root/a67 is written. Stop between stages: touch /root/a67/root/STOP_A67.
source /root/a66/env_a66.sh
BASE=/root/a67   # after the env file: it sets BASE=/root/a66
export A67_HOST_BASE=$BASE
R=$BASE/root
W=${A67_WORKERS:?set A67_WORKERS}
C=$BASE/chain.out
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" ops/a67_plans_on_60c.py "$@" --final-root "$R" --a66-root /root/a66/root --a42-root /root/a66/refs/a42/root; }
charged() { python3 -c "import json,os;p='$R/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
[ -f "$C" ] && { echo "chain.out exists: not relaunching"; exit 1; }
log "chain start pid $$ workers $W"
echo $$ > "$BASE/chain.pid"
[ -f "$R/STOP_A67" ] && { log "STOP_A67 before identity"; exit 3; }
log "identity start"
run identity --workers 4 --chunk 4 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 4; }
run identity-compare >> "$BASE/identity.out" 2>&1 || { log "IDENTITY COMPARE ERROR"; exit 4; }
log "identity $(tail -1 "$BASE/identity.out" | cut -c1-600) charged $(charged)"
grep -q '"verdict": "PASS"' "$R/IDENTITY_A67.json" || { log "IDENTITY FAILED: STOP (nothing more is run)"; exit 5; }
[ -f "$R/STOP_A67" ] && { log "STOP_A67 before confirm"; exit 3; }
log "confirm start"
run confirm --workers "$W" --chunk 96 >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED"; exit 6; }
log "confirm done; charged $(charged)"
run analyse --out "$R" >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 8; }
log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-600)"
log "A67_ALL_DONE"
