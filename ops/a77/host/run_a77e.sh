#!/bin/bash
# AMENDMENT 77 evaluation chain (root; detached; launched ONCE; never in a retry loop; no automatic replay):
#   bash /root/a77e/run_a77e.sh <policy_00084.pt> <MAPPO20C sha256 as recorded in REGISTRATION.md> \
#        <training BOUND_00084.json> <training PLAN.json> [ARMS]
#   ARMS default U0,MAPPO_CT,MAPPO20C; registered fallback U0,MAPPO20C (only if MAPPO_CT could not be retrieved).
#   bind MAPPO20C (refused unless the file = the sha = the iteration-84 receipt of the registered PLAN 57f51364)
#   -> identity (4 episodes) -> identity-compare vs the stored A50 identity wave (FAILED: STOP, nothing else runs)
#   -> confirm (the arms on 997600-997695) -> analyse (A77_RESULTS_D2800_p20.json). Only /root/a77e is written.
# Stop between stages: touch /root/a77e/root/STOP_A77.
set -uo pipefail
BASE=/root/a77e
if [ "${A77E_CHAIN:-}" != 1 ]; then
  [ -f "$BASE/chain.out" ] && { echo "chain.out exists: not relaunching"; exit 1; }
  [ "$#" -ge 4 ] || { echo "usage: run_a77e.sh POLICY SHA RECEIPT PLAN [ARMS]"; exit 2; }
  A77E_CHAIN=1 setsid -f nohup bash "$0" "$(readlink -f "$1")" "$2" "$(readlink -f "$3")" "$(readlink -f "$4")" \
    "${5:-U0,MAPPO_CT,MAPPO20C}" < /dev/null > "$BASE/run.nohup" 2>&1
  echo "A77 evaluation chain started detached; follow $BASE/chain.out"; exit 0
fi
POL=$1 SHA=$2 REC=$3 PLAN=$4 ARMS=$5
source "$BASE/env_a77e.sh"
R=$BASE/root
W=${A77_WORKERS:-46}
C=$BASE/chain.out
D=ops/a77_mappo20c.py
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" $D "$@" --final-root "$R"; }
charged() { python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$R/CHARGES.jsonl')))" 2>/dev/null || echo 0; }
log "chain start pid $$ workers $W arms $ARMS"
echo $$ > "$BASE/chain.pid"
[ -f "$BASE/SETUP_OK" ] || { log "SETUP_OK missing"; exit 2; }
FREE=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge 12 ] || { log "free disk ${FREE}G < 12G: not launching"; exit 2; }
if [ ! -f "$R/BOUND_A77_MAPPO20C.json" ]; then
  run bind --which MAPPO20C --policy "$POL" --sha "$SHA" --train-receipt "$REC" --train-plan "$PLAN" >> "$C" 2>&1 \
    || { log "BIND MAPPO20C REFUSED"; exit 3; }
fi
case ",$ARMS," in *,MAPPO_CT,*) [ -f "$R/BOUND_A77_MAPPO_CT.json" ] || { log "MAPPO_CT not bound: STOP (or use the registered fallback U0,MAPPO20C)"; exit 3; } ;; esac
run dry --arms "$ARMS" > "$BASE/dry_go.json" 2>&1 || { log "DRY FAILED"; exit 3; }
[ -f "$R/STOP_A77" ] && { log "STOP_A77 before identity"; exit 4; }
log "identity start"
run identity --workers 4 --chunk 4 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 5; }
run identity-compare --a50-root /root/a66/refs/a50/root >> "$BASE/identity.out" 2>&1 || { log "IDENTITY COMPARE ERROR"; exit 5; }
log "identity $(tail -1 "$BASE/identity.out" | cut -c1-400) charged $(charged)"
grep -q '"verdict": "PASS"' "$R/IDENTITY_A77.json" || { log "IDENTITY FAILED: STOP (nothing more is run)"; exit 6; }
[ -f "$R/STOP_A77" ] && { log "STOP_A77 before confirm"; exit 4; }
log "confirm start"
run confirm --arms "$ARMS" --workers "$W" --chunk 96 >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED"; exit 7; }
log "confirm done; charged $(charged)"
run analyse --arms "$ARMS" --out "$R" >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 8; }
log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-900)"
(cd "$R" && find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$BASE/SHA256SUMS_A77_ROOT")
log "A77_EVAL_DONE"
