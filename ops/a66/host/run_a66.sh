#!/bin/bash
# AMENDMENT 66 host chain (detached, launched ONCE; never inside a retry loop; no automatic replay).
#   per panel (p2 first, then p1): identity (4 episodes) -> identity-compare (FAILED: that panel is not run and the
#   chain STOPS) -> confirm (the panel's MAPPO arms on its 96 worlds) -> analysis against the read-only stored roots.
# Only /root/a66 is written. Stop between stages: touch /root/a66/root/STOP_A66.
BASE=/root/a66
source "$BASE/env_a66.sh"
R=$BASE/root
F=$BASE/refs
W=${A66_WORKERS:?set A66_WORKERS}
C=$BASE/chain.out
D=ops/a66_mappo60c.py
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" $D "$@" --final-root "$R"; }
charged() { python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$R/CHARGES.jsonl')))"; }
[ -f "$C" ] && { echo "chain.out exists: not relaunching"; exit 1; }
log "chain start pid $$ workers $W"
echo $$ > "$BASE/chain.pid"
for P in p2 p1; do
  if [ $P = p1 ]; then REFS="--a42-root $F/a42/root"; else REFS="--a50-root $F/a50/root --a52-root $F/a52/root --a54-root $F/a54/root"; fi
  [ -f "$R/STOP_A66" ] && { log "STOP_A66 before identity $P"; exit 3; }
  log "identity $P start"
  run identity --panel $P --workers 4 --chunk 4 >> "$BASE/identity_$P.out" 2>&1 || { log "IDENTITY RUN FAILED $P"; exit 4; }
  run identity-compare --panel $P $REFS >> "$BASE/identity_$P.out" 2>&1 || { log "IDENTITY COMPARE ERROR $P"; exit 4; }
  log "identity $P $(tail -1 "$BASE/identity_$P.out") charged $(charged)"
  grep -q '"verdict": "PASS"' "$R/IDENTITY_A66_$P.json" || { log "IDENTITY FAILED $P: STOP (nothing more is run)"; exit 5; }
  [ -f "$R/STOP_A66" ] && { log "STOP_A66 before confirm $P"; exit 3; }
  log "confirm $P start"
  run confirm --panel $P --workers $W --chunk 96 >> "$BASE/confirm_$P.out" 2>&1 || { log "CONFIRM FAILED $P"; exit 6; }
  log "confirm $P done; charged $(charged)"
  run analyse --panel $P $REFS --out "$R" >> "$BASE/analyse_$P.out" 2>&1 || { log "ANALYSE FAILED $P"; exit 8; }
  log "analysis $P: $(tail -1 "$BASE/analyse_$P.out" | cut -c1-400)"
done
log "A66_ALL_DONE"
