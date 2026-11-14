#!/bin/bash
# AMENDMENT 54 host chain (detached, launched ONCE; never inside a retry loop; no automatic replay).
#   1. MAPPO60B on the 96 A50 confirmation worlds (one dispatch; no identity re-run: registered deviation)
#   2. analysis A54_RESULTS_D2800_p60.json against read-only copies of the A50 root (a50ref) and A52 root (a52ref)
# Only /root/a54 is written. Stop between stages: touch /root/a54/root/STOP_A54.
BASE=/root/a54
source "$BASE/env_a54.sh"
R=$BASE/root
W=${A54_WORKERS:?set A54_WORKERS}
C=$BASE/chain.out
D=ops/a54_mappo60b.py
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" $D "$@" --final-root "$R"; }
[ -f "$C" ] && { echo "chain.out exists: not relaunching"; exit 1; }
log "chain start pid $$ workers $W"
echo $$ > "$BASE/chain.pid"

[ -f "$R/STOP_A54" ] && { log "STOP_A54 before confirm"; exit 3; }
log "confirm MAPPO60B start"
run confirm --workers $W --chunk 97 >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED"; exit 6; }
log "confirm done; charged $(python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$R/CHARGES.jsonl')))")"

run analyse --a50-root "$BASE/a50ref/root" --a52-root "$BASE/a52ref/root" --out "$R" >> "$BASE/analyse.out" 2>&1 \
  || { log "ANALYSE FAILED"; exit 8; }
log "done: $(tail -1 "$BASE/analyse.out")"
log "A54_ALL_DONE"
