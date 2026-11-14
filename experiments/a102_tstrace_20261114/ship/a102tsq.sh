#!/bin/bash
# A102TS SUMO queue (Box 3). Started BY HAND only, ONCE; refuses relaunch (QUEUE_STARTED marker). Runs the single A102TS
# stage "run" (2 episodes x 1,200 = 2,400 sim-s, world 988700, MAPPO60C + S5 bank 577a0564 on MAPPO60C 2abed317, raw
# kept, 2 workers), then the zero-SUMO post-run check. Log /root/a102tsq.log ("ALLOCATION START"; last line A102TS_DONE).
# No retry. Written 2026-11-14 by the prep agent; NOT started by it.
L=/root/a102tsq.log
B=/root/a102ts
finish() { echo A102TS_DONE >> $L; exit "$1"; }
refuse() { echo "$(date '+%F %T') REFUSED: $*" >> $L; finish 1; }
if [ -e $B/QUEUE_STARTED ]; then echo "$(date '+%F %T') REFUSED: $B/QUEUE_STARTED exists (the queue runs once)" >> $L; exit 1; fi
date '+%F %T' > $B/QUEUE_STARTED || exit 1
[ -f $B/SETUP_OK ] || refuse "no SETUP_OK"
grep -q PREPARE_OK $B/chain_prepare.out 2>/dev/null || refuse "prepare not done"
grep -q DRY_OK $B/chain_dry.out 2>/dev/null || refuse "dry not OK"
grep -q PREPHYSICS_OK $B/chain_prephysics.out 2>/dev/null || refuse "prephysics not OK"
[ -f $B/chain_run.out ] && refuse "run already launched"
[ -f $B/root/CHARGES.jsonl ] && refuse "root already charged"
echo "$(date '+%F %T') a102ts queue pid $$ world 988700 arms MAPPO60C BANK_SEARCHED(577a0564) executor 2abed317 cap 2400 workers 2" >> $L
echo "$(date '+%F %T') $B run start (ALLOCATION START)" >> $L
bash $B/run_a102ts.sh run > $B/nohup_run.out 2>&1
r=$?
CH=$(python3 -c "import json,os;p='$B/root/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)")
echo "$(date '+%F %T') a102ts run end rc $r charged $CH" >> $L
grep -q A102TS_RUN_DONE $B/chain_run.out 2>/dev/null || { echo "$(date '+%F %T') RUN FAILED (no A102TS_RUN_DONE); check not run" >> $L; finish 8; }
bash $B/run_a102ts.sh check > /dev/null 2>&1
c=$?
echo "$(date '+%F %T') check rc $c: $(grep -E 'MATCH|MISMATCH|executor_sha256_loaded|A102TS_CHECK ' $B/check.out | tr -s ' ' | tr '\n' ';' | cut -c1-1800)" >> $L
finish 0
