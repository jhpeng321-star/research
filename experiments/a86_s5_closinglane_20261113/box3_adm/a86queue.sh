#!/bin/bash
# A86 S5 SUMO admission (Box 3, /root/a86adm): waits for the A89 queue to end, then mech -> admit -> bank ONCE, no retry,
# 46 workers. NO confirm. A failed stage ends the chain. Written 2026-11-14 by the prep agent; NOT started by it.
L=/root/a86queue.log
B=/root/a86adm
[ -f $B/root/A80_CANDIDATES.json ] || { echo "$(date '+%F %T') REFUSED: $B not pinned" >> $L; exit 1; }
[ -f $B/chain_mech.out ] && { echo "$(date '+%F %T') REFUSED: $B already launched" >> $L; exit 1; }
echo "$(date '+%F %T') a86 queue pid $$ waiting for A89Q_DONE" >> $L
while ! grep -q A89Q_DONE /root/a89queue.log 2>/dev/null; do sleep 30; done
echo "$(date '+%F %T') $B start (ALLOCATION START)" >> $L
export A80_BASE=$B A80_WORKERS=46
( cd $B && bash $B/run_a80.sh mech > $B/nohup_mech.out 2>&1 && bash $B/run_a80.sh admit > $B/nohup_admit.out 2>&1 && bash $B/run_a80.sh bank > $B/nohup_bank.out 2>&1 )
echo "$(date '+%F %T') $B end rc $?" >> $L
echo A86Q_DONE >> $L
