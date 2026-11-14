#!/bin/bash
# A85 S4 SUMO admission (Box 3, /root/a85adm): refuses unless the S4 list is pinned; waits for the A89 queue to end and,
# if the A86 queue was started (/root/a86queue.log exists), for A86Q_DONE; then mech -> admit -> bank ONCE, no retry,
# 46 workers. NO confirm. A failed stage ends the chain. Written 2026-11-14 by the prep agent; NOT started by it.
L=/root/a85queue.log
B=/root/a85adm
[ -f $B/root/A80_CANDIDATES.json ] || { echo "$(date '+%F %T') REFUSED: $B/root/A80_CANDIDATES.json missing (run pin first)" | tee -a $L; exit 1; }
[ -f $B/chain_mech.out ] && { echo "$(date '+%F %T') REFUSED: $B already launched" | tee -a $L; exit 1; }
echo "$(date '+%F %T') a85 queue pid $$ waiting for A89Q_DONE (+ A86Q_DONE if the a86 queue exists)" >> $L
while ! grep -q A89Q_DONE /root/a89queue.log 2>/dev/null || { [ -f /root/a86queue.log ] && ! grep -q A86Q_DONE /root/a86queue.log; }; do sleep 30; done
[ -f $B/root/A80_CANDIDATES.json ] || { echo "$(date '+%F %T') REFUSED after wait: not pinned" >> $L; exit 1; }
echo "$(date '+%F %T') $B start (ALLOCATION START)" >> $L
export A80_BASE=$B A80_WORKERS=46
( cd $B && bash $B/run_a80.sh mech > $B/nohup_mech.out 2>&1 && bash $B/run_a80.sh admit > $B/nohup_admit.out 2>&1 && bash $B/run_a80.sh bank > $B/nohup_bank.out 2>&1 )
echo "$(date '+%F %T') $B end rc $?" >> $L
echo A85Q_DONE >> $L
