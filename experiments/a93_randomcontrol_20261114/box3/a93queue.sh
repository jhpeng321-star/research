#!/bin/bash
# A93 random-draw control SUMO admission (Box 3, /root/a93adm; AMENDMENT 93): refuses unless the list is pinned and
# not yet launched; waits for A85Q_DONE in /root/a85queue.log; then SKIPS (one log line, exit) if an a82 chain is busy:
#   (a) any process whose command line contains "a82" (ps -eo args | grep '[a]82': run_a82.sh of a82four / a82p20,
#       the a82 driver, a82queue.sh), or
#   (b) /root/a82four/chain_confirm.out exists without a finish marker (A82F_CONFIRM_DONE or a FAILED line).
# Otherwise mech -> admit -> bank ONCE, no retry, 46 workers, NO confirm. A failed stage ends the chain.
# Written 2026-11-14 by the A93 prep agent; NOT started by it.
L=/root/a93queue.log
B=/root/a93adm
[ -f $B/root/A80_CANDIDATES.json ] || { echo "$(date '+%F %T') REFUSED: $B/root/A80_CANDIDATES.json missing (run pin first)" | tee -a $L; exit 1; }
[ -f $B/chain_mech.out ] && { echo "$(date '+%F %T') REFUSED: $B already launched" | tee -a $L; exit 1; }
echo "$(date '+%F %T') a93 queue pid $$ waiting for A85Q_DONE" >> $L
while ! grep -q A85Q_DONE /root/a85queue.log 2>/dev/null; do sleep 30; done
[ -f $B/root/A80_CANDIDATES.json ] || { echo "$(date '+%F %T') REFUSED after wait: not pinned" >> $L; exit 1; }
[ -f $B/chain_mech.out ] && { echo "$(date '+%F %T') REFUSED after wait: $B already launched" >> $L; exit 1; }
BUSY=""
P=$(ps -eo pid,args | grep '[a]82' | cut -c1-160)
[ -n "$P" ] && BUSY="a82 process: $(echo "$P" | head -3 | tr '\n' ';')"
F=/root/a82four/chain_confirm.out
if [ -f $F ] && ! grep -qE 'A82F_CONFIRM_DONE|FAILED' $F; then BUSY="$BUSY $F has no finish marker"; fi
if [ -n "$BUSY" ]; then echo "$(date '+%F %T') $BUSY" >> $L; echo "A93Q_SKIPPED busy" >> $L; exit 0; fi
echo "$(date '+%F %T') $B start (ALLOCATION START)" >> $L
export A80_BASE=$B A80_WORKERS=46
( cd $B && bash $B/run_a80.sh mech > $B/nohup_mech.out 2>&1 && bash $B/run_a80.sh admit > $B/nohup_admit.out 2>&1 && bash $B/run_a80.sh bank > $B/nohup_bank.out 2>&1 )
echo "$(date '+%F %T') $B end rc $?" >> $L
echo A93Q_DONE >> $L
