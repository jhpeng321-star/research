#!/bin/bash
# A101 S-E1-RANK admission screen (Box 3; AMENDMENT 101): roots /root/a101adm_a, _b, _c in turn.
# Refuses unless every root is pinned and none has been launched. No waits / busy checks for older queues.
# Per root: mech -> admit -> bank ONCE each, 46 workers, no confirm. A failed root logs its rc; the queue continues.
# Written 2026-11-14 by the A101 prep agent; NOT started by it.
L=/root/a101q.log
ROOTS="/root/a101adm_a /root/a101adm_b /root/a101adm_c"
for B in $ROOTS; do
  [ -f $B/root/A80_CANDIDATES.json ] || { echo "$(date '+%F %T') REFUSED: $B not pinned" | tee -a $L; exit 1; }
  for S in mech admit bank confirm; do
    [ -f $B/chain_$S.out ] && { echo "$(date '+%F %T') REFUSED: $B already launched ($S)" | tee -a $L; exit 1; }
  done
done
echo "$(date '+%F %T') a101 queue pid $$ roots: $ROOTS" >> $L
for B in $ROOTS; do
  echo "$(date '+%F %T') $B start (ALLOCATION START)" >> $L
  ( export A80_BASE=$B A80_WORKERS=46; cd $B && bash $B/run_a80.sh mech > $B/nohup_mech.out 2>&1 && bash $B/run_a80.sh admit > $B/nohup_admit.out 2>&1 && bash $B/run_a80.sh bank > $B/nohup_bank.out 2>&1 )
  echo "$(date '+%F %T') $B end rc $?" >> $L
done
echo A101Q_DONE >> $L
