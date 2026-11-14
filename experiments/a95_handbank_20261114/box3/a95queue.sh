#!/bin/bash
# A95(a) hand-written plan admission (Box 3; AMENDMENT 95): roots /root/a95adm then /root/a95adm_b (only if it exists).
# Refuses unless every existing root is pinned and not launched; waits for A93Q_DONE or A93Q_SKIPPED in
# /root/a93queue.log; skips (one line + "A95Q_SKIPPED busy", exit) if an a82 chain is busy:
#   (a) a process whose command line matches '[r]un_a82\.sh|[a]82four' (the a82 runner of a82four / a82p20 stays
#       alive for the whole stage; anything under /root/a82four), or
#   (b) /root/a82four/chain_confirm.out exists without A82F_CONFIRM_DONE or a FAILED line.
# The busy check is repeated before the second root. Per root: mech -> admit -> bank ONCE, 46 workers, no confirm.
# Written 2026-11-14 by the A95 prep agent; NOT started by it.
L=/root/a95queue.log
ROOTS="/root/a95adm"; [ -d /root/a95adm_b ] && ROOTS="$ROOTS /root/a95adm_b"
chk() { for B in $ROOTS; do
  [ -f $B/root/A80_CANDIDATES.json ] || { echo "$(date '+%F %T') REFUSED$1: $B not pinned" | tee -a $L; return 1; }
  [ -f $B/chain_mech.out ] && { echo "$(date '+%F %T') REFUSED$1: $B already launched" | tee -a $L; return 1; }
done; return 0; }
busy() {
  local P F=/root/a82four/chain_confirm.out BUSY=""
  P=$(ps -eo pid,args | grep -E '[r]un_a82\.sh|[a]82four' | cut -c1-160)
  [ -n "$P" ] && BUSY="a82 process: $(echo "$P" | head -3 | tr '\n' ';')"
  if [ -f $F ] && ! grep -qE 'A82F_CONFIRM_DONE|FAILED' $F; then BUSY="$BUSY $F has no finish marker"; fi
  [ -n "$BUSY" ] && { echo "$(date '+%F %T') $BUSY" >> $L; echo "A95Q_SKIPPED busy" >> $L; return 0; }
  return 1; }
chk "" || exit 1
echo "$(date '+%F %T') a95 queue pid $$ roots: $ROOTS; waiting for A93Q_DONE / A93Q_SKIPPED" >> $L
while ! grep -qE 'A93Q_DONE|A93Q_SKIPPED' /root/a93queue.log 2>/dev/null; do sleep 30; done
chk " after wait" || exit 1
for B in $ROOTS; do
  busy && exit 0
  echo "$(date '+%F %T') $B start (ALLOCATION START)" >> $L
  ( export A80_BASE=$B A80_WORKERS=46; cd $B && bash $B/run_a80.sh mech > $B/nohup_mech.out 2>&1 && bash $B/run_a80.sh admit > $B/nohup_admit.out 2>&1 && bash $B/run_a80.sh bank > $B/nohup_bank.out 2>&1 )
  echo "$(date '+%F %T') $B end rc $?" >> $L
done
echo A95Q_DONE >> $L
