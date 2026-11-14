#!/bin/bash
# A98 SUMO chain (Box 3, /root/a98). Started BY HAND. Waits for A94Q_DONE in /root/a94queue.log (refuses if the A94 chain
# stopped: /root/a94/CHAIN_STOPPED or no A94_CONFIRM_DONE), refuses unless pinned, then waits while another SUMO stage
# script runs on this box (busy patterns below; queue wait-loops are NOT counted, so no deadlock), then mech -> confirm
# (+ analysis) ONCE, no retry, 46 workers. Any failure writes /root/a98/CHAIN_STOPPED. Log /root/a98queue.log, last line
# A98Q_DONE. Written 2026-11-14 by the prep agent; NOT started by it.
L=/root/a98queue.log
B=/root/a98
done_() { echo "$(date '+%F %T') $*" >> $L; echo A98Q_DONE >> $L; exit 1; }
chainstop() { [ -f $B/CHAIN_STOPPED ] || printf '%s\n' "{\"stage\": \"queue\", \"time\": \"$(date "+%F %T")\", \"reason\": \"$*\"}" > $B/CHAIN_STOPPED; done_ "CHAIN_STOPPED $*"; }
# busy(): a SUMO stage script of another package running now (stage scripts only; their queue loops excluded)
busy() { ps -eo pid=,args= | grep -E '[/]root/a94/run_a94\.sh|[/]root/a96[a-z0-9_]*/run_a96\.sh|[/]root/a82four/run_a82\.sh|[/]root/a95adm[a-z0-9_]*/run_' | grep -v "^ *$$ " | cut -c1-200; }
[ -f $B/SETUP_OK ] && grep -q PIN_OK $B/chain_pin.out 2>/dev/null || done_ "REFUSED: $B not pinned"
[ -f $B/chain_mech.out ] && done_ "REFUSED: $B mech already launched"
[ -f $B/CHAIN_STOPPED ] && done_ "REFUSED: $B/CHAIN_STOPPED present"
echo "$(date '+%F %T') a98 queue pid $$ waiting for A94Q_DONE" >> $L
while ! grep -q A94Q_DONE /root/a94queue.log 2>/dev/null; do sleep 30; done
echo "$(date '+%F %T') A94Q_DONE seen" >> $L
[ -f /root/a94/CHAIN_STOPPED ] && done_ "REFUSED: /root/a94/CHAIN_STOPPED present (A94 chain failed)"
grep -q A94_CONFIRM_DONE /root/a94/chain_confirm.out 2>/dev/null || done_ "REFUSED: /root/a94 has no A94_CONFIRM_DONE"
while [ -n "$(busy)" ]; do echo "$(date '+%F %T') busy: $(busy | head -2 | tr '\n' ';')" >> $L; sleep 60; done
export A98_WORKERS=46
echo "$(date '+%F %T') $B mech start (ALLOCATION START)" >> $L
bash $B/run_a98.sh mech > $B/nohup_mech.out 2>&1; r=$?
echo "$(date '+%F %T') mech end rc $r" >> $L
[ $r -eq 0 ] && grep -q A98_MECH_DONE $B/chain_mech.out && [ ! -f $B/CHAIN_STOPPED ] || chainstop "mech rc $r (no A98_MECH_DONE)"
echo "$(date '+%F %T') confirm start" >> $L
bash $B/run_a98.sh confirm > $B/nohup_confirm.out 2>&1; r=$?
echo "$(date '+%F %T') confirm end rc $r" >> $L
[ $r -eq 0 ] && grep -q A98_CONFIRM_DONE $B/chain_confirm.out && [ ! -f $B/CHAIN_STOPPED ] || chainstop "confirm rc $r (no A98_CONFIRM_DONE)"
echo "$(date '+%F %T') $(grep 'analysis:' $B/chain_confirm.out | tail -1 | cut -c1-900)" >> $L
echo A98Q_DONE >> $L
