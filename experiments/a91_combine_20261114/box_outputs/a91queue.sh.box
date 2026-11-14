#!/bin/bash
# A91 SUMO chain (Box 3, /root/a91): waits for A86Q_DONE in /root/a86queue.log and, if /root/a85queue.log exists, for
# A85Q_DONE in it; if /root/a85queue.log does not exist by 01:40 box-local time it stops waiting for A85 and proceeds.
# Then mech -> confirm (+ analyse) ONCE, no retry, 46 workers. Any failure writes /root/a91/CHAIN_STOPPED (run_a91.sh
# writes it itself; this script also writes one if a stage ends without its DONE line). Log /root/a91queue.log, last
# line A91Q_DONE. Written 2026-11-14 by the prep agent; NOT started by it.
L=/root/a91queue.log
B=/root/a91
export A91_WORKERS=46 A91_TURNS_SHA256=f10e7689d4fd32d6f204324fc02c8d48669d7cf89fc08327b84dfc8c36b2aff2 A91_PANEL_SHA256=e5d51c5ad882c161bf134e080474aae0892a74c3e98a53a32433d8903042d386
chainstop() { [ -f $B/CHAIN_STOPPED ] || printf '%s\n' "{\"stage\": \"queue\", \"time\": \"$(date "+%F %T")\", \"reason\": \"$*\"}" > $B/CHAIN_STOPPED; echo "$(date '+%F %T') CHAIN_STOPPED $*" >> $L; echo A91Q_DONE >> $L; exit 1; }
[ -f $B/SETUP_OK ] && grep -q PIN_OK $B/chain_pin.out 2>/dev/null || { echo "$(date '+%F %T') REFUSED: $B not pinned" >> $L; exit 1; }
[ -f $B/chain_mech.out ] && { echo "$(date '+%F %T') REFUSED: $B mech already launched" >> $L; exit 1; }
[ -f $B/CHAIN_STOPPED ] && { echo "$(date '+%F %T') REFUSED: $B/CHAIN_STOPPED present" >> $L; exit 1; }
DEADLINE=$(date -d "today 01:40" +%s)
[ "$(date +%s)" -gt $(( DEADLINE + 43200 )) ] && DEADLINE=$(date -d "tomorrow 01:40" +%s)   # started after 13:40: next 01:40
echo "$(date '+%F %T') a91 queue pid $$ waiting for A86Q_DONE (+ A85Q_DONE if /root/a85queue.log exists; a85 wait skipped if absent at $(date -d @$DEADLINE '+%F %T'))" >> $L
while ! grep -q A86Q_DONE /root/a86queue.log 2>/dev/null; do sleep 30; done
echo "$(date '+%F %T') A86Q_DONE seen" >> $L
while true; do
  if [ -f /root/a85queue.log ]; then
    grep -q A85Q_DONE /root/a85queue.log && { echo "$(date '+%F %T') A85Q_DONE seen" >> $L; break; }
  elif [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "$(date '+%F %T') /root/a85queue.log absent at the 01:40 deadline: not waiting for A85" >> $L; break
  fi
  sleep 30
done
echo "$(date '+%F %T') $B mech start (ALLOCATION START)" >> $L
bash $B/run_a91.sh mech > $B/nohup_mech.out 2>&1; r=$?
echo "$(date '+%F %T') a91 mech end rc $r" >> $L
[ $r -eq 0 ] && grep -q A91_MECH_DONE $B/chain_mech.out && [ ! -f $B/CHAIN_STOPPED ] || chainstop "mech rc $r (no A91_MECH_DONE)"
echo "$(date '+%F %T') a91 confirm start" >> $L
bash $B/run_a91.sh confirm > $B/nohup_confirm.out 2>&1; r=$?
echo "$(date '+%F %T') a91 confirm end rc $r" >> $L
[ $r -eq 0 ] && grep -q A91_CONFIRM_DONE $B/chain_confirm.out && [ ! -f $B/CHAIN_STOPPED ] || chainstop "confirm rc $r (no A91_CONFIRM_DONE)"
echo "$(date '+%F %T') $(grep 'analysis:' $B/chain_confirm.out | tail -1 | cut -c1-600)" >> $L
echo A91Q_DONE >> $L
