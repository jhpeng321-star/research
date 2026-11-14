#!/bin/bash
# A96 part (d) SUMO chain (Box 3, /root/a96pd). Started BY HAND only; it waits for nothing. Refuses unless setup +
# stage + bank(s) + pin are done (shas from /root/a96pd/PINNED.env, written by run_a96.sh pin; the panel sha must
# equal the registered one below). Then mech -> confirm (+ analyse) ONCE, no retry, 46 workers. Any failure writes
# /root/a96pd/CHAIN_STOPPED. Log /root/a96pd_queue.log, last line A96PD_DONE. Written 2026-11-14 by the prep
# agent; NOT started by it.
L=/root/a96pd_queue.log
B=/root/a96pd
REG_PANEL=3c879f6ff47fa75ebbf5f5055f4ed35ad7e54be96a704dffa743287b86aa6485
chainstop() { [ -f $B/CHAIN_STOPPED ] || printf '%s\n' "{\"stage\": \"queue\", \"time\": \"$(date "+%F %T")\", \"reason\": \"$*\"}" > $B/CHAIN_STOPPED; echo "$(date '+%F %T') CHAIN_STOPPED $*" >> $L; echo A96PD_DONE >> $L; exit 1; }
refuse() { echo "$(date '+%F %T') REFUSED: $*" >> $L; echo A96PD_DONE >> $L; exit 1; }
[ -f $B/SETUP_OK ] && grep -q PIN_OK $B/chain_pin.out 2>/dev/null && [ -f $B/PINNED.env ] || refuse "$B not pinned"
[ -f $B/chain_mech.out ] && refuse "$B mech already launched"
[ -f $B/CHAIN_STOPPED ] && refuse "$B/CHAIN_STOPPED present"
source $B/PINNED.env
[ "$A96_PANEL_SHA256" = "$REG_PANEL" ] || refuse "pinned panel $A96_PANEL_SHA256 != registered $REG_PANEL"
[ ${#A96_BANK_SHA256} = 64 ] && { [ ${#A96_HAND_SHA256} = 64 ] || [ "$A96_HAND_SHA256" = none ]; } || refuse "PINNED.env malformed"
export A96_WORKERS=46 A96_BANK_SHA256 A96_HAND_SHA256 A96_PANEL_SHA256
echo "$(date '+%F %T') a96pd queue pid $$ bank $A96_BANK_SHA256 hand $A96_HAND_SHA256 panel $A96_PANEL_SHA256" >> $L
echo "$(date '+%F %T') $B mech start (ALLOCATION START)" >> $L
bash $B/run_a96.sh mech > $B/nohup_mech.out 2>&1; r=$?
echo "$(date '+%F %T') mech end rc $r" >> $L
[ $r -eq 0 ] && grep -q A96_MECH_DONE $B/chain_mech.out && [ ! -f $B/CHAIN_STOPPED ] || chainstop "mech rc $r (no A96_MECH_DONE)"
echo "$(date '+%F %T') confirm start" >> $L
bash $B/run_a96.sh confirm > $B/nohup_confirm.out 2>&1; r=$?
echo "$(date '+%F %T') confirm end rc $r" >> $L
[ $r -eq 0 ] && grep -q A96_CONFIRM_DONE $B/chain_confirm.out && [ ! -f $B/CHAIN_STOPPED ] || chainstop "confirm rc $r (no A96_CONFIRM_DONE)"
echo "$(date '+%F %T') $(grep 'analysis:' $B/chain_confirm.out | tail -1 | cut -c1-900)" >> $L
echo A96PD_DONE >> $L
