#!/bin/bash
# A99 PANEL SUMO chain (Box 4, /root/a99panel). Started BY HAND only (coordinator); waits for nothing; refuses if it was
# ever launched before (mkdir lock), if setup / stage / pin are not done, or if the pinned shas differ from the registered
# ones below. mech (8 episodes) -> confirm (760 episodes) + analysis, each ONCE, no retry, 46 workers. Any failure writes
# /root/a99panel/CHAIN_STOPPED. Log /root/a99panelq.log, last line A99PANEL_DONE. Written by the prep agent; NOT started.
L=/root/a99panelq.log
B=/root/a99panel
REG_PANEL=6c161a56e1c913e35441570774f1df6ee62999a0ca9e11b27a309ab3b0dfc5ff
REG_S4=682c01ae1deb28c774701ff0a5102cd77c2addf2ca302e4e42a90409b7302ab2
REG_HAND=59acb7471c51da728ffe7e1f0600082a8e6486a0455108a21df41d7b851bd946
refuse() { echo "$(date '+%F %T') REFUSED: $*" >> $L; echo A99PANEL_DONE >> $L; exit 1; }
chainstop() { [ -f $B/CHAIN_STOPPED ] || printf '%s\n' "{\"stage\": \"queue\", \"time\": \"$(date "+%F %T")\", \"reason\": \"$*\"}" > $B/CHAIN_STOPPED; echo "$(date '+%F %T') CHAIN_STOPPED $*" >> $L; echo A99PANEL_DONE >> $L; exit 1; }
mkdir /root/a99panelq.launched 2>/dev/null || refuse "already launched once (/root/a99panelq.launched exists)"
[ -f $B/SETUP_OK ] && grep -q STAGE_OK $B/chain_stage.out 2>/dev/null && grep -q PIN_OK $B/chain_pin.out 2>/dev/null && [ -f $B/PINNED.env ] || refuse "$B not set up / staged / pinned"
[ -e $B/chain_mech.out ] && refuse "$B mech already launched"
[ -e $B/chain_confirm.out ] && refuse "$B confirm already launched"
[ -f $B/CHAIN_STOPPED ] && refuse "$B/CHAIN_STOPPED present"
[ -e $B/root/CHARGES.jsonl ] && refuse "$B/root/CHARGES.jsonl exists (something already ran)"
source $B/PINNED.env
[ "$A99P_PANEL_SHA256" = "$REG_PANEL" ] || refuse "pinned panel $A99P_PANEL_SHA256 != registered $REG_PANEL"
[ "$A99P_S4_BANK_SHA256" = "$REG_S4" ] && [ "$A99P_HAND_BANK_SHA256" = "$REG_HAND" ] || refuse "pinned bank shas differ from the registered ones"
export A99P_WORKERS=46
echo "$(date '+%F %T') a99panel queue pid $$ panel $A99P_PANEL_SHA256 S4 $A99P_S4_BANK_SHA256 hand $A99P_HAND_BANK_SHA256" >> $L
echo "$(date '+%F %T') ALLOCATION START A99 PANEL D2800_p20 984400-984591: 4 arms x 192 x 1,200 = 921,600 sim-s cap (mech 9,600 inside)" >> $L
echo "$(date '+%F %T') mech start" >> $L
bash $B/run_a99p.sh mech > $B/nohup_mech.out 2>&1; r=$?
echo "$(date '+%F %T') mech end rc $r" >> $L
[ $r -eq 0 ] && grep -q A99P_MECH_DONE $B/chain_mech.out && [ ! -f $B/CHAIN_STOPPED ] || chainstop "mech rc $r (no A99P_MECH_DONE)"
echo "$(date '+%F %T') confirm start" >> $L
bash $B/run_a99p.sh confirm > $B/nohup_confirm.out 2>&1; r=$?
echo "$(date '+%F %T') confirm end rc $r" >> $L
[ $r -eq 0 ] && grep -q A99P_CONFIRM_DONE $B/chain_confirm.out && [ ! -f $B/CHAIN_STOPPED ] || chainstop "confirm rc $r (no A99P_CONFIRM_DONE)"
echo "$(date '+%F %T') $(grep 'analysis:' $B/chain_confirm.out | tail -1 | cut -c1-900)" >> $L
echo A99PANEL_DONE >> $L
