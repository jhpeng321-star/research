#!/bin/bash
# A96(b) S4-p20 admission (Box 3, /root/a96adm_s4). NOT started by the prep agent. No auto-wait. Refuses unless the
# root is pinned (A96_CANDIDATES.json) and not launched and the reference is sealed (/root/a96ref/root/A96_REF.json);
# skips ("A96Q_S4_SKIPPED busy") if another a94 / a82four / a96 SUMO job runs; then pin-ref (zero SUMO, if not yet)
# -> mech -> admit -> bank ONCE, 46 workers. Log /root/a96q_s4.log, last line A96Q_S4_DONE.
L=/root/a96q_s4.log
B=/root/a96adm_s4
# busy(): another a94 / a82four / a96 SUMO job on this box -> 0 (busy). Narrow patterns (lines naming this queue script itself, incl. its own subshells, excluded):
#   [/]root/a94/run_a94\.sh | [a]94queue\.sh             (A94/A95 60 % panel chain and its queue)
#   [/]root/a82four/run_a82\.sh                          (A82 part B four-arm chain)
#   [/]root/a96[a-z0-9_]*/run_a96\.sh                    (any A96 root's stage script, incl. other agents' a96p* roots)
#   [a]96q_[a-z0-9_]*\.sh                                (any other A96 queue script)
#   + /root/a82four/chain_confirm.out without A82F_CONFIRM_DONE or a FAILED line
busy() {
  local P F=/root/a82four/chain_confirm.out B=""
  P=$(ps -eo pid=,args= | awk -v me="$(basename "$0")" 'index($0, me) == 0' | grep -E '[/]root/a94/run_a94\.sh|[a]94queue\.sh|[/]root/a82four/run_a82\.sh|[/]root/a96[a-z0-9_]*/run_a96\.sh|[a]96q_[a-z0-9_]*\.sh' | cut -c1-160)
  [ -n "$P" ] && B="process: $(echo "$P" | head -3 | tr '\n' ';')"
  if [ -f $F ] && ! grep -qE 'A82F_CONFIRM_DONE|FAILED' $F; then B="$B $F has no finish marker"; fi
  [ -n "$B" ] && { echo "$(date '+%F %T') BUSY $B" >> $L; return 0; }
  return 1
}
[ -f $B/root/A96_CANDIDATES.json ] || { echo "$(date '+%F %T') REFUSED: $B not pinned" | tee -a $L; exit 1; }
[ -f $B/chain_mech.out ] && { echo "$(date '+%F %T') REFUSED: $B already launched" | tee -a $L; exit 1; }
[ -f /root/a96ref/root/A96_REF.json ] || { echo "$(date '+%F %T') REFUSED: reference not sealed" | tee -a $L; exit 1; }
busy && { echo "A96Q_S4_SKIPPED busy" >> $L; exit 0; }
if [ ! -f $B/root/A96_REFPIN.json ]; then bash $B/run_a96.sh pinref > $B/nohup_pinref.out 2>&1 || { echo "$(date '+%F %T') pin-ref failed" >> $L; echo A96Q_S4_DONE >> $L; exit 1; }; fi
echo "$(date '+%F %T') $B start (ALLOCATION START) pid $$" >> $L
export A96_WORKERS=46
( bash $B/run_a96.sh mech > $B/nohup_mech.out 2>&1 && bash $B/run_a96.sh admit > $B/nohup_admit.out 2>&1 && bash $B/run_a96.sh bank > $B/nohup_bank.out 2>&1 )
echo "$(date '+%F %T') $B end rc $?" >> $L
echo A96Q_S4_DONE >> $L
