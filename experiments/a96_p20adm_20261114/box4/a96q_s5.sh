#!/bin/bash
# A96(b) S5-p20 admission (BOX 4, /root/a96adm_s5). NOT started by the prep agent. No auto-wait. Refuses unless the
# root is pinned (A96_CANDIDATES.json) and not launched and the reference is sealed (/root/a96ref/root/A96_REF.json);
# skips ("A96Q_S5_SKIPPED busy") if another A77 / A96 / A68-69 SUMO job runs; then pin-ref (zero SUMO, if not yet)
# -> mech -> admit -> bank ONCE, 46 workers. Log /root/a96q_s5.log, last line A96Q_S5_DONE.
L=/root/a96q_s5.log
B=/root/a96adm_s5
# busy() (Box 4): another SUMO job on this box -> 0 (busy). Narrow patterns (lines naming this queue script excluded):
#   [g]o_train\.sh | [w]ms\.control\.mappo\.train        (A77 MAPPO20C training chain / trainer, incl. sanity_post)
#   [/]root/a96[a-z0-9_]*/run_a96\.sh | [a]96q_[a-z0-9_]*\.sh   (any A96 stage script / other A96 queue)
#   [/]root/a6[89]/run_a6[89]\.sh                        (A68 / A69 chains present on this box)
busy() {
  local P B=""
  P=$(ps -eo pid=,args= | awk -v me="$(basename "$0")" 'index($0, me) == 0' | grep -E '[g]o_train\.sh|[w]ms\.control\.mappo\.train|[/]root/a96[a-z0-9_]*/run_a96\.sh|[a]96q_[a-z0-9_]*\.sh|[/]root/a6[89]/run_a6[89]\.sh' | cut -c1-160)
  [ -n "$P" ] && B="process: $(echo "$P" | head -3 | tr '\n' ';')"
  [ -n "$B" ] && { echo "$(date '+%F %T') BUSY $B" >> $L; return 0; }
  return 1
}
[ -f $B/root/A96_CANDIDATES.json ] || { echo "$(date '+%F %T') REFUSED: $B not pinned" | tee -a $L; exit 1; }
[ -f $B/chain_mech.out ] && { echo "$(date '+%F %T') REFUSED: $B already launched" | tee -a $L; exit 1; }
[ -f /root/a96ref/root/A96_REF.json ] || { echo "$(date '+%F %T') REFUSED: reference not sealed" | tee -a $L; exit 1; }
busy && { echo "A96Q_S5_SKIPPED busy" >> $L; exit 0; }
if [ ! -f $B/root/A96_REFPIN.json ]; then bash $B/run_a96.sh pinref > $B/nohup_pinref.out 2>&1 || { echo "$(date '+%F %T') pin-ref failed" >> $L; echo A96Q_S5_DONE >> $L; exit 1; }; fi
echo "$(date '+%F %T') $B start (ALLOCATION START) pid $$" >> $L
export A96_WORKERS=46
( bash $B/run_a96.sh mech > $B/nohup_mech.out 2>&1 && bash $B/run_a96.sh admit > $B/nohup_admit.out 2>&1 && bash $B/run_a96.sh bank > $B/nohup_bank.out 2>&1 )
echo "$(date '+%F %T') $B end rc $?" >> $L
echo A96Q_S5_DONE >> $L
