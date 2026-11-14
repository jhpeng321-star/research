#!/bin/bash
# A96(b) reference + HAND-p20 admission (Box 3). NOT started by the prep agent. Usage (MAPPO20C sha as registered):
#   A96_MAPPO20C_SHA256=<sha> bash /root/a96q_ref_hand.sh
# Waits for A94Q_DONE in /root/a94queue.log AND for /root/a82four/inputs/{policy_00084.pt,PLAN.json,BOUND_00084.json}
# with policy sha256 == A96_MAPPO20C_SHA256; then (zero SUMO) binds MAPPO20C in /root/a96ref if not yet bound; then,
# unless busy (-> "A96Q_SKIPPED busy", exit): reference stage (MAPPO20C x 24 worlds, 28,800 sim-s) -> pin-ref of the
# hand root -> mech -> admit -> bank of /root/a96adm_hand, each ONCE, 46 workers. Any failure ends the chain.
# Log /root/a96q_ref_hand.log, last line A96Q_REF_HAND_DONE.
L=/root/a96q_ref_hand.log
REF=/root/a96ref
H=/root/a96adm_hand
IN=/root/a82four/inputs
SHA=${A96_MAPPO20C_SHA256:?set A96_MAPPO20C_SHA256}
[ ${#SHA} = 64 ] || { echo "bad sha" | tee -a $L; exit 1; }
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
fin() { echo "$(date '+%F %T') $*" >> $L; echo A96Q_REF_HAND_DONE >> $L; exit 1; }
[ -f $H/root/A96_CANDIDATES.json ] || { echo "$(date '+%F %T') REFUSED: hand root not pinned" | tee -a $L; exit 1; }
for f in $REF/chain_ref.out $H/chain_mech.out; do [ -f $f ] && { echo "$(date '+%F %T') REFUSED: $f exists (launched)" | tee -a $L; exit 1; }; done
echo "$(date '+%F %T') a96 ref+hand queue pid $$ waiting for A94Q_DONE + MAPPO20C files (sha ${SHA:0:12})" >> $L
while ! grep -q A94Q_DONE /root/a94queue.log 2>/dev/null || [ ! -f $IN/policy_00084.pt ] || [ ! -f $IN/PLAN.json ] || [ ! -f $IN/BOUND_00084.json ] \
      || [ "$(sha256sum $IN/policy_00084.pt | cut -c1-64)" != "$SHA" ]; do sleep 30; done
echo "$(date '+%F %T') wait over" >> $L
[ -f $REF/chain_ref.out ] && fin "REFUSED after wait: reference launched"
if [ ! -f $REF/root/BOUND_A77_MAPPO20C.json ]; then
  A96_MAPPO20C_POLICY=$IN/policy_00084.pt A96_MAPPO20C_SHA256=$SHA A96_TRAIN_RECEIPT=$IN/BOUND_00084.json A96_TRAIN_PLAN=$IN/PLAN.json \
    bash $REF/run_a96.sh bind > $REF/nohup_bind.out 2>&1 || fin "BIND FAILED (see $REF/nohup_bind.out)"
fi
grep -q "\"sha256\": \"$SHA\"" $REF/root/BOUND_A77_MAPPO20C.json || fin "bound MAPPO20C sha is not $SHA"
busy && { echo "A96Q_SKIPPED busy" >> $L; exit 0; }
export A96_WORKERS=46
echo "$(date '+%F %T') $REF ref start (ALLOCATION START 28,800)" >> $L
bash $REF/run_a96.sh ref > $REF/nohup_ref.out 2>&1; r=$?
grep -q A96_REF_DONE $REF/chain_ref.out 2>/dev/null && [ $r = 0 ] || fin "reference stage rc $r (no A96_REF_DONE): STOP"
bash $H/run_a96.sh pinref > $H/nohup_pinref.out 2>&1 || fin "hand pin-ref failed: STOP"
busy && { echo "A96Q_SKIPPED busy" >> $L; exit 0; }
echo "$(date '+%F %T') $H start (ALLOCATION START)" >> $L
( bash $H/run_a96.sh mech > $H/nohup_mech.out 2>&1 && bash $H/run_a96.sh admit > $H/nohup_admit.out 2>&1 && bash $H/run_a96.sh bank > $H/nohup_bank.out 2>&1 )
echo "$(date '+%F %T') $H end rc $?" >> $L
echo A96Q_REF_HAND_DONE >> $L
