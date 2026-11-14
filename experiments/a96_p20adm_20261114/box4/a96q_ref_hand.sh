#!/bin/bash
# A96(b) reference + HAND-p20 admission (BOX 4). NOT started by the prep agent. Usage (MAPPO20C sha as registered):
#   A96_MAPPO20C_SHA256=<sha> bash /root/a96q_ref_hand.sh
# Waits for /home/ubuntu/a77/TRAIN_DONE AND /home/ubuntu/a77/run84/{production/policy_00084.pt,PLAN.json,BOUND_00084.json}
# (paths overridable: A96_POLICY, A96_PLAN, A96_RECEIPT)
# with policy sha256 == A96_MAPPO20C_SHA256; then (zero SUMO) binds MAPPO20C in /root/a96ref if not yet bound; then,
# unless busy (-> "A96Q_SKIPPED busy", exit): reference stage (MAPPO20C x 24 worlds, 28,800 sim-s) -> pin-ref of the
# hand root -> mech -> admit -> bank of /root/a96adm_hand, each ONCE, 46 workers. Any failure ends the chain.
# Log /root/a96q_ref_hand.log, last line A96Q_REF_HAND_DONE.
L=/root/a96q_ref_hand.log
REF=/root/a96ref
H=/root/a96adm_hand
POL=${A96_POLICY:-/home/ubuntu/a77/run84/production/policy_00084.pt}
PLAN=${A96_PLAN:-/home/ubuntu/a77/run84/PLAN.json}
REC=${A96_RECEIPT:-/home/ubuntu/a77/run84/BOUND_00084.json}
TD=/home/ubuntu/a77/TRAIN_DONE
SHA=${A96_MAPPO20C_SHA256:?set A96_MAPPO20C_SHA256}
[ ${#SHA} = 64 ] || { echo "bad sha" | tee -a $L; exit 1; }
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
fin() { echo "$(date '+%F %T') $*" >> $L; echo A96Q_REF_HAND_DONE >> $L; exit 1; }
[ -f $H/root/A96_CANDIDATES.json ] || { echo "$(date '+%F %T') REFUSED: hand root not pinned" | tee -a $L; exit 1; }
for f in $REF/chain_ref.out $H/chain_mech.out; do [ -f $f ] && { echo "$(date '+%F %T') REFUSED: $f exists (launched)" | tee -a $L; exit 1; }; done
echo "$(date '+%F %T') a96 ref+hand queue (Box 4) pid $$ waiting for $TD + MAPPO20C files (sha ${SHA:0:12})" >> $L
n=0
while [ ! -f $TD ] || [ ! -f $POL ] || [ ! -f $PLAN ] || [ ! -f $REC ] || [ "$(sha256sum $POL | cut -c1-64)" != "$SHA" ]; do
  sleep 30; n=$((n+1)); [ $((n % 40)) = 0 ] && echo "$(date '+%F %T') still waiting (TRAIN_DONE $([ -f $TD ] && echo yes || echo no), receipt $([ -f $REC ] && echo yes || echo no))" >> $L
done
echo "$(date '+%F %T') wait over; 60 s grace for the training chain to exit" >> $L; sleep 60
[ -f $REF/chain_ref.out ] && fin "REFUSED after wait: reference launched"
if [ ! -f $REF/root/BOUND_A77_MAPPO20C.json ]; then
  A96_MAPPO20C_POLICY=$POL A96_MAPPO20C_SHA256=$SHA A96_TRAIN_RECEIPT=$REC A96_TRAIN_PLAN=$PLAN \
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
