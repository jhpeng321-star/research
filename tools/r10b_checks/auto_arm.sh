#!/usr/bin/env bash
# Auto-arm: wait for the round-10b RUN_SUMMARY.json, then run all R10B checks (CPU, <=24 workers) -> VERDICT_R10B_CHECKS.md.
# Launch: setsid nohup bash auto_arm.sh > auto_arm.log 2>&1 &   (PID in auto_arm.pid).  Laptop independent.
L=/root/autodl-tmp/c4/v5
CHK=$L/chk_r10b
RUN=$L/runs/c4_wm_v10_round10b
echo $$ > $CHK/auto_arm.pid
echo "auto_arm $$ started $(date "+%F %T"); waiting for $RUN/RUN_SUMMARY.json"
while [ ! -f "$RUN/RUN_SUMMARY.json" ]; do sleep 120; done
echo "RUN_SUMMARY seen $(date "+%F %T")"
sleep 90   # let the final checkpoints and receipts finish writing
for i in 1 2 3 4 5 6; do
  if [ -f "$RUN/wm_s_stage300.pt" ] || [ -f "$RUN/wm_s_best.pt" ]; then break; fi
  sleep 60
done
WORKERS=24 bash $CHK/r10b_checks/run_all.sh full
RC=$?
echo "run_all full exit $RC $(date "+%F %T")"
# best-effort copy of the verdict to cpuN (ssh alias may not exist from this host)
if [ -f $CHK/VERDICT_R10B_CHECKS.md ]; then
  scp -q -o ConnectTimeout=20 $CHK/VERDICT_R10B_CHECKS.md cpuN:autopilot_r10/ 2>/dev/null && echo "copied to cpuN" || echo "cpuN copy skipped"
fi
echo "auto_arm done $(date "+%F %T")"

