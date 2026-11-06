#!/usr/bin/env bash
# Auto-arm (gpuR, laptop independent): wait for the round-11 RUN_SUMMARY.json AND for the round-10b checks to finish
# (chk_r10b/VERDICT_R10B_CHECKS.md exists or its auto_arm PID is gone), then run all R11 checks (CPU only, nice 10,
# <= 20 workers) -> chk_r11/VERDICT_R11_CHECKS.md (M1-M8 incl. the keep-10b rule).
# Launch: cd $CHK && setsid nohup bash tools/r11_checks/auto_arm.sh > auto_arm.log 2>&1 < /dev/null &   (PID in auto_arm.pid)
L=/root/autodl-tmp/c4/v5
CHK=${R11_CHK:-$L/chk_r11}
RUN=${R11_RUN:-$L/runs/c4_wm_v10_round11}
R10B_ARM_PID=${R10B_ARM_PID:-413441}
WORKERS=${WORKERS:-20}
echo $$ > "$CHK/auto_arm.pid"
echo "r11 auto_arm $$ started $(date '+%F %T'); waiting for $RUN/RUN_SUMMARY.json and the round-10b checks"
while [ ! -f "$RUN/RUN_SUMMARY.json" ]; do sleep 120; done
echo "RUN_SUMMARY seen $(date '+%F %T')"
while [ ! -f "$L/chk_r10b/VERDICT_R10B_CHECKS.md" ] && kill -0 "$R10B_ARM_PID" 2>/dev/null; do sleep 120; done
echo "round-10b checks finished or gone $(date '+%F %T')"
sleep 90   # let the final checkpoints and receipts finish writing
for i in 1 2 3 4 5 6; do
  if [ -f "$RUN/wm_s_stage300.pt" ] || [ -f "$RUN/wm_s_best.pt" ]; then break; fi
  sleep 60
done
export R11_REPO=${R11_REPO:-$L/repo_r11_4d8de32b} R11_TOOLS=${R11_TOOLS:-$CHK/tools} R11_CHK=$CHK R11_RUN=$RUN
WORKERS=$WORKERS nice -n 10 bash "$CHK/tools/r11_checks/run_all.sh" full
RC=$?
echo "r11 run_all full exit $RC $(date '+%F %T')"
echo "r11 auto_arm done $(date '+%F %T')"
