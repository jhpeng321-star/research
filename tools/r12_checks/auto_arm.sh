#!/usr/bin/env bash
# R12 post-run auto-arm (gpuR, laptop independent).  Waits for runs/c4_wm_v10_round12/RUN_SUMMARY.json with the
# round-12 training PID gone (or for a mid-run STOP), AND for the round-11 checks to finish (chk_r11/VERDICT_R11_CHECKS.md
# exists or the round-11 auto_arm PID is gone); then runs the registered round-12 checks with the second seed set
# (SEEDS2=1 bash tools/r12_checks/run_all.sh full; CPU only, nice 10, <= 20 workers) -> chk_r12/VERDICT_R12_CHECKS.md,
# then the telemetry reads and the post-run merge-law support recount (96 windows, 16 shards, nice 15).
# On a mid-run STOP it exits without running the full checks (the coordinator decides).
# Launch: cd $REPO && setsid nohup bash tools/r12_checks/auto_arm.sh $REPO > $L/chk_r12/auto_arm.log 2>&1 < /dev/null &
set -u -o pipefail
L=${R12_ROOT:-/root/autodl-tmp/c4/v5}
REPO=${1:?usage: auto_arm.sh REPO_DIR}
PY=${R12_PYTHON:-/root/miniconda3/bin/python}
CHK=${R12_CHK:-$L/chk_r12}
RUN=${R12_RUN:-$L/runs/c4_wm_v10_round12}
R11_ARM_PID=${R11_ARM_PID:-426390}
WORKERS=${WORKERS:-20}
mkdir -p "$CHK/results"
echo $$ > "$CHK/auto_arm.pid"
echo "r12 auto_arm $$ started $(date '+%F %T'); repo=$REPO; waiting for $RUN/RUN_SUMMARY.json and the round-11 checks"
while true; do
  if [ -f "$CHK/MIDRUN_STOP" ]; then echo "mid-run STOP recorded; full checks not run $(date '+%F %T')"; exit 0; fi
  TPID=$(cat "$L/gate_r12.pid" 2>/dev/null || echo "")
  if [ -f "$RUN/RUN_SUMMARY.json" ] && { [ -z "$TPID" ] || ! kill -0 "$TPID" 2>/dev/null; }; then break; fi
  sleep 120
done
echo "round-12 RUN_SUMMARY seen, training gone $(date '+%F %T')"
while [ ! -f "$L/chk_r11/VERDICT_R11_CHECKS.md" ] && kill -0 "$R11_ARM_PID" 2>/dev/null; do sleep 120; done
echo "round-11 checks finished or gone $(date '+%F %T')"
sleep 60
export R12_REPO=$REPO R12_TOOLS=$REPO/tools R12_CHK=$CHK R12_RUN=$RUN
SEEDS2=1 WORKERS=$WORKERS nice -n 10 bash "$REPO/tools/r12_checks/run_all.sh" full
echo "r12 run_all full exit $? $(date '+%F %T')"
export PYTHONPATH=$REPO/src:$REPO:$REPO/tools
cd "$REPO" || exit 2
nice -n 10 $PY tools/r12_checks/telemetry_reads.py "$RUN" "$CHK/results/TELEMETRY_R12.json" "$CHK/results/TELEMETRY_R12.md" \
  2>&1 | grep -v -i warn | tail -2
CK=$RUN/wm_s_stage300.pt
[ -f "$CK" ] || CK=$RUN/wm_s_best.pt
R9=$L/runs/c4_wm_v10_round9/wm_s_stage300.pt
SUP=$CHK/support
mkdir -p "$SUP"
for i in $(seq 0 15); do
  OMP_NUM_THREADS=1 nice -n 15 $PY tools/r12_support_recount.py --roots "$L/data/v5_native" "$L/data/genrand_plain" \
    --r9-ckpt "$R9" --extra "$CK" --windows 96 --shard "$i" --n-shards 16 --threads 1 \
    --out "$SUP/support_s$i.json" > "$SUP/support_s$i.log" 2>&1 &
done
wait
nice -n 10 $PY tools/r12_support_recount.py --roots x --merge "$SUP"/support_s*.json --out "$CHK/results/SUPPORT_R12.json" \
  2>&1 | tail -3
echo "r12 auto_arm done $(date '+%F %T')"
