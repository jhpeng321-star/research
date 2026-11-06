#!/usr/bin/env bash
# Round-12 waiter (gpuR, laptop independent).  Waits until round 11 has finished (runs/c4_wm_v10_round11/RUN_SUMMARY.json
# exists AND round 11's training PID is gone), then exec's scripts/round12_gate.sh with R12_PRIOR=off (R12_REVIEW C1) and
# the default R12_INIT (round-9 final stage300, else best).  The gate itself refuses stale artifacts
# (runs/smoke_r12, runs/c4_wm_v10_round12, logs, lock); this waiter also refuses up front if the round-12 run dir exists.
# Because the waiter exec's the gate and the gate exec's the full run, the PID in $L/gate_r12.pid is the round-12
# training PID (used by tools/r12_checks/midrun_arm.sh for the registered mid-run STOP rule).
# Launch: cd $REPO && setsid nohup bash scripts/round12_wait.sh $REPO > $L/gate_r12.out 2>&1 < /dev/null &
set -u -o pipefail
L=${R12_ROOT:-/root/autodl-tmp/c4/v5}
REPO=${1:?usage: round12_wait.sh REPO_DIR}
R11_RUN=${R11_RUN:-$L/runs/c4_wm_v10_round11}
R11_PID=${R11_PID:-423697}
echo $$ > "$L/gate_r12.pid"
echo "round12_wait $$ started $(date '+%F %T'): repo=$REPO; waiting for $R11_RUN/RUN_SUMMARY.json and PID $R11_PID gone"
if [ -e "$L/runs/c4_wm_v10_round12" ]; then echo "refusing: $L/runs/c4_wm_v10_round12 exists"; exit 4; fi
while [ ! -f "$R11_RUN/RUN_SUMMARY.json" ] || kill -0 "$R11_PID" 2>/dev/null; do sleep 60; done
echo "round 11 finished $(date '+%F %T'); sleeping 60 s for its final writes"
sleep 60
if [ -e "$L/runs/c4_wm_v10_round12" ]; then echo "refusing: $L/runs/c4_wm_v10_round12 exists"; exit 4; fi
echo "exec round12_gate.sh with R12_PRIOR=off $(date '+%F %T')"
cd "$REPO" || exit 2
export R12_PRIOR=off
exec bash scripts/round12_gate.sh "$REPO"
