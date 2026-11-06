#!/usr/bin/env bash
# WM-E gpuA runner (runs ON gpuA): start the resumable wz3to2 WM fine-tune detached, on its own PID.
#   bash ops/wzwm_gpua_run.sh CODE_DIR RUN_ROOT
# CODE_DIR = deployed `git archive` of c4/wz-wm-train (CODE_SHA inside); RUN_ROOT under /root/autodl-fs
# (autodl-tmp is ~97% full). Re-running after an interruption resumes (tools.wzwm.run train).
# Refuses on low disk, a missing PLAN.json, or a live runner of this root. Never signals other PIDs.
set -euo pipefail
CODE=${1:?CODE_DIR}; ROOT=${2:?RUN_ROOT}
PY=${WZWM_PYTHON:-/root/miniconda3/bin/python}
MIN_GB_ROOT=${WZWM_MIN_FREE_GB:-5}
free_gb() { df -Pk "$1" | awk 'NR==2{printf "%d", $4/1048576}'; }
case "$ROOT" in /root/autodl-fs/*|/autodl-fs/*) ;; *) echo "REFUSED: RUN_ROOT must be on /root/autodl-fs"; exit 3;; esac
[ -f "$ROOT/PLAN.json" ] || { echo "REFUSED: $ROOT/PLAN.json missing (run tools.wzwm.run prepare first)"; exit 3; }
mkdir -p "$ROOT/tmp"
[ "$(free_gb "$ROOT")" -ge "$MIN_GB_ROOT" ] || { echo "REFUSED: < ${MIN_GB_ROOT} GB free on $ROOT"; exit 3; }
[ "$(free_gb /)" -ge 1 ] || { echo "REFUSED: < 1 GB free on / (overlay)"; exit 3; }
if [ -f "$ROOT/RUNNER_PID" ]; then
  OLD=$(cat "$ROOT/RUNNER_PID")
  if [ -r "/proc/$OLD/cmdline" ] && tr '\0' ' ' < "/proc/$OLD/cmdline" | grep -q "tools.wzwm.run train --root $ROOT"; then
    echo "ALREADY RUNNING pid=$OLD"; exit 0
  fi
fi
cd "$CODE"
export TMPDIR="$ROOT/tmp" PYTHONPATH="$CODE/src:$CODE" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
setsid nohup nice -n 10 "$PY" -B -m tools.wzwm.run train --root "$ROOT" >> "$ROOT/runner.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$ROOT/RUNNER_PID"
sleep 3
if [ -r "/proc/$PID/stat" ]; then
  echo "STARTED pid=$PID start_ticks=$(awk '{print $22}' "/proc/$PID/stat") root=$ROOT code=$(cat "$CODE/CODE_SHA" 2>/dev/null)"
else
  echo "EXITED EARLY pid=$PID; tail of runner.log:"; tail -20 "$ROOT/runner.log"; exit 4
fi
