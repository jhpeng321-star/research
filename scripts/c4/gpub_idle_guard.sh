#!/usr/bin/env bash
# gpub_idle_guard.sh -- gpuB cost guard (GPUB_LOOP.md).  Shuts the box down (sync; shutdown -h now)
# only after LIMIT_MIN (90) CONTINUOUS minutes in which NONE of these is present:
#   * a PID listed in /root/autodl-tmp/c4/gpubloop/OUR_PIDS that is still alive (/proc/<pid>);
#   * a process whose command line matches our job patterns: `-m wms train`, wms.control.policy14_cli,
#     wms.control.train_cli, scripts/c4/loop14.py, gpubloop_driver.sh, loop14_checks/accept.py, sumo;
#   * any GPU compute process (nvidia-smi), e.g. a co-located round-16 training;
#   * the override file /root/autodl-tmp/c4/KEEP_ALIVE.
# Read-only checks (ps / nvidia-smi); never signals or kills anything.  Log: /root/autodl-tmp/c4/idle_guard.log
set -u
C4=/root/autodl-tmp/c4; LOG="$C4/idle_guard.log"; PIDS="$C4/gpubloop/OUR_PIDS"; KEEP="$C4/KEEP_ALIVE"
LIMIT_MIN="${LIMIT_MIN:-90}"; STEP_S=300; idle=0; last=""
PAT='-m wms train|wms\.control\.(policy14_cli|train_cli)|scripts/c4/loop14\.py|gpubloop_driver\.sh|loop14_checks/accept\.py|/bin/sumo( |$)'
ts() { date '+%F %T %Z'; }
active() {
  [ -e "$KEEP" ] && { echo "KEEP_ALIVE file"; return 0; }
  if [ -f "$PIDS" ]; then
    for p in $(cat "$PIDS"); do [ -d "/proc/$p" ] && { echo "recorded pid $p"; return 0; }; done
  fi
  m="$(ps -eo pid=,args= | grep -E -- "$PAT" | grep -v -E 'grep|gpub_idle_guard' | head -1)"
  [ -n "$m" ] && { echo "process: ${m:0:160}"; return 0; }
  g="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | head -1)"
  [ -n "$g" ] && { echo "gpu app: $g"; return 0; }
  return 1
}
echo "$(ts) idle guard pid $$ start (limit ${LIMIT_MIN} min, check every ${STEP_S} s)" >> "$LOG"
since=""   # epoch of the first idle observation in the current idle streak
while true; do
  if why="$(active)"; then
    [ -n "$since" ] || [ "$last" != "active" ] && echo "$(ts) active: $why" >> "$LOG"
    since=""; last=active; idle=0
  else
    now="$(date +%s)"; [ -n "$since" ] || since="$now"
    idle=$(( (now - since) / 60 )); last=idle
    echo "$(ts) idle ${idle} min (streak since $(date -d "@$since" '+%T'))" >> "$LOG"
    if [ "$idle" -ge "$LIMIT_MIN" ]; then
      echo "$(ts) SHUTDOWN: ${idle} continuous idle minutes (sync; shutdown -h now)" >> "$LOG"
      sync; shutdown -h now; exit 0
    fi
  fi
  echo "$(ts) idle=${idle}min" > "$C4/idle_guard.status"
  sleep "$STEP_S"
done
