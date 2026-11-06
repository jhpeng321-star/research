#!/usr/bin/env bash
# gpubloop_driver.sh CONFIG -- gpuB driver for the LOOP14 coop-OFF arm (GPUB_LOOP.md).
# Runs `loop14.py next` phase after phase.  The orchestrator does every gate itself; this only
# sequences it:
#   0 ok / 4 WM rejected (the cycle still completes)  -> next phase
#   5 GPU busy (shared-GPU headroom guard)             -> wait 300 s, retry the same phase (<= BUSY_MAX_S)
#   2 refused (incl. "loop stopped")                   -> print status, exit
#   3 phase failed                                     -> print status, exit (the operator decides `retry`)
# Stop between phases: touch <loop_root>/STOP_DRIVER (the running phase finishes first).
# Never signals any process.
set -u
CFG="$1"; P="${P:-/root/miniconda3/bin/python}"
cd "$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="$("$P" -c 'import json,sys;print(json.load(open(sys.argv[1]))["loop_root"])' "$CFG")"
BUSY_MAX_S="${BUSY_MAX_S:-43200}"; busy=0
ts() { date -u +%FT%TZ; }
echo "$(ts) driver pid $$ start cfg $CFG root $ROOT code $(pwd)"
while true; do
  if [ -e "$ROOT/STOP_DRIVER" ]; then echo "$(ts) STOP_DRIVER present; exiting before the next phase"; exit 0; fi
  echo "$(ts) next"
  "$P" scripts/c4/loop14.py next --config "$CFG"; rc=$?
  echo "$(ts) next exit $rc"
  case "$rc" in
    0|4) busy=0 ;;
    5) busy=$((busy + 300))
       if [ "$busy" -ge "$BUSY_MAX_S" ]; then echo "$(ts) GPU busy for ${busy} s; exiting"; exit 5; fi
       sleep 300 ;;
    *) "$P" scripts/c4/loop14.py status --config "$CFG"
       echo "$(ts) driver exits (rc $rc)"; exit "$rc" ;;
  esac
done
