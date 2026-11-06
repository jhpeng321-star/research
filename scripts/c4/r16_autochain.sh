#!/usr/bin/env bash
# Detached smoke waiter; install with the smoke's R16_*/WMS_* environment.
# Arguments pin the observed PID incarnation, so PID reuse cannot delay launch.
set -eu -o pipefail
PID="$1"; START="$2"
ROOT=/root/autodl-tmp/c4/v5
REPO="$ROOT/repo_r16_3b3c201d"
ts() { date -u +%FT%TZ; }
echo "$(ts) waiter $$ waiting for smoke $PID (start $START)"
while [ -r "/proc/$PID/stat" ]; do
    read -r state current < <(awk '{print $3, $22}' "/proc/$PID/stat") || break
    [ "$current" = "$START" ] && [ "$state" != Z ] || break
    sleep 15
done
if [ -e "$ROOT/STOP_R16_AUTOCHAIN" ]; then
    echo "$(ts) no launch: STOP_R16_AUTOCHAIN present"
elif grep -Fq 'Both R16 smoke gates passed' "$ROOT/smoke_r16_driver.log"; then
    echo "$(ts) smoke passed; launching R16 chain in $REPO"
    cd "$REPO"
    rc=0
    bash scripts/round16_gpuB.sh --chain || rc=$?
    echo "$(ts) R16 chain exited $rc"
    exit "$rc"
else
    echo "$(ts) no launch: smoke success marker absent"
fi
