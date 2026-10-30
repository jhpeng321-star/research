#!/bin/bash
# Read-only backfill of the ACTUAL (mix-OFF) trainer after the owned mix scorer exits.
set -euo pipefail
cd /root/autodl-tmp/c4/setupaudit20261030/fix
OWNED_MIX_PID=${1:?owned mix scorer PID required}
echo "OWNED_BACKFILL_WAITER_PID=$$ waiting_for=$OWNED_MIX_PID"
while ps -p "$OWNED_MIX_PID" -o args= >/dev/null; do sleep 10; done
/root/miniconda3/bin/python -c 'import json; d=json.load(open("mix_exact.json")); assert len(d["rows"]) == 6'
sleep 3
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [ -n "$apps" ]; then
  echo "Other GPU process present; backfill not started: $apps"
  exit 3
fi
exec bash run_measure.sh --out actual_exact.json --device cuda --n-starts 1 --reuse ../actual.json
