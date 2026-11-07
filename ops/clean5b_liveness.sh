#!/bin/bash
# Off-host liveness check (MAIN 2026-11-07; run on the laptop): every 5 min record whether every registered owner of
# <root> on <host> is alive, the disk free, and MEMORY_WATCH freshness; prints ALERT lines (report-only, never acts).
# usage: ops/clean5b_liveness.sh <host> <root> <local log>
H=$1; R=$2; LOG=$3
while true; do
  line=$(timeout 60 ssh -n -o ConnectTimeout=20 $H "R=$R; df -BG --output=avail / | tail -1; for f in \$R/owners/*.json; do p=\$(python3 -c \"import json;print(json.load(open('\$f'))['pid'])\"); [ -d /proc/\$p ] && echo \"\$(basename \$f .json)=alive\" || echo \"\$(basename \$f .json)=DEAD\"; done; echo memwatch_age_s=\$(( \$(date +%s) - \$(stat -c %Y \$R/MEMORY_WATCH.jsonl 2>/dev/null || echo 0) )); ls \$R | grep -E 'COMPLETE|STOP|FAULT' | tr '\n' ' '" 2>&1 | tr '\n' ' ')
  echo "$(date '+%F %T') $line" >> $LOG
  case "$line" in *DEAD*|*STOP*|*FAULT*) echo "$(date '+%F %T') ALERT $line" >> $LOG;; esac
  case "$line" in *MONITOR_COMPLETE*) exit 0;; esac
  sleep 300
done
