#!/usr/bin/env bash
# SUMO-budget cap (2026-10-13): stop each listed episode right after its first decision point
# is written (points.jsonl has >= 1 line); kills only that episode's python PID + its SUMO child.
# (v1 of this watcher added two scalar PIDs arithmetically in PowerShell, so the first two
#  stops missed and were redone by explicit PID; fixed with explicit arrays.)
B=E:/agent-work/c4-horizon/runs/active/C4-HORIZON-20261013/B
LEFT="$*"
while [ -n "$(echo $LEFT | tr -d ' ')" ]; do
  NEXT=""
  for j in $LEFT; do
    s=${j%%:*}; p=${j##*:}; d=$B/s${s}_p${p}
    n=$(cat $d/points.jsonl 2>/dev/null | wc -l)
    if [ "$n" -ge 1 ]; then
      powershell -c "\$p = @(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*branch_horizon.py --seed $s --penetration $p *' }); \$ids = @(\$p | ForEach-Object { \$_.ProcessId }); \$kids = @(Get-CimInstance Win32_Process | Where-Object { \$ids -contains \$_.ParentProcessId } | ForEach-Object { \$_.ProcessId }); \$all = @(\$ids) + @(\$kids); if (\$ids.Count -gt 0) { Stop-Process -Id \$all -Force -ErrorAction SilentlyContinue }; Write-Output (\"stopped \" + (\$all -join ','))"
      echo "{\"stopped_after_points\": $n, \"reason\": \"SUMO budget cap: 1 decision point for the later episodes\", \"time\": \"$(date -Iseconds)\"}" > $d/STOPPED.json
      echo "$(date +%T) $j stopped after $n point(s)"
    else
      NEXT="$NEXT $j"
    fi
  done
  LEFT="$NEXT"
  sleep 20
done
