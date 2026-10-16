#!/usr/bin/env bash
# Job wrapper started by c4_launch (nohup setsid).  Usage: c4_jobwrap.sh RUN_DIR LABEL -- cmd args...
# Writes pid, STATE (running|done|failed:RC), monitor.csv (every 60 s) and RUN_SUMMARY_C4.json.
# Runs the command exactly once: no retry.  LF only.
if LC_ALL=C grep -q $'\r' "${BASH_SOURCE[0]}"; then echo 'REFUSED: CRLF bytes in job wrapper' >&2; exit 2; fi
set -u
RUN_DIR=$1; LABEL=$2; shift 3          # drop the literal "--"
mkdir -p "$RUN_DIR"
echo $$ > "$RUN_DIR/pid"
echo running > "$RUN_DIR/STATE"
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
START_EPOCH=$(date +%s)
cd "${C4_REPO:-.}" 2>/dev/null || true

monitor() {
  local gpu util mem_used mem_avail disk
  [[ -f "$RUN_DIR/monitor.csv" ]] && mv "$RUN_DIR/monitor.csv" "$RUN_DIR/monitor.csv.prev_$(date +%s)"
  echo 'ts,load1,mem_used_mb,mem_avail_mb,gpu_mem_mib,gpu_util_pct,disk_free_gb' > "$RUN_DIR/monitor.csv"
  while true; do
    gpu=0; util=0
    if command -v nvidia-smi >/dev/null 2>&1; then
      read -r gpu util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' | tr ',' ' ') || true
    fi
    mem_used=$(free -m 2>/dev/null | awk '/^Mem:/ {print $3}'); mem_avail=$(free -m 2>/dev/null | awk '/^Mem:/ {print $7}')
    disk=$(df -Pk "$RUN_DIR" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}')
    echo "$(date -u +%H:%M:%S),$(cut -d' ' -f1 /proc/loadavg 2>/dev/null),${mem_used:-0},${mem_avail:-0},${gpu:-0},${util:-0},${disk:-0}" >> "$RUN_DIR/monitor.csv"
    sleep 60
  done
}
monitor &
MON=$!

echo "[$START_UTC] start $LABEL: $*"
"$@" &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null' TERM INT
wait "$CHILD"; RC=$?
if kill -0 "$CHILD" 2>/dev/null; then wait "$CHILD"; RC=$?; fi   # interrupted by our own trap
trap - TERM INT
END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SECONDS_RUN=$(( $(date +%s) - START_EPOCH ))
kill "$MON" 2>/dev/null; wait "$MON" 2>/dev/null

if [[ $RC -eq 0 ]]; then echo done > "$RUN_DIR/STATE"; else echo "failed:$RC" > "$RUN_DIR/STATE"; fi
PEAK_GPU=$(awk -F, 'NR>1 && $5+0>m {m=$5+0} END {print m+0}' "$RUN_DIR/monitor.csv" 2>/dev/null)
MIN_AVAIL=$(awk -F, 'NR>1 && (min==""||$4+0<min) {min=$4+0} END {print min+0}' "$RUN_DIR/monitor.csv" 2>/dev/null)
MAX_LOAD=$(awk -F, 'NR>1 && $2+0>m {m=$2+0} END {print m+0}' "$RUN_DIR/monitor.csv" 2>/dev/null)
COMMIT=$(cat "${C4_REPO:-.}/BUNDLE_COMMIT" 2>/dev/null || echo unknown)
CMD_JSON=$(printf '%s ' "$@" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
cat > "$RUN_DIR/RUN_SUMMARY_C4.json" <<EOF
{
  "label": "$LABEL",
  "run_dir": "$RUN_DIR",
  "host": "$(hostname)",
  "command": "$CMD_JSON",
  "start_utc": "$START_UTC",
  "end_utc": "$END_UTC",
  "seconds": $SECONDS_RUN,
  "exit_code": $RC,
  "state": "$(cat "$RUN_DIR/STATE")",
  "peak_gpu_mem_mib": ${PEAK_GPU:-0},
  "min_mem_available_mb": ${MIN_AVAIL:-0},
  "max_load1": ${MAX_LOAD:-0},
  "bundle_commit": "$COMMIT",
  "tool_summary": "out/RUN_SUMMARY.json (written by the tool itself when it finishes cleanly)"
}
EOF
echo "[$END_UTC] end $LABEL rc=$RC (${SECONDS_RUN}s)"
exit $RC
