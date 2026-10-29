#!/bin/bash
# own-PID GPU+host sampler: usage sampler.sh OUTFILE  (stop by touching OUTFILE.stop)
# columns: epoch, gpu_used_MiB, per-process "pid:MiB;" list, cgroup anon MiB, python/worker RSS sum MiB
OUT=$1
while [ ! -e "$OUT.stop" ]; do
  t=$(date +%s.%N)
  tot=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  procs=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ';')
  anon=$(awk '/^anon /{printf "%d", $2/1048576}' /sys/fs/cgroup/memory.stat 2>/dev/null)
  rss=$(ps -eo rss,args | awk '/policy14_cli|parallel_futures/ && !/awk/{s+=$1} END{printf "%d", s/1024}')
  echo "$t,$tot,$procs,$anon,$rss" >> "$OUT"
  sleep 2
done
