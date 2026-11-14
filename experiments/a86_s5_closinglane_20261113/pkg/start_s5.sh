#!/bin/bash
# A86 S5 one-shot start (gpuR). Runs S5 ONCE, never retries, never signals any process. Zero SUMO, never trains.
# Refuses if SETUP_OK is missing, the lock exists, out_s5 / out_s5.meta exist, or any compute process is on the GPU.
# Log: /root/autodl-tmp/a86/chain_s5.log ("S5 start <date>", "S5 rc <N> <date>").
set -u
B=/root/autodl-tmp/a86
OUT=$B/out_s5
exec >> $B/chain_s5.log 2>&1
mkdir $B/start_s5.lock 2>/dev/null || { echo "REFUSE: lock $B/start_s5.lock exists $(date)"; exit 3; }
echo "$$" > $B/start_s5.lock/pid
[ -e $B/SETUP_OK ] || { echo "S5 NOT STARTED: $B/SETUP_OK missing $(date)"; exit 3; }
if [ -e "$OUT" ] || [ -e "$OUT.meta" ]; then echo "S5 NOT STARTED: $OUT or $OUT.meta exists $(date)"; exit 3; fi
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
if [ "$busy" != "0" ]; then
  echo "S5 NOT STARTED: $busy compute process(es) on the GPU $(date)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; exit 4
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "S5 start $(date)"
bash $B/run_search_s5.sh $B/cfg/A86S5_SEARCH_CFG.json $OUT; rc=$?
echo "S5 rc $rc $(date)"
echo START_DONE
