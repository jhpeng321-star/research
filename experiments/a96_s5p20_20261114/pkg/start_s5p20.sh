#!/bin/bash
# A96 S5-p20 one-shot start (gpuR), copy of A86 start_s5.sh (f494996d), only paths/labels/cell changed. Runs S5-p20 ONCE, never retries, never signals any process. Zero SUMO, never trains.
# Refuses if SETUP_OK is missing, the lock exists, out_s5p20 / out_s5p20.meta exist, or any compute process is on the GPU.
# Log: /root/autodl-tmp/a96_s5p20/chain_s5p20.log ("S5P20 start <date>", "S5P20 rc <N> <date>", START_DONE).
set -u
B=/root/autodl-tmp/a96_s5p20
OUT=$B/out_s5p20
exec >> $B/chain_s5p20.log 2>&1
mkdir $B/start_s5p20.lock 2>/dev/null || { echo "REFUSE: lock $B/start_s5p20.lock exists $(date)"; exit 3; }
echo "$$" > $B/start_s5p20.lock/pid
[ -e $B/SETUP_OK ] || { echo "S5P20 NOT STARTED: $B/SETUP_OK missing $(date)"; exit 3; }
if [ -e "$OUT" ] || [ -e "$OUT.meta" ]; then echo "S5P20 NOT STARTED: $OUT or $OUT.meta exists $(date)"; exit 3; fi
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
if [ "$busy" != "0" ]; then
  echo "S5P20 NOT STARTED: $busy compute process(es) on the GPU $(date)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; exit 4
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "S5P20 start $(date)"
bash $B/run_search_s5p20.sh $B/cfg/A96S5P20_SEARCH_CFG.json $OUT; rc=$?
echo "S5P20 rc $rc $(date)"
echo START_DONE
