#!/bin/bash
# A96 S4-p20 queue (modelled on a85/queue_s4.sh 58389938): run S4-p20 ONCE (no retry). Zero SUMO, never trains.
# No predecessor to wait for (S4 finished 01:25 rc 0); instead it waits while another search python (a75ss/a85s4)
# is running, so the GPU is free. Refuses if out_s4p20 / out_s4p20.meta exist or another queue holds the lock.
# Never signals any process. Log: /root/autodl-tmp/a96_s4p20/chain_s4p20.log
set -u
B=/root/autodl-tmp/a96_s4p20
OUT=$B/out_s4p20
exec >> $B/chain_s4p20.log 2>&1
mkdir $B/queue_s4p20.lock 2>/dev/null || { echo "REFUSE: queue lock $B/queue_s4p20.lock exists $(date)"; exit 3; }
echo "$$" > $B/queue_s4p20.lock/pid
if [ -e "$OUT" ] || [ -e "$OUT.meta" ]; then echo "REFUSE: $OUT or $OUT.meta exists $(date)"; exit 3; fi
echo "S4p20 queue pid $$ $(date)"
while pgrep -f 'python -B -m (a75ss|a85s4) --states' > /dev/null; do sleep 60; done
if [ -e "$OUT" ] || [ -e "$OUT.meta" ]; then echo "REFUSE: $OUT or $OUT.meta appeared $(date)"; exit 3; fi
# same MPS client env as queue_s4.sh / chain_s2b.sh (daemon not started or stopped here). queue_s4.sh checked with
# pgrep -x on the full 23-char name, which cannot match (comm is truncated to 15 chars), so S4 logged "no MPS"
# while daemon pid 5702 was up. Logging only; the env is identical.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_a75_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_a75_log
pgrep -f '^nvidia-cuda-mps-control' > /dev/null && echo "mps control daemon alive" || echo "NOTE: no MPS control daemon (CUDA runs without MPS)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "S4p20 start $(date)"
bash $B/run_search_s4p20.sh $B/cfg/A96S4P20_SEARCH_CFG.json $OUT; rc=$?
echo "S4p20 rc $rc $(date)"
echo QUEUE_DONE
