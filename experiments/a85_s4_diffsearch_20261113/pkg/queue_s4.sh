#!/bin/bash
# A85 S4 queue: wait for the A79 chain's "S3b rc" line, then run S4 ONCE (no retry). Zero SUMO, never trains.
# Starts S4 even if S3b rc != 0 (independent run; logged). Refuses if out_s4 / out_s4.meta exist or another queue
# holds the lock. Never signals any process. Log: /root/autodl-tmp/a85/chain_s4.log
set -u
B=/root/autodl-tmp/a85
CHAINLOG=/root/autodl-tmp/a75/chain_s2b.log
CHAINPID=5708                      # bash /root/autodl-tmp/a75/chain_s2b.sh (identity-checked, read only)
OUT=$B/out_s4
exec >> $B/chain_s4.log 2>&1
mkdir $B/queue_s4.lock 2>/dev/null || { echo "REFUSE: queue lock $B/queue_s4.lock exists $(date)"; exit 3; }
echo "$$" > $B/queue_s4.lock/pid
if [ -e "$OUT" ] || [ -e "$OUT.meta" ]; then echo "REFUSE: $OUT or $OUT.meta exists $(date)"; exit 3; fi
echo "S4 queue pid $$ waiting for 'S3b rc' in $CHAINLOG $(date)"
while ! grep -q '^S3b rc' "$CHAINLOG" 2>/dev/null; do
  if ! tr '\0' ' ' < /proc/$CHAINPID/cmdline 2>/dev/null | grep -q 'chain_s2b.sh'; then
    sleep 5
    grep -q '^S3b rc' "$CHAINLOG" 2>/dev/null && break
    echo "S4 NOT STARTED: chain pid $CHAINPID gone without an 'S3b rc' line $(date)"; exit 2
  fi
  sleep 60
done
echo "seen: $(grep '^S3b rc' "$CHAINLOG" | tail -1)"
S3RC=$(grep '^S3b rc' "$CHAINLOG" | tail -1 | awk '{print $3}')
[ "$S3RC" = "0" ] || echo "NOTE: S3b rc=$S3RC (not 0); S4 is independent and starts anyway"
sleep 30                           # let the S3b workers release the GPU
if [ -e "$OUT" ] || [ -e "$OUT.meta" ]; then echo "REFUSE: $OUT or $OUT.meta appeared $(date)"; exit 3; fi
# same MPS client env as chain_s2b.sh (daemon started by the A79 swap; not started or stopped here)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_a75_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_a75_log
pgrep -x nvidia-cuda-mps-control > /dev/null && echo "mps control daemon alive" || echo "NOTE: no MPS control daemon (CUDA runs without MPS)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "S4 start $(date)"
bash $B/run_search_s4.sh $B/cfg/A85S4_SEARCH_CFG.json $OUT; rc=$?
echo "S4 rc $rc $(date)"
echo QUEUE_DONE
