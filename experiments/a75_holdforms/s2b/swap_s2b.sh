#!/bin/bash
# A79: stop own S3 queue + S2 (by PID, identity-checked), start MPS, launch chain once.
set -u
A=/root/autodl-tmp/a75; P=/root/autodl-tmp/a75p20
[ -e $A/out_s2b ] && { echo "REFUSE: out_s2b exists"; exit 3; }
chk() { tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null | grep -q -- "$2"; }
chk 4694 'queue_s3.sh' || { echo "pid 4694 is not the S3 queue"; exit 4; }
chk 1903 'a75ss --states data/regime_STATES.json --cell D2800_p60' || { echo "pid 1903 is not S2"; exit 4; }
chk 1899 'run_search_a75.sh' || { echo "pid 1899 mismatch"; exit 4; }
chk 1897 'run_search_a75.sh' || { echo "pid 1897 mismatch"; exit 4; }
desc() { for c in $(pgrep -P $1); do echo $c; desc $c; done; }
Q=$(desc 4694); S=$(desc 1903)
echo "S3 queue children: $Q"; echo "S2 workers: $(echo $S | wc -w)"
ls $A/out_s2/crn_log | wc -l > /tmp/s2_tasks_done.txt
kill 4694 $Q 2>/dev/null
kill 1897 1899 2>/dev/null
kill 1903 $S 2>/dev/null
for i in $(seq 1 30); do n=0; for p in 4694 1897 1899 1903 $Q $S; do [ -d /proc/$p ] && n=$((n+1)); done; [ $n -eq 0 ] && break; sleep 2; done
echo "still alive: $n"
[ $n -eq 0 ] || { echo "STOP INCOMPLETE - not launching"; exit 5; }
echo "STOPPED $(date) by owner decision (Amendment 79): S2 unfinished, crn tasks done $(cat /tmp/s2_tasks_done.txt), LOG lines $(wc -l < $A/out_s2/LOG_D2800_p60.jsonl); superseded by out_s2b" > $A/out_s2.meta/STOPPED.txt
echo "WITHDRAWN $(date) (Amendment 79): S3 queue pid 4694 stopped before any search; superseded by out_s3b" > $P/QUEUE_S3.withdrawn
sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_a75_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_a75_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d; echo "mps rc $?"; sleep 2
pgrep -x nvidia-cuda-mps > /tmp/mps_a75.pid; echo "mps pid $(cat /tmp/mps_a75.pid)"
setsid -f nohup bash $A/chain_s2b.sh < /dev/null > /dev/null 2>&1
sleep 45; cat $A/chain_s2b.log; tail -5 $A/out_s2b.meta/run.log | cut -c1-300
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader | head -20
date
