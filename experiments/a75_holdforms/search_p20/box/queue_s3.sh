#!/bin/bash
# A75-S3 queue: wait for S2 to finish (QUEUE_DONE in /root/autodl-tmp/a75/chain_s2.log, read only), then run the
# p20 search ONCE into out_s3. Never touches S2 files/processes. Log: /root/autodl-tmp/a75p20/chain_s3.log
set -u
P=/root/autodl-tmp/a75p20
S2LOG=/root/autodl-tmp/a75/chain_s2.log
[ -e $P/QUEUE_S3.started ] && { echo "REFUSE: queue already started once"; exit 3; }
date +%s > $P/QUEUE_S3.started
echo "queue pid $$ waiting for QUEUE_DONE in $S2LOG $(date)"
until grep -q '^QUEUE_DONE' "$S2LOG" 2>/dev/null; do sleep 60; done
echo "S2 done seen $(date); S2 line: $(grep 'S2 rc' "$S2LOG" | tail -1)"
# S2's python must be gone before the GPU is used (bracket pattern: never matches this grep itself)
while pgrep -f '[a]75ss --states data/regime_STATES.json --cell D2800_p60' > /dev/null; do sleep 30; done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "S3 start $(date)"
bash $P/run_search_s3.sh $P/cfg/A75S3_SEARCH_CFG_p20.json $P/out_s3
echo "S3 rc $? $(date)"
echo S3_QUEUE_DONE
