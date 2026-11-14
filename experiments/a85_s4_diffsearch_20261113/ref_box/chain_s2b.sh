#!/bin/bash
# A79 chain: S2b (p60) then S3b (p20), each once, under the MPS daemon started beforehand. Stops if S2b fails.
set -u
A=/root/autodl-tmp/a75; P=/root/autodl-tmp/a75p20
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_a75_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_a75_log
exec >> $A/chain_s2b.log 2>&1
echo "chain pid $$ S2b start $(date)"
bash $A/run_search_s2b.sh $A/cfg/A75S2B_SEARCH_CFG.json $A/out_s2b; rc=$?
echo "S2b rc $rc $(date)"
if [ "$rc" -ne 0 ]; then echo "CHAIN_STOPPED after S2b failure (no retry)"; echo QUEUE_DONE; exit 1; fi
echo "S3b start $(date)"
bash $P/run_search_s3b.sh $P/cfg/A75S3B_SEARCH_CFG_p20.json $P/out_s3b
echo "S3b rc $? $(date)"
echo QUEUE_DONE
