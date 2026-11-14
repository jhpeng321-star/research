#!/bin/bash
# E1 sensitivity arm WM-ACT (post-registration, pre-outcome): final-score the new finalists with the frozen WM.
# GPU box only, zero SUMO, one launch; finished seeds are kept on a re-launch; nothing existing is overwritten
# (new directory wm_act_scores).   run_wm_act.sh preflight | reuse
MODE=${1:?preflight | reuse}
A=/root/autodl-tmp/a68
D=$A/claim_validation_20261113_0231
set -e
mkdir -p $D/wm_act_scores
cd $A/code_split/code_v12r7b
for N in 11 23 42 67 89 103; do
  WZPROP_SPLIT_ACTIONS=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src:. nice -n 5 \
    /root/miniconda3/bin/python -B $D/wm_act_score.py $A/cfg/A47_SEARCH_CFG.json $D/wm_jobs/WM_POOL_seed$N.json \
    $D/wm_scores/WM_SCORES_seed$N.json $D/wm_act_scores $A/ckpt/ckpt/wm_s.pt 10 $MODE \
    >> $D/wm_act_scores/run_act_${MODE}_seed$N.log 2>&1
done
echo WM_ACT_ALL_DONE $MODE >> $D/wm_act_scores/run_act_${MODE}.done
