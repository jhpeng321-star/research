#!/bin/bash
# E1-FIX (Amendment 74, POST HOC REPAIR): rescoring of the registered pools with the frozen WM, three registered fixes.
# GPU box only, zero SUMO. Finished seeds are kept on a re-launch; nothing existing is overwritten.
#   run_wm_fix.sh preflight            zero rollouts, prints the plan per seed
#   run_wm_fix.sh main                 screen (K_screen) + final (2 x K_final), max_rows 64 -> wm_fix_scores/
#   run_wm_fix.sh repeat               gate G0: final stage again, max_rows 32          -> wm_fix_scores_repeat/
# optional: SEEDS="11 23" WORKERS=10 SCORE_FIELD=score
MODE=${1:?preflight | main | repeat}
A=/root/autodl-tmp/a68
D=$A/claim_validation_20261113_0231
SEEDS=${SEEDS:-"11 23 42 67 89 103"}
WORKERS=${WORKERS:-10}
SCORE_FIELD=${SCORE_FIELD:-score}
set -e
case $MODE in
  preflight|main) O=$D/wm_fix_scores; X="--max-rows 64" ;;
  repeat) O=$D/wm_fix_scores_repeat; X="--max-rows 32 --main-dir $D/wm_fix_scores" ;;
  *) echo "mode must be preflight, main or repeat"; exit 2 ;;
esac
mkdir -p $O
cd $A/code_split/code_v12r7b
for N in $SEEDS; do
  WZPROP_SPLIT_ACTIONS=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src:.:$D nice -n 5 \
    /root/miniconda3/bin/python -B $D/wm_fix_score.py $MODE $A/cfg/A47_SEARCH_CFG.json $D/wm_jobs/WM_POOL_seed$N.json \
    $O $A/ckpt/ckpt/wm_s.pt --workers $WORKERS --score-field $SCORE_FIELD $X \
    >> $O/run_fix_${MODE}_seed$N.log 2>&1
done
echo WM_FIX_ALL_DONE $MODE "$SEEDS" >> $O/run_fix_${MODE}.done
