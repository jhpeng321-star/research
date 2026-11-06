#!/bin/bash
# WM-C morning chain (2026-11-07). Steps are run BY HAND in order; each line is one step.
# Preconditions:
#   (G) GPU steps only after WM-E's fine-tune has finished (STATUS.json state != training).
#   (Q) The SUMO step only after WM_QUAL_GATE PASS on the WM-E final checkpoint (S-WMSEARCH amendment 1).
set -u
B=/root/autodl-fs/wmc_rankcheck_20261106          # gpuA work root (code_v3 = port 1d153d96 + wzprop)
FINAL=${FINAL:?set FINAL=/root/autodl-fs/wzwm/ft_r1/<final dir>/wm_s.pt}

# --- gpuA (G): refresh code_v3/tools/wzprop from the latest c4/wz-wm-propose archive first
# 1) final rank check, full protocol (PLANGEN K8 + solos for S-labelled) + the claim-4 regime check (T only)
#    bash $B/run_ckpt.sh wme_final $FINAL 4 4
# 2) strategy-level search vs solo2 per demand cell (zero SUMO), parallel WM workers on the free GPU
cd $B/code_v3 && PYTHONPATH=src:. nohup nice -n 10 /root/miniconda3/bin/python -B -m tools.wzprop strategy \
  --states data/regime_STATES.json --cells D3200_p60,D2800_p60,D2400_p60,D3200_p40 --out $B/strategy_final \
  --wm wma:WMA_READY.json --wm-arg checkpoint=$FINAL --wm-arg map_spec=maps/wz3to2/MAP_SPEC.json \
  --wm-arg device=cuda --wm-arg max_rows=48 --workers 6 --mem-fraction 0.12 > $B/strategy_final.log 2>&1 &
# -> $B/strategy_final/ARMS.json (wm / structured / random / incumbent per cell), STRATEGY_SEARCH_<cell>.json

# --- cpuV (Q): S-WMSEARCH amendment 1, MAPPO u0 base via the S-REGIMEU0 harness
#   S0=/home/ubuntu/imitreplay_20261106/src_52248a4b; W=/home/ubuntu/wmsearch_20261107
#   mkdir -p $W && cp -a $S0 $W/src && tar -xf wzprop_<sha>.tar -C $W/src/pytools   (tools/wzprop into pytools/tools)
#   . /home/ubuntu/cpus_env/env.sh $W/src
#   export COOPMWM_MAP_SPEC=$W/src/pytools/assets/research/newmap_candidates/wz3to2/MAP_SPEC.json COOPMWM_G5_RUNTIME=1 CUDA_VISIBLE_DEVICES=
#   cd $W/src/pytools
#   python -B -m tools.wzprop.wmsearch_u0 plan --root $W/root --regimeu0-root /home/ubuntu/imitreplay_20261106/regimeu0_v5 \
#       --arms $W/ARMS.json --cell D3200_p60 --seeds 989770-989785          (268,800 sim-s <= 280,000)
#   nohup nice -n 19 python -B -m tools.wzprop.wmsearch_u0 run --root $W/root --workers 16 > $W/run.log 2>&1 &
#   python -B -m tools.wzprop.wmsearch_u0 analyze --root $W/root --out $W/ANALYSIS.json
