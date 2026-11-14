#!/bin/bash
cd /home/ubuntu/trigclf_20261109/src
export COOPMWM_MAP_SPEC=/home/ubuntu/trigclf_20261109/src/assets/research/newmap_candidates/wz3to2/MAP_SPEC.json CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
D=/home/ubuntu/demox_7c7211d7
nice -n 19 /home/ubuntu/miniconda3/bin/python -m tools.coopmwm.learned_trigger collect --plan $D/PLAN_wm_host.json --guidance $D/GUIDANCE_UNION_R6_UNC_NOF2B_PRIORITY.json --tapes $D/tapes_wm --workers 8 --every 5 --out /home/ubuntu/trigclf_20261109/CANDS.jsonl.gz > /home/ubuntu/trigclf_20261109/collect.log 2>&1
echo "EXIT $?" >> /home/ubuntu/trigclf_20261109/collect.log
