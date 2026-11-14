#!/bin/bash
# A75-S3 sanity (mechanics only, not evidence): tiny cfg, CPU only (CUDA hidden), 2 workers, p20 states. Zero SUMO.
set -u
P=/root/autodl-tmp/a75p20; A=$P/a75; OUT=$P/sanity_tiny_cpu
[ -e "$OUT" ] && { echo "REFUSE: $OUT exists"; exit 3; }
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WZPROP_SPLIT_ACTIONS=32
cd $A/code_split/code_v12r7b
PYTHONPATH=src:.:$A/a75ship nice -n 19 /root/miniconda3/bin/python -B -m a75ss --states $P/p20/regime_STATES_p20.json \
  --cell D2800_p20 --config $P/cfg/SANITY_TINY_CPU_p20.json --out "$OUT" --ckpt $A/ckpt/ckpt/wm_s.pt \
  --map-spec maps/wz3to2/MAP_SPEC.json --device cpu --max-rows 64 --workers 2
echo "SANITY rc $?"
