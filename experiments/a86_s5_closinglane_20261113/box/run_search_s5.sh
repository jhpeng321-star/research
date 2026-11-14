#!/bin/bash
# A86 S5 (gpuR RTX 4090 24 GB, 3 workers, no MPS) copy of: A85 S4 launcher run_search_s4.sh (293ed036).
# Zero SUMO; frozen WM-S wm_s.pt 6da9d333; never trains.
#   run_search_s5.sh <cfg.json> <outdir>     one launch per out dir; refuses an existing dir
# Same as run_search_s4.sh except: own bundle copy A=/root/autodl-tmp/a86/a75 (a75_search_box.tgz 9eb32080),
# ship /root/autodl-tmp/a86/ship (a75ss/a75_wmforms/wm_fix_crn/a85s4 unchanged + a86s5.py), module a86s5,
# --workers 3 (24 GB card: gpuR S1b OOM'd at 5 workers, 3.8-5.0 GB per worker), no MPS client env (no daemon on gpuR),
# NICHE_TOP_*.json added to SHA256SUMS. --max-rows 64 and everything that scores a plan are unchanged.
set -u
A=/root/autodl-tmp/a86/a75
SHIP=/root/autodl-tmp/a86/ship
CFGF=$1; OUT=$2; M=$OUT.meta
[ -e "$OUT" ] && { echo "REFUSE: $OUT exists"; exit 3; }
[ -e "$M" ] && { echo "REFUSE: $M exists"; exit 3; }
mkdir -p "$M"
cd $A/code_split/code_v12r7b
export WZPROP_SPLIT_ACTIONS=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
date +%s > "$M/T_START"
sha256sum "$CFGF" > "$M/CFG_SHA256"
sha256sum $SHIP/*.py > "$M/SHIP_SHA256"
PYTHONPATH=src:.:$SHIP /root/miniconda3/bin/python -B -m a86s5 --states data/regime_STATES.json --cell D2800_p60 \
  --config "$CFGF" --out "$OUT" --ckpt $A/ckpt/ckpt/wm_s.pt --map-spec maps/wz3to2/MAP_SPEC.json \
  --device cuda --max-rows 64 --workers 3 > "$M/run.log" 2>&1
echo $? > "$M/RC"
date +%s > "$M/T_END"
( cd "$OUT" && sha256sum STRATEGY_SEARCH_*.json LOG_*.jsonl ARMS.json TOP_OFFER_CONDITIONS_*.json NICHE_TOP_*.json > "$M/SHA256SUMS" 2>/dev/null )
