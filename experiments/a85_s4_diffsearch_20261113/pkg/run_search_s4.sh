#!/bin/bash
# A85 S4 (MPS env, 14 workers) copy of: A79 S2b launcher /root/autodl-tmp/a75/run_search_s2b.sh (f6c4afeb).
# Zero SUMO; frozen WM-S wm_s.pt 6da9d333; never trains.
#   run_search_s4.sh <cfg.json> <outdir>     one launch per out dir; refuses an existing dir
# Same as run_search_s2b.sh except: own bundle copy A=/root/autodl-tmp/a85/a75 (a75_search_box.tgz 9eb32080,
# like a75p20/a75), PYTHONPATH ship = /root/autodl-tmp/a85/a85ship (a75ss/a75_wmforms/wm_fix_crn unchanged + a85s4.py),
# module a85s4 (a75ss.main routed through the S4 lane-set filter) instead of a75ss.
set -u
A=/root/autodl-tmp/a85/a75
SHIP=/root/autodl-tmp/a85/a85ship
CFGF=$1; OUT=$2; M=$OUT.meta
[ -e "$OUT" ] && { echo "REFUSE: $OUT exists"; exit 3; }
[ -e "$M" ] && { echo "REFUSE: $M exists"; exit 3; }
mkdir -p "$M"
cd $A/code_split/code_v12r7b
export WZPROP_SPLIT_ACTIONS=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
date +%s > "$M/T_START"
sha256sum "$CFGF" > "$M/CFG_SHA256"
sha256sum $SHIP/*.py > "$M/SHIP_SHA256"
PYTHONPATH=src:.:$SHIP /root/miniconda3/bin/python -B -m a85s4 --states data/regime_STATES.json --cell D2800_p60 \
  --config "$CFGF" --out "$OUT" --ckpt $A/ckpt/ckpt/wm_s.pt --map-spec maps/wz3to2/MAP_SPEC.json \
  --device cuda --max-rows 64 --workers 14 > "$M/run.log" 2>&1
echo $? > "$M/RC"
date +%s > "$M/T_END"
( cd "$OUT" && sha256sum STRATEGY_SEARCH_*.json LOG_*.jsonl ARMS.json TOP_OFFER_CONDITIONS_*.json > "$M/SHA256SUMS" 2>/dev/null )
