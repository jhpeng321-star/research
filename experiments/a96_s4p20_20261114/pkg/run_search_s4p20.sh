#!/bin/bash
# A96 S4-p20: exact copy of the A85 S4 launcher /root/autodl-tmp/a85/run_search_s4.sh (293ed036), cell D2800_p20.
# Zero SUMO; frozen WM-S wm_s.pt 6da9d333; never trains.
#   run_search_s4p20.sh <cfg.json> <outdir>     one launch per out dir; refuses an existing dir
# Same as run_search_s4.sh except: own bundle copy A=/root/autodl-tmp/a96_s4p20/a75 (a75_search_box.tgz 9eb32080),
# ship /root/autodl-tmp/a96_s4p20/a85ship (byte copy of a85/a85ship), --states = registered p20 file (7860a5d1,
# read only from a75p20, sha-checked and recorded as in run_search_s3b.sh), --cell D2800_p20.
set -u
A=/root/autodl-tmp/a96_s4p20/a75
SHIP=/root/autodl-tmp/a96_s4p20/a85ship
STATES=/root/autodl-tmp/a75p20/p20/regime_STATES_p20.json
CFGF=$1; OUT=$2; M=$OUT.meta
[ -e "$OUT" ] && { echo "REFUSE: $OUT exists"; exit 3; }
[ -e "$M" ] && { echo "REFUSE: $M exists"; exit 3; }
[ "$(sha256sum "$STATES" | cut -c1-64)" = 7860a5d194fa95bec102b8f113c4565606c09a5cde523a42a996ae20e67d4f8a ] \
  || { echo "REFUSE: states sha"; exit 4; }
mkdir -p "$M"
cd $A/code_split/code_v12r7b
export WZPROP_SPLIT_ACTIONS=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
date +%s > "$M/T_START"
sha256sum "$CFGF" > "$M/CFG_SHA256"
sha256sum "$STATES" > "$M/STATES_SHA256"
sha256sum $SHIP/*.py > "$M/SHIP_SHA256"
PYTHONPATH=src:.:$SHIP /root/miniconda3/bin/python -B -m a85s4 --states "$STATES" --cell D2800_p20 \
  --config "$CFGF" --out "$OUT" --ckpt $A/ckpt/ckpt/wm_s.pt --map-spec maps/wz3to2/MAP_SPEC.json \
  --device cuda --max-rows 64 --workers 14 > "$M/run.log" 2>&1
echo $? > "$M/RC"
date +%s > "$M/T_END"
( cd "$OUT" && sha256sum STRATEGY_SEARCH_*.json LOG_*.jsonl ARMS.json TOP_OFFER_CONDITIONS_*.json > "$M/SHA256SUMS" 2>/dev/null )
