#!/bin/bash
# A96 S5-p20 (gpuR RTX 4090 24 GB, 3 workers, no MPS) copy of: A86 S5 launcher run_search_s5.sh (2d3ecca6).
# Zero SUMO; frozen WM-S wm_s.pt 6da9d333; never trains.
#   run_search_s5p20.sh <cfg.json> <outdir>     one launch per out dir; refuses an existing dir
# Same as run_search_s5.sh except ONLY the cell: --cell D2800_p20, --states = registered p20 file (7860a5d1, as S3b;
# sha-checked), own install dir /root/autodl-tmp/a96_s5p20 (bundle copy a75, ship), STATES_SHA256 sidecar.
set -u
A=/root/autodl-tmp/a96_s5p20/a75
SHIP=/root/autodl-tmp/a96_s5p20/ship
STATES=/root/autodl-tmp/a96_s5p20/p20/regime_STATES_p20.json
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
PYTHONPATH=src:.:$SHIP /root/miniconda3/bin/python -B -m a86s5 --states "$STATES" --cell D2800_p20 \
  --config "$CFGF" --out "$OUT" --ckpt $A/ckpt/ckpt/wm_s.pt --map-spec maps/wz3to2/MAP_SPEC.json \
  --device cuda --max-rows 64 --workers 3 > "$M/run.log" 2>&1
echo $? > "$M/RC"
date +%s > "$M/T_END"
( cd "$OUT" && sha256sum STRATEGY_SEARCH_*.json LOG_*.jsonl ARMS.json TOP_OFFER_CONDITIONS_*.json NICHE_TOP_*.json > "$M/SHA256SUMS" 2>/dev/null )
