#!/bin/bash
# A75-S3 (new hold forms, 20 % CAVs) WM search launcher. Zero SUMO; frozen WM-S wm_s.pt 6da9d333; never trains.
#   run_search_s3.sh <cfg.json> <outdir>    one launch per out dir; refuses an existing dir
# Same as a75/run_search_a75.sh (66cef683) except: own bundle copy A=/root/autodl-tmp/a75p20/a75,
# --states = registered p20 file (7860a5d1, copied from laptop), --cell D2800_p20.
set -u
A=/root/autodl-tmp/a75p20/a75
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
PYTHONPATH=src:.:$A/a75ship /root/miniconda3/bin/python -B -m a75ss --states "$STATES" --cell D2800_p20 \
  --config "$CFGF" --out "$OUT" --ckpt $A/ckpt/ckpt/wm_s.pt --map-spec maps/wz3to2/MAP_SPEC.json \
  --device cuda --max-rows 64 --workers 10 > "$M/run.log" 2>&1
echo $? > "$M/RC"
date +%s > "$M/T_END"
( cd "$OUT" && sha256sum STRATEGY_SEARCH_*.json LOG_*.jsonl ARMS.json TOP_OFFER_CONDITIONS_*.json > "$M/SHA256SUMS" 2>/dev/null )
