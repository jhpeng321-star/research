#!/bin/bash
# A79 S2b (MPS, 14 workers) copy of: A75 hold-forms WM search launcher (zero SUMO; frozen WM-S wm_s.pt 6da9d333; never trains)
#   run_search_a75.sh <cfg.json> <outdir>     one launch per out dir; refuses an existing dir
# Code tree code_split/code_v12r7b (strategy_search 02e119c1 untouched) + a75ship/ (a75ss, a75_wmforms, wm_fix_crn).
# Sidecars go to <outdir>.meta/ (T_START, CFG_SHA256, run.log, RC, T_END, SHA256SUMS).
set -u
A=/root/autodl-tmp/a75
CFGF=$1; OUT=$2; M=$OUT.meta
[ -e "$OUT" ] && { echo "REFUSE: $OUT exists"; exit 3; }
[ -e "$M" ] && { echo "REFUSE: $M exists"; exit 3; }
mkdir -p "$M"
cd $A/code_split/code_v12r7b
export WZPROP_SPLIT_ACTIONS=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
date +%s > "$M/T_START"
sha256sum "$CFGF" > "$M/CFG_SHA256"
PYTHONPATH=src:.:$A/a75ship /root/miniconda3/bin/python -B -m a75ss --states data/regime_STATES.json --cell D2800_p60 \
  --config "$CFGF" --out "$OUT" --ckpt $A/ckpt/ckpt/wm_s.pt --map-spec maps/wz3to2/MAP_SPEC.json \
  --device cuda --max-rows 64 --workers 14 > "$M/run.log" 2>&1
echo $? > "$M/RC"
date +%s > "$M/T_END"
( cd "$OUT" && sha256sum STRATEGY_SEARCH_*.json LOG_*.jsonl ARMS.json TOP_OFFER_CONDITIONS_*.json > "$M/SHA256SUMS" 2>/dev/null )
