#!/usr/bin/env bash
# Launch WM-D shard-loop GPU workers on gpuA (only when the card is free of WM-E's training):
#   bash gpu_workers.sh CODE RUN CKPT_DIR TOML NWORK FRACTION MAXROWS "SHARDS..."
# CKPT_DIR holds wm_s.pt (+ physics.json) and TRAIN_WORLDS.json (the checkpoint's train/val worlds).
# TOML '-' = the checkpoint's own physics.json/config. Worker j gets every NWORK-th listed shard.
set -u
C=${1:?}; RUN=${2:?}; CK=${3:?}; TOML=${4:?}; NW=${5:?}; FR=${6:?}; MR=${7:?}; SHARDS=${8:?}
DATA=/root/autodl-fs/wzqual/data
mkdir -p "$RUN"
read -r -a ALL <<< "$SHARDS"
for j in $(seq 0 $((NW-1))); do
  L=""
  for ((i=j; i<${#ALL[@]}; i+=NW)); do L="$L ${ALL[$i]}"; done
  [ -z "$L" ] && continue
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WZQ_GPU_FRACTION=$FR WZQ_NICE=19 \
  WZQ_PYTHON=/root/miniconda3/bin/python /usr/bin/setsid nohup bash $C/tools/wzqual/ops/shard_loop.sh \
    g$j "$L" 80 $C "$DATA/tapes/runs/*/s*/prefix/TAPE.npz" $DATA "$RUN" "$CK/wm_s.pt" "$CK/TRAIN_WORLDS.json" \
    cuda 1 "$TOML" --identity $DATA/identity/IDENTITY.json --samples 4 --start-stride 240 --batch-starts 6 \
    --batch-moments 1 --max-rows "$MR" > "$RUN/g$j.log" 2>&1 < /dev/null &
done
sleep 5
echo "gpu workers: $(ls $RUN/g*.loop.pid 2>/dev/null | wc -l)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
