#!/usr/bin/env bash
# WM-D: score one checkpoint with the IDENTICAL qualification command as the WM2 baseline (runs ON gpuA).
#   bash run_ckpt.sh CODE RUN_NAME CHECKPOINT WORLDS_MANIFEST [TOML|-]
# Layout (gpuA cgroup quota 20 cores, card shared with WM-E): 4 CPU distributional shards x 1 thread
# + 1 GPU action-response process (memory fraction 0.25 ~ 11.8 GB, <= 80 rows per call), all nice 19.
# TOML '-' (default) = the checkpoint's own physics.json/config (fine-tuned WM-E checkpoints);
# the WM2 baseline used configs/c4_wm_wz3to2_ft_b.toml (no physics.json next to WM2).
# Merge after all five shards finished:
#   cd CODE && PYTHONPATH=$CODE/src:$CODE python -m tools.wzqual.run_qual merge \
#       --out /root/autodl-fs/wzqual/runs/RUN_NAME/FINAL --shards /root/autodl-fs/wzqual/runs/RUN_NAME/*_s?
set -euo pipefail
C=${1:?CODE}; NAME=${2:?RUN_NAME}; CKPT=${3:?CHECKPOINT}; MAN=${4:?WORLDS_MANIFEST}; TOML=${5:--}
D=/root/autodl-fs/wzqual/runs/$NAME
DATA=/root/autodl-fs/wzqual/data
T="$DATA/tapes/runs/*/s*/prefix/TAPE.npz"
L=$C/tools/wzqual/ops/qual_launch.sh
export WZQ_PYTHON=${WZQ_PYTHON:-/root/miniconda3/bin/python}
[ -e "$D" ] && { echo "REFUSED: $D exists"; exit 3; }
mkdir -p "$D"
sha256sum "$CKPT" > "$D/CHECKPOINT.sha256"
for i in 0 1 2 3; do
  bash "$L" "$C" "$T" "$DATA" "$D/dist_s$i" "$CKPT" "$MAN" cpu 1 "$TOML" \
    --identity "$DATA/identity/IDENTITY.json" --skip-action --start-shard "$i/4" --batch-starts 1
done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WZQ_GPU_FRACTION=0.25 bash "$L" "$C" "$T" "$DATA" \
  "$D/act_s0" "$CKPT" "$MAN" cuda 1 "$TOML" --identity "$DATA/identity/IDENTITY.json" \
  --skip-distributional --batch-moments 1 --max-rows 80
