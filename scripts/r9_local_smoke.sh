#!/usr/bin/env bash
# LOCAL CPU micro-smoke for the round-9 recipe (Git Bash on Windows). Not a gpuR script.
# Usage: scripts/r9_local_smoke.sh <data_dir> <ckpt> <out_dir> [extra wms --set args...]
#   <data_dir> must hold v5_native/ and genrand_plain/ (trimmed C4 manifests, >=1 train and 1 val episode each).
# Uses the REAL entrypoint (python -m wms train --config configs/c4_wm_v10round9.toml --mode run); only data sources,
# paired terms (pairs300 absent locally), validation size, device and the curriculum are overridden.
set -euo pipefail
DATA=$1; CK=$2; OUT=$3; shift 3
PY=${PY:-C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe}
export PYTHONPATH="src;." OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} WMS_LATEST_EVERY=1
export WMS_DATA_V5_NATIVE=$DATA/v5_native WMS_DATA_GENRAND_PLAIN=$DATA/genrand_plain
exec $PY ${R9_ENTRY:--m wms} train --config configs/c4_wm_v10round9.toml --mode run --out "$OUT" \
  --set 'data.c4_sources=[{name="native_v5",path="v5_native",mix_ratio=0.6,window_stride_s=10.0,use_val=true},{name="genrand_plain",path="genrand_plain",mix_ratio=0.4,window_stride_s=10.0,use_val=true}]' \
  --set loss.agg_delay.paired_weight=0.0 --set loss.paired_diff.weight=0.0 \
  --set selection.val_batches=1 --set selection.val_max_windows=2 \
  --set "model.init_from=\"$CK\"" --set loss.loss_share_logging=false --set loss.brake_follow_cav_weight=2.0 \
  --set optim.batch_size=4 --set 'runtime.device="cpu"' \
  --set "curriculum.unroll_s=[30.0,60.0]" --set "curriculum.stage_steps=[2,2]" \
  --set "curriculum.truncation_steps_per_stage=[60,60]" --set curriculum.max_hours=0.5 \
  --set optim.batch_size_per_stage=[1,1] "$@"
