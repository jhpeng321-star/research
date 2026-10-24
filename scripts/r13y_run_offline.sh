#!/bin/sh
# Inference only. Parent directory must already exist; output must be fresh.
set -eu
ROOT=/root/autodl-tmp/c4/v5/diag_yield
test -d "$ROOT"
test ! -e "$ROOT/r13y_offline_final"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
nohup nice -n 19 /root/miniconda3/bin/python -B "$ROOT/code_final/scripts/diag_r13y_offline.py" \
  --repo "$ROOT/code_final" --factorial-dir /root/autodl-tmp/c4/v5/diag_blowup \
  --checkpoint /root/autodl-tmp/c4/v5/runs/c4_wm_v10_round12c/wm_s.pt \
  --states /root/autodl-tmp/c4/v5/chk_r10b/states192.json.gz \
  --train-states-sha256 6b7d15f6081ad703339c9e149b217cf382e8cf0b8aa83fb3ab13b786df3a37ec \
  --spot-states "$ROOT/r13y_spot300_train.json.gz" \
  --train-spot-sha256 4920072649a44fb76c5aacc385407151f9091779a61a15eb52d4d36157b99a04 \
  --out "$ROOT/r13y_offline_final" --workers 4 --run > "$ROOT/r13y_offline_final.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$ROOT/r13y_offline_final.pid"
printf 'started own CPU-only inference PID %s\n' "$pid"
nohup sh "$ROOT/code_final/scripts/r13y_verify.sh" > "$ROOT/verification_final.log" 2>&1 < /dev/null &
verify_pid=$!
printf '%s\n' "$verify_pid" > "$ROOT/verification_final.pid"
printf 'queued own sequential verification PID %s\n' "$verify_pid"
