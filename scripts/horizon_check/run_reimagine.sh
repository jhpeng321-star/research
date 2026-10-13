#!/usr/bin/env bash
# Adds point-estimate WM tags (C4, C4R) to the 4 sampling-mode episodes once each is DONE.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=src C4_ALLOWED_ROOTS=E:/research-dense OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       OPENBLAS_NUM_THREADS=1 TEMP=E:/agent-work/tmp TMP=E:/agent-work/tmp
PY=C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe
R=E:/research-dense/runs/active
B=runs/active/C4-HORIZON-20261013/B
one() {
  e=$B/$1
  while [ ! -e $e/DONE.json ]; do sleep 60; done
  [ -e $e/reimagine_pe.jsonl ] && exit 0
  $PY scripts/horizon_check/reimagine_B.py --episode $e --suffix pe \
    --wm C4=$R/WM-S-C4-20261013/fetched/gpuW/c4_wm_main/wm_s_best.pt,mean,1 \
    --wm C4R=$R/WM-S-C4-20261013/fetched/gpuW/loop_it1_refresh/wm_s_best.pt,mean,1 \
    > $B/reimagine_$1.log 2>&1
}
export -f one; export PY R B
printf "s4121_p0.2\ns4122_p0.2\ns4123_p0.2\ns4124_p0.2\n" | xargs -P 4 -I{} bash -c 'one {}'
