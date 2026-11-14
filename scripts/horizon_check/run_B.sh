#!/usr/bin/env bash
# Part B queue: 16 episodes (seeds 4121-4128 x penetration 0.2/0.4), N concurrent workers.
# usage: bash scripts/horizon_check/run_B.sh <N> [job-list...]   (job = seed:pen)
# WM tags: C4 / C4R = driver point estimate (scorer path, n_samples 1).
# The first 4 episodes (s4121-4124 p0.2) ran C4/C4R in sampling mode before the sampling bias
# was found; reimagine_B.py adds point-estimate tags to them by replaying the native base.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=src C4_ALLOWED_ROOTS=E:/research-dense OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       OPENBLAS_NUM_THREADS=1 TEMP=E:/agent-work/tmp TMP=E:/agent-work/tmp
PY=C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe
R=E:/research-dense/runs/active
B=runs/active/C4-HORIZON-20261013/B
mkdir -p $B
N=$1; shift
JOBS="$*"
if [ -z "$JOBS" ]; then
  for p in 0.2 0.4; do for s in 4121 4122 4123 4124 4125 4126 4127 4128; do JOBS="$JOBS $s:$p"; done; done
fi
one() {
  s=${1%%:*}; p=${1##*:}
  [ -e $B/s${s}_p${p}/DONE.json ] && exit 0
  $PY scripts/horizon_check/branch_horizon.py --seed $s --penetration $p --out $B/s${s}_p${p} \
    --controllers random damping $R/WM-S-C4-20261013/fetched/cpuT_mappo_main/policy_latest.pt \
    --labels random damping mappo \
    --wm C4=$R/WM-S-C4-20261013/fetched/gpuW/c4_wm_main/wm_s_best.pt,mean,1 \
    --wm C4R=$R/WM-S-C4-20261013/fetched/gpuW/loop_it1_refresh/wm_s_best.pt,mean,1 \
    > $B/s${s}_p${p}.log 2>&1
}
export -f one; export PY R B
echo $JOBS | tr ' ' '\n' | xargs -P $N -I{} bash -c 'one {}'
