#!/usr/bin/env bash
# gpuR memory trial (probe only; no optimizer step, no checkpoint). Usage: r9_mem_trial.sh <repo_dir> <out_dir> <120|300> [old]
L=/root/autodl-tmp/c4/v5
REPO=$1; OUTD=$2; STAGE=$3; OLD=${4:-}
ulimit -n 65536
cd "$REPO" || exit 2
CK=$L/runs/c4_wm_v10_round8/wm_s_stage300.pt
export PYTHONPATH=src:. OMP_NUM_THREADS=12 WMS_JAM_WORKERS=12
export WMS_DATA_V5_NATIVE=$L/data/v5_native WMS_DATA_GENRAND_PLAIN=$L/data/genrand_plain WMS_DATA_GENRAND_BRANCH=$L/data/genrand_branch WMS_DATA_PAIRS300=$L/data/pairs300
export R9_PROBE_OUT=$OUTD/probe_${STAGE}${OLD}_b${B:-4}${TAG:-}.jsonl R9_PROBE_STEPS=${R9_PROBE_STEPS:-2}
[ -n "$OLD" ] && export R9_PROBE_OLD=1
mkdir -p $OUTD
TR=${TR:-120}; [ "$STAGE" = 300 ] && TR=60
/root/miniconda3/bin/python scripts/r9_mem_probe.py train --config configs/c4_wm_v10round9.toml --mode run --out $OUTD/run_${STAGE}${OLD}_b${B:-4} \
  --set model.init_from=$CK --set loss.loss_share_logging=false --set loss.brake_follow_cav_weight=2.0 --set optim.batch_size=${B:-4} --set model.checkpoint_forward=${CKF:-false} \
  --set "curriculum.unroll_s=[$STAGE.0]" --set "curriculum.stage_steps=[${NSTEPS:-3}]" --set "curriculum.truncation_steps_per_stage=[$TR]" \
  --set "optim.batch_size_per_stage=[${B:-4}]" --set "optim.lr_per_stage=[5.0e-5]" --set curriculum.max_hours=0.3 $EXTRA > $OUTD/log_${STAGE}${OLD}_b${B:-4}.txt 2>&1
echo exit=$? >> $OUTD/log_${STAGE}${OLD}_b${B:-4}.txt
