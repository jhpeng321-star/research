#!/bin/bash
# WMEV morning runbook (2026-11-07): claim 4a = "the WM predicts the expected benefit of a cooperation strategy by
# traffic condition" on the MAPPO u0 base, vs S-REGIMEU0 + S-LOWSHARE-U0 (SUMO).  Zero SUMO.  Steps are run BY HAND,
# in order; each block is one step.  Code: branch c4/wmev (tools/wzprop/regime_u0.py, regime_u0_compare.py).
#
# Preconditions (all must hold before ANY GPU step):
#   (G1) WM-E finished: /root/autodl-fs/wzwm/ft_r1/STATUS.json "state" != "training" and runner PID 416453 gone.
#   (G2) Shared gpuA (20-core quota, rollouts CPU-bound, 1 core per process): WM-D qualification has priority;
#        WM-C <= 8 processes, WMEV <= 6 processes, nice 10 (WM-C/WMEV agreement 2026-11-07).  If WM-D's or WM-C's
#        step time degrades clearly, halve the WMEV workers (stop own PIDs from PIDS.txt only, never by name).
#   (G3) FINAL = WM-E final checkpoint (/root/autodl-fs/wzwm/ft_r1/phases/B/a*/run/wm_s.pt, CHECKPOINT_MANIFEST.json)
#        with its physics.json / train_config.json beside it.
set -u
SHA=${SHA:?set SHA=<c4/wmev commit from WMEV_READY.json>}
FINAL=${FINAL:?set FINAL=/root/autodl-fs/wzwm/ft_r1/phases/B/a1/run/wm_s.pt}
B=/root/autodl-fs/wmev_20261107
U0=/root/wzwmdata20261106/inputs/policy_00169.pt            # speed-only MAPPO169, sha256 d9042cc5...
PY=/root/miniconda3/bin/python

# --- 0 (LOCAL, git-bash on the Windows host): archive (LF: autocrlf off) + chunked upload (scp drops > ~3 MB)
#   cd E:/agent-work/c4-wmev/_wmev
#   git -C E:/research-dense -c core.autocrlf=false archive --format=tar.gz -o wmev_$SHA.tgz $SHA -- src configs \
#       tools/wzprop tools/coopmwm assets/research/newmap_candidates/wz3to2 wmev_data ops/wmev_morning.sh
#   split -b 1m wmev_$SHA.tgz part_ && ssh gpuA "mkdir -p $B/run && rm -f $B/code.tgz" && \
#       for f in part_*; do ssh gpuA "cat >> $B/code.tgz" < $f; done && rm part_*
#   sha256sum wmev_$SHA.tgz; ssh gpuA "sha256sum $B/code.tgz"          # must match

# --- 1 (gpuA): unpack + input identity checks (refuse on any mismatch)
mkdir -p $B/code $B/run && tar -xzf $B/code.tgz -C $B/code
sha256sum $U0 | grep -q '^d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca' || echo 'REFUSE: u0 sha'
sha256sum $B/code/wmev_data/u0_STATES_t300.json | grep -q '^fface5533530da8e' || echo 'REFUSE: states sha'
sha256sum $FINAL | tee $B/CHECKPOINT_SHA256.txt
cd $B/code
export PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       COOPMWM_MAP_SPEC=$B/code/assets/research/newmap_candidates/wz3to2/MAP_SPEC.json

# --- 2 (gpuA CPU, any time): CPU tests on the real checkpoint (~2-3 min; tapes test skipped off the Windows host)
WMEV_U0=$U0 WMEV_WM2=$FINAL $PY -m pytest -q tools/wzprop/test_regime_u0_cpu.py -p no:cacheprovider

# --- 3 (gpuA GPU, G1-G3): calibrate ONE call at the planned packing (~3-5 min): s/step + peak memory
nice -n 10 $PY -B -m tools.wzprop.regime_u0 cost --calibrate --device cuda --threads 1 --mem-fraction 0.10 \
  --states wmev_data/u0_STATES_t300.json --checkpoint $FINAL --u0 $U0 --arms auto --K 8 --max-rows 32 \
  --steps 40 --workers 6 | tee $B/CALIBRATION.json
# Registered plan choice (before any result): the FIRST plan whose estimate.wall_minutes <= 90 with 6 workers
# (every plan has <= 32 rows per call, so the one calibration above applies to all of them):
#   P1  "--K 8"                                           61 states, H 900   (1 state per call)
#   P2  "--K 4 --states-per-call 2"                       61 states, H 900
#   P3  "--K 4 --states-per-call 2 --max-per-cell 6"      40 states, H 900
#   P4  "--K 4 --states-per-call 2 --max-per-cell 6 --H 600"  (the comparison is then [300, 900) s on BOTH sides)
# Cost of each plan: re-run step 3 WITHOUT --calibrate (and without --K 8) with --s-per-step <calibration.s_per_step>
# and the plan flags.  Night prior (CPU-only build, WM-C's GPU rate): 0.3-0.6 s/step -> P1 98-196 min, P2 50-100,
# P3 29-59, P4 20-39 min wall (6 workers, efficiency 0.8).
# Memory: calibration.memory.peak_gb_full_H must stay below 0.10 x 48 GB = 4.8 GB, else use --max-rows 16 (and
# halve K or states-per-call accordingly; redo step 3).
PLAN_FLAGS=${PLAN_FLAGS:?set PLAN_FLAGS to the chosen plan, e.g. "--K 4 --states-per-call 2"}

# --- 4 (gpuA GPU): 6 shard workers (resumable: re-running the same command skips scored states)
for i in 0 1 2 3 4 5; do
  nohup nice -n 10 $PY -B -m tools.wzprop.regime_u0 score --device cuda --threads 1 --mem-fraction 0.10 \
    --states wmev_data/u0_STATES_t300.json --checkpoint $FINAL --u0 $U0 --arms auto --max-rows 32 \
    --shard $i/6 --out $B/run/SCORES_part$i.jsonl $PLAN_FLAGS > $B/run/part$i.log 2>&1 &
  echo "$! part$i $(date +%s)" >> $B/run/PIDS.txt
done
# Monitor: wc -l $B/run/SCORES_part*.jsonl ; grep -h '"error"' $B/run/SCORES_part*.jsonl | head ; nvidia-smi
# Stop (only if needed): kill <pid> for the PIDs in $B/run/PIDS.txt; never pkill/taskkill by name.

# --- 5 (gpuA GPU, optional, only if step 4 finished early): horizon check vs the same worlds' SUMO native tapes
#   nohup nice -n 10 $PY -B -m tools.wzprop.regime_u0 score --device cuda --threads 1 --mem-fraction 0.10 \
#     --states wmev_data/u0_STATES_t300.json --checkpoint $FINAL --u0 $U0 --arms u0,native --K 2 --max-rows 32 \
#     --max-per-cell 3 --out $B/run/NATIVE_part0.jsonl > $B/run/native0.log 2>&1 &

# --- 6 (LOCAL, zero GPU): copy back + analyze into a NEW directory
#   D=E:/agent-work/c4-wmev/wmev_runs/final_$(date +%H%M) ; mkdir -p $D
#   for i in 0 1 2 3 4 5; do ssh gpuA "cat $B/run/SCORES_part$i.jsonl" > $D/SCORES_part$i.jsonl; \
#       ssh gpuA "cat $B/run/SCORES_part$i.jsonl.meta.json" > $D/SCORES_part$i.jsonl.meta.json; done
#   (optional) ssh gpuA "cat $B/run/NATIVE_part0.jsonl" > $D/NATIVE_part0.jsonl
#   cd E:/agent-work/c4-wmev && PYTHONPATH="src;." python -m tools.wzprop.regime_u0_compare analyze \
#       $(for i in 0 1 2 3 4 5; do echo --scores $D/SCORES_part$i.jsonl; done) [--scores $D/NATIVE_part0.jsonl] \
#       --sumo-windows wmev_data/SUMO_U0_WINDOWS.json --native-ref wmev_data/u0_NATIVE_REF_t300.json \
#       --label WME_final --out $D/WMEV_RESULT.json
#   -> WMEV_RESULT.md: per cell x strategy WM vs SUMO benefit, sign / rank / calibration / regime pattern /
#      group-vs-best-solo, at H = the stable horizon H* (horizon check), plus all H in WMEV_RESULT.json "all_H".
