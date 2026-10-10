#!/usr/bin/env bash
# LPWO LF comparators. Authored only; execution needs separately staged TRAIN data.
# Local flag identity: black_box_traj=--blackbox-traj; black_box_id=--black-box.
# External run/receipt flag identity: PENDING outside receipt confirmation.
# MAX_HOURS is a TOTAL wall-clock cap including BOTH models, smoke and gates.
set -euo pipefail

GPU=${GPU:-gpuA}
case "$GPU" in
  gpuA|gpuB) ;;
  *) echo "GPU must be gpuA or gpuB" >&2; exit 2 ;;
esac
: "${PY:?set PY to the staged host Python executable}"
: "${CODE:?set CODE to the staged repository root}"
: "${OUT:?set OUT to a new chain output directory}"
: "${TRAIN_WINDOWS:?set TRAIN_WINDOWS to windows_train_noPI_noWall.json}"
: "${VAL_WINDOWS:?set VAL_WINDOWS to windows_val_noPI_noWall.json}"
: "${EXCLUDED_WINDOWS:?set EXCLUDED_WINDOWS to excluded_runs_PI_wall.json}"
: "${BASE_PACK:?set BASE_PACK}"
: "${BASE_IDENTITIES:?set BASE_IDENTITIES}"
: "${TRAIN_OUTCOMES:?set TRAIN_OUTCOMES}"
: "${DSPEED_PACK:?set DSPEED_PACK}"
: "${DSPEED_IDENTITIES:?set DSPEED_IDENTITIES}"
: "${DSPEED_OVERLAY:?set DSPEED_OVERLAY}"
: "${DAMPING_PACK:?set DAMPING_PACK}"
: "${DAMPING_IDENTITIES:?set DAMPING_IDENTITIES}"
: "${DAMPING_OVERLAY:?set DAMPING_OVERLAY}"
MAX_HOURS=${MAX_HOURS:-8}
SMOKE_STAGE_STEPS=10,10,10,10
FULL_STAGE_STEPS=2500,2500,1875,1250
STAGES=10,30,60,120
# Default to no loader children: 10 workers with trainer prefetch_factor=4 can
# queue 40 batches plus duplicated dataset/process overhead. Override only with
# an explicit RAM allocation; this chain caps loader workers and CPU threads at 4.
NUM_WORKERS=${NUM_WORKERS:-0}
CPU_THREADS=${CPU_THREADS:-4}
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS"
command -v timeout >/dev/null || { echo "GNU timeout required for hard cap" >&2; exit 2; }
for name in PY CODE OUT TRAIN_WINDOWS VAL_WINDOWS EXCLUDED_WINDOWS BASE_PACK BASE_IDENTITIES TRAIN_OUTCOMES DSPEED_PACK DSPEED_IDENTITIES DSPEED_OVERLAY DAMPING_PACK DAMPING_IDENTITIES DAMPING_OVERLAY; do
  [[ "${!name}" == /* ]] || { echo "$name must be an absolute staged-host path" >&2; exit 2; }
done
for f in "$TRAIN_WINDOWS" "$VAL_WINDOWS" "$EXCLUDED_WINDOWS"; do
  [[ -f "$f" ]] || { echo "missing window list: $f" >&2; exit 2; }
done
[[ -d "$CODE/output/wd01/wm_n1/wms" ]] || { echo "missing trainer" >&2; exit 2; }
[[ -d "$(dirname "$OUT")" ]] || { echo "OUT parent must exist" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "refusing existing OUT: $OUT" >&2; exit 2; }
CAP_S=$("$PY" -c 'import math,sys; h=float(sys.argv[1]); assert math.isfinite(h) and 0<h<=8, "MAX_HOURS must be in (0,8]"; print(max(1,int(h*3600)))' "$MAX_HOURS")
"$PY" -c 'import sys; w,t=map(int,sys.argv[1:]); assert 0<=w<=4 and 1<=t<=4, "NUM_WORKERS in [0,4], CPU_THREADS in [1,4] required"' "$NUM_WORKERS" "$CPU_THREADS"
START_S=$(date +%s)
DEADLINE_S=$((START_S + CAP_S))
mkdir "$OUT"
OUT=$(cd "$OUT" && pwd)
log(){ printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$OUT/chain.log"; }
trap 'rc=$?; log "exit rc=$rc gpu=$GPU elapsed_s=$(($(date +%s)-START_S)) cap_s=$CAP_S"' EXIT
log "LF comparator flags pending outside receipt confirmation; black_box_traj=--blackbox-traj black_box_id=--black-box"
log "gpu=$GPU cap_s=$CAP_S stages=$STAGES stage_steps=$FULL_STAGE_STEPS"
log "cpu_threads=$CPU_THREADS loader_workers=$NUM_WORKERS; aggregate host RAM unmeasured"
export WM_N1_TRAIN_OUTCOMES="$TRAIN_OUTCOMES"
export WM_N1_TRAIN_IDENTITIES="$BASE_IDENTITIES"
cd "$CODE/output/wd01/wm_n1"

# Remaining-time timeout applies to EACH subprocess, with no post-cap kill grace.
# Foreground mode is deliberately absent so timeout also kills loader children.
bounded(){
  local remaining_s=$((DEADLINE_S - $(date +%s)))
  (( remaining_s > 0 )) || { log "MAX_HOURS exhausted"; return 124; }
  timeout --signal=KILL "${remaining_s}s" "$@"
}
run(){
  local dest=$1 tag=$2 steps=$3
  local flags=()
  case "$tag" in
    black_box_traj) flags=(--blackbox-traj) ;;
    black_box_id) flags=(--black-box) ;;
    *) log "unknown model tag $tag"; return 2 ;;
  esac
  [[ ! -e "$dest" ]] || { log "refusing existing stage $dest"; return 2; }
  log "$tag command_flags=${flags[*]} destination=$dest stage_steps=$steps"
  bounded "$PY" -m wms.cli train \
    --pack "$BASE_PACK" --identities "$BASE_IDENTITIES" \
    --pack "$DSPEED_PACK" --identities "$DSPEED_IDENTITIES" --overlay "$DSPEED_OVERLAY" \
    --pack "$DAMPING_PACK" --identities "$DAMPING_IDENTITIES" --overlay "$DAMPING_OVERLAY" \
    --outcomes "$TRAIN_OUTCOMES" \
    --window-list "$TRAIN_WINDOWS" --val-window-list "$VAL_WINDOWS" \
    --exclude-window-list "$EXCLUDED_WINDOWS" \
    --out "$dest" --device cuda --threads "$CPU_THREADS" --num-workers "$NUM_WORKERS" --batch-size 8 \
    --sumo-fidelity --start-shift-prob 0.75 --truncation-steps 20 \
    --unroll-s 10 --unroll-s 30 --unroll-s 60 --unroll-s 120 \
    --stage-steps "$steps" "${flags[@]}"
  # The eval adapter reads this sibling: preserve receipt physics, not defaults.
  bounded "$PY" -c '
import json, sys
from pathlib import Path
out=Path(sys.argv[1])
receipt=json.loads((out/"TRAIN_RECEIPT.json").read_text())
physics=receipt["physics"]
assert physics["dt_s"]==0.5
with (out/"physics.json").open("x") as f:
    json.dump(physics, f, indent=2, allow_nan=False)
' "$dest"
}

# Both smoke gates must pass before either full training starts.
for tag in black_box_traj black_box_id; do
  log "$tag smoke allocated steps=$SMOKE_STAGE_STEPS"
  run "$OUT/smoke_$tag" "$tag" "$SMOKE_STAGE_STEPS" > "$OUT/smoke_$tag.log" 2>&1
  bounded "$PY" -m wms.gate_stages "$OUT/smoke_$tag/train_log.jsonl" \
    --stages "$STAGES" --require-terms position,speed,lane > "$OUT/gate_$tag.log" 2>&1
  # gate_stages covers all horizons; require every row's terms and exact counts too.
  bounded "$PY" -c '
import json, math, sys
from collections import Counter
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
assert len(rows)==40 and Counter(r["stage_s"] for r in rows)=={10:10,30:10,60:10,120:10}
for r in rows:
    assert r["finite"] is True and r["gate_monotone"] is True and r["gate_g0"]==0
    assert r["valid_targets"]>0 and r["unroll_steps"]==int(2*r["stage_s"])
    assert all(math.isfinite(float(r[k])) for k in ("loss","position","speed","lane","seconds","gate_g_last"))
    curve=r["gate_curve"]
    assert curve and curve[0]==0 and all(math.isfinite(float(v)) for v in curve)
    assert all(b>=a for a,b in zip(curve,curve[1:]))
print("multi-stage finite/count/target gate passed")
' "$OUT/smoke_$tag/train_log.jsonl" >> "$OUT/gate_$tag.log" 2>&1
  log "$tag smoke GATE PASSED"
done
for tag in black_box_traj black_box_id; do
  log "$tag full allocated steps=$FULL_STAGE_STEPS"
  run "$OUT/full_$tag" "$tag" "$FULL_STAGE_STEPS" > "$OUT/full_$tag.log" 2>&1
  log "$tag full completed"
done
log "LF chain completed"
