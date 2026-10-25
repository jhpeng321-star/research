#!/usr/bin/env bash
# S-RAMPSPEED (R16) operator script.  Root-agnostic: run root R16_ROOT (default /root/autodl-tmp/c4/v5,
# the same layout on gpuB and gpuR); run it from the prepared copy <root>/repo_r16_<sha8>.
# Exactly one mode:
#   (none)       CPU config validation only
#   --smoke      preflight (configs = R14 + R16 change, data manifests, init sha = 13E b2 FINAL,
#                offline evidence hashes + re-analysis) + R16/R14/13E core tests on CPU, then
#                smoke A (2 x 120 s from 13E b2) and smoke B (2 x 120 + 2 x 300 s from smoke A), each
#                with the strict 13E-machinery verdict + ramp-tensor checks; writes smoke_r16.complete.json
#   --launch-a   production A (30 x 120 s from 13E b2; needs the smoke receipt), then its strict verdict
#   --launch-b   production B (40 x 120 + 60 x 300 s) from A FINAL (needs the A production verdict)
#   --chain      --launch-a, then --launch-b
# Shared GPU (a policy loop also runs on gpuB): no exclusive GPU guard; this process caps its own CUDA
# memory with WMS_GPU_MEM_FRACTION (default 0.40 of 48 GB = ~19.7 GB; 13E/R14 peaked at ~2 GB).
# The R14 stop-rule waiters (paired d11 checks on gpuR-only state files) are not part of R16.
# Never signals or deletes anything; refuses to overwrite any prior artifact.
set -eu -o pipefail -o noclobber
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PYTHON=${R16_PYTHON:-/root/miniconda3/bin/python}
[ -x "$PYTHON" ] || { echo "set R16_PYTHON to the deployment Python" >&2; exit 2; }
MODE=config
case "${1:-}" in
    "") ;;
    --smoke|--launch-a|--launch-b|--chain) MODE=${1#--} ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
esac
L=${R16_ROOT:-/root/autodl-tmp/c4/v5}
export R16_ROOT=$L
OFFLINE=${R16_OFFLINE:-$L/diag_r16/R16_OFFLINE_REPORT.json}
TAG=r16; RUN=c4_wm_v10_round16
A=configs/c4_wm_v10round16.toml; B=configs/c4_wm_v10round16_stage_b.toml
CK=$L/runs/c4_wm_v10_round13e_b2/wm_s.pt
export PYTHONPATH="$REPO/src:$REPO/tests:$REPO" PYTHONDONTWRITEBYTECODE=1
GATE=("$PYTHON" -B -m wms.train.round16_gate)

disk_guard() {
    local NEED_MB=$1 FREE_KB
    FREE_KB=$(df -Pk "$L" | awk 'NR==2 {print $4}')
    [ "${FREE_KB:-0}" -ge $(( NEED_MB * 1024 )) ] || {
        echo "disk guard: < ${NEED_MB} MB free on $L; refusing $MODE" >&2; exit 6; }
}
fresh() {
    for artifact in "$@"; do
        [ ! -e "$artifact" ] && [ ! -L "$artifact" ] || { echo "prior artifact: $artifact" >&2; exit 4; }
    done
}

CUDA_VISIBLE_DEVICES= "${GATE[@]}" config
if [ "$MODE" = config ]; then exit 0; fi
if [ "$MODE" = chain ]; then
    bash "$REPO/scripts/round16_gpuB.sh" --launch-a || exit $?
    exec bash "$REPO/scripts/round16_gpuB.sh" --launch-b
fi
ulimit -n 65536
# gpuB container quota is 20 CPUs shared with a policy loop: ~4 cores here (runtime.threads = 2 in the recipe).
export OMP_NUM_THREADS=4 WMS_JAM_WORKERS=4 WMS_LATEST_EVERY=20
export WMS_GPU_MEM_FRACTION=${WMS_GPU_MEM_FRACTION:-0.40}
export WMS_DATA_V5_NATIVE="$L/data/v5_native" WMS_DATA_GENRAND_PLAIN="$L/data/genrand_plain"
export WMS_DATA_GENRAND_BRANCH="$L/data/genrand_branch" WMS_DATA_PAIRS300="$L/data/pairs300"
export WMS_DATA_LOWPEN_NATIVE="$L/data/lowpen_20261025/lowpen_native"
export WMS_DATA_LOWPEN_GENRAND_PLAIN="$L/data/lowpen_20261025/lowpen_genrand_plain"
export WMS_DATA_LOWPEN_GENRAND_BRANCH="$L/data/lowpen_20261025/lowpen_genrand_branch"
COMMON=(--set loss.loss_share_logging=false --set loss.brake_follow_cav_weight=2.0
    --set optim.batch_size=4 --set runtime.device=cuda)

train_verdict() {   # STAGE CONFIG INIT DIR LOG REPORT PRODUCTION(0/1) [overrides...]
    local STAGE=$1 CONFIG=$2 INIT=$3 DIR=$4 LOG=$5 REPORT=$6 PROD=$7 CODE=0
    shift 7
    fresh "$DIR" "$LOG" "$REPORT"
    "$PYTHON" -B -u -m wms train --config "$CONFIG" --mode run --out "$DIR" \
        "${COMMON[@]}" --set "model.init_from=$INIT" "$@" > "$LOG" 2>&1 || CODE=$?
    local FLAG=()
    [ "$PROD" = 1 ] && FLAG=(--production)
    CUDA_VISIBLE_DEVICES= "${GATE[@]}" verdict --stage "$STAGE" --run-dir "$DIR" --log "$LOG" \
        --train-exit-code "$CODE" --init-checkpoint "$INIT" --report "$REPORT" ${FLAG[@]+"${FLAG[@]}"}
}

if [ "$MODE" = smoke ]; then
    fresh "$L/smoke_${TAG}.complete.json" "$L/runs/${TAG}_gate_core_tests"
    disk_guard 2000
    CUDA_VISIBLE_DEVICES= "${GATE[@]}" preflight --root "$L" --offline-report "$OFFLINE"
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        nice -n 10 "$PYTHON" -B -m pytest -q -p no:cacheprovider --noconftest \
        tests/test_r16_ramp_speed_adapt.py tests/test_r16_gate.py \
        tests/test_r13e_emergency.py tests/test_r13e_secure_gap.py tests/test_r13y_yield.py \
        tests/test_r14_lane_secure_gap.py tests/test_r14_lowshare.py tests/test_loop14_emergency_onset.py \
        --basetemp "$L/runs/${TAG}_gate_core_tests"
    train_verdict a "$A" "$CK" "$L/runs/smoke_${TAG}_a" "$L/smoke_${TAG}_a.log" "$L/smoke_${TAG}_a.gate.json" 0 \
        --set 'curriculum.stage_steps=[2]' --set selection.val_max_windows=4 --set curriculum.max_hours=1.5
    train_verdict b "$B" "$L/runs/smoke_${TAG}_a/wm_s.pt" "$L/runs/smoke_${TAG}_b" "$L/smoke_${TAG}_b.log" \
        "$L/smoke_${TAG}_b.gate.json" 0 \
        --set 'curriculum.stage_steps=[2,2]' --set 'optim.batch_size_per_stage=[4,4]' \
        --set selection.val_max_windows=4 --set curriculum.max_hours=1.5
    "$PYTHON" -B - "$L" "$OFFLINE" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, offline = Path(sys.argv[1]), Path(sys.argv[2])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
doc = {"a_sha256": sha(root / "runs/smoke_r16_a/wm_s.pt"), "b_sha256": sha(root / "runs/smoke_r16_b/wm_s.pt"),
       "a_gate": sha(root / "smoke_r16_a.gate.json"), "b_gate": sha(root / "smoke_r16_b.gate.json"),
       "offline_sha256": sha(offline)}
for s in "ab":
    assert json.loads((root / f"smoke_r16_{s}.gate.json").read_text())["ok"] is True
with open(root / "smoke_r16.complete.json", "x") as fh:
    json.dump(doc, fh, indent=1)
print(json.dumps(doc))
PY
    echo "Both R16 smoke gates passed. Production: scripts/round16_gpuB.sh --chain (or --launch-a)."
    exit 0
fi

check_smoke() {
    "$PYTHON" -B - "$L" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
doc = json.loads((root / "smoke_r16.complete.json").read_text())
assert doc["a_sha256"] == sha(root / "runs/smoke_r16_a/wm_s.pt") and doc["b_sha256"] == sha(root / "runs/smoke_r16_b/wm_s.pt")
for s in "ab":
    assert doc[f"{s}_gate"] == sha(root / f"smoke_r16_{s}.gate.json")
    assert json.loads((root / f"smoke_r16_{s}.gate.json").read_text())["ok"] is True
PY
}
STAGE=${MODE#launch-}
check_smoke || { echo "smoke receipt missing/invalid: run --smoke first" >&2; exit 3; }
CUDA_VISIBLE_DEVICES= "${GATE[@]}" preflight --root "$L" --offline-report "$OFFLINE" > /dev/null
disk_guard 1500
OUT="$L/runs/${RUN}_$STAGE"
if [ "$STAGE" = a ]; then
    INIT=$CK; CONFIG=$A
else
    INIT=$L/runs/${RUN}_a/wm_s.pt; CONFIG=$B
    "$PYTHON" -B -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["ok"] is True else 3)' \
        "$L/production_${TAG}_a.gate.json" || { echo "A production verdict missing/failed" >&2; exit 3; }
fi
printf '%s\n' "$$" > "$L/production_${TAG}_$STAGE.pid"
date -Is > "$L/production_${TAG}_$STAGE.start"
train_verdict "$STAGE" "$CONFIG" "$INIT" "$OUT" "$L/production_${TAG}_$STAGE.log" \
    "$L/production_${TAG}_$STAGE.gate.json" 1
