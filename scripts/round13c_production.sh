#!/usr/bin/env bash
# Separate one-shot production phase, reusing a completed smoke of identical src/config trees.
# This supports an operational-only waiter repair without repeating smoke training.
set -eu -o pipefail -o noclobber
L=${R13C_ROOT:-/root/autodl-tmp/c4/v5}
REPO=${1:?repo argument required}
PYTHON=${R13C_PYTHON:-/root/miniconda3/bin/python}
cd "$REPO"
ulimit -n 65536
test -f "$L/registration_r13c_operational.json"
test -f "$L/smoke_r13c.gate.json"
test ! -e "$L/runs/c4_wm_v10_round13c"
"$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["ok"] and len(d["checkpoints"])==3; assert all(c["temporal_emission"]==0 and c["core_bitwise_frozen"] for c in d["checkpoints"])' "$L/smoke_r13c.gate.json"
"$PYTHON" -c 'import hashlib,json,pathlib,sys; d=json.load(open(sys.argv[1])); assert hashlib.sha256(pathlib.Path("configs/c4_wm_v10round13c.toml").read_bytes()).hexdigest()==d["config_sha256"]; assert hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest()==d["init_sha256"]' "$L/registration_r13c_operational.json" "$L/runs/c4_wm_v10_round12c/wm_s.pt"
cat "$L/smoke_r13c.verdict" >> "$L/smoke_r13c.log"
mkdir "$L/.round13c_production.lock"
export PYTHONPATH=src:. OMP_NUM_THREADS=12 WMS_JAM_WORKERS=12 WMS_LATEST_EVERY=20
export WMS_DATA_V5_NATIVE="$L/data/v5_native" WMS_DATA_GENRAND_PLAIN="$L/data/genrand_plain"
export WMS_DATA_GENRAND_BRANCH="$L/data/genrand_branch" WMS_DATA_PAIRS300="$L/data/pairs300"
START_TICKS=$("$PYTHON" -c 'import pathlib,sys; print(pathlib.Path("/proc/"+sys.argv[1]+"/stat").read_text().rsplit(")",1)[1].split()[19])' "$$")
mkdir -p "$L/chk_r13c"
for MODE in midrun postrun; do
    nohup "$PYTHON" -u -B "$REPO/scripts/round13c_waiter.py" --repo "$REPO" --root "$L" \
        --pid "$$" --start-ticks "$START_TICKS" --mode "$MODE" \
        > "$L/chk_r13c/${MODE}_waiter.log" 2>&1 < /dev/null &
    WAITER=$!
    printf '%s\n' "$WAITER" > "$L/chk_r13c/${MODE}_waiter.pid"
    kill -0 "$WAITER"
done
printf '%s\n' "$$" > "$L/production_r13c.pid"
date -Is > "$L/production_r13c.start"
exec "$PYTHON" -u -m wms train --config configs/c4_wm_v10round13c.toml --mode run \
    --out "$L/runs/c4_wm_v10_round13c" --set "model.init_from=$L/runs/c4_wm_v10_round12c/wm_s.pt" \
    --set loss.loss_share_logging=false --set loss.brake_follow_cav_weight=2.0 \
    --set optim.batch_size=4 --set runtime.device=cuda
