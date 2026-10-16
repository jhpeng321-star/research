#!/usr/bin/env bash
# V6 long paired TRAIN collection; based on collect_v5.sh's resumable boxops phases.
# TODO: supply a registered, exposure-audited TRAIN seed block via SEEDS,
# split disjointly between boxes. No new block is allocated or defaulted here.
#
# MEASURED real smoke (2026-10-15, SUMO 1.27.1, CPU-only, affinity <=2 CPUs,
# one worker, one math-library thread): seed4107/m2700/p0.2/textbook;
# warmup60/base240/cadence120/branch120, 2 random + native, repeat_check=0.
# 1 base + 3 branches = 660 simulated seconds; 0 collisions / teleports.
# Collection 26.795 s wall (base 14.239 s, branch phase 10.483 s); conversion
# with wms.data.c4_episodes 8.847 s. Raw 2,313,561 bytes / 4 NPZ episodes;
# converted 14,182,350 bytes / 4 episodes / 56 NPY arrays. Each branch has
# one full 120 s window, identical 60 s prehistory, and the same pair key.
# Evidence/source: _out/v6longpair/smoke_real/SMOKE_RECEIPT.json and
# _out/v6longpair/smoke_real/paired/branch/c4_episodes (relative to this repo).
#
# Sizing PROJECTION from that smoke, not a two-box throughput measurement:
# defaults cost 300+900 + 6*(4+1)*120 + 120 repeat = 4,920 sim-s/base.
# Conservatively scale all sim time by 26.795/660: ~200 worker-wall-s/base
# (~0.0555 worker-h; not measured CPU utilization). For TWO 48-core boxes
# / ~2 h, start with 512 bases/box, WORKERS=46 explicitly, 1,024 total.
# 12 worker waves/box at 2x the measured cost take ~80 min; reserve 30 min
# for conversion/setup/I/O (~110 min total). Later congested states, other
# cases and box CPUs may be slower: check the first batch before scaling.
# Total: 6,144 pair groups, 30,720 branches, 5,038,080 sim-s. Reserve ~80 GB
# raw+converted/box (smoke bytes scaled by ~7.86x recorded state count/base).
# The driver's printed estimate still uses its historical calibration.
# BRANCH_ARMS counts random arms; a native reference is always additional.
#
# Usage: SEEDS=<registered-subset> WORKERS=46 bash scripts/boxops/collect_v6_longpair.sh RUN_ID
# Local smoke: SEEDS=4107 WORKERS=1 PENETRATIONS=0.2 PEN_WEIGHTS=1 DEMANDS=2700 \
#   WARMUP=60 HORIZON=240 BRANCH_CADENCE=120 BRANCH_HORIZON=120 BRANCH_ARMS=2 \
#   BRANCH_REPEAT_CHECK=0 bash scripts/boxops/collect_v6_longpair.sh smoke
# Output source: $REPO/_out/v6longpair/RUN_ID/paired/branch/c4_episodes
# (actual write-once destination is recorded in paired.CONVERTED). No upload.
set -euo pipefail
usage() {
    echo 'Usage: SEEDS=<registered-training-subset> collect_v6_longpair.sh RUN_ID | --help'
    echo 'One generic_random base per seed, long random/native branches, then wms.data.c4_episodes.'
    echo 'Defaults: textbook, warmup300/horizon900, cadence120/branch120, 4 random + native; WORKERS=1.'
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
[[ $# == 1 ]] || { usage >&2; exit 2; }
: "${SEEDS:?SEEDS is mandatory; supply a registered TRAIN block}"
POPULATION=${POPULATION:-textbook}
PENETRATIONS=${PENETRATIONS:-0.05,0.1,0.2,0.4}
PEN_WEIGHTS=${PEN_WEIGHTS:-1,1,2,2}
DEMANDS=${DEMANDS:-1900,2100,2400,2700}
WORKERS=${WORKERS:-1}
WARMUP=${WARMUP:-300}
HORIZON=${HORIZON:-900}
BRANCH_CADENCE=${BRANCH_CADENCE:-120}
BRANCH_ARMS=${BRANCH_ARMS:-4}
BRANCH_HORIZON=${BRANCH_HORIZON:-120}
BRANCH_REPEAT_CHECK=${BRANCH_REPEAT_CHECK:-1}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${REPO:-$(cd "$HERE/../.." && pwd)}
source "$HERE/common.sh"
RUN_ID=$1; valid_id "$RUN_ID"; safe_path "$RUN_ID"
for name in WORKERS HORIZON BRANCH_CADENCE BRANCH_ARMS BRANCH_HORIZON; do
    [[ ${!name} =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
done
for name in WARMUP BRANCH_REPEAT_CHECK; do
    [[ ${!name} =~ ^(0|[1-9][0-9]*)$ ]] || die "$name must be a nonnegative integer"
done
(( BRANCH_HORIZON >= 120 )) || die 'longpair requires BRANCH_HORIZON >= 120'
(( WARMUP + HORIZON <= 2400 )) || die 'warmup + horizon exceeds the 2400 s fixture'
(( HORIZON >= BRANCH_CADENCE + BRANCH_HORIZON && WARMUP + BRANCH_CADENCE >= 60 )) ||
    die 'timing must allow at least one full branch with 60 s prehistory'
export CUDA_VISIBLE_DEVICES=''
activate_box
"$PYTHON" "$HERE/ops.py" require-population "$REPO"
OUT=${RUN_ROOT:-$REPO/_out/v6longpair}/$RUN_ID
begin_run "$OUT" collect "population=$POPULATION" "seeds=$SEEDS" \
    "pens=$PENETRATIONS" "weights=$PEN_WEIGHTS" "demands=$DEMANDS" "workers=$WORKERS" \
    "warmup=$WARMUP" "horizon=$HORIZON" "cadence=$BRANCH_CADENCE" \
    "arms=$BRANCH_ARMS" "branch_h=$BRANCH_HORIZON" "repeat_check=$BRANCH_REPEAT_CHECK"
ARGS=(--out "$OUT/paired" --plain-seeds '' --branch-seeds "$SEEDS" --workers "$WORKERS"
      --population "$POPULATION" --penetrations "$PENETRATIONS" --pen-weights "$PEN_WEIGHTS"
      --demands "$DEMANDS" --warmup "$WARMUP" --horizon "$HORIZON" --gr-mode speed
      --branch-cadence "$BRANCH_CADENCE" --branch-arms "$BRANCH_ARMS"
      --branch-horizon "$BRANCH_HORIZON" --branch-repeat-check "$BRANCH_REPEAT_CHECK"
      --no-physics-log --no-convert)
phase estimate "$PYTHON" -m traffic_shepherd.c4.genrand_driver "${ARGS[@]}" --estimate-only
phase paired "$PYTHON" -m traffic_shepherd.c4.genrand_driver "${ARGS[@]}"
# This wrapper invokes wms.data.c4_episodes on complete base/branch episode roots,
# excluding preserved incomplete ancestors on resume; paired metadata is retained.
convert_phase paired "$OUT/paired/branch"
"$PYTHON" "$HERE/ops.py" summary "$OUT"
printf 'Converted source receipt: %s\n' "$OUT/paired.CONVERTED"
