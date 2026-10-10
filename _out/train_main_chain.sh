#!/usr/bin/env bash
# TRAIN-only main chain. Default: CPU-only preflight; GPU requires explicit run.
# Usage (on the box, after code and original read-only packs are available):
#   BOX=gpuA ORIGINAL_ROOT=/root/autodl-tmp SW_ROOT=/path/to/switching \
#   WINDOWS_TRAIN=/path/to/train_noPIwall.json WINDOWS_VAL=/path/to/monitor.json \
#   EXCLUDE=/path/to/pi_wall_excluded.json bash train_main_chain.sh preflight
#   SW_STAGE_SRC=/local/train-export SW_ROOT=/new/destination ... bash ... stage
#   W=/new/run/directory ... bash train_main_chain.sh run
# stage uses LOCAL rsync only, a six-path allowlist, and a NEW SW_ROOT.
# Coordinator SSH rsync examples are in train_main_data.json; never invoked here.
# Original packs remain read-only with mixed frozen metadata and TRAIN outcomes.
set -Eeuo pipefail
die() { printf 'REFUSED: %s\n' "$*" >&2; exit 2; }
[[ $# -le 1 ]] || die 'one mode only: preflight | stage | run'
MODE=${1:-preflight}
case "$MODE" in preflight|stage|run) ;; *) die "unknown mode $MODE" ;; esac
# This check precedes all Python/data access (including when invoked via bash).
if LC_ALL=C grep -q $'\r' "${BASH_SOURCE[0]}"; then die 'chain has CRLF/CR bytes; require LF'; fi
BOX=${BOX:-gpuA}
case "$BOX" in
  gpuA) PY=${PY:-/root/autodl-tmp/gpu1/env/rd_gpu/bin/python} ;;
  gpuB) PY=${PY:-/root/autodl-tmp/gpu1/env/rd_gpu/bin/python} ;;
  *) die 'BOX must be gpuA or gpuB' ;;
esac
# Both boxes use the proven gpu1 env layout unless PY is explicitly overridden.
# PY is one interpreter executable, never a "conda run ..." command string.
ORIGINAL_ROOT=${ORIGINAL_ROOT:-/root/autodl-tmp}
: "${SW_ROOT:?set SW_ROOT to the TRAIN switching export or new staging destination}"
: "${WINDOWS_TRAIN:?set WINDOWS_TRAIN to externally built original leave-PI-and-wall-out TRAIN windows}"
: "${WINDOWS_VAL:?set WINDOWS_VAL to externally built original TRAIN-scoped monitoring windows}"
: "${EXCLUDE:?set EXCLUDE to externally built original PI-and-wall exclusion list}"
CODE_ROOT=${CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
WM_CODE=${WM_CODE:-$CODE_ROOT/output/wd01/wm_n1}
W=${W:-$ORIGINAL_ROOT/wms_train_main/work}
PACK_V3=${PACK_V3:-$ORIGINAL_ROOT/wmn1_m1_45c9032_20261003_A1/work/pack}
PACK_DSPEED=${PACK_DSPEED:-$ORIGINAL_ROOT/wmn1_a9_data/pack_dspeed}
PACK_DAMPING=${PACK_DAMPING:-$ORIGINAL_ROOT/wmn1_a10_data/pack_damping}
SIDECAR_ROOT=${SIDECAR_ROOT:-$ORIGINAL_ROOT/wmn1_a11_20261004T172721Z/work/sidecars}
IDENTITIES_V3=${IDENTITIES_V3:-$SIDECAR_ROOT/identities}
IDENTITIES_DSPEED=${IDENTITIES_DSPEED:-$SIDECAR_ROOT/identities_dspeed}
IDENTITIES_DAMPING=${IDENTITIES_DAMPING:-$ORIGINAL_ROOT/wmn1_a10_data/identities_damping}
OVERLAY_DSPEED=${OVERLAY_DSPEED:-$ORIGINAL_ROOT/wmn1_a9_data/labels_overlay_dspeed}
OVERLAY_DAMPING=${OVERLAY_DAMPING:-$ORIGINAL_ROOT/wmn1_a10_data/labels_overlay_damping}
TRAIN_OUTCOMES=${TRAIN_OUTCOMES:-$SIDECAR_ROOT/labels_cov_20261004T072826Z}
MERGE_PARAMS=${MERGE_PARAMS:-$CODE_ROOT/_out/merge_params.json}
SW_PACK=${SW_PACK:-$SW_ROOT/pack_switching_train}
SW_IDENTITIES=${SW_IDENTITIES:-$SW_ROOT/identities_switching_train}
SW_DRIVER=${SW_DRIVER:-$SW_ROOT/driver_params_switching_train}
SW_WINDOWS_TRAIN=${SW_WINDOWS_TRAIN:-$SW_ROOT/windows_train_hetdata.json}
SW_WINDOWS_VAL=${SW_WINDOWS_VAL:-} # optional, world-disjoint TRAIN-scoped monitor
SW_TIMELINE_INDEX=${SW_TIMELINE_INDEX:-$SW_ROOT/ACTION_TIMELINE_INDEX_train.json}
ZERO_SHOT_HELDOUT=${ZERO_SHOT_HELDOUT:-$SW_ROOT/ZERO_SHOT_HELDOUT.json} # metadata only
OMP=${OMP:-2}
SWITCH_MIX_RATIO=${SWITCH_MIX_RATIO:-0.5}
EVENT_OVERSAMPLE=${EVENT_OVERSAMPLE:-2}
CAV_DRIVE=${CAV_DRIVE:-teacher}
NUM_WORKERS=${NUM_WORKERS:-0} # lazy closures are not pickleable; mmap, <6 GB/Python
MAX_HOURS=${MAX_HOURS:-10}
MODEL_KIND=${MODEL_KIND:-structured} # structured | blackbox_traj | blackbox_id
ROLLOUT_MODE=${ROLLOUT_MODE:-residual} # residual | hybrid
LANE_HEAD=${LANE_HEAD:-off} # off | history (requires hybrid)
# Braking defaults are OFF: multipliers of 1 and propagation weight of 0.
ASYM_SPEED_LOSS=${ASYM_SPEED_LOSS:-1}
DECEL_WEIGHT=${DECEL_WEIGHT:-1}
DECEL_THRESHOLD=${DECEL_THRESHOLD:-3}
PROPAGATION_LOSS=${PROPAGATION_LOSS:-0}
PROP_WINDOW=${PROP_WINDOW:-3}
PROP_MAX_GAP_M=${PROP_MAX_GAP_M:-}
RESIDUAL_CAP_NEG_MPS2=${RESIDUAL_CAP_NEG_MPS2:-}
FLAGS=${FLAGS:-} # extras, whitespace-separated scalar arguments; no eval
BRAKING_FLAGS=${BRAKING_FLAGS:-} # opt-in only, empty by default
readonly STAGES_CSV=10,30,60,120 STAGE_STEPS=2500,2500,1875,1250
readonly SMOKE_STAGE_STEPS=6,6,6,6
[[ "$NUM_WORKERS" == 0 ]] || die 'NUM_WORKERS must be 0 until lazy closure pickling is supported'
[[ "$OMP" =~ ^[1-9][0-9]*$ ]] || die 'OMP must be a positive integer'
MODEL=()
case "$MODEL_KIND" in
  structured) ;;
  blackbox_traj) MODEL+=(--blackbox-traj) ;;
  blackbox_id) MODEL+=(--black-box) ;;
  *) die 'MODEL_KIND must be structured, blackbox_traj or blackbox_id' ;;
esac
case "$ROLLOUT_MODE" in residual|hybrid) ;; *) die 'ROLLOUT_MODE must be residual or hybrid' ;; esac
case "$LANE_HEAD" in off|history) ;; *) die 'LANE_HEAD must be off or history' ;; esac
[[ "$LANE_HEAD" != history || "$ROLLOUT_MODE" == hybrid ]] || die 'LANE_HEAD=history requires ROLLOUT_MODE=hybrid'
# The trajectory factory ignores mode/head; refuse rather than mislabel a run.
[[ "$MODEL_KIND" != blackbox_traj || ( "$ROLLOUT_MODE" == residual && "$LANE_HEAD" == off ) ]] || die 'blackbox_traj requires residual rollout and lane head off'
MODEL+=(--rollout-mode "$ROLLOUT_MODE" --lane-head "$LANE_HEAD")
EXTRA=(); BRAKING=()
read -r -a EXTRA <<< "$FLAGS"
read -r -a BRAKING <<< "$BRAKING_FLAGS"
# Extras cannot replace the training contract or introduce non-TRAIN input paths.
# Model/mode/head are env-only; remaining options must be exact CLI help tokens.
for arg in "${EXTRA[@]}" "${BRAKING[@]}"; do
  case "$arg" in
    --pack*|--identities*|--overlay*|--out*|--window*|--val-*|--exclude*|--leave-*|--switch-*|--zero-*|--heldout-*|--cav-*|--stage-*|--unroll-*|--steps*|--device*|--num-workers*|--gpu-share*|--traffic-*|--stability-*|--conservation-*|--loss-share*|--black*|--geometry-only|--driver-sidecar*|--state-noise*|--threads*|--batch-size*|--truncation-steps*|--start-shift-prob*|--rollout*|--lane-head*)
      die "extra cannot override fixed contract: $arg" ;;
  esac
done
BRAKING_ENV=(--asym-speed-loss "$ASYM_SPEED_LOSS" --decel-weight "$DECEL_WEIGHT"
             --decel-threshold "$DECEL_THRESHOLD" --propagation-loss "$PROPAGATION_LOSS"
             --prop-window "$PROP_WINDOW")
if [[ -n "$PROP_MAX_GAP_M" ]]; then BRAKING_ENV+=(--prop-max-gap-m "$PROP_MAX_GAP_M"); fi
if [[ -n "$RESIDUAL_CAP_NEG_MPS2" ]]; then BRAKING_ENV+=(--residual-cap-neg-mps2 "$RESIDUAL_CAP_NEG_MPS2"); fi
command -v "$PY" >/dev/null || die "missing interpreter $PY"
# Resolve the effective braking contract with CLI last-value-wins semantics:
# env defaults, then FLAGS, then BRAKING_FLAGS. Stdlib only, no pack access.
TERMS=$("$PY" - "$MODEL_KIND" "${BRAKING_ENV[@]}" "${EXTRA[@]}" "${BRAKING[@]}" <<'PY'
import argparse, math, sys
model, *argv = sys.argv[1:]
parser = argparse.ArgumentParser(allow_abbrev=False)
for option, default in (("asym-speed-loss", 1), ("decel-weight", 1),
                        ("decel-threshold", 3), ("propagation-loss", 0),
                        ("prop-window", 3), ("prop-max-gap-m", None),
                        ("residual-cap-neg-mps2", None)):
    parser.add_argument('--' + option, type=float, default=default)
args, _ = parser.parse_known_args(argv)
for key, value in vars(args).items():
    if value is None: continue
    minimum = 1 if key in ('asym_speed_loss', 'decel_weight') else 0
    if not math.isfinite(value) or value < minimum or (
            minimum == 0 and key != 'propagation_loss' and value == 0):
        raise SystemExit(f'invalid braking value {key}: {value}')
terms = ['position', 'speed', 'lane', 'traffic', 'stability', 'conservation']
enabled = [('asym_speed', args.asym_speed_loss > 1),
           ('decel_weight', args.decel_weight > 1),
           ('propagation', args.propagation_loss > 0)]
if model != 'structured' and (any(on for _, on in enabled) or args.residual_cap_neg_mps2 is not None):
    raise SystemExit('enabled braking requires MODEL_KIND=structured')
terms.extend(term for term, on in enabled if on)
print(','.join(terms))
PY
) || die 'invalid braking contract'
readonly TERMS
# Black-box defaults stay inert; explicit extras still pass the same help guard.
if [[ "$MODEL_KIND" != structured ]]; then BRAKING_ENV=(); fi

# Host checks use stdlib only. No torch import, GPU probe, or hidden data scan.
host_check() {
  "$PY" - "$@" <<'PY'
import json, math, os, pathlib, shutil, sys, time
mode, *args = sys.argv[1:]
if mode == 'disk':
    path, needed = args
    p = pathlib.Path(path)
    while not p.exists():
        if p == p.parent: raise SystemExit('no existing disk-check ancestor')
        p = p.parent
    free = shutil.disk_usage(p).free
    print(json.dumps({'path': path, 'free_bytes': free, 'required_bytes': int(needed)}))
    if free < int(needed): raise SystemExit('insufficient disk: require >=10 GiB remaining')
elif mode == 'idle':
    def cpu():
        v = list(map(int, pathlib.Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
        return sum(v), v[3] + v[4]
    a, b = cpu(); time.sleep(1); c, d = cpu()
    busy = (1 - (d-b) / max(1, c-a)) * (os.cpu_count() or 1)
    load = os.getloadavg()[0]
    print(json.dumps({'busy_cores': busy, 'load1': load, 'idle_limit_cores': 1.0}))
    if busy > 1.0 or load > 1.0: raise SystemExit('CPU contention before timing; no retry')
elif mode == 'hours':
    hours = float(args[0])
    if not math.isfinite(hours) or hours <= 0: raise SystemExit('MAX_HOURS must be finite and positive')
PY
}
readonly MIN_FREE_BYTES=10737418240
host_check hours "$MAX_HOURS"

# Stage allowlist: never copy a source root, val/test packs, raw runs or sealed data.
# Reject symlinks and restricted descendants before walking them or invoking rsync.
stage() {
  : "${SW_STAGE_SRC:?stage requires an explicit LOCAL SW_STAGE_SRC}"
  [[ ! -e "$SW_ROOT" && ! -L "$SW_ROOT" ]] || die 'SW_ROOT already exists; no overwrite or resume'
  [[ "$SW_STAGE_SRC" != *:* && "$SW_ROOT" != *:* ]] || die 'stage accepts local paths only; use coordinator SSH rsync separately'
  [[ -d "$(dirname -- "$SW_ROOT")" ]] || die 'staging destination parent must exist'
  command -v rsync >/dev/null || die 'local rsync is required for stage'
  local -a paths=(pack_switching_train identities_switching_train driver_params_switching_train
                 ACTION_TIMELINE_INDEX_train.json windows_train_hetdata.json ZERO_SHOT_HELDOUT.json)
  local bytes source_sizes
  source_sizes=$("$PY" - "$SW_STAGE_SRC" "${paths[@]}" <<'PY'
import json, os, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
deny = re.compile(r'(?:^|[^a-z0-9])(cal(?:ibration)?|holdout|sealed|chunk[-_]5|j4b|inc1|val|test)(?:$|[^a-z0-9])', re.I)
if any(deny.search(part) for part in root.parts): raise SystemExit('restricted staging source root')
def size(p, heldout_metadata=False):
    if p.is_symlink(): raise SystemExit(f'stage rejects symlink: {p}')
    if not heldout_metadata and deny.search(p.name): raise SystemExit(f'restricted stage path: {p}')
    if p.is_file(): return p.stat().st_size
    if not p.is_dir(): raise SystemExit(f'missing/unsupported stage source: {p}')
    with os.scandir(p) as entries:
        return sum(size(pathlib.Path(e.path)) for e in entries)
if root.is_symlink() or not root.is_dir(): raise SystemExit('source must be a real directory')
total = 0
for name in sys.argv[2:]:
    n = size(root / name, name == 'ZERO_SHOT_HELDOUT.json')
    print(json.dumps({'source': str(root / name), 'measured_bytes': n}))
    total += n
print(total)
PY
  ) || die 'source allowlist/size check failed'
  bytes=${source_sizes##*$'\n'}
  [[ "$bytes" =~ ^[0-9]+$ ]] || die 'invalid measured source byte count'
  # Estimate ~7.7 GiB (one TRAIN pack + timelines, excluding val/test).
  # Exact logical bytes, not the estimate, drive the pre-copy capacity refusal.
  host_check disk "$SW_ROOT" "$((bytes + MIN_FREE_BYTES))"
  mkdir -- "$SW_ROOT" # atomic reservation; failure leaves existing data untouched
  trap 'rc=$?; printf "stage failed rc=%s; partial destination retained\n" "$rc" >> "$SW_ROOT/stage.log"' ERR
  printf 'source=%s\nexact_source_bytes=%s\nestimate_train_only_gib=7.7\n' "$SW_STAGE_SRC" "$bytes" > "$SW_ROOT/stage.log"
  printf '%s\n' "$source_sizes" >> "$SW_ROOT/stage.log"
  local p
  for p in "${paths[@]}"; do
    # No --delete, update, resume, links, or whole-tree transfer.
    rsync -rt --itemize-changes -- "$SW_STAGE_SRC/$p" "$SW_ROOT/" >> "$SW_ROOT/stage.log" 2>&1
  done
  if [[ -n "$SW_WINDOWS_VAL" ]]; then
    printf 'optional SW_WINDOWS_VAL must be supplied separately as TRAIN-scoped metadata: %s\n' "$SW_WINDOWS_VAL" >> "$SW_ROOT/stage.log"
  fi
  host_check disk "$SW_ROOT" "$MIN_FREE_BYTES" >> "$SW_ROOT/stage.log"
  printf 'staging complete; originals were not copied; run preflight next\n' >> "$SW_ROOT/stage.log"
  trap - ERR
}
if [[ "$MODE" == stage ]]; then stage; exit 0; fi

for d in "$WM_CODE/wms" "$PACK_V3" "$PACK_DSPEED" "$PACK_DAMPING" \
         "$IDENTITIES_V3" "$IDENTITIES_DSPEED" "$IDENTITIES_DAMPING" \
         "$OVERLAY_DSPEED" "$OVERLAY_DAMPING" "$TRAIN_OUTCOMES" \
         "$SW_PACK" "$SW_IDENTITIES" "$SW_DRIVER"; do
  [[ -d "$d" ]] || die "missing required directory $d"
done
for f in "$WINDOWS_TRAIN" "$WINDOWS_VAL" "$EXCLUDE" "$MERGE_PARAMS" \
         "$SW_WINDOWS_TRAIN" "$SW_TIMELINE_INDEX" "$ZERO_SHOT_HELDOUT" \
         "$SW_PACK/manifest.json" "$SW_PACK/INDEX.json" \
         "$SW_PACK/controller_timeline.npy" "$SW_PACK/issued_acceleration_mps2.npy" \
         "$SW_PACK/issued_target_speed_mps.npy" "$SW_IDENTITIES/TRAIN_IDENTITY_RECEIPT.json" \
         "$SW_IDENTITIES/identities.npy" "$SW_IDENTITIES/global_positions.npy" \
         "$SW_IDENTITIES/vehicle_map.json" "$SW_DRIVER/driver_params.npy"; do
  [[ -f "$f" ]] || die "missing required file $f"
done
[[ -z "$SW_WINDOWS_VAL" || -f "$SW_WINDOWS_VAL" ]] || die "missing SW_WINDOWS_VAL $SW_WINDOWS_VAL"
# Refuse restricted input locations before opening metadata. Original mixed
# frozen metadata contents are left unchanged; ZERO_SHOT_HELDOUT is metadata.
"$PY" - "$PACK_V3" "$PACK_DSPEED" "$PACK_DAMPING" "$IDENTITIES_V3" \
         "$IDENTITIES_DSPEED" "$IDENTITIES_DAMPING" "$OVERLAY_DSPEED" "$OVERLAY_DAMPING" \
         "$TRAIN_OUTCOMES" "$WINDOWS_TRAIN" "$WINDOWS_VAL" "$EXCLUDE" "$MERGE_PARAMS" \
         "$SW_PACK" "$SW_IDENTITIES" "$SW_DRIVER" "$SW_WINDOWS_TRAIN" "$SW_WINDOWS_VAL" \
         "$SW_TIMELINE_INDEX" "$ZERO_SHOT_HELDOUT" <<'PY'
import pathlib, re, sys
deny = re.compile(r'(?:^|[^a-z0-9])(cal(?:ibration)?|holdout|sealed|chunk[-_]5|j4b|inc1|test)(?:$|[^a-z0-9])', re.I)
for i, name in enumerate(sys.argv[1:]):
    if not name: continue
    p = pathlib.Path(name)
    parts = p.parts[:-1] if i == len(sys.argv[1:])-1 and p.name == 'ZERO_SHOT_HELDOUT.json' else p.parts
    if any(deny.search(part) for part in parts): raise SystemExit(f'restricted input location: {p}')
PY
# No non-TRAIN data files are opened; mixed frozen pack metadata is retained.
"$PY" - "$WINDOWS_TRAIN" "$WINDOWS_VAL" "$SW_WINDOWS_TRAIN" "$SW_TIMELINE_INDEX" \
         "$SW_IDENTITIES/TRAIN_IDENTITY_RECEIPT.json" "$SW_WINDOWS_VAL" <<'PY'
import json, pathlib, sys
for position, name in enumerate(sys.argv[1:]):
    if not name: continue
    p = pathlib.Path(name)
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f'require explicit TRAIN-scoped metadata: {p}')
    # Frozen LPO/LPWO lists declare split per window, not at the top level.
    # Accept their actual builder schema without rewriting receipt-pinned files.
    rows = data.get('windows')
    declared = data.get('split')
    expected_lists = ({'windows_train_noPI', 'windows_train_noPI_noWall'},
                      {'windows_val', 'windows_val_noPI_noWall'})
    legacy_list = (position < 2 and declared is None
                   and data.get('schema') == 'wm_s_lpo_windows/1'
                   and data.get('list') in expected_lists[position])
    if declared != 'TRAIN' and not legacy_list:
        raise SystemExit(f'require explicit TRAIN-scoped metadata: {p}')
    if legacy_list or rows is not None:
        if (not isinstance(rows, list) or not rows
                or data.get('n_windows') != len(rows)
                or any(not isinstance(row, dict) or row.get('split') != 'TRAIN' for row in rows)):
            raise SystemExit(f'require nonempty explicitly TRAIN windows and exact count: {p}')
PY
export WM_N1_TRAIN_OUTCOMES="$TRAIN_OUTCOMES" WM_N1_TRAIN_IDENTITIES="$IDENTITIES_V3"
export OMP_NUM_THREADS="$OMP" MKL_NUM_THREADS="$OMP"
export PYTHONPATH="$WM_CODE${PYTHONPATH:+:$PYTHONPATH}"

COMMON=(--pack "$PACK_V3" --identities "$IDENTITIES_V3" --outcomes "$TRAIN_OUTCOMES"
        --pack "$PACK_DSPEED" --identities "$IDENTITIES_DSPEED" --overlay "$OVERLAY_DSPEED"
        --pack "$PACK_DAMPING" --identities "$IDENTITIES_DAMPING" --overlay "$OVERLAY_DAMPING"
        --window-list "$WINDOWS_TRAIN" --val-window-list "$WINDOWS_VAL"
        --exclude-window-list "$EXCLUDE" --leave-pi-wall-out
        --switch-pack "$SW_PACK" --switch-identities "$SW_IDENTITIES"
        --switch-window-list "$SW_WINDOWS_TRAIN" --switch-timeline-index "$SW_TIMELINE_INDEX"
        --zero-shot-heldout "$ZERO_SHOT_HELDOUT" --switch-mix-ratio "$SWITCH_MIX_RATIO" --event-oversample "$EVENT_OVERSAMPLE"
        --switch-val-frac .1 --heldout-family-policy drop --heldout-max-drop-frac .85
        --cav-drive "$CAV_DRIVE" --device cuda --threads "$OMP" --num-workers "$NUM_WORKERS"
        --batch-size 8 --sumo-fidelity --start-shift-prob 0 --truncation-steps 20
        --entries-v2 --entry-pad 8 --merge-cal --merge-cal-params "$MERGE_PARAMS"
        --traffic-loss --traffic-weight 0.1 --traffic-ramp-start-s 0 --traffic-ramp-end-s 1
        --projection --projection-min-gap-m 2.0 --residual-cap-mps2 1.5 --lane-logit-cap 3.0
        --stability-weight 0.01 --stability-margin 1.0 --stability-probes 1
        --conservation-metric --conservation-weight 0.01 --loss-share-logging)
# Ramp 0..1 s explicitly activates the traffic objective even in the 10 s smoke.
# SW_DRIVER is staged/checked, but ground-truth driver parameters never enter loss.
if [[ -n "$SW_WINDOWS_VAL" ]]; then COMMON+=(--switch-val-window-list "$SW_WINDOWS_VAL"); fi
UNROLL=(--unroll-s 10 --unroll-s 30 --unroll-s 60 --unroll-s 120)
HELP=$("$PY" -m wms.cli train --help 2>&1) || die "CLI help failed: $HELP"
for arg in "${COMMON[@]}" "${MODEL[@]}" "${UNROLL[@]}" --stage-steps --out "${BRAKING_ENV[@]}" "${EXTRA[@]}" "${BRAKING[@]}"; do
  case "$arg" in
    --*) option=${arg%%=*}
         # Exact option token, not substring (e.g. --projection vs --projection-min-gap-m).
         [[ " $HELP " =~ (^|[^[:alnum:]_-])${option}([^[:alnum:]_-]|$) ]] || die "MISSING FLAG $option" ;;
  esac
done
for module in wms.gate_stages wms.plan_steps; do
  MOD_HELP=$("$PY" -m "$module" --help 2>&1) || die "$module help failed: $MOD_HELP"
  if [[ "$module" == wms.plan_steps ]]; then options=(--stage-steps --stages --max-hours)
  else options=(--stages --require-terms --ramp-end-s); fi
  for option in "${options[@]}"; do
    [[ " $MOD_HELP " =~ (^|[^[:alnum:]_-])${option}([^[:alnum:]_-]|$) ]] || die "$module MISSING FLAG $option"
  done
done
host_check disk "$SW_ROOT" "$MIN_FREE_BYTES"
host_check disk "$W" "$MIN_FREE_BYTES"
host_check idle
[[ ! -e "$W" && ! -L "$W" ]] || die "output exists: $W; choose a fresh W (failed runs are preserved)"
printf 'preflight passed BOX=%s PY=%s stages=%s steps=%s workers=0 max_hours=%s model=%s rollout=%s lane_head=%s terms=%s\n' "$BOX" "$PY" "$STAGES_CSV" "$STAGE_STEPS" "$MAX_HOURS" "$MODEL_KIND" "$ROLLOUT_MODE" "$LANE_HEAD" "$TERMS"
if [[ "$MODE" == preflight ]]; then exit 0; fi
[[ -d "$(dirname -- "$W")" ]] || die 'run output parent must exist'
mkdir -- "$W"
trap 'rc=$?; printf "FAILED rc=%s line=%s; all artifacts retained; no retry\n" "$rc" "$LINENO" >> "$W/chain.log"; exit "$rc"' ERR
trap 'printf "INTERRUPTED; artifacts retained\n" >> "$W/chain.log"; exit 130' INT TERM
printf 'RUN BOX=%s stage_steps=%s smoke=%s max_hours=%s\n' "$BOX" "$STAGE_STEPS" "$SMOKE_STAGE_STEPS" "$MAX_HOURS" > "$W/chain.log"
printf '%s\n' "$HELP" > "$W/cli_help.txt"
printf '%q ' "$PY" -m wms.cli train "${COMMON[@]}" "${MODEL[@]}" "${UNROLL[@]}" "${BRAKING_ENV[@]}" "${EXTRA[@]}" "${BRAKING[@]}" > "$W/common_command.txt"
printf '\n' >> "$W/common_command.txt"

# Smoke timing wrapper samples CPU every second and RSS per child process.
# A contaminated sample invalidates timing, terminates the process group, and
# refuses the full run. It never restarts training. Full run retains RSS guard.
guarded_train() {
  local name=$1 steps=$2; shift 2
  "$PY" - "$W" "$name" "$OMP" "$PY" -m wms.cli train \
    "${COMMON[@]}" "${MODEL[@]}" "${UNROLL[@]}" "${BRAKING_ENV[@]}" "${EXTRA[@]}" "${BRAKING[@]}" \
    --stage-steps "$steps" --out "$W/$name" <<'PY'
import json, os, pathlib, signal, subprocess, sys, time
work, name, omp, *command = sys.argv[1:]
root = pathlib.Path(work)
def interrupted(signum, frame):
    raise SystemExit(128 + signum)
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
def cpu():
    v = list(map(int, pathlib.Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    return sum(v), v[3] + v[4]
def rss_group(pgid):
    result = []
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            if os.getpgid(int(p.name)) != pgid: continue
            rss = int((p / 'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
            result.append({'pid': int(p.name), 'rss_bytes': rss})
        except (FileNotFoundError, ProcessLookupError, PermissionError): pass
    return result
child = None
try:
    with (root / f'{name}.log').open('x') as log, (root / f'{name}_host.jsonl').open('x') as samples:
        # A separate immediate idle check protects timing after preflight work.
        a, b = cpu(); time.sleep(1); c, d = cpu()
        before = (1 - (d-b)/max(1,c-a)) * (os.cpu_count() or 1)
        if name == 'smoke_stages' and (before > 1 or os.getloadavg()[0] > 1):
            raise RuntimeError('CPU contention immediately before timing')
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        samples.write(json.dumps({'event': 'started', 'pid': child.pid, 'command': command}) + '\n'); samples.flush()
        a, b = cpu()
        while True:
            time.sleep(1)
            c, d = cpu(); busy = (1 - (d-b)/max(1,c-a)) * (os.cpu_count() or 1)
            rss = rss_group(child.pid)
            row = {'time': time.time(), 'busy_cores': busy, 'load1': os.getloadavg()[0], 'rss': rss}
            samples.write(json.dumps(row) + '\n'); samples.flush()
            if any(p['rss_bytes'] >= 6_000_000_000 for p in rss):
                raise RuntimeError('per-process RSS reached 6 GB; no retry')
            # Allow trainer threads plus one coordinator core, never whole-host saturation.
            if name == 'smoke_stages' and busy > int(omp) + 1:
                raise RuntimeError('CPU contention during timing; projection invalid; no retry')
            rc = child.poll()
            if rc is not None:
                samples.write(json.dumps({'event': 'finished', 'returncode': rc}) + '\n')
                raise SystemExit(rc)
except BaseException:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try: child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait()
    raise
PY
}
guarded_train smoke_stages "$SMOKE_STAGE_STEPS"
"$PY" -m wms.gate_stages "$W/smoke_stages/train_log.jsonl" \
  --stages "$STAGES_CSV" --require-terms "$TERMS" --ramp-end-s 1 \
  > "$W/smoke_stages_check.txt" 2>&1
# Require actual coverage, not merely four lines (duplicates would pass wc -l).
"$PY" - "$W/smoke_stages/loss_share.jsonl" "$W/smoke_stages_check.txt" "$W/smoke_stages/train_log.jsonl" "$TERMS" <<'PY'
import json, math, pathlib, sys
terms = set(sys.argv[4].split(','))
rows = [json.loads(s) for s in pathlib.Path(sys.argv[1]).read_text().splitlines() if s.strip()]
stages = set()
for row in rows:
    share = row.get('loss_share', {})
    if not terms <= share.keys(): raise SystemExit('missing loss-share term')
    if any(isinstance(share[t], bool) or not isinstance(share[t], (int, float)) or not math.isfinite(share[t]) or share[t] < 0 for t in terms):
        raise SystemExit('invalid loss-share value')
    stages.add(float(row['stage_s']))
if stages != {10., 30., 60., 120.}: raise SystemExit('loss shares must cover exactly four stages')
counts = {s: 0 for s in stages}
for line in pathlib.Path(sys.argv[3]).read_text().splitlines():
    if not line.strip(): continue
    row = json.loads(line)
    stage = float(row['stage_s'])
    if stage not in counts: raise SystemExit('unexpected smoke stage')
    counts[stage] += 1
    for term in terms | {'seconds'}:
        value = row.get(term)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SystemExit(f'missing/nonfinite smoke term {term}')
    if row['seconds'] <= 0: raise SystemExit('invalid smoke step timing')
if any(n != 6 for n in counts.values()): raise SystemExit('require exactly six smoke steps per stage')
result = json.loads(pathlib.Path(sys.argv[2]).read_text().splitlines()[-1])
if result.get('status') != 'passed': raise SystemExit('smoke gate did not pass')
PY
"$PY" -m wms.plan_steps "$W/smoke_stages/train_log.jsonl" \
  --stage-steps "$STAGE_STEPS" --stages "$STAGES_CSV" --max-hours "$MAX_HOURS" \
  > "$W/projected_hours.txt" 2> "$W/projected_hours.err"
# Independent finite/cap/budget check: a planner success must not silently shrink.
"$PY" - "$W/smoke_stages/plan_steps.json" "$MAX_HOURS" <<'PY'
import json, math, pathlib, sys
p = json.loads(pathlib.Path(sys.argv[1]).read_text())
h = float(p['projected_hours'])
if not math.isfinite(h) or h <= 0 or h > float(sys.argv[2]): raise SystemExit('projected hours refused')
if p['stage_steps'] != [2500, 2500, 1875, 1250] or p['stages_s'] != [10, 30, 60, 120]:
    raise SystemExit('planner changed fixed budget')
PY
host_check disk "$W" "$MIN_FREE_BYTES" >> "$W/chain.log"
printf 'four-stage smoke and explicit-hours gate passed; starting full\n' >> "$W/chain.log"
guarded_train full "$STAGE_STEPS"
printf 'full finished rc=0\n' >> "$W/chain.log"
