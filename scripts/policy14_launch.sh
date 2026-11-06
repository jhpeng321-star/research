#!/usr/bin/env bash
# POLICY14 staging launcher. Foreground, sequential, from scratch; LF only.
# Partial guard preparation: exact ROBUST support/lane-action limits remain
# blocked on evidence availability; see POLICY14_PREP.md before launch.
set -Eeuo pipefail

die() { printf 'REFUSED: %s\n' "$*" >&2; exit 2; }
usage() {
  cat <<'HELP'
Usage: bash scripts/policy14_launch.sh WM_CKPT WM_CONFIG REAL_RETURNS [options]
  --qualification FILE   Required user-supplied qualified-WM evidence JSON
  --wm-member CKPT CONFIG Add another qualified frozen WM member (repeatable)
  --output-root DIR      New campaign directory; parent must already exist
  --device cpu|cuda      Default cuda; --smoke requires/uses cpu
  --max-updates N        Per seed, 1..600 (default 600; smoke 1..2, default 2)
  --smoke                Disposable CPU integration run; no scientific claims
  --dry-run              Print three commands; no imports, writes or execution
  --help                 Show this help

Fixed seeds: 965140 965141 965142, run sequentially, no resume/init arguments.
Stage: /root/autodl-tmp/c4/v5/policy14_stage
Environment: PY (default python3), THREADS (default 8, range 1..8).
See POLICY14_REGISTRATION_DRAFT.md for the evidence JSON contract and gates.
HELP
}

if [[ ${1:-} == --help ]]; then usage; exit 0; fi
[[ $# -ge 3 ]] || { usage >&2; die 'supply WM_CKPT, WM_CONFIG and REAL_RETURNS in that order'; }
WM_CKPTS=("$1"); WM_CONFIGS=("$2"); REAL_RETURNS=$3; shift 3
STAGE=/root/autodl-tmp/c4/v5/policy14_stage
CONFIG=$STAGE/configs/c4_policy14.toml
OUT=/root/autodl-tmp/c4/v5/runs/c4_policy14_20261026
QUALIFICATION=
DEVICE=cuda; DEVICE_SET=0; MAX_UPDATES=600; UPDATES_SET=0
SMOKE=0; DRY_RUN=0
PY=${PY:-python3}; THREADS=${THREADS:-8}
while (( $# )); do
  case "$1" in
    --qualification|--output-root|--device|--max-updates)
      [[ $# -ge 2 && -n $2 ]] || die "$1 requires a value"
      case "$1" in
        --qualification) QUALIFICATION=$2 ;;
        --output-root) OUT=$2 ;;
        --device) DEVICE=$2; DEVICE_SET=1 ;;
        --max-updates) MAX_UPDATES=$2; UPDATES_SET=1 ;;
      esac
      shift 2 ;;
    --wm-member)
      [[ $# -ge 3 && -n $2 && -n $3 ]] || die '--wm-member requires CKPT CONFIG'
      WM_CKPTS+=("$2"); WM_CONFIGS+=("$3"); shift 3 ;;
    --smoke) SMOKE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n $QUALIFICATION ]] || die '--qualification FILE is required (never auto-qualified)'
[[ $DEVICE == cpu || $DEVICE == cuda ]] || die '--device must be cpu or cuda'
[[ $THREADS =~ ^[1-8]$ ]] || die 'THREADS must be an integer in 1..8'
[[ $MAX_UPDATES =~ ^[1-9][0-9]{0,2}$ ]] || die '--max-updates must be an integer in 1..600'
(( MAX_UPDATES <= 600 )) || die 'per-seed budget is at most 600 updates'
if (( SMOKE )); then
  (( ! DEVICE_SET )) || [[ $DEVICE == cpu ]] || die '--smoke requires --device cpu'
  DEVICE=cpu
  (( UPDATES_SET )) || MAX_UPDATES=2
  (( MAX_UPDATES <= 2 )) || die 'smoke budget is at most two updates per seed'
  # Keep disposable artifacts separate even when using the default output root.
  OUT=${OUT}_smoke
fi
[[ $OUT == /* && $OUT != / && $OUT != */ ]] || die '--output-root must be an absolute path without a trailing slash'

SEEDS=(965140 965141 965142)
make_command() {
  CMD=("$PY" -m wms.control.policy14_cli --config "$CONFIG"
       --wm-ckpt "${WM_CKPTS[@]}" --wm-config "${WM_CONFIGS[@]}"
       --real-returns "$REAL_RETURNS" --device "$DEVICE" --seed "$1"
       --max-updates "$MAX_UPDATES" --output-dir "$OUT/seed_$1")
  (( ! SMOKE )) || CMD+=(--smoke)
}
print_command() { printf '%q ' "$@"; printf '\n'; }

if (( DRY_RUN )); then
  printf 'DRY RUN ONLY: no inputs qualified, no Python/GPU/SUMO execution, no writes.\n'
  printf 'Required before execution: 2026-10-26 or later (Asia/Shanghai), staged CLI/config,\n'
  printf 'lowmem config checks, qualified hash-matched evidence, approved TRAIN returns, new output root.\n'
  printf 'qualification=%q\nworking_directory=%q\noutput_root=%q\n' "$QUALIFICATION" "$STAGE" "$OUT"
  printf 'export OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q NUMEXPR_NUM_THREADS=%q VECLIB_MAXIMUM_THREADS=%q\n' \
    "$THREADS" "$THREADS" "$THREADS" "$THREADS" "$THREADS"
  printf 'export PYTHONPATH=%q\n' "$STAGE/src:$STAGE"
  for seed in "${SEEDS[@]}"; do make_command "$seed"; print_command "${CMD[@]}"; done
  exit 0
fi

# No date override or automatic wait/launch: the operator invokes this on/after Oct 26.
TODAY=$(TZ=Asia/Shanghai date +%F)
[[ $TODAY > 2026-10-25 ]] || die "not before 2026-10-26 Asia/Shanghai (today $TODAY)"
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ $REPO == "$STAGE" ]] || die "stage this source tree at $STAGE before execution (found $REPO)"
[[ -f $CONFIG && -f $STAGE/src/wms/control/policy14_cli.py ]] || die 'main must stage c4_policy14.toml and policy14_cli.py first'
command -v "$PY" >/dev/null 2>&1 || die "Python executable not found: $PY"
[[ -s $QUALIFICATION && -f $QUALIFICATION ]] || die 'qualification evidence JSON is missing/empty'
[[ -s $REAL_RETURNS && -f $REAL_RETURNS ]] || die 'REAL_RETURNS must be a nonempty approved TRAIN file'
for i in "${!WM_CKPTS[@]}"; do
  [[ -s ${WM_CKPTS[$i]} && -f ${WM_CKPTS[$i]} ]] || die "missing WM checkpoint: ${WM_CKPTS[$i]}"
  [[ -s ${WM_CONFIGS[$i]} && -f ${WM_CONFIGS[$i]} ]] || die "missing WM config: ${WM_CONFIGS[$i]}"
done
# Resolve caller-relative inputs before changing to the staged source directory.
QUALIFICATION=$(readlink -f -- "$QUALIFICATION")
REAL_RETURNS=$(readlink -f -- "$REAL_RETURNS")
for i in "${!WM_CKPTS[@]}"; do
  WM_CKPTS[$i]=$(readlink -f -- "${WM_CKPTS[$i]}")
  WM_CONFIGS[$i]=$(readlink -f -- "${WM_CONFIGS[$i]}")
done
[[ ! -e $OUT && ! -L $OUT ]] || die "output root already exists: $OUT (choose a new root)"
OUT_PARENT=$(cd -- "$(dirname -- "$OUT")" && pwd -P) || die 'output parent must already exist'
OUT=$OUT_PARENT/$(basename -- "$OUT")
[[ ! -e $OUT && ! -L $OUT ]] || die 'resolved output root already exists'

export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS
export PYTHONPATH="$STAGE/src:$STAGE"
# Explicit TRAIN numeric sources, independent of the selected WM run's data tables.
export WMS_DATA_V5_NATIVE_Q=${WMS_DATA_V5_NATIVE_Q:-/root/autodl-tmp/c4/v5/data/v5_native_q}
export WMS_DATA_GENRAND_PLAIN_Q=${WMS_DATA_GENRAND_PLAIN_Q:-/root/autodl-tmp/c4/v5/data/genrand_plain_q}
# Preserve the CLI path guard; add only the explicit input parents and campaign parent.
export C4_ALLOWED_ROOTS="$STAGE:$OUT_PARENT:$(cd -- "$(dirname -- "$REAL_RETURNS")" && pwd -P)"
C4_ALLOWED_ROOTS+=":$WMS_DATA_V5_NATIVE_Q:$WMS_DATA_GENRAND_PLAIN_Q"
for i in "${!WM_CKPTS[@]}"; do
  C4_ALLOWED_ROOTS+=":$(cd -- "$(dirname -- "${WM_CKPTS[$i]}")" && pwd -P)"
  C4_ALLOWED_ROOTS+=":$(cd -- "$(dirname -- "${WM_CONFIGS[$i]}")" && pwd -P)"
done

# Standard-library-only preflight. A supplied assertion is checked, never created.
# Hashing a checkpoint does not deserialize it or import a model/runtime.
PREFLIGHT=$("$PY" - "$CONFIG" "$QUALIFICATION" "$REAL_RETURNS" "$THREADS" \
  "${#WM_CKPTS[@]}" "${WM_CKPTS[@]}" "${WM_CONFIGS[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    path = guarded(path)
    require(path.is_file() and path.stat().st_size > 0, f"missing/empty file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


config_hashes = {}


def guarded(path):
    pattern = r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth"
    require(not re.search(pattern, str(path), re.I), "restricted input path")
    path = Path(path).resolve(strict=True)
    require(not re.search(pattern, str(path), re.I), "restricted resolved input path")
    return path


def config(path, seen=()):
    path = guarded(path)
    require(path not in seen and len(seen) < 16, "config inheritance cycle/depth")
    config_hashes[str(path)] = digest(path)
    with path.open("rb") as stream:
        raw = json.load(stream) if path.suffix == ".json" else tomllib.load(stream)
    raw = raw.get("config", raw)
    parent = raw.pop("extends", None)
    base = config(path.parent / parent, (*seen, path)) if parent else {}

    def merge(left, right):
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(left.get(key), dict):
                merge(left[key], value)
            else:
                left[key] = value
        return left

    return merge(base, raw)


try:
    policy_path, marker_path, returns_path, threads, count = sys.argv[1:6]
    count, threads = int(count), int(threads)
    checkpoints = sys.argv[6:6 + count]
    wm_configs = sys.argv[6 + count:]
    require(len(wm_configs) == count, "WM checkpoint/config count mismatch")
    require(len({Path(p).resolve() for p in checkpoints}) == count, "duplicate WM member")
    raw = config(policy_path)
    trainer, cf, run = (raw.get(k, {}) for k in ("trainer", "counterfactual", "run"))
    require(type(trainer.get("batch_size")) is int and trainer["batch_size"] in (1, 2), "batch_size must be 1 or 2")
    require(trainer.get("horizon") == 10, "horizon must be 10 control seconds")
    require(trainer.get("checkpoint_imagination") is True, "checkpoint_imagination must be true")
    require(raw.get("lanes", {}).get("consequence_mask") is True, "CAVCUTIN consequence mask must be enabled")
    require(trainer.get("low_penetration_horizon_cap") is True, "observed low-penetration cap must be enabled")
    require(trainer.get("wm_safety_report_only") is True, "WM safety must remain report-only")
    require(raw.get("policy", {}).get("accel_max_mps2") == 1.2, "TRAIN acceleration ceiling must be 1.2")
    require(cf.get("native_accel_max_mps2") == 1.2, "native reference must share the TRAIN acceleration ceiling")
    require(cf.get("max_starts") == 1, "counterfactual.max_starts must be 1")
    require(trainer.get("credit_profile_every", 0) == 0, "disable memory-spiking credit profiling")
    require(type(run.get("threads")) is int and 1 <= run["threads"] <= threads <= 8, "run.threads must be <= THREADS <= 8")
    for section in (run, trainer, raw.get("policy", {})):
        require(not any(section.get(k) for k in ("resume", "init_policy", "init_from")), "policy must start from scratch")
    marker = guarded(marker_path)
    marker_hash = digest(marker)
    qualification = json.loads(marker.read_text(encoding="utf-8"))
    require(qualification.get("status") == "qualified", "marker status must be qualified")
    require(isinstance(qualification.get("qualified_by"), str) and qualification["qualified_by"].strip(), "qualified_by is required")
    evidence = qualification.get("evidence_file")
    require(isinstance(evidence, str) and evidence.strip(), "evidence_file is required")
    report = guarded(marker.parent / evidence)
    require(report != marker, "qualification must reference a separate evidence report")
    require(digest(report) == qualification.get("evidence_sha256"), "evidence report hash mismatch")
    members = qualification.get("members")
    require(isinstance(members, list) and len(members) == count, "qualification must cover every WM member in CLI order")
    records = []
    for ckpt, cfg, member in zip(checkpoints, wm_configs, members):
        ckpt_hash, cfg_hash = digest(ckpt), digest(cfg)
        require(member.get("checkpoint_sha256") == ckpt_hash, f"unqualified checkpoint/hash mismatch: {ckpt}")
        require(member.get("wm_config_sha256") == cfg_hash, f"unqualified WM config/hash mismatch: {cfg}")
        config(cfg)  # Validate/record inheritance without loading data.
        sidecars = {name: digest(Path(ckpt).with_name(name)) for name in ("physics.json", "TRAIN_RECEIPT.json")}
        records.append(dict(checkpoint=str(Path(ckpt).resolve()), checkpoint_sha256=ckpt_hash,
                            wm_config=str(Path(cfg).resolve()), wm_config_sha256=cfg_hash, sidecars=sidecars))
    require(len({r["checkpoint_sha256"] for r in records}) == count, "copied checkpoints are not distinct ensemble members")
    print(json.dumps(dict(qualification_file=str(marker), qualification_sha256=marker_hash,
                         qualified_by=qualification["qualified_by"], evidence_file=str(report),
                         evidence_sha256=qualification["evidence_sha256"], members=records,
                         config_sha256=config_hashes, resolved_policy_config=raw,
                         real_returns=str(Path(returns_path).resolve()), real_returns_sha256=digest(returns_path),
                         epistemic="unavailable" if count == 1 else "ensemble disagreement; calibration required"), indent=2))
except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
    sys.exit(f"REFUSED: {error}")
PY
) || die 'preflight failed; no training started'

# Atomic new-root reservation prevents an accidental second invocation from resuming.
mkdir -- "$OUT"
printf '%s\n' "$PREFLIGHT" > "$OUT/INPUT_AUDIT.json"
cp -- "$QUALIFICATION" "$OUT/QUALIFICATION.supplied.json"
cd -- "$STAGE"
for seed in "${SEEDS[@]}"; do
  make_command "$seed"
  {
    printf 'started_utc=%s\nseed=%s\nsmoke=%s\nthreads=%s\n' "$(date -u +%FT%TZ)" "$seed" "$SMOKE" "$THREADS"
    printf 'command='; print_command "${CMD[@]}"
  } > "$OUT/seed_${seed}.launch.txt"
  # Foreground and fail-fast: the next seed starts only after exit 0, no retries.
  "${CMD[@]}" > "$OUT/seed_${seed}.train.log" 2>&1
done
printf 'Completed three sequential POLICY14 runs: %s\n' "$OUT"
