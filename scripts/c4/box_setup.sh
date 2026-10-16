#!/usr/bin/env bash
# One-time setup of a C4 box (GPU box or CPU box).  LF only.  Safe to rerun (idempotent, never overwrites).
#
#   scp -r <bundle dir> gpu-host:/workspace/c4/bundle        # the bundle dir also holds box_setup.sh + unpack_bundle.py
#   ssh gpu-host 'cd /workspace/c4/bundle && bash box_setup.sh --detach'    # survives ssh drops
#   ssh gpu-host 'tail -f /workspace/c4/setup.log; cat /workspace/c4/SETUP_REPORT.txt'
#
# Options:  --role gpu|cpu|both   which data to unpack (cpu: code, ckpts, lists, sidecars; no big packs)
#           --bundle DIR          default: the directory of this script if it holds SHA256SUMS, else $C4_HOME/bundle
#           --skip-sumo --skip-tests --no-verify-tars --detach
# Env:      C4_HOME (default /workspace/c4), PY, MIN_DISK_GB (default 60 gpu / 15 cpu), STRICT_TESTS=1
#
# What it does: environment report (python, torch/CUDA, pandas==2.3.3, pyarrow, onnxruntime, SUMO 1.27.1,
# nproc + cgroup CPU quota, RAM, disk) -> unpack + sha verify -> write env.sh/UNPACK_OK -> unit-test smoke.
# It installs only: pandas==2.3.3, pyarrow, the requirements.txt pins and (if missing) eclipse-sumo/traci/libsumo 1.27.1
# via pip.  It never touches torch, CUDA, drivers or system packages.
if LC_ALL=C grep -q $'\r' "${BASH_SOURCE[0]}"; then echo 'REFUSED: CRLF bytes in script' >&2; exit 2; fi
set -uo pipefail
SELF_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
C4_HOME=${C4_HOME:-/workspace/c4}
ROLE=both; BUNDLE=""; SKIP_SUMO=0; SKIP_TESTS=0; VERIFY=1; DETACH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) ROLE=$2; shift 2 ;;
    --bundle) BUNDLE=$2; shift 2 ;;
    --skip-sumo) SKIP_SUMO=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --no-verify-tars) VERIFY=0; shift ;;
    --detach) DETACH=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
case "$ROLE" in gpu|cpu|both) ;; *) echo "--role must be gpu, cpu or both" >&2; exit 2 ;; esac
if [[ -z "$BUNDLE" ]]; then
  if [[ -f "$SELF_DIR/SHA256SUMS" ]]; then BUNDLE=$SELF_DIR; else BUNDLE=$C4_HOME/bundle; fi
fi
mkdir -p "$C4_HOME"
if [[ "$DETACH" == 1 ]]; then
  nohup setsid bash "${BASH_SOURCE[0]}" --role "$ROLE" --bundle "$BUNDLE" \
    $([[ $SKIP_SUMO == 1 ]] && echo --skip-sumo) $([[ $SKIP_TESTS == 1 ]] && echo --skip-tests) \
    $([[ $VERIFY == 0 ]] && echo --no-verify-tars) > "$C4_HOME/setup.log" 2>&1 < /dev/null &
  echo "box_setup running detached (pid $!); log: $C4_HOME/setup.log"
  exit 0
fi

REPORT=$C4_HOME/SETUP_REPORT.txt
: > "$REPORT"
say() { printf '%s\n' "$*" | tee -a "$REPORT"; }
FAILS=0; WARNS=0
ok() { say "  OK    $*"; }
warn() { say "  WARN  $*"; WARNS=$((WARNS + 1)); }
fail() { say "  FAIL  $*"; FAILS=$((FAILS + 1)); }

say "== C4 box setup $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) role=$ROLE bundle=$BUNDLE home=$C4_HOME"
UNPACK_PY=$SELF_DIR/unpack_bundle.py
[[ -f "$UNPACK_PY" ]] || UNPACK_PY=$BUNDLE/unpack_bundle.py
[[ -f "$UNPACK_PY" ]] || { fail "unpack_bundle.py not found next to box_setup.sh or in the bundle"; exit 1; }
[[ -f "$BUNDLE/SHA256SUMS" ]] || { fail "$BUNDLE/SHA256SUMS missing (transfer incomplete?)"; exit 1; }

# ---------------------------------------------------------------- python
say "-- python"
PY=${PY:-}
for cand in "$PY" /workspace/gpu1/env/rd_gpu/bin/python /root/miniconda3/bin/python python3 python; do
  [[ -n "$cand" ]] || continue
  if command -v "$cand" >/dev/null 2>&1; then PY=$(command -v "$cand"); break; fi
done
[[ -n "$PY" ]] || { fail "no python interpreter"; exit 1; }
PYV=$("$PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' && ok "python $PYV at $PY" || fail "python >= 3.10 required (found $PYV)"

pipi() { "$PY" -m pip install --no-input --disable-pip-version-check "$@" >> "$C4_HOME/setup_pip.log" 2>&1; }
modver() { "$PY" - "$1" <<'PYEOF' 2>/dev/null
import importlib, sys
try:
    print(importlib.import_module(sys.argv[1]).__version__)
except Exception:
    print("missing")
PYEOF
}
say "-- python packages (pins: pandas==2.3.3 is required; pandas 3 breaks the outcome-sidecar dtype check)"
V=$(modver numpy);  [[ "$V" != missing ]] && ok "numpy $V" || { pipi numpy && ok "numpy installed" || fail "numpy"; }
V=$(modver pandas)
if [[ "$V" == 2.3.3 ]]; then ok "pandas $V"; else
  say "  pandas is '$V', installing pandas==2.3.3"; pipi "pandas==2.3.3" && ok "pandas $(modver pandas)" || fail "pandas==2.3.3 install failed (see $C4_HOME/setup_pip.log)"
fi
V=$(modver pyarrow); [[ "$V" != missing ]] && ok "pyarrow $V" || { pipi pyarrow && ok "pyarrow $(modver pyarrow)" || fail "pyarrow"; }
V=$(modver onnxruntime); [[ "$V" != missing ]] && ok "onnxruntime $V" || { pipi "onnxruntime==1.20.1" && ok "onnxruntime installed" || warn "onnxruntime (needed for the CIRCLES damping controller in SUMO)"; }
V=$(modver torch)
if [[ "$V" == missing ]]; then
  [[ "$ROLE" == cpu ]] && warn "torch missing (CPU role: needed by the learned policy / WM check in SUMO)" || fail "torch missing (do not pip-install blindly; use the box's CUDA build)"
else
  CUDA=$("$PY" -c 'import torch; print("cuda=%s devices=%d name=%s mem_gb=%.0f" % (torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", torch.cuda.get_device_properties(0).total_memory/2**30 if torch.cuda.is_available() else 0))' 2>/dev/null)
  ok "torch $V ($CUDA)"
  if [[ "$ROLE" != cpu && "$CUDA" != cuda=True* ]]; then fail "gpu role but torch.cuda.is_available() is False"; fi
fi
V=$(modver pytest); [[ "$V" != missing ]] && ok "pytest $V" || { pipi pytest && ok "pytest installed" || warn "pytest not installable; tests skipped"; }

# ---------------------------------------------------------------- SUMO
if [[ "$SKIP_SUMO" == 0 ]]; then
  say "-- SUMO (need 1.27.1 for the CIRCLES scenarios; sumo binary + traci)"
  SUMOV=$(sumo --version 2>/dev/null | head -1 || true)
  if [[ "$SUMOV" != *1.27.1* ]]; then
    say "  sumo on PATH: '${SUMOV:-none}'; installing eclipse-sumo/traci/libsumo 1.27.1 with pip"
    if pipi "eclipse-sumo==1.27.1" "traci==1.27.1" "libsumo==1.27.1"; then
      SUMO_HOME=$("$PY" -c 'import sumo; print(sumo.SUMO_HOME)' 2>/dev/null || true)
      [[ -n "$SUMO_HOME" ]] && export SUMO_HOME PATH="$SUMO_HOME/bin:$PATH"
      SUMOV=$(sumo --version 2>/dev/null | head -1 || true)
    else
      warn "pip could not install eclipse-sumo==1.27.1 (no wheel / no network); see $C4_HOME/setup_pip.log"
    fi
  fi
  if [[ "$SUMOV" == *1.27.1* ]]; then ok "$SUMOV ($(command -v sumo))"
  else
    fail "SUMO 1.27.1 unavailable. Fallbacks: (a) apt: sudo add-apt-repository ppa:sumo/stable && sudo apt-get install sumo sumo-tools (check the version afterwards); (b) build tag v1_27_1 from github.com/eclipse-sumo/sumo; then rerun with SUMO_HOME set"
  fi
  V=$(modver traci); [[ "$V" != missing ]] && ok "traci importable" || fail "traci missing"
fi

# ---------------------------------------------------------------- resources
say "-- resources"
NPROC=$(nproc 2>/dev/null || echo 1)
QUOTA=$NPROC
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  read -r q p < /sys/fs/cgroup/cpu.max || true
  if [[ "${q:-max}" != max && "${p:-0}" -gt 0 ]]; then QUOTA=$(( (q + p - 1) / p )); fi
elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
  q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
  if [[ "$q" -gt 0 && "$p" -gt 0 ]]; then QUOTA=$(( (q + p - 1) / p )); fi
fi
(( QUOTA > NPROC )) && QUOTA=$NPROC
if (( QUOTA < NPROC )); then warn "nproc=$NPROC but the cgroup CPU quota allows only $QUOTA cores -> use $QUOTA workers"; else ok "cores: nproc=$NPROC, usable=$QUOTA"; fi
MEM_GB=$(free -g 2>/dev/null | awk '/^Mem:/ {print $2}'); ok "RAM ${MEM_GB:-?} GB total, $(free -g 2>/dev/null | awk '/^Mem:/ {print $7}') GB available"
FREE=$(df -Pk "$C4_HOME" | awk 'NR==2 {printf "%d", $4/1048576}')
NEED=${MIN_DISK_GB:-$([[ "$ROLE" == cpu ]] && echo 15 || echo 60)}
(( FREE >= NEED )) && ok "disk: ${FREE} GB free under $C4_HOME (need ${NEED})" || fail "disk: only ${FREE} GB free under $C4_HOME (need ${NEED})"
if command -v nvidia-smi >/dev/null 2>&1; then ok "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | head -1)"; else
  [[ "$ROLE" == cpu ]] && ok "no GPU (cpu role)" || warn "nvidia-smi not found"
fi
say "   (sparse pack arrays: the apparent size of data_cache is ~75 GB for ~17 GB of real data; check 'du -sh' vs 'du -sh --apparent-size' after unpacking)"
if (( FAILS > 0 )); then say "== environment has $FAILS failure(s); not unpacking. Fix and rerun."; exit 1; fi

# ---------------------------------------------------------------- unpack
say "-- unpack"
ONLY=""
# CPU box = SUMO collection/evaluation: code, policy-loader checkpoints, window lists only
if [[ "$ROLE" == cpu ]]; then ONLY="--only c4_code,c4_ckpt,c4_data_c4_wm_lists"; fi
REPO=$C4_HOME/rd
UNPACK_OPTS=()
[[ "$VERIFY" == 1 ]] || UNPACK_OPTS+=(--no-verify-tars)
# shellcheck disable=SC2086
if "$PY" "$UNPACK_PY" --bundle "$BUNDLE" --root "$REPO" $ONLY "${UNPACK_OPTS[@]}" 2>&1 | tee -a "$C4_HOME/setup.log"; [[ ${PIPESTATUS[0]} -eq 0 ]]; then
  ok "bundle unpacked and sha256-verified into $REPO"
else
  fail "unpack failed (see above); nothing marked ready"; exit 1
fi
"$PY" - "$BUNDLE/BUNDLE_MANIFEST.json" > "$REPO/BUNDLE_COMMIT" <<'PYEOF'
import json, sys
print(json.load(open(sys.argv[1]))["git"]["commit"])
PYEOF
# shell scripts must be LF
BAD=$(grep -rlI $'\r' "$REPO/scripts/c4" --include=*.sh 2>/dev/null || true)
[[ -z "$BAD" ]] && ok "scripts/c4/*.sh have LF endings" || fail "CRLF in: $BAD"
chmod +x "$REPO"/scripts/c4/*.sh 2>/dev/null || true
cat > "$C4_HOME/env.sh" <<ENVEOF
# written by box_setup.sh $(date -u +%Y-%m-%dT%H:%M:%SZ)
PY=$PY
C4_CORES=$QUOTA
${SUMO_HOME:+SUMO_HOME=$SUMO_HOME}
ENVEOF
(( FAILS == 0 )) && touch "$C4_HOME/UNPACK_OK"

# ---------------------------------------------------------------- smoke
if [[ "$SKIP_TESTS" == 0 ]]; then
  say "-- unit-test smoke (tests/unit, CPU)"
  ( cd "$REPO" && PYTHONPATH="$REPO/src:$REPO" OMP_NUM_THREADS=2 timeout 1500 "$PY" -m pytest -q -x tests/unit -p no:cacheprovider 2>&1 | tail -15 ) | tee -a "$REPORT"
  RC=${PIPESTATUS[0]}
  if [[ $RC -eq 0 ]]; then ok "unit tests pass"; else
    if [[ "${STRICT_TESTS:-0}" == 1 ]]; then fail "unit tests failed (rc $RC)"; else warn "unit tests failed or timed out (rc $RC); read the output above before launching"; fi
  fi
  ( cd "$REPO" && PYTHONPATH="$REPO/src:$REPO" "$PY" -m traffic_shepherd inspect >/dev/null 2>&1 ) && ok "traffic_shepherd inspect" || warn "traffic_shepherd inspect failed"
  if [[ "$ROLE" != cpu && -f "$REPO/configs/c4_wm.toml" ]]; then
    # preflight needs the R5b init checkpoint (WMS_DATA_CKPT_R5B_BEST) and the zero-shot guard file; the guard is NOT in the
    # bundle (restricted-token name), so until it is placed by hand this check is expected to fail (OPS_NOTES section 11).
    ( cd "$REPO" && WMS_DATA_CKPT_R5B_BEST="${WMS_DATA_CKPT_R5B_BEST:-$REPO/data_cache/ckpt_r5b/wm_s_best.pt}" PYTHONPATH="$REPO/src:$REPO" "$PY" -m wms preflight --config configs/c4_wm.toml --out "$C4_HOME/runs/preflight_setup_$(date +%s)" 2>&1 | tail -25 ) | tee -a "$REPORT"
    [[ ${PIPESTATUS[0]} -eq 0 ]] && ok "wms preflight configs/c4_wm.toml" || warn "wms preflight did not pass (expected until the zero-shot guard file is placed; see OPS_NOTES section 11)"
  fi
fi

say "== setup finished: $FAILS failure(s), $WARNS warning(s).  Next: bash $REPO/scripts/c4/run_wm.sh preflight   (GPU box)"
(( FAILS == 0 ))
