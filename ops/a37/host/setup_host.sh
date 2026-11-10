#!/bin/bash
# AMENDMENT 37 host setup (run as root from the shipped dir; idempotent; never deletes anything).
#   1. python env ($BASE/venv via uv, else system python3 -m venv): torch 2.8.0 CPU, numpy 2.3.2, eclipse-sumo 1.27.1
#   2. the staged plant + bindings (a32 relay stage.tar / src7.tar) at the absolute paths the sealed bindings name
#      (/root/autodl-fs/nm_finaleval_p20c_bank, /autodl-fs/data, /home/ubuntu/...): symlinks into $BASE/stage, created
#      only when absent; existing paths are accepted only if the pinned shas match
#   3. src = plant SRC b5268408_5ecc4358 + the A37 overlay (git blobs of the branch commit), final root $BASE/root
#   4. env file, zero-SUMO preflight (runner setup-check), CPU tests, prepare + dry for every setting
# Usage: bash setup_host.sh   (from the ship dir)  -> writes $BASE/SETUP_OK on success
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a37ct
mkdir -p "$BASE"
LOG=$BASE/setup.log
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date '+%F %T') $*"; }
die() { say "SETUP FAILED: $*"; echo "$*" > "$BASE/SETUP_FAILED"; exit 1; }
rm -f "$BASE/SETUP_FAILED" 2>/dev/null
say "setup start ship=$SHIP"
[ "$(id -u)" = 0 ] || die "must run as root (sudo)"
cd "$SHIP" && sha256sum -c --quiet SHA256SUMS || die "ship SHA256SUMS mismatch"
OVL=$(cat "$SHIP/OVERLAY_ID")
NCPU=$(nproc)
FREE_GB=$(df -Pk "$BASE" | awk 'NR==2{print int($4/1048576)}')
MEM_GB=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
say "host $(hostname) nproc=$NCPU mem_avail_gb=$MEM_GB free_gb=$FREE_GB arch=$(uname -m) os=$(. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME") overlay=$OVL"
[ "$(uname -m)" = x86_64 ] || die "x86_64 required"
[ "$FREE_GB" -ge 12 ] || die "need >= 12 GB free on $BASE (have $FREE_GB)"

# ---------------------------------------------------------------- 1. python
PY=$BASE/venv/bin/python
if [ ! -x "$PY" ]; then
  if ! command -v uv >/dev/null 2>&1 && [ ! -x /root/.local/bin/uv ] && [ ! -x /root/.cargo/bin/uv ]; then
    say "installing uv (astral.sh)"
    (curl -LsSf https://astral.sh/uv/install.sh || wget -qO- https://astral.sh/uv/install.sh) | sh || say "uv install failed"
  fi
  UV=$(command -v uv || ls /root/.local/bin/uv /root/.cargo/bin/uv 2>/dev/null | head -1)
  if [ -n "$UV" ]; then
    "$UV" venv --python 3.12 "$BASE/venv" || die "uv venv failed"
    "$UV" pip install --python "$PY" pip || die "uv pip bootstrap failed"
  else
    python3 -m venv "$BASE/venv" || die "no uv and python3 -m venv failed (install python3-venv)"
  fi
fi
"$PY" -m pip install -q --no-input --upgrade pip >/dev/null 2>&1
"$PY" -c "import torch" 2>/dev/null || "$PY" -m pip install -q --no-input torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
  || "$PY" -m pip install -q --no-input torch --index-url https://download.pytorch.org/whl/cpu || die "torch install failed"
"$PY" -m pip install -q --no-input numpy==2.3.2 eclipse-sumo==1.27.1 traci==1.27.1 sumolib==1.27.1 libsumo==1.27.1 \
  pytest psutil pandas scipy cloudpickle || die "pip install failed"
"$PY" - <<'EOF' || die "python env check failed"
import importlib.metadata as m, sys, torch, numpy
v = {p: m.version(p) for p in ('eclipse-sumo', 'torch', 'numpy')}
print('python', sys.version.split()[0], v)
assert v['eclipse-sumo'] == '1.27.1', v
EOF

# ---------------------------------------------------------------- 2. staged plant + bindings
if [ ! -f "$BASE/stage/.done" ]; then
  mkdir -p "$BASE/stage" "$BASE/stage/src7"
  tar xf "$SHIP/stage.tar" -C "$BASE/stage" --no-same-owner || die "stage.tar extract"
  tar xf "$SHIP/src7.tar" -C "$BASE/stage/src7" --no-same-owner || die "src7.tar extract"
  mkdir -p "$BASE/stage/root/autodl-fs/nm_finaleval_p20c_bank/a32_banks"
  cp "$SHIP/GUIDANCE_C4SIT_priv_UNION_UNC.json" "$BASE/stage/root/autodl-fs/nm_finaleval_p20c_bank/a32_banks/"
  touch "$BASE/stage/.done"
fi
H=$BASE/stage/root/mining20261106/cpuv_home
link() {   # link <abs path> <target>: create only when absent
  if [ -e "$1" ] || [ -L "$1" ]; then say "exists: $1 -> $(readlink -f "$1")"; else mkdir -p "$(dirname "$1")"; ln -s "$2" "$1"; say "linked $1 -> $2"; fi
}
link /root/autodl-fs/nm_finaleval_p20c_bank "$BASE/stage/root/autodl-fs/nm_finaleval_p20c_bank"
link /root/autodl-fs/nm_finaleval_p20x_stage "$BASE/stage/autodl-fs/data/nm_finaleval_p20x_stage"
[ -e /autodl-fs/data ] || link /autodl-fs/data /root/autodl-fs
mkdir -p /home/ubuntu
for x in mappospeed cpus_env coopmwm20261106_clean4 e3final; do link "/home/ubuntu/$x" "$H/$x"; done
link /home/ubuntu/coopmwm20261104_launchcoop_src7 "$BASE/stage/src7/coopmwm20261104_launchcoop_src7"
link /home/ubuntu/nm_finaleval_p20x_stage /autodl-fs/data/nm_finaleval_p20x_stage
[ -f /root/autodl-fs/nm_finaleval_p20c_bank/a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json ] || {
  mkdir -p /root/autodl-fs/nm_finaleval_p20c_bank/a32_banks
  cp "$SHIP/GUIDANCE_C4SIT_priv_UNION_UNC.json" /root/autodl-fs/nm_finaleval_p20c_bank/a32_banks/; }
pin() { [ "$(sha256sum "$1" 2>/dev/null | cut -c1-64)" = "$2" ] || die "pin mismatch: $1 (want $2)"; say "pin ok ${2:0:8} $1"; }
pin /root/autodl-fs/nm_finaleval_p20c_bank/prerun_plant/A_checkpoint_000.pt 5ea2296470c2af2855f9e292b03eeaa62bafdfec08146712d7347d7ce117d734
pin /home/ubuntu/mappospeed/production2/policy_00169.pt d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca
pin /autodl-fs/data/nm_finaleval_p20c_bank/oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json fd385a2b8b1cbaf33897f8617b6d4c0abeb9f2dea40c9b72dca91d718bfc3c82
pin /autodl-fs/data/nm_finaleval_p20c_bank/oracle_banks/LEARNED_TRIGGER_MODEL.json 8e0ddd3e241ce0172736909133f0b745c58b6bf795e06ca5364e5755f7e3b720
pin /root/autodl-fs/nm_finaleval_p20c_bank/a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json 39f384a29fb8e952a94f918234c52bb3ff2059f2155b99f17b2f5c26a38ec239
pin /home/ubuntu/coopmwm20261106_clean4/inputs/map/wz3to2/wz3to2.net.xml b62778986347e8c21bca90d7ddcc8d9faeb288290a4dbdc3f53ee64ae8c21dfa

# ---------------------------------------------------------------- 3. src + final root
SRCDIR=$BASE/src_$OVL
if [ ! -d "$SRCDIR" ]; then
  cp -a "$BASE/stage/root/autodl-fs/nm_finaleval_p20c_stage/src_p20c_b5268408_5ecc4358" "$SRCDIR.tmp" || die "src copy"
  tar xf "$SHIP/overlay.tar" -C "$SRCDIR.tmp" --no-same-owner || die "overlay extract"
  mv "$SRCDIR.tmp" "$SRCDIR"
fi
ln -sfn "$SRCDIR" "$BASE/src"
R=$BASE/root
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks"
B=/root/autodl-fs/nm_finaleval_p20c_bank
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || cp "$B/$f" "$R/$f" || die "mirror $f"
  cmp -s "$B/$f" "$R/$f" || die "final root $f differs from the binding root"
done

# ---------------------------------------------------------------- 4. env, preflight, tests, dry
SHIM=/home/ubuntu/cpus_env/libmkl_intel_shim.so
cat > "$BASE/env_a37.sh" <<EOF
BASE=$BASE
SRC=$BASE/src
PY=$PY
export SUMO_HOME=\$(\$PY -c 'import sumo, os; print(os.path.dirname(sumo.__file__))')
export PATH="\$SUMO_HOME/bin:$BASE/venv/bin:\$PATH"
export PYTHONPATH="\$SRC/pytools:\$SRC/runtime_src:\$SRC/WM_PACKAGE:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
[ -f $SHIM ] && export LD_PRELOAD=$SHIM
export COOPMWM_MAP_SPEC=/home/ubuntu/coopmwm20261106_clean4/inputs/map/wz3to2/MAP_SPEC.json COOPMWM_G5_RUNTIME=1
cd \$SRC
EOF
# shellcheck disable=SC1090
source "$BASE/env_a37.sh"
"$PY" -c "import ctypes" 2>&1 | grep -qi "cannot be preloaded" && { say "LD_PRELOAD shim does not load here; disabled"; sed -i '/LD_PRELOAD/d' "$BASE/env_a37.sh"; unset LD_PRELOAD; }
nice -n 19 "$PY" -m tools.coopmwm.nm_eval_runner setup-check --checkpoint /home/ubuntu/mappospeed/production2/policy_00169.pt \
  --sha256 d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca > "$BASE/setup_check.json" 2>>"$LOG" \
  || die "runner setup-check failed (see $BASE/setup_check.json)"
say "runner setup-check PASS (zero SUMO)"
A37_MIRROR=$R nice -n 19 "$PY" -m pytest -q -p no:cacheprovider ops/test_a37_celltailor.py \
  pytools/tools/coopmwm/test_a37_cellbank_cpu.py pytools/tools/coopmwm/test_a32_arms_cpu.py \
  pytools/tools/coopmwm/test_a35_transfer_cpu.py > "$BASE/tests.out" 2>&1 || { tail -30 "$BASE/tests.out"; die "CPU tests failed"; }
say "tests: $(tail -1 "$BASE/tests.out")"
"$PY" ops/a37_celltailor.py prepare --final-root "$R" --setting a35_D3200_p20 > "$BASE/prepare.json" || die "prepare"
for s in a37v_D3200_p20 a37v_D2400_p20 a35_D3200_p20 a35_D2400_p20 a33; do
  "$PY" ops/a37_celltailor.py dry --final-root "$R" --setting $s > "$BASE/dry_$s.json" 2>>"$LOG" || die "dry $s"
done
say "dry OK; a37_sha256 $(grep -o '"a37_sha256": "[0-9a-f]*"' "$BASE/dry_a33.json")"
WORKERS=$(( NCPU - 2 )); MW=$(( MEM_GB * 10 / 15 )); [ "$MW" -lt "$WORKERS" ] && WORKERS=$MW
[ "$WORKERS" -ge 4 ] || die "too few workers ($WORKERS)"
echo "$WORKERS" > "$BASE/WORKERS"
date '+%F %T' > "$BASE/SETUP_OK"
say "SETUP_OK workers=$WORKERS overlay=$OVL"
