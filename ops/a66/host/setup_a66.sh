#!/bin/bash
# AMENDMENT 66 host setup (new host; run as root from the ship dir; idempotent; never deletes anything).
# Adapted copy of ops/a50/host/setup_host.sh (a497230b): base /root/a66, python from /opt/py312, src = plant SRC
# b5268408_5ecc4358 + the A50 overlay a497230b9c8a + the A52 / A54 / A66 driver files, env_a66.sh, zero-SUMO
# preflight + the same CPU tests. No A50 prepare / dry (no A50 root is written here). Zero SUMO.
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a66
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
say "host $(hostname) nproc=$(nproc --all) arch=$(uname -m) overlay=$OVL a66=$(cat "$SHIP/A66_CODE_TIP")"

# ---------------------------------------------------------------- 1. python
PY=$BASE/venv/bin/python
if [ ! -x "$PY" ]; then
  /opt/py312/bin/python3.12 -m venv "$BASE/venv" || die "venv failed"
fi
nice -n 19 "$PY" -m pip install -q --no-input --upgrade pip >/dev/null 2>&1
"$PY" -c "import torch" 2>/dev/null || nice -n 19 "$PY" -m pip install -q --no-input torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
  || die "torch install failed"
nice -n 19 "$PY" -m pip install -q --no-input numpy==2.3.2 eclipse-sumo==1.27.1 traci==1.27.1 sumolib==1.27.1 libsumo==1.27.1 \
  sumo-data==1.27.1 onnxruntime==1.29.0 pytest psutil pandas scipy cloudpickle || die "pip install failed"
SUMOBIN=$("$PY" -c 'import sumo, os; print(os.path.join(os.path.dirname(sumo.__file__), "bin", "sumo"))')
ldd "$SUMOBIN" | grep -q "not found" && die "sumo has missing libraries"
"$SUMOBIN" --version | head -1 | grep -q "1.27.1" || die "sumo --version is not 1.27.1"
"$PY" -c "
import importlib.metadata as m, sys
v = {p: m.version(p) for p in ('eclipse-sumo', 'torch', 'numpy', 'onnxruntime', 'pandas', 'scipy')}
print('python', sys.version.split()[0], v)
assert v['eclipse-sumo'] == '1.27.1' and v['torch'].startswith('2.8.0') and v['numpy'] == '2.3.2', v
" || die "python env check failed"
"$PY" -m pip freeze > "$BASE/pip_freeze.txt"

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
link() {
  if [ -e "$1" ] || [ -L "$1" ]; then say "exists: $1 -> $(readlink -f "$1")"; else mkdir -p "$(dirname "$1")"; ln -s "$2" "$1"; say "linked $1 -> $2"; fi
}
link /root/autodl-fs/nm_finaleval_p20c_bank "$BASE/stage/root/autodl-fs/nm_finaleval_p20c_bank"
link /root/autodl-fs/nm_finaleval_p20x_stage "$BASE/stage/autodl-fs/data/nm_finaleval_p20x_stage"
[ -e /autodl-fs/data ] || link /autodl-fs/data /root/autodl-fs
for x in mappospeed cpus_env coopmwm20261106_clean4 e3final; do link "/home/ubuntu/$x" "$H/$x"; done
link /home/ubuntu/coopmwm20261104_launchcoop_src7 "$BASE/stage/src7/coopmwm20261104_launchcoop_src7"
link /home/ubuntu/nm_finaleval_p20x_stage /autodl-fs/data/nm_finaleval_p20x_stage
pin() { [ "$(sha256sum "$1" 2>/dev/null | cut -c1-64)" = "$2" ] || die "pin mismatch: $1 (want $2)"; say "pin ok ${2:0:8} $1"; }
pin /root/autodl-fs/nm_finaleval_p20c_bank/prerun_plant/A_checkpoint_000.pt 5ea2296470c2af2855f9e292b03eeaa62bafdfec08146712d7347d7ce117d734
pin /home/ubuntu/mappospeed/production2/policy_00169.pt d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca
pin /autodl-fs/data/nm_finaleval_p20c_bank/oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json fd385a2b8b1cbaf33897f8617b6d4c0abeb9f2dea40c9b72dca91d718bfc3c82
pin /autodl-fs/data/nm_finaleval_p20c_bank/oracle_banks/LEARNED_TRIGGER_MODEL.json 8e0ddd3e241ce0172736909133f0b745c58b6bf795e06ca5364e5755f7e3b720
pin /root/autodl-fs/nm_finaleval_p20c_bank/a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json 39f384a29fb8e952a94f918234c52bb3ff2059f2155b99f17b2f5c26a38ec239
pin /home/ubuntu/coopmwm20261106_clean4/inputs/map/wz3to2/wz3to2.net.xml b62778986347e8c21bca90d7ddcc8d9faeb288290a4dbdc3f53ee64ae8c21dfa

# ---------------------------------------------------------------- 3. src + final root + policies + stored roots
SRCDIR=$BASE/src_${OVL}_a66
if [ ! -d "$SRCDIR" ]; then
  cp -a "$BASE/stage/root/autodl-fs/nm_finaleval_p20c_stage/src_p20c_b5268408_5ecc4358" "$SRCDIR.tmp" || die "src copy"
  tar xf "$SHIP/overlay.tar" -C "$SRCDIR.tmp" --no-same-owner || die "overlay extract"
  tar xf "$SHIP/a66_overlay.tar" -C "$SRCDIR.tmp" --no-same-owner || die "a66 overlay extract"
  mv "$SRCDIR.tmp" "$SRCDIR"
fi
ln -sfn "$SRCDIR" "$BASE/src"
R=$BASE/root
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks" "$BASE/inputs" "$BASE/refs"
B=/root/autodl-fs/nm_finaleval_p20c_bank
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || cp "$B/$f" "$R/$f" || die "mirror $f"
  cmp -s "$B/$f" "$R/$f" || die "final root $f differs from the binding root"
done
for f in policy_00048.pt policy_00024.pt; do [ -f "$BASE/inputs/$f" ] || cp "$SHIP/$f" "$BASE/inputs/$f" || die "policy $f"; done
pin "$BASE/inputs/policy_00048.pt" 7f0c88695161bae6a243801df94e92fbd27c87551d655be61d338fd369364443
pin "$BASE/inputs/policy_00024.pt" 30971d7effb46539a8cf3a3d34da10eb0ac841d8c600be368676669b801b5ff9
if [ ! -f "$BASE/refs.done" ]; then
  tar xf "$SHIP/refs.tar" -C "$BASE/refs" --no-same-owner || die "refs extract"
  (cd "$BASE/refs" && sha256sum -c --quiet "$SHIP/REFS_SHA256SUMS") || die "refs sha mismatch"
  chmod -R a-w "$BASE/refs"
  touch "$BASE/refs.done"
fi

# ---------------------------------------------------------------- 4. env, preflight, tests
SHIM=/home/ubuntu/cpus_env/libmkl_intel_shim.so
cat > "$BASE/env_a66.sh" <<EOF
BASE=$BASE
SRC=$BASE/src
PY=$PY
export SUMO_HOME=\$(\$PY -c 'import sumo, os; print(os.path.dirname(sumo.__file__))')
export PATH="\$SUMO_HOME/bin:$BASE/venv/bin:\$PATH"
export PYTHONPATH="\$SRC/pytools:\$SRC/runtime_src:\$SRC/WM_PACKAGE:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
[ -f $SHIM ] && export LD_PRELOAD=$SHIM
export A50_HOST_BASE=/root/a50v A52_HOST_BASE=/root/a52 A54_HOST_BASE=/root/a54 A66_HOST_BASE=$BASE COOPMWM_MAP_SPEC=/home/ubuntu/coopmwm20261106_clean4/inputs/map/wz3to2/MAP_SPEC.json COOPMWM_G5_RUNTIME=1
cd \$SRC
EOF
# shellcheck disable=SC1090
source "$BASE/env_a66.sh"
"$PY" -c "import ctypes" 2>&1 | grep -qi "cannot be preloaded" && { say "LD_PRELOAD shim does not load here; disabled"; sed -i '/LD_PRELOAD/d' "$BASE/env_a66.sh"; unset LD_PRELOAD; }
nice -n 19 "$PY" -m tools.coopmwm.nm_eval_runner setup-check --checkpoint /home/ubuntu/mappospeed/production2/policy_00169.pt \
  --sha256 d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca > "$BASE/setup_check.json" 2>>"$LOG" \
  || die "runner setup-check failed (see $BASE/setup_check.json)"
say "runner setup-check PASS (zero SUMO)"
A37_MIRROR=$R nice -n 19 "$PY" -m pytest -q -p no:cacheprovider ops/test_a42_bigp60.py ops/test_a37_celltailor.py \
  pytools/tools/coopmwm/test_a42_bigbank_cpu.py pytools/tools/coopmwm/test_a37_cellbank_cpu.py \
  pytools/tools/coopmwm/test_a32_arms_cpu.py pytools/tools/coopmwm/test_a35_transfer_cpu.py \
  ops/test_a45_divp60.py pytools/tools/coopmwm/test_a45_diverse_cpu.py ops/test_a50_fullp60.py > "$BASE/tests.out" 2>&1 || { tail -30 "$BASE/tests.out"; die "CPU tests failed"; }
say "tests: $(tail -1 "$BASE/tests.out")"
date '+%F %T' > "$BASE/SETUP_OK"
say "SETUP_OK overlay=$OVL"
