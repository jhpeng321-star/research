#!/bin/bash
# AMENDMENT 40 host setup (root, from the shipped dir; idempotent; never deletes or overwrites anything it did not
# create; never touches /root/a37ct). Separate root dir /root/a40gv.
#   1. python: reuse /root/a37ct/venv when it has eclipse-sumo 1.27.1 + torch (read-only use), else own venv
#   2. staged plant + bindings at the absolute paths the sealed bindings name: if A37 already linked them they are
#      only VERIFIED (pins); missing links are created into $BASE/stage (extracted from the shipped stage.tar)
#   3. src = plant SRC b5268408_5ecc4358 + the A40 overlay (git blobs of the branch commit, incl. the wz3to2v80 map),
#      final root $BASE/root = mirror of the binding root files (cmp-checked)
#   4. env file, zero-SUMO runner setup-check, CPU tests, dry for every A40 setting -> $BASE/SETUP_OK
# Zero SUMO. Usage: bash setup_a40.sh  (from the ship dir)
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a40gv
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
say "host $(hostname) nproc=$NCPU mem_avail_gb=$MEM_GB free_gb=$FREE_GB overlay=$OVL"
[ "$FREE_GB" -ge 10 ] || die "need >= 10 GB free on $BASE (have $FREE_GB)"

# ---------------------------------------------------------------- 1. python
PY=""
if [ -x /root/a37ct/venv/bin/python ] && /root/a37ct/venv/bin/python - <<'EOF' >/dev/null 2>&1
import importlib.metadata as m, torch, numpy, traci
assert m.version('eclipse-sumo') == '1.27.1'
EOF
then PY=/root/a37ct/venv/bin/python; say "python: reusing the A37 venv (read-only)"; fi
if [ -z "$PY" ]; then
  PY=$BASE/venv/bin/python
  if [ ! -x "$PY" ]; then
    UV=$(command -v uv || ls /root/.local/bin/uv /root/.cargo/bin/uv 2>/dev/null | head -1)
    if [ -n "$UV" ]; then "$UV" venv --python 3.12 "$BASE/venv" && "$UV" pip install --python "$PY" pip || die "uv venv failed"
    else python3 -m venv "$BASE/venv" || die "python3 -m venv failed"; fi
  fi
  "$PY" -c "import torch" 2>/dev/null || "$PY" -m pip install -q --no-input torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    || die "torch install failed"
  "$PY" -m pip install -q --no-input numpy==2.3.2 eclipse-sumo==1.27.1 traci==1.27.1 sumolib==1.27.1 libsumo==1.27.1 \
    pytest psutil pandas scipy cloudpickle || die "pip install failed"
fi
"$PY" -c "import importlib.metadata as m; v=m.version('eclipse-sumo'); print('sumo', v); assert v=='1.27.1'" || die "sumo version"

# ---------------------------------------------------------------- 2. staged plant + bindings
if [ ! -e /root/autodl-fs/nm_finaleval_p20c_bank ] || [ ! -e /home/ubuntu/mappospeed ]; then
  if [ ! -f "$BASE/stage/.done" ]; then
    mkdir -p "$BASE/stage/src7"
    tar xf "$SHIP/stage.tar" -C "$BASE/stage" --no-same-owner || die "stage.tar extract"
    tar xf "$SHIP/src7.tar" -C "$BASE/stage/src7" --no-same-owner || die "src7.tar extract"
    mkdir -p "$BASE/stage/root/autodl-fs/nm_finaleval_p20c_bank/a32_banks"
    cp "$SHIP/GUIDANCE_C4SIT_priv_UNION_UNC.json" "$BASE/stage/root/autodl-fs/nm_finaleval_p20c_bank/a32_banks/"
    touch "$BASE/stage/.done"
  fi
  STG=$BASE/stage
else
  STG=""
  say "binding paths already present (A37 setup): verifying pins only"
fi
link() { if [ -e "$1" ] || [ -L "$1" ]; then say "exists: $1"; else mkdir -p "$(dirname "$1")"; ln -s "$2" "$1"; say "linked $1 -> $2"; fi; }
if [ -n "$STG" ]; then
  H=$STG/root/mining20261106/cpuv_home
  link /root/autodl-fs/nm_finaleval_p20c_bank "$STG/root/autodl-fs/nm_finaleval_p20c_bank"
  link /root/autodl-fs/nm_finaleval_p20x_stage "$STG/autodl-fs/data/nm_finaleval_p20x_stage"
  [ -e /autodl-fs/data ] || link /autodl-fs/data /root/autodl-fs
  mkdir -p /home/ubuntu
  for x in mappospeed cpus_env coopmwm20261106_clean4 e3final; do link "/home/ubuntu/$x" "$H/$x"; done
  link /home/ubuntu/coopmwm20261104_launchcoop_src7 "$STG/src7/coopmwm20261104_launchcoop_src7"
  link /home/ubuntu/nm_finaleval_p20x_stage /autodl-fs/data/nm_finaleval_p20x_stage
fi
[ -f /root/autodl-fs/nm_finaleval_p20c_bank/a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json ] || {
  mkdir -p /root/autodl-fs/nm_finaleval_p20c_bank/a32_banks
  cp "$SHIP/GUIDANCE_C4SIT_priv_UNION_UNC.json" /root/autodl-fs/nm_finaleval_p20c_bank/a32_banks/; }
pin() { [ "$(sha256sum "$1" 2>/dev/null | cut -c1-64)" = "$2" ] || die "pin mismatch: $1 (want $2)"; say "pin ok ${2:0:8} $1"; }
B=/root/autodl-fs/nm_finaleval_p20c_bank
pin $B/prerun_plant/A_checkpoint_000.pt 5ea2296470c2af2855f9e292b03eeaa62bafdfec08146712d7347d7ce117d734
pin /home/ubuntu/mappospeed/production2/policy_00169.pt d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca
pin $B/oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json fd385a2b8b1cbaf33897f8617b6d4c0abeb9f2dea40c9b72dca91d718bfc3c82
pin $B/oracle_banks/LEARNED_TRIGGER_MODEL.json 8e0ddd3e241ce0172736909133f0b745c58b6bf795e06ca5364e5755f7e3b720
pin $B/a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json 39f384a29fb8e952a94f918234c52bb3ff2059f2155b99f17b2f5c26a38ec239
pin /home/ubuntu/coopmwm20261106_clean4/inputs/map/wz3to2/wz3to2.net.xml b62778986347e8c21bca90d7ddcc8d9faeb288290a4dbdc3f53ee64ae8c21dfa

# ---------------------------------------------------------------- 3. src + final root
PLANT_SRC=/root/autodl-fs/nm_finaleval_p20c_stage/src_p20c_b5268408_5ecc4358
[ -d "$PLANT_SRC" ] || PLANT_SRC=$(ls -d /root/a37ct/stage/root/autodl-fs/nm_finaleval_p20c_stage/src_p20c_b5268408_5ecc4358 \
  "$BASE"/stage/root/autodl-fs/nm_finaleval_p20c_stage/src_p20c_b5268408_5ecc4358 2>/dev/null | head -1)
[ -d "$PLANT_SRC" ] || die "plant source src_p20c_b5268408_5ecc4358 not found"
SRCDIR=$BASE/src_$OVL
if [ ! -d "$SRCDIR" ]; then
  cp -a "$PLANT_SRC" "$SRCDIR.tmp" || die "src copy"
  tar xf "$SHIP/overlay.tar" -C "$SRCDIR.tmp" --no-same-owner || die "overlay extract"
  mv "$SRCDIR.tmp" "$SRCDIR"
fi
ln -sfn "$SRCDIR" "$BASE/src"
MAPV=$BASE/src/pytools/assets/research/newmap_candidates/wz3to2v80
pin $MAPV/wz3to2v80.net.xml 1fed6f59cfcfa63d2e30f1344eac5e59a1597eca77c7d9edbae3e0bc08161f03
R=$BASE/root
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks" "$BASE/p20c_ref"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || cp "$B/$f" "$R/$f" || die "mirror $f"
  cmp -s "$B/$f" "$R/$f" || die "final root $f differs from the binding root"
done
cp -n "$SHIP"/p20c_ref/*.json "$BASE/p20c_ref/" || die "p20c identity refs"

# ---------------------------------------------------------------- 4. env, preflight, tests, dry
SHIM=/home/ubuntu/cpus_env/libmkl_intel_shim.so
cat > "$BASE/env_a40.sh" <<EOF
BASE=$BASE
SRC=$BASE/src
PY=$PY
export SUMO_HOME=\$(\$PY -c 'import sumo, os; print(os.path.dirname(sumo.__file__))')
export PATH="\$SUMO_HOME/bin:\$(dirname $PY):\$PATH"
export PYTHONPATH="\$SRC/pytools:\$SRC/runtime_src:\$SRC/WM_PACKAGE:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
[ -f $SHIM ] && export LD_PRELOAD=$SHIM
export MAPV=$MAPV/MAP_SPEC.json
export MAP3=/home/ubuntu/coopmwm20261106_clean4/inputs/map/wz3to2/MAP_SPEC.json
export COOPMWM_MAP_SPEC=\$MAPV COOPMWM_G5_RUNTIME=1
cd \$SRC
EOF
# shellcheck disable=SC1090
source "$BASE/env_a40.sh"
"$PY" -c "import ctypes" 2>&1 | grep -qi "cannot be preloaded" && { say "LD_PRELOAD shim does not load here; disabled"; sed -i '/LD_PRELOAD/d' "$BASE/env_a40.sh"; unset LD_PRELOAD; }
nice -n 19 "$PY" -m tools.coopmwm.nm_eval_runner setup-check --checkpoint /home/ubuntu/mappospeed/production2/policy_00169.pt \
  --sha256 d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca > "$BASE/setup_check.json" 2>>"$LOG" \
  || die "runner setup-check failed (see $BASE/setup_check.json)"
say "runner setup-check PASS (zero SUMO)"
A40_MIRROR=$R COOPMWM_MAP_SPEC=$MAP3 nice -n 19 "$PY" -m pytest -q -p no:cacheprovider pytools/tools/coopmwm/test_a40_geomvar_cpu.py \
  pytools/tools/coopmwm/test_a35_transfer_cpu.py pytools/tools/coopmwm/test_a32_arms_cpu.py ops/test_a40_geomvar.py \
  > "$BASE/tests.out" 2>&1 || { tail -30 "$BASE/tests.out"; die "CPU tests failed"; }
say "tests: $(tail -1 "$BASE/tests.out")"
for s in a40_cal_D2400 a40_cal_D2800 a40_cal_D3200 a40_D2400_p20 a40_D2800_p20 a40_D3200_p20; do
  "$PY" ops/a40_geomvar.py dry --final-root "$R" --setting $s > "$BASE/dry_$s.json" 2>>"$LOG" || die "dry $s"
done
grep -q '"learned_trigger_sha256": "41fc8d1a' "$BASE/dry_a40_D2800_p20.json" || die "LEARNED_TRIGGER digest is not 41fc8d1a"
grep -q '"a32_sha256": "1c51f22c' "$BASE/dry_a40_D2800_p20.json" || die "a32 digest is not 1c51f22c"
say "dry OK (LT 41fc8d1a, a32 1c51f22c)"
WORKERS=$(( NCPU - 4 )); MW=$(( MEM_GB * 10 / 15 )); [ "$MW" -lt "$WORKERS" ] && WORKERS=$MW
[ "$WORKERS" -ge 4 ] || die "too few workers ($WORKERS)"
echo "$WORKERS" > "$BASE/WORKERS"
date '+%F %T' > "$BASE/SETUP_OK"
say "SETUP_OK workers=$WORKERS overlay=$OVL"
