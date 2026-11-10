#!/bin/bash
# A38 box bootstrap (48-core Linux, run as root): unpack the bundle under BASE, mirror the frozen paths the plant
# pins (/home/ubuntu/{e3final,mappospeed,cpus_env,coopmwm20261106_clean4,coopmwm20261104_launchcoop_src7},
# /root/autodl-fs/nm_finaleval_p20c_{stage,bank}) as symlinks into BASE (never overwrites an existing real path),
# overlay the c4/a38-road2 files onto the frozen source, write env_a38.sh. Python: miniconda3 py3.10-3.12 with
# torch (CPU) + numpy + eclipse-sumo==1.27.1 (SUMO_HOME from the wheel). Zero physics.
set -euo pipefail
BASE=${BASE:-/root/a38}
B=$BASE/bundle
mkdir -p "$B" && cd "$B"
[ -f a38_bundle.tar ] || { echo "copy a38_bundle.tar to $B first"; exit 1; }
tar xf a38_bundle.tar
sha256sum -c --quiet MANIFEST.sha256
mkdir -p "$BASE/stage" && tar xf stage.tar -C "$BASE/stage" && tar xf src7.tar -C "$BASE/stage"
link() { if [ -e "$2" ] && [ ! -L "$2" ]; then echo "exists, kept: $2"; else mkdir -p "$(dirname "$2")"; ln -sfn "$1" "$2"; fi; }
H=$BASE/stage/root/mining20261106/cpuv_home
for d in e3final mappospeed cpus_env coopmwm20261106_clean4; do link "$H/$d" "/home/ubuntu/$d"; done
link "$BASE/stage/coopmwm20261104_launchcoop_src7" /home/ubuntu/coopmwm20261104_launchcoop_src7
link "$BASE/stage/root/autodl-fs/nm_finaleval_p20c_stage" /root/autodl-fs/nm_finaleval_p20c_stage
mkdir -p /autodl-fs && link "$BASE/stage/autodl-fs/data" /autodl-fs/data
SRC=$BASE/src
rm -rf "$SRC" && cp -a "$BASE/stage/root/autodl-fs/nm_finaleval_p20c_stage/src_p20c_b5268408_5ecc4358" "$SRC"
# repo layout -> frozen stage layout: ops/* -> $SRC/ops, everything else -> $SRC/pytools
( cd "$B/overlay/repo" && find . -type f | while read -r f; do
    case "$f" in ./ops/*) dst="$SRC/$f";; *) dst="$SRC/pytools/$f";; esac
    mkdir -p "$(dirname "$dst")"; cp -f "$f" "$dst"
    [ "$(sha256sum < "$f" | cut -c1-64)" = "$(sha256sum < "$dst" | cut -c1-64)" ] || { echo "overlay copy mismatch $f"; exit 1; }
  done )
# fresh mirror root: the p20c bank-root bindings + prerun plant, own empty ledger (as a35)
ROOT=$BASE/root
mkdir -p "$ROOT"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json; do cp -n "$BASE/stage/root/autodl-fs/nm_finaleval_p20c_bank/$f" "$ROOT/"; done
cp -an "$BASE/stage/root/autodl-fs/nm_finaleval_p20c_bank/prerun_plant" "$ROOT/" 2>/dev/null || true
cp -an "$BASE/stage/root/autodl-fs/nm_finaleval_p20c_bank/oracle_banks" "$ROOT/" 2>/dev/null || true
cp -an "$BASE/stage/root/autodl-fs/nm_finaleval_p20c_bank/a32_banks" "$ROOT/" 2>/dev/null || true
link "$ROOT" /root/autodl-fs/nm_finaleval_p20c_bank
PY=${PY:-$(command -v python3)}
SUMO_HOME=$($PY -c 'import sumo,os;print(os.path.dirname(sumo.__file__))')
cat > "$BASE/env_a38.sh" <<ENV
export BASE=$BASE SRC=$SRC ROOT=$ROOT PY=$PY
export SUMO_HOME=$SUMO_HOME
export PATH="\$SUMO_HOME/bin:\$PATH"
export PYTHONPATH="\$SRC/pytools:\$SRC/runtime_src:\$SRC/WM_PACKAGE:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
[ -f /home/ubuntu/cpus_env/libmkl_intel_shim.so ] && export LD_PRELOAD=/home/ubuntu/cpus_env/libmkl_intel_shim.so
export MAP2=\$SRC/pytools/assets/research/newmap_candidates/wz2to1/MAP_SPEC.json
export MAP3=/home/ubuntu/coopmwm20261106_clean4/inputs/map/wz3to2/MAP_SPEC.json
export COOPMWM_G5_RUNTIME=1
cd \$SRC
ENV
$PY -c 'import torch, numpy, traci; print("py ok", torch.__version__, numpy.__version__)'
"$SUMO_HOME/bin/sumo" --version | head -1
echo "BOOTSTRAP_OK $(date '+%F %T')" | tee "$BASE/BOOTSTRAP_OK"
