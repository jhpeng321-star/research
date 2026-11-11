#!/bin/bash
# AMENDMENT 60 host setup on Box 1 (zero SUMO): /root/a60 only. src = byte copy of the A56 src (/root/a56/src, the
# A42 overlay + A56 ops) + the A60 overlay (overlay.tar from c4/a60-wmvalue); final roots rootA (60A) and rootC (60C)
# mirror the binding root files exactly as A50/A56 (cmp-checked). Never writes /root/a42v, /root/a56, /root/a57, /root/a59.
set -uo pipefail
BASE=/root/a60
SHIP=$BASE/ship
say() { echo "$(date '+%F %T') $*"; }
die() { say "SETUP FAILED: $*"; echo "$*" > "$BASE/SETUP_FAILED"; exit 1; }
cd "$SHIP" && sha256sum -c --quiet SHA256SUMS || die "ship SHA256SUMS mismatch"
[ -d "$BASE/src" ] && die "$BASE/src exists (setup runs once)"
cp -a /root/a56/src "$BASE/src.tmp" || die "src copy"
tar xf "$SHIP/overlay.tar" -C "$BASE/src.tmp" --no-same-owner || die "overlay extract"
mv "$BASE/src.tmp" "$BASE/src"
B=/root/autodl-fs/nm_finaleval_p20c_bank
for R in "$BASE/rootA" "$BASE/rootC"; do
  mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks" "$R/a60_banks"
  for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
           oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
           a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
    [ -f "$R/$f" ] || cp "$B/$f" "$R/$f" || die "mirror $f"
    cmp -s "$B/$f" "$R/$f" || die "final root $f differs from the binding root"
  done
done
cat > "$BASE/env_a60.sh" <<EOF
BASE=$BASE
SRC=$BASE/src
PY=/root/a42v/venv/bin/python
export SUMO_HOME=\$(\$PY -c "import sumo, os; print(os.path.dirname(sumo.__file__))")
export PATH="\$SUMO_HOME/bin:/root/a42v/venv/bin:\$PATH"
export PYTHONPATH="\$SRC/pytools:\$SRC/runtime_src:\$SRC/WM_PACKAGE:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
[ -f /home/ubuntu/cpus_env/libmkl_intel_shim.so ] && export LD_PRELOAD=/home/ubuntu/cpus_env/libmkl_intel_shim.so
export A42_HOST_BASE=/root/a42v A60_HOST_BASE=$BASE COOPMWM_MAP_SPEC=/home/ubuntu/coopmwm20261106_clean4/inputs/map/wz3to2/MAP_SPEC.json COOPMWM_G5_RUNTIME=1
cd \$SRC
EOF
cat "$SHIP/BASE_CODE" > "$BASE/BASE_CODE"
say "setup OK"
touch "$BASE/SETUP_OK"
