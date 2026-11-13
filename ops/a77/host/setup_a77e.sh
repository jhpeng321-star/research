#!/bin/bash
# AMENDMENT 77 evaluation host setup (run as root from the ship dir; idempotent; never deletes; ZERO SUMO).
# Needs a finished A66 stack (/root/a66/SETUP_OK: plant, bindings, refs incl. the A50 identity wave, python env); on a
# brand-new box run the A66 ship's setup_a66.sh first (it needs /opt/py312, which setup_a77_train.sh builds).
# /root/a77e/src = HARD-LINK clone (cp -al) of the A66 tree + the A77 files as new files; the runner is REPLACED by a
# patched copy (never edited in place: /root/a66/src keeps its bytes, pinned below; 6-line table edit checked).
# /root/a77e/root = copies of the p20c binding-root files. Binds IDENT_U0 (+ MAPPO_CT when its file is shipped).
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a77e
FROM=/root/a66
RUNNER_A66=838ccf86fe3ccab03bc9fc4d59db22312771c40f8079ff71a89bf52a5d0a4862
U0=/home/ubuntu/mappospeed/production2/policy_00169.pt
die() { echo "SETUP FAILED: $*"; mkdir -p "$BASE"; echo "$*" > "$BASE/SETUP_FAILED"; exit 1; }
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
[ "$(id -u)" = 0 ] || die "must run as root"
(cd "$SHIP" && sha256sum -c --quiet SHA256SUMS) || die "ship SHA256SUMS mismatch"
[ -f $FROM/SETUP_OK ] || die "$FROM is not a completed A66 setup (run the A66 ship setup_a66.sh first)"
SRC66=$(readlink -f $FROM/src)
mkdir -p "$BASE"
S=$BASE/src
if [ ! -d "$S" ]; then cp -al "$SRC66" "$S" || die "link tree"; fi
[ -L "$S" ] && die "$S is a symlink"
source $FROM/env_a66.sh
PY=$FROM/venv/bin/python
T=$S/pytools/tools/coopmwm
if ! grep -q "^A77_RANGES" "$T/nm_eval_runner.py"; then
  pin "$T/nm_eval_runner.py" $RUNNER_A66
  cp "$T/nm_eval_runner.py" "$BASE/nm_eval_runner.py.new" || die "runner copy"
  "$PY" "$SHIP/patch_tables.py" tables "$BASE/nm_eval_runner.py.new" || die "runner patch"
  mv "$T/nm_eval_runner.py" "$T/nm_eval_runner.py.a66" && mv "$BASE/nm_eval_runner.py.new" "$T/nm_eval_runner.py" || die "runner"
fi
pin "$SRC66/pytools/tools/coopmwm/nm_eval_runner.py" $RUNNER_A66
N=$(diff "$SRC66/pytools/tools/coopmwm/nm_eval_runner.py" "$T/nm_eval_runner.py" | grep -c "^[<>]")
[ "$N" = 6 ] || die "runner diff is $N lines, not the 6-line A77 table edit"
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
mkdir -p "$S/ops/a77/host"
put "$SHIP/a77_mappo20c.py" "$S/ops/a77_mappo20c.py"
put "$SHIP/test_a77_mappo20c.py" "$S/ops/test_a77_mappo20c.py"
for f in nm_eval_panel_wz3to2_a77c_D2800_p20.json SEED_AUDIT.json; do put "$SHIP/$f" "$S/ops/a77/$f"; done
put "$SHIP/patch_tables.py" "$S/ops/a77/host/patch_tables.py"
B=/root/autodl-fs/nm_finaleval_p20c_bank
R=$BASE/root
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks" "$BASE/inputs"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || cp "$B/$f" "$R/$f" || die "mirror $f"
  cmp -s "$B/$f" "$R/$f" || die "$R/$f differs from the binding root"
done
pin "$U0" d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca
[ -f "$FROM/refs/a50/root/WAVE_A50_IDENT_U0__0000-0003.json" ] || die "A50 identity wave missing in $FROM/refs/a50/root"
sed -e "s#^BASE=/root/a66#BASE=$BASE#" -e "s#^SRC=/root/a66/src#SRC=$S#" $FROM/env_a66.sh > "$BASE/env_a77e.sh.new"
echo "export A77_HOST_BASE=$BASE" >> "$BASE/env_a77e.sh.new"
if [ -f "$BASE/env_a77e.sh" ]; then cmp -s "$BASE/env_a77e.sh" "$BASE/env_a77e.sh.new" || die "env differs"; rm "$BASE/env_a77e.sh.new"
else mv "$BASE/env_a77e.sh.new" "$BASE/env_a77e.sh"; fi
put "$SHIP/run_a77e.sh" "$BASE/run_a77e.sh"; chmod +x "$BASE/run_a77e.sh"
source "$BASE/env_a77e.sh"
D=ops/a77_mappo20c.py
[ -f "$R/BOUND_A77_IDENT_U0.json" ] || nice -n 19 "$PY" $D bind --which IDENT_U0 --policy "$U0" --final-root "$R" || die "bind IDENT_U0"
if [ -f "$SHIP/mappo_ct_policy_00012.pt" ]; then
  put "$SHIP/mappo_ct_policy_00012.pt" "$BASE/inputs/mappo_ct_policy_00012.pt"
  pin "$BASE/inputs/mappo_ct_policy_00012.pt" 069a4f85ab4ac7bdc5e01404dbe903f9b21957d5e1478c34d0929e5dcb2020c3
  [ -f "$R/BOUND_A77_MAPPO_CT.json" ] || nice -n 19 "$PY" $D bind --which MAPPO_CT --policy "$BASE/inputs/mappo_ct_policy_00012.pt" \
    --final-root "$R" || die "bind MAPPO_CT"
else echo "MAPPO_CT file not shipped: only the registered fallback --arms U0,MAPPO20C can run"; fi
nice -n 19 "$PY" $D dry --final-root "$R" > "$BASE/dry.json" 2>&1 || { cat "$BASE/dry.json"; die "dry failed"; }
nice -n 19 "$PY" -m tools.coopmwm.nm_eval_runner setup-check --checkpoint "$U0" \
  --sha256 d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca > "$BASE/setup_check.json" 2>&1 \
  || die "runner setup-check failed (zero SUMO)"
A37_MIRROR=$R A77_A50_REFS=$FROM/refs/a50/root nice -n 19 "$PY" -m pytest -q -p no:cacheprovider ops/test_a77_mappo20c.py \
  ops/test_a42_bigp60.py --basetemp=/dev/shm/a77e_test_$$ > "$BASE/tests.out" 2>&1 || { tail -30 "$BASE/tests.out"; die "CPU tests failed"; }
echo "tests: $(tail -1 "$BASE/tests.out")"
echo "tree diff vs the A66 tree:"; diff -rq "$SRC66" "$S" -x __pycache__ | cut -c1-200
echo "runner $(sha "$T/nm_eval_runner.py")  driver $(sha "$S/ops/a77_mappo20c.py")"
du -sh "$BASE" | cut -c1-80
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
