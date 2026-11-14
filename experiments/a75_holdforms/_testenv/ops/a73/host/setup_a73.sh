#!/bin/bash
# AMENDMENT 73 host setup (Box 4; run as root; zero SUMO; never deletes; writes only /root/a73; disk-frugal).
# /root/a73/src = HARD-LINK clone (cp -al, no data blocks) of /root/a69/src + the A73 files as new small files; the
# runner is REPLACED by a patched copy (never edited in place: /root/a69/src keeps its own bytes, checked below).
# /root/a73/root = the same binding files as the A69 root (hard links, compared with the binding root).
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a73
FROM=/root/a69
S=$BASE/src
RUNNER_A69=62496e864567700a051aaf3d26ee16fb64e1dbe5b0a34b7d28688e0defb67836
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
[ -L $FROM/src ] && die "$FROM/src is a symlink"
[ -f $FROM/SETUP_OK ] || die "$FROM is not a completed A69 setup"
mkdir -p "$BASE"
if [ ! -d "$S" ]; then cp -al $FROM/src "$S" || die "link tree"; fi
[ -L "$S" ] && die "$S is a symlink"
[ "$(readlink -f $S)" = "$S" ] || die "$S does not resolve to itself"
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
PY=$(grep "^PY=" $FROM/env_a69.sh | cut -d= -f2)
[ -x "$PY" ] || die "python of the A69 env not found"
T=$S/pytools/tools/coopmwm
mkdir -p "$S/ops/a73"
if ! grep -q "^A73_RANGES" "$T/nm_eval_runner.py"; then
  pin "$T/nm_eval_runner.py" $RUNNER_A69
  [ -e "$T/nm_eval_runner.py.a69" ] && die ".a69 runner backup already exists"
  cp "$T/nm_eval_runner.py" "$BASE/nm_eval_runner.py.new" || die "runner copy"
  "$PY" "$SHIP/patch_tables.py" tables "$BASE/nm_eval_runner.py.new" || die "runner patch"
  mv "$T/nm_eval_runner.py" "$T/nm_eval_runner.py.a69" && mv "$BASE/nm_eval_runner.py.new" "$T/nm_eval_runner.py" || die "runner"
fi
pin $FROM/src/pytools/tools/coopmwm/nm_eval_runner.py $RUNNER_A69
N=$(diff $FROM/src/pytools/tools/coopmwm/nm_eval_runner.py "$T/nm_eval_runner.py" | grep -c "^[<>]")
[ "$N" = 7 ] || die "runner diff is $N lines, not the 7-line table edit"
pin "$T/a44_lcvariant.py" 4e7a97f3183290ca5f1ad161cb3557b27d9b3695eb596a18f3ee3097e92aac0c
pin "$S/ops/a44_lcmech.py" 7a639ed279251c920dc922ef29e2efa0a0dc64bef11238b93eaacb3f10529b60
put "$SHIP/a73_lcside20.py" "$S/ops/a73_lcside20.py"
put "$SHIP/test_a73_lcside20.py" "$S/ops/test_a73_lcside20.py"
for f in patch_tables.py CANDIDATES_LC_D2800_p20.json nm_eval_panel_wz3to2_a73m_D2800_p20.json \
         nm_eval_panel_wz3to2_a73v_D2800_p20.json nm_eval_panel_wz3to2_a73c_D2800_p20.json; do put "$SHIP/$f" "$S/ops/a73/$f"; done
if [ -f "$SHIP/PINS_A73.json" ]; then put "$SHIP/PINS_A73.json" "$S/ops/a73/PINS_A73.json"; fi
B=/root/autodl-fs/nm_finaleval_p20c_bank
R=$BASE/root
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || ln "$FROM/root/$f" "$R/$f" || die "mirror $f"
  cmp -s "$B/$f" "$R/$f" || die "$R/$f differs from the binding root"
done
sed -e "s#^BASE=/root/a69#BASE=/root/a73#" -e "s#^SRC=/root/a69/src#SRC=/root/a73/src#" $FROM/env_a69.sh > "$BASE/env_a73.sh.new"
echo "export A73_HOST_BASE=/root/a73" >> "$BASE/env_a73.sh.new"
if [ -f "$BASE/env_a73.sh" ]; then cmp -s "$BASE/env_a73.sh" "$BASE/env_a73.sh.new" || die "env differs"; rm "$BASE/env_a73.sh.new"; else mv "$BASE/env_a73.sh.new" "$BASE/env_a73.sh"; fi
put "$SHIP/run_a73.sh" "$BASE/run_a73.sh"; chmod +x "$BASE/run_a73.sh"
echo "tree diff vs the A69 tree:"
diff -rq $FROM/src "$S" -x __pycache__ | cut -c1-200
echo "runner $(sha "$T/nm_eval_runner.py")  driver $(sha "$S/ops/a73_lcside20.py")"
du -sh "$BASE" | cut -c1-80
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
