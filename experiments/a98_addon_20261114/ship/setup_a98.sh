#!/bin/bash
# AMENDMENT 98 (A98) host setup (root; zero SUMO; never deletes; writes only /root/a98). Source tree = REAL copy of the
# A94 tree /root/a94/src (hash-checked against the A94 ship; /root/a94 is only read). New files: ops/a98_addon.py,
# ops/test_a98_addon.py, ops/a98_e2e.py, ops/a98/A94_BANK_SEARCHED_D2800_p60.json (the two-search bank REBUILT here with
# the unchanged build_bank_a94.py, sha must be 40b16ed9). Root = byte copies of the A94 root binding files + the A94
# panel file / pin (panel sha d85999dd checked). No A94 outcome file is read.
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a98
FROM=/root/a94
S=$BASE/src
R=$BASE/root
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
[ -f "$BASE/SETUP_OK" ] && die "$BASE/SETUP_OK exists (setup runs once)"
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
( cd "$SHIP" && sha256sum -c --quiet SHIP_FILES.sha256 ) || die "shipped files differ from SHIP_FILES.sha256"
pin $FROM/src/pytools/tools/coopmwm/nm_eval_runner.py 47fb9f76117c3862ac0f173a64891599f84d6b9a16c8b5a04ed8a33ebd2397f5
pin $FROM/src/ops/a94_bank.py c8b05145822f8250a626036a9dddf5214b550542c8c775c847a7f693a66338ea
pin $FROM/src/ops/a94/build_bank_a94.py c8746b4a467cc6c7e3fb01d27978ff0c90d6a35a3c6cc7f0c13ca4aa166ac91b
pin $FROM/src/ops/a81_bigconfirm.py c755a5510bc7c1c5f9f9279e7b2f40a77c388b5ccd60a35db2207b7d2dc4c4bb
pin $FROM/root/A94_PANEL_C.json 720d19e1df5d95c1b769170b62e234c1498c1c616fafa209c1880b5a474095ac
grep -q "panel_sha256 d85999ddaaf3eb583775d66dbabca168b4432966bc01e7193dfc79f244f7946c" $FROM/root/A94_PANEL_C.sha256 || die "A94 panel pin"
pin /root/a85adm/root/a80_banks/A80_BANK_D2800_p60.json a428cd1d320196578d55c84eea4b9b391f5e5c03928692a90dff0ac96667d7a8
pin /root/a86adm/root/a80_banks/A80_BANK_D2800_p60.json 577a0564944b2062e441d7ee603e7fe8090bede52a5dd6e8774d41ed9cb2d8b6
PY=$(grep "^PY=" $FROM/env_a94.sh | cut -d= -f2)
[ -x "$PY" ] || die "python of the A94 env not found"
mkdir -p "$BASE"
if [ ! -d "$S" ]; then
  mkdir -p "$S" && ( cd $FROM/src && tar --exclude=__pycache__ --exclude='*.twosearch_*' --exclude='*.v0e6c35b4_*' \
      --exclude='*.prev_*' -cf - . ) | ( cd "$S" && tar -xpf - ) || die "copy tree"
fi
diff -rq --no-dereference -x __pycache__ -x '*.twosearch_*' -x '*.v0e6c35b4_*' -x '*.prev_*' $FROM/src "$S" > "$BASE/tree_diff_before.txt"
[ -s "$BASE/tree_diff_before.txt" ] && die "copied tree differs"
mkdir -p "$S/ops/a98"
put "$SHIP/a98_addon.py" "$S/ops/a98_addon.py"
put "$SHIP/test_a98_addon.py" "$S/ops/test_a98_addon.py"
put "$SHIP/a98_e2e.py" "$S/ops/a98_e2e.py"
B2=$S/ops/a98/A94_BANK_SEARCHED_D2800_p60.json
if [ ! -f "$B2" ]; then
  ( source $FROM/env_a94.sh >/dev/null 2>&1; "$PY" -I -c "import sys; sys.path[:0]=['$S/pytools','$S/ops','$S']; sys.argv=sys.argv[1:]; exec(open('$S/ops/a94/build_bank_a94.py').read())" \
      b searched --s5-bank /root/a86adm/root/a80_banks/A80_BANK_D2800_p60.json --s4-bank /root/a85adm/root/a80_banks/A80_BANK_D2800_p60.json \
      --out-dir "$S/ops/a98" ) > "$BASE/build_s4s5.out" 2>&1 || { cat "$BASE/build_s4s5.out"; die "two-search bank build"; }
fi
pin "$B2" 40b16ed95add6fcd63385dabcbf7e8a03fe8dbf20d0418e58e5aa8feee30ee04
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json A94_PANEL_C.json A94_PANEL_C.sha256; do
  [ -f "$R/$f" ] || cp "$FROM/root/$f" "$R/$f" || die "copy $f"
  cmp -s "$FROM/root/$f" "$R/$f" || die "$R/$f differs from the A94 root"
done
sed -e "s#^BASE=/root/a94#BASE=$BASE#" -e "s#^SRC=/root/a94/src#SRC=$S#" -e "s#A94_HOST_BASE=/root/a94#A98_HOST_BASE=$BASE#" \
    $FROM/env_a94.sh > "$BASE/env_a98.sh.new"
grep -q "^SRC=$S$" "$BASE/env_a98.sh.new" && grep -q "A98_HOST_BASE=$BASE" "$BASE/env_a98.sh.new" || die "env switch"
grep -q "/root/a94" "$BASE/env_a98.sh.new" && die "env: an a94 path remains"
if [ -f "$BASE/env_a98.sh" ]; then cmp -s "$BASE/env_a98.sh" "$BASE/env_a98.sh.new" || die "env differs"; else cp "$BASE/env_a98.sh.new" "$BASE/env_a98.sh"; fi
put "$SHIP/run_a98.sh" "$BASE/run_a98.sh"; chmod +x "$BASE/run_a98.sh"
( source "$BASE/env_a98.sh" && cd "$S" && "$PY" - <<'EOF'
import sys
sys.path[:0] = ['ops', '.']
from tools.coopmwm import a75_holdforms as H, nm_eval_runner as R
import a98_addon as M, a75_d2forms as D2
assert H.a75_digest() == D2.A75_SHA256 and tuple(R.A94_RANGES) == (M.BLOCK,)
M.point(); M.A94._runner_check()
print('code identity ok: a75', H.a75_digest()[:8], 'arms', M.ARMS, 'caps', M.MECH_CAP, M.CONFIRM_CAP, M.CAP_SIM_S)
EOF
) || die "code identity check"
diff -rq -x __pycache__ -x '*.twosearch_*' -x '*.v0e6c35b4_*' -x '*.prev_*' $FROM/src "$S" | cut -c1-200
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
