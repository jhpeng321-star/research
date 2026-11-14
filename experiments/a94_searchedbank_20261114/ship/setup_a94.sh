#!/bin/bash
# AMENDMENTS 94 + 95 (A94) host setup (root; zero SUMO; never deletes; writes only /root/a94). Derived from setup_a91.sh
# (e418d9d3). Source tree = REAL copy (cp -a, no hard links) of the finished A91 tree /root/a91/src (diff-verified).
# ONE changed file: pytools/tools/coopmwm/nm_eval_runner.py (A91 91e40a94 + A94_RANGES, patch_tables_a94.py).
# New files under ops/: a94_bank.py, test_a94_bank.py, a94_analyse_e2e.py, a94_mech_e2e.py, a94/build_bank_a94.py,
# a94/patch_tables_a94.py, a94/SEED_AUDIT_A94.json. The bank files are NOT built here (run_a94.sh bank, after the S4
# admission). Root binding files = byte copies of the A91 root binding files (checked equal to the A66 root).
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a94
FROM=/root/a91
S=$BASE/src
R=$BASE/root
RUN_A91_SHA=91e40a948c874ad057d64c751cd585362c816811694dc8a578f76032cc613700
DRV_A81_SHA=c755a5510bc7c1c5f9f9279e7b2f40a77c388b5ccd60a35db2207b7d2dc4c4bb
A80_SHA=48c5c448c2a85d4e2020def54cb17259979c89f2e0cc341829d6e99de0d0e289
BANK_SHA=c325ec1da034d0ad22d242f97e926c30cfa998981f16cef935b1b6498565c30c
S5_BANK_SHA=577a0564944b2062e441d7ee603e7fe8090bede52a5dd6e8774d41ed9cb2d8b6
S5_ADM_SHA=78b25e4557e73720da881610b6fad7eeacd6d94cfacf957a1044b09284b22e33
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
[ -f "$BASE/SETUP_OK" ] && die "$BASE/SETUP_OK exists (setup runs once)"
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
( cd "$SHIP" && sha256sum -c --quiet SHIP_FILES.sha256 ) || die "shipped files differ from SHIP_FILES.sha256"
[ -f $FROM/SETUP_OK ] || die "$FROM is not a completed A91 setup"
grep -q A91_CONFIRM_RUN_DONE $FROM/chain_confirm.out || die "A91 confirm is not done"
RT=pytools/tools/coopmwm/nm_eval_runner.py
pin "$FROM/src/$RT" $RUN_A91_SHA
pin "$FROM/src/ops/a81_bigconfirm.py" $DRV_A81_SHA
pin "$FROM/src/ops/a80_s2b.py" $A80_SHA
pin /root/a75d2c/root/a75d2_banks/A75D2_BANK_D2800_p60.json $BANK_SHA
pin /root/a86adm/root/a80_banks/A80_BANK_D2800_p60.json $S5_BANK_SHA
pin /root/a86adm/root/A80_ADMISSION_D2800_p60.json $S5_ADM_SHA
PY=$(grep "^PY=" $FROM/env_a91.sh | cut -d= -f2)
[ -x "$PY" ] || die "python of the A91 env not found"
mkdir -p "$BASE"
if [ ! -d "$S" ]; then cp -a $FROM/src "$S" || die "copy tree"; fi
[ -L "$S" ] && die "$S is a symlink"
diff -rq --no-dereference $FROM/src "$S" -x __pycache__ > "$BASE/tree_diff_before.txt"; [ -s "$BASE/tree_diff_before.txt" ] && die "copied tree differs before patching"
mkdir -p "$S/ops/a94"
put "$SHIP/a94_bank.py" "$S/ops/a94_bank.py"
put "$SHIP/test_a94_bank.py" "$S/ops/test_a94_bank.py"
put "$SHIP/a94_analyse_e2e.py" "$S/ops/a94_analyse_e2e.py"
put "$SHIP/a94_mech_e2e.py" "$S/ops/a94_mech_e2e.py"
put "$SHIP/build_bank_a94.py" "$S/ops/a94/build_bank_a94.py"
put "$SHIP/patch_tables_a94.py" "$S/ops/a94/patch_tables_a94.py"
put "$SHIP/SEED_AUDIT_A94.json" "$S/ops/a94/SEED_AUDIT_A94.json"
if ! grep -q "^A94_RANGES = ((988700, 988899),)$" "$S/$RT"; then
  [ "$(sha "$S/$RT")" = $RUN_A91_SHA ] || die "$S/$RT is neither the A91 runner nor the A94 runner"
  "$PY" -I "$SHIP/patch_tables_a94.py" tables "$S/$RT" || die "A94 table patch"
  [ "$(diff "$FROM/src/$RT" "$S/$RT" | grep -c '^<')" = 1 ] || die "the runner patch changed more than the tail line"
fi
pin "$FROM/src/$RT" $RUN_A91_SHA
[ "$(stat -c %i "$S/$RT")" != "$(stat -c %i "$FROM/src/$RT")" ] || die "the A94 runner shares the A91 inode"
echo "runner A91 ${RUN_A91_SHA:0:8} (unchanged) -> A94 $(sha "$S/$RT")"
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || cp "$FROM/root/$f" "$R/$f" || die "copy $f"
  cmp -s "/root/a66/root/$f" "$R/$f" || die "$R/$f differs from the A66 root"
  cmp -s "$FROM/root/$f" "$R/$f" || die "$R/$f differs from the A91 root"
done
pin /root/a66/inputs/policy_00084.pt 2abed31707b61121569af2936c23689827b52b28422df162751b230aa6a93de8
[ -d /root/a75/a70ref/root ] && [ -d /root/a75/ref70/evaluation ] || die "A70 references missing"
sed -e "s#^BASE=/root/a91#BASE=$BASE#" -e "s#^SRC=/root/a91/src#SRC=$S#" -e "/A91_HOST_BASE/d" \
    $FROM/env_a91.sh > "$BASE/env_a94.sh.new"
echo "export A94_HOST_BASE=$BASE" >> "$BASE/env_a94.sh.new"
grep -q "^SRC=$S$" "$BASE/env_a94.sh.new" || die "env: SRC not switched"
grep -q "^export A75D2_HOST_BASE=/root/a75d2c$" "$BASE/env_a94.sh.new" || die "env: A75D2_HOST_BASE is not /root/a75d2c"
grep -q "a91" "$BASE/env_a94.sh.new" && die "env: an a91 path remains"
if [ -f "$BASE/env_a94.sh" ]; then cmp -s "$BASE/env_a94.sh" "$BASE/env_a94.sh.new" || die "env differs"; else cp "$BASE/env_a94.sh.new" "$BASE/env_a94.sh"; fi
put "$SHIP/run_a94.sh" "$BASE/run_a94.sh"; chmod +x "$BASE/run_a94.sh"
( source "$BASE/env_a94.sh" && cd "$S" && A94_HAND_SHA256=none "$PY" - <<'EOF'
import sys
sys.path[:0] = ['ops', '.']
from tools.coopmwm import a75_holdforms as H, nm_eval_runner as R
import a75_d2forms as D2
import a81_bigconfirm as B
import a94_bank as M
import a80_s2b as A80
import nm_finaleval as N
assert H.a75_digest() == D2.A75_SHA256, ('a75 digest', H.a75_digest())
assert R.A75_ARMS == ('A75_PLAN', 'A75_BANK') and tuple(R.A70_RANGES) == ((996300, 996399),)
assert tuple(R.A81_RANGES) == ((990100, 990499),) and tuple(R.A89_RANGES) == ((988400, 988499),)
assert tuple(R.A91_RANGES) == ((988600, 988699),) and tuple(R.A94_RANGES) == (M.BLOCK,)
M.point()
assert B._runner_check()
assert 'a70c_D2800_p60' in N.SETTINGS and 'a94c_D2800_p60' not in N.SETTINGS
print('code identity ok: a75', H.a75_digest()[:8], 'A94 block', M.BLOCK, 'seeds', M.SEEDS_C[0], M.SEEDS_C[-1])
EOF
) || die "code identity check"
echo "tree diff vs the A91 tree (expected: runner + the A94 files):"
diff -rq $FROM/src "$S" -x __pycache__ | cut -c1-200
du -sh "$BASE" | cut -c1-80
df -BG --output=avail "$BASE" | tail -1
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
