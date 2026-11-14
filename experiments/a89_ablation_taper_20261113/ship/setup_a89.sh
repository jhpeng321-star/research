#!/bin/bash
# AMENDMENT 89 (A89) host setup (root; zero SUMO; never deletes; writes only /root/a89). Derived from setup_a81.sh.
# Source tree = REAL copy (cp -a, no hard links) of the finished A81 tree /root/a81big/src (all files cmp-verified).
# ONE changed file: pytools/tools/coopmwm/nm_eval_runner.py (A81 538bd0bf + A89_RANGES, patch_tables_a89.py).
# New files under ops/: a89_sixarm.py, test_a89_sixarm.py, a80_s2b.py (= registered A80 driver 48c5c448, byte copy
# from /root/a78s2b/src/ops), a89/build_a89_arms.py, a89/patch_tables_a89.py, a89/SEED_AUDIT_A89.json,
# a89/A89_ARMS_D2800_p60.json (built here, zero SUMO, from c325ec1d + the E1R candidate bank 057c1f06).
# Root binding files = byte copies of the A81 root binding files (checked equal to the A66 root).
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a89
FROM=/root/a81big
S=$BASE/src
R=$BASE/root
RUN_A81_SHA=538bd0bf88e595e6c42452cbfa98193b7a3c642006e448899617862c3f9b8609
DRV_A81_SHA=c755a5510bc7c1c5f9f9279e7b2f40a77c388b5ccd60a35db2207b7d2dc4c4bb
A80_SHA=48c5c448c2a85d4e2020def54cb17259979c89f2e0cc341829d6e99de0d0e289
BANK_SHA=c325ec1da034d0ad22d242f97e926c30cfa998981f16cef935b1b6498565c30c
E1R_CANDS=/root/a80e1r/root/a80_banks/A80_CANDS_D2800_p60.json
E1R_CANDS_SHA=057c1f066621ad1d7bf3da3f4da2529c39e530848b81a4bc3befeecc1777833a
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
[ -f "$BASE/SETUP_OK" ] && die "$BASE/SETUP_OK exists (setup runs once)"
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
( cd "$SHIP" && sha256sum -c --quiet SHIP_FILES.sha256 ) || die "shipped files differ from SHIP_FILES.sha256"
[ -f $FROM/SETUP_OK ] || die "$FROM is not a completed A81 setup"
grep -q A81_CONFIRM_DONE $FROM/chain_confirm.out || die "A81 confirm is not done"
RT=pytools/tools/coopmwm/nm_eval_runner.py
pin "$FROM/src/$RT" $RUN_A81_SHA
pin "$FROM/src/ops/a81_bigconfirm.py" $DRV_A81_SHA
pin "$FROM/root/a75d2_banks/A75D2_BANK_D2800_p60.json" $BANK_SHA
pin /root/a78s2b/src/ops/a80_s2b.py $A80_SHA
pin "$E1R_CANDS" $E1R_CANDS_SHA
PY=$(grep "^PY=" $FROM/env_a81.sh | cut -d= -f2)
[ -x "$PY" ] || die "python of the A81 env not found"
mkdir -p "$BASE"
if [ ! -d "$S" ]; then cp -a $FROM/src "$S" || die "copy tree"; fi
[ -L "$S" ] && die "$S is a symlink"
diff -rq --no-dereference $FROM/src "$S" -x __pycache__ > "$BASE/tree_diff_before.txt"; [ -s "$BASE/tree_diff_before.txt" ] && die "copied tree differs before patching"
[ "$(find "$S" -type f -links +1 | wc -l)" = 0 ] || echo "note: files with >1 link inside $S (inherited from $FROM hard-link clone; the copy has its own inodes)"
mkdir -p "$S/ops/a89"
put "$SHIP/a89_sixarm.py" "$S/ops/a89_sixarm.py"
put "$SHIP/test_a89_sixarm.py" "$S/ops/test_a89_sixarm.py"
put /root/a78s2b/src/ops/a80_s2b.py "$S/ops/a80_s2b.py"
put "$SHIP/build_a89_arms.py" "$S/ops/a89/build_a89_arms.py"
put "$SHIP/patch_tables_a89.py" "$S/ops/a89/patch_tables_a89.py"
put "$SHIP/SEED_AUDIT_A89.json" "$S/ops/a89/SEED_AUDIT_A89.json"
if ! grep -q "^A89_RANGES = ((988400, 988499),)$" "$S/$RT"; then
  [ "$(sha "$S/$RT")" = $RUN_A81_SHA ] || die "$S/$RT is neither the A81 runner nor the A89 runner"
  "$PY" -I "$SHIP/patch_tables_a89.py" tables "$S/$RT" || die "A89 table patch"
  [ "$(diff "$FROM/src/$RT" "$S/$RT" | grep -c '^<')" = 1 ] || die "the runner patch changed more than the tail line"
fi
pin "$FROM/src/$RT" $RUN_A81_SHA
[ "$(stat -c %i "$S/$RT")" != "$(stat -c %i "$FROM/src/$RT")" ] || die "the A89 runner shares the A81 inode"
echo "runner A81 ${RUN_A81_SHA:0:8} (unchanged) -> A89 $(sha "$S/$RT")"
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || cp "$FROM/root/$f" "$R/$f" || die "copy $f"
  cmp -s "/root/a66/root/$f" "$R/$f" || die "$R/$f differs from the A66 root"
  cmp -s "$FROM/root/$f" "$R/$f" || die "$R/$f differs from the A81 root"
done
pin /root/a66/inputs/policy_00084.pt 2abed31707b61121569af2936c23689827b52b28422df162751b230aa6a93de8
[ -d /root/a75/a70ref/root ] && [ -d /root/a75/ref70/evaluation ] || die "A70 references missing"
sed -e "s#^BASE=/root/a81big#BASE=$BASE#" -e "s#^SRC=/root/a81big/src#SRC=$S#" -e "/A81_HOST_BASE/d" \
    $FROM/env_a81.sh > "$BASE/env_a89.sh.new"
echo "export A89_HOST_BASE=$BASE" >> "$BASE/env_a89.sh.new"
grep -q "^SRC=$S$" "$BASE/env_a89.sh.new" || die "env: SRC not switched"
grep -q "^export A75D2_HOST_BASE=/root/a75d2c$" "$BASE/env_a89.sh.new" || die "env: A75D2_HOST_BASE is not /root/a75d2c"
if [ -f "$BASE/env_a89.sh" ]; then cmp -s "$BASE/env_a89.sh" "$BASE/env_a89.sh.new" || die "env differs"; else cp "$BASE/env_a89.sh.new" "$BASE/env_a89.sh"; fi
put "$SHIP/run_a89.sh" "$BASE/run_a89.sh"; chmod +x "$BASE/run_a89.sh"
ARMS=$S/ops/a89/A89_ARMS_D2800_p60.json
if [ ! -f "$ARMS" ]; then
  ( source "$BASE/env_a89.sh" && cd "$S" && "$PY" -I -c "import sys; sys.path[:0]=['$S/pytools','$S/ops','$S']; sys.argv=['b','$FROM/root/a75d2_banks/A75D2_BANK_D2800_p60.json','$E1R_CANDS','$ARMS','$E1R_CANDS_SHA']; exec(open('ops/a89/build_a89_arms.py').read())" ) > "$BASE/build_arms.out" 2>&1 || { cat "$BASE/build_arms.out"; die "arms bank build"; }
fi
cat "$BASE/build_arms.out"
echo "arms bank $(sha "$ARMS")"
( source "$BASE/env_a89.sh" && cd "$S" && A89_ARMS_SHA256=$(sha "$ARMS") "$PY" - <<'EOF'
import sys, json
sys.path[:0] = ['ops', '.']
from tools.coopmwm import a75_holdforms as H, nm_eval_runner as R
import a75_d2forms as D2
import a81_bigconfirm as B
import a89_sixarm as M
import a80_s2b as A80
import nm_finaleval as N
assert H.a75_digest() == D2.A75_SHA256, ('a75 digest', H.a75_digest())
assert R.A75_ARMS == ('A75_PLAN', 'A75_BANK') and tuple(R.A70_RANGES) == ((996300, 996399),)
assert tuple(R.A81_RANGES) == ((990100, 990499),) and tuple(R.A89_RANGES) == (M.BLOCK,)
M.point()
assert B._runner_check()
assert 'a70c_D2800_p60' in N.SETTINGS and 'a89c_D2800_p60' not in N.SETTINGS
# TAPER spec == the E1R draw strategy of mut_1fca373046a3c66e (canonical)
doc = json.load(open('ops/a89/A89_ARMS_D2800_p60.json'))
for arm, fam in M.ARM_FAMILY.items():
    c = H.make_core('A75_PLAN', M.CELL, doc, family=fam)
    print(arm, fam, c.plans[0]['key'], 'extended' if c.plans[0]['extended'] else 'plain', len(c.plans[0]['spec']['roles']), 'roles')
print('code identity ok: a75', H.a75_digest()[:8], 'A89 block', M.BLOCK)
EOF
) || die "code identity check"
echo "tree diff vs the A81 tree (expected: runner + the A89 files):"
diff -rq $FROM/src "$S" -x __pycache__ | cut -c1-200
du -sh "$BASE" | cut -c1-80
df -BG --output=avail "$BASE" | tail -1
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
