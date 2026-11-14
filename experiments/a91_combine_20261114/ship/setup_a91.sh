#!/bin/bash
# AMENDMENT 91 (A91) host setup (root; zero SUMO; never deletes; writes only /root/a91). Derived from setup_a89.sh.
# Source tree = REAL copy (cp -a, no hard links) of the finished A89 tree /root/a89/src (all files cmp-verified).
# ONE changed file: pytools/tools/coopmwm/nm_eval_runner.py (A89 982ec0cf + A91_RANGES, patch_tables_a91.py).
# New files under ops/: a91_combine.py, test_a91_combine.py, a91_analyse_e2e.py, a91/build_a91_turns.py,
# a91/patch_tables_a91.py, a91/SEED_AUDIT_A91.json, a91/A89_ARMS_D2800_p60.json (byte copy, de9a30a9),
# a91/A91_TURNS_D2800_p60.json (built here, zero SUMO, from de9a30a9).
# Root binding files = byte copies of the A89 root binding files (checked equal to the A66 root).
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a91
FROM=/root/a89
S=$BASE/src
R=$BASE/root
RUN_A89_SHA=982ec0cf56ea6803f9e294567278ae5ca96677a97e4e871f2e3d40e3ae9f934f
DRV_A81_SHA=c755a5510bc7c1c5f9f9279e7b2f40a77c388b5ccd60a35db2207b7d2dc4c4bb
A80_SHA=48c5c448c2a85d4e2020def54cb17259979c89f2e0cc341829d6e99de0d0e289
ARMS_SHA=de9a30a944eda470777cbd3247c6ca81db2745547549cde3e2bff154e23824c2
BANK_SHA=c325ec1da034d0ad22d242f97e926c30cfa998981f16cef935b1b6498565c30c
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
[ -f "$BASE/SETUP_OK" ] && die "$BASE/SETUP_OK exists (setup runs once)"
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
( cd "$SHIP" && sha256sum -c --quiet SHIP_FILES.sha256 ) || die "shipped files differ from SHIP_FILES.sha256"
[ -f $FROM/SETUP_OK ] || die "$FROM is not a completed A89 setup"
grep -q A89_CONFIRM_RUN_DONE $FROM/chain_confirm.out || die "A89 confirm is not done"
RT=pytools/tools/coopmwm/nm_eval_runner.py
pin "$FROM/src/$RT" $RUN_A89_SHA
pin "$FROM/src/ops/a81_bigconfirm.py" $DRV_A81_SHA
pin "$FROM/src/ops/a80_s2b.py" $A80_SHA
pin "$FROM/src/ops/a89/A89_ARMS_D2800_p60.json" $ARMS_SHA
pin /root/a75d2c/root/a75d2_banks/A75D2_BANK_D2800_p60.json $BANK_SHA
PY=$(grep "^PY=" $FROM/env_a89.sh | cut -d= -f2)
[ -x "$PY" ] || die "python of the A89 env not found"
mkdir -p "$BASE"
if [ ! -d "$S" ]; then cp -a $FROM/src "$S" || die "copy tree"; fi
[ -L "$S" ] && die "$S is a symlink"
diff -rq --no-dereference $FROM/src "$S" -x __pycache__ > "$BASE/tree_diff_before.txt"; [ -s "$BASE/tree_diff_before.txt" ] && die "copied tree differs before patching"
mkdir -p "$S/ops/a91"
put "$SHIP/a91_combine.py" "$S/ops/a91_combine.py"
put "$SHIP/test_a91_combine.py" "$S/ops/test_a91_combine.py"
put "$SHIP/a91_analyse_e2e.py" "$S/ops/a91_analyse_e2e.py"
put "$SHIP/build_a91_turns.py" "$S/ops/a91/build_a91_turns.py"
put "$SHIP/patch_tables_a91.py" "$S/ops/a91/patch_tables_a91.py"
put "$SHIP/SEED_AUDIT_A91.json" "$S/ops/a91/SEED_AUDIT_A91.json"
put "$S/ops/a89/A89_ARMS_D2800_p60.json" "$S/ops/a91/A89_ARMS_D2800_p60.json"
pin "$S/ops/a91/A89_ARMS_D2800_p60.json" $ARMS_SHA
if ! grep -q "^A91_RANGES = ((988600, 988699),)$" "$S/$RT"; then
  [ "$(sha "$S/$RT")" = $RUN_A89_SHA ] || die "$S/$RT is neither the A89 runner nor the A91 runner"
  "$PY" -I "$SHIP/patch_tables_a91.py" tables "$S/$RT" || die "A91 table patch"
  [ "$(diff "$FROM/src/$RT" "$S/$RT" | grep -c '^<')" = 1 ] || die "the runner patch changed more than the tail line"
fi
pin "$FROM/src/$RT" $RUN_A89_SHA
[ "$(stat -c %i "$S/$RT")" != "$(stat -c %i "$FROM/src/$RT")" ] || die "the A91 runner shares the A89 inode"
echo "runner A89 ${RUN_A89_SHA:0:8} (unchanged) -> A91 $(sha "$S/$RT")"
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || cp "$FROM/root/$f" "$R/$f" || die "copy $f"
  cmp -s "/root/a66/root/$f" "$R/$f" || die "$R/$f differs from the A66 root"
  cmp -s "$FROM/root/$f" "$R/$f" || die "$R/$f differs from the A89 root"
done
pin /root/a66/inputs/policy_00084.pt 2abed31707b61121569af2936c23689827b52b28422df162751b230aa6a93de8
[ -d /root/a75/a70ref/root ] && [ -d /root/a75/ref70/evaluation ] || die "A70 references missing"
sed -e "s#^BASE=/root/a89#BASE=$BASE#" -e "s#^SRC=/root/a89/src#SRC=$S#" -e "/A89_HOST_BASE/d" \
    $FROM/env_a89.sh > "$BASE/env_a91.sh.new"
echo "export A91_HOST_BASE=$BASE" >> "$BASE/env_a91.sh.new"
grep -q "^SRC=$S$" "$BASE/env_a91.sh.new" || die "env: SRC not switched"
grep -q "^export A75D2_HOST_BASE=/root/a75d2c$" "$BASE/env_a91.sh.new" || die "env: A75D2_HOST_BASE is not /root/a75d2c"
if [ -f "$BASE/env_a91.sh" ]; then cmp -s "$BASE/env_a91.sh" "$BASE/env_a91.sh.new" || die "env differs"; else cp "$BASE/env_a91.sh.new" "$BASE/env_a91.sh"; fi
put "$SHIP/run_a91.sh" "$BASE/run_a91.sh"; chmod +x "$BASE/run_a91.sh"
TURNS=$S/ops/a91/A91_TURNS_D2800_p60.json
if [ ! -f "$TURNS" ]; then
  ( source "$BASE/env_a91.sh" && cd "$S" && "$PY" -I -c "import sys; sys.path[:0]=['$S/pytools','$S/ops','$S']; sys.argv=['b','$S/ops/a91/A89_ARMS_D2800_p60.json','$TURNS']; exec(open('ops/a91/build_a91_turns.py').read())" ) > "$BASE/build_turns.out" 2>&1 || { cat "$BASE/build_turns.out"; die "turns bank build"; }
fi
cat "$BASE/build_turns.out"
echo "turns bank $(sha "$TURNS")"
( source "$BASE/env_a91.sh" && cd "$S" && A91_TURNS_SHA256=$(sha "$TURNS") "$PY" - <<'EOF'
import sys, json
sys.path[:0] = ['ops', '.']
from tools.coopmwm import a75_holdforms as H, nm_eval_runner as R
import a75_d2forms as D2
import a81_bigconfirm as B
import a91_combine as M
import a80_s2b as A80
import nm_finaleval as N
assert H.a75_digest() == D2.A75_SHA256, ('a75 digest', H.a75_digest())
assert R.A75_ARMS == ('A75_PLAN', 'A75_BANK') and tuple(R.A70_RANGES) == ((996300, 996399),)
assert tuple(R.A81_RANGES) == ((990100, 990499),) and tuple(R.A89_RANGES) == ((988400, 988499),)
assert tuple(R.A91_RANGES) == (M.BLOCK,)
M.point()
assert B._runner_check()
assert 'a70c_D2800_p60' in N.SETTINGS and 'a91c_D2800_p60' not in N.SETTINGS
doc = json.load(open('ops/a91/A89_ARMS_D2800_p60.json'))
for arm, fam in M.ARM_FAMILY.items():
    c = H.make_core('A75_PLAN', M.CELL, doc, family=fam)
    print(arm, fam, c.plans[0]['key'], 'extended' if c.plans[0]['extended'] else 'plain', len(c.plans[0]['spec']['roles']), 'roles')
t = json.load(open('ops/a91/A91_TURNS_D2800_p60.json'))
c = H.make_core('A75_BANK', M.CELL, t)
print('BANK_TURNS priority', [(p['family'], p['key'], type(k).__name__) for p, k in zip(c.plans, c.cores)])
print('code identity ok: a75', H.a75_digest()[:8], 'A91 block', M.BLOCK)
EOF
) || die "code identity check"
echo "tree diff vs the A89 tree (expected: runner + the A91 files):"
diff -rq $FROM/src "$S" -x __pycache__ | cut -c1-200
du -sh "$BASE" | cut -c1-80
df -BG --output=avail "$BASE" | tail -1
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
