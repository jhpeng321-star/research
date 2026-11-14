#!/bin/bash
# AMENDMENT 81 (S-BIGCONFIRM-A81) host setup (run as root; zero SUMO; never deletes; writes only /root/a81big).
# Source tree = HARD-LINK clone (cp -al) of the finished A75-D2 tree /root/a75d2c/src. ONE changed file:
#   pytools/tools/coopmwm/nm_eval_runner.py copied to a NEW inode, patched additively (A81_RANGES = ((990100, 990499),),
#   ops/a81/host/patch_tables_a81.py), renamed over the link (the A75-D2 inode keeps its bytes; checked).
# New files: ops/a81_bigconfirm.py, ops/test_a81_bigconfirm.py, ops/a81/host/patch_tables_a81.py, ops/a81/SEED_AUDIT.json.
# Root binding files = the A68 root binding files (hard links, as A75-D2), checked equal to the A66 root.
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a81big
FROM=/root/a75d2c
A68=/root/a68
S=$BASE/src
R=$BASE/root
RUN_D2_SHA=ec095e50fde9eaa885b3f8dcf9c10199bddeb1cae6f50730da34e2a27dceb84e
BANK_SHA=c325ec1da034d0ad22d242f97e926c30cfa998981f16cef935b1b6498565c30c
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
[ -f "$BASE/SETUP_OK" ] && die "$BASE/SETUP_OK exists (setup runs once)"
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
( cd "$SHIP" && sha256sum -c --quiet SHIP_FILES.sha256 ) || die "shipped files differ from SHIP_FILES.sha256"
[ -L $FROM/src ] && die "$FROM/src is a symlink"
[ -f $FROM/SETUP_OK ] || die "$FROM is not a completed A75-D2 setup"
grep -q A75D2_CONFIRM_DONE $FROM/chain_confirm.out || die "A75-D2 confirm is not done"
RT=pytools/tools/coopmwm/nm_eval_runner.py
pin "$FROM/src/$RT" $RUN_D2_SHA
pin "$FROM/root/a75d2_banks/A75D2_BANK_D2800_p60.json" $BANK_SHA
ST0=$(sha "$FROM/src/ops/nm_finaleval.py")
PY=$(grep "^PY=" $FROM/env_a75d2.sh | cut -d= -f2)
[ -x "$PY" ] || die "python of the A75-D2 env not found"
mkdir -p "$BASE"
if [ ! -d "$S" ]; then cp -al $FROM/src "$S" || die "link tree"; fi
[ -L "$S" ] && die "$S is a symlink"
[ "$(readlink -f $S)" = "$S" ] || die "$S does not resolve to itself"
mkdir -p "$S/ops/a81/host"
put "$SHIP/a81_bigconfirm.py" "$S/ops/a81_bigconfirm.py"
put "$SHIP/test_a81_bigconfirm.py" "$S/ops/test_a81_bigconfirm.py"
put "$SHIP/patch_tables_a81.py" "$S/ops/a81/host/patch_tables_a81.py"
put "$SHIP/SEED_AUDIT.json" "$S/ops/a81/SEED_AUDIT.json"
# runner: new inode, additive patch, rename over the link
if ! grep -q "^A81_RANGES = ((990100, 990499),)$" "$S/$RT"; then
  [ "$(sha "$S/$RT")" = $RUN_D2_SHA ] || die "$S/$RT is neither the A75-D2 runner nor the A81 runner"
  P=$BASE/patch.tmp
  [ -e "$P" ] && die "$P exists (inspect; never removed automatically)"
  mkdir "$P" && cp "$S/$RT" "$P/runner.py" || die "copy"
  "$PY" -I "$SHIP/patch_tables_a81.py" tables "$P/runner.py" || die "A81 table patch"
  [ "$(diff "$S/$RT" "$P/runner.py" | grep -c '^<')" = 1 ] || die "the runner patch changed more than the tail line"
  mv -f "$P/runner.py" "$S/$RT" || die "install patched runner"
  mv "$P" "$BASE/patch.done" || die "keep patch dir"
fi
pin "$FROM/src/$RT" $RUN_D2_SHA
[ "$(sha "$FROM/src/ops/nm_finaleval.py")" = "$ST0" ] || die "A75-D2 settings changed"
[ "$(stat -c %i "$S/$RT")" != "$(stat -c %i "$FROM/src/$RT")" ] || die "the A81 runner still shares the A75-D2 inode"
echo "runner A75-D2 ${RUN_D2_SHA:0:8} (unchanged) -> A81 $(sha "$S/$RT")"
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || ln "$A68/root/$f" "$R/$f" || die "mirror $f"
  cmp -s "/root/a66/root/$f" "$R/$f" || die "$R/$f differs from the A66 root"
  cmp -s "$FROM/root/$f" "$R/$f" || die "$R/$f differs from the A75-D2 root"
done
pin /root/a66/inputs/policy_00084.pt 2abed31707b61121569af2936c23689827b52b28422df162751b230aa6a93de8
[ -d /root/a75/a70ref/root ] && [ -d /root/a75/ref70/evaluation ] || die "A70 references missing"
sed -e "s#^BASE=/root/a75d2c#BASE=$BASE#" -e "s#^SRC=/root/a75d2c/src#SRC=$S#" -e "/A75D2_MECH_SEED0/d" \
    $FROM/env_a75d2.sh > "$BASE/env_a81.sh.new"
echo "export A81_HOST_BASE=$BASE" >> "$BASE/env_a81.sh.new"
grep -q "^SRC=$S$" "$BASE/env_a81.sh.new" || die "env: SRC not switched to $S"
grep -q "^export A75D2_HOST_BASE=/root/a75d2c$" "$BASE/env_a81.sh.new" || die "env: A75D2_HOST_BASE (stored-reference specs) is not /root/a75d2c"
if [ -f "$BASE/env_a81.sh" ]; then cmp -s "$BASE/env_a81.sh" "$BASE/env_a81.sh.new" || die "env differs"; else cp "$BASE/env_a81.sh.new" "$BASE/env_a81.sh"; fi
put "$SHIP/run_a81.sh" "$BASE/run_a81.sh"; chmod +x "$BASE/run_a81.sh"
( source "$BASE/env_a81.sh" && cd "$S" && "$PY" - <<'EOF'
import sys
sys.path[:0] = ['ops', '.']
from tools.coopmwm import a75_holdforms as H, nm_eval_runner as R
import a75_d2forms as D2
import a81_bigconfirm as M
import nm_finaleval as N
assert H.a75_digest() == D2.A75_SHA256, ('a75 digest', H.a75_digest())
assert R.A75_ARMS == ('A75_PLAN', 'A75_BANK') and tuple(R.A70_RANGES) == ((996300, 996399),)
assert tuple(R.A81_RANGES) == (M.BLOCK,) and M.block_free(M.BLOCK, R.EVAL_HELDOUT_RANGES)
assert 'a70c_D2800_p60' in N.SETTINGS and 'a81c_D2800_p60' not in N.SETTINGS
print('code identity ok: a75', H.a75_digest()[:8], 'A81 block', M.BLOCK)
EOF
) || die "code identity check"
echo "tree diff vs the A75-D2 tree (expected: runner + the four A81 files):"
diff -rq $FROM/src "$S" -x __pycache__ | cut -c1-200
du -sh "$BASE" | cut -c1-80
df -BG --output=avail "$BASE" | tail -1
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
