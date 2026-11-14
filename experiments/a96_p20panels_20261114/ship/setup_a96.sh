#!/bin/bash
# AMENDMENT 96 parts (c) / (d) host setup (root; zero SUMO; never deletes; writes only /root/a96p<part>).
#   bash setup_a96.sh c   -> /root/a96pc  (matched panel, block 984400-984599, runner + A96C_RANGES)
#   bash setup_a96.sh d   -> /root/a96pd  (earlier-search panel 988200-988391, the A82 part-B runner unchanged)
# Source tree = REAL copy (cp -a) of the never-launched A82 part-B tree /root/a82four/src (diff-verified; /root/a82four is
# only read). New files: ops/a96_p20.py, ops/test_a96_p20.py, ops/a96_e2e.py, ops/a96/{build_bank_a96.py,
# patch_tables_a96c.py, SEED_AUDIT_A96C.json}. Root binding files = byte copies of /root/a82four/root (cmp-checked);
# the A32 DUO_HOLD identity chunk = byte copy of /root/a82four/refs/a32 (sha pinned).
set -uo pipefail
PART=${1:?part c | d}
case "$PART" in c|d) ;; *) echo "part c | d"; exit 2 ;; esac
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a96p$PART
FROM=/root/a82four
S=$BASE/src
R=$BASE/root
RUN82_SHA=2de0f1c07feb3e1d509367a11e65d392c6f6cd93a4bb605d92ec672d53e21d76
DRV82_SHA=5181bf4ba5aaf610de4c820b4b508ab617ba622c2f5855bf4f0c621d95a21f0a
A80_SHA=48c5c448c2a85d4e2020def54cb17259979c89f2e0cc341829d6e99de0d0e289
A32W_SHA=007cef740b64adf0eee02890c37f388142b26d4126eee11c9594a85847baec65
P192_SHA=1648eb4b6d578a0256896120f49531d99edc4badf1e9e87e21762eb432462a89
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
[ -f "$BASE/SETUP_OK" ] && die "$BASE/SETUP_OK exists (setup runs once)"
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
( cd "$SHIP" && sha256sum -c --quiet SHIP_FILES.sha256 ) || die "shipped files differ from SHIP_FILES.sha256"
[ -f $FROM/SETUP_OK ] || die "$FROM is not a completed A82 part-B setup"
[ -e $FROM/chain_mech.out ] || [ -e $FROM/root/CHARGES.jsonl ] && die "$FROM was launched (expected never launched)"
RT=pytools/tools/coopmwm/nm_eval_runner.py
pin "$FROM/src/$RT" $RUN82_SHA
pin "$FROM/src/ops/a82_fourarm.py" $DRV82_SHA
pin "$FROM/src/ops/a80_s2b.py" $A80_SHA
pin "$FROM/refs/a32/WAVE_DUO_HOLD__0000-0047.json" $A32W_SHA
pin "$FROM/panels/A82F_PANEL_C_192.json" $P192_SHA
PY=$(grep "^PY=" $FROM/env_a82.sh | cut -d= -f2)
[ -x "$PY" ] || die "python of the A82 env not found"
mkdir -p "$BASE"
if [ ! -d "$S" ]; then cp -a $FROM/src "$S" || die "copy tree"; fi
[ -L "$S" ] && die "$S is a symlink"
diff -rq --no-dereference $FROM/src "$S" -x __pycache__ > "$BASE/tree_diff_before.txt"; [ -s "$BASE/tree_diff_before.txt" ] && die "copied tree differs before patching"
mkdir -p "$S/ops/a96"
put "$SHIP/a96_p20.py" "$S/ops/a96_p20.py"
put "$SHIP/test_a96_p20.py" "$S/ops/test_a96_p20.py"
put "$SHIP/a96_e2e.py" "$S/ops/a96_e2e.py"
put "$SHIP/build_bank_a96.py" "$S/ops/a96/build_bank_a96.py"
put "$SHIP/patch_tables_a96c.py" "$S/ops/a96/patch_tables_a96c.py"
put "$SHIP/SEED_AUDIT_A96C.json" "$S/ops/a96/SEED_AUDIT_A96C.json"
if [ "$PART" = c ] && ! grep -q "^A96C_RANGES = ((984400, 984599),)$" "$S/$RT"; then
  [ "$(sha "$S/$RT")" = $RUN82_SHA ] || die "$S/$RT is neither the A82 runner nor the A96 (c) runner"
  "$PY" -I "$SHIP/patch_tables_a96c.py" tables "$S/$RT" || die "A96C table patch"
  [ "$(diff "$FROM/src/$RT" "$S/$RT" | grep -c '^<')" = 1 ] || die "the runner patch changed more than the tail line"
  [ "$(stat -c %i "$S/$RT")" != "$(stat -c %i "$FROM/src/$RT")" ] || die "the A96 runner shares the A82 inode"
fi
pin "$FROM/src/$RT" $RUN82_SHA
echo "runner A82 ${RUN82_SHA:0:8} (unchanged) -> A96 part $PART $(sha "$S/$RT")"
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks" "$BASE/refs/a32"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || cp "$FROM/root/$f" "$R/$f" || die "copy $f"
  cmp -s "$FROM/root/$f" "$R/$f" || die "$R/$f differs from the A82 root"
done
put "$FROM/refs/a32/WAVE_DUO_HOLD__0000-0047.json" "$BASE/refs/a32/WAVE_DUO_HOLD__0000-0047.json"
sed -e "s#^BASE=/root/a82four#BASE=$BASE#" -e "s#^SRC=/root/a82four/src#SRC=$S#" \
    -e "s#A82F_HOST_BASE=/root/a82four#A96_HOST_BASE=$BASE A96_PART=$PART#" $FROM/env_a82.sh > "$BASE/env_a96.sh.new"
grep -q "^SRC=$S$" "$BASE/env_a96.sh.new" || die "env: SRC not switched"
grep -q "A96_HOST_BASE=$BASE A96_PART=$PART " "$BASE/env_a96.sh.new" || die "env: A96_HOST_BASE / A96_PART not set"
grep -q "a82four" "$BASE/env_a96.sh.new" && die "env: an a82four path remains"
if [ -f "$BASE/env_a96.sh" ]; then cmp -s "$BASE/env_a96.sh" "$BASE/env_a96.sh.new" || die "env differs"; else cp "$BASE/env_a96.sh.new" "$BASE/env_a96.sh"; fi
put "$SHIP/run_a96.sh" "$BASE/run_a96.sh"; chmod +x "$BASE/run_a96.sh"
echo "$PART" > "$BASE/PART"
( source "$BASE/env_a96.sh" && cd "$S" && A96_HAND_SHA256=none "$PY" - <<'EOF'
import os, sys
sys.path[:0] = ['ops', '.']
from tools.coopmwm import a75_holdforms as H, nm_eval_runner as R
import a96_p20 as M, a76_d2p20 as M76, a80_s2b, a77_mappo20c, a32_ablation
assert H.a75_digest() == M.A75_SHA256 == M76.A75_SHA256, 'a75 digest'
assert tuple(R.A82_RANGES) == ((988200, 988599),) and tuple(getattr(R, M.RUNNER_ATTR)) == (M.BLOCK,)
M._runner()
assert M.PART == os.environ['A96_PART']
print('code identity ok: part', M.PART, 'a75', H.a75_digest()[:8], 'block', M.BLOCK, 'seeds', M.seeds_of(192)[0],
      M.seeds_of(192)[-1], 'arms(no hand)', M.ARMS)
EOF
) || die "code identity check"
echo "tree diff vs the A82 tree (expected: [c: runner] + the A96 files):"
diff -rq $FROM/src "$S" -x __pycache__ | cut -c1-200
du -sh "$BASE" | cut -c1-80
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
