#!/bin/bash
# AMENDMENT 76 (A76-D2P20) host setup (run as root; zero SUMO; never deletes; writes only $A76_BASE, default /root/a76p20).
# usage: [A76_BASE=/root/a76p20] [A76_PANELS_FROM=<other A76 root>] bash setup_a76p20.sh
# Source tree $BASE/src = HARD-LINK clone (cp -al) of /root/a69/src (A68 tree + A44 executor + merged runner 62496e86;
#   the p20c plant / bindings code; A73 used the same base at 20 %) + the A76 files as new small files. The runner is
#   copied to a NEW inode, patched additively (A75 arms = ops/a75/host/patch_runner_a75.py; A76 seed blocks =
#   patch_tables_a76.py) and renamed over the link: /root/a69/src keeps its bytes (checked). nm_finaleval is unchanged.
# Root $BASE/root = the p20c bank-root binding files (hard links from /root/a69/root, compared with the p20c bank root),
#   exactly the A73 list. Nothing under /root/a75d2c, /root/a75d2b, /root/a75d2, /root/a73 is read or written.
# A76_PANELS_FROM (re-binding a new comparator only): hard-link the three pinned panels of an earlier A76 root so the
#   new root runs the SAME worlds (disclose in its registration).
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=${A76_BASE:-/root/a76p20}
FROM=/root/a69
S=$BASE/src
R=$BASE/root
RUN69_SHA=62496e864567700a051aaf3d26ee16fb64e1dbe5b0a34b7d28688e0defb67836
A44_SHA=4e7a97f3183290ca5f1ad161cb3557b27d9b3695eb596a18f3ee3097e92aac0c
A75_FILE_SHA=72046db6ad9a3f117105ffdf029ac993ec3d421026da6a1ca4095d5f7be620eb
FIX_SHA=c9999d06668b1beb6ba5dad325322d6104a2e997d5ed9b6e1d1eb78dd6e991a4
RUN76_SHA=79fc9efd3df81065479bd8895daf02f2761da9246c997f3731ede3987dd1cd55   # A69 runner + A75 arms + A76 blocks (probed off-tree)
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
case "$BASE" in /root/a76p20*) ;; *) die "A76_BASE must be /root/a76p20* (got $BASE)";; esac
[ -f "$BASE/SETUP_OK" ] && die "$BASE/SETUP_OK exists (setup runs once)"
( cd "$SHIP" && sha256sum -c --quiet SHIP_FILES.sha256 ) || die "shipped files differ from SHIP_FILES.sha256"
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
[ -L $FROM/src ] && die "$FROM/src is a symlink"
[ -f $FROM/SETUP_OK ] || die "$FROM is not a completed A69 setup"
RT=pytools/tools/coopmwm/nm_eval_runner.py
pin "$FROM/src/$RT" $RUN69_SHA
pin "$FROM/src/pytools/tools/coopmwm/a44_lcvariant.py" $A44_SHA
pin "$FROM/src/ops/a44/FIXTURE_PLANS.json" $FIX_SHA
PY=$(grep "^PY=" $FROM/env_a69.sh | cut -d= -f2)
[ -x "$PY" ] || die "python of the A69 env not found"
mkdir -p "$BASE"
if [ ! -d "$S" ]; then cp -al $FROM/src "$S" || die "link tree"; fi
[ -L "$S" ] && die "$S is a symlink"
[ "$(readlink -f $S)" = "$S" ] || die "$S does not resolve to itself"
T=$S/pytools/tools/coopmwm
mkdir -p "$S/ops/a75" "$S/ops/a75d2" "$S/ops/a76/host"
put "$SHIP/a75_holdforms.py" "$T/a75_holdforms.py"
pin "$T/a75_holdforms.py" $A75_FILE_SHA
put "$SHIP/a75_specs.py" "$S/ops/a75/a75_specs.py"
put "$SHIP/a75_d2forms.py" "$S/ops/a75_d2forms.py"
for f in CANDIDATES_D2_D2800_p60.json CANDIDATES_D2_D2800_p60.csv CANDIDATES_D2_D2800_p60.sha256 REF_PINS.json EXAMPLE_PLANS.json; do
  put "$SHIP/$f" "$S/ops/a75d2/$f"
done
put "$SHIP/a76_d2p20.py" "$S/ops/a76_d2p20.py"
put "$SHIP/test_a76_d2p20.py" "$S/ops/test_a76_d2p20.py"
for f in patch_tables_a76.py patch_runner_a75.py; do put "$SHIP/$f" "$S/ops/a76/host/$f"; done
put "$SHIP/SEED_AUDIT.json" "$S/ops/a76/SEED_AUDIT.json"
# runner: new inode, additive patches, rename over the link
if ! grep -q "^A76_RANGES = " "$S/$RT"; then
  [ "$(sha "$S/$RT")" = $RUN69_SHA ] || die "$S/$RT is neither the A69 runner nor the A76 runner"
  P=$BASE/patch.tmp
  [ -e "$P" ] && die "$P exists (inspect; never removed automatically)"
  mkdir "$P" && cp "$S/$RT" "$P/runner.py" || die "copy runner"
  "$PY" -I "$SHIP/patch_runner_a75.py" arms "$P/runner.py" || die "A75 arms patch"
  "$PY" -I "$SHIP/patch_tables_a76.py" tables "$P/runner.py" || die "A76 table patch"
  pin "$P/runner.py" $RUN76_SHA
  mv -f "$P/runner.py" "$S/$RT" || die "install patched runner"
  mv "$P" "$BASE/patch.done" || die "keep patch dir"
fi
pin "$FROM/src/$RT" $RUN69_SHA
[ "$(stat -c %i "$S/$RT")" != "$(stat -c %i "$FROM/src/$RT")" ] || die "the A76 runner still shares the A69 inode"
N=$(diff "$FROM/src/$RT" "$S/$RT" | grep -c '^<')
[ "$N" = 5 ] || die "runner diff removes $N lines (expected the 5 replaced lines of the two additive patches)"
pin "$S/$RT" $RUN76_SHA
echo "runner A69 ${RUN69_SHA:0:8} (unchanged) -> A76 $(sha "$S/$RT")"
B=/root/autodl-fs/nm_finaleval_p20c_bank
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || ln "$FROM/root/$f" "$R/$f" || die "mirror $f"
  cmp -s "$B/$f" "$R/$f" || die "$R/$f differs from the p20c binding root"
done
if [ -n "${A76_PANELS_FROM:-}" ]; then
  for f in A76_PANEL_M.json A76_PANEL_V.json A76_PANEL_C.json; do
    [ -f "$R/$f" ] || ln "$A76_PANELS_FROM/$f" "$R/$f" || die "panel link $f"
  done
fi
sed -e "s#^BASE=/root/a69#BASE=$BASE#" -e "s#^SRC=/root/a69/src#SRC=$S#" $FROM/env_a69.sh > "$BASE/env_a76.sh.new"
echo "export A76_HOST_BASE=$BASE" >> "$BASE/env_a76.sh.new"
grep -q "^SRC=$S$" "$BASE/env_a76.sh.new" || die "env: SRC not switched to $S"
if [ -f "$BASE/env_a76.sh" ]; then cmp -s "$BASE/env_a76.sh" "$BASE/env_a76.sh.new" || die "env differs"; rm "$BASE/env_a76.sh.new"; else mv "$BASE/env_a76.sh.new" "$BASE/env_a76.sh"; fi
put "$SHIP/run_a76p20.sh" "$BASE/run_a76p20.sh"; chmod +x "$BASE/run_a76p20.sh"
# code identity on the box: executor digest, runner arms + blocks, candidates = the seeded draw, p20 banks execute
( source "$BASE/env_a76.sh" && cd "$S" && nice -n 19 "$PY" - <<'EOF'
import sys
sys.path[:0] = ['ops', '.']
from tools.coopmwm import a75_holdforms as H, nm_eval_runner as R
import a76_d2p20 as M
assert H.a75_digest() == M.A75_SHA256, ('a75 digest', H.a75_digest())
assert R.A75_ARMS == ('A75_PLAN', 'A75_BANK') and R.A44_ARMS == ('A44_PLAN', 'A44_BANK')
assert tuple(R.A76_RANGES) == ((997100, 997199), (997200, 997299), (997300, 997399))
assert M.check_blocks_free(R.A76_RANGES, R.EVAL_HELDOUT_RANGES)
c = M.candidates()
assert len(c) == 36 and M.cands_bank_doc(c)
M.examples_bank20()
M.ident_bank_doc()
print('code identity ok: a75', H.a75_digest()[:8], 'blocks', R.A76_RANGES, 'candidates', len(c))
EOF
) || die "code identity check"
echo "tree diff vs the A69 tree (expected: runner, a75_holdforms, ops/a75/a75_specs.py, a75_d2forms + ops/a75d2, the A76 driver/test, ops/a76):"
diff -rq $FROM/src "$S" -x __pycache__ | cut -c1-200
du -sh "$BASE" | cut -c1-80
df -BG --output=avail "$BASE" | tail -1
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
