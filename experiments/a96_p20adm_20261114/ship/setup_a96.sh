#!/bin/bash
# AMENDMENT 96 part (b) host setup (zero SUMO, never deletes, writes only $A96_BASE). Same recipe as A82 part A:
# $BASE/src = REAL COPY (cp -a) of the finished A76 tree /root/a76p20/src (verified file by file) + a80_s2b, a77_mappo20c,
# a82_s3b20 (the registered A80 / A77 / A82 files, sha pinned) + a96_p20adm.py; runner = A76 runner 79fc9efd +
# patch_tables_a96.py (A96_RANGES 988900-988999, one additive row; sha pinned after patching). $BASE/root = copies of
# the p20c bank-root binding files (cmp with the A76 root). /root/a76p20 and the p20c bank root are only READ.
#   A96_BASE=/root/a96ref|/root/a96adm_*|/root/a96synth_* bash setup_a96.sh
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=${A96_BASE:?A96_BASE}
case "$BASE" in /root/a96ref|/root/a96adm_hand|/root/a96adm_s4|/root/a96adm_s5|/root/a96synth_ref|/root/a96synth_adm) ;; *) echo "A96_BASE not an A96 base"; exit 2;; esac
FROM=/root/a76p20
S=$BASE/src
R=$BASE/root
B=/root/autodl-fs/nm_finaleval_p20c_bank
RUN76_SHA=79fc9efd3df81065479bd8895daf02f2761da9246c997f3731ede3987dd1cd55
A75_FILE_SHA=72046db6ad9a3f117105ffdf029ac993ec3d421026da6a1ca4095d5f7be620eb
A80_SHA=48c5c448c2a85d4e2020def54cb17259979c89f2e0cc341829d6e99de0d0e289
A77_SHA=308894c671d6ddb84ecc688058b8316a762b286b47e182c59ddd7dfcc2a9156c
A82A_SHA=2b33146bfb5df716959d8a2033ff59f78ba48bc2bd4591237ae9c1a76706fdd5
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
[ -f "$BASE/SETUP_OK" ] && die "$BASE/SETUP_OK exists (setup runs once)"
( cd "$SHIP" && sha256sum -c --quiet SHIP_FILES.sha256 ) || die "shipped files differ from SHIP_FILES.sha256"
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
[ -f $FROM/SETUP_OK ] || die "$FROM is not a completed A76 setup"
[ -L $FROM/src ] && die "$FROM/src is a symlink"
RT=pytools/tools/coopmwm/nm_eval_runner.py
pin "$FROM/src/$RT" $RUN76_SHA
pin "$FROM/src/pytools/tools/coopmwm/a75_holdforms.py" $A75_FILE_SHA
PY=$(grep "^PY=" $FROM/env_a76.sh | cut -d= -f2)
[ -x "$PY" ] || die "python of the A76 env not found"
mkdir -p "$BASE"
if [ ! -d "$S" ]; then nice -n 19 cp -a "$FROM/src" "$S" || die "copy tree"; fi
[ -L "$S" ] && die "$S is a symlink"
( cd "$FROM/src" && find . -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum ) > "$BASE/a76_tree.sha256"
( cd "$S" && sha256sum -c --quiet "$BASE/a76_tree.sha256" ) || die "the copy differs from the A76 tree"
echo "tree copy ok: $(wc -l < "$BASE/a76_tree.sha256") files identical to /root/a76p20/src"
mkdir -p "$S/ops/a96"
put "$SHIP/a80_s2b.py" "$S/ops/a80_s2b.py"; pin "$S/ops/a80_s2b.py" $A80_SHA
put "$SHIP/a77_mappo20c.py" "$S/ops/a77_mappo20c.py"; pin "$S/ops/a77_mappo20c.py" $A77_SHA
put "$SHIP/a82_s3b20.py" "$S/ops/a82_s3b20.py"; pin "$S/ops/a82_s3b20.py" $A82A_SHA
put "$SHIP/a96_p20adm.py" "$S/ops/a96_p20adm.py"
for f in patch_tables_a96.py SEED_AUDIT_A96.json; do put "$SHIP/$f" "$S/ops/a96/$f"; done
if ! grep -q "^A96_RANGES = " "$S/$RT"; then
  [ "$(sha "$S/$RT")" = $RUN76_SHA ] || die "$S/$RT is neither the A76 runner nor the A96 runner"
  P=$BASE/patch.tmp
  [ -e "$P" ] && die "$P exists (inspect; never removed automatically)"
  mkdir "$P" && cp "$S/$RT" "$P/runner.py" || die "copy runner"
  "$PY" -I "$SHIP/patch_tables_a96.py" tables "$P/runner.py" || die "A96 table patch"
  mv -f "$P/runner.py" "$S/$RT" || die "install patched runner"
  mv "$P" "$BASE/patch.done" || die "keep patch dir"
fi
pin "$FROM/src/$RT" $RUN76_SHA
echo "runner A96: $(sha "$S/$RT")"
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || cp "$B/$f" "$R/$f" || die "mirror $f"
  cmp -s "$B/$f" "$R/$f" || die "$R/$f differs from the p20c binding root"
  cmp -s "$FROM/root/$f" "$R/$f" || die "$R/$f differs from the A76 root binding"
done
sed -e "s#^BASE=/root/a76p20#BASE=$BASE#" -e "s#^SRC=/root/a76p20/src#SRC=$S#" -e "s#A76_HOST_BASE=/root/a76p20#A76_HOST_BASE=$BASE#" $FROM/env_a76.sh > "$BASE/env_a96.sh.new"
grep -q "^SRC=$S$" "$BASE/env_a96.sh.new" || die "env: SRC not switched to $S"
echo "export A96_HOST_BASE=$BASE A76_COMP_SHA256=d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca" >> "$BASE/env_a96.sh.new"
if [ -f "$BASE/env_a96.sh" ]; then cmp -s "$BASE/env_a96.sh" "$BASE/env_a96.sh.new" || die "env differs"; rm "$BASE/env_a96.sh.new"; else mv "$BASE/env_a96.sh.new" "$BASE/env_a96.sh"; fi
put "$SHIP/run_a96.sh" "$BASE/run_a96.sh"; chmod +x "$BASE/run_a96.sh"
( source "$BASE/env_a96.sh" && cd "$S" && nice -n 19 "$PY" - <<'PYEOF'
import sys
sys.path[:0] = ['ops', '.']
from tools.coopmwm import a75_holdforms as H, nm_eval_runner as R
import a96_p20adm as M96, a82_s3b20 as MA, a76_d2p20 as M76
assert H.a75_digest() == MA.A75_SHA256 == M76.A75_SHA256 == M96.A75_SHA256, ('a75 digest', H.a75_digest())
assert R.A75_ARMS == ('A75_PLAN', 'A75_BANK')
assert tuple(R.A96_RANGES) == ((988900, 988999),) and not hasattr(R, 'A82_RANGES')
assert M76.check_blocks_free(R.A96_RANGES, R.EVAL_HELDOUT_RANGES) and M76.check_blocks_free(R.A76_RANGES, R.EVAL_HELDOUT_RANGES)
print('code identity ok: a75', H.a75_digest()[:8], 'A96_RANGES', R.A96_RANGES)
PYEOF
) || die "code identity check"
echo "tree diff vs the A76 tree (expected: runner, a80_s2b, a77_mappo20c, a82_s3b20, a96_p20adm, ops/a96):"
diff -rq $FROM/src "$S" -x __pycache__ | cut -c1-200
du -sh "$BASE" | cut -c1-80
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
