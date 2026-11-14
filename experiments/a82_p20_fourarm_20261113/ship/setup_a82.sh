#!/bin/bash
# AMENDMENT 82 (DRAFT) host setup, zero SUMO, never deletes, writes only the chosen base.
#   bash setup_a82.sh admit   -> base ${A82_BASE:-/root/a82p20}  (part A: S3b admission at 20 %)
#   bash setup_a82.sh four    -> base ${A82F_BASE:-/root/a82four} (part B: four-arm confirmation at 20 %)
# Source tree $BASE/src = a REAL COPY (cp -a, no hard links) of the finished A76 tree /root/a76p20/src (A69 tree + A75
# arms + A76 blocks; runner 79fc9efd) + the A82 files as new files. Part A keeps the A76 runner bytes; part B gets a new
# runner file = A76 runner + patch_tables_a82.py (block 988200-988599), sha pinned. /root/a76p20 is only READ.
# Root $BASE/root = COPIES of the p20c bank-root binding files (as A76, compared with /root/autodl-fs/nm_finaleval_p20c_bank).
set -uo pipefail
MODE=${1:?mode: admit | four}
SHIP=$(cd "$(dirname "$0")" && pwd)
case "$MODE" in
  admit) BASE=${A82_BASE:-/root/a82p20}; case "$BASE" in /root/a82p20*) ;; *) echo "A82_BASE must be /root/a82p20*"; exit 2;; esac ;;
  four) BASE=${A82F_BASE:-/root/a82four}; case "$BASE" in /root/a82four*) ;; *) echo "A82F_BASE must be /root/a82four*"; exit 2;; esac ;;
  *) echo "mode admit | four"; exit 2 ;;
esac
FROM=/root/a76p20
S=$BASE/src
R=$BASE/root
B=/root/autodl-fs/nm_finaleval_p20c_bank
RUN76_SHA=79fc9efd3df81065479bd8895daf02f2761da9246c997f3731ede3987dd1cd55
RUN82_SHA=2de0f1c07feb3e1d509367a11e65d392c6f6cd93a4bb605d92ec672d53e21d76
A75_FILE_SHA=72046db6ad9a3f117105ffdf029ac993ec3d421026da6a1ca4095d5f7be620eb
A80_SHA=48c5c448c2a85d4e2020def54cb17259979c89f2e0cc341829d6e99de0d0e289
A77_SHA=308894c671d6ddb84ecc688058b8316a762b286b47e182c59ddd7dfcc2a9156c
A32W_SHA=007cef740b64adf0eee02890c37f388142b26d4126eee11c9594a85847baec65
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
mkdir -p "$S/ops/a82"
put "$SHIP/a80_s2b.py" "$S/ops/a80_s2b.py"; pin "$S/ops/a80_s2b.py" $A80_SHA
put "$SHIP/a77_mappo20c.py" "$S/ops/a77_mappo20c.py"; pin "$S/ops/a77_mappo20c.py" $A77_SHA
for f in a82_s3b20.py test_a82_s3b20.py a82_fourarm.py test_a82_fourarm.py; do put "$SHIP/$f" "$S/ops/$f"; done
for f in SYNTH_TOP_OFFER_CONDITIONS_D2800_p20.json patch_tables_a82.py SEED_AUDIT_A82.json; do put "$SHIP/$f" "$S/ops/a82/$f"; done
if [ "$MODE" = four ] && ! grep -q "^A82_RANGES = " "$S/$RT"; then
  [ "$(sha "$S/$RT")" = $RUN76_SHA ] || die "$S/$RT is neither the A76 runner nor the A82 runner"
  P=$BASE/patch.tmp
  [ -e "$P" ] && die "$P exists (inspect; never removed automatically)"
  mkdir "$P" && cp "$S/$RT" "$P/runner.py" || die "copy runner"
  "$PY" -I "$SHIP/patch_tables_a82.py" tables "$P/runner.py" || die "A82 table patch"
  pin "$P/runner.py" $RUN82_SHA
  mv -f "$P/runner.py" "$S/$RT" || die "install patched runner"
  mv "$P" "$BASE/patch.done" || die "keep patch dir"
fi
pin "$FROM/src/$RT" $RUN76_SHA
if [ "$MODE" = four ]; then pin "$S/$RT" $RUN82_SHA; else pin "$S/$RT" $RUN76_SHA; fi
for RR in "$R" "$BASE/synth/root"; do
  mkdir -p "$RR/prerun_plant" "$RR/oracle_banks" "$RR/a32_banks"
  for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
           oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
           a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
    [ -f "$RR/$f" ] || cp "$B/$f" "$RR/$f" || die "mirror $f"
    cmp -s "$B/$f" "$RR/$f" || die "$RR/$f differs from the p20c binding root"
    cmp -s "$FROM/root/$f" "$RR/$f" || die "$RR/$f differs from the A76 root binding"
  done
done
if [ "$MODE" = four ]; then
  mkdir -p "$BASE/refs/a32"
  put "$SHIP/WAVE_DUO_HOLD__0000-0047.json" "$BASE/refs/a32/WAVE_DUO_HOLD__0000-0047.json"
  pin "$BASE/refs/a32/WAVE_DUO_HOLD__0000-0047.json" $A32W_SHA
fi
sed -e "s#^BASE=/root/a76p20#BASE=$BASE#" -e "s#^SRC=/root/a76p20/src#SRC=$S#" $FROM/env_a76.sh > "$BASE/env_a82.sh.new"
grep -q "^SRC=$S$" "$BASE/env_a82.sh.new" || die "env: SRC not switched to $S"
grep -q "A76_HOST_BASE=/root/a76p20" "$BASE/env_a82.sh.new" || die "env: A76_HOST_BASE is not /root/a76p20"
if [ "$MODE" = admit ]; then echo "export A82_HOST_BASE=$BASE A82_A76_ROOT=/root/a76p20/root" >> "$BASE/env_a82.sh.new"
else echo "export A82F_HOST_BASE=$BASE A82_A32_WAVE_SHA256=$A32W_SHA" >> "$BASE/env_a82.sh.new"; fi
if [ -f "$BASE/env_a82.sh" ]; then cmp -s "$BASE/env_a82.sh" "$BASE/env_a82.sh.new" || die "env differs"; rm "$BASE/env_a82.sh.new"; else mv "$BASE/env_a82.sh.new" "$BASE/env_a82.sh"; fi
put "$SHIP/run_a82.sh" "$BASE/run_a82.sh"; chmod +x "$BASE/run_a82.sh"
echo "$MODE" > "$BASE/MODE"
( source "$BASE/env_a82.sh" && cd "$S" && A82_MODE=$MODE nice -n 19 "$PY" - <<'EOF'
import os, sys
sys.path[:0] = ['ops', '.']
from tools.coopmwm import a75_holdforms as H, nm_eval_runner as R
import a82_s3b20 as MA, a82_fourarm as MB, a76_d2p20 as M76, a80_s2b as M80, a77_mappo20c as A77
assert H.a75_digest() == MA.A75_SHA256 == M76.A75_SHA256 == MB.A75_SHA256, ('a75 digest', H.a75_digest())
assert R.A75_ARMS == ('A75_PLAN', 'A75_BANK')
assert tuple(R.A76_RANGES) == ((997100, 997199), (997200, 997299), (997300, 997399))
assert M76.check_blocks_free(R.A76_RANGES, R.EVAL_HELDOUT_RANGES)
if os.environ['A82_MODE'] == 'four':
    assert tuple(R.A82_RANGES) == (MB.BLOCK,) and MB.block_free(MB.BLOCK, R.EVAL_HELDOUT_RANGES)
else:
    assert not hasattr(R, 'A82_RANGES')
raw = open('ops/a82/SYNTH_TOP_OFFER_CONDITIONS_D2800_p20.json', 'rb').read()
doc = MA.candidates_doc(raw, MA.sha_bytes(raw))
print('code identity ok: a75', H.a75_digest()[:8], 'synthetic candidates', doc['n_candidates'], 'mode', os.environ['A82_MODE'])
EOF
) || die "code identity check"
echo "tree diff vs the A76 tree (expected: runner [four], a80_s2b, a77_mappo20c, the A82 driver/test files, ops/a82):"
diff -rq $FROM/src "$S" -x __pycache__ | cut -c1-200
du -sh "$BASE" | cut -c1-80
df -BG --output=avail "$BASE" | tail -1
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
