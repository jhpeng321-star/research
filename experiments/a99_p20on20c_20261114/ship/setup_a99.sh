#!/bin/bash
# AMENDMENT 99 host setup (zero SUMO; never deletes; writes only $A99_BASE). $BASE/src = REAL COPY (cp -a) of
# /root/a96adm_s5/src (the A96(b) tree incl. the 11-14 nm_finaleval cache fix fa98b552), verified file by file;
# + ops/a99_on20c.py; $BASE/root = copies of the 8 p20c binding files (cmp with the p20c bank root and /root/a96adm_s5/root).
#   A99_BASE=/root/a99ident|/root/a99adm_s4|/root/a99adm_s5|/root/a99adm_hand bash setup_a99.sh
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=${A99_BASE:?A99_BASE}
case "$BASE" in /root/a99ident|/root/a99adm_s4|/root/a99adm_s5|/root/a99adm_hand) ;; *) echo "A99_BASE not an A99 base"; exit 2;; esac
FROM=/root/a96adm_s5
S=$BASE/src
R=$BASE/root
B=/root/autodl-fs/nm_finaleval_p20c_bank
die() { echo "SETUP FAILED: $*"; exit 1; }
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
[ "$(id -u)" = 0 ] || die "must run as root"
[ -e "$BASE" ] && die "$BASE exists (new roots only)"
pin "$FROM/src/ops/nm_finaleval.py" fa98b5526296f67e96c7670d7baedf4e943df49074822daa9f6fdd73a69fc132
pin "$FROM/src/ops/a96_p20adm.py" 85cdef84094625cff8e5a9c7f99cf99eb38e586e1ea1b6f4514de98cad8ffdff
pin "$FROM/src/pytools/tools/coopmwm/nm_eval_runner.py" 0c5c2a4b53052ae25103e9b8ac7b895b86b590de4e4d94b77d2110d9065fa8c8
pin "$SHIP/a99_on20c.py" "$(cat "$SHIP/a99_on20c.sha256" | cut -c1-64)"
mkdir -p "$BASE"
nice -n 19 ionice -c3 cp -a "$FROM/src" "$S" || die "copy tree"
[ -L "$S" ] && die "$S is a symlink"
( cd "$FROM/src" && find . -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum ) > "$BASE/a96s5_tree.sha256"
( cd "$S" && sha256sum -c --quiet "$BASE/a96s5_tree.sha256" ) || die "the copy differs from /root/a96adm_s5/src"
echo "tree copy ok: $(wc -l < "$BASE/a96s5_tree.sha256") files identical to /root/a96adm_s5/src"
cp "$SHIP/a99_on20c.py" "$S/ops/a99_on20c.py" || die "driver"
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  cp "$B/$f" "$R/$f" || die "mirror $f"
  cmp -s "$B/$f" "$R/$f" || die "$R/$f differs from the p20c binding root"
  cmp -s "$FROM/root/$f" "$R/$f" || die "$R/$f differs from the a96adm_s5 root binding"
done
sed -e "s#/root/a96adm_s5#$BASE#g" "$FROM/env_a96.sh" > "$BASE/env_a99.sh"
grep -q "^SRC=$S$" "$BASE/env_a99.sh" || die "env: SRC not switched"
grep -q a96adm "$BASE/env_a99.sh" && die "env still names a96adm"
cp "$SHIP/run_a99.sh" "$BASE/run_a99.sh" && chmod +x "$BASE/run_a99.sh"
echo "tree diff vs /root/a96adm_s5/src (expected: ops/a99_on20c.py only):"
diff -rq "$FROM/src" "$S" -x __pycache__ | cut -c1-200
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
