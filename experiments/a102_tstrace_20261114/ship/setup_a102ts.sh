#!/bin/bash
# A102TS host setup (Box 3; root; ZERO SUMO; never deletes; writes only /root/a102ts). Runs once.
# Source tree = REAL copy (cp -a; symlinks kept as in A94) of the finished A94 tree /root/a94/src, diff-verified,
# NOT patched. Root inputs = byte copies of the A94 root binding files (cmp vs /root/a94/root AND /root/a66/root) and
# of the A94 panel file + pin (cmp vs /root/a94/root). /root/a94 is only READ.
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a102ts
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
grep -q A94_CONFIRM_DONE $FROM/chain_confirm.out || die "A94 confirm is not done"
pin $FROM/src/ops/a94_bank.py c8b05145822f8250a626036a9dddf5214b550542c8c775c847a7f693a66338ea
pin $FROM/src/ops/a81_bigconfirm.py c755a5510bc7c1c5f9f9279e7b2f40a77c388b5ccd60a35db2207b7d2dc4c4bb
pin $FROM/src/ops/a75_d2forms.py 8a2533ef58f5e4dc295808830cfb4bdc39e20c417cf98459dc96f5b7762551cf
pin $FROM/src/pytools/tools/coopmwm/nm_eval_runner.py 47fb9f76117c3862ac0f173a64891599f84d6b9a16c8b5a04ed8a33ebd2397f5
pin $FROM/src/pytools/tools/coopmwm/a75_holdforms.py 72046db6ad9a3f117105ffdf029ac993ec3d421026da6a1ca4095d5f7be620eb
pin $FROM/src/ops/a94/A94_BANK_SEARCHED_D2800_p60.json 577a0564944b2062e441d7ee603e7fe8090bede52a5dd6e8774d41ed9cb2d8b6
pin $FROM/root/a94_banks/A94_BANK_SEARCHED_D2800_p60.json 577a0564944b2062e441d7ee603e7fe8090bede52a5dd6e8774d41ed9cb2d8b6
pin $FROM/root/A94_PANEL_C.json 720d19e1df5d95c1b769170b62e234c1498c1c616fafa209c1880b5a474095ac
pin /root/a66/inputs/policy_00084.pt 2abed31707b61121569af2936c23689827b52b28422df162751b230aa6a93de8
pin $FROM/env_a94.sh 91df1f9b03cf21f9efb6b22283b6a15a25236380dd8b2cefd8ca3bc01bd70dce
grep -q "^bank 577a0564944b2062e441d7ee603e7fe8090bede52a5dd6e8774d41ed9cb2d8b6 " $FROM/PINNED_SHAS || die "A94 PINNED_SHAS bank is not the S5 bank"
PY=$(grep "^PY=" $FROM/env_a94.sh | cut -d= -f2)
[ -x "$PY" ] || die "python of the A94 env not found"
mkdir -p "$BASE"
if [ ! -d "$S" ]; then cp -a $FROM/src "$S" || die "copy tree"; fi
[ -L "$S" ] && die "$S is a symlink"
diff -rq --no-dereference $FROM/src "$S" -x __pycache__ > "$BASE/tree_diff.txt"
[ -s "$BASE/tree_diff.txt" ] && die "copied tree differs from /root/a94/src"
echo "tree: byte copy of /root/a94/src ($(find "$S" -type f -not -path '*__pycache__*' | wc -l) files, diff -rq empty)"
for f in a102_ts.py a102ts_check.py run_a102ts.sh; do put "$SHIP/$f" "$BASE/$f"; done
chmod +x "$BASE/run_a102ts.sh"
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || cp "$FROM/root/$f" "$R/$f" || die "copy $f"
  cmp -s "/root/a66/root/$f" "$R/$f" || die "$R/$f differs from the A66 root"
  cmp -s "$FROM/root/$f" "$R/$f" || die "$R/$f differs from the A94 root"
done
for f in A94_PANEL_C.json A94_PANEL_C.sha256; do
  [ -f "$R/$f" ] || cp "$FROM/root/$f" "$R/$f" || die "copy $f"
  cmp -s "$FROM/root/$f" "$R/$f" || die "$R/$f differs from the A94 root"
done
grep -q "panel_sha256 d85999ddaaf3eb583775d66dbabca168b4432966bc01e7193dfc79f244f7946c$" "$R/A94_PANEL_C.sha256" || die "panel pin is not d85999dd"
sed -e "s#^BASE=/root/a94#BASE=$BASE#" -e "s#^SRC=/root/a94/src#SRC=$S#" -e "/A94_HOST_BASE/d" \
    $FROM/env_a94.sh > "$BASE/env_a102ts.sh.new"
echo "export A94_HOST_BASE=$BASE" >> "$BASE/env_a102ts.sh.new"
grep -q "^SRC=$S$" "$BASE/env_a102ts.sh.new" || die "env: SRC not switched"
[ "$(grep -c '/root/a94' "$BASE/env_a102ts.sh.new")" = 0 ] || die "env: a /root/a94 path remains"
if [ -f "$BASE/env_a102ts.sh" ]; then cmp -s "$BASE/env_a102ts.sh" "$BASE/env_a102ts.sh.new" || die "env differs"; else cp "$BASE/env_a102ts.sh.new" "$BASE/env_a102ts.sh"; fi
du -sh "$BASE" | cut -c1-80
df -BG --output=avail "$BASE" | tail -1
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
