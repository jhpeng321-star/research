#!/bin/bash
# AMENDMENT 75 stage D1, CONFIRM pieces (Box 4; run as root; zero SUMO; never deletes; writes only /root/a75).
# /root/a75/src (identity + admission tree) is NOT changed. The confirmation needs the A70 panel tables, so:
# /root/a75/src_c = HARD-LINK clone of /root/a75/src in which the runner and the settings file are REPLACED by patched
#                   NEW files (the registered A70 table-only edit, ops/a70/host/patch_tables.py; the linked originals
#                   keep their bytes, checked) + the registered A70 driver + the A70 panel file.
# /root/a75/a70ref/root = the stored A70 wave receipts of MAPPO60C and DUO_HOLD_60C + the A70 selection receipt (pinned)
# /root/a75/ref70 = their 2 x 96 per-vehicle reports (pinned in REF_PINS.json)
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a75
S=$BASE/src
C=$BASE/src_c
R=$BASE/root
PANEL_SHA=1b4b59ee4ab717b931647fbae7cfa489726cd2cbacad30e268c57e32cb2c78e1
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
[ -f $BASE/SETUP_OK ] || die "$BASE is not a completed A75 setup"
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
PY=$(grep "^PY=" $BASE/env_a75.sh | cut -d= -f2)
[ -x "$PY" ] || die "python not found"
RUN=pytools/tools/coopmwm/nm_eval_runner.py
SET=ops/nm_finaleval.py
RUN0=$(sha "$S/$RUN"); SET0=$(sha "$S/$SET")
pin "$SHIP/a70_smallbank.py" 72af622e0c2188d8616da6a45faf3b25ff80bd27feced7467c66c5a5ae716da0
pin "$SHIP/patch_tables.py" d0e75a032c863e5d028a4d0391d8e36aafafa288e25305e4e71ac70c900d1241
pin "$SHIP/nm_eval_panel_wz3to2_a70c_D2800_p60.json" 835e423adf8fb9bb2cee98c645a845b43f23ce6ec073a5c6bf70db65a045a1d4
if [ ! -d "$C" ]; then cp -al "$S" "$C" || die "link tree"; fi
[ -L "$C" ] && die "$C is a symlink"
if ! grep -q "^A70_RANGES" "$C/$RUN"; then
  [ -e "$BASE/patch.tmp" ] && die "$BASE/patch.tmp exists"
  mkdir "$BASE/patch.tmp" && cp "$C/$RUN" "$BASE/patch.tmp/runner.py" && cp "$C/$SET" "$BASE/patch.tmp/settings.py" || die "copies"
  "$PY" "$SHIP/patch_tables.py" tables "$BASE/patch.tmp/runner.py" "$BASE/patch.tmp/settings.py" || die "table patch"
  "$PY" "$SHIP/patch_tables.py" pin "$BASE/patch.tmp/settings.py" "$SHIP/a70_smallbank.py" $PANEL_SHA || die "pin patch"
  # mv replaces the directory entry (new inode): the files linked from /root/a75/src and /root/a68/src are untouched
  mv "$BASE/patch.tmp/runner.py" "$C/$RUN" && mv "$BASE/patch.tmp/settings.py" "$C/$SET" || die "install patched files"
fi
pin "$S/$RUN" "$RUN0"; pin "$S/$SET" "$SET0"
pin /root/a68/src/$RUN "$RUN0"; pin /root/a68/src/$SET "$SET0"
pin "$SHIP/a70_smallbank.py" 72af622e0c2188d8616da6a45faf3b25ff80bd27feced7467c66c5a5ae716da0
put "$SHIP/a70_smallbank.py" "$C/ops/a70_smallbank.py"
put "$SHIP/nm_eval_panel_wz3to2_a70c_D2800_p60.json" "$C/ops/nm_eval_panel_wz3to2_a70c_D2800_p60.json"
[ -e "$S/ops/a70_smallbank.py" ] && die "the admission tree got the A70 driver"
N1=$(diff "$S/$RUN" "$C/$RUN" | grep -c "^[<>]"); N2=$(diff "$S/$SET" "$C/$SET" | grep -c "^<")
echo "runner diff lines $N1 (table only), settings removed lines $N2"
[ "$N2" = 0 ] || die "the settings patch removed lines"
diff "$S/$RUN" "$C/$RUN"
A=$BASE/a70ref/root
mkdir -p "$A/prerun_plant" "$A/oracle_banks" "$A/a32_banks"
for f in A70_SELECTION_D2800_p60.json WAVE_A70C_MAPPO60C__0000-0095.json WAVE_A70C_DUO_HOLD_60C__0000-0095.json \
         RESERVED_A70C_MAPPO60C__0000-0095.json RESERVED_A70C_DUO_HOLD_60C__0000-0095.json; do put "$SHIP/a70root/$f" "$A/$f"; done
pin "$A/A70_SELECTION_D2800_p60.json" ba50b7c5590603abad6eceb8602f9f06563fd1f4e1a4f1b3763410a7672a8404
pin "$A/WAVE_A70C_MAPPO60C__0000-0095.json" 7aa9c5dc4fe473f60c248bc6f79f056661182cd854ba8f9e8915de8c200991a0
pin "$A/WAVE_A70C_DUO_HOLD_60C__0000-0095.json" a8eb833605870358481f676895512db9e591b8d3c5137cdeb729860b39f51d15
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$A/$f" ] || ln "$R/$f" "$A/$f" || die "mirror $f"
done
X=$BASE/ref70
mkdir -p "$X" "$BASE/ref70.tmp"
tar -xf "$SHIP/ref70_reports.tar" -C "$BASE/ref70.tmp" || die "reference reports"
( cd "$BASE/ref70.tmp" && find evaluation -type f | sort ) > "$BASE/ref70.list"
[ "$(wc -l < "$BASE/ref70.list")" = 192 ] || die "reference reports: not 192 files"
while read -r f; do mkdir -p "$X/$(dirname "$f")"; put "$BASE/ref70.tmp/$f" "$X/$f"; done < "$BASE/ref70.list"
sed -e "s#^SRC=/root/a75/src\$#SRC=/root/a75/src_c#" $BASE/env_a75.sh > "$BASE/env_a75c.sh.new"
grep -q "^SRC=/root/a75/src_c$" "$BASE/env_a75c.sh.new" || die "env: SRC not switched"
if [ -f "$BASE/env_a75c.sh" ]; then cmp -s "$BASE/env_a75c.sh" "$BASE/env_a75c.sh.new" || die "env differs"; else cp "$BASE/env_a75c.sh.new" "$BASE/env_a75c.sh"; fi
put "$SHIP/run_a75_confirm.sh" "$BASE/run_a75_confirm.sh"; chmod +x "$BASE/run_a75_confirm.sh"
put "$SHIP/check_refs70.py" "$C/ops/a75d1/check_refs70.py"
echo "tree diff admission tree vs confirm tree:"
diff -rq "$S" "$C" -x __pycache__ | cut -c1-200
( source "$BASE/env_a75c.sh" && "$PY" ops/a75d1/check_refs70.py "$A" /root/a66/root "$X" ) || die "A70 reference binding check"
echo "runner_c $(sha "$C/$RUN")  settings_c $(sha "$C/$SET")"
du -sh "$BASE" | cut -c1-80
date "+%F %T" > "$BASE/CONFIRM_TREE_OK"
echo CONFIRM_TREE_OK
