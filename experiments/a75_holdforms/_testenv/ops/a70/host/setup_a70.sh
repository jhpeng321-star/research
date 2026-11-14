#!/bin/bash
# AMENDMENT 70 host setup (Box 3; run as root; zero SUMO; never deletes; never writes /root/a68, /root/a66, /root/a67).
# /root/a70/src = REAL copy of /root/a68/src + the table-only A70 edits (patch_tables.py) + the A70 driver.
# Root /root/a70/root = the same mirrored binding files as the A68 root + a byte copy of the frozen A42 BIG_BANK bank
# (identity only).   usage: setup_a70.sh tree | pin <panel_sha256>
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a70
S=$BASE/src
R=$BASE/root
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
pin() { [ "$(sha256sum "$1" 2>/dev/null | cut -c1-64)" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
PY=/root/a66/venv/bin/python
case "${1:?tree | pin}" in
tree)
  [ -L /root/a68/src ] && die "/root/a68/src is a symlink"
  mkdir -p "$BASE/panels"
  if [ ! -d "$S" ]; then cp -a /root/a68/src "$S" || die "copy tree"; fi
  [ -L "$S" ] && die "$S is a symlink"
  [ "$(readlink -f $S)" = "$S" ] || die "$S does not resolve to itself"
  mkdir -p "$S/ops/a70"
  "$PY" "$SHIP/patch_tables.py" tables "$S/pytools/tools/coopmwm/nm_eval_runner.py" "$S/ops/nm_finaleval.py" || die "patch"
  put "$SHIP/a70_smallbank.py" "$S/ops/a70_smallbank.py"
  put "$SHIP/patch_tables.py" "$S/ops/a70/patch_tables.py"
  mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks" "$R/a70_banks"
  B=/root/autodl-fs/nm_finaleval_p20c_bank
  for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
           oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
           a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
    [ -f "$R/$f" ] || cp "$B/$f" "$R/$f" || die "mirror $f"
    cmp -s "$B/$f" "$R/$f" || die "$R/$f differs from the binding root"
    cmp -s /root/a68/root/$f "$R/$f" || die "$R/$f differs from the A68 root"
  done
  BANK=a70_banks/A42_BIG_BANK_D2800_p60.json
  [ -f "$R/$BANK" ] || cp /root/a68/root/a68_banks/A42_BIG_BANK_D2800_p60.json "$R/$BANK" || die "bank copy"
  pin "$R/$BANK" 7e7dc76b8a0cd5c158205a0e3f6ef96d73a36a483c46d03ac422428dc8c573ff
  pin /root/a66/inputs/policy_00084.pt 2abed31707b61121569af2936c23689827b52b28422df162751b230aa6a93de8
  pin "$R/a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json" 39f384a29fb8e952a94f918234c52bb3ff2059f2155b99f17b2f5c26a38ec239
  sed -e 's#^BASE=/root/a68#BASE=/root/a70#' -e 's#^SRC=/root/a68/src#SRC=/root/a70/src#' /root/a68/env_a68.sh > "$BASE/env_a70.sh.new"
  echo 'export A70_HOST_BASE=/root/a70 A71_HOST_BASE=/root/a71' >> "$BASE/env_a70.sh.new"
  if [ -f "$BASE/env_a70.sh" ]; then cmp -s "$BASE/env_a70.sh" "$BASE/env_a70.sh.new" || die "env differs"; rm "$BASE/env_a70.sh.new"; else mv "$BASE/env_a70.sh.new" "$BASE/env_a70.sh"; fi
  put "$SHIP/run_a70.sh" "$BASE/run_a70.sh"
  echo "tree diff vs the A68 tree:"
  diff -rq /root/a68/src "$S" -x __pycache__ | cut -c1-200
  diff /root/a68/src/pytools/tools/coopmwm/nm_eval_runner.py "$S/pytools/tools/coopmwm/nm_eval_runner.py"
  diff /root/a68/src/ops/nm_finaleval.py "$S/ops/nm_finaleval.py" | cut -c1-160
  date '+%F %T' > "$BASE/SETUP_TREE_OK"; echo SETUP_TREE_OK ;;
pin)
  SHA=${2:?panel sha}
  P=$S/ops/nm_eval_panel_wz3to2_a70c_D2800_p60.json
  put "$BASE/panels/nm_eval_panel_wz3to2_a70c_D2800_p60.json" "$P"
  [ -f "$S/ops/nm_finaleval.py.a70unpinned" ] || cp "$S/ops/nm_finaleval.py" "$S/ops/nm_finaleval.py.a70unpinned"
  "$PY" "$SHIP/patch_tables.py" pin "$S/ops/nm_finaleval.py" "$S/ops/a70_smallbank.py" "$SHA" || die "pin patch"
  sha256sum "$P" "$S/ops/a70_smallbank.py" "$S/ops/nm_finaleval.py" "$S/pytools/tools/coopmwm/nm_eval_runner.py"
  date '+%F %T' > "$BASE/SETUP_OK"; echo SETUP_OK ;;
*) die "unknown mode" ;;
esac
