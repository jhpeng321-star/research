#!/bin/bash
# AMENDMENT 68 host setup (Box 3; run as root; idempotent; zero SUMO; never deletes or overwrites anything).
# New final root /root/a68/root = the same mirrored binding files as the A66/A67 roots + a byte copy of the frozen A42
# BIG_BANK bank. Source tree /root/a68/src is a REAL copy of the A66 tree (never the A66 tree itself) with the two
# table-only edits; this script adds the A68 driver, the pinned settings file and the two panel files to it.
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a68
R=$BASE/root
S=$BASE/src
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
[ -L "$S" ] && die "$S is a symlink"
[ "$(readlink -f $S)" = "$S" ] || die "$S does not resolve to itself"
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks" "$R/a68_banks" "$S/ops/a68"
B=/root/autodl-fs/nm_finaleval_p20c_bank
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || cp "$B/$f" "$R/$f" || die "mirror $f"
  cmp -s "$B/$f" "$R/$f" || die "final root $f differs from the binding root"
  cmp -s /root/a66/root/$f "$R/$f" || die "final root $f differs from the A66 root"
done
BANK=a68_banks/A42_BIG_BANK_D2800_p60.json
[ -f "$R/$BANK" ] || cp /root/a66/refs/a42/root/a42_banks/A42_BIG_BANK_D2800_p60.json "$R/$BANK" || die "bank copy"
pin() { [ "$(sha256sum "$1" 2>/dev/null | cut -c1-64)" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
pin "$R/$BANK" 7e7dc76b8a0cd5c158205a0e3f6ef96d73a36a483c46d03ac422428dc8c573ff
pin /root/a67/root/a67_banks/A42_BIG_BANK_D2800_p60.json 7e7dc76b8a0cd5c158205a0e3f6ef96d73a36a483c46d03ac422428dc8c573ff
pin /root/a66/inputs/policy_00084.pt 2abed31707b61121569af2936c23689827b52b28422df162751b230aa6a93de8
pin "$R/a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json" 39f384a29fb8e952a94f918234c52bb3ff2059f2155b99f17b2f5c26a38ec239
pin "$R/oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json" fd385a2b8b1cbaf33897f8617b6d4c0abeb9f2dea40c9b72dca91d718bfc3c82
put() {  # put <shipped file> <target>: copy once; an existing target must be byte-identical
  if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi
}
put "$SHIP/a68_research60c.py" "$S/ops/a68_research60c.py"
put "$SHIP/nm_eval_trainblock_wz3to2_a68v_D2800_p60.json" "$S/ops/nm_eval_trainblock_wz3to2_a68v_D2800_p60.json"
put "$SHIP/nm_eval_panel_wz3to2_a68c_D2800_p60.json" "$S/ops/nm_eval_panel_wz3to2_a68c_D2800_p60.json"
# the settings file: the unpinned build-time version is replaced ONCE by the pinned one (kept as .unpinned)
if ! cmp -s "$SHIP/nm_finaleval.py" "$S/ops/nm_finaleval.py"; then
  [ -f "$S/ops/nm_finaleval.py.unpinned" ] && die "nm_finaleval.py differs and .unpinned already exists"
  mv "$S/ops/nm_finaleval.py" "$S/ops/nm_finaleval.py.unpinned" && cp "$SHIP/nm_finaleval.py" "$S/ops/nm_finaleval.py" || die "settings"
fi
pin "$S/ops/nm_eval_trainblock_wz3to2_a68v_D2800_p60.json" 5caece782df34dbd5f07d9da7fbeda9e2cfbf009001b1dcb7483266516211f3e
pin "$S/ops/nm_eval_panel_wz3to2_a68c_D2800_p60.json" 91f48bc86f67de6352ae9c6438983e491ece889d1e5808385dca9b641f79f209
echo "tree diff vs the A66 tree (expected: the two table files, the A68 driver/panels, .unpinned):"
diff -rq /root/a66/src_a497230b9c8a_a66 "$S" -x __pycache__ | cut -c1-200
sha256sum "$S/ops/a68_research60c.py" "$S/ops/nm_finaleval.py" "$S/pytools/tools/coopmwm/nm_eval_runner.py"
date '+%F %T' > "$BASE/SETUP_OK"
echo SETUP_OK
