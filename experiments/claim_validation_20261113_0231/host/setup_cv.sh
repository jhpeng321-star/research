#!/bin/bash
# Claim validation 20261113_0231 host setup (Box 3; run as root; idempotent; zero SUMO; never deletes or overwrites).
# Pattern of ops/a68/host/setup_a68.sh: the final root gets the SAME mirrored binding files as the A66 / A68 roots
# (U0 = frozen base MAPPO binding, oracle binding, pre-run plant). Only /root/cv_20261113_0231 is written.
#   setup_cv.sh [ROOT]      default ROOT = /root/cv_20261113_0231/root   (a scratch root is used for dry builds)
set -uo pipefail
BASE=/root/cv_20261113_0231
R=${1:-$BASE/root}
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
case "$R" in $BASE/*) ;; *) die "root must be under $BASE";; esac
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks" "$R/cv_banks"
B=/root/autodl-fs/nm_finaleval_p20c_bank
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || cp "$B/$f" "$R/$f" || die "mirror $f"
  cmp -s "$B/$f" "$R/$f" || die "final root $f differs from the binding root"
done
pin() { [ "$(sha256sum "$1" 2>/dev/null | cut -c1-64)" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
pin "$BASE/inputs/A42_BIG_BANK_D2800_p60.json" 7e7dc76b8a0cd5c158205a0e3f6ef96d73a36a483c46d03ac422428dc8c573ff
pin "$R/a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json" 39f384a29fb8e952a94f918234c52bb3ff2059f2155b99f17b2f5c26a38ec239
pin "$R/oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json" fd385a2b8b1cbaf33897f8617b6d4c0abeb9f2dea40c9b72dca91d718bfc3c82
sha256sum "$R/BOUND_U0.json" "$R/ORACLE_BINDING.json" "$R/PRERUN_PLAN.json"
grep -o '"[a-z_]*sha[a-z0-9_]*": "[0-9a-f]\{64\}"' "$R/BOUND_U0.json" | head -6
echo SETUP_OK "$R"
