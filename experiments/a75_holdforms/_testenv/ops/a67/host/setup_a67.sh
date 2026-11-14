#!/bin/bash
# AMENDMENT 67 host setup (Box 3; run as root; idempotent; zero SUMO; never deletes or overwrites anything).
# New final root /root/a67/root = the same mirrored binding files as the A66 root + a byte copy of the frozen A42
# BIG_BANK bank; the A67 driver is added to the A66 source tree (no existing file is changed).
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a67
R=$BASE/root
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks" "$R/a67_banks"
B=/root/autodl-fs/nm_finaleval_p20c_bank
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || cp "$B/$f" "$R/$f" || die "mirror $f"
  cmp -s "$B/$f" "$R/$f" || die "final root $f differs from the binding root"
  cmp -s /root/a66/root/$f "$R/$f" || die "final root $f differs from the A66 root"
done
BANK=a67_banks/A42_BIG_BANK_D2800_p60.json
[ -f "$R/$BANK" ] || cp /root/a66/refs/a42/root/a42_banks/A42_BIG_BANK_D2800_p60.json "$R/$BANK" || die "bank copy"
pin() { [ "$(sha256sum "$1" 2>/dev/null | cut -c1-64)" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
pin "$R/$BANK" 7e7dc76b8a0cd5c158205a0e3f6ef96d73a36a483c46d03ac422428dc8c573ff
pin /root/a66/refs/a42/root/ADMISSION_D2800_p60.json 5ed0317ce39f6c6bb7a6756695fb50a64bf96513b17768a49ab58489a18f0cf9
pin /root/a66/inputs/policy_00084.pt 2abed31707b61121569af2936c23689827b52b28422df162751b230aa6a93de8
pin "$R/a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json" 39f384a29fb8e952a94f918234c52bb3ff2059f2155b99f17b2f5c26a38ec239
pin "$R/oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json" fd385a2b8b1cbaf33897f8617b6d4c0abeb9f2dea40c9b72dca91d718bfc3c82
D=/root/a66/src/ops/a67_plans_on_60c.py
if [ -f "$D" ]; then cmp -s "$SHIP/a67_plans_on_60c.py" "$D" || die "$D exists with different content"; else cp "$SHIP/a67_plans_on_60c.py" "$D" || die "driver copy"; fi
sha256sum "$D"
date '+%F %T' > "$BASE/SETUP_OK"
echo SETUP_OK
