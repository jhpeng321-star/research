#!/bin/bash
# AMENDMENT 69 host setup (Box 3; run as root; zero SUMO; never deletes; never writes /root/a68, /root/a66, /root/a67).
# /root/a69/src = REAL copy of /root/a68/src + the A44 files (variant, merged runner, mechanics driver) + the A69 driver.
# Roots /root/a69/a44m/root (A44 mechanics) and /root/a69/root (A69) = the same mirrored binding files as the A68 root.
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a69
S=$BASE/src
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
[ -L /root/a68/src ] && die "/root/a68/src is a symlink"
mkdir -p "$BASE"
if [ ! -d "$S" ]; then cp -a /root/a68/src "$S" || die "copy tree"; fi
[ -L "$S" ] && die "$S is a symlink"
[ "$(readlink -f $S)" = "$S" ] || die "$S does not resolve to itself"
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
pin() { [ "$(sha256sum "$1" 2>/dev/null | cut -c1-64)" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
T=$S/pytools/tools/coopmwm
mkdir -p "$S/ops/a44" "$S/ops/a69"
if ! cmp -s "$SHIP/nm_eval_runner.py" "$T/nm_eval_runner.py"; then
  [ -f "$T/nm_eval_runner.py.a68" ] && die "runner differs and .a68 already exists"
  cmp -s /root/a68/src/pytools/tools/coopmwm/nm_eval_runner.py "$T/nm_eval_runner.py" || die "runner is not the A68 runner"
  mv "$T/nm_eval_runner.py" "$T/nm_eval_runner.py.a68" && cp "$SHIP/nm_eval_runner.py" "$T/nm_eval_runner.py" || die "runner"
fi
put "$SHIP/a44_lcvariant.py" "$T/a44_lcvariant.py"
put "$SHIP/a44_lcmech.py" "$S/ops/a44_lcmech.py"
put "$SHIP/a69_lcside.py" "$S/ops/a69_lcside.py"
put "$SHIP/vlcmp.py" "$S/ops/a69/vlcmp.py"
for f in FIXTURE_PLANS.json SEED_AUDIT_RAW.json nm_eval_trainblock_wz3to2_a44m_D2800_p60.json seed_audit.py; do put "$SHIP/a44/$f" "$S/ops/a44/$f"; done
pin "$T/a44_lcvariant.py" 4e7a97f3183290ca5f1ad161cb3557b27d9b3695eb596a18f3ee3097e92aac0c
pin "$T/nm_eval_runner.py" 62496e864567700a051aaf3d26ee16fb64e1dbe5b0a34b7d28688e0defb67836
pin "$S/ops/a44/nm_eval_trainblock_wz3to2_a44m_D2800_p60.json" a579283bb0f7a4039715f80df9c9a4c7636a9261702e96fc6773555a18ffca8b
B=/root/autodl-fs/nm_finaleval_p20c_bank
for R in $BASE/a44m/root $BASE/root; do
  mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks"
  for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
           oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
           a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
    [ -f "$R/$f" ] || cp "$B/$f" "$R/$f" || die "mirror $f"
    cmp -s "$B/$f" "$R/$f" || die "$R/$f differs from the binding root"
    cmp -s /root/a68/root/$f "$R/$f" || die "$R/$f differs from the A68 root"
  done
done
sed -e 's#^BASE=/root/a68#BASE=/root/a69#' -e 's#^SRC=/root/a68/src#SRC=/root/a69/src#' /root/a68/env_a68.sh > "$BASE/env_a69.sh.new"
echo 'export A44_HOST_BASE=/root/a69/a44m A69_HOST_BASE=/root/a69' >> "$BASE/env_a69.sh.new"
if [ -f "$BASE/env_a69.sh" ]; then cmp -s "$BASE/env_a69.sh" "$BASE/env_a69.sh.new" || die "env differs"; rm "$BASE/env_a69.sh.new"; else mv "$BASE/env_a69.sh.new" "$BASE/env_a69.sh"; fi
put "$SHIP/run_a69.sh" "$BASE/run_a69.sh"; chmod +x "$BASE/run_a69.sh"
echo "tree diff vs the A68 tree:"
diff -rq /root/a68/src "$S" -x __pycache__ | cut -c1-200
grep -n "a68" "$BASE/env_a69.sh" | cut -c1-200
date '+%F %T' > "$BASE/SETUP_OK"
echo SETUP_OK
