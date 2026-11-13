#!/bin/bash
# AMENDMENT 75 stage D1 host setup (Box 4; run as root; zero SUMO; never deletes; writes only /root/a75; disk-frugal).
# /root/a75/src  = HARD-LINK clone (cp -al, no data blocks) of /root/a68/src (the tree that produced the A68 admission
#                  and its MAPPO60C reference) + the A75 files as new small files. NO existing file is changed: runner,
#                  executor, settings and the A68 driver keep the A68 bytes (checked below with diff -rq).
# /root/a75/root = the same binding files as the A68 root (hard links, compared with the A66 root).
# /root/a75/ref68 = the 24 per-vehicle reports of the stored A68 MAPPO60C admission wave (pinned in REF_PINS.json).
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a75
FROM=/root/a68
S=$BASE/src
R=$BASE/root
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
[ -L $FROM/src ] && die "$FROM/src is a symlink"
[ -f $FROM/SETUP_OK ] || die "$FROM is not a completed A68 setup"
[ -f $FROM/root/A68_ADMISSION_D2800_p60.json ] || die "$FROM/root has no A68 admission receipt"
mkdir -p "$BASE"
if [ ! -d "$S" ]; then cp -al $FROM/src "$S" || die "link tree"; fi
[ -L "$S" ] && die "$S is a symlink"
[ "$(readlink -f $S)" = "$S" ] || die "$S does not resolve to itself"
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
PY=$(grep "^PY=" $FROM/env_a68.sh | cut -d= -f2)
[ -x "$PY" ] || die "python of the A68 env not found"
mkdir -p "$S/ops/a75d1"
[ -e "$S/ops/a75_directhold.py" ] && ! cmp -s "$SHIP/a75_directhold.py" "$S/ops/a75_directhold.py" && die "driver differs"
put "$SHIP/a75_directhold.py" "$S/ops/a75_directhold.py"
put "$SHIP/test_a75_directhold.py" "$S/ops/test_a75_directhold.py"
for f in CANDIDATES_D1_D2800_p60.json CANDIDATES_D1_D2800_p60.sha256 REF_PINS.json; do put "$SHIP/$f" "$S/ops/a75d1/$f"; done
pin "$S/ops/a75d1/CANDIDATES_D1_D2800_p60.json" "$(cat "$SHIP/CANDIDATES_FILE.sha256")"
pin "$S/ops/a75d1/REF_PINS.json" "$(cat "$SHIP/REF_PINS_FILE.sha256")"
pin "$S/ops/nm_eval_trainblock_wz3to2_a68v_D2800_p60.json" 5caece782df34dbd5f07d9da7fbeda9e2cfbf009001b1dcb7483266516211f3e
mkdir -p "$BASE/panels"
for f in $FROM/panels/*.json; do [ -f "$BASE/panels/$(basename "$f")" ] || ln "$f" "$BASE/panels/$(basename "$f")" || die "panel link"; done
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || ln "$FROM/root/$f" "$R/$f" || die "mirror $f"
  cmp -s "/root/a66/root/$f" "$R/$f" || die "$R/$f differs from the A66 root"
done
pin /root/a66/inputs/policy_00084.pt 2abed31707b61121569af2936c23689827b52b28422df162751b230aa6a93de8
# the stored reference reports (new files; an existing one must be byte-identical)
X=$BASE/ref68
mkdir -p "$X" "$BASE/ref68.tmp"
tar -xf "$SHIP/ref68_reports.tar" -C "$BASE/ref68.tmp" || die "reference reports"
( cd "$BASE/ref68.tmp" && find evaluation -type f | sort ) > "$BASE/ref68.list"
[ "$(wc -l < "$BASE/ref68.list")" = 24 ] || die "reference reports: not 24 files"
while read -r f; do mkdir -p "$X/$(dirname "$f")"; put "$BASE/ref68.tmp/$f" "$X/$f"; done < "$BASE/ref68.list"
sed -e "s#^BASE=/root/a68#BASE=/root/a75#" -e "s#^SRC=/root/a68/src#SRC=/root/a75/src#" $FROM/env_a68.sh > "$BASE/env_a75.sh.new"
echo "export A75_HOST_BASE=/root/a75" >> "$BASE/env_a75.sh.new"
grep -q "A68_HOST_BASE=/root/a68" "$BASE/env_a75.sh.new" || die "env: A68_HOST_BASE is not /root/a68"
if [ -f "$BASE/env_a75.sh" ]; then cmp -s "$BASE/env_a75.sh" "$BASE/env_a75.sh.new" || die "env differs"; else cp "$BASE/env_a75.sh.new" "$BASE/env_a75.sh"; fi
put "$SHIP/run_a75.sh" "$BASE/run_a75.sh"; chmod +x "$BASE/run_a75.sh"
echo "tree diff vs the A68 tree (expected: only the A75 driver, its test and ops/a75d1):"
diff -rq $FROM/src "$S" -x __pycache__ | cut -c1-200
echo "runner $(sha "$S/pytools/tools/coopmwm/nm_eval_runner.py")  executor $(sha "$S/pytools/tools/coopmwm/oracle_plan.py")"
echo "driver $(sha "$S/ops/a75_directhold.py")  candidates $(sha "$S/ops/a75d1/CANDIDATES_D1_D2800_p60.json")"
du -sh "$BASE" | cut -c1-80
df -BG --output=avail "$BASE" | tail -1
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
