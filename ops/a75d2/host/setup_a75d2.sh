#!/bin/bash
# AMENDMENT 75 stage D2 host setup (run as root; zero SUMO; never deletes; writes only /root/a75d2; disk-frugal).
# usage: A75D2_MECH_SEED0=<block start> bash setup_a75d2.sh [ref68_reports.tar]
# /root/a75d2/src  = HARD-LINK clone (cp -al) of /root/a68/src + the A75 executor / D2 files as NEW small files.
#                    The runner is the ONLY changed file: copied to a new inode first (cp + mv, so the A68 inode and
#                    bytes are never touched), then patched additively (A75_PLAN / A75_BANK arms + A75D2_MECH_RANGE).
# /root/a75d2/root = the A68 root binding files (hard links, compared with the A66 root).
# /root/a75d2/ref68 = the 24 per-vehicle reports of the stored A68 MAPPO60C admission wave (pinned in REF_PINS.json):
#                    from the tar argument, else hard links of the D1 install /root/a75/ref68 (read-only).
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a75d2
FROM=/root/a68
S=$BASE/src
R=$BASE/root
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
[ -f "$BASE/SETUP_OK" ] && die "$BASE/SETUP_OK exists (setup runs once)"
TAR=${1:-}
S0=${A75D2_MECH_SEED0:-}
[ -n "$S0" ] || die "A75D2_MECH_SEED0 unset: the mechanics seed block is assigned by the coordinator at registration"
[[ "$S0" =~ ^9[0-9]{5}$ ]] && [ $((S0 % 100)) = 0 ] || die "A75D2_MECH_SEED0=$S0 is not a 100-seed block start"
for b in "964000 964199" "962000 962999" "977000 977999" "989600 989699"; do
  set -- $b; [ "$S0" -le "$2" ] && [ $((S0+99)) -ge "$1" ] && die "A75D2_MECH_SEED0=$S0 overlaps forbidden $1-$2"
done
[ -L $FROM/src ] && die "$FROM/src is a symlink"
[ -f $FROM/SETUP_OK ] || die "$FROM is not a completed A68 setup"
[ -f $FROM/root/A68_ADMISSION_D2800_p60.json ] || die "$FROM/root has no A68 admission receipt"
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
# the shipped files are the registered ones
( cd "$SHIP" && sha256sum -c --quiet SHIP_FILES.sha256 ) || die "shipped files differ from SHIP_FILES.sha256"
PY=$(grep "^PY=" $FROM/env_a68.sh | cut -d= -f2)
[ -x "$PY" ] || die "python of the A68 env not found"
mkdir -p "$BASE"
RUN68="$FROM/src/pytools/tools/coopmwm/nm_eval_runner.py"
RUN68_SHA=$(sha "$RUN68")
if [ ! -d "$S" ]; then cp -al $FROM/src "$S" || die "link tree"; fi
[ -L "$S" ] && die "$S is a symlink"
[ "$(readlink -f $S)" = "$S" ] || die "$S does not resolve to itself"
T=$S/pytools/tools/coopmwm
[ -f "$T/a44_lcvariant.py" ] || die "the A68 tree has no a44_lcvariant.py (A75 executor base)"
mkdir -p "$S/ops/a75d2" "$S/ops/a75"
put "$SHIP/a75_holdforms.py" "$T/a75_holdforms.py"
put "$SHIP/a75_specs.py" "$S/ops/a75/a75_specs.py"
put "$SHIP/a75_d2forms.py" "$S/ops/a75_d2forms.py"
put "$SHIP/test_a75_d2forms.py" "$S/ops/test_a75_d2forms.py"
for f in CANDIDATES_D2_D2800_p60.json CANDIDATES_D2_D2800_p60.csv CANDIDATES_D2_D2800_p60.sha256 REF_PINS.json EXAMPLE_PLANS.json; do
  put "$SHIP/$f" "$S/ops/a75d2/$f"
done
mkdir -p "$S/ops/a75d2/host"
put "$SHIP/patch_tables_d2.py" "$S/ops/a75d2/host/patch_tables_d2.py"
# runner: new inode, additive patch, then rename over the link (the A68 file keeps its inode and bytes)
RUN=$T/nm_eval_runner.py
if ! grep -q "^A75D2_MECH_RANGE = ($S0, $((S0+99)))$" "$RUN"; then
  [ "$(stat -c %i "$RUN")" = "$(stat -c %i "$RUN68")" ] || cmp -s "$RUN" "$RUN68" || die "$RUN is neither the A68 runner nor patched"
  cp "$RUN" "$RUN.a75d2.new" || die "runner copy"
  "$PY" -I "$SHIP/patch_runner_a75.py" arms "$RUN.a75d2.new" || die "A75 arms patch"
  "$PY" -I "$SHIP/patch_tables_d2.py" tables "$RUN.a75d2.new" "$S0" || die "D2 table patch"
  mv -f "$RUN.a75d2.new" "$RUN" || die "runner rename"
fi
[ "$(sha "$RUN68")" = "$RUN68_SHA" ] || die "the A68 runner bytes changed (must never happen)"
[ "$(stat -c %i "$RUN")" != "$(stat -c %i "$RUN68")" ] || die "the D2 runner still shares the A68 inode"
echo "runner A68 $RUN68_SHA (unchanged)  D2 $(sha "$RUN")"
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
X=$BASE/ref68
if [ ! -d "$X/evaluation" ]; then
  if [ -n "$TAR" ]; then
    mkdir -p "$BASE/ref68.tmp" && tar -xf "$TAR" -C "$BASE/ref68.tmp" || die "reference reports tar"
    ( cd "$BASE/ref68.tmp" && find evaluation -type f | sort ) > "$BASE/ref68.list"
    while read -r f; do mkdir -p "$X/$(dirname "$f")"; put "$BASE/ref68.tmp/$f" "$X/$f"; done < "$BASE/ref68.list"
  elif [ -d /root/a75/ref68/evaluation ]; then
    mkdir -p "$X" && cp -al /root/a75/ref68/evaluation "$X/evaluation" || die "link D1 reference reports"
  else
    die "no reference reports: pass ref68_reports.tar or install D1 (/root/a75/ref68) first"
  fi
fi
[ "$(find "$X/evaluation" -type f -name vehicles_report.json | wc -l)" = 24 ] || die "reference reports: not 24 files"
sed -e "s#^BASE=/root/a68#BASE=/root/a75d2#" -e "s#^SRC=/root/a68/src#SRC=/root/a75d2/src#" $FROM/env_a68.sh > "$BASE/env_a75d2.sh.new"
echo "export A75D2_HOST_BASE=/root/a75d2" >> "$BASE/env_a75d2.sh.new"
echo "export A75D2_MECH_SEED0=$S0" >> "$BASE/env_a75d2.sh.new"
grep -q "A68_HOST_BASE=/root/a68" "$BASE/env_a75d2.sh.new" || die "env: A68_HOST_BASE is not /root/a68"
if [ -f "$BASE/env_a75d2.sh" ]; then cmp -s "$BASE/env_a75d2.sh" "$BASE/env_a75d2.sh.new" || die "env differs"; else cp "$BASE/env_a75d2.sh.new" "$BASE/env_a75d2.sh"; fi
put "$SHIP/run_a75d2.sh" "$BASE/run_a75d2.sh"; chmod +x "$BASE/run_a75d2.sh"
# code identity on the box: executor digest, runner arms + block, candidates = the seeded draw
( source "$BASE/env_a75d2.sh" && cd "$S" && "$PY" - <<'EOF'
import sys
sys.path[:0] = ['ops', '.']
from tools.coopmwm import a75_holdforms as H, nm_eval_runner as R
import a75_d2forms as M
assert H.a75_digest() == M.A75_SHA256, ('a75 digest', H.a75_digest())
assert R.A75_ARMS == ('A75_PLAN', 'A75_BANK')
s0 = M.mech_seed0()
assert tuple(R.A75D2_MECH_RANGE) == (s0, s0+99) and M.check_block_free(s0, R.EVAL_HELDOUT_RANGES)
assert len(M.candidates()) == 36
M.examples()
print('code identity ok: a75', H.a75_digest()[:8], 'block', s0, s0+99)
EOF
) || die "code identity check"
echo "tree diff vs the A68 tree (expected: runner, a75_holdforms, ops/a75/a75_specs.py, the D2 driver/test, ops/a75d2):"
diff -rq $FROM/src "$S" -x __pycache__ | cut -c1-200
du -sh "$BASE" | cut -c1-80
df -BG --output=avail "$BASE" | tail -1
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
