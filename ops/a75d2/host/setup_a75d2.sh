#!/bin/bash
# AMENDMENT 75 stage D2 host setup v2 (run as root; zero SUMO; never deletes; writes only /root/a75d2b).
# usage: A75D2_MECH_SEED0=<block start> bash setup_a75d2.sh [ref68_reports.tar]
# v2 (2026-11-13): the v1 base /root/a68/src has no A44 executor / runner, so the source tree is now a HARD-LINK clone
#   (cp -al) of /root/a69/src (= /root/a68/src + the A44 variant files + the merged A68+A44 runner 62496e86). New base dir
#   /root/a75d2b (the partial v1 /root/a75d2 is never read, changed or removed).
# Changed files: pytools/tools/coopmwm/nm_eval_runner.py and ops/nm_finaleval.py, each copied to a NEW inode, patched
#   additively, then renamed over the link (the A69 / A68 inodes keep their bytes; checked): runner = the registered A70
#   table edit (ops/a70/host/patch_tables.py) + A75 arms (patch_runner_a75.py) + the D2 mechanics block
#   (patch_tables_d2.py) -> sha pinned for 997400; settings = the registered A70 a70c setting + its panel pin.
# New files: a75_holdforms.py, ops/a75/a75_specs.py, the D2 driver / test / ops/a75d2, ops/a70_smallbank.py, the A70 panel.
# Worlds / panels / bindings / references stay the A68 ones (root = the A68 root binding files, hard links).
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a75d2b
FROM=/root/a69
A68=/root/a68
S=$BASE/src
R=$BASE/root
TAR=${1:-}
RUN69_SHA=62496e864567700a051aaf3d26ee16fb64e1dbe5b0a34b7d28688e0defb67836
A44_SHA=4e7a97f3183290ca5f1ad161cb3557b27d9b3695eb596a18f3ee3097e92aac0c
RUN_D2_SHA=ec095e50fde9eaa885b3f8dcf9c10199bddeb1cae6f50730da34e2a27dceb84e
A70_PANEL=1b4b59ee4ab717b931647fbae7cfa489726cd2cbacad30e268c57e32cb2c78e1
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
[ -f "$BASE/SETUP_OK" ] && die "$BASE/SETUP_OK exists (setup runs once)"
S0=${A75D2_MECH_SEED0:-}
[ -n "$S0" ] || die "A75D2_MECH_SEED0 unset: the mechanics seed block is assigned by the coordinator at registration"
[[ "$S0" =~ ^9[0-9]{5}$ ]] && [ $((S0 % 100)) = 0 ] || die "A75D2_MECH_SEED0=$S0 is not a 100-seed block start"
for b in "964000 964199" "962000 962999" "977000 977999" "989600 989699"; do
  set -- $b; [ "$S0" -le "$2" ] && [ $((S0+99)) -ge "$1" ] && die "A75D2_MECH_SEED0=$S0 overlaps forbidden $1-$2"
done
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
( cd "$SHIP" && sha256sum -c --quiet SHIP_FILES.sha256 ) || die "shipped files differ from SHIP_FILES.sha256"
for d in $FROM/src $A68/src; do [ -L $d ] && die "$d is a symlink"; done
[ -f $FROM/SETUP_OK ] || die "$FROM is not a completed A69 setup"
[ -f $A68/SETUP_OK ] || die "$A68 is not a completed A68 setup"
[ -f $A68/root/A68_ADMISSION_D2800_p60.json ] || die "$A68/root has no A68 admission receipt"
RT=pytools/tools/coopmwm/nm_eval_runner.py
ST=ops/nm_finaleval.py
pin "$FROM/src/$RT" $RUN69_SHA
pin "$FROM/src/pytools/tools/coopmwm/a44_lcvariant.py" $A44_SHA
cmp -s "$FROM/src/$ST" "$A68/src/$ST" || die "the A69 settings file is not the A68 one"
ST0=$(sha "$FROM/src/$ST"); RT68=$(sha "$A68/src/$RT")
PY=$(grep "^PY=" $A68/env_a68.sh | cut -d= -f2)
[ -x "$PY" ] || die "python of the A68 env not found"
mkdir -p "$BASE"
if [ ! -d "$S" ]; then cp -al $FROM/src "$S" || die "link tree"; fi
[ -L "$S" ] && die "$S is a symlink"
[ "$(readlink -f $S)" = "$S" ] || die "$S does not resolve to itself"
T=$S/pytools/tools/coopmwm
mkdir -p "$S/ops/a75d2/host" "$S/ops/a75"
put "$SHIP/a75_holdforms.py" "$T/a75_holdforms.py"
put "$SHIP/a75_specs.py" "$S/ops/a75/a75_specs.py"
put "$SHIP/a75_d2forms.py" "$S/ops/a75_d2forms.py"
put "$SHIP/test_a75_d2forms.py" "$S/ops/test_a75_d2forms.py"
put "$SHIP/a70_smallbank.py" "$S/ops/a70_smallbank.py"
put "$SHIP/nm_eval_panel_wz3to2_a70c_D2800_p60.json" "$S/ops/nm_eval_panel_wz3to2_a70c_D2800_p60.json"
for f in CANDIDATES_D2_D2800_p60.json CANDIDATES_D2_D2800_p60.csv CANDIDATES_D2_D2800_p60.sha256 REF_PINS.json EXAMPLE_PLANS.json; do
  put "$SHIP/$f" "$S/ops/a75d2/$f"
done
put "$SHIP/patch_tables_d2.py" "$S/ops/a75d2/host/patch_tables_d2.py"
pin "$S/ops/a70_smallbank.py" 72af622e0c2188d8616da6a45faf3b25ff80bd27feced7467c66c5a5ae716da0
pin "$S/ops/nm_eval_panel_wz3to2_a70c_D2800_p60.json" 835e423adf8fb9bb2cee98c645a845b43f23ce6ec073a5c6bf70db65a045a1d4
# runner + settings: new inodes, additive patches, rename over the links
if ! grep -q "^A75D2_MECH_RANGE = ($S0, $((S0+99)))$" "$S/$RT"; then
  [ "$(sha "$S/$RT")" = $RUN69_SHA ] || die "$S/$RT is neither the A69 runner nor the D2 runner"
  [ "$(sha "$S/$ST")" = "$ST0" ] || die "$S/$ST is not the A68 settings file"
  P=$BASE/patch.tmp
  [ -e "$P" ] && die "$P exists (inspect; never removed automatically)"
  mkdir "$P" && cp "$S/$RT" "$P/runner.py" && cp "$S/$ST" "$P/settings.py" && cp "$SHIP/a70_smallbank.py" "$P/a70_smallbank.py" || die "copies"
  "$PY" -I "$SHIP/patch_tables_a70.py" tables "$P/runner.py" "$P/settings.py" || die "A70 table patch"
  "$PY" -I "$SHIP/patch_tables_a70.py" pin "$P/settings.py" "$P/a70_smallbank.py" $A70_PANEL || die "A70 pin patch"
  cmp -s "$P/a70_smallbank.py" "$SHIP/a70_smallbank.py" || die "the A70 driver would change"
  "$PY" -I "$SHIP/patch_runner_a75.py" arms "$P/runner.py" || die "A75 arms patch"
  "$PY" -I "$SHIP/patch_tables_d2.py" tables "$P/runner.py" "$S0" || die "D2 table patch"
  [ "$S0" != 997400 ] || pin "$P/runner.py" $RUN_D2_SHA
  [ "$(diff "$S/$ST" "$P/settings.py" | grep -c '^<')" = 0 ] || die "the settings patch removed lines"
  mv -f "$P/runner.py" "$S/$RT" && mv -f "$P/settings.py" "$S/$ST" || die "install patched files"
  mv "$P" "$BASE/patch.done" || die "keep patch dir"
fi
pin "$FROM/src/$RT" $RUN69_SHA; pin "$A68/src/$RT" "$RT68"; pin "$FROM/src/$ST" "$ST0"; pin "$A68/src/$ST" "$ST0"
[ "$(stat -c %i "$S/$RT")" != "$(stat -c %i "$FROM/src/$RT")" ] || die "the D2 runner still shares the A69 inode"
[ "$(stat -c %i "$S/$ST")" != "$(stat -c %i "$FROM/src/$ST")" ] || die "the D2 settings still share the A69 inode"
echo "runner A69 ${RUN69_SHA:0:8} (unchanged) -> D2 $(sha "$S/$RT")   settings A68 ${ST0:0:8} (unchanged) -> D2 $(sha "$S/$ST")"
mkdir -p "$BASE/panels"
for f in $A68/panels/*.json; do [ -f "$BASE/panels/$(basename "$f")" ] || ln "$f" "$BASE/panels/$(basename "$f")" || die "panel link"; done
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  [ -f "$R/$f" ] || ln "$A68/root/$f" "$R/$f" || die "mirror $f"
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
sed -e "s#^BASE=/root/a68#BASE=$BASE#" -e "s#^SRC=/root/a68/src#SRC=$S#" $A68/env_a68.sh > "$BASE/env_a75d2.sh.new"
echo "export A75D2_HOST_BASE=$BASE" >> "$BASE/env_a75d2.sh.new"
echo "export A75D2_MECH_SEED0=$S0" >> "$BASE/env_a75d2.sh.new"
grep -q "A68_HOST_BASE=/root/a68" "$BASE/env_a75d2.sh.new" || die "env: A68_HOST_BASE is not /root/a68"
grep -q "^SRC=$S$" "$BASE/env_a75d2.sh.new" || die "env: SRC not switched to $S"
if [ -f "$BASE/env_a75d2.sh" ]; then cmp -s "$BASE/env_a75d2.sh" "$BASE/env_a75d2.sh.new" || die "env differs"; else cp "$BASE/env_a75d2.sh.new" "$BASE/env_a75d2.sh"; fi
put "$SHIP/run_a75d2.sh" "$BASE/run_a75d2.sh"; chmod +x "$BASE/run_a75d2.sh"
# code identity on the box: executor digest, runner arms + blocks, settings, candidates = the seeded draw
( source "$BASE/env_a75d2.sh" && cd "$S" && "$PY" - <<'EOF'
import sys
sys.path[:0] = ['ops', '.']
from tools.coopmwm import a75_holdforms as H, nm_eval_runner as R
import a75_d2forms as M
import nm_finaleval as N
assert H.a75_digest() == M.A75_SHA256, ('a75 digest', H.a75_digest())
assert R.A75_ARMS == ('A75_PLAN', 'A75_BANK') and R.A44_ARMS == ('A44_PLAN', 'A44_BANK')
assert tuple(R.A70_RANGES) == ((996300, 996399),) and 'a70c_D2800_p60' in N.SETTINGS
s0 = M.mech_seed0()
assert tuple(R.A75D2_MECH_RANGE) == (s0, s0+99) and M.check_block_free(s0, R.EVAL_HELDOUT_RANGES)
assert len(M.candidates()) == 36
M.examples()
print('code identity ok: a75', H.a75_digest()[:8], 'block', s0, s0+99)
EOF
) || die "code identity check"
echo "tree diff vs the A69 tree (expected: runner, settings, a75_holdforms, ops/a75/a75_specs.py, a70_smallbank + A70 panel, the D2 driver/test, ops/a75d2):"
diff -rq $FROM/src "$S" -x __pycache__ | cut -c1-200
du -sh "$BASE" | cut -c1-80
df -BG --output=avail "$BASE" | tail -1
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
