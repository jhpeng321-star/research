#!/bin/bash
# AMENDMENT 80 (A80-S2B) host setup (run as root; zero SUMO; never deletes; writes only $A80_BASE, default /root/a78s2b).
# Same build as A75-D2 v3 (ops/a75d2/host/setup_a75d2.sh) WITHOUT the D2 mechanics block: the source tree is a HARD-LINK
# clone (cp -al) of /root/a69/src + every /root/a68/src file missing from it (byte-identical check of everything shared);
# runner and settings are copied to NEW inodes, patched additively (A70 tables + A70 setting pin, A75 arms) and renamed
# over the links (the A69 / A68 inodes keep their bytes; checked). New files: a75_holdforms d1af7e0f, ops/a75/a75_specs,
# the A75-D2 driver (unchanged), a70_smallbank + A70 panel, ops/a75d2/{EXAMPLE_PLANS,REF_PINS}.json, the A80 driver /
# test / fixtures. Root binding files = hard links of the A68 root (cmp with the A66 root); reference reports = hard
# links of the D1 copies (/root/a75/ref68). /root/a75d2c and /root/a76p20 are never read or written.
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=${A80_BASE:-/root/a78s2b}
FROM=/root/a69
A68=/root/a68
S=$BASE/src
R=$BASE/root
RUN69_SHA=62496e864567700a051aaf3d26ee16fb64e1dbe5b0a34b7d28688e0defb67836
A44_SHA=4e7a97f3183290ca5f1ad161cb3557b27d9b3695eb596a18f3ee3097e92aac0c
A70_PANEL=1b4b59ee4ab717b931647fbae7cfa489726cd2cbacad30e268c57e32cb2c78e1
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
case "$BASE" in /root/a75d2c*|/root/a76p20*|/root/a69|/root/a68|/root/a66) die "$BASE is another stage's base" ;; esac
[ -f "$BASE/SETUP_OK" ] && die "$BASE/SETUP_OK exists (setup runs once)"
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
( cd $A68/src && find . -type f ! -path '*/__pycache__/*' | sort ) > "$BASE/a68_files.list"
: > "$BASE/A68_SYNC.list.new"
while read -r f; do
  if [ ! -e "$S/$f" ]; then mkdir -p "$S/$(dirname "$f")" && ln "$A68/src/$f" "$S/$f" || die "link $f"; echo "$f" >> "$BASE/A68_SYNC.list.new"; fi
done < "$BASE/a68_files.list"
[ -f "$BASE/A68_SYNC.list" ] || mv "$BASE/A68_SYNC.list.new" "$BASE/A68_SYNC.list"
echo "A68 files linked into the clone: $(wc -l < "$BASE/A68_SYNC.list")"
while read -r f; do
  case "$f" in ./$RT|./$ST) continue ;; esac
  cmp -s "$A68/src/$f" "$S/$f" || die "A68 file differs in the clone: $f"
done < "$BASE/a68_files.list"
pin "$S/ops/a68/CANDIDATES_D2800_p60.json" bff1a7bbf0212bbbcbbee8de3f9b7de7f0e1c58708ed4d31dc0fbbb9c0d79b22
pin "$S/ops/a68/CANDIDATES_D2800_p60.sha256" be1a5df3585dd66f08556fed54f8ddf1eb0b8f4a5eccc60dad6f2505960ad69a
mkdir -p "$S/ops/a75d2" "$S/ops/a75" "$S/ops/a80"
put "$SHIP/a75_holdforms.py" "$T/a75_holdforms.py"
put "$SHIP/a75_specs.py" "$S/ops/a75/a75_specs.py"
put "$SHIP/a75_d2forms.py" "$S/ops/a75_d2forms.py"
put "$SHIP/a70_smallbank.py" "$S/ops/a70_smallbank.py"
put "$SHIP/nm_eval_panel_wz3to2_a70c_D2800_p60.json" "$S/ops/nm_eval_panel_wz3to2_a70c_D2800_p60.json"
for f in REF_PINS.json EXAMPLE_PLANS.json; do put "$SHIP/$f" "$S/ops/a75d2/$f"; done
put "$SHIP/a80_s2b.py" "$S/ops/a80_s2b.py"
put "$SHIP/test_a80_s2b.py" "$S/ops/test_a80_s2b.py"
for f in SYNTH_TOP_OFFER_CONDITIONS_D2800_p60.json E2E_TINY_TOP_OFFER_CONDITIONS_D2800_p60.json; do put "$SHIP/$f" "$S/ops/a80/$f"; done
pin "$T/a75_holdforms.py" 72046db6ad9a3f117105ffdf029ac993ec3d421026da6a1ca4095d5f7be620eb
pin "$S/ops/a75_d2forms.py" 8a2533ef58f5e4dc295808830cfb4bdc39e20c417cf98459dc96f5b7762551cf
pin "$S/ops/a70_smallbank.py" 72af622e0c2188d8616da6a45faf3b25ff80bd27feced7467c66c5a5ae716da0
pin "$S/ops/nm_eval_panel_wz3to2_a70c_D2800_p60.json" 835e423adf8fb9bb2cee98c645a845b43f23ce6ec073a5c6bf70db65a045a1d4
pin "$S/ops/a75d2/EXAMPLE_PLANS.json" c7d6b8ef3a7860441fa5e54ad57acf62c86b4bc1352e6b1a8ba82b4e5a3f4db6
pin "$S/ops/a75d2/REF_PINS.json" 9741e2e320821fa157b77626044539b92349bfe1f9e3d1e0b4b3f0ed0663d97b
# runner + settings: new inodes, additive patches (A70 tables + setting pin, A75 arms), rename over the links
if ! grep -q "^A75_ARMS = " "$S/$RT"; then
  [ "$(sha "$S/$RT")" = $RUN69_SHA ] || die "$S/$RT is neither the A69 runner nor an A80 runner"
  [ "$(sha "$S/$ST")" = "$ST0" ] || die "$S/$ST is not the A68 settings file"
  P=$BASE/patch.tmp
  [ -e "$P" ] && die "$P exists (inspect; never removed automatically)"
  mkdir "$P" && cp "$S/$RT" "$P/runner.py" && cp "$S/$ST" "$P/settings.py" && cp "$SHIP/a70_smallbank.py" "$P/a70_smallbank.py" || die "copies"
  "$PY" -I "$SHIP/patch_tables_a70.py" tables "$P/runner.py" "$P/settings.py" || die "A70 table patch"
  "$PY" -I "$SHIP/patch_tables_a70.py" pin "$P/settings.py" "$P/a70_smallbank.py" $A70_PANEL || die "A70 pin patch"
  cmp -s "$P/a70_smallbank.py" "$SHIP/a70_smallbank.py" || die "the A70 driver would change"
  "$PY" -I "$SHIP/patch_runner_a75.py" arms "$P/runner.py" || die "A75 arms patch"
  [ "$(diff "$S/$ST" "$P/settings.py" | grep -c '^<')" = 0 ] || die "the settings patch removed lines"
  mv -f "$P/runner.py" "$S/$RT" && mv -f "$P/settings.py" "$S/$ST" || die "install patched files"
  mv "$P" "$BASE/patch.done" || die "keep patch dir"
fi
pin "$FROM/src/$RT" $RUN69_SHA; pin "$A68/src/$RT" "$RT68"; pin "$FROM/src/$ST" "$ST0"; pin "$A68/src/$ST" "$ST0"
[ "$(stat -c %i "$S/$RT")" != "$(stat -c %i "$FROM/src/$RT")" ] || die "the A80 runner still shares the A69 inode"
[ "$(stat -c %i "$S/$ST")" != "$(stat -c %i "$FROM/src/$ST")" ] || die "the A80 settings still share the A69 inode"
echo "runner A69 ${RUN69_SHA:0:8} (unchanged) -> A80 $(sha "$S/$RT")   settings A68 ${ST0:0:8} (unchanged) -> A80 $(sha "$S/$ST")"
mkdir -p "$BASE/panels"
for f in $A68/panels/*.json; do [ -f "$BASE/panels/$(basename "$f")" ] || ln "$f" "$BASE/panels/$(basename "$f")" || die "panel link"; done
MIRROR="BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
 oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json"
for RR in "$R" "$BASE/synth/root"; do
  mkdir -p "$RR/prerun_plant" "$RR/oracle_banks" "$RR/a32_banks"
  for f in $MIRROR; do
    [ -f "$RR/$f" ] || ln "$A68/root/$f" "$RR/$f" || die "mirror $f"
    cmp -s "/root/a66/root/$f" "$RR/$f" || die "$RR/$f differs from the A66 root"
  done
done
pin /root/a66/inputs/policy_00084.pt 2abed31707b61121569af2936c23689827b52b28422df162751b230aa6a93de8
X=$BASE/ref68
if [ ! -d "$X/evaluation" ]; then
  [ -d /root/a75/ref68/evaluation ] || die "no reference reports (/root/a75/ref68 of D1)"
  mkdir -p "$X" && cp -al /root/a75/ref68/evaluation "$X/evaluation" || die "link D1 reference reports"
fi
[ "$(find "$X/evaluation" -type f -name vehicles_report.json | wc -l)" = 24 ] || die "reference reports: not 24 files"
sed -e "s#^BASE=/root/a68#BASE=$BASE#" -e "s#^SRC=/root/a68/src#SRC=$S#" $A68/env_a68.sh > "$BASE/env_a80.sh.new"
echo "export A80_HOST_BASE=$BASE" >> "$BASE/env_a80.sh.new"
grep -q "A68_HOST_BASE=/root/a68" "$BASE/env_a80.sh.new" || die "env: A68_HOST_BASE is not /root/a68"
grep -q "^SRC=$S$" "$BASE/env_a80.sh.new" || die "env: SRC not switched to $S"
if [ -f "$BASE/env_a80.sh" ]; then cmp -s "$BASE/env_a80.sh" "$BASE/env_a80.sh.new" || die "env differs"; else cp "$BASE/env_a80.sh.new" "$BASE/env_a80.sh"; fi
put "$SHIP/run_a80.sh" "$BASE/run_a80.sh"; chmod +x "$BASE/run_a80.sh"
( source "$BASE/env_a80.sh" && cd "$S" && "$PY" - <<'EOF'
import sys
sys.path[:0] = ['ops', '.']
from tools.coopmwm import a75_holdforms as H, nm_eval_runner as R
import a80_s2b as M
import a75_d2forms as D2
import nm_finaleval as N
assert H.a75_digest() == M.A75_SHA256 == D2.A75_SHA256, ('a75 digest', H.a75_digest())
assert R.A75_ARMS == ('A75_PLAN', 'A75_BANK') and R.A44_ARMS == ('A44_PLAN', 'A44_BANK')
assert tuple(R.A70_RANGES) == ((996300, 996399),) and 'a70c_D2800_p60' in N.SETTINGS and 'a68v_D2800_p60' in N.SETTINGS
assert not hasattr(R, 'A75D2_MECH_RANGE')
D2.examples()
cov = M.covered_tokens()
raw = open('ops/a80/SYNTH_TOP_OFFER_CONDITIONS_D2800_p60.json', 'rb').read()
doc = M.candidates_doc(raw, M.sha_bytes(raw))
import a68_research60c as A68
assert A68.candidates() and A68.CANDS_PIN.read_text().split()[0] == '194610bd3efe8168c852b04819025e878cbe5d3b59718abe1898ccc289b77186'
print('code identity ok: a75', H.a75_digest()[:8], 'covered tokens', len(cov), 'synthetic candidates', doc['n_candidates'],
      'mech plans', len(doc['mech_plans']))
EOF
) || die "code identity check"
echo "tree diff vs the A69 tree (expected: runner, settings, a75_holdforms, ops/a75, a70_smallbank + A70 panel, a75_d2forms, ops/a75d2, the A80 files, A68 sync):"
diff -rq $FROM/src "$S" -x __pycache__ | cut -c1-200 | head -40
du -sh "$BASE" | cut -c1-80
df -BG --output=avail "$BASE" | tail -1
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
