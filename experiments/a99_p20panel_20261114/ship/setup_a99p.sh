#!/bin/bash
# A99 PANEL host setup (Box 4, root; zero SUMO; never deletes; writes only /root/a99panel).
#   bash /root/a99panel_ship/setup_a99p.sh
# src  = REAL copy (cp -a) of the A99 admission tree /root/a99adm_s4/src (= /root/a96adm_s5/src + ops/a99_on20c.py; carries
#        the 11-14 nm_finaleval cache fix fa98b552), verified file by file; then the runner table row A96C_RANGES
#        (984400-984599) appended (patch_runner_a99p.py; one changed tail line) + ops/a96_p20.py (A96 c/d driver b29c45d4,
#        unchanged) + ops/a99_panel.py + ops/test_a99_panel.py + ops/a99/patch_runner_a99p.py.
# root = byte copies of the 8 p20c binding files (cmp with /root/a99adm_s4/root and the p20c bank root);
# panels/A96C_PANEL_C_192.json = the A96(c) staged panel (Box 3 /root/a96pc/root/A96C_PANEL_C.json, file 0bb84f3c).
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a99panel
FROM=/root/a99adm_s4
S=$BASE/src
R=$BASE/root
B=/root/autodl-fs/nm_finaleval_p20c_bank
RT=pytools/tools/coopmwm/nm_eval_runner.py
die() { echo "SETUP FAILED: $*"; exit 1; }
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; echo "pin ok ${2:0:8} $1"; }
[ "$(id -u)" = 0 ] || die "must run as root"
[ -e "$BASE" ] && die "$BASE exists (new root only)"
( cd "$SHIP" && sha256sum -c --quiet SHIP_FILES.sha256 ) || die "shipped files differ from SHIP_FILES.sha256"
grep -q A99Q_DONE /root/a99q.log || die "A99 admission queue not done"
pin "$FROM/src/ops/nm_finaleval.py" fa98b5526296f67e96c7670d7baedf4e943df49074822daa9f6fdd73a69fc132
pin "$FROM/src/$RT" 0c5c2a4b53052ae25103e9b8ac7b895b86b590de4e4d94b77d2110d9065fa8c8
pin "$FROM/src/ops/a99_on20c.py" f3620a8cf48371959b9b3fc36ef9da7eb13afc433df37f6de13c4b9f7b7c28a5
pin "$FROM/src/ops/a76_d2p20.py" 3a1b7b5763276340e541ffa3294840ba9767e22f69417af341e53ce514505a6e
pin "$FROM/src/ops/a75_d2forms.py" 8a2533ef58f5e4dc295808830cfb4bdc39e20c417cf98459dc96f5b7762551cf
pin "$FROM/src/ops/a80_s2b.py" 48c5c448c2a85d4e2020def54cb17259979c89f2e0cc341829d6e99de0d0e289
pin "$FROM/src/pytools/tools/coopmwm/a75_holdforms.py" 72046db6ad9a3f117105ffdf029ac993ec3d421026da6a1ca4095d5f7be620eb
pin "$SHIP/A96C_PANEL_C_192.json" 0bb84f3c129ca7347236337cb4bcc316b447a0186f6478fa2ca9451646145c56
pin /root/a99adm_s4/root/a96_banks/A96_BANK_D2800_p20.json 682c01ae1deb28c774701ff0a5102cd77c2addf2ca302e4e42a90409b7302ab2
pin /root/a99adm_hand/root/a96_banks/A96_BANK_D2800_p20.json 59acb7471c51da728ffe7e1f0600082a8e6486a0455108a21df41d7b851bd946
[ -e /root/a99adm_s5/root/a96_banks/A96_BANK_D2800_p20.json ] && die "an S5 bank exists: the package assumes S5 empty"
mkdir -p "$BASE"
nice -n 19 ionice -c3 cp -a "$FROM/src" "$S" || die "copy tree"
[ -L "$S" ] && die "$S is a symlink"
( cd "$FROM/src" && find . -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum ) > "$BASE/a99s4_tree.sha256"
( cd "$S" && sha256sum -c --quiet "$BASE/a99s4_tree.sha256" ) || die "the copy differs from $FROM/src"
echo "tree copy ok: $(wc -l < "$BASE/a99s4_tree.sha256") files identical to $FROM/src"
mkdir -p "$S/ops/a99"
cp "$SHIP/patch_runner_a99p.py" "$S/ops/a99/patch_runner_a99p.py" || die "patch script"
PY=$(grep "^PY=" $FROM/env_a99.sh | cut -d= -f2)
[ -x "$PY" ] || die "python of the A99 env not found"
"$PY" -I "$S/ops/a99/patch_runner_a99p.py" tables "$S/$RT" || die "runner table patch"
[ "$(diff "$FROM/src/$RT" "$S/$RT" | grep -c '^<')" = 1 ] || die "the runner patch changed more than the tail line"
[ "$(stat -c %i "$S/$RT")" != "$(stat -c %i "$FROM/src/$RT")" ] || die "runner shares the A99 inode"
pin "$S/$RT" 47afb40951b237f4d2be01a7e908b8b704c57250345058443312a92ef84a4020
pin "$FROM/src/$RT" 0c5c2a4b53052ae25103e9b8ac7b895b86b590de4e4d94b77d2110d9065fa8c8
for f in a96_p20.py a99_panel.py test_a99_panel.py; do cp "$SHIP/$f" "$S/ops/$f" || die "install $f"; done
mkdir -p "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks" "$BASE/panels"
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
         oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
         a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  cp "$FROM/root/$f" "$R/$f" || die "copy $f"
  cmp -s "$FROM/root/$f" "$R/$f" || die "$R/$f differs from the A99 root"
  cmp -s "$B/$f" "$R/$f" || die "$R/$f differs from the p20c binding root"
done
cp "$SHIP/A96C_PANEL_C_192.json" "$BASE/panels/A96C_PANEL_C_192.json" || die "panel copy"
pin "$BASE/panels/A96C_PANEL_C_192.json" 0bb84f3c129ca7347236337cb4bcc316b447a0186f6478fa2ca9451646145c56
sed -e "s#^BASE=$FROM\$#BASE=$BASE#" -e "s#^SRC=$FROM/src\$#SRC=$S#" -e "s#A76_HOST_BASE=$FROM#A76_HOST_BASE=$BASE#" \
    -e "s#A96_HOST_BASE=$FROM #A96_HOST_BASE=$BASE A96_PART=c #" $FROM/env_a99.sh > "$BASE/env_a99p.sh"
grep -q "^SRC=$S$" "$BASE/env_a99p.sh" || die "env: SRC not switched"
grep -q "A96_HOST_BASE=$BASE A96_PART=c " "$BASE/env_a99p.sh" || die "env: A96_HOST_BASE / A96_PART not set"
grep -q "a99adm" "$BASE/env_a99p.sh" && die "env: an a99adm path remains"
cp "$SHIP/run_a99p.sh" "$BASE/run_a99p.sh" && chmod +x "$BASE/run_a99p.sh"
echo "tree diff vs $FROM/src (expected: runner + ops/a96_p20.py, ops/a99_panel.py, ops/test_a99_panel.py, ops/a99):"
diff -rq "$FROM/src" "$S" -x __pycache__ | cut -c1-200
date "+%F %T" > "$BASE/SETUP_OK"
echo SETUP_OK
