#!/bin/bash
# AMENDMENT 77 (S-MAPPO20C) training host setup. Run as user ubuntu (sudo -n only for apt + CPython install) from the
# unpacked ship dir:  bash /home/ubuntu/a77ship/setup_a77_train.sh
# Idempotent; ZERO SUMO, ZERO training; writes only /home/ubuntu/a77 (+ /opt/py312 when CPython is missing).
#   1 CPython 3.12.16 at /opt/py312 (exists on Box 3; else built from the shipped / python.org source, as for A65)
#   2 venv /home/ubuntu/a77/venv = the A65 pip freeze (requirements_a65.txt, sha 91b7b227) - STOP on any difference
#   3 code /home/ubuntu/a77/code = a51_code_4c2abb14.tgz (sha 800f210c, SOURCE_MANIFEST 644/644) + the A77 overlay
#     (2 NEW files only: the D2800_p20 config and the map adapter; nothing in the bundle is replaced)
#   4 inputs: u0 d9042cc5, A65 sanity_pre4 rows 3fe97fb7 (host identity reference); ledger; CPU tests (zero SUMO)
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
B=/home/ubuntu/a77
mkdir -p "$B"
LOG=$B/setup.log
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date '+%F %T') $*"; }
die() { say "SETUP FAILED: $*"; echo "$*" > "$B/SETUP_FAILED"; exit 1; }
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
pin() { [ "$(sha "$1")" = "$2" ] || die "pin mismatch: $1 (want $2)"; say "pin ok ${2:0:8} $1"; }
rm -f "$B/SETUP_FAILED"
[ "$(id -un)" = ubuntu ] || die "run as user ubuntu (paths are /home/ubuntu/a77, registered in the PLAN)"
[ "$(readlink -f "$B")" = "$B" ] || die "$B must be a real directory (the PLAN records this path)"
cd "$SHIP" && sha256sum -c --quiet SHA256SUMS || die "ship SHA256SUMS mismatch"
say "setup start host=$(hostname) nproc=$(nproc) mem=$(free -g | awk 'NR==2{print $2}')G disk_free=$(df -BG --output=avail /home | tail -1)"
[ "$(ulimit -Hn)" -ge 65536 ] || die "open-file hard limit $(ulimit -Hn) < 65536 (supervisor requirement)"

# ---------------------------------------------------------------- 1. CPython 3.12.16
PY312=/opt/py312/bin/python3.12
if ! { [ -x $PY312 ] && $PY312 -c 'import sys,ssl,sqlite3,bz2,lzma,ctypes,zlib; assert sys.version.split()[0]=="3.12.16"'; }; then
  say "building CPython 3.12.16 into /opt/py312 (as A65 on Box 3)"
  sudo -n true || die "sudo -n unavailable: CPython build needs apt + make altinstall"
  sudo -n DEBIAN_FRONTEND=noninteractive apt-get update -q || die "apt update"
  sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -q build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
    libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev uuid-dev tk-dev wget curl \
    ca-certificates rsync pigz || die "apt install"
  mkdir -p "$B/pybuild" && cd "$B/pybuild" || die "pybuild dir"
  if [ -f "$SHIP/Python-3.12.16.tgz" ]; then cp "$SHIP/Python-3.12.16.tgz" .; else wget -q https://www.python.org/ftp/python/3.12.16/Python-3.12.16.tgz || die "download"; fi
  pin Python-3.12.16.tgz 710c3e75dac8b2f258d9cd0f76e5c4c2e9175fac9e249c98901552f7cf33b7dd
  tar xzf Python-3.12.16.tgz && cd Python-3.12.16 || die "untar"
  ./configure --prefix=/opt/py312 --with-ensurepip=install > ../configure.log 2>&1 || die "configure"
  nice -n 10 make -j"$(nproc)" > ../make.log 2>&1 || die "make"
  sudo -n make altinstall > ../install.log 2>&1 || die "make altinstall"
fi
$PY312 -c 'import sys,ssl,sqlite3,bz2,lzma,ctypes,zlib; print("python", sys.version.split()[0])' || die "python check"

# ---------------------------------------------------------------- 2. venv = the A65 freeze
V=$B/venv
if [ ! -f "$V/FREEZE_OK" ]; then
  [ -e "$V" ] && die "$V exists without FREEZE_OK (inspect; nothing is deleted)"
  $PY312 -m venv "$V" || die "venv"
  nice -n 10 "$V/bin/python" -m pip install -q --no-input -r "$SHIP/requirements_a65.txt" \
      --extra-index-url https://download.pytorch.org/whl/cpu > "$B/pip.log" 2>&1 || { tail -5 "$B/pip.log"; die "pip install"; }
  F=$("$V/bin/python" -m pip freeze 2>/dev/null | sha256sum | cut -c1-64)
  [ "$F" = 91b7b227b4dd4c4783cf52c5ab10a0a2db9ea595d711a9fbdd8cc03d7cc72688 ] || {
    "$V/bin/python" -m pip freeze > "$B/pip_freeze_differs.txt"; die "pip freeze $F != A65 91b7b227 (see pip_freeze_differs.txt)"; }
  "$V/bin/python" -m pip freeze > "$V/FREEZE_OK"
fi
SUMO_HOME=$V/lib/python3.12/site-packages/sumo
"$SUMO_HOME/bin/sumo" --version | head -1 | grep -q "1.27.1" || die "sumo --version is not 1.27.1"
ldd "$SUMO_HOME/bin/sumo" | grep -q "not found" && die "sumo has missing libraries"
say "venv ok (freeze 91b7b227 = A65), sumo 1.27.1"

# ---------------------------------------------------------------- 3. code = A51 bundle + 2 new overlay files
C=$B/code
if [ ! -f "$B/CODE_OK" ]; then
  [ -e "$C" ] && die "$C exists without CODE_OK"
  pin "$SHIP/a51_code_4c2abb14.tgz" 800f210c8f11913ee73d376d35010a7a42043ac71669b002690d69db3a2df1e0
  mkdir -p "$C" && tar xzf "$SHIP/a51_code_4c2abb14.tgz" -C "$C" --no-same-owner || die "code extract"
  for f in configs/c4_mappo_speed_wz3to2_d2800p20_a77.toml tools/nm_mappo/n1_adapter_a77.py; do
    [ -e "$C/$f" ] && die "overlay file $f already in the A51 bundle"
    cp "$SHIP/overlay/$f" "$C/$f" || die "overlay $f"
  done
  touch "$B/CODE_OK"
fi
"$V/bin/python" - "$C" <<'EOF' || die "SOURCE_MANIFEST check"
import hashlib, json, sys
from pathlib import Path
c = Path(sys.argv[1]); m = json.loads((c/'SOURCE_MANIFEST.json').read_text())
bad = [n for n, d in m.items() if hashlib.sha256((c/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != d]
print('SOURCE_MANIFEST', len(m) - len(bad), '/', len(m), 'ok'); sys.exit(1 if bad else 0)
EOF
pin "$C/configs/c4_mappo_speed_wz3to2_d2800p20_a77.toml" 76dc251b9d12a6372b31a283c4b21b29773855d93bcf2e849ab32432f1b4283f

# ---------------------------------------------------------------- 4. inputs, ledger, tests
mkdir -p "$B/inputs" "$B/ledger"
for f in mappo_u0_policy_00169.pt a65_sanity_pre4_rows.jsonl; do [ -f "$B/inputs/$f" ] || cp "$SHIP/$f" "$B/inputs/$f" || die "input $f"; done
pin "$B/inputs/mappo_u0_policy_00169.pt" d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca
pin "$B/inputs/a65_sanity_pre4_rows.jsonl" 3fe97fb70d3b8804570b41bbc04eb5a8bbb9e4e1c6f7eaea73b07e916bae9acb
cmp -s "$SHIP/charge.py" "$B/ledger/charge.py" || cp "$SHIP/charge.py" "$B/ledger/charge.py" || die "ledger"
if [ ! -d "$B/testlib" ]; then nice -n 10 "$V/bin/python" -m pip install -q --no-input --target "$B/testlib" pytest > "$B/pip_test.log" 2>&1 || die "pytest (separate --target dir; venv untouched)"; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
A77_CODE=$C PYTHONPATH=$B/testlib nice -n 10 "$V/bin/python" -m pytest -q -p no:cacheprovider "$SHIP/test_a77_train.py" \
  > "$B/tests.out" 2>&1 || { tail -30 "$B/tests.out"; die "A77 CPU tests failed"; }
say "tests: $(tail -1 "$B/tests.out")"
F=$("$V/bin/python" -m pip freeze 2>/dev/null | sha256sum | cut -c1-64)
[ "$F" = 91b7b227b4dd4c4783cf52c5ab10a0a2db9ea595d711a9fbdd8cc03d7cc72688 ] || die "venv changed during setup"
date '+%F %T' > "$B/SETUP_OK"
say "SETUP_OK  (next: register the PLAN sha, then go_train.sh)"
