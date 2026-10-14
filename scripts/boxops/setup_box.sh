#!/usr/bin/env bash
set -euo pipefail
usage() { echo 'Usage: setup_box.sh [--help] (after fetch_code.sh; apt + ~/c4env + 30 simulated-second TraCI smoke)'; }
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
[[ $# == 0 ]] || { usage >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
mkdir -p "$HOME/c4" "$HOME/.ssh"
exec 8>"$HOME/c4/.setup.lock"
flock -n 8 || die 'setup is already running'
exec > >(tee -a "$HOME/setup.log") 2>&1
echo "setup start $(date -u +%FT%TZ)"
if [[ -f $HOME/c4/SETUP_DONE ]]; then
    mv "$HOME/c4/SETUP_DONE" "$HOME/c4/SETUP_DONE.previous-$(date -u +%Y%m%dT%H%M%S)-$$"
fi
chmod 700 "$HOME/.ssh"
if [[ ! -f $RELAY_KEY ]]; then ssh-keygen -q -t ed25519 -N '' -f "$RELAY_KEY" -C c4-box-relay; fi
chmod 600 "$RELAY_KEY"
[[ -f $RELAY_KEY.pub ]] || ssh-keygen -y -f "$RELAY_KEY" >"$RELAY_KEY.pub"
echo 'Authorize this PUBLIC key on the relay:'
cat "$RELAY_KEY.pub"
[[ -f $REPO/traffic_shepherd/c4/collect.py ]] || die 'fetch code first (see BOXOPS.md bootstrap)'
SUDO=()
if ((EUID != 0)); then SUDO=(sudo -n); fi
export DEBIAN_FRONTEND=noninteractive
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y python3 python3-venv python3-pip rsync git openssh-client \
    curl ca-certificates build-essential util-linux libgomp1 \
    libxrender1 libxext6 libxft2 libgl1 libglu1-mesa libfontconfig1 libxcursor1 \
    libxinerama1 libxrandr2 libxi6
[[ -x $VENV/bin/python ]] || python3 -m venv "$VENV"
"$PYTHON" -m pip install 'pip==25.1.1' 'setuptools==80.9.0' 'wheel==0.45.1'
# No fallback to an unpinned/newer torch if the requested wheel is unavailable.
"$PYTHON" -m pip install 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cpu
"$PYTHON" -m pip install -r "$HERE/requirements-box.txt"
"$PYTHON" -m pip install --no-deps --no-build-isolation -e "$REPO"
"$PYTHON" -m pip check
activate_box
"$PYTHON" -m pip freeze >"$HOME/c4/INSTALLED_FREEZE.txt"
sumo --version
SMOKE=$HOME/c4/setup-smoke/$(date -u +%Y%m%dT%H%M%S)-$$
"$PYTHON" -m traffic_shepherd.c4.collect --policy native --seeds 4107 --penetrations 0.1 \
    --demands 2400 --warmup 0 --horizon 30 --workers 1 --no-physics-log --out "$SMOKE"
"$PYTHON" "$HERE/ops.py" check-collection "$SMOKE" 1
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" >"$HOME/c4/SETUP_DONE"
echo "SETUP_DONE: $HOME/c4/SETUP_DONE"
