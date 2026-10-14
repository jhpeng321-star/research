#!/usr/bin/env bash
set -euo pipefail
usage() { echo 'Usage: fetch_code.sh [ARCHIVE_ID] [--help] (default c4-code; pulls ARCHIVE_ID.tar.gz + .sha256 from relay)'; }
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
[[ $# -le 1 ]] || { usage >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
ARCHIVE_ID=${1:-c4-code}
valid_id "$ARCHIVE_ID"; safe_path "$ARCHIVE_ID"; safe_path "$REPO"
mkdir -p "$HOME/c4/downloads" "$HOME/.ssh"
exec 8>"$HOME/c4/.fetch.lock"
flock -n 8 || die 'fetch is already running'
chmod 700 "$HOME/.ssh"
if [[ ! -f $RELAY_KEY ]]; then ssh-keygen -q -t ed25519 -N '' -f "$RELAY_KEY" -C c4-box-relay; fi
[[ -f $RELAY_KEY.pub ]] || ssh-keygen -y -f "$RELAY_KEY" >"$RELAY_KEY.pub"
echo 'Authorize this PUBLIC key on relay before fetching:'
cat "$RELAY_KEY.pub"
relay_config
ARCHIVE=$HOME/c4/downloads/$ARCHIVE_ID.tar.gz
retry_transfer rsync -a --partial --timeout=120 -e "$RSYNC_SSH" \
    "$RELAY_HOST:$RELAY_DIR/$ARCHIVE_ID.tar.gz.sha256" "$ARCHIVE.sha256"
retry_transfer rsync -a --partial --timeout=120 -e "$RSYNC_SSH" \
    "$RELAY_HOST:$RELAY_DIR/$ARCHIVE_ID.tar.gz" "$ARCHIVE"
python3 "$HERE/ops.py" extract "$ARCHIVE" "$REPO"
echo "verified code installed: $REPO"
