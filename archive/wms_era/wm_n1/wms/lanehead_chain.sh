#!/bin/bash
# Same C1 data, windows, multi-stage gate, fixed step budget and MAX_HOURS as
# old_hist_chain.sh, with only the history-conditioned lane head enabled.
set -u
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
W=${W:-/root/autodl-tmp/wms_oldhist_20261007/work}
mkdir -p "$W"
# The historical chain is CRLF-checked out on some Windows worktrees. Normalize
# its copied launch script in the writable run directory so Bash on the target
# Linux host sees valid LF input; source data and the workspace remain untouched.
sed 's/\r$//' "$HERE/old_hist_chain.sh" > "$W/lanehead_old_hist_chain.sh"
export W LANE_HEAD=history
exec bash "$W/lanehead_old_hist_chain.sh" "$@"
