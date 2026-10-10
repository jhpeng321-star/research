#!/usr/bin/env bash
# Fetch WM-S-R3-20261010 results from gpuA and prepare shutdown (local script,
# Git Bash). Streams the finished run dir back as ONE uncompressed tar, hashes
# it locally, compares against a fresh box-side stream hash, and then only
# PRINTS the shutdown command - it never runs it. One attempt per remote step,
# no loops, no kills; on mismatch the artifact is retained and the script
# stops with exit 2. Uncompressed tar keeps the two streams byte-comparable
# (gzip -f would embed fresh timestamps and break the comparison; the run is
# finished, so two back-to-back tar streams of the same tree are identical).
# Usage:
#   bash fetch_and_shutdown.sh                       # defaults below
#   RUN_NAME=... FETCH_DEST=... bash fetch_and_shutdown.sh
set -Eeuo pipefail
unset MSYS_NO_PATHCONV || true

RUN_NAME=${RUN_NAME:-r3_main_a1}
REMOTE_DIR=/root/autodl-tmp/wms_train_r3
DEST=${FETCH_DEST:-/e/research-dense/runs/active/WM-S-R3-20261010/box/gpuA}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
TAR_PATH=$DEST/gpuA_${RUN_NAME}_${STAMP}.tar

die() { printf 'REFUSED: %s\n' "$*" >&2; exit 2; }
mkdir -p "$DEST"
[[ ! -e "$TAR_PATH" ]] || die "output exists: $TAR_PATH; never overwrite a prior fetch"

ssh gpuA "test -d $REMOTE_DIR/$RUN_NAME" \
  || die "no run dir on gpuA: $REMOTE_DIR/$RUN_NAME"
if ssh gpuA "pgrep -f 'wms\.cli train' >/dev/null"; then
  die "a wms.cli train process is still running on gpuA; fetch only after the run finished"
fi

printf 'streaming %s/%s -> %s\n' "$REMOTE_DIR" "$RUN_NAME" "$TAR_PATH"
ssh gpuA "cd $REMOTE_DIR && tar -cf - $RUN_NAME" > "$TAR_PATH" \
  || die 'stream failed; partial file retained for inspection'

LOCAL_HASH=$(sha256sum "$TAR_PATH" | awk '{print $1}')
BOX_HASH=$(ssh gpuA "cd $REMOTE_DIR && tar -cf - $RUN_NAME | sha256sum" | awk '{print $1}')
printf 'local sha256 %s\nbox   sha256 %s\n' "$LOCAL_HASH" "$BOX_HASH"
if [[ "$LOCAL_HASH" == "$BOX_HASH" ]]; then
  printf '%s  %s\n' "$LOCAL_HASH" "$(basename "$TAR_PATH")" > "$TAR_PATH.sha256"
  printf 'VERIFY: MATCH (%s bytes)\n' "$(wc -c < "$TAR_PATH")"
else
  printf 'VERIFY: MISMATCH; artifact retained; do not trust it and do not shut the box down on this basis\n' >&2
  exit 2
fi

# Optional tiny launch log next to the run dir (single best-effort attempt).
scp -q "gpuA:$REMOTE_DIR/launch_${RUN_NAME}.log" "$DEST/" 2>/dev/null \
  || printf 'note: optional launch log not fetched\n'

cat <<'SHUTDOWN'
Fetch verified. To release the box, run this yourself when ready
(this script deliberately does not run it):

  ssh gpuA 'sudo shutdown -h now'      # or power off from the AutoDL console

Then record the sha256 above and the artifact path in
E:/research-dense/runs/active/WM-S-R3-20261010/REGISTRATION.md.
SHUTDOWN
