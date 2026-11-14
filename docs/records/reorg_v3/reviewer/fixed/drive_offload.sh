#!/usr/bin/env bash
# Conservative backup-only review copy. No deletion authorization is emitted.
# Run in a Windows-path-aware Bash/Python/rclone environment (e.g. Git Bash).
# Each invocation selects ONE explicit item. Sealed/frozen-waiver subsets and
# worktree archives are omitted pending a real file manifest and owner decision.
# Dry-run makes no files, contacts no remote and runs no helper.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXECUTE=0; OWNER_IDLE=0; ITEM=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --owner-confirmed-idle) OWNER_IDLE=1; shift ;;
    --item) [ "$#" -ge 2 ] || exit 2; ITEM="$2"; shift 2 ;;
    *) echo "usage: $0 --item ID [--execute --owner-confirmed-idle]" >&2; exit 2 ;;
  esac
done
case "$ITEM" in
  partB_bundles) LOCAL="E:/research-dense/output/partB_bundles" ;;
  upload_v2|upload_v2c|raw_chunk1_p40|pack_v3|shards_v3)
    LOCAL="D:/rd-staging/wmn1_feat/$ITEM" ;;
  chunk1|chunk1_fix27|chunk2|index|chunk3|chunk4|truearms_chunk2|truearms_chunk3|walls_pilot_chunk4)
    LOCAL="D:/rd-staging/wmn1_feat/shards_v2/$ITEM" ;;
  cpu_payload) LOCAL="D:/rd-staging/cpu_payload" ;;
  *) echo "unknown or excluded item: $ITEM" >&2; exit 2 ;;
esac
# A separate raw-tree namespace avoids confusing existing .tar.zst archives
# with directory mirrors, and does not overwrite the old recovery archives.
REMOTE="gdrive:research-dense-cold/reorg_v3_reviewed_backups/$ITEM"
if [ "$EXECUTE" -eq 0 ]; then
  printf '[DRY] backup %s -> %s; checksum/size one-way check; retain local source\n' "$LOCAL" "$REMOTE"
  exit 0
fi
[ "$OWNER_IDLE" -eq 1 ] || { echo "Execute requires an owner-maintained zero-writer window" >&2; exit 2; }
command -v rclone >/dev/null || { echo "rclone required" >&2; exit 3; }
command -v python >/dev/null || { echo "python required" >&2; exit 3; }
HELPER="$SCRIPT_DIR/backup_receipt.py"
python "$HELPER" preflight "$LOCAL"
TS="$(date -u +%Y%m%dT%H%M%SZ)_$$_$RANDOM"
OUTDIR="$SCRIPT_DIR/journal/drive_$TS"
# Helper verifies the log parent and journals directory creation durably.
python "$HELPER" init "$SCRIPT_DIR" "$OUTDIR"
JOURNAL="$OUTDIR/receipt.jsonl"
python "$HELPER" record "$JOURNAL" before_copy "$LOCAL" "$REMOTE"
# No --immutable: that flag prevents repairing a truncated raw-tree object.
# --checksum allows resuming unchanged files; this new namespace is backup-only.
if ! rclone copy "$LOCAL" "$REMOTE" --checksum --transfers 2 --checkers 4 --retries 3 >"$OUTDIR/copy.log" 2>&1; then
  python "$HELPER" record "$JOURNAL" copy_failed "$LOCAL" "$REMOTE"; exit 1
fi
python "$HELPER" record "$JOURNAL" before_check "$LOCAL" "$REMOTE"
if ! rclone check "$LOCAL" "$REMOTE" --one-way --checkers 4 --combined "$OUTDIR/check.txt" >"$OUTDIR/check.log" 2>&1; then
  python "$HELPER" record "$JOURNAL" check_failed "$LOCAL" "$REMOTE"; exit 1
fi
# Default rclone check can fall back to size when a common hash is unavailable.
# Require a real MD5 for EVERY source member, compare remotely without download,
# then re-hash the local source to detect ordinary mid-backup changes.
rclone md5sum "$LOCAL" >"$OUTDIR/local_before.md5"
rclone md5sum "$REMOTE" >"$OUTDIR/remote.md5"
rclone md5sum "$LOCAL" >"$OUTDIR/local_after.md5"
python "$HELPER" compare "$OUTDIR/local_before.md5" "$OUTDIR/remote.md5" "$OUTDIR/local_after.md5"
python "$HELPER" record "$JOURNAL" backup_verified "$LOCAL" "$REMOTE"
printf 'Backup verified; local source retained. Receipt: %s\n' "$JOURNAL"
# backup_verified deliberately does NOT match the original deleter's verified.
