#!/usr/bin/env bash
# =============================================================================
# drive_offload.sh -- REORG_PLAN_V3 Google-Drive offload + server-side verify.
#
# For every item listed below:
#   1. rclone copy <local> <remote>          (upload; skipped if already done)
#   2. rclone check <local> <remote> --one-way   (server-side checksum compare)
#   3. append a verification record to the manifest
#
# Local deletion is NEVER done here. DELETE_AFTER_VERIFY.ps1 reads the manifest
# and deletes only items whose check passed.
#
# SAFETY
#   * DRY-RUN BY DEFAULT: pass --execute to actually upload/verify.
#   * rclone.conf / credential files are never opened by this script.
#   * No `rclone sync` and no `--delete-*` flags are ever used.
#   * Hard-coded item list; no shell globbing of sources.
#   * Does not touch sealed/CAL/HOLDOUT/chunk-5/J4B/INC1 contents: sealed is
#     uploaded as OPAQUE BYTES only (rclone reads bytes, never opens contents).
#
# Delivered UNEXECUTED (planning only).
# =============================================================================
set -uo pipefail

REMOTE="gdrive"
ROOT="gdrive:research-dense-cold"
OUTDIR="${OUTDIR:-E:/rd-wt/reorg3/architect/journal}"
EXECUTE=0
for a in "$@"; do
  case "$a" in
    --execute) EXECUTE=1 ;;
    -h|--help) echo "usage: $0 [--execute]"; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

TS="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUTDIR"
MANIFEST="$OUTDIR/drive_verify_${TS}.jsonl"
echo "{\"manifest\":\"$MANIFEST\",\"execute\":$EXECUTE,\"ts\":\"$TS\"}" > "$MANIFEST"

if ! command -v rclone >/dev/null 2>&1; then
  echo "FATAL: rclone not installed / not on PATH" >&2
  exit 3
fi
if ! rclone listremotes 2>/dev/null | grep -qx "${REMOTE}:"; then
  echo "FATAL: rclone remote '${REMOTE}:' is not configured" >&2
  exit 3
fi

# item = local|remote|expected_bytes|note
ITEMS=(
"E:/research-dense-c-migrated|$ROOT/01_Code_And_Restore/research-dense-c-migrated|59400761|C:->E: migration copy"
"E:/research-dense/output/partB_bundles|$ROOT/03_Experiment_Archives/partB_bundles|563055757|superseded bundles; keep shipped zips+sha downstream"
"E:/research-dense/output/evidence_regeneration_v1|$ROOT/03_Experiment_Archives/evidence_regeneration_v1|24079558957|OWNER-GATED frozen evidence"
"E:/research-dense/runs/campaign_stage1_chunk1_p40_superseded|$ROOT/03_Experiment_Archives/chunk1_p40_superseded|0|OWNER-GATED (WM-42/44/56); archive only the superseded subset"
"E:/research-dense/sealed|$ROOT/04_Restricted_Archives_Do_Not_Open/sealed|66391419642|OWNER-GATED sealed; OPAQUE bytes only, never open contents"
"D:/rd-staging/wmn1_feat/upload_v2|$ROOT/02_World_Model_Data/wm_n1/upload_v2|6861918208|VERIFIED 7/7 2026-10-06"
"D:/rd-staging/wmn1_feat/upload_v2c|$ROOT/02_World_Model_Data/wm_n1/upload_v2c|6912930816|VERIFIED 7/7 2026-10-06"
"D:/rd-staging/wmn1_feat/raw_chunk1_p40|$ROOT/03_Experiment_Archives/raw_chunk1_p40|4888943104|VERIFIED 294/294 2026-10-06"
"D:/rd-staging/wmn1_feat/shards_v2/chunk1|$ROOT/03_Experiment_Archives/shards_v2/chunk1|6569813020|VERIFIED 2026-10-06"
"D:/rd-staging/wmn1_feat/shards_v2/chunk1_fix27|$ROOT/03_Experiment_Archives/shards_v2/chunk1_fix27|1174405120|VERIFIED 2026-10-06"
"D:/rd-staging/wmn1_feat/shards_v2/chunk2|$ROOT/03_Experiment_Archives/shards_v2/chunk2|6739553280|VERIFIED 2026-10-06"
"D:/rd-staging/wmn1_feat/shards_v2/index|$ROOT/03_Experiment_Archives/shards_v2/index|1048576|VERIFIED 2026-10-06"
"D:/rd-staging/wmn1_feat/shards_v2/chunk3|$ROOT/03_Experiment_Archives/shards_v2/chunk3|6733930496|TRUNCATED on Drive -> re-upload"
"D:/rd-staging/wmn1_feat/shards_v2/chunk4|$ROOT/03_Experiment_Archives/shards_v2/chunk4|10233461760|TRUNCATED on Drive -> re-upload"
"D:/rd-staging/wmn1_feat/shards_v2/truearms_chunk2|$ROOT/03_Experiment_Archives/shards_v2/truearms_chunk2|1569724928|TRUNCATED on Drive -> re-upload"
"D:/rd-staging/wmn1_feat/shards_v2/truearms_chunk3|$ROOT/03_Experiment_Archives/shards_v2/truearms_chunk3|1569724928|TRUNCATED on Drive -> re-upload"
"D:/rd-staging/wmn1_feat/shards_v2/walls_pilot_chunk4|$ROOT/03_Experiment_Archives/shards_v2/walls_pilot_chunk4|2448420864|TRUNCATED on Drive -> re-upload"
"D:/rd-staging/wmn1_feat/pack_v3|$ROOT/02_World_Model_Data/wm_n1/pack_v3|52403736576|not on Drive; pack_v3 mirrored gpuA/gpuB 18/18 sha"
"D:/rd-staging/wmn1_feat/shards_v3|$ROOT/02_World_Model_Data/wm_n1/shards_v3|54962298880|not on Drive"
"D:/rd-staging/cpu_payload|$ROOT/02_World_Model_Data/wm_n1/cpu_payload|1200834816|not on Drive"
)

verify_count=0; fail_count=0; skip_count=0
for item in "${ITEMS[@]}"; do
  IFS='|' read -r LOCAL REMOTE_PATH EXPECTED NOTE <<< "$item"
  if [ ! -e "$LOCAL" ]; then
    echo "{\"local\":\"$LOCAL\",\"status\":\"missing_local\",\"note\":\"$NOTE\"}" >> "$MANIFEST"
    skip_count=$((skip_count+1)); continue
  fi
  LOCAL_SIZE="$(stat -c %s "$LOCAL" 2>/dev/null || echo 0)"
  if [ "$EXPECTED" != "0" ] && [ "$LOCAL_SIZE" != "$EXPECTED" ]; then
    echo "WARN: $LOCAL size $LOCAL_SIZE != expected $EXPECTED" >&2
  fi

  if [ "$EXECUTE" -eq 0 ]; then
    echo "[DRY] copy  $LOCAL -> $REMOTE_PATH  ($LOCAL_SIZE bytes)"
    echo "[DRY] check $LOCAL  vs  $REMOTE_PATH  --one-way"
    echo "{\"local\":\"$LOCAL\",\"remote\":\"$REMOTE_PATH\",\"expected_bytes\":$EXPECTED,\"local_bytes\":$LOCAL_SIZE,\"status\":\"dry_run\",\"note\":\"$NOTE\"}" >> "$MANIFEST"
    continue
  fi

  rclone copy "$LOCAL" "$REMOTE_PATH" --checksum --immutable --transfers 4 --checkers 8 --retries 3
  COPY_RC=$?
  # server-side checksum comparison (one-way: local is the source of truth)
  rclone check "$LOCAL" "$REMOTE_PATH" --one-way --checkers 8 \
    --combined "$OUTDIR/check_$(echo "$LOCAL" | tr '/: ' '___').txt" >/dev/null 2>&1
  CHECK_RC=$?
  if [ "$COPY_RC" -eq 0 ] && [ "$CHECK_RC" -eq 0 ]; then
    STATUS="verified"; verify_count=$((verify_count+1))
  else
    STATUS="failed"; fail_count=$((fail_count+1))
  fi
  echo "{\"local\":\"$LOCAL\",\"remote\":\"$REMOTE_PATH\",\"expected_bytes\":$EXPECTED,\"local_bytes\":$LOCAL_SIZE,\"copy_rc\":$COPY_RC,\"check_rc\":$CHECK_RC,\"status\":\"$STATUS\",\"note\":\"$NOTE\",\"ts\":\"$TS\"}" >> "$MANIFEST"
done

echo "drive_offload done: verified=$verify_count failed=$fail_count skipped=$skip_count"
echo "verification manifest: $MANIFEST"
echo "Next: run DELETE_AFTER_VERIFY.ps1 (report-only) and review before -Execute."
