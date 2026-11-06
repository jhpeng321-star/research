#!/usr/bin/env bash
# V9ENTRY policy on WM v5m (2026-10-18).  OWNER APPROVAL REQUIRED before running (multi-hour GPU run).  LF only.
# Mirrors the v8 policy run (runs/c4_policy_v8_on_v5m: launch_policy_v5.sh, configs/c4_policy_v8.toml, v5m,
# 600 updates, C4_GPU_MEM_FRACTION 0.6, WMS_DATA_V6_POLICY=loop_r2/data/v6_policy_iter004) except
# trainer.entry_queue_state = true and the start sources' queue-array copies (v5_native_q, genrand_plain_q).
#
#   bash scripts/c4/launch_policy_v9entry.sh [OUT] [extra train_cli args]
#   (default OUT /root/autodl-tmp/c4/v5/runs/c4_policy_v9entry_on_v5m; CONFIG/C4_GPU_MEM_FRACTION overridable)
if LC_ALL=C grep -q $'\r' "${BASH_SOURCE[0]}"; then echo 'REFUSED: CRLF bytes in script' >&2; exit 2; fi
set -Eeuo pipefail
L=${C4_V5_ROOT:-/root/autodl-tmp/c4/v5}
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${1:-$L/runs/c4_policy_v9entry_on_v5m}; shift || true
export WMS_DATA_V5_NATIVE_Q=$L/data/v5_native_q
export WMS_DATA_GENRAND_PLAIN_Q=$L/data/genrand_plain_q
export WMS_DATA_V6_POLICY=$L/loop_r2/data/v6_policy_iter004
for d in "$WMS_DATA_V5_NATIVE_Q" "$WMS_DATA_GENRAND_PLAIN_Q" "$WMS_DATA_V6_POLICY"; do
  [[ -f "$d/C4_EPISODES_MANIFEST.json" ]] || { echo "REFUSED: no C4 source at $d" >&2; exit 2; }
done
export CONFIG=${CONFIG:-configs/c4_policy_v9entry_v5m.toml}
export C4_GPU_MEM_FRACTION=${C4_GPU_MEM_FRACTION:-0.6}
WM_CKPT=${WM_CKPT:-$L/wm_v5m/wm_s_best_ap.pt}
bash "$REPO/scripts/c4/launch_policy_v5.sh" "$WM_CKPT" "$OUT" --max-updates "${MAX_UPDATES:-600}" "$@"
{
  echo "WMS_DATA_V5_NATIVE_Q=$WMS_DATA_V5_NATIVE_Q"
  echo "WMS_DATA_GENRAND_PLAIN_Q=$WMS_DATA_GENRAND_PLAIN_Q"
  echo "WMS_DATA_V6_POLICY=$WMS_DATA_V6_POLICY"
  echo "queue_manifest_sha256 v5_native_q=$(sha256sum "$WMS_DATA_V5_NATIVE_Q/C4_EPISODES_MANIFEST.json" | cut -d' ' -f1)"
  echo "queue_manifest_sha256 genrand_plain_q=$(sha256sum "$WMS_DATA_GENRAND_PLAIN_Q/C4_EPISODES_MANIFEST.json" | cut -d' ' -f1)"
} >> "$OUT/LAUNCH.txt"
