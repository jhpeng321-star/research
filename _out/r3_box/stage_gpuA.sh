#!/usr/bin/env bash
# Local staging for WM-S-R3-20261010 on gpuA. Run from Git Bash in ws-r3-stage.
# Builds the code tar (git archive of the integrated branch HEAD) and the
# ~21.6 MB lpwo_v3 + sidecars_packv3_v2 bundle, hashes both, uploads once with
# scp, verifies the hashes on the box, unpacks into /root/autodl-tmp/wms_r3,
# and prints the exact single launch command. It NEVER launches training and
# NEVER loops on ssh: exactly one attempt per remote step; on any failure,
# stop and inspect (ssh drops are expected; never wrap ssh in retry loops).
# Usage:
#   STAGE_DRY_RUN=1 bash stage_gpuA.sh   # local build + verification only
#   bash stage_gpuA.sh                   # full staging (box must be ON)
set -Eeuo pipefail
unset MSYS_NO_PATHCONV || true

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
OUT=_out/r3_box
mkdir -p "$OUT"
die() { printf 'REFUSED: %s\n' "$*" >&2; exit 2; }

# Integrated HEAD: tracked files must be committed so the archive is exactly
# the reviewed tree. Untracked local files (e.g. _out artifacts) are allowed.
DIRTY=$(git status --porcelain --untracked-files=no)
[[ -z "$DIRTY" ]] || die "tracked changes present; commit first: $DIRTY"
HEAD=$(git rev-parse HEAD)
SHORT=${HEAD:0:12}
CODE_TAR=$OUT/wms_r3_code_${SHORT}.tar
DATA_TAR=$OUT/wms_r3_data_${SHORT}.tar

# Receipt-pinned integrity of the v3 lists (tracked BUILD_RECEIPT.json values).
sha256sum -c --quiet <<'EOF' || die 'lpwo_v3 lists fail their BUILD_RECEIPT hashes'
eab66b55b1a7ae978cd2790b4f4d4179d360909129eb061efde57ed44c00ab3c  _out/lpwo_v3/windows_train_noPI_noWall.json
cd078ddc18dc22468055a6f508c97260a8ce0f016f2c284d761070ee38ca13cd  _out/lpwo_v3/windows_val_noPI_noWall.json
7c42ae2025b75ad6c0a8856d269312ddb952ae973b000b92a3293f9e476d0169  _out/lpwo_v3/excluded_runs_PI_wall_v3.json
f294e520207fb1d8f2a064ac6f4bef3d5d4fd6e3006235223fd6b538a359498c  _out/lpwo_v3/eval_worlds.json
e7b102fc1f48251f393b6e545c0fd74a6e769fa8d5b9e3248043409f7d174db8  _out/lpwo_v3/holdout_eval_dropped_windows.json
EOF

# The pack_v3 sidecar payload (identities/ + outcomes/, ~21.1 MB) is not a git
# artifact; it must be placed under _out/sidecars_packv3_v2 first (verified
# ws-packv3fix bundle; see _out/R3_STAGE.md). Overridable for tests only.
SIDECARS_SRC=${SIDECARS_SRC:-$ROOT/_out/sidecars_packv3_v2}
[[ -d "$SIDECARS_SRC/identities" && -d "$SIDECARS_SRC/outcomes" ]] \
  || die "sidecar payload missing under $SIDECARS_SRC (identities/ + outcomes/); copy the verified ws-packv3fix bundle there first (see _out/R3_STAGE.md)"

# Receipt-pinned integrity of the payload (tracked BUILD_RECEIPT.json values).
sha256sum -c --quiet <<'EOF' || die 'sidecar payload fails BUILD_RECEIPT hashes'
1ce43bae52494d856f5877375804dc46e6c3d64ec72068f8161394693532c9de  _out/sidecars_packv3_v2/identities/identities.npy
bfdaf0de3aa78305a174bf60604881552d87a0c2b7eabc04dabe1193c9c59017  _out/sidecars_packv3_v2/identities/global_positions.npy
4c345b59caa2d70217470995e43cebf96d43af08622d4bb9b8fa1c16bc17330c  _out/sidecars_packv3_v2/identities/vehicle_map.json
46097207587f8c55cebc3dba35a5ccc96661f00118ec51cb7490e5be68f2f3d2  _out/sidecars_packv3_v2/identities/TRAIN_IDENTITY_RECEIPT.json
5d7de14bbf1aca698cde47450d1cc0ba30e3052d3fc2d47e252acdca34afbcd5  _out/sidecars_packv3_v2/outcomes/SHA256SUMS
f418173f995d5b1815367da5e55f17f358d4cfa4dbd5ecc77674505bbf0865ca  _out/sidecars_packv3_v2/outcomes/TRAIN_ISOLATION_RECEIPT.json
1e0ec6de0e2cdfd386fd84165879cac88b21d8cf1627d71108668d2306839ef  _out/sidecars_packv3_v2/outcomes/base_train_outcomes.parquet
705e8195d0cf523d74d2299f5fa960db5feed9a7ad68041613d0478a2def14dc  _out/sidecars_packv3_v2/outcomes/overlay_outcomes.parquet
EOF

printf 'building code tar (git archive %s)\n' "$SHORT"
git archive --format=tar --prefix=wms_r3/ -o "$CODE_TAR" HEAD
printf 'building data bundle tar (lpwo_v3 + sidecars_packv3_v2)\n'
tar -C _out --transform 's,^,wms_r3/,' -cf "$ROOT/$DATA_TAR" lpwo_v3 sidecars_packv3_v2

( cd "$OUT" && sha256sum "$(basename "$CODE_TAR")" "$(basename "$DATA_TAR")" > "SHA256SUMS_${SHORT}.txt" )
cat "$OUT/SHA256SUMS_${SHORT}.txt"

if [[ "${STAGE_DRY_RUN:-0}" == 1 ]]; then
  printf 'STAGE_DRY_RUN=1: local artifacts verified and ready; no ssh/scp performed.\n'
  exit 0
fi

# Upload once, verify on the box, unpack. One attempt per step; never retry.
scp -p "$CODE_TAR" "$DATA_TAR" "$OUT/SHA256SUMS_${SHORT}.txt" gpuA:/root/autodl-tmp/ \
  || die 'scp failed; inspect manually, do not loop'
ssh gpuA "cd /root/autodl-tmp && sha256sum -c SHA256SUMS_${SHORT}.txt" \
  || die 'box sha256 mismatch; inspect and remove the uploaded files manually'
ssh gpuA 'test ! -e /root/autodl-tmp/wms_r3' \
  || die '/root/autodl-tmp/wms_r3 already exists on gpuA; inspect or move it aside manually first'
ssh gpuA "tar -xf /root/autodl-tmp/wms_r3_code_${SHORT}.tar -C /root/autodl-tmp \
  && tar -xf /root/autodl-tmp/wms_r3_data_${SHORT}.tar -C /root/autodl-tmp \
  && mkdir -p /root/autodl-tmp/wms_r3/_out/r3_box /root/autodl-tmp/wms_train_r3" \
  || die 'box unpack failed; inspect manually'
scp -p "$OUT/r3_env.sh" gpuA:/root/autodl-tmp/wms_r3/_out/r3_box/r3_env.sh \
  || die 'env upload failed'

{
  echo "date_utc=$(date -u +%Y%m%dT%H%M%SZ)"
  echo "head=$HEAD"
  echo "code_tar=$(basename "$CODE_TAR")"
  echo "data_tar=$(basename "$DATA_TAR")"
  echo "sha256sums=SHA256SUMS_${SHORT}.txt"
  echo "box=gpuA code_root=/root/autodl-tmp/wms_r3 W=/root/autodl-tmp/wms_train_r3/r3_main_a1"
} > "$OUT/STAGE_RECEIPT_${SHORT}.txt"

cat <<'LAUNCH'
STAGED. Launch steps (box stays on; do NOT wrap ssh in retry loops):

  ssh gpuA                       # one interactive session
    bash -l
    source /root/autodl-tmp/wms_r3/_out/r3_box/r3_env.sh
    cd /root/autodl-tmp/wms_r3
    bash _out/train_main_chain.sh preflight

If preflight passes, launch EXACTLY ONCE (the single launch command):

  nohup setsid bash _out/train_main_chain.sh run > /root/autodl-tmp/wms_train_r3/launch_r3_main_a1.log 2>&1 &

If ssh drops after launching, do NOT relaunch: reconnect and check
/root/autodl-tmp/wms_train_r3/r3_main_a1/chain.log instead.
LAUNCH
