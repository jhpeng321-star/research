#!/bin/bash
# A38 laptop: build the box bundle (zero physics). Inputs: the a32 relay stage (frozen plant source, e3final runtime,
# MAPPO d9042cc5, p20c bank-root bindings; the package gpuB ran amendments 32/33/35 from), src7.tar (launchcoop_src7
# symlink targets), the c4/a38-road2 overlay (36 files changed since 5ecc4358 incl. the wz2to1 map assets), the sealed
# p20c U0/LT first-wave records for the plant identity gate, and these box scripts. Output: OUT/a38_bundle.tar + sha.
set -euo pipefail
OUT=${1:-E:/agent-work/a38/bundle}
SRC=E:/agent-work/a38-road2
REL=E:/agent-work/a32/relay
REF=E:/research-dense/runs/active/C4-V7FT-20261017/paper_data/eval/nm_finaleval_p20c_bank
mkdir -p "$OUT/b/overlay/repo" "$OUT/b/p20c_ref" "$OUT/b/scripts"
HEAD=$(git -C "$SRC" rev-parse HEAD)
[ -z "$(git -C "$SRC" status --porcelain -- ops tools assets)" ] || { echo "worktree dirty: commit first"; exit 1; }
( cd "$SRC" && git diff --name-only 5ecc4358 HEAD | tar cf - -T - ) | tar xf - -C "$OUT/b/overlay/repo"
( cd "$SRC" && git diff --name-only 5ecc4358 HEAD ) > "$OUT/b/overlay/FILES.txt"
( cd "$OUT/b/overlay/repo" && find . -type f | sort | xargs sha256sum ) > "$OUT/b/overlay/SHA256SUMS"
echo "$HEAD" > "$OUT/b/overlay/HEAD"
for f in WAVE_U0__0000-0035.json WAVE_LEARNED_TRIGGER__0000-0023.json; do cp "$REF/$f" "$OUT/b/p20c_ref/"; done
cp E:/agent-work/a38/box/*.sh E:/agent-work/a38/box/*.py "$OUT/b/scripts/"
sed -i 's/\r$//' "$OUT/b/scripts/"*.sh
cp "$REL/stage.tar" "$REL/src7.tar" "$OUT/b/"
( cd "$OUT/b" && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256 )
tar cf "$OUT/a38_bundle.tar" -C "$OUT/b" .
sha256sum "$OUT/a38_bundle.tar" | tee "$OUT/a38_bundle.tar.sha256"
