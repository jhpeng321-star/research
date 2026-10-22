#!/usr/bin/env bash
# Ship the round-9 code delta to gpuR into a FRESH repo dir (never overwrites, never deletes on gpuR).
#
#   scripts/ship_r9_gpuR.sh <local-commit> [--local-only | --stage-only] [--skip-base-check] [--base <commit>]
#   --local-only: build + self-verify the tarball locally, never contact gpuR.
#
# What it does (run from any directory inside the local worktree, Git Bash):
#   1. Resolves <local-commit> -> SHA8 and the BASE commit (default 21ca456f = round-8 code; gpuR's
#      repo_r9_base matches it on every src/configs/tools/scripts file; see R9_LAUNCH_PREP.md).
#   2. (default) Verifies on gpuR that repo_r9_base still matches BASE: LF-normalised md5 of every text file
#      under src configs tools scripts tests. Mismatch in src/ or configs/ aborts.
#   3. Delta = tracked files added/modified between BASE and <local-commit>. Blobs are read with
#      `git cat-file blob` (byte exact, LF as committed, no autocrlf/eol conversion) into a staging dir,
#      tarred, and an md5 manifest is written. Deleted files are only reported (not applied).
#   4. scp tar+manifest to gpuR:/root/autodl-tmp/c4/v5/ship_r9_<SHA8>/.
#   5. Full mode: refuses if repo_r9_<SHA8> exists; checks free disk; `cp -a repo_r9_base repo_r9_<SHA8>`;
#      extracts the tar there; md5-verifies EVERY shipped file; byte-compiles src/wms (no GPU, no training).
#      --stage-only: extracts into ship_r9_<SHA8>/check/ only (no base copy, a few MB) and md5-verifies.
# Exit 0 only if every shipped file verifies. Nothing is launched.
set -euo pipefail

REMOTE=gpuR
L=/root/autodl-tmp/c4/v5
BASE=21ca456f
STAGE_ONLY=0
LOCAL_ONLY=0
SKIP_BASE=0
COMMIT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --stage-only) STAGE_ONLY=1;;
    --local-only) LOCAL_ONLY=1; SKIP_BASE=1;;
    --skip-base-check) SKIP_BASE=1;;
    --base) shift; BASE="$1";;
    -*) echo "unknown option $1" >&2; exit 64;;
    *) COMMIT="$1";;
  esac
  shift
done
[ -n "$COMMIT" ] || { echo "usage: $0 <local-commit> [--stage-only] [--skip-base-check] [--base <commit>]" >&2; exit 64; }

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
SHA=$(git rev-parse --short=8 "$COMMIT^{commit}")
BASE_FULL=$(git rev-parse "$BASE^{commit}")
git merge-base --is-ancestor "$BASE_FULL" "$SHA" || { echo "BASE $BASE is not an ancestor of $SHA" >&2; exit 65; }
WORK="$ROOT/runs/active/ship_r9_$SHA"
[ ! -e "$WORK" ] || { echo "local staging dir exists: $WORK (choose a new commit or remove it yourself)" >&2; exit 66; }
mkdir -p "$WORK/files"
echo "[ship] commit=$SHA base=${BASE_FULL:0:8} remote=$REMOTE stage_only=$STAGE_ONLY"

PYBIN=${PYBIN:-C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe}
[ -x "$PYBIN" ] || PYBIN=python

# ---- 2. base check -------------------------------------------------------------------------------------
if [ "$SKIP_BASE" = 0 ]; then
  git ls-tree -r --name-only "$BASE_FULL" -- src configs tools scripts tests \
    | grep -v -E '\.(npz|npy|pt|pkl|pyc|png|jpg)$' > "$WORK/base_paths.txt"
  # One `git cat-file --batch` process (no per-file forks; forks failed intermittently on this box). Blobs may
  # legitimately contain CRLF: normalise like the remote side does (CRLF -> LF) before hashing.
  "$PYBIN" - "$BASE_FULL" "$WORK/base_paths.txt" "$WORK/base_md5.txt" <<'PY'
import hashlib, subprocess, sys
base, paths_file, out_file = sys.argv[1:4]
paths = [p.rstrip("\n") for p in open(paths_file, encoding="utf-8") if p.strip()]
crlf, lf = bytes([13, 10]), bytes([10])
proc = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
with open(out_file, "w", encoding="utf-8", newline="\n") as out:
    for p in paths:
        proc.stdin.write((base + ":" + p + "\n").encode()); proc.stdin.flush()
        header = proc.stdout.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            sys.exit("cannot read blob %s:%s: %r" % (base, p, header))
        data = proc.stdout.read(int(header[2])); proc.stdout.read(1)
        out.write("%s  %s\n" % (hashlib.md5(data.replace(crlf, lf)).hexdigest(), p))
proc.stdin.close(); proc.wait()
PY
  scp -q "$WORK/base_md5.txt" "$REMOTE:/tmp/r9_base_md5.txt"
  ssh "$REMOTE" "cd $L/repo_r9_base && /root/miniconda3/bin/python - <<'PY'
import hashlib, sys
bad = []
missing = []
for line in open('/tmp/r9_base_md5.txt'):
    md5, path = line.rstrip('\n').split('  ', 1)
    try:
        data = open(path, 'rb').read().replace(b'\r\n', b'\n')
    except FileNotFoundError:
        missing.append(path); continue
    if hashlib.md5(data).hexdigest() != md5:
        bad.append(path)
print('base-check: mismatched', len(bad), 'missing', len(missing))
for p in bad: print('  MISMATCH', p)
for p in missing: print('  MISSING ', p)
hard = [p for p in bad if p.startswith(('src/', 'configs/'))]
sys.exit(1 if hard else 0)
PY" || { echo "[ship] base check FAILED (src/configs differ from $BASE); inspect above" >&2; exit 67; }
fi

# ---- 3. delta + tar + manifest --------------------------------------------------------------------------
git diff --name-status --no-renames "$BASE_FULL" "$SHA" > "$WORK/delta_status.txt"
awk '$1=="A"||$1=="M"{print $2}' "$WORK/delta_status.txt" > "$WORK/ship_paths.txt"
awk '$1=="D"{print $2}' "$WORK/delta_status.txt" > "$WORK/deleted_not_applied.txt"
N=$(wc -l < "$WORK/ship_paths.txt")
echo "[ship] shipping $N files; $(wc -l < "$WORK/deleted_not_applied.txt") deletions NOT applied (listed in deleted_not_applied.txt)"
: > "$WORK/MANIFEST.md5"
while IFS= read -r p; do
  mkdir -p "$WORK/files/$(dirname "$p")"
  git cat-file blob "$SHA:$p" > "$WORK/files/$p"
  printf '%s  %s\n' "$(md5sum < "$WORK/files/$p" | cut -d' ' -f1)" "$p" >> "$WORK/MANIFEST.md5"
done < "$WORK/ship_paths.txt"
# shell scripts must be LF on gpuR
while IFS= read -r p; do
  if grep -q -P '\r$' "$WORK/files/$p"; then echo "[ship] ERROR: CRLF in shipped $p" >&2; exit 68; fi
done < <(grep -E '\.sh$' "$WORK/ship_paths.txt" || true)
( cd "$WORK/files" && tar --force-local --no-same-owner -cf "../r9_delta_$SHA.tar" -T "../ship_paths.txt" )
TARBYTES=$(stat -c %s "$WORK/r9_delta_$SHA.tar")
echo "[ship] tar bytes=$TARBYTES"
# local self-check: extract to scratch and verify manifest
mkdir -p "$WORK/selfcheck"
tar --force-local -xf "$WORK/r9_delta_$SHA.tar" -C "$WORK/selfcheck"
( cd "$WORK/selfcheck" && md5sum -c --quiet "$WORK/MANIFEST.md5" ) || { echo "[ship] local tar self-check FAILED" >&2; exit 73; }
[ "$(tar --force-local -tf "$WORK/r9_delta_$SHA.tar" | wc -l)" = "$N" ] || { echo "[ship] tar member count != $N" >&2; exit 74; }
echo "[ship] local tar self-check OK ($N files)"

if [ "$LOCAL_ONLY" = 1 ]; then echo "[ship] local-only: tar at $WORK/r9_delta_$SHA.tar; nothing sent."; exit 0; fi

# ---- 4. upload ------------------------------------------------------------------------------------------
RDIR="$L/ship_r9_$SHA"
ssh "$REMOTE" "mkdir $RDIR" || { echo "[ship] remote $RDIR exists; refusing to reuse" >&2; exit 69; }
scp -q "$WORK/r9_delta_$SHA.tar" "$WORK/MANIFEST.md5" "$WORK/ship_paths.txt" "$REMOTE:$RDIR/"
LOCAL_TAR_MD5=$(md5sum < "$WORK/r9_delta_$SHA.tar" | cut -d' ' -f1)
REMOTE_TAR_MD5=$(ssh "$REMOTE" "md5sum < $RDIR/r9_delta_$SHA.tar | cut -d' ' -f1")
[ "$LOCAL_TAR_MD5" = "$REMOTE_TAR_MD5" ] || { echo "[ship] tar md5 mismatch after scp" >&2; exit 70; }

# ---- 5. extract + verify --------------------------------------------------------------------------------
if [ "$STAGE_ONLY" = 1 ]; then
  ssh "$REMOTE" "set -e; mkdir $RDIR/check; tar -xf $RDIR/r9_delta_$SHA.tar -C $RDIR/check; cd $RDIR/check; md5sum -c --quiet ../MANIFEST.md5 && echo '[ship] remote stage-only verify OK ($N files)'; df -h /root/autodl-tmp | tail -1"
  echo "[ship] stage-only done; no repo dir created."
  exit 0
fi
ssh "$REMOTE" "set -euo pipefail
cd $L
DEST=repo_r9_$SHA
[ ! -e \$DEST ] || { echo \"\$DEST exists; refusing\"; exit 71; }
NEED_KB=\$(( \$(du -sk repo_r9_base | cut -f1) + \$(du -sk $RDIR | cut -f1) + 524288 ))
AVAIL_KB=\$(df -k /root/autodl-tmp | tail -1 | awk '{print \$4}')
echo \"need ~\${NEED_KB} KB, avail \${AVAIL_KB} KB\"
[ \$AVAIL_KB -gt \$NEED_KB ] || { echo 'insufficient disk'; exit 72; }
cp -a repo_r9_base \$DEST
tar -xf $RDIR/r9_delta_$SHA.tar -C \$DEST
cd \$DEST && md5sum -c --quiet $RDIR/MANIFEST.md5 && echo \"[ship] all $N shipped files md5-verified in \$DEST\"
PYTHONPATH=src:. /root/miniconda3/bin/python -m compileall -q src/wms >/dev/null && echo '[ship] compileall src/wms OK'
df -h /root/autodl-tmp | tail -1"
echo "[ship] done: $L/repo_r9_$SHA ready. Nothing launched."
