#!/usr/bin/env bash
# WM-E: ship the committed c4/wz-wm-train tree to gpuA as an immutable code dir (run locally, git bash).
#   bash ops/wzwm_deploy_gpua.sh [COMMIT]   -> /root/autodl-fs/wzwm/code_<sha12>
set -euo pipefail
C=$(git rev-parse "${1:-HEAD}"); S=${C:0:12}; DEST=/root/autodl-fs/wzwm/code_$S
TAR=$(mktemp -d)/code_$S.tar
PATHS="src tools configs traffic_shepherd pyproject.toml"
git cat-file -e "$C:assets/research/newmap_candidates" 2>/dev/null && PATHS="$PATHS assets/research/newmap_candidates"
git -c core.autocrlf=false -c core.eol=lf archive --format=tar "$C" $PATHS > "$TAR"   # slim (~30 MB), LF
SUM=$(sha256sum "$TAR" | cut -d' ' -f1)
for i in 1 2 3; do
  if ssh -o ConnectTimeout=20 -o BatchMode=yes gpuA "mkdir -p /root/autodl-fs/wzwm && cat > /root/autodl-fs/wzwm/code_$S.tar" < "$TAR"; then break; fi
  sleep $((20 * i))
done
ssh -o ConnectTimeout=20 -o BatchMode=yes gpuA "set -e; cd /root/autodl-fs/wzwm; echo \"$SUM  code_$S.tar\" | sha256sum -c -; \
  if [ ! -d $DEST ]; then mkdir $DEST.part && tar -xf code_$S.tar -C $DEST.part && echo $C > $DEST.part/CODE_SHA && mv $DEST.part $DEST; fi; \
  cat $DEST/CODE_SHA"
