#!/bin/bash
# Install the drop_raw fix on Box 4. Nothing is deleted: the first driver + test are kept beside the new ones as *.v1
# (same inode as before). New files are new inodes in /root/a75/src and /root/a75/src_c. Zero SUMO.
set -eu
SHIP=${1:?ship dir}
BASE=/root/a75
OLD=c49abc08ce8883c1f77cc7e24972aea3485ec72d3953228473546442ed71e1a8
sha() { sha256sum "$1" | cut -c1-64; }
[ -f "$BASE/DRIVER_FIX.sha256" ] && { echo "fix already installed"; exit 1; }
pgrep -f "[a]75_directhold.py" > /dev/null && { echo "a driver is running: not installing"; exit 1; }
( cd "$SHIP" && sha256sum -c SHA256SUMS ) || { echo "ship does not verify"; exit 1; }
for T in src src_c; do
  [ "$(sha "$BASE/$T/ops/a75_directhold.py")" = "$OLD" ] || { echo "$T: unexpected driver"; exit 1; }
  [ -e "$BASE/$T/ops/a75_directhold.py.v1" ] && { echo "$T: .v1 exists"; exit 1; }
done
for T in src src_c; do
  for f in a75_directhold.py test_a75_directhold.py; do
    mv "$BASE/$T/ops/$f" "$BASE/$T/ops/$f.v1"
    cp "$SHIP/$f" "$BASE/$T/ops/$f.new" && mv "$BASE/$T/ops/$f.new" "$BASE/$T/ops/$f"
    cmp "$SHIP/$f" "$BASE/$T/ops/$f"
  done
  [ "$(sha "$BASE/$T/ops/a75_directhold.py.v1")" = "$OLD" ]
done
cp "$SHIP/run_a75_resume.sh" "$BASE/run_a75_resume.sh"
echo "$(sha "$SHIP/a75_directhold.py")  a75_directhold.py" > "$BASE/DRIVER_FIX.sha256"
echo "FIX_INSTALLED driver $(sha "$BASE/src/ops/a75_directhold.py") test $(sha "$BASE/src/ops/test_a75_directhold.py") resume $(sha "$BASE/run_a75_resume.sh")"
