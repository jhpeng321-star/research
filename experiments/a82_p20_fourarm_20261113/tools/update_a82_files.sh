#!/bin/bash
# Replace ONLY the A82-owned driver/test/run files in the two A82 bases with the shipped versions (zero SUMO); logged.
set -uo pipefail
SHIP=/root/a82_ship
( cd $SHIP && sha256sum -c --quiet SHIP_FILES.sha256 ) || { echo "ship check failed"; exit 1; }
for BASE in /root/a82p20 /root/a82four; do
  [ -f $BASE/SETUP_OK ] || continue
  for f in a82_s3b20.py test_a82_s3b20.py a82_fourarm.py test_a82_fourarm.py run_a82.sh; do
    T=$BASE/src/ops/$f
    [ "$f" = run_a82.sh ] && T=$BASE/run_a82.sh
    if cmp -s "$SHIP/$f" "$T"; then continue; fi
    old=$(sha256sum "$T" | cut -c1-64)
    cp "$SHIP/$f" "$T"
    echo "$(date '+%F %T') $f $old -> $(sha256sum "$T" | cut -c1-64) (ship $(sha256sum /tmp/a82_ship.tgz | cut -c1-12))" >> $BASE/UPDATES.log
  done
  echo "== $BASE"; tail -5 $BASE/UPDATES.log 2>/dev/null
done
