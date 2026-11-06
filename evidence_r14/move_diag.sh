#!/usr/bin/env bash
# S-R14 prep: archive finished gpuR diagnosis dirs to cpuN (sha256 verified), then delete on gpuR.
# diag_brake is EXCLUDED: diag_brake/d11_run5.py + cfworker5.py are live dependencies of the running 13E gate checks.
set -u
L=/root/autodl-tmp/c4/v5; A='~/c4/archive_gpuR_diag'
LOG=E:/agent-work/c4-v10r14/_r14ops/move_diag.log
verify() { ssh cpuN "cd $A && test -f $1.sha256 && nice -n 19 sha256sum -c --quiet $1.sha256 >/dev/null 2>&1 && find $1 -type l -printf '%p -> %l\n' | sort | cmp -s - $1.links && echo OK \$(wc -l < $1.sha256) \$(find $1 -type f | wc -l)"; }
ssh cpuN "mkdir -p $A"
for d in diag_long diag_yield diag_blowup diag_fact diag_gapacc diag_r12croll diag_cavcutin diag_r10; do
  echo "$(date) == $d" >> $LOG
  ssh gpuR "cd $L && test -d $d && ! test -L $d && find $d -type f -print0 | sort -z | xargs -0 sha256sum > /tmp/r14mv_$d.sha256 && find $d -type l -printf '%p -> %l\n' | sort > /tmp/r14mv_$d.links && wc -l < /tmp/r14mv_$d.sha256" >> $LOG 2>&1 || { echo "manifest fail $d" >> $LOG; continue; }
  ssh gpuR "cat /tmp/r14mv_$d.sha256" | ssh cpuN "cat > $A/$d.sha256.new"
  ssh gpuR "cat /tmp/r14mv_$d.links" | ssh cpuN "cat > $A/$d.links"
  ssh cpuN "cd $A && if cmp -s $d.sha256.new $d.sha256; then rm $d.sha256.new; echo same; else mv $d.sha256.new $d.sha256; echo new; fi" >> $LOG
  v=$(verify $d)
  if [ "${v%% *}" != OK ]; then
    ssh gpuR "cd $L && nice -n 19 tar czf - $d" | ssh cpuN "cd $A && nice -n 19 tar xzf -" || { echo "transfer fail $d" >> $LOG; continue; }
    v=$(verify $d)
  fi
  echo "verify $d: [$v]" >> $LOG
  set -- $v x x x
  if [ "$1" = OK ] && [ "$2" = "$3" ] && [ "$2" -gt 0 ]; then
    ssh gpuR "cd $L && test -d $d && rm -rf -- $L/$d && echo deleted $d" >> $LOG 2>&1
  else
    echo "NOT deleting $d" >> $LOG
  fi
done
ssh gpuR "df -h /root/autodl-tmp | tail -1" >> $LOG
echo "$(date) done" >> $LOG
