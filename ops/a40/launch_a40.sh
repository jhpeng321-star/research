#!/bin/bash
# AMENDMENT 40 one-command launcher (laptop, Git Bash):  bash ops/a40/launch_a40.sh HOST PORT [USER]
#   (PORT "-" = use the ssh config alias HOST as is; SSH_KEY=<file> adds -i <file>; GO=1 also touches A40_GO = MAIN's
#    explicit decision to start without waiting for A37_ALL_DONE)
#   1. ship dir from the COMMITTED branch tip (git blobs): overlay.tar (ops/* + tools/coopmwm/* changed since the
#      plant overlay 5ecc4358 -> ops/, pytools/tools/; the wz3to2v80 map -> pytools/assets/), setup_a40.sh, run_a40.sh,
#      the a32 relay stage.tar / src7.tar, the DUO bank, the sealed p20c first-wave records (identity gate refs)
#   2. upload into /tmp/a40ship_<id> (a file is skipped when the remote copy has the same sha; stage.tar / src7.tar are
#      copied host-side from an A37 ship dir when one with the same sha exists), setup_a40.sh DETACHED + polling
#   3. run_a40.sh DETACHED exactly once (refused when /root/a40gv/chain.out exists); it WAITS for the A37 chain to end
#      (A37_ALL_DONE) unless A40_GO exists. Separate root /root/a40gv; nothing under /root/a37ct is written.
# Never launches inside a retry loop; never deletes anything remote; no passwords (key auth only).
set -uo pipefail
HOST=${1:?HOST}; PORT=${2:?PORT}; USR=${3:-root}
WT=E:/agent-work/a40-geomvar
A=E:/agent-work/a40
RELAY=E:/agent-work/a32/relay
REF=E:/research-dense/runs/active/C4-V7FT-20261017/paper_data/eval/nm_finaleval_p20c_bank
DUO=E:/agent-work/a32/stage/nm_finaleval_p20c_bank/a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json
PP=""; [ "$PORT" = - ] || PP="-p $PORT"; PS=""; [ "$PORT" = - ] || PS="-P $PORT"
OPT="${SSH_KEY:+-i $SSH_KEY} -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=accept-new"
SSH="ssh $PP $OPT $USR@$HOST"
log() { echo "$(date '+%F %T') LAUNCH $*" | tee -a "$A/PROGRESS.md"; }
die() { log "BLOCKED: $*"; exit 1; }

git -C "$WT" diff --quiet HEAD -- ops tools/coopmwm assets/research/newmap_candidates/wz3to2v80 \
  || die "uncommitted changes in ops/, tools/coopmwm or the wz3to2v80 map (commit first)"
COMMIT=$(git -C "$WT" rev-parse HEAD); OVL=${COMMIT:0:12}
SHIP=$A/ship_$OVL
FILES="SHA256SUMS OVERLAY_ID setup_a40.sh run_a40.sh overlay.tar GUIDANCE_C4SIT_priv_UNION_UNC.json p20c_ref.tar src7.tar stage.tar"
if [ ! -f "$SHIP/SHA256SUMS" ]; then
  STG=$SHIP/overlay_src
  mkdir -p "$STG"
  for f in $( (git -C "$WT" diff --name-only --diff-filter=d 5ecc4358 "$COMMIT" -- ops tools/coopmwm;
               git -C "$WT" ls-tree -r --name-only "$COMMIT" -- ops/a40 assets/research/newmap_candidates/wz3to2v80) | sort -u); do
    case "$f" in tools/coopmwm/*|assets/*) d="pytools/$f" ;; ops/*) d="$f" ;; *) continue ;; esac
    mkdir -p "$STG/$(dirname "$d")"
    git -C "$WT" show "$COMMIT:$f" > "$STG/$d" || die "git show $f"
  done
  [ "$(sha256sum "$STG/pytools/assets/research/newmap_candidates/wz3to2v80/wz3to2v80.net.xml" | cut -c1-64)" = \
    1fed6f59cfcfa63d2e30f1344eac5e59a1597eca77c7d9edbae3e0bc08161f03 ] || die "shipped wz3to2v80 net bytes differ"
  (cd "$STG" && tar --force-local -cf ../overlay.tar .) || die "overlay tar"
  git -C "$WT" show "$COMMIT:ops/a40/host/setup_a40.sh" > "$SHIP/setup_a40.sh"
  git -C "$WT" show "$COMMIT:ops/a40/host/run_a40.sh" > "$SHIP/run_a40.sh"
  mkdir -p "$SHIP/p20c_ref" && cp "$REF/WAVE_U0__0000-0035.json" "$REF/WAVE_LEARNED_TRIGGER__0000-0023.json" "$SHIP/p20c_ref/"
  (cd "$SHIP" && tar --force-local -cf p20c_ref.tar p20c_ref) || die "p20c_ref tar"
  cp "$RELAY/stage.tar" "$RELAY/src7.tar" "$DUO" "$SHIP/" || die "relay tars / DUO bank"
  echo "$OVL" > "$SHIP/OVERLAY_ID"
  (cd "$SHIP" && sha256sum overlay.tar setup_a40.sh run_a40.sh stage.tar src7.tar GUIDANCE_C4SIT_priv_UNION_UNC.json \
     p20c_ref.tar OVERLAY_ID | sed 's/ \*/  /' > SHA256SUMS)
  grep -q $'\r' "$SHIP/setup_a40.sh" "$SHIP/run_a40.sh" && die "CRLF in shipped .sh"
  log "ship $SHIP built from $COMMIT"
fi

ID=$($SSH 'id -u' 2>/dev/null) || die "ssh $USR@$HOST:$PORT failed (key auth required; no passwords)"
SUDO=""; [ "$ID" = 0 ] || { $SSH 'sudo -n true' >/dev/null 2>&1 || die "non-root user without passwordless sudo"; SUDO="sudo -n"; }
log "host $HOST:$PORT user $USR uid $ID nproc $($SSH nproc) $($SSH "df -Ph / | tail -1")"
log "A37 state: $($SSH "$SUDO tail -1 /root/a37ct/chain.out 2>/dev/null" || echo none)"
if $SSH "$SUDO test -f /root/a40gv/chain.out"; then
  log "chain.out already exists on the host: NOT relaunching (status only)"
  $SSH "$SUDO tail -5 /root/a40gv/chain.out"
  [ "${GO:-0}" = 1 ] && { $SSH "$SUDO touch /root/a40gv/A40_GO" && log "A40_GO touched (MAIN decision)"; }
  exit 0
fi
RS=/tmp/a40ship_$OVL
$SSH "mkdir -p $RS" || die "remote mkdir"
for f in $FILES; do
  want=$(grep " $f\$" "$SHIP/SHA256SUMS" | cut -c1-64)
  have=$($SSH "sha256sum $RS/$f 2>/dev/null | cut -c1-64")
  if [ -n "$want" ] && [ "$want" = "$have" ]; then continue; fi
  if [ -n "$want" ]; then          # host-side copy from an A37 ship dir with the identical file
    src=$($SSH "for x in /tmp/a37ship_*/$f; do [ -f \"\$x\" ] && [ \"\$(sha256sum \"\$x\" | cut -c1-64)\" = $want ] && { echo \"\$x\"; break; }; done" 2>/dev/null)
    if [ -n "$src" ] && $SSH "cp '$src' $RS/$f"; then log "host-side copy $f from $src"; continue; fi
  fi
  scp $PS $OPT "$SHIP/$f" "$USR@$HOST:$RS/$f" || die "upload $f"
done
$SSH "cd $RS && tar xf p20c_ref.tar && sha256sum -c --quiet SHA256SUMS" || die "remote ship sha check"
log "uploaded to $RS"
if ! $SSH "$SUDO test -f /root/a40gv/SETUP_OK"; then
  $SSH "$SUDO setsid nohup bash $RS/setup_a40.sh > /tmp/a40_setup.out 2>&1 < /dev/null & echo setup-started" || die "setup launch"
  log "setup_a40.sh started (detached); polling SETUP_OK / SETUP_FAILED"
  for i in $(seq 1 120); do
    sleep 30
    if $SSH "$SUDO test -f /root/a40gv/SETUP_OK" 2>/dev/null; then break; fi
    if $SSH "$SUDO test -f /root/a40gv/SETUP_FAILED" 2>/dev/null; then
      $SSH "$SUDO tail -40 /root/a40gv/setup.log"; die "host setup failed: $($SSH "$SUDO cat /root/a40gv/SETUP_FAILED")"; fi
  done
  $SSH "$SUDO test -f /root/a40gv/SETUP_OK" || die "setup did not finish in 60 min (see /root/a40gv/setup.log)"
fi
$SSH "$SUDO tail -6 /root/a40gv/setup.log"
[ "${GO:-0}" = 1 ] && { $SSH "$SUDO touch /root/a40gv/A40_GO" && log "A40_GO touched (MAIN decision: start without waiting for A37)"; }
$SSH "$SUDO cp $RS/run_a40.sh /root/a40gv/run_a40.sh && $SUDO setsid nohup bash /root/a40gv/run_a40.sh > /root/a40gv/chain.nohup 2>&1 < /dev/null & echo chain-started" \
  || die "chain launch (check the host before any relaunch)"
sleep 5
log "chain launched on $HOST:$PORT (overlay $OVL): pid $($SSH "$SUDO cat /root/a40gv/chain.pid 2>/dev/null"); waits for A37_ALL_DONE or A40_GO"
echo "offloader: nohup bash $WT/ops/a40/offload_a40.sh $HOST $PORT $USR >> $A/offload.log 2>&1 &"
