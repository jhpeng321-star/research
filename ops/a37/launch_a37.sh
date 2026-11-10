#!/bin/bash
# AMENDMENT 37 one-command launcher (laptop, Git Bash):  bash ops/a37/launch_a37.sh HOST PORT [USER]
#   (PORT "-" = use the ssh config alias HOST as is; SSH_KEY=<file> adds -i <file>)
#   1. ship dir from the COMMITTED branch tip (git blobs, LF): overlay.tar (ops/* + tools/coopmwm/* changed since the
#      plant overlay 5ecc4358, + ops/a37/*), setup_host.sh, run_a37.sh, the a32 relay stage.tar / src7.tar, the DUO bank
#   2. upload (skipped when the remote copy has the same sha), setup_host.sh DETACHED (setsid nohup) + status polling
#   3. the chain run_a37.sh DETACHED, launched exactly once (refused when a chain pid is alive or chain.out exists)
#   4. prints the offloader command (run it in the background on the laptop; it keeps the laptop copy current)
# Never launches inside a retry loop; never deletes anything remote.
set -uo pipefail
HOST=${1:?HOST}; PORT=${2:?PORT}; USR=${3:-root}
WT=E:/agent-work/a37-celltailor
A=E:/agent-work/a37
RELAY=E:/agent-work/a32/relay
PP=""; [ "$PORT" = - ] || PP="-p $PORT"; PS=""; [ "$PORT" = - ] || PS="-P $PORT"
OPT="${SSH_KEY:+-i $SSH_KEY} -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=accept-new"
SSH="ssh $PP $OPT $USR@$HOST"
log() { echo "$(date '+%F %T') LAUNCH $*" | tee -a "$A/PROGRESS.md"; }
die() { log "BLOCKED: $*"; exit 1; }

git -C "$WT" diff --quiet HEAD -- ops tools/coopmwm || die "uncommitted changes in ops/ or tools/coopmwm (commit first)"
COMMIT=$(git -C "$WT" rev-parse HEAD); OVL=${COMMIT:0:12}
SHIP=$A/ship_$OVL
if [ ! -f "$SHIP/SHA256SUMS" ]; then
  STG=$SHIP/overlay_src
  mkdir -p "$STG"
  for f in $( (git -C "$WT" diff --name-only --diff-filter=d 5ecc4358 "$COMMIT" -- ops tools/coopmwm;
               git -C "$WT" ls-tree -r --name-only "$COMMIT" -- ops/a37) | sort -u); do
    case "$f" in tools/coopmwm/*) d="pytools/$f" ;; ops/*) d="$f" ;; *) continue ;; esac
    mkdir -p "$STG/$(dirname "$d")"
    git -C "$WT" show "$COMMIT:$f" > "$STG/$d" || die "git show $f"
  done
  (cd "$STG" && tar --force-local -cf ../overlay.tar .) || die "overlay tar"
  git -C "$WT" show "$COMMIT:ops/a37/host/setup_host.sh" > "$SHIP/setup_host.sh"
  git -C "$WT" show "$COMMIT:ops/a37/host/run_a37.sh" > "$SHIP/run_a37.sh"
  cp "$RELAY/stage.tar" "$RELAY/src7.tar" "$SHIP/" || die "relay tars"
  cp E:/agent-work/a32/stage/nm_finaleval_p20c_bank/a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json "$SHIP/"
  echo "$OVL" > "$SHIP/OVERLAY_ID"
  (cd "$SHIP" && sha256sum overlay.tar setup_host.sh run_a37.sh stage.tar src7.tar GUIDANCE_C4SIT_priv_UNION_UNC.json OVERLAY_ID \
     | sed 's/ \*/  /' > SHA256SUMS)
  grep -q $'\r' "$SHIP/setup_host.sh" "$SHIP/run_a37.sh" && die "CRLF in shipped .sh"
  log "ship $SHIP built from $COMMIT"
fi

ID=$($SSH 'id -u' 2>/dev/null) || die "ssh $USR@$HOST:$PORT failed (key auth required; no passwords)"
SUDO=""; [ "$ID" = 0 ] || { $SSH 'sudo -n true' >/dev/null 2>&1 || die "non-root user without passwordless sudo"; SUDO="sudo -n"; }
log "host $HOST:$PORT user $USR uid $ID nproc $($SSH nproc) $($SSH "df -Ph / | tail -1")"
if $SSH "$SUDO test -f /root/a37ct/chain.out"; then
  log "chain.out already exists on the host: NOT relaunching (status only)"
  $SSH "$SUDO tail -5 /root/a37ct/chain.out"
  exit 0
fi
RS=/tmp/a37ship_$OVL
$SSH "mkdir -p $RS" || die "remote mkdir"
for f in SHA256SUMS OVERLAY_ID setup_host.sh run_a37.sh overlay.tar GUIDANCE_C4SIT_priv_UNION_UNC.json src7.tar stage.tar; do
  want=$(grep " $f\$" "$SHIP/SHA256SUMS" | cut -c1-64)
  have=$($SSH "sha256sum $RS/$f 2>/dev/null | cut -c1-64")
  if [ -n "$want" ] && [ "$want" = "$have" ]; then continue; fi
  scp $PS $OPT "$SHIP/$f" "$USR@$HOST:$RS/$f" || die "upload $f"
done
log "uploaded to $RS"
if ! $SSH "$SUDO test -f /root/a37ct/SETUP_OK"; then
  $SSH "$SUDO setsid nohup bash $RS/setup_host.sh > /tmp/a37_setup.out 2>&1 < /dev/null & echo setup-started" || die "setup launch"
  log "setup_host.sh started (detached); polling SETUP_OK / SETUP_FAILED"
  for i in $(seq 1 120); do
    sleep 30
    if $SSH "$SUDO test -f /root/a37ct/SETUP_OK" 2>/dev/null; then break; fi
    if $SSH "$SUDO test -f /root/a37ct/SETUP_FAILED" 2>/dev/null; then
      $SSH "$SUDO tail -40 /root/a37ct/setup.log"; die "host setup failed: $($SSH "$SUDO cat /root/a37ct/SETUP_FAILED")"; fi
  done
  $SSH "$SUDO test -f /root/a37ct/SETUP_OK" || die "setup did not finish in 60 min (see /root/a37ct/setup.log)"
fi
$SSH "$SUDO tail -6 /root/a37ct/setup.log"
$SSH "$SUDO cp $RS/run_a37.sh /root/a37ct/run_a37.sh && $SUDO setsid nohup bash /root/a37ct/run_a37.sh > /root/a37ct/chain.nohup 2>&1 < /dev/null & echo chain-started" \
  || die "chain launch (check the host before any relaunch)"
sleep 5
log "chain launched on $HOST:$PORT (overlay $OVL): $($SSH "$SUDO cat /root/a37ct/chain.pid 2>/dev/null")"
echo "offloader: nohup bash $WT/ops/a37/offload_a37.sh $HOST $PORT $USR >> $A/offload.log 2>&1 &"
