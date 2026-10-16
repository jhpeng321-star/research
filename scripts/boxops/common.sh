#!/usr/bin/env bash
# Shared boxops runtime. Source this file; no remote operations happen on import.
set -euo pipefail
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    echo 'Usage: source scripts/boxops/common.sh (shared functions, not a launcher)'
    exit 0
fi
BOXOPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${REPO:-$HOME/c4/repo}
VENV=${VENV:-$HOME/c4env}
RELAY_HOST=${RELAY_HOST:-user@relay.example.com}
RELAY_PORT=${RELAY_PORT:-22}
RELAY_DIR=${RELAY_DIR:-/workspace/c4/relay}
RELAY_DATA=${RELAY_DATA:-/workspace/c4/data_v5}
RELAY_KEY=${RELAY_KEY:-$HOME/.ssh/relay_key}
TRANSFER_RETRIES=${TRANSFER_RETRIES:-8}
RETRY_DELAY=${RETRY_DELAY:-15}
PYTHON=${PYTHON:-$VENV/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

die() { echo "boxops: $*" >&2; exit 1; }
valid_id() { [[ $1 =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ && $1 != *..* ]] || die 'invalid run/archive ID'; }
safe_path() {
    local lower=${1,,} token
    for token in sealed holdout heldout cal_split chunk5 j4b inc1; do
        [[ $lower != *"$token"* ]] || die 'restricted path refused'
    done
}
relay_config() {
    [[ $RELAY_HOST =~ ^[A-Za-z0-9_.@-]+$ && $RELAY_PORT =~ ^[0-9]+$ ]] || die 'invalid relay address'
    [[ $RELAY_DIR =~ ^/[A-Za-z0-9_./-]+$ && $RELAY_DATA =~ ^/[A-Za-z0-9_./-]+$ ]] || die 'invalid relay directory'
    [[ $RELAY_KEY != *[[:space:]]* ]] || die 'relay key path must not contain whitespace'
    safe_path "$RELAY_DIR"; safe_path "$RELAY_DATA"
    SSH=(ssh -p "$RELAY_PORT" -i "$RELAY_KEY" -o BatchMode=yes -o IdentitiesOnly=yes
         -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=15
         -o ServerAliveCountMax=3)
    RSYNC_SSH="ssh -p $RELAY_PORT -i $RELAY_KEY -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3"
}
retry_transfer() {
    local attempt rc=1
    [[ $TRANSFER_RETRIES =~ ^[1-9][0-9]*$ && $RETRY_DELAY =~ ^[0-9]+$ ]] || die 'invalid retry settings'
    for ((attempt=1; attempt<=TRANSFER_RETRIES; attempt++)); do
        if "$@"; then return 0; else rc=$?; fi
        echo "transfer attempt $attempt/$TRANSFER_RETRIES failed (rc=$rc)" >&2
        if ((attempt < TRANSFER_RETRIES)); then sleep "$RETRY_DELAY"; fi
    done
    return "$rc"
}
activate_box() {
    safe_path "$REPO"
    [[ -x $PYTHON && -f $REPO/traffic_shepherd/c4/collect.py ]] || die 'run fetch_code.sh and setup_box.sh first'
    export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
    SUMO_HOME="$("$PYTHON" -c 'import sumo; print(sumo.SUMO_HOME)')"
    export SUMO_HOME PATH="$VENV/bin:$SUMO_HOME/bin:$PATH"
    cd "$REPO"
}
lock_run() {
    mkdir -p "$1/logs"
    exec 9>"$1/.boxops.lock"
    flock -n 9 || die "another launcher owns $1"
}
begin_run() {
    OUT=$1; shift
    safe_path "$OUT"
    lock_run "$OUT"
    "$PYTHON" "$BOXOPS_DIR/ops.py" begin "$OUT" "$REPO" "$@"
    trap 'rc=$?; trap - EXIT; "$PYTHON" "$BOXOPS_DIR/ops.py" end "$OUT" "$rc" || true; exit "$rc"' EXIT
}
phase() {
    local name=$1; shift
    [[ ! -f $OUT/$name.DONE ]] || { echo "skip completed phase: $name"; return 0; }
    echo "phase $name: $(date -u +%FT%TZ)"
    { printf '\n# %s ' "$(date -u +%FT%TZ)"; printf '%q ' "$@"; printf '\n'; } >>"$OUT/logs/$name.log"
    "$@" >>"$OUT/logs/$name.log" 2>&1
    touch "$OUT/$name.DONE"
}
convert_phase() {
    local name=$1 source=$2 dest
    [[ ! -f $OUT/$name.CONVERTED ]] || return 0
    dest=$source/c4_episodes
    # The converter requires an empty destination. Keep interrupted output for diagnosis.
    if [[ -d $dest ]] && [[ -n $(ls -A "$dest") ]]; then
        dest=$source/c4_episodes.attempt-$(date -u +%Y%m%dT%H%M%S)-$$
    fi
    "$PYTHON" "$BOXOPS_DIR/ops.py" convert "$source" "$dest" >>"$OUT/logs/convert_$name.log" 2>&1
    printf '%s\n' "$dest" >"$OUT/$name.CONVERTED"
}
push_run() {
    local subdir=${1:-} remote
    [[ ! -f $OUT/PUSHED ]] || { echo 'already pushed'; return 0; }
    relay_config
    remote=$RELAY_DATA/$RUN_ID${subdir:+/$subdir}
    "$PYTHON" "$BOXOPS_DIR/ops.py" summary "$OUT"
    # End metadata must reach the relay too, before success is marked locally.
    "$PYTHON" "$BOXOPS_DIR/ops.py" end "$OUT" "${RUN_RC:-0}"
    retry_transfer "${SSH[@]}" "$RELAY_HOST" "mkdir -p '$remote'" >>"$OUT/logs/push.log" 2>&1
    retry_transfer rsync -az --partial --timeout=120 -e "$RSYNC_SSH" \
        --exclude='.boxops.lock' --exclude='PUSHED' --exclude='scenarios/' \
        "$OUT/" "$RELAY_HOST:$remote/" >>"$OUT/logs/push.log" 2>&1
    touch "$OUT/PUSHED"
    retry_transfer rsync -az --partial --timeout=120 -e "$RSYNC_SSH" \
        "$OUT/PUSHED" "$RELAY_HOST:$remote/PUSHED" >>"$OUT/logs/push.log" 2>&1 || {
        mv "$OUT/PUSHED" "$OUT/PUSH_PENDING"; return 1;
    }
}
