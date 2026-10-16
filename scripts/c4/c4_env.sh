#!/usr/bin/env bash
# Shared helpers for the C4 box scripts.  SOURCE this file (do not execute).  LF only.
#
# Layout on a box (override with C4_HOME):
#   $C4_HOME/bundle/   the transferred bundle (tars + SHA256SUMS + BUNDLE_MANIFEST.json)
#   $C4_HOME/rd/       repo root (unpacked code + data_cache); configs/data_registry.toml resolves here
#   $C4_HOME/runs/     one fresh directory per run: STATE, pid, launch.log, monitor.csv, RUN_SUMMARY_C4.json, out/
#   $C4_HOME/env.sh    written by box_setup.sh (interpreter, SUMO, cores, WMS_DATA_*)
#
# Rules these helpers enforce: new output directory per run (never overwritten), nohup + setsid +
# pid file, STATE file (running|done|failed:RC), RUN_SUMMARY_C4.json, no retry loops.

C4_HOME=${C4_HOME:-/workspace/c4}
C4_REPO=${C4_REPO:-$C4_HOME/rd}
C4_RUNS=${C4_RUNS:-$C4_HOME/runs}
C4_SCRIPTS=${C4_SCRIPTS:-$C4_REPO/scripts/c4}

c4_log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
c4_die() { printf 'REFUSED: %s\n' "$*" >&2; exit 2; }

# usage: c4_refuse_crlf file...   (box pitfall: CRLF .sh from Windows tars breaks bash)
c4_refuse_crlf() {
  local f
  for f in "$@"; do
    if LC_ALL=C grep -q $'\r' "$f"; then c4_die "$f has CRLF bytes; require LF (re-pack with pack_bundle.py)"; fi
  done
}

# Persisted environment from box_setup.sh (PY, SUMO paths, cores); already-set variables win.
if [[ -f "$C4_HOME/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$C4_HOME/env.sh"
fi

# ---- interpreter -----------------------------------------------------------------------
c4_find_python() {
  local cand
  for cand in "${PY:-}" /workspace/gpu1/env/rd_gpu/bin/python /root/miniconda3/bin/python python3 python; do
    [[ -n "$cand" ]] || continue
    if command -v "$cand" >/dev/null 2>&1; then command -v "$cand"; return 0; fi
  done
  return 1
}
PY=${PY:-$(c4_find_python || true)}

# ---- usable CPU cores (cgroup quota aware: containers often show host nproc) -------------
c4_usable_cores() {
  local n quota period
  n=$(nproc 2>/dev/null || echo 1)
  if [[ -r /sys/fs/cgroup/cpu.max ]]; then                      # cgroup v2
    read -r quota period < /sys/fs/cgroup/cpu.max || true
    if [[ "${quota:-max}" != max && "${period:-0}" -gt 0 ]]; then
      quota=$(( (quota + period - 1) / period ))
      (( quota < n )) && n=$quota
    fi
  elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then       # cgroup v1
    quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    if [[ "$quota" -gt 0 && "$period" -gt 0 ]]; then
      quota=$(( (quota + period - 1) / period ))
      (( quota < n )) && n=$quota
    fi
  fi
  echo "$n"
}
C4_CORES=${C4_CORES:-$(c4_usable_cores)}
OMP=${OMP:-4}

# ---- environment for wms / traffic_shepherd -------------------------------------------------
c4_export_env() {
  local sep=:   # os.pathsep of the interpreter (Linux box); ';' only when testing from Git Bash on Windows
  case "$(uname -s 2>/dev/null)" in MINGW*|MSYS*|CYGWIN*) sep=';' ;; esac
  export C4_HOME C4_REPO C4_RUNS
  export PYTHONPATH="$C4_REPO/src${sep}$C4_REPO${PYTHONPATH:+$sep$PYTHONPATH}"
  export OMP_NUM_THREADS="$OMP" MKL_NUM_THREADS="$OMP"
  export PYTHONUNBUFFERED=1
  # logical dataset names that are not in configs/data_registry.toml (already-set variables win):
  # configs/c4_wm.toml model.init_from = "ckpt_r5b_best" resolves through WMS_DATA_CKPT_R5B_BEST.
  : "${WMS_DATA_CKPT_R5B_BEST:=$C4_REPO/data_cache/ckpt_r5b/wm_s_best.pt}"
  export WMS_DATA_CKPT_R5B_BEST
  # wms.control.train_cli refuses any config/checkpoint/data/output path outside the repo root unless it is
  # listed here (os.pathsep-separated).  The run directories live under $C4_RUNS, outside the repo.
  : "${C4_ALLOWED_ROOTS:=$C4_RUNS${sep}$C4_HOME}"
  export C4_ALLOWED_ROOTS
  if [[ -n "${SUMO_HOME:-}" ]]; then export SUMO_HOME PATH="$SUMO_HOME/bin:$PATH"; fi
}

c4_require_repo() {
  [[ -d "$C4_REPO/src/wms" ]] || c4_die "repo not unpacked at $C4_REPO (run box_setup.sh)"
  [[ -n "${PY:-}" ]] || c4_die "no python interpreter found (set PY=...)"
  [[ -f "$C4_HOME/UNPACK_OK" ]] || c4_die "$C4_HOME/UNPACK_OK missing: box_setup.sh has not completed successfully"
}

# The switching pack's zero-shot guard (registry dataset zero_shot_heldout) is REQUIRED by configs/c4_wm.toml
# (data.switching).  Its name contains a restricted token, so the bundle never ships it; place the
# one file on the box by hand (or sets WMS_DATA_ZERO_SHOT_HELDOUT).  Fail closed with a clear message.
c4_require_zero_shot_guard() {
  local p=${WMS_DATA_ZERO_SHOT_HELDOUT:-$C4_REPO/data_cache/wmn1_switching_train/ZERO_SHOT_HELDOUT.json}
  [[ -f "$p" ]] || c4_die "zero-shot guard file missing: $p. configs/c4_wm.toml (data.switching) needs it and the bundle does not ship it (restricted-token name). Place it by hand (scp of the single original file) or train without the switching direct-mix source. See OPS_NOTES.md section 11."
}

# ---- resource guards ----------------------------------------------------------------------------
c4_free_gb() { df -Pk "$1" | awk 'NR==2 {printf "%d", $4/1048576}'; }

c4_require_disk() {   # usage: c4_require_disk MIN_GB
  mkdir -p "$C4_RUNS"
  local free; free=$(c4_free_gb "$C4_RUNS")
  (( free >= $1 )) || c4_die "only ${free} GB free under $C4_RUNS (need $1)"
}

c4_gpu_used_mib() {   # prints GPU memory in use (MiB); 0 when there is no usable nvidia-smi
  local v=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)
  fi
  [[ "$v" =~ ^[0-9]+$ ]] || v=0
  echo "$v"
}

c4_require_gpu_idle() {   # refuse to start next to someone else's job (box has ONE GPU)
  local used; used=$(c4_gpu_used_mib)
  (( used < ${GPU_BUSY_MIB:-2000} )) || c4_die "GPU already holds ${used} MiB; stop that job first (GPU_BUSY_MIB overrides)"
}

# ---- run directories -----------------------------------------------------------------------------
# usage: c4_prepare_run NAME   -> sets RUN_ID RUN_DIR OUT_DIR (refuses to touch an existing attempt)
c4_prepare_run() {
  local name=$1 state pid n
  RUN_ID=${RUN_ID:-${name}_$(date -u +%Y%m%dT%H%M%SZ)}
  RUN_DIR=$C4_RUNS/$RUN_ID
  if [[ -e "$RUN_DIR" ]]; then
    state=$(cat "$RUN_DIR/STATE" 2>/dev/null || echo none)
    pid=$(cat "$RUN_DIR/pid" 2>/dev/null || echo 0)
    case "$state" in
      done*) c4_log "$RUN_ID already finished (STATE=$state); nothing to do"; exit 0 ;;
      running*)
        if [[ "$pid" -gt 0 ]] && kill -0 "$pid" 2>/dev/null; then c4_die "$RUN_ID is still running (pid $pid)"; fi
        state="running-but-pid-dead" ;;
    esac
    if [[ "${RESUME:-0}" == 1 ]]; then      # same directory, for tools that resume themselves (eval campaign)
      OUT_DIR=$RUN_DIR/out; export RUN_ID RUN_DIR OUT_DIR
      c4_log "resuming $RUN_ID (previous STATE=$state)"
      return 0
    fi
    if [[ "${RESTART:-0}" != 1 ]]; then
      c4_die "$RUN_DIR exists with STATE=$state. It is preserved, never overwritten. To continue, rerun with RESTART=1 (new directory ${RUN_ID}_rN; set RESTART_CKPT=<checkpoint> to warm-start) or, for self-resuming tools such as the eval campaign, RESUME=1."
    fi
    n=2; while [[ -e "$C4_RUNS/${RUN_ID}_r$n" ]]; do n=$((n + 1)); done
    PREV_RUN_DIR=$RUN_DIR
    RUN_ID=${RUN_ID}_r$n; RUN_DIR=$C4_RUNS/$RUN_ID
  fi
  OUT_DIR=$RUN_DIR/out
  mkdir -p "$RUN_DIR"
  export RUN_ID RUN_DIR OUT_DIR PREV_RUN_DIR
}

# usage: c4_launch LABEL cmd args...   (background unless FOREGROUND=1; never loops/retries)
c4_launch() {
  local label=$1; shift
  c4_log "run dir: $RUN_DIR"
  printf '%s\n' "$*" > "$RUN_DIR/COMMAND.txt"
  if [[ "${FOREGROUND:-0}" == 1 ]]; then
    bash "$C4_SCRIPTS/c4_jobwrap.sh" "$RUN_DIR" "$label" -- "$@"
    return $?
  fi
  local launcher=(nohup)
  command -v setsid >/dev/null 2>&1 && launcher=(nohup setsid)
  "${launcher[@]}" bash "$C4_SCRIPTS/c4_jobwrap.sh" "$RUN_DIR" "$label" -- "$@" \
      >> "$RUN_DIR/launch.log" 2>&1 < /dev/null &
  echo $! > "$RUN_DIR/launcher.pid"
  sleep 2
  c4_log "launched $label: launcher pid $(cat "$RUN_DIR/launcher.pid")"
  c4_log "follow:  tail -f $RUN_DIR/launch.log     state: cat $RUN_DIR/STATE     summary: $RUN_DIR/RUN_SUMMARY_C4.json"
}
