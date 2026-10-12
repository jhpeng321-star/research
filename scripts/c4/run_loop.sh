#!/usr/bin/env bash
# C4 Dreamer loop: collect SUMO episodes with the current policy -> convert into WM data ->
# fine-tune the world model -> fine-tune the policy in imagination; ITERS iterations.  LF only.
#
#   bash run_loop.sh plan            # print the stage commands, start nothing
#   bash run_loop.sh run             # one detached loop driver (nohup/setsid, pid file, STATE)
#
# Required env (run): WM_CKPT_0, POLICY_CKPT_0, COLLECT_SEEDS_FILE
#   COLLECT_SEEDS_FILE  JSON of the REGISTERED collection seeds (audited, disjoint from eval_seeds.json);
#                       passed to collect together with --iteration K.  Refused when missing.
# Optional env:
#   ITERS=3  START_ITER=1  STAGES=collect,convert,wm,policy   (a subset splits the loop across machines:
#                CPU box runs collect,convert; GPU box runs wm,policy; copy iter_K/convert and the
#                policy/WM checkpoints between them with scp/rsync, then rerun with the other STAGES)
#   LOOP_ID (default loop_<ts>)   RESUME_LOOP=1 re-enter an existing loop dir (done stages are skipped)
#   PENETRATIONS=0.05,0.1,0.2,0.4   DEMANDS=2400,2600,2700
#   COLLECT_WORKERS (default usable cores - 2)   COLLECT_MODULE=traffic_shepherd.c4.collect
#   CONVERT_MODULE=traffic_shepherd.c4.to_pack
#   WM_FT_CONFIG=configs/c4_wm.toml   POLICY_FT_CONFIG=configs/c4_policy.toml   POLICY_MODULE=wms.control.train_cli
#   WM_FT_SET / POLICY_FT_SET   extra "--set" items (space separated k=v), e.g. fewer steps for fine-tuning
#   RESTART_STAGE=1   rerun a FAILED stage once in a fresh directory (the failed one is renamed, never deleted)
# Interfaces assumed (reconcile with the collect/convert workers; see OPS_NOTES.md):
#   collect: --policy-ckpt --penetrations --demands --seeds-file --iteration --workers --out ; writes <out>/episodes/*
#   convert: --episodes <collect out> --out <dir> --iteration K ; writes <dir>/pack (wm_n1_pack) + <dir>/identities
#   The WM config refers to the logical datasets c4_iter_pack / c4_iter_identities (env WMS_DATA_C4_ITER_PACK,
#   WMS_DATA_C4_ITER_IDENTITIES = newest iteration; WMS_DATA_C4_ITER_PACK_<k> for every k).
if LC_ALL=C grep -q $'\r' "${BASH_SOURCE[0]}"; then echo 'REFUSED: CRLF bytes in script' >&2; exit 2; fi
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$HERE/c4_env.sh"
c4_refuse_crlf "$HERE/c4_env.sh" "$HERE/c4_jobwrap.sh"

MODE=${1:-}; [[ -n "$MODE" ]] || c4_die "usage: run_loop.sh plan|run"; shift
case "$MODE" in plan|run|__body) ;; *) c4_die "mode must be plan or run" ;; esac
c4_require_repo; c4_export_env; cd "$C4_REPO"

ITERS=${ITERS:-3}; START_ITER=${START_ITER:-1}
STAGES=${STAGES:-collect,convert,wm,policy}
PENETRATIONS=${PENETRATIONS:-0.05,0.1,0.2,0.4}
DEMANDS=${DEMANDS:-2400,2600,2700}
COLLECT_WORKERS=${COLLECT_WORKERS:-$(( C4_CORES > 3 ? C4_CORES - 2 : 1 ))}
COLLECT_MODULE=${COLLECT_MODULE:-traffic_shepherd.c4.collect}
CONVERT_MODULE=${CONVERT_MODULE:-traffic_shepherd.c4.to_pack}
POLICY_MODULE=${POLICY_MODULE:-wms.control.train_cli}
WM_FT_CONFIG=${WM_FT_CONFIG:-configs/c4_wm.toml}
POLICY_FT_CONFIG=${POLICY_FT_CONFIG:-configs/c4_policy.toml}
WM_CKPT_NAMES=${WM_CKPT_NAMES:-wm_s_best.pt wm_s.pt}
POLICY_CKPT_NAMES=${POLICY_CKPT_NAMES:-policy_best.pt policy.pt}
WM_FT_SET=${WM_FT_SET:-}; POLICY_FT_SET=${POLICY_FT_SET:-}
COLLECT_SEEDS_FILE=${COLLECT_SEEDS_FILE:-}
WM_CKPT_0=${WM_CKPT_0:-}; POLICY_CKPT_0=${POLICY_CKPT_0:-}
RESTART_STAGE=${RESTART_STAGE:-0}

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

first_ckpt() {   # usage: first_ckpt DIR name1 name2 ...  (prints the first existing file)
  local d=$1 n; shift
  for n in "$@"; do if [[ -f "$d/$n" ]]; then echo "$d/$n"; return 0; fi; done
  return 1
}

if [[ "$MODE" == run ]]; then
  if has_stage collect; then
    [[ -n "$COLLECT_SEEDS_FILE" && -f "$COLLECT_SEEDS_FILE" ]] \
      || c4_die "COLLECT_SEEDS_FILE (registered, exposure-audited collection seeds) is required for the collect stage"
    [[ -f "$POLICY_CKPT_0" || $START_ITER -gt 1 ]] || c4_die "POLICY_CKPT_0 missing (stage b output)"
  fi
  if has_stage wm; then [[ -f "$WM_CKPT_0" || $START_ITER -gt 1 ]] || c4_die "WM_CKPT_0 missing"; fi
fi

LOOP_ID=${LOOP_ID:-loop_$(date -u +%Y%m%dT%H%M%SZ)}
LOOP_DIR=$C4_RUNS/$LOOP_ID

# ---------------------------------------------------------------- stage runner (no retry loops)
stage_run() {   # usage: stage_run ITER NAME cmd...   -> runs synchronously in $LOOP_DIR/iter_K/NAME
  local k=$1 name=$2; shift 2
  local d=$LOOP_DIR/iter_$k/$name state
  if [[ -e "$d" ]]; then
    state=$(cat "$d/STATE" 2>/dev/null || echo none)
    if [[ "$state" == done ]]; then c4_log "iter $k $name: already done, skipping"; return 0; fi
    if [[ "$RESTART_STAGE" != 1 ]]; then
      c4_log "iter $k $name exists with STATE=$state (preserved). Rerun with RESTART_STAGE=1 to redo it in a fresh directory."
      return 1
    fi
    mv "$d" "$d.failed_$(date -u +%H%M%S)"
  fi
  mkdir -p "$d"
  c4_log "iter $k: $name -> $d"
  printf '%s\n' "$*" > "$d/COMMAND.txt"
  if ! bash "$C4_SCRIPTS/c4_jobwrap.sh" "$d" "loop_${LOOP_ID}_i${k}_$name" -- "$@" >> "$d/launch.log" 2>&1; then
    c4_log "iter $k: $name FAILED (state $(cat "$d/STATE")); stopping the loop. Log: $d/launch.log"
    return 1
  fi
}

set_args() { local item; for item in $1; do printf '%s\n' --set "$item"; done; }

loop_body() {
  local k wm_ck pol_ck prev out_collect out_convert out_wm out_pol
  local -a extra
  wm_ck=$WM_CKPT_0; pol_ck=$POLICY_CKPT_0
  for (( k = START_ITER; k < START_ITER + ITERS; k++ )); do
    out_collect=$LOOP_DIR/iter_$k/collect/out
    out_convert=$LOOP_DIR/iter_$k/convert/out
    out_wm=$LOOP_DIR/iter_$k/wm/out
    out_pol=$LOOP_DIR/iter_$k/policy/out
    prev=$LOOP_DIR/iter_$((k - 1))     # an earlier iteration (this run or a previous one) supplies the inputs
    if [[ -d "$prev" ]]; then
      wm_ck=$(first_ckpt "$prev/wm/out" $WM_CKPT_NAMES || echo "$wm_ck")
      pol_ck=$(first_ckpt "$prev/policy/out" $POLICY_CKPT_NAMES || echo "$pol_ck")
    fi
    if has_stage collect; then
      [[ -f "$pol_ck" ]] || { c4_log "no policy checkpoint for iteration $k"; return 1; }
      stage_run "$k" collect "$PY" -m "$COLLECT_MODULE" --policy-ckpt "$pol_ck" \
        --penetrations "$PENETRATIONS" --demands "$DEMANDS" --seeds-file "$COLLECT_SEEDS_FILE" \
        --iteration "$k" --workers "$COLLECT_WORKERS" --out "$out_collect" || return 1
    fi
    if has_stage convert; then
      stage_run "$k" convert "$PY" -m "$CONVERT_MODULE" --episodes "$out_collect" --out "$out_convert" \
        --iteration "$k" || return 1
    fi
    if has_stage wm; then
      c4_require_gpu_idle
      export WMS_DATA_C4_ITER_PACK=$out_convert/pack WMS_DATA_C4_ITER_IDENTITIES=$out_convert/identities
      export "WMS_DATA_C4_ITER_PACK_$k=$out_convert/pack"
      export WMS_DATA_CKPT_PREV=$wm_ck
      mapfile -t extra < <(set_args "$WM_FT_SET")
      stage_run "$k" wm "$PY" -m wms train --config "$WM_FT_CONFIG" --mode run --out "$out_wm" \
        --set 'model.init_from="ckpt_prev"' --set "runtime.omp=$OMP" --set "runtime.threads=$OMP" \
        "${extra[@]}" || return 1
      wm_ck=$(first_ckpt "$out_wm" $WM_CKPT_NAMES) || { c4_log "WM fine-tune wrote no checkpoint"; return 1; }
    fi
    if has_stage policy; then
      c4_require_gpu_idle
      mapfile -t extra < <(set_args "$POLICY_FT_SET")
      stage_run "$k" policy "$PY" -m "$POLICY_MODULE" --config "$POLICY_FT_CONFIG" --out "$out_pol" \
        --set "${POLICY_WM_KEY:-wm.checkpoint}=\"$wm_ck\"" --set "${POLICY_INIT_KEY:-policy.init_from}=\"$pol_ck\"" \
        "${extra[@]}" || return 1
      pol_ck=$(first_ckpt "$out_pol" $POLICY_CKPT_NAMES) || { c4_log "policy wrote no checkpoint"; return 1; }
    fi
    c4_log "iteration $k complete: wm=$wm_ck policy=$pol_ck"
  done
}

case "$MODE" in
  plan)
    echo "loop dir : $LOOP_DIR"
    echo "iterations $START_ITER..$((START_ITER + ITERS - 1)); stages: $STAGES; collect workers $COLLECT_WORKERS (usable cores $C4_CORES)"
    echo "collect  : $PY -m $COLLECT_MODULE --policy-ckpt <policy> --penetrations $PENETRATIONS --demands $DEMANDS --seeds-file <COLLECT_SEEDS_FILE> --iteration K --workers $COLLECT_WORKERS --out iter_K/collect/out"
    echo "convert  : $PY -m $CONVERT_MODULE --episodes iter_K/collect/out --out iter_K/convert/out --iteration K"
    echo "wm       : $PY -m wms train --config $WM_FT_CONFIG --mode run --out iter_K/wm/out --set model.init_from=\"ckpt_prev\" $WM_FT_SET"
    echo "policy   : $PY -m $POLICY_MODULE --config $POLICY_FT_CONFIG --out iter_K/policy/out --set wm.checkpoint=<wm> $POLICY_FT_SET"
    ;;
  run)
    if [[ -e "$LOOP_DIR" && "${RESUME_LOOP:-0}" == 1 ]]; then
      pid=$(cat "$LOOP_DIR/pid" 2>/dev/null || echo 0)
      if [[ "$pid" -gt 0 ]] && kill -0 "$pid" 2>/dev/null; then c4_die "loop $LOOP_ID is still running (pid $pid)"; fi
      RUN_ID=$LOOP_ID; RUN_DIR=$LOOP_DIR; OUT_DIR=$LOOP_DIR/out; export RUN_ID RUN_DIR OUT_DIR
    else
      RUN_ID=$LOOP_ID
      c4_prepare_run loop
      LOOP_ID=$RUN_ID; LOOP_DIR=$RUN_DIR
    fi
    # the driver is this same script re-entered as __body inside the job wrapper
    export LOOP_ID ITERS START_ITER STAGES PENETRATIONS DEMANDS COLLECT_WORKERS COLLECT_MODULE CONVERT_MODULE \
           POLICY_MODULE WM_FT_CONFIG POLICY_FT_CONFIG WM_FT_SET POLICY_FT_SET WM_CKPT_0 POLICY_CKPT_0 \
           COLLECT_SEEDS_FILE RESTART_STAGE
    c4_launch loop bash "$HERE/run_loop.sh" __body
    ;;
  __body)
    loop_body
    ;;
esac
