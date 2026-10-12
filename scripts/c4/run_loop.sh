#!/usr/bin/env bash
# C4 Dreamer loop: collect SUMO episodes with the current policy -> (convert) -> fine-tune the world model on
# them -> fine-tune the policy in imagination; ITERS iterations.  LF only.
#
#   bash run_loop.sh plan            # print the stage commands, start nothing
#   bash run_loop.sh run             # one detached loop driver (nohup/setsid, pid file, STATE)
#
# Required env (run): WM_CKPT_0 (stage-a WM checkpoint), POLICY_CKPT_0 (stage-b policy_best.pt)
# Optional env:
#   ITERS=3  START_ITER=1  STAGES=collect,convert,wm,policy   (a subset splits the loop across machines:
#                CPU box runs collect,convert; GPU box runs wm,policy; copy iter_K/collect/out, iter_K/convert/out and
#                the policy/WM checkpoints between them with scp/rsync, then rerun with the other STAGES)
#   LOOP_ID (default loop_<ts>)   RESUME_LOOP=1 re-enter an existing loop dir (done stages are skipped)
#   PENETRATIONS=0.05,0.1,0.2,0.4  DEMANDS=2400,2600,2700  WARMUP=300  HORIZON=600  COLLECT_WORKERS (default cores - 2)
#   COLLECT_SEEDS=4107-4148   seed spec for traffic_shepherd.c4.collect (it accepts ONLY the exposed development seeds
#                4107-4148 and refuses sealed/test/holdout seeds itself).  COLLECT_SEEDS_<K> overrides iteration K.
#                Evaluation seeds (920xxx, eval_seeds.json) are disjoint from these by construction; checked when the file exists.
#   COLLECT_ARGS         extra words appended to the collect command (e.g. "--no-physics-log")
#   WM_FT_CONFIG=configs/c4_wm.toml  POLICY_FT_CONFIG=configs/c4_policy.toml  POLICY_MODULE=wms.control.train_cli
#   WM_FT_SET            extra "--set" items for the WM refresh (space separated k=v), e.g. "curriculum.stage_steps=[200,300,200]"
#   POLICY_FT_ARGS       extra CLI words for the policy refresh, e.g. "--max-updates 500"
#   POLICY_DATA_CONFIG=configs/c4_wm.toml   WM config supplying the policy's TRAIN start states (see run_policy.sh; "none" = unchanged)
#   RESTART_STAGE=1   rerun a FAILED stage once in a fresh directory (the failed one is renamed, never deleted)
#
# WM-REFRESH DATA HOOK (built in parallel on branch c4/loop: recorded C4 episodes -> extra WM training source).
# The hook's exact interface is not final, so it is parameterised.  Placeholders usable in the variables below:
#   @ITER@ iteration number   @EPISODES@ iter_K/collect/out   @CONVERT@ iter_K/convert/out   @WM_PREV@ WM checkpoint entering the iteration
#   LOOP_CONVERT_CMD   shell command for the "convert" stage (empty: the stage is skipped and episodes are consumed as-is), e.g.
#                      LOOP_CONVERT_CMD='python -m traffic_shepherd.c4.<converter> --episodes @EPISODES@ --out @CONVERT@'
#   LOOP_DATA_ARGS     words appended to the `wms train` command of the WM refresh (config --set overrides or flags that
#                      register the new source), e.g. LOOP_DATA_ARGS='--set data.<extra>.path=@CONVERT@'  (no spaces inside a word)
#   LOOP_DATA_ENV      space-separated NAME=VALUE pairs exported before the WM refresh (registry datasets via WMS_DATA_<NAME>)
# With LOOP_DATA_ARGS empty the wm stage REFUSES (it would only re-fine-tune on the original data); LOOP_ALLOW_NO_DATA=1 overrides.
# Collect/policy interfaces below are the real ones (traffic_shepherd/c4/collect.py, wms/control/train_cli.py).
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
WARMUP=${WARMUP:-300}; HORIZON=${HORIZON:-600}
COLLECT_WORKERS=${COLLECT_WORKERS:-$(( C4_CORES > 3 ? C4_CORES - 2 : 1 ))}
COLLECT_MODULE=${COLLECT_MODULE:-traffic_shepherd.c4.collect}
COLLECT_SEEDS=${COLLECT_SEEDS:-4107-4148}
COLLECT_ARGS=${COLLECT_ARGS:-}
POLICY_MODULE=${POLICY_MODULE:-wms.control.train_cli}
WM_FT_CONFIG=${WM_FT_CONFIG:-configs/c4_wm.toml}
POLICY_FT_CONFIG=${POLICY_FT_CONFIG:-configs/c4_policy.toml}
POLICY_DATA_CONFIG=${POLICY_DATA_CONFIG:-configs/c4_wm.toml}
WM_CKPT_NAMES=${WM_CKPT_NAMES:-wm_s_best.pt wm_s.pt}
POLICY_CKPT_NAMES=${POLICY_CKPT_NAMES:-policy_best.pt policy_latest.pt}
WM_FT_SET=${WM_FT_SET:-}; POLICY_FT_ARGS=${POLICY_FT_ARGS:-}
LOOP_CONVERT_CMD=${LOOP_CONVERT_CMD:-}; LOOP_DATA_ARGS=${LOOP_DATA_ARGS:-}; LOOP_DATA_ENV=${LOOP_DATA_ENV:-}
LOOP_ALLOW_NO_DATA=${LOOP_ALLOW_NO_DATA:-0}
EVAL_SEEDS_FILE=${EVAL_SEEDS_FILE:-$C4_REPO/runs/active/WM-S-C4-20261013/eval_seeds.json}
WM_CKPT_0=${WM_CKPT_0:-}; POLICY_CKPT_0=${POLICY_CKPT_0:-}
RESTART_STAGE=${RESTART_STAGE:-0}

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

first_ckpt() {   # usage: first_ckpt DIR name1 name2 ...  (prints the first existing file)
  local d=$1 n; shift
  for n in "$@"; do if [[ -f "$d/$n" ]]; then echo "$d/$n"; return 0; fi; done
  return 1
}

subst() {   # usage: subst TEXT ITER EPISODES CONVERT WM_PREV   -> placeholders replaced
  local t=$1
  t=${t//@ITER@/$2}; t=${t//@EPISODES@/$3}; t=${t//@CONVERT@/$4}; t=${t//@WM_PREV@/$5}
  printf '%s' "$t"
}

seeds_for() {   # usage: seeds_for K -> COLLECT_SEEDS_K if set, else COLLECT_SEEDS
  local v="COLLECT_SEEDS_$1"
  printf '%s' "${!v:-$COLLECT_SEEDS}"
}

check_seeds_disjoint() {   # usage: check_seeds_disjoint SPEC  (refuses overlap with the evaluation seed file, if present)
  [[ -f "$EVAL_SEEDS_FILE" ]] || return 0
  "$PY" - "$1" "$EVAL_SEEDS_FILE" <<'PYEOF'
import json, sys
spec, path = sys.argv[1], sys.argv[2]
seeds = set()
for part in spec.split(","):
    a, _, b = part.partition("-")
    seeds.update(range(int(a), int(b or a) + 1))
found = set()
def walk(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "seed" and isinstance(v, int):
                found.add(v)
            else:
                walk(v)
    elif isinstance(node, list):
        for v in node:
            walk(v)
walk(json.load(open(path, encoding="utf-8")))
bad = sorted(seeds & found)
if bad:
    sys.exit("collection seeds overlap evaluation seeds: %s" % bad[:10])
PYEOF
}

if [[ "$MODE" == run ]]; then
  if has_stage collect; then
    [[ -f "$POLICY_CKPT_0" || $START_ITER -gt 1 ]] || c4_die "POLICY_CKPT_0 missing (stage b output)"
    for (( k = START_ITER; k < START_ITER + ITERS; k++ )); do
      check_seeds_disjoint "$(seeds_for "$k")" || c4_die "iteration $k collection seeds rejected"
    done
  fi
  if has_stage wm; then
    [[ -f "$WM_CKPT_0" || $START_ITER -gt 1 ]] || c4_die "WM_CKPT_0 missing"
    if [[ -z "$LOOP_DATA_ARGS" && "$LOOP_ALLOW_NO_DATA" != 1 ]]; then
      c4_die "wm stage needs the c4/loop data hook: set LOOP_DATA_ARGS (and LOOP_CONVERT_CMD), see the header of run_loop.sh, or LOOP_ALLOW_NO_DATA=1 to only re-fine-tune on the original data"
    fi
    c4_require_zero_shot_guard
  fi
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

policy_box_config() {   # the policy trainer's start-state data comes from POLICY_DATA_CONFIG (see run_policy.sh)
  if [[ "$POLICY_DATA_CONFIG" == none ]]; then echo "$POLICY_FT_CONFIG"; return; fi
  local cfg=$LOOP_DIR/policy_box.toml
  printf 'extends = "%s"\n\n[data]\nwm_config = "%s"\n' "$C4_REPO/$POLICY_FT_CONFIG" "$C4_REPO/$POLICY_DATA_CONFIG" > "$cfg"
  echo "$cfg"
}

loop_body() {
  local k wm_ck pol_ck prev out_collect out_convert out_wm out_pol data_words kv name value
  local -a extra data_extra pol_extra
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
      # shellcheck disable=SC2086
      stage_run "$k" collect "$PY" -m "$COLLECT_MODULE" --policy "$pol_ck" \
        --penetrations "$PENETRATIONS" --demands "$DEMANDS" --seeds "$(seeds_for "$k")" \
        --warmup "$WARMUP" --horizon "$HORIZON" --workers "$COLLECT_WORKERS" --out "$out_collect" $COLLECT_ARGS || return 1
    fi
    if has_stage convert; then
      if [[ -n "$LOOP_CONVERT_CMD" ]]; then
        stage_run "$k" convert bash -c "$(subst "$LOOP_CONVERT_CMD" "$k" "$out_collect" "$out_convert" "$wm_ck")" || return 1
      else
        c4_log "iter $k: no LOOP_CONVERT_CMD, convert stage skipped (episodes are consumed as recorded)"
      fi
    fi
    if has_stage wm; then
      c4_require_gpu_idle
      for kv in $LOOP_DATA_ENV; do
        name=$(subst "${kv%%=*}" "$k" "$out_collect" "$out_convert" "$wm_ck")
        value=$(subst "${kv#*=}" "$k" "$out_collect" "$out_convert" "$wm_ck")
        export "$name=$value"
      done
      export WMS_DATA_CKPT_PREV=$wm_ck
      mapfile -t extra < <(set_args "$WM_FT_SET")
      data_words=$(subst "$LOOP_DATA_ARGS" "$k" "$out_collect" "$out_convert" "$wm_ck")
      read -r -a data_extra <<< "$data_words"
      stage_run "$k" wm "$PY" -m wms train --config "$WM_FT_CONFIG" --mode run --out "$out_wm" \
        --set 'model.init_from="ckpt_prev"' --set "runtime.omp=$OMP" --set "runtime.threads=$OMP" \
        "${extra[@]}" "${data_extra[@]}" || return 1
      wm_ck=$(first_ckpt "$out_wm" $WM_CKPT_NAMES) || { c4_log "WM fine-tune wrote no checkpoint"; return 1; }
    fi
    if has_stage policy; then
      c4_require_gpu_idle
      read -r -a pol_extra <<< "$POLICY_FT_ARGS"
      stage_run "$k" policy "$PY" -m "$POLICY_MODULE" --config "$(policy_box_config)" --wm-ckpt "$wm_ck" --device cuda \
        --output-dir "$out_pol" --init-policy "$pol_ck" "${pol_extra[@]}" || return 1
      pol_ck=$(first_ckpt "$out_pol" $POLICY_CKPT_NAMES) || { c4_log "policy wrote no checkpoint"; return 1; }
    fi
    c4_log "iteration $k complete: wm=$wm_ck policy=$pol_ck"
  done
}

case "$MODE" in
  plan)
    echo "loop dir : $LOOP_DIR"
    echo "iterations $START_ITER..$((START_ITER + ITERS - 1)); stages: $STAGES; collect workers $COLLECT_WORKERS (usable cores $C4_CORES)"
    echo "collect  : $PY -m $COLLECT_MODULE --policy <policy_best.pt> --penetrations $PENETRATIONS --demands $DEMANDS --seeds $COLLECT_SEEDS --warmup $WARMUP --horizon $HORIZON --workers $COLLECT_WORKERS --out iter_K/collect/out $COLLECT_ARGS"
    echo "convert  : ${LOOP_CONVERT_CMD:-<skipped: LOOP_CONVERT_CMD unset>}"
    echo "wm       : $PY -m wms train --config $WM_FT_CONFIG --mode run --out iter_K/wm/out --set model.init_from=ckpt_prev $WM_FT_SET ${LOOP_DATA_ARGS:-<LOOP_DATA_ARGS unset: stage refuses>}"
    echo "policy   : $PY -m $POLICY_MODULE --config <box config: $POLICY_FT_CONFIG + data.wm_config=$POLICY_DATA_CONFIG> --wm-ckpt <wm> --device cuda --output-dir iter_K/policy/out --init-policy <policy> $POLICY_FT_ARGS"
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
    export LOOP_ID ITERS START_ITER STAGES PENETRATIONS DEMANDS WARMUP HORIZON COLLECT_WORKERS COLLECT_MODULE COLLECT_SEEDS COLLECT_ARGS \
           POLICY_MODULE WM_FT_CONFIG POLICY_FT_CONFIG POLICY_DATA_CONFIG WM_FT_SET POLICY_FT_ARGS LOOP_CONVERT_CMD LOOP_DATA_ARGS \
           LOOP_DATA_ENV LOOP_ALLOW_NO_DATA WM_CKPT_0 POLICY_CKPT_0 RESTART_STAGE
    for (( k = START_ITER; k < START_ITER + ITERS; k++ )); do
      v="COLLECT_SEEDS_$k"
      [[ -z "${!v:-}" ]] || export "$v"
    done
    c4_launch loop bash "$HERE/run_loop.sh" __body
    ;;
  __body)
    loop_body
    ;;
esac
