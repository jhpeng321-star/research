# The Dreamer data loop: SUMO collection, WM refresh, policy resume

The C4 policy ([C4_DESIGN.md](C4_DESIGN.md)) is trained in imagination, so it can exploit any
blind spot of the world model. The data loop closes that gap: the current policy is run in SUMO,
its episodes are added to the WM training data, the WM is fine-tuned, and the policy continues
training on the refreshed WM, dreaming partly from the states it actually visits in SUMO. The
orchestrator is `scripts/c4/dreamer_loop.py` (standard library only); formats are in
[DATA_FORMAT.md](DATA_FORMAT.md). Paths are relative to the repository root.

## 1. One cycle

| step | command (run by the orchestrator) | output |
|---|---|---|
| collect | `python -m traffic_shepherd.c4.collect --policy <policy snapshot .pt> --population textbook --seeds <range> --case-plan weighted-cycle --penetrations ... --pen-weights ... --demands ... --warmup ... --horizon ... --workers N --no-physics-log --out <dir>` | raw episodes (`c4_sumo_episode/1`) |
| pull | `rsync` from the collector host into the loop's raw pool | `<loop_root>/raw/b<NNNN>/` |
| convert | `python -m wms.data.c4_episodes --episodes <pool episodes> --out <data_root>/<prefix><N>` | converted source (`wms_c4_episode_source/1`) |
| WM refresh | `python -m wms train --config <wm.config> --mode run --set model.init_from=<newest accepted WM> --set curriculum.stage_steps=<wm.stage_steps> [wm.extra_set ...]` with `WMS_DATA_V6_POLICY=<converted pool>` | `<wm_runs_root>/c4_wm_loop_r<NNN>/wm_s_best.pt` |
| sanity check | finite training losses, `wm_s_best.pt` and `physics.json` present, final validation `wm.val_metric` at every common horizon `<= (1 + wm.val_tolerance) x` the reference WM's | accept, or reject and keep the previous WM |
| policy switch | stop the running trainer, then `WMS_LOOP_REFRESH=1 python -m wms.control.train_cli --config <policy.config> --resume <newest policy ckpt> --wm-ckpt <accepted WM>` with `WMS_DATA_V6_POLICY=<that pool>` | `<policy work_root>/runs/c4_policy_loop_g<NNN>/` |

Relevant presets: the WM refresh config (default `configs/c4_wm_v6_refresh.toml`) adds a
`v6_policy` entry to `[[data.c4_sources]]` that resolves through `WMS_DATA_V6_POLICY`; the loop
policy config (default `configs/c4_policy_v6_loop.toml`) draws imagination starts by weight from
`native_v5`, `genrand_plain` and `v6_policy` (`[data] start_source_weights`, about half from the
policy's own episodes). Before the first conversion exists, `policy.config_without_policy_data` is
used instead.

On a loop resume (`WMS_LOOP_REFRESH=1`) the trainer accepts a different WM identity, skips the
saved sampler state when the start-window set changed, and resets its best-validation scores, so
each run directory selects its own `policy_best*.pt` on its own WM. The trainer stops cleanly when a
`STOP` file appears in its output directory (it writes `policy_latest.pt` first).

## 2. Modes

- **Pipelined** (`"mode": "pipelined"`, default). Three lanes advance on each polling tick
  (`poll_s`):
  - *Collectors*: each host in `collect.hosts` runs batches of `collect.batch_size` episodes back to
    back with the newest settled policy checkpoint and its own disjoint seed range. With
    `require_new_policy_ckpt` (default) a batch starts only for a checkpoint no earlier batch used,
    which keeps the seed block from being exhausted by fast collectors.
  - *WM lane*: refreshes run back to back; each converts the whole pool (or the most recent
    `convert.max_episodes`, or fewer when disk is short) into a fresh directory.
  - *Policy lane*: every accepted WM is handed to the policy.
- **Sequential** (`--sequential` or `"mode": "sequential"`). Iteration `k` runs `snapshot ->
  collect_launch -> collect_wait -> pull -> convert -> wm_launch -> wm_wait -> wm_check ->
  policy_stop -> policy_start` with `seeds.per_iteration` seeds, up to `max_iterations`. The policy
  keeps training on the previous WM during all steps except the switch.

## 3. Configuration (`configs/examples/dreamer_loop.example.json`)

The config is JSON, merged over `DEFAULTS` in `scripts/c4/dreamer_loop.py` (nested objects merge,
other values replace). Any string containing a restricted split token is refused.

| key | meaning |
|---|---|
| `mode` | `pipelined` or `sequential` |
| `loop_root` | state, logs, policy snapshots and the raw episode pool (required) |
| `data_root` | converted pools `<prefix><N>` (required) |
| `wm_runs_root` | WM refresh run directories (required) |
| `disk_paths`, `min_free_gb` | nothing new starts while free space on these paths is below `min_free_gb` |
| `poll_s`, `max_iterations`, `ssh_retries`, `max_step_errors` | tick length, sequential iteration cap, ssh retry count, consecutive-error limit |
| `hosts.wm` | must be local (the orchestrator runs where the WM trains): `code_root`, `python` |
| `hosts.policy` | local, or remote with `ssh`; needs `code_root`, `python`, `work_root` |
| `hosts.<collector>` | one entry per name in `collect.hosts`: `ssh`, `code_root`, `python`, `work_root` |
| `hosts.*.ssh` | list of ssh arguments ending with the destination, e.g. `["-p", "<port>", "<user>@<host>"]`; omit for the local host |
| `hosts.*.code_root` / `python` / `work_root` | deployed checkout of this repository, interpreter path, scratch root (each loop uses a fresh subdirectory) |
| `seeds.block`, `seeds.exclude`, `seeds.per_iteration` | seed range (must lie inside the training block hard-coded in the script), seeds to skip, sequential batch size |
| `collect.*` | `hosts`, `batch_size`, `require_new_policy_ckpt`, `min_interval_s`, `workers` (parallel SUMO processes), `warmup`, `horizon`, `population`, `case_plan`, `penetrations`, `pen_weights`, `demands`, `timeout_s`, `max_attempts`, `min_complete_fraction`, `wait_for_absent` (process patterns that pause a collector), `nice` |
| `convert.*` | `prefix` (default `v6_policy_iter`), `max_episodes` (0 = whole pool), `mb_per_episode` (disk estimate), `min_window`, `timeout_s`, `nice` |
| `wm.*` | `config`, `stage_steps`, `extra_set` (more `--set` overrides), `env` (e.g. `WMS_DATA_*` locations of the other sources), `timeout_s`, `val_metric`, `val_tolerance`, `min_new_episodes`, `max_refreshes`, optional `lane_calib` |
| `policy.*` | `config`, `config_without_policy_data`, `device`, `max_updates`, `extra_args`, `env`, `allowed_roots` (exported as `C4_ALLOWED_ROOTS` so the trainer may read data outside the checkout), `stop_timeout_s`, `adopted_wait_max_s`, `start_check_s`, `max_restarts` |
| `bootstrap` (optional) | `wm_ckpt`, `wm_val_log`, `policy_run_dir`, `adopt_policy_pid`, `policy_wm`: the same as the `--init-*` flags |

A minimal skeleton (placeholders in angle brackets):

```json
{
  "mode": "pipelined",
  "loop_root": "<scratch>/loop", "data_root": "<scratch>/data", "wm_runs_root": "<scratch>/runs",
  "hosts": {
    "wm":     {"code_root": "<checkout>", "python": "<venv>/bin/python"},
    "policy": {"code_root": "<checkout>", "python": "<venv>/bin/python", "work_root": "<scratch>"},
    "cpu":    {"ssh": ["<user>@<cpu-host>"], "code_root": "<remote checkout>",
               "python": "<remote venv>/bin/python", "work_root": "<remote scratch>"}
  },
  "collect": {"hosts": ["cpu"], "batch_size": 50, "workers": 8},
  "wm": {"env": {"WMS_DATA_V5_NATIVE": "<path>", "WMS_DATA_GENRAND_PLAIN": "<path>",
                 "WMS_DATA_V6_LONGPAIR": "<path>"}},
  "policy": {"env": {"WMS_DATA_V5_NATIVE": "<path>", "WMS_DATA_GENRAND_PLAIN": "<path>",
                     "WMS_DATA_V6_LONGPAIR": "<path>"}, "allowed_roots": ["<data dir>"]}
}
```

## 4. Running it

```
python scripts/c4/dreamer_loop.py dry-run --config <cfg.json> [init flags]   # print the plan
python scripts/c4/dreamer_loop.py run --config <cfg.json> \
    --init-wm <wm_s_best.pt> --init-wm-val-log <val_log.jsonl> \
    [--init-policy-run-dir <dir> --adopt-policy-pid <pid> --init-policy-wm <wm used by it>]
python scripts/c4/dreamer_loop.py status --config <cfg.json> [--live]
python scripts/c4/dreamer_loop.py stop --config <cfg.json>                   # writes <loop_root>/STOP
```

`--init-*` flags are honoured only when `<loop_root>/state.json` does not yet exist; afterwards
`run` resumes from the saved state. `stop` ends the orchestrator between ticks; detached jobs
(collections, refreshes, the policy trainer) keep running and are re-attached by the next `run`.
To resume, delete the `STOP` file and run again without `--init-*` flags. Adopting an already
running policy trainer: the loop waits for its next periodic `policy_<n>.pt` and the following
update, then sends SIGTERM to exactly that PID (at most one update is lost).

## 5. Files

| path | content |
|---|---|
| `<loop_root>/state.json` | full loop state, rewritten atomically after every transition |
| `<loop_root>/loop_log.jsonl` | append-only log of every action |
| `<loop_root>/seed_ledger.jsonl` | one line per batch/iteration: seeds, host |
| `<loop_root>/batches/b<NNNN>/` | policy snapshot (with sha256 in the state) and collection log |
| `<loop_root>/raw/b<NNNN>/` | pulled raw episodes (never deleted by the loop) |
| `<loop_root>/refreshes/r<NNN>/` | conversion list and log per refresh |
| `<wm_runs_root>/c4_wm_loop_r<NNN>[.aK]` | WM refresh runs (+ `.stdout.log`, `.pid`, `.EXIT`) |
| `<policy work_root>/runs/c4_policy_loop_g<NNN>/` | policy runs (`LAUNCH.txt`, `train.log`, pid, exit marker) |

In sequential mode, superseded converted directories created by the loop are deleted; raw episodes
are always kept.

## 6. Failure handling and safety

- Every transition is persisted; allocated seeds are recorded before anything runs and are never
  re-allocated, even if the batch fails.
- Transient errors (ssh, rsync, busy host) are logged and retried on the next tick; the loop exits
  after `max_step_errors` consecutive errors in one lane. ssh transport failures are retried with
  backoff; rsync skips `*.incomplete-*` directories.
- A failed collection batch is relaunched into the same directory (collection is resumable) up to
  `collect.max_attempts`. A rejected or failed refresh keeps the previous WM and does not restart
  the policy. A dead policy trainer is restarted from its newest checkpoint, falling back to the
  previous WM if it died right after a switch (`policy.max_restarts`).
- Signals go only to processes the loop launched (process-group leader, start time and command
  line checked) or to the one explicitly adopted PID; there is no pattern-based killing. Output
  directories are never reused (`.aN` suffixes, per-loop remote directories).
- The SUMO collection environment exports `SUMO_HOME` from the interpreter's `sumo` package, since
  non-interactive ssh shells usually lack it.

## 7. Caveats

- The converter's world-grouped validation split is drawn over the whole pool, so a policy world
  may move between train and validation across refreshes, and the validation set used by the
  sanity check changes slightly between refreshes.
- Converted pools take on the order of tens of MB per 900 s episode; with limited disk the loop
  converts only the most recent episodes.
- WM refresh and policy training on one GPU slow each other down; `policy.env` can set
  `C4_GPU_MEM_FRACTION` so the policy, not the WM refresh, runs out of memory first, or
  `hosts.policy` can point to a separate machine (which then needs the start-window data sources).
