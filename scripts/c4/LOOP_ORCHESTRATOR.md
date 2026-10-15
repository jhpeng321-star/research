# C4 Dreamer-loop orchestrator (`scripts/c4/dreamer_loop.py`)

Owner intent (2026-10-16): the policy collects real SUMO episodes, the world model (WM) refreshes on all
policy data so far, the policy continues (`--resume`) on the refreshed WM, and it dreams FROM the states it
actually visits in SUMO. The orchestrator runs on **gpuW** (always on, holds the data), stdlib only, under
`nohup`. Branch `c4/loop2`.

## Modes

**Pipelined (default, `"mode": "pipelined"`)**: three lanes advanced by one polling tick (`poll_s`, 60 s):

* **Collectors**: every host in `collect.hosts` runs SUMO batches of `collect.batch_size` (50) episodes back
  to back. Each batch uses the newest settled policy checkpoint (`policy_*.pt`, mtime at least 30 s old)
  and its own disjoint seed range from the approved C4_V6_LONGPAIR training block **965184–965999**
  (hard-coded guard; evaluation seeds 964000–964199 are refused). A finished batch is pulled with rsync to
  `<loop_root>/raw/b<NNNN>` (the pool, kept for good). **Seed budget:** by default
  (`require_new_policy_ckpt: true`) a batch starts only when there is a checkpoint that no earlier batch
  used. A 50-episode batch takes only about 3–5 min on the 48-core box, so without this rule the block would
  be used up in about an hour. `min_interval_s` adds a minimum time between batches.
* **WM lane**: refreshes run back to back. Each one converts the pool (all batches so far) into a fresh
  `<data_root>/v6_policy_iterNNN` (in the background, via a list file), then runs
  `python -m wms train --config configs/c4_wm_v6_refresh.toml --set model.init_from=<newest accepted WM>
  --set curriculum.stage_steps=[60,40]` with `WMS_DATA_V6_POLICY` pointing at that pool. The **sanity
  check** requires finite training losses, `wm_s_best.pt` and `physics.json`, and a final validation
  `mean_position` of at most (1 + 0.30) times the reference WM's last validation, for each common horizon.
  A failed refresh is logged (`wm_rejected`) and the previous WM is kept.
* **Policy lane**: every accepted WM is handed to the policy:
  1. Stop the trainer:
     * A trainer started by the loop gets a STOP file. `dreamer.fit` then finishes its current update and
       writes `policy_latest.pt`.
     * The adopted first trainer has no STOP support. The loop waits for its next periodic `policy_<n>.pt`
       and for update n+1 in `updates.jsonl` (so update n's validation and best-checkpoint saving are
       done), then sends SIGTERM to exactly that PID. At most one update is lost.
  2. Restart it with `WMS_LOOP_REFRESH=1 python -m wms.control.train_cli --config
     configs/c4_policy_v6_loop.toml --resume <ckpt> --wm-ckpt <new WM>` in a new
     `<policy work_root>/runs/c4_policy_loop_gNNN`, with `WMS_DATA_V6_POLICY` = that refresh's pool.

**Sequential (`--sequential` or `"mode": "sequential"`)**, the fallback. Iteration k runs the steps in a
fixed order:

1. snapshot
2. collect_launch (`seeds.per_iteration` seeds)
3. collect_wait
4. pull
5. convert
6. wm_launch
7. wm_wait
8. wm_check
9. policy_stop
10. policy_start
11. done

The policy keeps training during steps 2–8.

## Imagination start windows from the policy's own episodes

`configs/c4_policy_v6_loop.toml` is `c4_policy_v6_full.toml` with these changes:

* `wm_config = configs/c4_wm_v6_refresh.toml`
* `start_sources = [native_v5, genrand_plain, v6_policy]`
* `start_source_weights = {native_v5 = 0.25, genrand_plain = 0.25, v6_policy = 0.5}`

At each penetration the start source is drawn by weight (renormalised over the sources present), then a
uniform window of that source. Without the key, the sampler is unchanged; a unit test checks the draws are
identical to the old ones.

On a loop resume (`WMS_LOOP_REFRESH=1`) the trainer handles the changed state as follows:

* It skips the saved sampler state when the window set changed (`[loop-refresh] start windows changed`).
* It resets `best_validation` / `best_coop_validation` when the WM changed. Each run directory keeps its own
  `policy_best*.pt`, selected on its own WM.

A restart before the first conversion uses `config_without_policy_data` (`c4_policy_v6_full.toml`).

## Files

| path | content |
|---|---|
| `<loop_root>/state.json` | full state (atomic rewrite after every transition); a restart resumes from it |
| `<loop_root>/loop_log.jsonl` | append-only log of every action (`t`, `event`, details) |
| `<loop_root>/seed_ledger.jsonl` | one line per batch/iteration: seeds, block, host. **Copy into the experiment ledger.** |
| `<loop_root>/STOP` | stop request (see below) |
| `<loop_root>/batches/bNNNN/` | policy snapshot (+ sha256 in the state) and collect log of each batch |
| `<loop_root>/raw/bNNNN/` | pulled raw episodes (pool; never deleted by the loop) |
| `<loop_root>/refreshes/rNNN/` | convert list + log per refresh |
| `<wm_runs_root>/c4_wm_loop_rNNN[.aK]` (+ `.stdout.log`, `.pid`, `.EXIT`) | WM refresh runs |
| `<policy work_root>/runs/c4_policy_loop_gNNN` | policy runs (LAUNCH.txt, train.log, pid, EXIT) |
| CPU host `<work_root>/<loop_id>/bNNNN/` | remote collection dirs (unique per loop; never reused) |

## Start / stop / check

Run from the deployed code root on gpuW (`/root/autodl-tmp/c4/v5/repo_<sha>`; configs reference it as
`REPO_SHA`, so make a concrete copy with `sed`).

```bash
# status (no ssh unless --live)
/root/miniconda3/bin/python scripts/c4/dreamer_loop.py status --config /root/autodl-tmp/c4/v5/loop.json [--live]
# plan without executing
/root/miniconda3/bin/python scripts/c4/dreamer_loop.py dry-run --config /root/autodl-tmp/c4/v5/loop.json [init flags]
# graceful stop: finishes the current tick/step and exits; detached jobs (collection, WM refresh, policy)
# keep running and are picked up again by the next `run` with the same config (same loop_root)
/root/miniconda3/bin/python scripts/c4/dreamer_loop.py stop --config /root/autodl-tmp/c4/v5/loop.json
# resume after a stop or crash: delete <loop_root>/STOP, then run again WITHOUT --init-* flags
```

To stop everything, stop the loop first. Then stop the policy cleanly with `touch <policy run_dir>/STOP`
(loop-launched trainers only). Stop a refresh or collection by its PID from `state.json`; that is the
process-group leader, so use `kill -TERM -- -<pid>`.

### The exact start command for ~11:00 (first v6 policy checkpoint)

The watcher `autopolicy_v6.sh` starts the v6 policy in `/root/autodl-tmp/c4/v5/runs/c4_policy_v6_full`
(`pid` file inside). It runs on the lane-calibrated `lanecal_v6/wm_s_best_ap.pt`, or on
`c4_wm_v6_cont/wm_s_best.pt` if calibration failed. Check `autopolicy_v6.log` to see which.

```bash
SHA=<deployed loop2 commit, e.g. the one in the report>
cd /root/autodl-tmp/c4/v5/repo_$SHA
sed "s/REPO_SHA/$SHA/g" configs/c4_dreamer_loop_gpuw.json > /root/autodl-tmp/c4/v5/loop.json
P=/root/autodl-tmp/c4/v5/runs/c4_policy_v6_full
PWM=$(grep -o 'wm_ckpt=[^ ]*' $P/LAUNCH.txt | cut -d= -f2)      # the WM the running policy uses
/root/miniconda3/bin/python scripts/c4/dreamer_loop.py dry-run --config /root/autodl-tmp/c4/v5/loop.json \
  --init-wm /root/autodl-tmp/c4/v5/runs/c4_wm_v6_cont/wm_s_best.pt --init-policy-run-dir $P --adopt-policy-pid $(cat $P/pid)
setsid nohup /root/miniconda3/bin/python scripts/c4/dreamer_loop.py run --config /root/autodl-tmp/c4/v5/loop.json \
  --init-wm /root/autodl-tmp/c4/v5/runs/c4_wm_v6_cont/wm_s_best.pt \
  --init-wm-val-log /root/autodl-tmp/c4/v5/runs/c4_wm_v6_cont/val_log.jsonl \
  --init-policy-wm "$PWM" --init-policy-run-dir $P --adopt-policy-pid $(cat $P/pid) \
  > /root/autodl-tmp/c4/v5/loop.nohup.log 2>&1 < /dev/null &
```

Prerequisites:

* gpuW `~/.ssh/loop_key` is authorised on the CPU box (done 2026-10-16).
* The code root `repo-<sha>` is deployed on the box (`/home/ubuntu/c4/repo-<sha>`, done for the report's
  commit).
* `/root/autodl-tmp/c4/v5/loop` must not exist yet.

Adoption checks that the PID is a `wms.control.train_cli` whose command line contains the run dir. The
first switch waits for the trainer's next `policy_<n>.pt` (50 updates, about 20–25 min), then sends SIGTERM
to that PID only.

## Configuration (`configs/c4_dreamer_loop_gpuw.json`; defaults in `DEFAULTS`)

| key | meaning |
|---|---|
| `mode` | `pipelined` or `sequential` |
| `loop_root`, `data_root`, `wm_runs_root` | where the loop writes |
| `disk_paths`, `min_free_gb` | disk guard: nothing new starts while free disk is below `min_free_gb` (5 GB) |
| `hosts.wm` | must be local (the orchestrator runs where the WM trains) |
| `hosts.policy` | local or `{"ssh": [...], ...}` |
| `hosts.cpu`, `hosts.<other>` | collector hosts |
| `hosts.*.code_root`, `python`, `work_root` | per host |
| `collect.hosts` | collector list. Leave a box out of it to reserve it for evaluations. `wait_for_absent` (default `scripts/boxops/eval_v5.sh`) also pauses a collector while an evaluation runs on its host (read-only `pgrep`; nothing is killed). |
| `collect.batch_size`, `workers`, `warmup`, `horizon`, `penetrations`, `pen_weights`, `demands`, `timeout_s`, `max_attempts` | collection settings. A failed batch is relaunched into the same directory (collect resumes) up to `max_attempts`. |
| `seeds.block`, `seeds.exclude`, `seeds.per_iteration` (sequential) | 965990–965993 are excluded (used by the live test) |
| `convert.max_episodes` | 0 = the whole pool |
| `convert.mb_per_episode` | 21 MB measured on gpuW |
| `convert.min_window` | minimum window size |
| `wm.stage_steps` | [60, 40] |
| `wm.extra_set`, `wm.env`, `wm.timeout_s` | WM refresh launch settings |
| `wm.val_metric`, `wm.val_tolerance` | sanity check |
| `wm.min_new_episodes` | 0 = back to back |
| `wm.max_refreshes` | maximum number of refreshes |
| `wm.lane_calib` | optional ≤ 25 min lane-AP calibration per accepted WM (`tools/lanehazard_calib.py`), off by default |
| `policy.config`, `config_without_policy_data`, `device`, `max_updates` | policy run settings (`max_updates` = 100000, so a resumed run keeps going) |
| `policy.env`, `allowed_roots` (`C4_ALLOWED_ROOTS`) | policy environment |
| `policy.stop_timeout_s`, `adopted_wait_max_s`, `max_restarts` | stop and restart limits |

A dead policy is restarted from its newest checkpoint. If it dies right after a switch, it falls back to the
previous WM.

## Timing per cycle (estimates; gpuW 4080S shared by WM refresh and policy)

| part | time | basis |
|---|---|---|
| SUMO batch, 50 episodes, 46 workers | about 3–5 min | measured about 48 s wall per 900 s policy episode in the 2026-10-16 baseline evaluation; plus warmup |
| pull + convert (background) | about 1–2 s per episode, plus startup | 1 episode: 16 s including Python start |
| WM refresh `[60,40]` | about 18 + 25 = 43 min training + about 3–5 min data loading + validation | about 50 min with the GPU to itself; **slower while the policy shares the GPU** |
| policy switch, loop-launched trainer | ≤ 1 update (about 25–40 s) + about 3 min restart (start windows) | |
| policy switch, first (adopted) trainer | up to about 25 min wait for its next `policy_<n>.pt` | the policy keeps training meanwhile |

The first policy switch comes about 1–1.5 h after the start (first batch, then a refresh). After that, one
switch per refresh, about every **55–85 min**.

**Seed budget:** 812 seeds remain, about 2.5–3 batches/h, so the block lasts about 5.5–6.5 h. After that
the collectors stop, the WM lane refreshes the whole pool once more, and the loop finishes. A longer run
needs an owner-approved new seed block (`traffic_shepherd/c4/seeds.py` + `seeds.block`).

## Failure handling

* **Crash/restart.** Every transition is persisted. `run` with the same config resumes where it stopped and
  re-attaches to detached jobs by PID + `/proc` start time. Allocated seeds are never re-allocated.
* **Transient errors** (ssh drop, rsync, busy box) are logged as `lane_error` / `step_error` and retried
  next tick. The loop exits after `max_step_errors` (×3 for lanes) consecutive errors in one lane.
* **ssh** retries rc 255 with backoff; rsync uses `--partial`, retries, and `--exclude *.incomplete-*`.
* **Collection failure.** A failed batch is relaunched into the same directory (`collect` skips complete
  episodes) up to `max_attempts`, then marked failed. Its seeds stay recorded as used.
* **WM refresh failure** (process died, timeout, nonfinite loss, worse validation): the previous WM is
  kept, the policy is not restarted, and the next refresh starts.
* **Policy start failure.** The trainer is restarted from its newest checkpoint; if it died right after a
  switch, it falls back to the previous WM and pool (`max_restarts`).
* **Disk.** Status `paused_disk` while free disk < `min_free_gb`. Converted pools are about 21 MB per
  episode, and the running policy's pool cannot be deleted before the next one exists. So each new pool is
  capped at half the space above the guard (the most recent episodes; `convert_window_reduced` in the
  log). Raw episodes are about 0.75–1.5 MB each and are never deleted.
* **Safety.**
  * Signals go only to PIDs the loop launched (a process-group leader with a matching start time) or to the
    explicitly adopted PID (start time + `wms.control.train_cli` + run dir checked).
  * No pattern kills.
  * Output dirs are never reused (`.aN` suffixes; remote dirs per `loop_id`).
  * Paths containing sealed/holdout/heldout/cal_split/chunk5/j4b/inc1 are refused.

## Verification (2026-10-16)

* Unit tests: `tests/unit/test_dreamer_loop.py` (fake backend; both modes, resume, STOP, rejection, disk
  pause, seeds, PID safety) and `tests/unit/test_c4_loop_starts.py` (weighted start sources).
* The live test on gpuW + CPU box is reported in the integrator's report. It used
  `configs/c4_dreamer_loop_livetest.json`, seeds 965990–965993, outputs under
  `/root/autodl-tmp/c4/v5/loop_test/` and `~/c4/loop_test/`.

## Known risks

* **Disk on gpuW** (about 9 GB free) limits each converted pool to about 70–140 recent episodes. "All
  episodes" needs about 21 MB × pool, so expand `/root/autodl-tmp` or lower `min_free_gb`.
* **GPU sharing.** WM refresh and policy on one 4080S slow each other down. Set `hosts.policy` to the PRO
  6000 box when it is available. That box then needs:
  * `v5_native` and `genrand_plain` synced once;
  * a `v6_longpair` dir holding only `C4_EPISODES_MANIFEST.json` (start windows read only its manifest);
  * a gpuW ssh key.

  The loop ships each WM and each converted pool to it with rsync.
* **The converter's world split is random over the pool** (`--val-frac 0.2`), so a v6_policy world can move
  between train and val across refreshes. The validation comparison against the previous WM also uses a
  slightly different val set (new v6_policy windows). Treat `wm_rejected` with judgement.
* **The first adopted switch** costs up to one checkpoint interval of waiting. The SIGTERM lands about 1
  update after a periodic checkpoint, never during a checkpoint write.
