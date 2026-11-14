# A102TS: traced re-run of world 988700 for the time-space figure. Prepared, NOT launched

Prepared 2026-11-14 06:25-06:35 (Box 3 time). Zero SUMO, nothing charged, queue installed but not started.
REGISTRATION.md and the ledgers were not edited. No git.

## What it runs

- Root `/root/a102ts` (new). Ledger `/root/a102ts/root/CHARGES.jsonl`, phase `tstrace_a102`, cap 2,400 sim-s.
  The cap is enforced in code by the A42 runtime: `check_capacity(2400+1200)` refuses.
- One world: **988700**, the first A94 panel world. It was fixed in advance and not chosen from outcomes.
- Two jobs, with the same job ids and the same A94 spec builder as the panel jobs:
  - `a94c_mappo60c_0000`: MAPPO60C alone.
  - `a94c_bank_searched_0000`: A75_BANK on the S5 bank 577a0564, over MAPPO60C 2abed317.
- Budget: 2 x 1,200 = 2,400 sim-s. Workers: 2.
- **Raw output is kept.** No `--drop-raw` and no drop call. Each episode keeps `RESULT.json.gz` (about 17-20 MB),
  `vehicles.jsonl.gz` (about 1 MB) and `observable.jsonl.gz` (about 3-4 MB). Sizes are from the A94 drop receipts.

## Files and shas

| file | sha256 |
|---|---|
| `/root/a102ts/a102_ts.py` (driver, thin wrapper over the unchanged A94 / A81 drivers) | e6475393... |
| `/root/a102ts/a102ts_check.py` (post-run check) | 9a3006ca... |
| `/root/a102ts/run_a102ts.sh` (stages) | 40e71552... |
| `/root/a102ts/ship/setup_a102ts.sh` (ran once: SETUP_OK) | 6291b6de... |
| `/root/a102tsq.sh` (queue; NOT started) | 5e4202fd... |

- Inputs:
  - S5 bank 577a0564944b2062e441d7ee603e7fe8090bede52a5dd6e8774d41ed9cb2d8b6.
  - Executor policy_00084 2abed31707b61121569af2936c23689827b52b28422df162751b230aa6a93de8. Binding 4a73e50f.
  - Panel file 720d19e1; panel_sha256 d85999ddaaf3eb583775d66dbabca168b4432966bc01e7193dfc79f244f7946c.
  - a75 digest d1af7e0f.
  - Runner nm_eval_runner 47fb9f76, the same as A94.
- Tree: `/root/a102ts/src` is a `cp -a` of `/root/a94/src`. It has 1,526 files, `diff -rq` against the source is empty,
  and it is not patched.
  - Root binding files and the panel file / pin are byte copies, `cmp`-checked against `/root/a94/root`. The binding
    files are also checked against `/root/a66/root`.
  - The env is `env_a94.sh` with BASE / SRC / A94_HOST_BASE switched to `/root/a102ts`.

## How spec identity with the panel jobs was verified

- **By construction.** The specs come from the unchanged A94 function `a94_bank.arm_specs`, with the A94 tags, sliced
  to index 0.
  - The only changes are bookkeeping: phase label, cap, log file name and root.
  - The phase does not appear in any spec field, which the dry run checked.
- **Against the stored records** (`run_a102ts.sh dry`, DRY_OK). For each job, the spec fields the runner copies into the
  record were compared one by one with the stored A94 wave records. Those records are read-only, sha-pinned
  (e126789b / 44951b2a), and their receipts 693be701 / 9230f4ae are cited in `A94_RESULTS_D2800_p60.json` sources. The
  compared fields are checkpoint, world, window, world_id, job_id and panel_sha256.
  - MAPPO60C: 23 fields, **0 differences**.
  - BANK_SEARCHED: 32 fields. The **only difference** is `checkpoint.a75_bank.path`
    (`/root/a94/root/a94_banks/...` -> `/root/a102ts/root/a94_banks/...`). The same sha 577a0564 was checked on disk.
  - The remaining spec fields are identical between the two arms: `deterministic=true`, `noise_seed=988700`,
    `capture=null`.
- **Pre-physics** (`run_a102ts.sh prephysics`, PREPHYSICS_OK, charged 0, root file list unchanged). This is the
  pre-episode part of `a75_d2forms.execute`, which was passed:
  - plant CONFIG == PRERUN_PLAN;
  - runner ok / learned;
  - A75 / A37 arms and A94_RANGES;
  - map check ok;
  - U0 residual == PRERUN_PLAN.

  It also checked:
  - the executor binding sha and the executor file sha are both 2abed317;
  - a75 digest d1af7e0f;
  - the reservation prefix `a94`;
  - capacity: 2,400 passes and 3,600 refuses;
  - published before the run = 0/0.

## Post-run check (`a102ts_check.py`, run by the queue)

- (a) It checks that `executor_sha256_loaded` = 2abed317 in both re-run records.
- (b) It compares re-run `J_veh_s` with the stored A94 value for world 988700 and prints MATCH / MISMATCH with both
  numbers. The stored values are MAPPO60C 42646.69246811499 and BANK_SEARCHED 27625.72399731639. It also compares:
  - safety counters;
  - for the bank arm, the a75 `activations`, `activations_by_family`, `issued` and `acts`.
- (c) It lists the raw files with their bytes and sha. It also compares them with the shas sealed in the A94
  `DROPPED_RAW.json`; that comparison is informational only, because gzip headers may differ. Finally it probes
  `observable.jsonl.gz` for the figure fields.
- The script was tested on a synthetic `/dev/shm` root with a deliberate +1.0 J offset on the bank arm. It printed
  MISMATCH with both numbers, exited with rc 3 and printed A102TS_CHECK_DONE. The scratch root was removed afterwards.
- **Note:** `A94_RESULTS_D2800_p60.json` holds arm means and contrasts, not per-world J. The per-world value comes from
  the A94 wave record. The receipts of those wave records are the ones cited in the results' `sources`.

## Figure inputs

- The time-space figure script `fig/make_fig_ts_v10.py` needs `IN/<tag>/observable.jsonl.gz` for the tags `mappo60c`
  and `bank_searched`. Each line holds `time_s`, `vehicles[{vid, s_m, v_mps}]` and `actions[{vid, oracle_plan}]`.
- The A75_BANK hook (`a75_holdforms.make_hook`) writes `oracle_plan=<plan family>` on every acceleration request of a
  plan member. This is the same field the a58 inputs carry (checked in `a58x_big_bank_0000`). **No script change is
  needed.**
- MAPPO60C alone has no plan actions, so its member set is empty. The top panel is drawn without the black overdraw, as
  intended.
- Caveat: the S5 bank's `mut_069966492c9abdcc` is an LcStrategyCore plan. A member with a lane-target-only command and
  no acceleration request gets no `oracle_plan` action at that second, so it is not overdrawn there. The check prints
  the counts of `actions_with_oracle_plan` and `plan_member_vids`.
- After the run, copy the two files to the laptop:

  ```
  ssh ... "sudo -n cat /root/a102ts/root/evaluation/a94c_mappo60c_0000/observable.jsonl.gz" > E:/research-dense/runs/active/C4-V7FT-20261017/a94_ts_trace/fig_inputs/mappo60c/observable.jsonl.gz
  ```

  Do the same for `a94c_bank_searched_0000` into `bank_searched/`.
- The figure caption says "5-6-CAV holds". Check that wording against the S5 plans
  (`rand_8b08c77a63f7d622`, `mut_069966492c9abdcc`, `mut_97dda2be6b45ccf0`); I did not verify the group sizes.

## Launch (coordinator, after registration)

```
timeout 60 ssh -o BatchMode=yes -i ~/.ssh/rented_box ubuntu@124.220.72.157 'sudo -n bash -c "setsid -f nohup bash /root/a102tsq.sh < /dev/null > /root/a102tsq.nohup 2>&1"' < /dev/null
```

- The queue runs once: it writes `QUEUE_STARTED` and refuses a relaunch.
- Before starting, it requires PREPARE_OK, DRY_OK and PREPHYSICS_OK, no `chain_run.out` and no CHARGES.
- It logs `/root/a102tsq.log`, including the "ALLOCATION START" line.
- It runs `run_a102ts.sh run` (identity is re-checked before physics), then the check. The last line is `A102TS_DONE`.
- To stop before the run starts: `sudo -n touch /root/a102ts/root/STOP_A94`.
- The runner charges one canary job per process group first (nm_finaleval.run_waves), with at most 2 workers.
- **Expected wall time: about 5-12 min.** Each episode takes about 3-4 min on this box under load (A94: 768 episodes
  in 65 min on 46 workers). If the canary runs the two jobs one after the other, it takes up to about 2x. Box 3 is
  running the a101 queue (load about 44 on 48 cores), so 2 extra processes oversubscribe it slightly.

## Deviations / disclosures

1. Ledger phase `tstrace_a102` instead of `searchedbank_a94`, and cap 2,400. The job ids and wave names (`A94C_*`) are
   kept the same as the panel jobs, so the run is in a separate root under the same ids.
2. No mechanics gate before the run. Replay identity is judged afterwards by the J MATCH / MISMATCH of the check, which
   is exactly the A81 mechanics criterion (J exact).
3. `A94_HAND_SHA256=none` is set in the wrapper's environment so that only BANK_SEARCHED is built. That variable does not
   affect the BANK_SEARCHED or MAPPO60C specs.
4. `prepare` also byte-copied the frozen A75-D2 bank (c325ec1d) into `root/a94_banks`, because the A94 prepare does that.
   It is unused.
5. Files outside `/root/a102ts`: `/root/a102tsq.sh`, `/tmp/a102ts_setup.out` and `/tmp/a102ts_ls_before.txt`. The
   future `/root/a102tsq.log` and `/root/a102tsq.nohup` will also be outside it.
6. The stored A94 world-988700 records were read, but they were already known to the coordinator. No other panel world
   was read.
