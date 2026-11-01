# Exact SUMO plan grader (lever 5) — `tools/coopmwm/exact_grade.py`

**Grade** for every executed plan in arms B and R: `grade = J_without - J_with` (veh-s; positive means the plan helped). J is the increase in the cumulative `team_delay_vs` (road speed deficit + source waiting, the comparator J) over `[t0, t_end]`. `t_end = min(t0 + D + 120, episode end)`, so a window is truncated at 1200 s when `t0 + D > 1080`; the receipt says `truncated`. Totals come from the per-second accel_plant fingerprints (`branch_eval.totals_at`).
- **with-plan**: the real training episode's RESULT. No new physics is run.
- **without-plan**: one `accel_plant.episode` rerun to `t_end` with the same world, rollout checkpoint (`{arm}_checkpoint_{u-1}`, SHA-checked), `noise_seed` and runtime. Seconds 300..t0-1 replay the recorded one-second samples, as the existing `exact_plan` twins do. Every fingerprint from 300 to t0 must be bitwise equal to the original: the episode checks this inside its loop, and it is checked again after the run. On any difference the receipt is `stopped` with a `PrefixMismatch` reason. Stopped grades are never imputed and never retried.
## Fairness of the without-plan branch
- From t0, `GradeController` has no plan and no guide. The group and every other CAV follow the policy's own stochastic actions, which is the behaviour the PPO baseline describes.
- **Same noise draws (CRN).** The torch generator is fast-forwarded through the original's exact draw schedule for 300..t0-1:
  - per second: policy `randn(1,1)` then `randn(n,1)`;
  - `rand` per eligible guide query;
  - R only: `randint` per matched random plan;
  - `rand` for beta on every accepted proposal.
  The AR(1) noise memory is restored from the recorded t0 sample.
- This is checked bitwise at t0: the live own action must equal the recorded one (`crn_verified_t0`). After t0 the branch consumes the original's guide and beta draws at the same seconds but never acts on them. So each CAV gets the identical innovation while CAV membership matches. `aligned_until_s` records the first second where membership diverges; after that the draws are still valid and come from the same seeded stream.
- **Registered fallback** (`noise_mode_used='mean'`): if the t0 check fails, plan members use the policy's mean action and everyone else keeps sampling. The expected rate is 0, so treat any fallback as a code-drift alarm.
- **Why not mean action by default:** with-plan non-members are stochastic. A deterministic baseline would compare the plan against a different policy. CRN keeps the estimator unbiased and cancels shared noise.
- Later plans that start inside the with-plan window are listed in `overlapping_plan_starts`. They are part of the realised with-plan outcome (see open questions).
## Receipt schema (`coopmwm/exact_grade/v1`, `receipts/<grade_id>.json`, sealed with `receipt_sha256`)
- Identity: `grade_id = xg_<episode_id>_t<t0>`, `spec_sha256`, `episode_id`, `result_path`, `world_seed`, `arm`, `update`.
- `plan` (`id`, `members`, `duration_s`, `start_s`, `option`, `regime`, `expected_saving`, `plan_sha256`) and `window`.
- Outcome: `status` (`graded` / `stopped` / `failed`), `grade_veh_s`, `J_with_veh_s`, `J_without_veh_s`, plus `window_with` / `window_without` (all metric deltas, including collisions, teleports and hard_brake for safety).
- `noise`: `crn_verified_t0`, `noise_mode_used`, `aligned_until_s`.
- Also: `prefix`, `overlapping_plan_starts`, `charged_sim_s`, `attempt`, `host`, `pid`, `started_unix`, `finished_unix`.
- Specs (`queue/<id>.json`) are sealed with `spec_sha256`, which also binds `result_sha256` and `checkpoint_sha256`.
## Queue / worker (cpuQ)
- Directory layout: `queue/` (specs), `claims/` (O_EXCL claim per attempt holding pid, host and /proc start time), `attempts/` (history), `receipts/` (final, never overwritten). The grade root's `GRADER_CONFIG.json` holds `{run_root, phase:"exact_grade", path_map}`.
- Run it as `cd <pytools>; nohup python -m tools.coopmwm.exact_grade supervise --root <grade_root> --workers N &`. Workers are owned subprocesses at nice 19 with 1 torch thread, and their PIDs are logged in `WORKERS.jsonl`. The supervisor never kills a process.
- **Charging:** each attempt is charged before any physics via `accel_storage.reserve(..., phase='exact_grade')`. If the budget is refused, the supervisor writes `GRADER_BLOCKED.json` and stops; nothing runs and nothing is imputed.
- **Resume:** enqueue is idempotent (same SHA is a no-op; a different spec raises). A claim whose worker is dead goes to `attempts/` and the grade is re-queued, up to 2 attempts, then gets a `failed` receipt. A claim on a foreign host or a still-live PID is never taken over. Physics or executor errors produce a `failed` receipt with no retry.
**Budget:** each plan costs `sim_s = t_end` (from t=0, warmup included, like every accel_plant job; max 1200). Plans start between 300 and 900 s with D up to 300, so a plan typically costs 700–1100 sim-s, in line with the owner's ~1,000 estimate. `quote([(t0, D), ...])` gives the exact total.
## Integration recipe (S-COOPMWM `warm_runner.run_arm`, B and R)
0. Register the `exact_grade` phase in the budget quote, re-quoted within 7,500,000. Create the grade root and `GRADER_CONFIG.json`, then start the supervisor under nohup. **Gate:** the first B update must not apply until the supervisor PID is alive.
1. After `eps = wave(...)`, call `GradeQueue(grade_root).enqueue_episode(ep, meta)` for each episode. `meta`:
   - `run_root`; `result_path = root/jobs/<ep id>/RESULT.json`; `result_sha256 = digest(ep)`; `update = u`; `arm`;
   - `checkpoint_path = root/{arm}_checkpoint_{u-1:03}.pt`, its `sha`, and `checkpoint_update = u-1`;
   - `noise_seed = xg.train_noise_seed(seed, u, SEED)`;
   - `runtime = b['runtime']`, `sumo_checkpoint = b['checkpoint']`, the beta used, `warm_history`, `accel_bounds`.
2. Before `update(...)`:
   - `R = q.receipts()`;
   - `T = xg.temperature(R, b['awr']['temperature']*UNIT)`;
   - `W = xg.imitation_weights(R, T['temperature_veh_s'], consumed)`, where `consumed` = grade ids recorded in earlier `{arm}_UPDATE` files;
   - `pairs = [p for e in W['weights'].values() for p in xg.imitation_pairs(read(e['result_path']), e)]`;
   - `metrics = xg.exact_update(policy, opt, eps, rng, pairs, b['awr']['coefficient'])`. This replaces `accel_learner.update`: PPO on own steps only, critic on all current steps, no critic plan advantage.
3. Replace `trust(beta, running)` with `tr = xg.trust_evidence(R, beta)`; its beta and buckets have the same shape. Pass `[dict(executed=True)]*tr['graded']` as `cumulative` to `before_update`. `tr['harm_guard']` is guard 3b computed on exact grades.
4. Write `W`, `T`, the newly consumed ids and `xg.lag_report(q, u)` (grading lag per update) into `{arm}_UPDATE_{u}.json`. Add the loss to COOPMWM_DESIGN.md.

Temperature rule (registered here; confirm): once there are at least 10 grades, `T = max(1, RMS of all exact grades so far, both signs)`. Before that, T is the frozen AWR temperature (about uniform weights). Weight = `min(20, exp(g/T))` for g > 0, otherwise 0. Each plan is imitated once.
## Tests
`python -m pytest -q tools/coopmwm/test_exact_grade_cpu.py` gives 14 passed. Together with the awr, branch_eval and mppi_search suites: 50 passed (rl_driving, Windows, CPU). The fake plant runs behind the real `accel_plant.episode`, `Controller` and `Policy`. The tests cover:
- the grade equals an independent oracle (the original live controller with a scripted guide that never executes from t0), bitwise for every fingerprint, for both B and R (R includes the randint draws);
- a PrefixMismatch gives a STOP receipt, with no retry and no imitation;
- the post-hoc fingerprint check;
- the policy's own actions and shared noise; a membership split and the mean fallback;
- window and J arithmetic, truncation, and budget;
- only positive grades are imitated (temperature, clip, consumed);
- `exact_update` masking;
- queue idempotence, resume, orphan recovery, exhausted attempts, executor error, and budget blocked;
- trust and the harm guard.

The `supervise` and `accel_charge` paths are Linux-only (fcntl, /proc) and are not exercised here.
