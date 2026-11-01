# Exact SUMO plan grader (lever 5) — `tools/coopmwm/exact_grade.py`

**Grade** for every executed plan in arms B and R: `grade = J_without - J_with` (veh-s; positive means the plan helped). J is the increase in the cumulative `team_delay_vs` (road speed deficit + source waiting, the comparator J) over `[t0, t_end]`. `t_end = min(t0 + D + 120, episode end)`; a window cut at 1200 s is accepted by MAIN and flagged `truncated`. Totals come from the per-second accel_plant fingerprints (`branch_eval.totals_at`).
- **with-plan**: the real training episode's RESULT. No new physics is run.
- **without-plan**: one `accel_plant.episode` rerun to `t_end` with the same world, rollout checkpoint (`{arm}_checkpoint_{u-1}`, SHA-checked), `noise_seed` and runtime. Seconds 300..t0-1 replay the recorded one-second samples, as the `exact_plan` twins do. Every fingerprint from 300 to t0 must be bitwise equal to the original (checked inside the loop and again after the run). On any difference the receipt is `stopped` (PrefixMismatch), never imputed and never retried.
## Fairness of the without-plan branch (marginal effect of plan P, MAIN overlap rule)
- From t0 there is no guide. P's members, and every CAV not in another plan, follow the policy's own stochastic actions, which is the behaviour the PPO baseline describes.
- **Other plans:** every OTHER executed plan active in the window is replayed exactly as recorded: its per-second absolute accelerations, same members, same start/end/truncation. They are listed in `replayed_plans` and the per-second replay counts are checked against the tape.
- **Unresolved overlaps:** if another plan in the window shares members with P, the spec is `overlap_unresolved`. It gets a receipt with no physics and no charge, and is excluded from imitation and trust.
- **Same noise draws (CRN):** the torch generator is fast-forwarded through the original's exact draw schedule: policy `randn(1,1)` and `randn(n,1)` per second, `rand` per eligible guide query, R-only `randint` per matched random plan, and `rand` for beta on each accepted proposal. The AR(1) noise memory is restored at t0.
  - Verified bitwise at t0: the live own action must equal the recorded one (`crn_verified_t0`).
  - After t0 the branch consumes the original's guide and beta draws at the same seconds, so innovations stay identical while CAV membership matches. `aligned_until_s` records the first divergence.
- **Registered fallback:** if the t0 check fails, P's members use the policy's mean action (`noise_mode_used='mean'`). The expected rate is 0; treat any fallback as a code-drift alarm.
- **Why not mean action by default:** a deterministic baseline would compare the plan against a different policy. CRN keeps the estimate unbiased and cancels shared noise.
## Receipt schema (`coopmwm/exact_grade/v1`, `receipts/<grade_id>.json`, sealed with `receipt_sha256`)
- Identity: `grade_id = xg_<episode_id>_t<t0>`, `spec_sha256`, `episode_id`, `result_path`, `world_seed`, `arm`, `update`, `training_seed`.
- `plan` (`id`, `members`, `duration_s`, `start_s`, `option`, `regime`, `plan_sha256`), `window`, `replayed_plans`, `overlap_clash_members`, `overlapping_plan_starts`.
- Outcome: `status` (`graded` / `stopped` / `failed` / `overlap_unresolved`), `grade_veh_s`, `J_with_veh_s`, `J_without_veh_s`, `window_with` / `window_without` (all metric deltas, including safety).
- `noise` (`crn_verified_t0`, `noise_mode_used`, `aligned_until_s`, `replayed_plans`), plus `prefix`, `charged_sim_s`, `attempt`, `host`, `pid` and timestamps. Specs are sealed with `spec_sha256`, which binds `result_sha256` and `checkpoint_sha256`.
## Queue / worker (cpuQ)
- Layout: `queue/`, `claims/` (one O_EXCL claim per attempt: pid, host, /proc start time), `attempts/`, `receipts/` (never overwritten), `TEMPERATURE_<arm>_<seed>.json`. `GRADER_CONFIG.json` holds `{run_root, phase:"exact_grade", path_map}`.
- Run: `cd <pytools>; nohup python -m tools.coopmwm.exact_grade supervise --root <grade_root> --workers N &`. Workers run at nice 19 with 1 thread; PIDs are logged in `WORKERS.jsonl`. Nothing is ever killed.
- Each attempt is charged before any physics (`accel_storage.reserve`, phase `exact_grade`). If the budget is refused, `GRADER_BLOCKED.json` is written and grading stops.
- Enqueue is idempotent. A dead worker's claim is re-queued, at most 2 attempts in total, then the receipt is `failed`. Physics or executor errors fail with no retry.
- **Budget:** `sim_s = t_end` (from t=0, warmup included; max 1200; about 700–1100 per plan). `quote()` gives exact totals; unresolved plans cost 0.
- **Smoke** (one real completed episode, one plan, run once; idempotent): `python -m tools.coopmwm.exact_grade smoke --run-root <run root> --episode-id warm_u001_B_<seed> --grade-root <dir> [--binding <runtime binding json>] [--t0 T] [--charge accel|local] [--path-map '{}']`. It prints status, grade, prefix_pass, crn_verified_t0, aligned_until_s and replayed_plans.
## Integration recipe (S-COOPMWM `warm_runner.run_arm`, B and R)
0. Register the `exact_plan_grading` phase inside the 7.5M cap. Create the grade root and config, then start the supervisor under nohup. Gate: B's first update does not apply until the supervisor PID is alive. Run the smoke after the first B episode.
1. After `eps = wave(...)`, call `GradeQueue(gr).enqueue_episode(ep, meta)` for each episode. `meta`:
   - `run_root`; `result_path = root/jobs/<id>/RESULT.json`; `result_sha256 = digest(ep)`; `update = u`; `arm`; `training_seed = SEED`;
   - `checkpoint_path = root/{arm}_checkpoint_{u-1:03}.pt`, its `sha`, and `checkpoint_update = u-1`;
   - `noise_seed = xg.train_noise_seed(seed, u, SEED)`;
   - `runtime`, `sumo_checkpoint = b['checkpoint']`, beta, `warm_history`, `accel_bounds`.
2. Before `update(...)`:
   - `R = q.receipts()`; `T = q.frozen_temperature(arm, SEED)`;
   - `W = xg.imitation_weights(R, arm, T, SEED)`;
   - `pairs = [p for e in W['weights'].values() for p in xg.imitation_pairs(read(e['result_path']), e)]`;
   - `metrics = xg.exact_update(policy, opt, eps, rng, pairs, b['awr']['coefficient'])`. This replaces `accel_learner.update`: own-step PPO, all-step critic, imitation only from exact-positive plans.
3. Use `tr = xg.trust_evidence(R, beta, arm=arm, training_seed=SEED)` instead of `trust(beta, running)` (same beta/bucket shape). Pass `[dict(executed=True)]*tr['graded']` as `cumulative` to `before_update`. `tr['harm_guard']` is guard 3b on exact grades.
4. Write `W` (including `T`) and `xg.lag_report(q, u)` into `{arm}_UPDATE_{u}.json`, and the loss into COOPMWM_DESIGN.md.

**Temperature and imitation (MAIN):** weights are uniform (1.0 for each positive plan) until the 20th exact grade of the arm/seed. Then T = max(1 veh-s, RMS of the first 20 grades by completion, both signs) is frozen in `TEMPERATURE_<arm>_<seed>.json` (first write wins). Weight = `min(20, exp(g/T))` for g > 0, else 0. Every update imitates the whole buffer of exact-positive plans.
## Tests
`python -m pytest -q tools/coopmwm/test_exact_grade_cpu.py`: 16 passed (rl_driving, Windows, CPU), using a fake plant behind the real `accel_plant.episode`, `Controller` and `Policy`. Coverage:
- B and R grades equal an independent oracle bitwise: the original live controller with only P suppressed and other plans executing live (other plans are replayed);
- PrefixMismatch STOP; post-hoc fingerprint check; own actions and shared noise; replay equal to the recorded accelerations; membership split and mean fallback;
- `overlap_unresolved` with no physics, no charge and no imitation; J and window arithmetic and budget; uniform-then-frozen temperature and the whole buffer;
- `exact_update` masking; queue resume, orphan, exhausted, executor-error and budget-blocked cases; trust and harm guard; smoke.

`supervise` and `accel_charge` are Linux-only (fcntl, /proc) and not exercised here.
