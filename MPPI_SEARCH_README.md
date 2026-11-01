# tools/coopmwm/mppi_search.py: policy-centred group plan search (arm B, R)

This module is generic and discovery-pure. Plans are joint perturbations of the policy's own imagined acceleration trajectory. They are scored on WM futures that are paired across candidates. The O1-O4 families are never used as seeds; they appear only in `posthoc_label`. Everything is CPU numpy, deterministic given `seed`, and JSON-serialisable.

## API
- `group_components(rows, chain_m=100, max_size=6, exclude=committed_vids)` returns disjoint groups of 2-6 CAVs. Grouping uses 1-D chaining on `s_m` across all lanes and the ramp. Members are ordered leader first. For a component larger than 6, the rule takes the contiguous window of 6 that spans the most lanes, then the smallest spread, then the most downstream window. The remainder is re-chained and the rule repeats.
- `search(record, group, policy_traj, bounds, evaluate, regime, mappo_traj=None, covariance=None, library=None, seed=None, config=None)`
  - `evaluate(record, actions, K, H) -> {id: [K costs or {'team_delay':..}]}`. Comparators `policy` and `MAPPO` are always the first two actions in every call.
  - `policy_traj` and `mappo_traj` are absolute m/s², members (group order) x seconds, with length >= `max_duration_s`.
  - Returns `accepted`, `plan` (the Thompson choice), `option='MPPI'`, `expected_saving` (vs MAPPO), `expected_saving_vs_policy`, `topk` (each with `saving_vs_mappo`, `saving_vs_policy`, `p_best`), `probabilities` (sum to 1), `thompson`, `comparator_means`, `finalists`, `iterations`, `centre_open_loop_gap`, `scaling`, `screen`/`final` (K, H), `imagined_futures` and `config`.
- Plan dict fields: `members`, `duration_s`, `profile` (absolute, m x D per second, clipped to bounds), `knot_times_s` (<=10), `perturbation_knots`, `perturbation_profile`, `perturbation_canonical` (m x 10 over [0, 300] s), `canonical_span_s`, `origin` (one of centre, mappo, library, sampled), `family='MPPI'`, `regime`, `id`.
- `acceleration_at(plan, elapsed)` reads the profile. `plans.acceleration_at` delegates to it for plans that have a `profile`. `final_horizon(D)`, `knot_count(D)` and `scaling(m, n)` expose the rules below.
- `CovarianceModel(sigma)` holds one model per regime and size. Its methods are `.update(regime, perturbation_canonical, weight>0)`, `.get`, `.favoured_directions` and `.to_json`/`.from_json`.
- `PlanLibrary()` provides `.add(descriptor, plan, adv>0)` and `.nearest(descriptor, n)`. `describe(group, regime)` builds the descriptor. `learn(plan, adv, descriptor, cov, lib)` stores only SUMO-positive plans.
- `sample_unscored(group, policy_traj, bounds, regime, seed, covariance)` is for arm R: the same distribution, with no WM scoring.
- `posthoc_label(plan, family_seed_prototypes, policy_traj)` returns the nearest family. It returns 'new' if no family is nearer than the unchanged policy.

## Registered rules (`REGISTERED`; owner 2026-11-01: durations up to 300 s)
- **Durations:** {30, 60, 90, 120, 150, 180, 240, 300} s, with `max_duration_s` (default 300) as a parameter.
- **Knots:** n(D) = min(10, max(min(8, 1+D//10), 1+ceil(D/30))). For D=60, 180, 240 and 300 that gives 7, 8, 9 and 10 knots.
- **Noise:** z ~ N(mu, Sigma) on n(D_max)=10 canonical knots/member spanning [0, D_max] s from the decision (absolute frame). A plan of duration D uses the curve on [0, D) and then returns to the policy. Sigma_0 = 0.5² · kron(I_members, SE-kernel with a length of one knot). The 0.5 is the policy's registered exploration std. Members start isotropic.
  - CPU benchmark, savings vs a policy cost of 124.8: the absolute frame gave 106/95/74 for 2/4/6 members, against 84/61/23 for the normalised frame. A range/√12 = 1.3 sigma failed at 6 members.
- **Scaling:** d = members x n(D_max); population N = ceil4(max(32, 2d)); elites E = ceil(N/4); finalists = 2·top_k. With 2 members d=20, N=40, E=10; with 6 members d=60, N=120, E=30.
- **MPPI** (default) solves for the temperature that gives Kish ESS = E. `method='cem'` uses the top E instead.
- **Per-iteration update:** the covariance and the duration probabilities each use a one-pseudo-sample prior.
- **Horizons:**
  - Screen: K=4, H = min(600, max(120, D_max)) = 300.
  - Final: K=8, **H_final = min(600, max(300, D+120))**, where D is the longest finalist duration. All finalists and both comparators share that horizon and its futures.
  - Iterations: 3. Cost: (N+2)·4·3 + 10·8 futures, which is 584 for 2 members and 1544 for 6.
- **Top-k** (k=4): candidates with mean saving vs MAPPO > 0. Thompson uses 4000 bootstraps of the K8 future indices to get P(best); the executed plan is the winner of one more resample.
- **Learned covariance:** (d·Sigma_0 + n_eff·M_w)/(d + n_eff), where n_eff is the Kish count. This makes it invariant to the reward unit.
- **Library:** n nearest entries within the same regime and size, re-gridded onto the current canonical span, with no cutoff.

## Integration recipe (live service: `plan_service.main` / `acceleration.Controller`)
1. In `plan_service`, replace `search(r, bounds, lambda a,k,h: ...)` with `mppi_search.search(r, g, policy_traj, bounds, lambda rec,a,k,h: evaluate(rt,rec,a,k,h,chunk_actions=w,trim=False,pad_actions=w), regime, mappo_traj, cov, lib, seed)`.
   - `g` comes from `group_components(r['rows'], exclude=committed)`.
   - `regime` is `exploration.regime(rows)`.
   - `seed` is `int(digest(r)[:15],16)`.
   - Widen the request whitelist for the trajectories, or compute them in-service.
2. **policy_traj and mappo_traj:** take the group's applied raw accelerations per second for 300 s from a WM rollout of the 'policy' and 'MAPPO' comparators, on future 0 or as the mean over K. `centre_open_loop_gap` (open-loop centre minus closed-loop policy on the same futures) checks this extraction.
3. **MUST:** `plan_wm.profile_table` builds `range(300)` steps. With H_final up to 420 s (600 max), extend the table to H. Otherwise background/'policy' residuals silently stop at 300 s.
   - `wm.adapted_futures((H,))` is already parameterised for any H, with 2·H steps. Only the protocol `HORIZONS` (60/120/300/600) needs the H_final rule registered.
4. `plan_wm` now treats `'profile' in a` like knot plans: absolute values with policy-residual background. On the SUMO side, `Controller` calls `plans.acceleration_at`, which handles profiles. Replace its `len(group)==2` gate and `plans.group` with `group_components`.
5. **Trust:** `option`/`family` = 'MPPI', so the bucket is 'MPPI|regime'. `exploration.OPTIONS/BUCKETS` must add it, otherwise `trust()` raises at `keys.index`.
6. After a positive SUMO advantage, call `learn(plan, adv, describe(g, regime), cov, lib)`. Persist `cov.to_json()` and `lib.to_json()`. For arm R, use `sample_unscored`, which is matched by size and duration.

## Tests
`python -m pytest -q tools/coopmwm/test_mppi_search_cpu.py` passes 15 of 15, with system Python 3.14 and with rl_driving.
- The tests cover groups (ramp and multi-lane, HDVs ignored, exclude, the >6 rule), knots ≤10, durations ≤300, the H_final rule, bounds, origins, comparators first, and that family_seeds is never called.
- They check the profile accessor, improvement for 2 and 6 members (MPPI and CEM), and that a 270 s optimal hold is found with D ≥ 240 and H_final ≥ 360.
- They also cover top-k and Thompson (P sums to 1 and favours the best), rejection when MAPPO is optimal, the covariance shift and unit invariance, library seeding, the post-hoc label, and determinism.
- `test_acceleration_cpu.py` shows the same 8 environment-only failures before and after this change (missing tools.wmteach/leverprobe, fcntl and COOPMWM_MAPPO on Windows).
