# Stage 1c release results — plateau-decline demand, R1/R2 release rules vs reference holds (p20)

Date: 2026-09-04. Status: **information-pass complete on all four cells
(seeds 7401-7520, p = 0.20); no go/no-go criterion is attached to this
stage.** The p30 stratum remains deferred by order. Arm semantics follow
`experiments/idm/stage1c_release_contract.py`: five arms per seed —
D (damping only), W_ORACLE (hold to t=1200, plateau end, reference),
W_R1 (arrival/discharge + speed release rule), W_R2 (treatment-blind
density release rule), W_HOLD (hold to the 900 cap) — all branching from D
at t_d = 250 under the frozen damping controller and the frozen 10 m/s,
tau+3.0, 200-400 m band wall with 5 s taper and tau restore.

## Inventory and integrity

- Seeds completed: **120/120** (2200: 7401-7430, 2400: 7431-7460,
  2600: 7461-7490, 2700: 7491-7520); missing: none.
- Branches: **600** (5 per seed). Integrity failures: **0** — every wall
  branch is prefix-identical to D through t_d = 250 and diverges at t_d+1
  or later (distribution at 2600, representative: 84/120 arms at 251, the
  rest 252-274 — later only when no CAV is in the band at engagement; the
  wall changes nothing until an in-band CAV exists).
- D repeat identity: **12/12 exact** (every 10th seed).
- Zero collisions and zero teleports on all 600 branches.
- Terminal note (recorded limitation): every branch ends at
  t ≈ 1915-1917 s — after the 1800 s demand end, consistent with
  post-demand clearance — but the 1c runner does not propagate the env's
  termination reason into the trace metadata, so the strict
  natural-clearance evidence of earlier stages is not re-derivable from the
  seed JSONs; the uniform per-cell end time across all five arms is the
  recorded terminal consistency check.
- Cap semantics (as run): `wall_cap_s` is an absolute controller step, so
  "hard max 900" / W_HOLD taper at t = 901 = **650 s hold** after the
  t = 251 engagement, and W_ORACLE tapers at t = 1201 (plateau end,
  950 s hold). The doc said "hard max 900 s"; the run implements an
  absolute cap. Numbers below report the as-run semantics.

## Release timing (replayed from the recorded 1 Hz rule inputs)

Taper-start step (wall engages at t = 251; earliest rule release t = 430;
hard max t = 901; plateau ends t = 1200):

| cell | W_ORACLE | W_R1 | W_R2 | W_HOLD |
| --- | --- | --- | --- | --- |
| 2200 | 1201 (all) | 901 (all) | med 901, range [430, 901], **33% by rule** | 901 (all) |
| 2400 | 1201 (all) | 901 (all) | med 901, range [430, 901], **43% by rule** | 901 (all) |
| 2600 | 1201 (all) | 901 (all) | med 901, range [539, 901], **23% by rule** | 901 (all) |
| 2700 | 1201 (all) | 901 (all) | med 901, range [430, 901], **47% by rule** | 901 (all) |

Every release in every arm happens **inside the plateau** (t ≤ 1201): no
release rule or cap waits for the demand decline.

**R1 is provably inert.** On all 120 seeds the W_R1 branch digest is
**bit-identical to W_HOLD** — the rule never fires. Mechanism: the R1 speed
gate requires approach mean speed (200-800 m) ≥ 0.8 × 31.29 = 25.03 m/s,
but the 200-400 m part of that window is the walled band running at 10 m/s
for the whole hold, so the gate is unreachable while the wall is engaged;
the arrival ≤ 0.9 × discharge condition additionally requires an emptying
approach that plateau demand never produces. R1 as specified degenerates to
the fixed hold by construction, not by bad luck.

**R2 fires on 23-47% of seeds** (density < the treatment-blind 90th-percentile
threshold; median threshold 7.78 / 8.89 / 9.44 / 10.00 veh/km/lane across
cells), releasing 430-901 s into the plateau.

## Axes vs D (paired per seed; Δ = arm − D)

Median Δ incremental delay (s), interquartile range, fraction of seeds with
delay loss, and Wilcoxon signed-rank p (n = 30 per cell):

| cell | W_ORACLE | W_R1 (= W_HOLD) | W_R2 |
| --- | --- | --- | --- |
| 2200 | **+2752** [1796, 3825], loss 100%, p<1e-4 | +2307 [1108, 3554], loss 97%, p<1e-4 | +2014 [941, 3280], loss 90%, p<1e-4 |
| 2400 | +1972 [−2283, 3895], loss 63%, p=0.058 | +1323 [−2830, 2683], loss 57%, p=0.73 | +908 [−2621, 2635], loss 60%, p=0.66 |
| 2600 | **−3081** [−6307, −1037], loss 23%, p=0.006 | −4516 [−8267, −2235], loss 20%, p=3e-4 | **−4582** [−7961, −1677], loss 20%, p=3e-4 |
| 2700 | **−4921** [−8940, +727], loss 30%, p=8e-4 | −6283 [−10205, −560], loss 23%, p=1e-4 | **−7053** [−10399, −560], loss 23%, p=2e-4 |

The **sign flips with demand**: below the transition the held wall is a
significant delay *cost* (all 30/30 seeds lose at 2200); at and above 2600
the same hold is a significant delay *benefit* (−3.1k to −7.1k s median).
2400 is the heterogeneous hinge (p25 ≈ −2.3k to −2.8k: a quarter to a third
of seeds already improve).

Merge discharge (completed vehicles, median Δ): W_ORACLE loses discharge at
2600/2700 (−14.5 and −13.5, p<1e-3 — the 950 s hold pushes completions past
the branch end); the 901-capped arms hold discharge at 0 median everywhere
and *gain* on 43% of seeds at 2700 (p≈1.5e-3). D baselines per cell:
703 / 756 / 808 / 834 completed; D delay medians 10.2k / 17.7k / 31.2k /
38.7k s.

## Why the sign flips: onset timing vs the t=250 decision

D health-collapse onset median step: **393 / 192 / 180 / 161** across
2200 → 2700; onsets at or before the t = 250 engagement:
**5/26, 20/30, 27/30, 29/30** of D-collapsed seeds. The wall cannot prevent
a collapse that starts before it engages — at 2200 it can (and prevents
46% of would-be collapses), at ≥ 2600 it cannot (0%), it can only shorten
and meter them. This brackets the Part-1 boundary (beneficial-fraction sign
change between 2400 and 2500 in the generation stratum) from the
timing side: the same 251-step engagement is early enough at 2200 and too
late at ≥ 2600 in exactly the opposite senses (prevention possible but not
yet needed vs needed but no longer possible).

## Oracle recovery and duration repair

Recovery = among seeds where D sustains a health collapse (h < 0.15 for
≥ 60 s after t = 100), the fraction whose wall arm never sustains one:

| cell | D collapsed | W_ORACLE | W_R1 (= W_HOLD) | W_R2 |
| --- | ---: | ---: | ---: | ---: |
| 2200 | 26/30 | **46% (12/26)** | 46% (12/26) | 31% (8/26) |
| 2400 | 30/30 | 7% (2/30) | 7% (2/30) | 3% (1/30) |
| 2600 | 30/30 | 0% | 0% | 0% |
| 2700 | 30/30 | 0% | 0% | 0% |

Median collapse-duration repair (Δ duration, arm − D): −81 to −230 s per
cell-arm (D durations 482 / 758 / 879 / 927 s) — a 10-30% shortening at
2600-2700 where prevention is impossible. The wall also eliminates the only
physical collapses in the study (2 seeds in D at 2700 → 0 in every wall
arm).

## Safety flags (predeclared HeadroomMargins budgets, screening only)

- Collisions / teleports: **zero excess on all 600 branches**.
- Low-TTC exposure: budget exceeded on 0 / 1-2 / 3 / 2 seeds per cell per
  wall arm (2200 → 2700) — isolated, not systematic.
- **Hard braking is the persistent cost**: median hard-brake excess +98 to
  +300 vehicle-steps vs D, exceeding the max(2, 10%) budget on ≥ 97% of
  seeds in every cell and arm. Followers braking into the walled queue are
  the mechanism (damping only commands the p = 0.20 CAVs); this is a
  dose-linked price of every hold profile measured here, including on the
  cells where delay improves.

## R2 vs the fixed hold, paired on the seeds where R2 fired

On by-rule seeds only (n = 10 / 13 / 7 / 14 at 2200 / 2400 / 2600 / 2700),
median Δdelay R2 vs W_HOLD on the same seeds: 1133 vs 1408, 845 vs 1214,
−4518 vs −4017, −7469 vs −6283 s; R2 better on 6/10, 8/13, 2/7, 6/14 with
equal paired discharge. Earlier release buys a small, consistent median
delay improvement and never a discharge penalty — but the per-seed win rate
(50-62%) says the density threshold alone is not a sharp release signal.

## Reading (no gate attached)

1. The wall's plateau-decline value is **regime-dependent with a clean
   mechanism**: delay cost + 46% prevention at 2200; hinge at 2400;
   delay benefit + 10-30% duration repair + physical-collapse elimination
   at 2600-2700, with prevention impossible (onsets precede t = 250).
2. **R1 must be redesigned or dropped**: it is unreachable by
   construction while the band is walled (its speed gate includes the band
   it slows). Any speed-gate rule must be evaluated over a region outside
   the walled band (e.g., 400-800 m only), or evaluated after taper.
3. **R2 works and is safe but weak**: fires on 23-47% of seeds, small
   consistent delay gain over the fixed hold, zero discharge cost, but
   near coin-flip per seed. A stronger signal needs the collapse-onset
   timing, not the approach density alone.
4. The 1200 s oracle cap **over-holds at high demand** (delay benefit
   smaller than the 901 cap and discharge loss) — on plateau-decline
   demand there is no regime where holding past t = 901 is the better
   reference.
5. Consistent with Part 1/Part 3: the decision variable that matters is
   **when the collapse onsets relative to the decision step**, which is
   exactly the quantity the v1 gate fit found only weakly predictable
   (L1 ridge Δh R² 0.168, Spearman 0.417; no rung robustly beats L0 at
   λ=2). Stage 1c strengthens the case that the next predictor target
   should be onset-time-relative-to-t_d, not end-state collapse alone.

## Artifacts

- `runs/probe/stage1c_release_m{2200,2400,2600,2700}_p20_s*/`
  (manifest + per-seed JSON: 5 arm summaries, 1 Hz rule inputs, integrity,
  per-seed R2 threshold, D repeats every 10th seed).
- `runs/probe/stage1c_release/summary.json` — all tables above, generated by
  `experiments/idm/analyze_stage1c_release.py` (release timing replayed from
  rule inputs; axes, recovery, safety, flow, divergence distribution).
- Code: `stage1c_release_contract.py`, `run_stage1c_release.py`,
  `policies/stage1c_release.py`, `analyze_stage1c_release.py` (all new; no
  frozen-file edits; full suite 1017 passed before compute).
- Seeds 7401-7520 are registered under `stage1c_release` and excluded from
  prospective evaluation (`idm/contract.py`).
