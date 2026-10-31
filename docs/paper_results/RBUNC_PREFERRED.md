# S-RB-UNC — preferred motion-posterior continuation

Frozen plan: `RBUNC_PREFERRED_DESIGN.md` (`97d5d6c9`).
Existing raw/encoder panels remain preserved. This is a **deterministic legacy
IDM/MOBIL reference diagnostic**, not integrated WM-PHYS qualification.

## Runtime and calibration changes

- Consume FIT's `shrinkage_v1` exact joint finite weights. Its learned
  n/(n+19.35568) population mixing is already applied; no additional inflation,
  clipping, residual noise or parameter-wise independent draws.
- `e4365034` fixes final supplied-posterior selection to use observable VID when
  provided, preserving incumbent draws through packed-slot permutations across
  candidate actions. Invalid/duplicate present identities are rejected. Saved-ID
  and supplied posterior sources cannot silently override one another.
- `0d04093b` adds `with_motion_posteriors`: bind caller-generated FIT
  `condition_motion` finite weights to a fresh observation. Clear all saved-ID
  records and LC blocks, retain the same learned population for entrants, and
  hash the new posterior snapshot. Exact support ordering and causal likelihood
  inputs remain the caller's responsibility.
- Combined42CPUtestsPASS/12.88s. Actual FIT generic inference source and preferred
  artifact/policy compatibility smoke PASS on8 synthetic intervals and zero
  evidence: exact support/weights, explicit population prior, no old identity
  records, unchanged global RNG. Receipt `rbunc/causal_motion_bridge_smoke.json`.

## Preferred-versus-raw physical comparison

New exclusive `cpuQ:/home/ubuntu/rbunc20261031/preferred_v1`; supervisor553091,
four disjoint nice19/one-thread workers. Staged immutable code65f3ddae756c.
Only present-driver posterior changes. All32 nativeFIT states969400–969431,
sameK8/seed20261031/H120 physical horizon and source ablations on969400.
Arrival-fit bytes, Gamma-rate draws, request clocks, fitted newcomer population,
factual targets, encoder and starting observations are checked against the raw
panel rather than assumed unchanged. All checks PASS; all32 worlds complete,
all four worker PIDs553092–553095 exited0, terminal1793468815.2104843.
No UNC-owned scientific job remains running.

### Complete32/K8 result

- Team-delay mean within-K SD **748.35 [668.54,828.51] veh-s**, versus real
  across-world SD **4200.50 veh-s**. Road **383.08 [341.71,435.45] vs1140.37**;
  source waiting **589.76 [516.44,662.72] vs3475.68**. Brackets are95% whole-world
  bootstrap CIs on mean SD, not conditional coverage intervals.
- First-state team SD: both **850.71**, current-drivers-only **164.22**,
  arrivals/newcomers-only **874.61**, both-fixed **0.00 veh-s**. The8 fixed-source
  complete physical tapes hash identically. Request streams match exactly in
  driver-only/fixed ablations; current-driver spread is zero in arrivals-only.
  Nonlinear sources do not have additive variance/SD contributions.
- Team bias **+1142.82 veh-s**, empirical CRPS **980.79 veh-s**, range coverage
  **59.375%**. Raw comparison: bias+1142.11/CRPS980.68/coverage53.125%.
  Paired preferred-minus-raw team CRPS **+0.11 [−24.19,+24.32] veh-s**;
  16/32 worlds have lower CRPS. **No demonstrated team-forecast improvement.**
- Road CRPS **621.37**, paired change **−21.53 [−52.25,+8.14] veh-s**;
  source CRPS **463.56**, change **+4.66 [−4.88,+16.25] veh-s**. These are
  pointwise diagnostic CIs, not simultaneous qualification tests.
- Mean input per-driver SD increases relative to raw: v0 **1.627m/s**,
  headway **.232s**, accel **.258m/s²**, decel **1.035m/s²**, minGap **.545m**.
  More posterior parameter spread did not materially change team-delay spread or
  CRPS. The learned-arrival/newcomer source dominates this first-state diagnostic;
  this is not a globally identified causal variance fraction.
- Request-count mean within-K SD remains exactly **10.51/20.69/34.61 vehicles**
  at120/300/600s, versus Gamma-Poisson predictive **11.26/21.28/36.87**.
  Only request clocks, not traffic dynamics, were extended beyond120s here.
- All280 futures passed finite/conservation/no-slot/no-dropped-request and global
  RNG assertions. Independent pairwise-variance and empirical-CDF CRPS checks
  PASS96metric-world cells. Preferred/raw comparison independently verifies
  exact start/history/encoder receipts, observational identity/target hashes,
  rate draws, request schedules and fitted-population artifact SHA.

Raw bundle `rbunc/preferred_source_probe.json`, SHA256
`0455779281f78586450c31da3b71c43d873f2846b1ce17a47e694c3c15f1fba3`;
all dispatch/PID/terminal lineage is embedded. Summary `rbunc/preferred_summary.json`.
Reproduce without inference (write a NEW output path):

```sh
python -B tools/rbunc/analyze_addendum.py \
  --raw docs/paper_results/rbunc/preferred_source_probe.json \
  --compare-raw docs/paper_results/rbunc/fitted_source_probe.json \
  --out NEW_SUMMARY.json
```

## Integrated EVAL boundary

Latest INT handoff01:33 publishes live FULL-v1 (three owned gpuA workers) with
the preferred posterior and learned arrivals. That immutable runtime is retained.
No complete integrated candidate distribution score was available at this
continuation's freeze. No new spread multiplier is selected from unavailable
EVAL evidence or from the marginal across-world SD, which includes observed
state heterogeneity. Source-attribution checks are not conditional coverage.
The preferred posterior stays preferred on FIT's motion-only evidence; this
reference traffic diagnostic supplies no additional adoption win or rejection of
its integrated WM-PHYS outcome. Earlier encoder-input reference CRPS611.4 remains
substantially lower than either fitted variant, under its separate input contract.

For next INT builds, use commits above and the exact generic-causal bridge when
causal likelihood inputs exist. The bridge does not itself extract new motion or
claim a missing crosswalk. LCv2 stays separately opt-in because FIT's LC
validation still loses its baseline. Full120/300/600s physical qualification and
expected action shifts remain INT/EVAL's matched frozen panel.
