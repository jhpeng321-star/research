# S-ESLONG — owner-approved CPU build, 2026-10-30
Final tested code/config/test SHA: `a94549ee93ba8052d4a53d998d19766589e76baa`.
Delivery SHA: `git log -1 --format=%H -- docs/paper_results/ESLONG.md` (no self-hash).
Worktree `E:/agent-work/c4-eslong`, branch `c4/eslong`, base `9613f956`.

## Estimator and confidence
- Default `long_credit.estimator="pathwise"` preserves legacy arithmetic/RNG/Adam.
- Opt-in `antithetic`: perturb ONLY residual output weight+bias (137 scalars for
  hidden_dim=64); dense 60/90s remains the registered pathwise prefix objective.
- Draw M=2 independent Gaussian directions with a private update-index generator.
  `g_ES=mean_m[(L_plus-L_minus)*eps/(2*sigma)]`; L is loss (negative credit/scale).
  Starts, row geometry, K8 WM/driver/lane tapes, entrant seeds/identities and
  exploration draws are shared exactly, including across signs/directions.
- Each side retains native-relative gain-only row gates and signed JOINT q10.
  A row-pair additionally requires |median(plus-minus)| > 87.82 veh-s and
  q10(sign(median)*(plus-minus)) > 0; finite samples and nonzero row coefficient.
  Same registered sigma_300=43.91, strict 2-sigma margin; no behavioural cutoff.
  Pair masks apply symmetrically before each side's joint-q10 reduction.
  Negative orientation uses q90(diff)<0, equivalently q10(-diff)>0.
  Rejected directions contribute zero and remain in denominator M.
- Existing `long_grad_balance=true` caps ||g_ES|| at ||g_dense|| before addition;
  zero dense suppresses ES, nonfinite gradients fail before Adam, then clip=100.
  No long critic/value function. No backward through a full 300s rollout:
  full forwards use no_grad, long outputs are nondifferentiable, dense replay
  executes 61/91s including the exact next-input boundary; oversized segments fail.

## Defaults and scale
- `es_sigma=0` means sigma=exploration_std / RMS(local command head Jacobian).
  t0 frozen all-CAV features give `radius*sech²(z)*sqrt(||features||²+1)`;
  thus expected local pre-cap command RMS equals the scheduled additive std.
  Report sigma, head RMS, sigma/max(head RMS,1/sqrt(d)), predicted command RMS,
  actual t0 executed antithetic half-difference RMS, accepted/rejected row-pairs,
  accepted directions, raw ES norm, dense norm, cosine, balance and forward count.
  This is command-scale calibration: execution caps may suppress the measured
  executed response; no inflation/search to cross a hand-set action threshold.
  Inert t0/no additive std requires an explicit positive sigma (no silent floor).
- S-CAPREL may provide `policy.eslong_decode_slope(latent)` for its decode law.
  S-GATEMEAS may bind `trainer._long_credit_reference(trainer,imagined,totals,base)`
  returning paired reference totals (same geometry, detached here); copied to
  workers. Default remains registered native. Rebind that hook on resume.
- Saved gated300 config + update_count/RNG restore sigma policy/directions/Adam;
  historical receipts normalize missing estimator fields to pathwise defaults.

## Exact config and gpuA-alone cost projection
`configs/c4_policy14_candA_es.toml` extends `c4_policy14_candA_longgrad_skip.toml`:
```toml
[long_credit]
estimator = "antithetic"
es_directions = 2
es_sigma = 0.0
```
Inherited balance/skips ON, trained head preserved, explicit new-phase Adam-reset
flag retained (ordinary resume preserves Adam). Config depth cap 16→32 permits
the requested direct inheritance. Regenerate authority/source manifests for entry.
CUDA perturbations always use the existing forward pool (even with one worker).
For gpuA alone: `WMS_SPEED_PARALLEL_FORWARD=1`, `WMS_SPEED_FORWARD_WORKERS=2`,
`WMS_SPEED_FORWARD_MEMORY_FRACTION=.24`, `WMS_SPEED_FUTURE_WORKERS=1`,
`WMS_SPEED_FUTURE_BUDGET_CAP=.90`; retain segment=2 and mode-2 reward recipe.
Use a fresh output dir and explicit long_credit binding; estimator keys live in
gated300 metadata, outside trainer resume-allow. Do not relabel an old score gate.
K8 count=(1+2M)*8=40 full futures/update (32 perturbations); dense VJPs extra.
Measured input: 861s / eight serial 300s forwards ≈107.625s/future, ~10GiB/worker.
Serial projection: 4305s forward + 8*357*(91/300)=5171s/update (86.2min).
Two-worker ideal: 2152.5s forward +866s dense ≈3019s (50.3min/update);
150 updates ≈125.8h, vs serial ≈215.5h. Concurrency is NOT a timing guarantee.
Two forward workers ≈20GiB plus coordinator ≤~12GiB + contexts (~2.25GiB),
~34GiB on 47GB gpuA; use measured reservation/budget guard, not this projection.
Full unperturbed caches are NOT transported to ES workers, only CRN boundaries.

## Verification and risks
125 distinct CPU cases passed: ESLONG 15, LONGGRAD 29, gate/skip/resume 56,
LONGMEM 25; Ruff passed. Two CUDA and three older heavyweight WM cases excluded.
CPU synthetic tests: quadratic unbiased ungated estimator; confidence bias is
intentional, so gated ES is NOT claimed unbiased. CRN, sign symmetry, default-OFF
bitwise updates, resume, actual worker transport and real-WM entrants are tested.
Risks: M=2 high variance, finite-sigma smoothing, gate starvation, shared WM bias,
cap-suppressed actions, head-only search, expensive forwards and unqualified GPU
concurrency/memory. No GPU/SSH/SUMO or production-result claim from this build.
