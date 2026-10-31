# S-RB-UNC FIT source-attribution diagnostic

Owner panel: **969400–969431 inclusive**, native observations at t=300 s.
No driver labels, assignments, reserved-world paths, SUMO or training.
All 32 worlds are included without outcome selection.

## Frozen implementation and sampling

- Usable adapter increment: `1f918969`; follow-up uses the physical encoder's
  own parameter support (rather than legacy support for its first five values).
- cpuQ immutable code snapshot `rbunc20261031/code/adc30cdb37b1` canary;
  panel snapshot `70c0be77939f`. Sampler and probe bytes are identical between
  snapshots; the latter adds the owned panel supervisor.
- Frozen existing motion-encoder checkpoint:
  `/home/ubuntu/e3final/inputs/wm_s_best.pt`, SHA256
  `5b2569418bf651b8ef2c3a2abdc110ac6860c32dd92dca8cdf3b2f48b1c8cbb2`.
  This is the available S-SUMOCORE encoder, **not** the differently serialized
  WM2 `b8797242` binding of the integrator's main efficacy panel.
- Explicit inputs per world:
  `/home/ubuntu/wmdata20261031/jobs/s{seed}_native/{start.pt,start_history.npz}`.
  History allowlist is kin/mask/target_mask, ending at t0. Genuine last60s
  features use the frozen encoder's input contract. Audit hook denies prohibited
  path tokens for all indirect file opens too.
- Inferred HDV/CAV population pools frozen from current, genuinely observed
  drivers before padding. Current CAV settings stay fixed. Arrival CAV settings
  use the observed CAV inferred pool, HDVs the observed HDV inferred pool.
- Poisson demand MLE = (observed insertions since0 + observed t0 backlog)/300,
  per entrance. Observed CAV share and mainline entry lane share from pre-t0
  insertions only. No future rate/count realization is supplied.
- K8, private seed20261031, equal world identity across alternate source modes.
- All32: both sources sampled. First world969400 additionally: drivers-only,
  arrivals/newcomers-only, and both-fixed. This world is chosen by numeric order,
  not observed behavior. Fixed futures must have identical physical tape hashes.
- Driver parameter sample SD and request-count K8 SD at120/300/600s are saved.

## Physics attribution reference

While CF/integrated WM-PHYS is pending, this engineering probe uses deterministic
**legacy IDM/MOBIL**, all learned motion/lane/merge corrections absent, source
encoder frozen and copied by reference. Every sampled driver's settings remain
fixed until exit; newborn admission and motion use the same ordinal-bound draw.
Known stochastic modules fail closed, and the global RNG must remain bit-exact
across every physical rollout. Each future is a fresh causal rollout, no future
presence/junction/trajectory teacher and no re-anchoring.

Horizon120s; 128 padding slots; a no-slot refusal/dropped request invalidates the
probe rather than silently changing demand. Queue conservation is checked exactly.
Native CAVs use an explicitly deterministic IDM callback, retaining their current
lane. This is **not the main native SUMO execution-parity controller contract**.
It tests uncertainty source plumbing, not LC/CF parity or WM-PHYS efficacy.

Physical delay diagnostic = post-step on-road `0.5*max(0,1-v/31.29)` plus the
pre-step source rectangle. It is deliberately identified separately from the
registered main EVAL contract. Only after all sampling/rollout does the probe
open the factual `cumulative_120.json` for a descriptive across-world comparison.
No result calibrates the sampling scale or changes this frozen code.

## Dispatch and scope

- Local GPU unavailable; local CPU tests use one thread. cpuQ initially
  32logicalCPUs/load2.07 (inspection), load3.89 at panel preparation. One canary
  + at most8 single-thread workers, nice19, existing CPU conda interpreter.
- Own canary PID211301; supervisor PID215439; per-child own PID receipts under
  `/home/ubuntu/rbunc20261031/fit_probe_v1/`. Existing canary never duplicated.
  Queue new work only below load24. No processes signalled.
- Raw outputs: `w{seed}_h120.json`; all attempts/logs retained. No retry/refund.
- Report mean/RMS within-K SD versus real across-world SD, source-mode SD and
  frozen-mode tape equality. Those SDs answer different conditioning questions;
  they must not be tuned to equality. This is not a coverage qualification panel.
- Integrated WM-PHYS A must use the same adapter in S-RB-INT/S-RB-EVAL for its
  separately frozen120/300/600s efficacy analysis.

## Mechanical likelihood correction / retained v1

All32 v1 worlds completed. Its causal likelihood diagnostics exposed raw-IDM
residual MSE up to3.18e10 at compressed/censored gaps. The residual compared
executed observed acceleration against an unbounded internal IDM quantity.
Correct that mismatch using the model's **existing** plant acceleration bounds;
do not choose a new threshold or use factual delay to tune any scale. The source
attribution reference is otherwise unchanged. Regression verifies compressed-gap
raw acceleration below-1000 becomes the configured plant bound, with the expected
finite observation residual MSE.

- v2 immutable source: `/home/ubuntu/rbunc20261031/code/e448360130cc`.
- Separate output directory `fit_probe_v2`, own canary252452/supervisor252519.
- Same32 identities, K8, H120 and source ablations; all v1 attempts retained.
- v1 raw output and initial analysis are preserved under `docs/paper_results/rbunc`.
  A counting-label audit identified recorder `arrivals` as **destination exits**
  (`env.py:678`, `branching.py:374`), not entrance requests. Request-spread tables
  must compare with Poisson theory only; any recorded exit SD is separately named
  and explicitly incomparable. Team/source/road delay comparisons are unaffected.
