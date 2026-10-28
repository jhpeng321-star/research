# S-GATED300

Worktree `E:/agent-work/c4-gated300`, branch `c4/gated300`, based on
`c4/c4prep` `f5bbeec0`. Owner authority: 2026-10-28 22:05; CCT-1 registration
at 22:10 in the active C4 registration. Default OFF; zero SUMO.

## Frozen contract

Dense trusted credit remains at the existing training horizon, at most 90 s.
Candidate A was described as 10→90 s, but the independent audit establishes
that looplong currently makes medium/high-share starts fixed 90 s from C1
resume; low-share starts retain the existing base/cap behavior. Long credit
is bounded by a WM-specific CCT-1 artifact and 300 s.
Missing authority or H_C ≤90 means exact existing execution. Noise sigma in
veh-s: 90=25.38, 120=41.22, 180=42.11, 300=43.91 (chaotic-sensitivity proxy).
Confident means strictly |predicted dJ| >2 sigma. CCT-1 requires sign ≥0.80,
coverage ≥0.25, and mean absolute real dJ of wrong confident calls ≤2 sigma.
H_C is the longest contiguous passing prefix of 90/120/180/300 s.

Reward is −J, so paired reward is improvement = −(treatment−reference cost).
Each start and logical all/individual/group/coop/team credit row gates on its
K8 median and two-sided pessimistic sign: improvement q10>0; harm q90<0.
Total credit is dense + w_long × gated long, w_long=1.0 fixed prospectively,
never fitted/tuned to outcomes. No critic or value function for long credit.

## Implementation and candidate-A flags

`src/wms/control/gated300.py` contains the new config, authority binding,
gate, shared-prefix extraction, long objective and resume binding. Small hooks
in `dreamer.py`, `policy14.py`, `policy14_cli.py` and `looplong.py` connect it.
Absent `[long_credit]`, or `enabled=false`, follows the old execution path.
Missing `cct1_file` or H_C≤90 does not lengthen any rollout or add any loss.

Append to candidate A's resolved policy TOML **for an authorized new phase**:

```toml
[long_credit]
enabled = true
cct1_file = "/absolute/authority/CCT1_{wm_sha256}.json"
w_long = 1.0
score_training_horizon = true
```

All four defaults are respectively **false, empty path, 1.0, false**.
`{wm_sha256}` expands to the loaded checkpoint SHA256; a literal path also
works. The file must come from the registered **pairs300 verdict panel** and
name that exact WM. H_C and its passing prefix are rechecked against all three
registered inequalities and sigma values. An entrant mismatch, explicit CRN
mismatch, non-K8/non-q10 setup, stale WM identity or informational panel is
refused. There is no inherited WM2/WM3 authority for extending beyond 90 s.
The existing short-horizon authority/acceptance is still required by looplong.

Generate the WM-specific artifact using its existing measured horizon gate:

```text
python -B tools/gated300/cct1.py /absolute/pairs300_gate.json --panel pairs300 --out /absolute/authority/CCT1_<full-WM-sha256>.json
```

Retain candidate A's original C1 `64f70af4`, full residual/group/trunk actor,
q10/K8, max_starts=1, entrant ON, focus .6 and registered credit weights.
The equivalent six-row configuration (w_ind=w_coop=0, w_group=18.8,
cover_group_members=false, w_team=10.3, w_all=1) is supported identically.
No outcome-tuned coefficient or new behavioral threshold is introduced.

### Exact objective and sign/row semantics

For each start and physical paired treatment/native row, let
`I_K = sum_0^H_long reward(treatment) - sum_0^H_long reward(native).detach()`.
This is **improvement = -dJ**, not cost. The median is the interpolated .5
quantile of K8 (mean of the two central samples). The detached mask is ON iff
`abs(median(I_K)) > 2*sigma_H` and either `median>0 AND q10>0` or
`median<0 AND q90<0`. Equality to the noise threshold or zero tail is OFF.
OFF rows get exactly zero long credit/gradient. Native and zero-coefficient
rows are excluded from the gate-share denominator.

The individual/group/coop/team/all linear objective is collapsed to its
existing **effective row coefficients before gating**. Thus the dead individual
8.9−8.9 rows stay dead; group carries 49.7 and each flip carries −10.3 for a
three-member group. Derived coop/team quantities are not gated a second time:
that would break the owner-requested equivalent six-row form. For negative
coefficients the confidence sign still tests the physical treatment−native
dJ, then the signed coefficient enters the joint objective.

Dense credit keeps the existing mean-over-time suffix-return objective and
existing signed-joint q10 selection. The additional term is **t0 accrued
0..H_long credit**, not only the 90..H_long tail. After row gating it reuses
`antiexploit.tail_credit`'s q10 of the complete signed per-start sum (including
covariance and negative team coefficients), averaged over starts:
`total = dense + w_long * gated_long`. Dense and long share the dense return
scale; no long returns alter that scale. There is no critic bootstrap, TD
mixing, terminal value or long critic training.

### Prefix, memory and looplong/resume

Each K future is rolled **once** to H_long; the 90 s (or current shorter dense)
prefix is sliced from the same trajectory and its suffix returns recomputed
to stop at the dense boundary. Long totals come from those same rewards.
No separate 90 s rollout is simulated. Drivers/CRN/entry process continue in
the same future; entrant slots are reserved once for the longer rollout.

Existing LONGMEM supports this via full-K detached forward plus streamed
per-future exact VJP, segment checkpointing and both host offloads. Its backward
replay is autograd recomputation, not an independently drawn dense/long rollout.
The new long tensor is part of the same full-K VJP. S-POLSPEED may batch these
same K futures, retaining one max-horizon trajectory and both reductions; it
does not need two horizon passes. Real 300 s GPU fit/throughput is **unmeasured**
here; no GPU memory feasibility claim is made. Candidate-A settings remain
microbatch1/segment8/both offloads; do not turn off the memory path for 300 s.

`[long_credit]` is independent of `[horizon]`: looplong's auto-horizon rewrite
cannot erase its H_long. Accepted-WM swaps rebind the WM-specific artifact,
and missing authority turns the extension OFF. A mismatched existing artifact
refuses. The dense curriculum and its low-share rules are preserved.
Checkpoint payloads save config, exact WM identities and artifact digest;
resume rechecks the file and restores the binding. Explicit run sections win
over old checkpoint sections, including an explicit OFF. Refreshed WMs need an
explicit new binding. Old C1 checkpoints contain no new fields and remain usable.

### Optional panel/phase-gate/checkpoint-selection horizon (audit F3)

`score_training_horizon=true` makes every `independent_score` call resolve the
same current training curriculum/observed-share horizon, deterministic policy,
and K futures. Looplong configuration is resolved **before** the initial frozen
panel baseline, so phase gates do not accidentally compare 10 s with 90 s.
The option works independently when the long term is OFF.

When long credit is active, the same max-horizon rollout also scores H_long
with the same per-start all-CAV-row gate and q10. `J_dense`, `J_long`,
`native_J_long` retain physical costs; `J = J_dense - w_long * long_gated_gain`
is the explicit combined selection score. Native-relative gain includes that
same long term. The existing all-start native veto and earliest-minimum selector
consume it through their existing interfaces. Native-curve/selection receipts
retain full per-start dense/long scores and actual horizon lists, including
mixed low-share panels. RNG and training/scoring flags restore on exit.
This opt-in is a prospective scoring change, never retroactively applied to
the stopped candidate-A run.

### Per-update diagnostics

- `gated300_rows_on_share`: fraction of effective non-native rows ON.
- `gated300_mean_abs_dJ_long`: mean absolute K-sample dJ on effective rows;
  `gated300_mean_abs_median_dJ_long` also reports the median statistic.
- `gated300_median_pessimistic_sign_agreement`: two-sided sign stability
  (q10 for improvement, q90 for harm); `gated300_median_vs_q10_sign_agreement`
  separately logs the literal median-versus-q10 comparison.
- `gated300_long_credit_share`: absolute weighted selected long-row mass /
  (absolute dense-row mass + absolute long-row mass), zero for zero mass.
- Signed `gated300_dense_credit`, `gated300_long_credit`, H_long and w_long;
  long and dense risk-gradient audit components remain separate.

## CCT-1 calibration receipt (INFORMATIONAL, not the verdict panel)

`tools/gated300/cct1.py INPUT --panel calibration|validation|pairs300 --out FILE`
reads the existing gate's `rows[].dJ[h]` (flat or grouped arms), preserves every
arm/start, excludes real-zero pairs only from sign counts, and uses all pairs
for confidence coverage. No wrong confident signs means mean wrong magnitude 0.
World-cluster 95% percentile intervals use 2000 draws, seed 0. `H_C=0` means
even 90 s failed. The legacy threshold and horizon results are retained verbatim.
Calibration/validation artifacts have `policy_authorized=false`; only the
explicit `pairs300` verdict role can supply trainer authority. Output creation
is exclusive (no input/output overwrite).

One read-only gpuR transaction fetched exactly the three requested selection
files (no WM inference, simulator, or remote writes). Evidence and source-byte
SHA256s: `tools/gated300/evidence/*_source.json` and `*_CCT1.json`.
Each panel has 108 arm pairs / 36 world clusters, sampled K1 as recorded in
the original files. These are calibration diagnostics, NOT a WM adoption verdict.

Numbers below list horizons in order **90 / 120 / 180 / 300 s**:

- **MACRO selection 0000**, WM `b8797242`: confident sign agreement
  **.6500 / .6522 / .8333 / .8235**; coverage
  **.3704 / .4259 / .5000 / .4722**; wrong-call mean |real dJ| veh-s
  **29.00 / 26.02 / 165.40 / 414.47**. All horizons FAIL; informational H_C=0.
- **MACRO selection 0030**, WM `030c5870`: confident sign agreement
  **.6829 / .6667 / .7966 / .8596**; coverage
  **.3796 / .4444 / .5463 / .5278**; wrong-call mean |real dJ| veh-s
  **24.20 / 29.62 / 96.97 / 254.85**. All horizons FAIL; informational H_C=0.
- **W1_PAIRED selection 0010**, WM `8a04b7d7`: confident sign agreement
  **.7143 / .7083 / .7778 / .8182**; coverage
  **.3889 / .4444 / .5000 / .5093**; wrong-call mean |real dJ| veh-s
  **28.44 / 24.66 / 97.09 / 362.26**. All horizons FAIL; informational H_C=0.

All coverages clear .25; the 90/120 s accuracies fail .80. At 300 s the
accuracies clear .80 but wrong-call mean magnitudes exceed 87.82 veh-s.
Full unrounded values, intervals and unchanged all-pairs numbers are in JSON.
Six CCT-1 CPU tests passed on Python 3.13 / torch CPU before this step's commit.

## CPU verification and delivery boundary

Environment: Python 3.13.5, torch 2.14.1+cpu, CUDA hidden, OMP/MKL threads 1,
UTF-8 mode, repository root plus `src` on PYTHONPATH. Temporary test outputs
were confined to `E:/Temp/opencode/gated300_*`.

- **21/21 new CPU tests PASS** (`test_gated300.py` + `test_gated300_cct1.py`,
  final complete run 37.89 s). Covers strict thresholds, improvement/harm signs,
  sign-instability OFF, only-ON long gradients, explicit OFF/missing/H_C=0/90
  bitwise loss/gradient/Adam/updated-weight/RNG equality, binding refusals,
  actual 90 s prefixes of K8 120/300 s forward trajectories with exactly one
  rollout per future, equivalent nine/six-row loss/gradients, full-K LONGMEM
  VJP/update parity, uninterrupted-vs-resumed exact updates/Adam, changed-file
  resume refusal, explicit OFF precedence, optional matched scoring/RNG/flags,
  and long config surviving the actual auto-horizon `looplong.configure`.
- Production full residual actor, all five trainable modules, real WM code,
  K8, sampled lanes, entrant ON and streamed/checkpointed/offloaded backward:
  a synthetic CPU update PASS with nonzero long credit and all module gradient
  norms positive. Exhaustive ON replay/resume and this production ON update
  use compressed synthetic time (4 physical control steps labelled 120 s) and
  scaled synthetic rewards to exercise the gate; they are not 300 s traffic
  training evidence. The separate max-horizon forward tests use the real
  180/240/600 physics-step geometry for 90/120/300 s.
- Existing **LONGMEM: 25 PASS / 2 CUDA-only SKIP**, including production
  full-actor K8/entrant gradient/update/Adam/RNG parity (169.15 s CPU).
- Existing anti-exploit, credit-floor, looplong, coopw and recovery suites:
  **103 PASS / 2 SKIP**. Native-init: **21 PASS**, including three-update
  bitwise trainer/optimizer/all-tensor parity against unmodified c4/c4prep
  source, with `NATIVEINIT_BASELINE=E:/agent-work/c4-c4prep`.
- Existing launch/preparation/policy14 integration suites: **39 PASS**.
- Two existing native-init mock smoke-barrier tests fail on Windows because
  `str(Path(rel).parent)` produces backslashes while HASHES keys use slashes.
  **Both reproduce on unmodified c4/c4prep f5bbeec0**; no simulator is invoked.
  Initial missing-baseline/sparse-helper setup failures were resolved for the
  other native tests. The first exhaustive 120 s synthetic backward attempt
  exceeded a 120 s local tool timeout; no process remained. It was replaced
  by the short exhaustive VJP/resume fixture plus actual max-horizon forwards.

Reproduce the new tests (set PYTHONPATH/C4_ALLOWED_ROOTS for your environment):

```text
python -B -m pytest -q -o addopts= -p no:cacheprovider --basetemp E:/Temp/opencode/gated300_new tests/unit/test_gated300.py tests/unit/test_gated300_cct1.py
```

This delivery is CPU-qualified engineering code. No GPU smoke or policy phase
was launched, no SUMO was run, and no shared process/lock/budget was modified.
The registration's independent deployment/execution-parity blocker for Thursday's
candidate-A restart is not cleared by adding confidence-gated training credit.
