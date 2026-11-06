# POLICY14 registration draft — 2026-10-26

**Status: PREPARED / NOT LAUNCHED.** Proposed registration for the main integrator.
This document records a bounded policy-training plan; it supplies no qualification
verdict. A preparation receipt is appended to the external registry; the launcher
consumes a separately supplied WM qualification before training.

## Scope and build contract

- Stage the reviewed source tree at `/root/autodl-tmp/c4/v5/policy14_stage`.
  The source includes `configs/c4_policy14.toml` and `wms.control.policy14_cli`.
- Preparation includes a local CPU smoke. GPU training and SUMO collection are
  deferred to the registered run. Each actor starts from scratch in a new output.
- Earliest execution: **2026-10-26, Asia/Shanghai**. The launcher checks the date
  for both training and smoke, with no date override or automatic scheduled start.
  A dry run can be inspected before that date.
- CLI contract used verbatim:

  ```text
  python -m wms.control.policy14_cli --config CONFIG \
    --wm-ckpt CKPT [CKPT ...] --wm-config CONFIG [CONFIG ...] \
    --real-returns PATH --device cpu|cuda --seed SEED \
    --max-updates N --output-dir NEW_DIR [--smoke]
  ```

  Checkpoint/config lists are paired in order. At least **one** qualified frozen
  member is permitted. No implicit checkpoint, actor initialization, imitation,
  native residual prior or resume is allowed. Actor and group communication
  parameters start from scratch for each seed. A pretrained WM and critic fitting
  on approved real returns do not constitute pretrained actor weights.

## Resources, seeds, budget and outputs

- Hardware envelope: **one 24 GB GPU, 25 allocated CPUs**. Only one trainer runs
  at a time. Batch **1 or 2**, imagination horizon **10 control seconds**,
  `trainer.checkpoint_imagination = true`, `counterfactual.max_starts = 1`.
  Keep `trainer.credit_profile_every = 0`: historical lowmem configurations still
  allowed three starts, and credit profiling caused memory spikes/OOMs.
- Resolved `[run].threads` must be an integer in `1..THREADS`, with `THREADS` in
  `1..8` (default 8). OMP/MKL/OpenBLAS/NumExpr/vecLib thread caps match `THREADS`.
  The main CLI must honor the resolved torch thread cap; allocated 25 CPUs is
  not permission to set numerical-library threads to 25.
- Three **sequential** from-scratch optimization seeds, fixed order:
  **965140, 965141, 965142**. These are training RNG seeds, not new traffic-world
  identifiers or permission for simulation. Main must confirm their registration.
- Production ceiling: **600 actor updates per seed, 1,800 total**, one attempt
  per seed, no automatic retries, extensions or resume. `--max-updates` may only
  reduce this ceiling. A failure exits the launcher before the next seed; retain
  the partial artifacts and account for consumed work before any amendment.
- Each seed also uses **32 critic-only warm-start updates** (96 total) and a
  **12-update disposable free-road actor** readiness probe (36 total). These do
  not initialize the campaign actor. No external simulation is involved.
- At actor update **10**, the exact paired, independently rescored unshaped J
  must decrease from its initial value on fixed development starts, or the CLI
  saves its own checkpoint and exits before scaling. Passing that tiny local gate
  is not a native win, safety certificate, or authorization for SUMO.
- Disposable CPU smoke: at most **2 updates per seed, 6 total**, separate outputs,
  `--smoke` and `--device cpu`; reductions permitted. Main must record any CLI
  smoke-only horizon/batch overrides. Passing smoke demonstrates integration and
  finite gradients only: **no scientific efficacy, cooperation, transfer,
  calibration or safety claims follow from smoke**.
- Default production output root:
  `/root/autodl-tmp/c4/v5/runs/c4_policy14_20261026`.
  The launcher appends `_smoke` to the chosen root for smoke. Each campaign root
  must be new and its parent must exist; outputs are `seed_965140/`,
  `seed_965141/`, `seed_965142/`, plus per-seed launch/log files and a shared
  `INPUT_AUDIT.json` / copy of the supplied qualification marker. Choose a new
  explicit `--output-root` for an amended attempt.
- This launcher consumes **zero new SUMO episodes / simulation-seconds**. Charge
  existing WM data, real-return targets, current-policy refreshes, failed work,
  checkpoint screening and calibration data in the eventual cumulative ledger.
  Record elapsed time, peak VRAM/RAM, updates and artifact sizes; no GPU-hour or
  disk-throughput estimate is claimed without measurement. Resource failure is a
  stop for review, not permission to change another job or increase this budget.

## Explicit, user-supplied WM qualification evidence

`--qualification FILE` is mandatory, including for smoke and command inspection.
Dry-run mode prints the plan without opening/validating inputs or creating output;
it does **not** certify launch readiness. A real invocation fails closed unless
the user supplies a nonempty JSON marker referencing a separate nonempty evidence
report, with matching SHA-256 hashes for that report and every supplied WM
checkpoint/config pair. A marker has this structure (placeholders, not a verdict):

```json
{
  "status": "qualified",
  "qualified_by": "<responsible reviewer and review date>",
  "evidence_file": "<approved qualification report path>",
  "evidence_sha256": "<64 lowercase hex characters>",
  "members": [
    {
      "checkpoint_sha256": "<64 lowercase hex characters>",
      "wm_config_sha256": "<64 lowercase hex characters>"
    }
  ]
}
```

`evidence_file` may be absolute or relative to the marker's directory. The
`members` list must exactly cover CLI members in order; add one entry per
`--wm-member CKPT CONFIG`. Duplicate checkpoint paths or duplicate checkpoint
bytes are refused. Each checkpoint must also have nonempty adjacent `physics.json`
and `TRAIN_RECEIPT.json`; their hashes and all config inheritance hashes are
recorded in `INPUT_AUDIT.json`. The preflight uses Python 3.11+ standard libraries
only; it does not deserialize checkpoints or load training datasets.

**The marker is a reviewer's assertion backed by an evidence report, not an
automatically computed scientific pass.** File existence, an attractive filename,
training completion, NLL, a smoke verdict, head-zero initialization or an older
round's reputation cannot create qualification. Changing WM weights, physics,
inherited configs, observation/command paths or relevant rollout code requires
review and renewed evidence. Pin the source commit and inherited-config hashes
in the report; review them against the staged build before issuing the marker.

The report must cover current-code accounting/queue and command parity,
closed-loop action-conditioned cooperative/harmful intervention ranking beyond
nulls, near-field HDV follower response, pre-/post-merge recovery, source
discharge and safety at relevant horizons. Register tolerances on permitted
development data before reading outcomes, with episode/world-clustered intervals.
No automatic qualification probe, GPU smoke or SUMO job is launched by this script.

## Data, objective and credit contract

- Only explicitly approved **TRAIN-corpus** start windows and real-return files.
  Validate provenance, traffic seeds, split roles and source manifests before
  supplying them. **Evaluation seeds 964000–964199 are excluded from all training,
  critic fitting, tuning, qualification and smoke inputs.** No restricted,
  sealed, final-test, CAL/HOLDOUT or other prohibited data/path access is part of
  this plan. The launcher does not discover datasets or inspect such registries.
- Allowed WM/critic experience is native driving, undirected random direct-output
  actions and approved learned-policy trajectories. Designed-controller windows,
  demonstrations and controller-selection targets are excluded. Native is a
  comparison/reference, not an imitation target. Do not silently reuse the
  historical switching/direct-mix source; follow `DATAHYG_NOTES.md` provenance.
- Authoritative cost **J** is total on-road delay plus waiting at **every source**,
  in vehicle-seconds, over the **same fixed request population** and request
  schedule in every alternative. Include initially queued, requested-but-not-
  inserted and later-arriving vehicles. Report `J / N_requested` using a fixed,
  policy-independent denominator; never normalize by admitted/completed traffic.
  Entries and initial source backlog must be represented, with request/insert/
  exit conservation and no queue truncation, free blocked entries or fleet mismatch.
- Register one finite episode horizon (future probes: 300–600 s) and its terminal
  backlog rule before scoring. Report accrued J and remaining source/on-road
  backlog separately; use an audited policy-consistent tail where estimating
  beyond the 10 s imagination horizon. Do not erase unserved demand at truncation
  or hide a flat/zero tail behind a short imagined gain. Use near-undiscounted
  finite-horizon targets or explicitly justify discounting against that endpoint.
- Fit the critic to approved real-return targets with the same J, horizon,
  denominator and backlog convention. Native/random-action returns may warm-start
  an action-conditioned critic/representation, but are not unbiased current-policy
  values. Check independent bootstrap bias and refresh current-policy tails only
  in separately registered future loops. A bootstrap is an estimate, not observed J.
- Prepared `policy14_native_returns.npz` contains 12 existing TRAIN native-world
  600 s targets, seeds 969824–969835, from the permitted diagnostic source. Values
  come from each recorded `summary.json`'s exact window source-wait and speed-
  deficit integrals; no source queue is reconstructed from endpoints. Numeric
  state/backlog arrays provide the initial critic state. This p40/2400 corpus is
  a small off-policy warm-start, not an adequate current-policy/generalisation
  dataset. The exporter and per-summary hashes are supplied for reproducibility.
- The strict NPZ provider accepts only numeric/string arrays (`allow_pickle=False`),
  TRAIN provenance, declared exogenous requested population, 300–600 s horizon,
  the exact source-inclusive accounting convention, and permitted source kinds.
  `write_real_returns_npz` can instead include audited interval ledgers. The critic
  output and loss scales are 10,000 vehicle-seconds; predictions remain raw -J.
- Positive gain is `D(A) = J(native) - J(A)`. Begin with team gain plus sampled
  per-CAV marginal credit `M_i = J(G without i) - J(G)`, pairing world/event/vehicle
  randomness and holding other actions fixed. Stop gradients through reference
  baselines; do not claim that subtracting a detached baseline alone creates new
  pathwise marginal derivatives. Keep synergy diagnostic initially; if later
  added, retain positive individual/team terms and `w_coop <= w_ind`.
- Normalize with TRAIN return statistics shared across penetrations. Log reward,
  constraint, entropy and critic gradient scales, saturation and acceleration
  support. Verify acceleration learning on a simple free-road diagnostic and
  finite-difference/score-function credit where projection/lane/entry discontinuities
  defeat pathwise gradients. Diagnostic success is not traffic efficacy.
- Acceleration receives team pathwise derivatives. Sampled leave-one-out credit
  weights the removed CAV's categorical lane score-function term, averaged across
  sampled marginals. It is not mislabeled as extra continuous pathwise credit.
- Throughput, mainline/ramp and worst-group delay are separate diagnostics or
  constraints; avoid double-counting throughput in primary J. Any bounded potential
  shaping requires consistent terminal correction and unshaped independent scoring.

## Comparison, observation and execution parity

Primary scientific comparators are **native** and a **universal MAPPO** trained
across the same demand/penetration distribution, from scratch, at equal total
SUMO-data budgets. Give MAPPO identical permitted sensors, report histories,
message ages/missingness, actor capacity/communication, direct commands, safety
projection, lane semantics/dwell, action holds, caps and reward accounting.
Both may use privileged training-only critics; deployment actors may not use
hidden driver types, future merges, true penetration or privileged queues.
Reconcile decoded versus executable acceleration caps (nominal direct range
`[-3, 1.5] m/s²` and any population-specific execution cap) explicitly; never
compare policies with different effective caps. Use 1 s commands / 0.5 s physics.

Train on 5/10/20/40% CAVs and the approved varied demands; later generalization
checks at 15/30/60% require separately registered fresh development worlds.
Share target stratum weights across methods. Report every stratum and fixed-weight
aggregate with paired episode/world-clustered uncertainty, not best-seed results.
Account for WM pretraining and refresh/critic/calibration experience when matching
MAPPO's cumulative SUMO budget; shared historical data still needs disclosure.

Keep actor-only, belief-verified actor, sensed-only and labelled oracle-state
diagnostics distinct. Belief must model unobserved occupancy/arrivals with
uncertainty rather than treating unseen vehicles as empty road. Optional
`gs_mask` support may use the **existing mask fallback** when unavailable; record
the selected path and parity with MAPPO. This is an optional compatibility path,
not grounds to invent a replacement lane/masking mechanism or claim the optional
feature was tested. A lane-disabled pilot remains an explicitly labelled ablation.

## Promotion gates: imagination, then future SUMO, then deployment

1. **Before bounded scientific training:** human-reviewed, hash-pinned WM evidence
   as above; accounting/command parity; approved real-return targets; usable actor
   gradients and no worsened local safety. A smoke cannot satisfy scientific gates.
2. **After all three training seeds:** independently re-score a predeclared maximum
   of **two checkpoints per seed** (six candidates) on disjoint TRAIN-derived
   development worlds/noise, without training updates, with exact unshaped J and
   paired native/reference rollouts. Report all seeds and candidate checkpoints,
   mean/downside gains and clustered intervals. Require reproducible multi-seed
   native-relative imagined improvement and safety/fairness non-regression under
   prospectively registered tolerances. A training loss or optimized-on trajectory
   is not independent re-scoring. If imagination fails, repair credit/objective;
   do not increase this run's budget. This launcher trains only; it does not run
   the independent scorer or promote its own artifacts.
3. **Only after that gate and a new authorization:** future paired SUMO transfer
   pilot, initially **24 policy/native pairs of 600 s per checkpoint**. This is
   48 episodes and **28,800 controlled simulation-seconds per checkpoint**, plus
   all warmup/replay. At two checkpoints for each of three seeds, the maximum
   proposal is **144 pairs / 288 episodes / 172,800 controlled seconds**, before
   warmup/replay. Native reuse and universal-MAPPO arms must be itemized, never
   assumed free or silently included. Register fresh development world seeds,
   stratum allocation, tolerances, accounting and comparable MAPPO budget before
   execution. **No SUMO episode is authorized or run now.** If WM wins but SUMO
   loses, diagnose action-conditioned coverage/tail error before a separately
   budgeted refresh; do not call imagined gain a transfer result.
4. **Safety/fairness at every gate:** collisions, teleports, TTC exposure,
   hard-braking vehicle-seconds (`a < -3 m/s²`) and worst-group delay. Explicitly
   report **HDV followers of controlled CAVs**, pre-merge yielding and post-cut-in
   braking/recovery alongside all-road HDV and CAV safety. A CAV-follower-only TTC
   training constraint does not audit HDV follower harm. A capped/adaptive dual or
   projection never turns a violated safety gate into a pass.
5. **Separate deployment gate:** execute a learned direct-output plan only when
   calibrated **P(J_plan < J_safety-projected-hold-speed | belief) >= 0.8**.
   Otherwise abstain to that direct hold-speed fallback. Native remains the
   scientific benchmark; if an owner requires native as the probability reference,
   require that additional test explicitly. Only actor plans, their learned group
   variants and samples are candidates, not a MAPPO/PI/wall controller menu. An
   alternative replacing the actor plan must also beat that plan under the rule.
   Fit/test the complete candidate-search and abstention procedure on disjoint,
   separately authorized development SUMO episodes, including terminal-value error.
   Report reliability near 0.8, Brier score, precision, coverage and uncertainty on
   accepted-decision win rate by stratum; abstain with insufficient support. A 4/5
   vote is not calibration, and the threshold may not be lowered to gain coverage.

One WM member is allowed for bounded learning, but **epistemic uncertainty is
unavailable with one member**, not proven zero. Driver/arrival sampling is outcome
variation, not model disagreement. More members should be independently trained
bootstrap models, not copied checkpoints; report stochastic versus ensemble
variation separately. Neither an ensemble nor a supplied qualification marker
establishes the calibrated deployment claim.

## Exact launcher commands

The following uses explicit operator-supplied paths. `WM_CKPT`, `WM_CONFIG`,
`REAL_RETURNS` and `QUALIFICATION` must already name approved artifacts; the
launcher never manufactures qualification or discovers substitute data.

```bash
bash /root/autodl-tmp/c4/v5/policy14_stage/scripts/policy14_launch.sh \
  "$WM_CKPT" "$WM_CONFIG" "$REAL_RETURNS" \
  --qualification "$QUALIFICATION" \
  --device cuda --max-updates 600 \
  --output-root /root/autodl-tmp/c4/v5/runs/c4_policy14_20261026 \
  --dry-run
```

On/after Oct 26, after registration and qualification review, remove only
`--dry-run` to execute the three foreground sequential training runs. For a
disposable CPU integration campaign, use `--smoke --device cpu --max-updates 2`
and its own output root (the script appends `_smoke`). Add each further qualified
member with `--wm-member "$WM_CKPT_2" "$WM_CONFIG_2"`. `PY` selects Python 3.11+
with the main implementation's dependencies; `THREADS=8` is the default.
The main CLI retains responsibility for TRAIN provenance/split/path checks and
writing per-run resolved configs, metrics and checkpoints. No arbitrary extra CLI
arguments are forwarded by this launcher.

## Guidance used and interpretation

Read `E:/agent-work/astra-insight/POLICY_ADVICE.md` in full (2026-10-24), especially
its corrected fleet-matched diagnosis, short-imagination/real-return-tail advice,
multi-seed independent scoring, full-system safety and separate qualification /
deployment gates. Also consulted current `c4_policy_v6_lowmem.toml`,
`c4_policy_v6_loop_lowmem.toml` and their policy inheritance, v8/v9entry queue
settings, `docs/WM_C4_DESIGN.md`, `DATAHYG_NOTES.md`, `SRCWAIT_NOTES.md`,
`SRCGRAD_NOTES.md`, `MAPPO_NOTES.md`, `POPULATION_NOTES.md` and launch conventions.
Older design language about automatic penetration generalization, empty unseen
space or rewarding throughput does not override the current advice/contracts.
Historical evaluation plans do not authorize reuse of evaluation seeds here.
