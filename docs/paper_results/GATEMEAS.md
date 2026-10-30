# S-GATEMEAS — owner-approved build, 2026-10-30 14:00
Final code/test SHA: `bb7a245af6982843e9876651d2da52cba4b7e0dd` (base `9613f956`).
Final delivery SHA: `git log -1 --format=%H -- docs/paper_results/GATEMEAS.md`.
Status: CPU build/tests complete; no SSH, GPU or SUMO execution; interim charged 0 sim-s.

## Four opt-in switches (all default OFF)
- `[long_credit] ctrl0_reference=false`: ON adds one paired CTRL0 row per start.
  Same mode-2 always-commanded executor, native lanes, CRN driver/lane/entry tapes.
  Non-intervened CAVs/entrants on treatment rows use residual zero under that law;
  explicit learned-native base rows remain the dense-credit baseline. Thus the ON
  option also changes the shared dense-prefix treatment counterfactual executions.
  Long reference is detached CTRL0, accrued t0..300 s, with no critic/value calls.
  Gain-only gate unchanged: median >87.82 veh-s (2*43.91), q10>0; decisions detached.
  Identical contiguous row reductions prevent alignment-only fictitious gains.
- `[long_credit] score_training_horizon=false`: port of DEADZONE2's opt-in preset,
  rather than merging its diagnosis/replay changes. ON rescores resolved training
  dense 60/90 s (or actual low-share base horizon) and qualified 300 s trajectories.
  Freeze curriculum position at the phase start; use a NEW baseline for each new
  phase/horizon regime. Training still advances normally. Do not compare with old
  10 s J receipts; start a new measurement phase when stepping from 60 to 90 s.
  `frozen_scoring_baseline.json` binds WM IDs, long config, panel SHA, raw and
  composite scores. Same-phase continuation retains it; WM swaps record a new one.
  `rescores.jsonl` and phase-panel results separate `J_raw_dense`, `CTRL0_J_dense`,
  `raw_dJ_vs_CTRL0_dense`, and corresponding `_long` fields from
  `composite_gated_score = J_dense - w_long*q10(gated long improvement)`.
  Raw dJ is ours-minus-CTRL0 (lower better); raw_gain uses the opposite sign.
- `[counterfactual] headroom_starts=false`: existing `start_focus` already supplies
  a 60% label-only demand/penetration focus-cell PROPOSAL, not queue targeting.
  ON adds weight 1+headroom in [1,2] using initial RSU speed deficit in cells near/
  reaching the merge, existing ramp queue observations/backlog, and a CAV group.
  No new behavioural thresholds, policy features, oracle outcomes or eval inputs.
  The final cell mass can change after the tilt. VAL remains uniform; sampler
  telemetry includes pressure, proposal-weight estimates, ESS and fallback bound.
- `[long_credit] k16=false`: ON uses 16 futures at qualified H_long=300; the same
  trajectories provide K16 dense prefixes. q10 interpolation uses ranks 1/2 at .5.
  Same-row forward/cache cost doubles; sigma and shared WM bias are unchanged.
  Example nine native rows -> ten CTRL0 rows: 48,000 row-s, factor 20/9=2.22 vs K8;
  live graph per replay scales by 10/9, not 2.22; zero-VJP skips can save replay.
  Arithmetic estimate only; no measured runtime. Fitted K8 anchor/ensemble off.

## Preset and S-ESLONG hook
`configs/c4_policy14_candA_gatemeas.toml` opts into the first three; K16 stays OFF.
It preserves longgrad/longskip, existing learned driver/execution settings and the
actor head. Use a newly labelled phase after the owned production gate/STOP;
rebuild the existing hash-bound runtime authority for this config/source tree.
S-ESLONG (`c4-eslong`) can import `gated300.add_ctrl0_reference(variants)` and
`reference_index(variants)`; set `long_credit.ctrl0_reference=true` on its bound
mode-2 trainer. `imagine` supplies `gated300_long_difference` and reference labels;
the helper is idempotent/RNG-free. `cost_estimate(rows, starts, ...)` returns costs.

## Interim preparation (not executed)
`tools/candA/interim_a.py CHECKPOINT.pt` prints the registration block; add
`--exposure-audit AUDIT.json` for the hash-bound append text. Instructions/receipt
schema: `tools/candA/S_INTERIM_A_REGISTRATION.md`. Append BEFORE `--run`.
Eight fresh worlds (identities only in the script), two per fixed focus cell;
ours vs exact-zero CTRL0, 300 s native warmup + 600 s intervention each, cap 14,400
sim-s. CPU-only one-shot output, precharged attempts, no automatic retries; paired
initial-state hashes, raw fleet+source J, safety/drop receipts and world bootstrap.
Exposure audit is still required; no training/selection use or efficacy claim.

## CPU receipts
`GATEMEAS_CPU.json`: integrated suite 117 passed/3 expensive legacy cases deselected;
final feature suite 18 passed after exact row-law/reduction fixes; final interim
preparation checks 2 passed. Ruff, diff check and 16-file config-loader chain PASS.
OFF identity covers metrics, parameters, gradients, Adam and RNG bit-for-bit.
Toy: dense J=220 vs CTRL0=200 (harm), long J=800 vs 1000 (gain), composite=20.
Real-WM synthetic K8/K16 x normal/memory replay: every non-native zero-actor row has
exactly zero long difference, including entrants; no critic calls; finite updates.
Those tests compress physical time to check wiring, not 300 s model fidelity.
Reproduce: `PYTHONPATH=src:. python -m pytest tests/unit/test_gatemeas.py -q`
(Windows uses `PYTHONPATH=src;.`); full suite command is in the JSON receipt.
