# S-RB-LANE — SUMO LC2013 highway rebuild and FIT diagnostic

> Historical diagnosticv1, preserved. Latest source/scoring corrections and
> precision results: [iteration diary](../LC2013_PRECISION_ITERATIONS.md) and
> [current handoff](../../HANDOFF.md). Coordinate contractv2 starts at candidate9.

## Result for integration
**Implemented, default OFF; improved recorded-event recall, NOT qualified as
full SUMO parity or long-horizon distributional fidelity.** Final source model
is commit `127a106f` on `c4/rb-lane` (base `f0db3c82`); subsequent commit adds
receipts/report only. Owner WM-PHYS authorization: REGISTRATION.md:5683.

- New tensor physics: `src/wms/physics/lc2013.py`; adapter:
  `src/wms/model/lc2013_adapter.py`. Small guarded edits in spec/wms, no factory
  changes, no new learned weights. Default model behavior remains unchanged.
- Opt in with `ModelSpec(lane_model="lc2013", lane_head="off")`. Strategic
  route/ramp-end urgency, cooperative lane clearing, speed gain, keep-right,
  source fixed-point wish accumulation/lookahead memory, SUMO-IDM two-sided
  secure gaps/minGap/lcAssertive, follower-help speed advice, and default
  right-overtaking prevention. Sublane OFF. Lane decision after motion.
- Only directly comparable motion-inferred IDM or explicitly named LC settings
  map. Existing encoder MOBIL politeness/threshold are NOT LC2013 parameters;
  LC parameters default to SUMO population values. No impatience gap heuristic.
- Existing learned acceleration/merge corrections must be disabled by WM-PHYS
  integration. Build the LC spec; do not merely change `.lane_model` on a live
  first-passage/hazard model that retains conflicting old modules.

## Fixed diagnostic and provenance
Only WMDATA FIT worlds **969400–969431**, five arms, 160 episodes. Only six
allowlisted observable view files per episode; all hashed. Population settings,
60s causal wish warmup (240–300s), score first60s (300–360s), same-ID HDVs on
normal road in adjacent frames. Both models receive recorded post-motion
longitudinal state with PRE-change lanes. CAV bodies constrain gaps, their
decisions are excluded. No SUMO, GPU, training, hidden-setting or prohibited-path
reads; no process signalling. cpuQ nice19, one CPU thread, CUDA hidden.

The first LC candidate completed in187.56s and is retained (`panel_results/`).
A source-only amendment adds right-overtake prevention using SUMO's own
congestion rule (MSVehicle.cpp:1494-1495), not an invented threshold. Final
candidate completed in183.37s (`panel2_results/`); completed MOBIL scores reused
after exact input-hash, cohort, physics and horizon checks. No parameter fitting
or outcome-based candidate selection. Both dispatches exited; no running jobs.

## Final paired event scores
- **Exact recorded changes reproduced: LC 1,136/3,255 =34.90%; MOBIL
  244/3,255 =7.50%.** The historical **1.4%** is a different panel/contract and
  cannot be treated as this paired baseline.
- Mainline only: LC794/2454 (32.36%) versus MOBIL9/2454 (0.37%).
- Overtake1→0: LC587/1592 (36.87%) versus MOBIL8/1592 (0.50%).
- Return0→1: LC207/862 (24.01%) versus MOBIL1/862 (0.12%).
- Ramp merge2→1: LC342/801 (42.70%) versus MOBIL235/801 (29.34%).
- ±2s one-to-one positive-step matches: LC1160/3255 (35.64%) versus
  MOBIL277/3255 (8.51%).
- **False-positive proposals remain substantial:** LC44,333 positive decision
  steps, exact precision2.56%; MOBIL7,637 steps, exact precision3.19%. Persistent
  wishes can propose repeatedly until the recorded lane actually changes.
- Compressing consecutive positive proposals into runs: LC5,159 runs,
  ±2s matches861 (recall26.45%, precision16.69%); MOBIL2,367 runs, matches273
  (recall8.39%, precision11.53%). Matched run timing MAE0.472s vs0.727s;
  position MAE8.242m vs0.866m. These compare different matched event subsets.
- Exact-event position/timing errors are identically zero under longitudinal
  teacher forcing; they are not evidence of accurate trajectory prediction.
  Positive-run counts are not autonomous lane-change rates. No breakdown,
  clearing, delay-distribution or controller-effect claim follows from this test.

## Verification and external helper check
- **62 CPU tests PASS**, 26.05s; 21 LC/model/scorer tests plus41 existing lane,
  WM-lane and native-CAV regressions. JUnit: `tools/rblane/receipts/CPU_TESTS.xml`.
- Completeness160/160, exact aggregate reassembly, current model/scorer SHA256,
  paired input/cohort equivalence and MOBIL-score reuse all PASS:
  `tools/rblane/receipts/VERIFIED.json`. Final SUMMARY SHA256
  `984df884e6d2a030d7b6989a45ff35ed4785d3770f32ec74a08f913b0981aeaa`.
- Used completed GLM `rb-shared/targets/lc_events.csv`: all3220 scored events
  after the first tape frame agree by world/arm/ID/time/direction; max position
  discrepancy0.005m is target rounding. The35 missing events all occur at300.5s:
  helper raw tape STARTS300.5s and cannot detect its initial lane transition,
  whereas views include300s. Original MISMATCH retained, explicit boundary
  reconciliation PASS, no inference repeated. `TARGET_CROSSCHECK.json` and
  `TARGET_RECONCILIATION.json`. `lcspec/DONE.md` was not present at completion.

## Known parity gaps / next integration gate
Synchronous messages and independent proposal gaps differ from SUMO's serial
downstream-first updates; the runtime geometrically arbitrates front-to-back.
Full leader-overtake/fall-behind/counterchange blocker reservation and all
follower-negotiation branches are not ported. Generic route queued occupation
must be supplied by caller; defaults use scenario ramp end and through lanes.
Nonlegacy keep-right follower adjustment, nonzero speed-gain-lookahead,
stochastic lcOvertakeRight, emergency/nonhighway special cases are outside this
subset. Source inventory and exact formulas: `docs/LC2013_SOURCE_NOTES.md`.
Use the integrated CF/merge/LC candidate for S-RB-EVAL's distributional gate;
this local diagnostic by itself does not establish the owner goal.
# Precision-continuation notice (2026-11-01)

The original report below is preserved as the completed **diagnosticv1** result.
Subsequent source fixes, per-motivation false-positive diagnostics, frozen-encoder
and published motion-point sensitivities are in
[`../LC2013_PRECISION_ITERATIONS.md`](../LC2013_PRECISION_ITERATIONS.md).
Candidate9 corrects upstream coordinate origins in both runtime and scorer;
itsv2 measurements must not be pooled with the old paired baseline. See current
[`../../HANDOFF.md`](../../HANDOFF.md) for source/tests/live status. Precision
remains an unresolved issue; no distributional qualification is established.
