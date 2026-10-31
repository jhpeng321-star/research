# S-C1PIDIAG — fixed recorded-state attribution (2026-10-31)

Owner authority: S-C1PIDIAG line 5574, shared C4-V7FT registration, read before
planning. No SUMO, training, fitting or black-box retraining is authorized.

## Fixed panel, before scientific computation

`docs/paper_results/c1pidiag/PANEL.json` binds 40 `pi_u_same` and 30 `wall_same`
starts: first eligible entries sorted by `(world,t0_s,sid)` in the refresh index
SHA256 c3c7d3f0008b44b9831989c74932502644ae17529183b2da0208c83133c94094.
Forbidden identities are excluded before opening state files; no error-based
selection/replacement. Retain original indices and source hashes. 53 distinct
worlds overall; cohorts are separate and are never pooled. PI is same-world
controller transfer, not unseen-world generalization. Wall history availability
is an input property, never a reason to replace a chosen start.

## Frozen models and staged ablations

- C1 structured `87ed02e0`, 13E `952ec059`, WM2 `b8797242`; frozen original
  BB-traj `afaaa32c` comparator. Verify full hashes and physics receipts.
- History grid: snapshot, observed last 5 s, genuine observed last 60 s where
  available. Reconstruct ONLY from the same recorded episode at times <=t0;
  never use post-t0 frames or a native twin as the PI history. Missing history
  is unavailable, never repeated snapshots labelled full history.
- Settings grid: inferred vs existing population-default diagnostic switch.
  Document that this switch also changes the conditioning embedding. In old
  residual mode physical inferred settings may be inert; preserve that fact.
- Teacher-forced primary: all available history/settings cells, baseline
  components, forecast once to 120 s and score at 60/120 s. "Teacher-forced"
  means CAVs only; HDVs remain free-running, not one-step teacher forcing.
- Components on each model's longest genuine available history, inferred and
  population: residual on/off x lanes original/MOBIL-prior. Remove only the
  acceleration correction for residual-off, preserve lane module; MOBIL-only
  bypasses learned lane logits and uses original IDM/MOBIL physical prior.
  Deduplicate baseline cells. These are diagnostic interventions, not fits.
- Entries OFF for the primary historical contract, including WM2+mix (entrant
  mix is dormant). Separately evaluate WM2 autonomous entries+learned mix only
  if source queue/demand inputs are recorded and compatible; otherwise mark
  unavailable, never invent demand/backlog. Full-road scope is held constant.
- Then conditional closed loop on all 70 starts: old baseline, frozen BB-traj,
  each new-model baseline and its best teacher-forced config per cohort. Best
  means smallest common-ID 60 s median; ties prefer fewer changed switches,
  then longer genuine history, inferred settings, lexicographic config ID.
  This within-panel selection is exploratory. Use refresh common external
  CAV plant, same recorded presence schedule, no autonomous-feedback claim.
  Wall membership/timing from refresh is explicitly conditional replay.

## Measurement, controls and resource admission

Export per-start per-ID errors, survival counts, signed position/speed drift,
60/120 s absolute position errors and paired WM-minus-BB-traj differences on
common surviving HDV IDs. Aggregate median of start medians; 2,000 whole-world
bootstrap draws, seed 0; pair first, do not subtract marginal intervals. Also
retain baseline-compatible unpaired-ID medians to reconcile refresh numbers.
Use identical identity/arm RNG streams across ablations; report runtime/source
hashes and errors. No model/component claim from unavailable or failed cells.
Training coverage comparison uses C1GAP exposure exports, not truth files.

gpuA initial nvidia-smi: 10,701 MiB free, GPU 100%; BELOW 12 GB admission.
Use existing gpuA Python CPU, nice19, one thread/worker, <=24 workers, only if
free cores permit. Recheck load before dispatch. Own PIDs/start ticks only;
write-once outputs, preserve failures, no deletions/new environments/push.
Denied paths and seed blocks are guarded before input opens. No raw truth or
vType/assignment files. Commit this plan and shared registration before inference.

## Coordination and deliverable

Read `c4-c1closed/C1CLOSED_DESIGN.md`: its full-history 40-world WMDATA autonomous
feedback/entries panel is distinct. This task diagnoses legacy PI/wall starts;
it does not duplicate or substitute for S-C1CLOSED. Final
`docs/paper_results/C1PIDIAG.md` <=60 lines, attribution, world-cluster CIs,
dominant measured cause (or honest unresolved cause), fair-history re-score,
switch tradeoffs and old-C1 Paper-B recommendation if transfer is still lost.

## Prospective history-recovery addendum (before auxiliary forecasts)

Original pre-t0 60 s is absent, but each fixed episode records the following
60 s. Add an explicitly separate, outcome-independent clock-shift diagnostic:
same 70 episode identities, t0' = original t0 + 60 s, all vehicles observed at
t0' as the new roster. Frames in [t0'-60,t0'] become observed history; frames
strictly after t0' are prediction targets. No original-t0 prediction receives
these frames. This does not replace the original fixed panel.

Rebuild stable-ID kin/mask/target-mask arrays from the same arm's recorded
trajectory, causally carry junction lane labels, and compare snapshot/5/60 s
x inferred/population for C1/13E/WM2 and frozen BB-traj on these SAME shifted
starts. Preserve the recorded remaining endpoint clocks: PI 60/120 s, wall
60 s (last-label tolerance <=0.5 s); unavailable 120 s wall stays unavailable.
Report endpoint attrition. No component winner is chosen from this auxiliary
panel; this is the registered full-history test, with 60 s primary. Entries
remain unavailable. Run sequentially after the original grid, <=24 workers.

## Source-inspection component clarification (before extra forecasts)

Current `accel_delta_mps2` includes attention/temporal residual, courtesy, learned
yield and ramp-speed adaptation. Thus original r0 is **all learned acceleration
corrections off** (A0), not attention residual alone. Add frozen single-switch
teacher-forced checks on the same original starts/longest available history,
inferred settings: attention+temporal residual off, courtesy off, learned yield
off for 13E/WM2, plus ramp-speed-adaptation off for WM2 (490 units). Keep all
other components unchanged. This source-defined split is registered before
looking at any panel aggregate. After scoring, conditional closed-loop-check
the best extra switch per model/cohort if it improves on that model's original
baseline; selection is exploratory. Run after both earlier panels, <=24 workers.
