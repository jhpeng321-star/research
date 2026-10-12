# WM-S-LANECHECK-20261012 — in progress

## Scope and constraints

Offline, CPU-only analysis of recorded wall_same (67 states) and optional seen_controls trajectories, plus reuse of the existing physics+true-driver-type rollouts. No training, SUMO execution, GPU, or edits outside this run directory. True driver types are evaluation-only. Excluded paths/files will not be opened.

## Question

Where do WM-S's MOBIL+merge decisions disagree with recorded SUMO LC2013 HDV lane changes, and which lane-change categories account for wrong-lane rollout error at 60/120 s?

## Status — measured first pass

The first one-step pass covers all 67 `wall_same` state windows (and the `seen_controls` second opinion), with existing per-tick recorded future neighborhoods and no rollout drift. At 0.5 s resolution, default MOBIL produced 1,009 wanted-change episodes vs 952 recorded changes and matched only 14 at the identical transition/direction (precision 1.39%, recall 1.47%). True-type IDM settings with model-default lane settings produced 717 episodes, 11 exact matches (1.53% precision, 1.16% recall). The lane rule therefore has both excess unmatched intents and missed recorded events; tuning the aggregate change rate alone cannot resolve lane timing.

On the reused physics+true-type+MOBIL rollout, 179/214 wrong-lane HDVs at 120 s are mainline-left/overtake-category cars. Their absolute position-error sum is 48.1 km (75.5% of the wrong-lane subset's 63.7 km; 20.4% of all matched-HDV absolute position error). Attribution is descriptive: latest divergent lane-change category, not a causal intervention.

## Methods and interpretation

- For each 0.5 s recorded transition, the exact recorded neighborhood is passed to `wms.physics.mobil.mobil_merge_logits`; no predicted states enter this score. Ties use WM-S `argmax` behavior. Default is the R4 sumo-fidelity physics configuration. True-type changes only the five evaluation-only IDM settings from `out/truth.json`; MOBIL policy parameters remain at WM-S defaults (politeness .2, threshold .1 m/s², merge gap 6 m, urgency 1). This distinction matters because SUMO lane parameters are not MOBIL parameters.
- Lane `-1` is forward-filled before change detection; a counted event requires the driver present on both frames and both lanes valid. An intent episode collapses adjacent wanted ticks per car/direction. Precision/recall require an exact same 0.5 s transition and direction. Every wall_same recorded change stores its nearest same-car/same-direction MOBIL intent offset: negative = MOBIL earlier, positive = later, no matching intent anywhere = never.
- Categories use lane indices: 2→1 ramp merge, 1→0 mainline-left/overtake, 0→1 mainline-right/return. “Jam” is local speed <5 m/s. Proximity is within 25 m of any CAV, and separately within 25 m of a listed wall member.
- Rollout attribution reuses `variant_phys_true.pkl`. Each wrong-lane HDV at a horizon is assigned the latest lane-change category on either path; this is descriptive, not a counterfactual causal decomposition. Position-error share uses summed absolute longitudinal error over matched HDVs.

## One-step agreement and disagreement categories (measured)

`wall_same`: 680,987 valid HDV decisions. Recorded changes: 952 (0.140%/HDV-step). Default MOBIL wanted 4,460 vehicle-steps (0.655%) collapsed to 1,009 intent episodes; true-type wanted 3,303 steps (0.485%) / 717 episodes. Unmatched intent episodes: 995 default, 706 true-type. Exact event precision/recall: **1.39% / 1.47%** default; **1.53% / 1.16%** true-type. The true-type settings cut intents by 29%, but do not improve exact timing agreement.

| wall_same category | recorded events | default intents (exact matches) | true-type intents (exact matches) | recorded changes near CAV / wall member | recorded changes <5 m/s | recorded merge position; default / true-type wanted merge position |
|---|---:|---:|---:|---:|---:|---|
| Ramp→mainline | 160 | 221 (3) | 198 (3) | 123 / 0 | 146 (91%) | mean s=61.1 m; intent mean s=61.0 / 62.1 m |
| Mainline-left / overtake (1→0) | 621 | 740 (11) | 483 (8) | 269 / 2 | 298 (48%) | not applicable |
| Mainline-right / return (0→1) | 171 | 48 (0) | 36 (0) | 6 / 3 | 0 | not applicable |

Timing offset of nearest same-direction intent (recorded changes): default ramp 77 early / 3 on-time / 0 late / 80 never; left 101 / 11 / 0 / 509; right 1 / 0 / 0 / 170. True-type: ramp 77 / 3 / 0 / 80; left 66 / 8 / 0 / 547; right 1 / 0 / 0 / 170. So a ramp change is usually preceded by a MOBIL urge (~7.9 s mean earlier among the 80 with a candidate); mainline-left misses dominate the never category. Across all recorded changes 759 default and 797 true-type have no same-direction urge anywhere in the trajectory. There are essentially no late MOBIL urges in this set.

Recorded events near any CAV: 398/952 (41.8%); near wall members: 5/952 (0.5%). Wanted episodes near CAVs: 604/1,009 default (59.9%), 428/717 true-type (59.7%); near wall members: 32 (3.2%) / 25 (3.5%). Ramp changes are predominantly jam-speed (recorded 91%, wanted 90%); mainline-left changes are split around half jam (48% recorded; 42% / 38% wanted); recorded rightward returns occur at free speed in this sample. `seen_controls` confirms the pattern at larger scale: 8,314 recorded changes; exact precision/recall 1.63%/1.29% default and 2.01%/1.14% true-type.

## Free-running rollout view (measured, saved physics + true types + MOBIL)

At 60 s, 279 matched HDVs are wrong-lane. Latest-divergence categories: left/overtake 209 cars (mean |ds| 259 m; 54.2 km total; 78.2% of wrong-lane error), ramp-merge 6 (1,077 m mean; 6.5 km; 9.3%), right-return 39 (165 m mean; 6.45 km; 9.3%), other 25 (87 m mean; 2.18 km; 3.1%). At 120 s, 214 are wrong-lane: left/overtake 179 (268 m mean; 48.1 km; **75.5%** of wrong-lane error and **20.4%** of all matched-HDV |ds|), right-return 19 (732 m; 13.9 km; 21.8%), other 16 (108 m; 1.72 km; 2.7%). No ramp-merge category remains among wrong-lane cars at 120 s. These counts describe cars, not independent events; position drift and lane divergence co-occur.

## What the campaign config says (measured) and why it differs from MOBIL

Inspected campaign scenario XML, generated route/vTypes, and launcher/generator source (`traffic_shepherd/simulation.py`, `scenarios.py`). SUMO 1.27.1 runs with `--step-length 0.5`; `actionStepLength` is not overridden. The campaign `.sumocfg` has no lateral-resolution or lane-change override: lane-change model is SUMO's LC2013 default, without the sublane model. No lane-change duration is set, so trajectories contain discrete lane occupancy transitions rather than sublane lateral sweeps. VTypes explicitly set cooperative yielding and impatience: aggressive `lcCooperative=.3`, `lcAssertive=.6`, `impatience=.7`; normal `.6`, unset assertiveness (SUMO default), `.3`; cautious `1.0`, unset assertiveness, `0.0`. `lcStrategic`, `lcSpeedGain`, `lcKeepRight`, and `lcLookaheadLeft` are absent from these vTypes and therefore use SUMO defaults.

WM-S MOBIL instead evaluates local same-tick IDM acceleration incentives and 6 m safe gaps with a 0.1 m/s² threshold (default path); urgency relaxes the mandatory ramp merge over the fixed merge zone. True-type mode adds the IDM follower-gain term with fixed politeness .2. R4 uses `merge_cal=False`, so it has no calibrated cooperative yielding. There is no keep-right reward for returning right, per-driver impatience, explicit LC2013 strategic look-ahead, or LC2013 speed-gain thresholds. SUMO's cooperative/assertive/impatience behaviors and strategic/speed-gain/keep-right objectives can therefore request lane changes that this local MOBIL rule never requests; noise and SUMO's internal decision state add further timing variation. This is a concrete mechanism consistent with the observed omissions, not proof that any single setting caused each miss.

## Round-6 options (evidence vs inference)

1. **Rule-first LC2013 features (inference, implementation-only):** add an explicit keep-right return incentive, strategic merge look-ahead, and urgency/cooperation conditioned on the recorded vType settings, retaining MOBIL safety as a hard feasibility veto. The one-step measurements show 170/171 recorded mainline-right changes are never proposed and most ramp changes are proposed too early, which motivates this design; do not treat the current state-replay score as efficacy evidence.
2. **Learn lane-event timing (inference, training candidate after R5 scoring):** predict stay/left/right change hazards with a direction- and category-aware event loss rather than tuning the mean lane-flip rate. The measured exact timing match is only 1–2%; intent-count correction alone leaves most misses (and is not the current result's objective). Use observable histories only; true vTypes remain evaluation-only.
3. **Hybrid gate (inference, proposed):** keep MOBIL as safety/adjacency envelope, learn when to trigger feasible lane events, then calibrate on frozen one-step and saved-rollout diagnostics before any authorized round-6 run. This best separates safety feasibility from the demonstrated LC2013 timing/strategy mismatch.

## Verification and artifacts

`scripts/lanecheck.py` reproduces `results.json` from the read-only source artifacts. `results.json` is 521,929 bytes (<5 MB) and contains every wall_same recorded event with per-model nearest-intent timing, aggregates for both state sets, and 60/120 s rollout attribution. `python -m py_compile runs/active/WM-S-LANECHECK-20261012/scripts/lanecheck.py` passed; the CPU analysis completed with one worker. No SUMO run, training, or GPU was used.

**Close-out scope note:** the repository-wide experiment ledger, ROUTE, OPEN_QUESTIONS, INDEX, and generated summary were not changed because the requested write boundary permits changes only in this run directory and `E:/agent-work/tmp`. The scientific evidence record itself is complete here; ledger/navigation integration remains the only close-out follow-up if the write boundary is expanded.
