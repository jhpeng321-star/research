# IDM v5.5 timing-aligned Gate-F and conditional wall-pilot protocol

Status: executed, 2026-09-01. Development seeds only. Results are reported in
`docs/IDM_V5_5_T880_GATE_F_AND_WALL_PILOT_RESULTS.md`.

## Timing and outcome

The exact physical confirmations in the density mapper are 966, 975, 999 and
999 seconds. The proposed wall is activated at absolute SUMO time 880 s. Its
twentieth wall command advances the simulator from 899 to 900 s; 900 s is the
planned release boundary back to identical analytical damping.

The primary outcome is prevention or shortening of sustained premerge
congestion: at least five premerge vehicles below 15 m/s together with
premerge mean speed at or below 15 m/s for 60 consecutive seconds. Report
incidence, total qualifying duration, longest run, first onset/confirmation,
and recovery. `h < 0.15` remains a diagnostic feature only.

## Stage 1: treatment-blind Gate F

Run only damping through absolute time 880 on the already-opened 2700/p30
development seeds 4401--4406. No wall command is allowed and no treatment
outcome may be read.

A seed is admitted only when the unchanged v5.4 wall contract provides:

- members for every occupied lane;
- exact full occupied-lane coverage;
- initial, predicted maximum and predicted terminal spans no greater than 50 m;
- nonnegative projected terminal runway beyond the 50 m release boundary for
  the complete unchanged 20-second treatment;
- the frozen front/rear gap, TTC and acceleration activation-safety pass;
- projected duration 20/20 seconds.

The conditional six-seed wall pilot is authorized only if at least four seeds
pass. Four is the minimum that could later retain at least two beneficial and
two safe nonbeneficial independent outcomes. Gate-F selection never uses the
known control outcomes.

## Stage 2: conditional matched pilot

If Stage 1 passes, rerun every seed as frozen analytical damping and identical
damping plus the unchanged v5.4 wall from 880 to 900. Use the exact 1080 s or
proven natural-clearance horizon. Preserve exact predecision identity,
out-of-band damping and release to damping.

Do not change duration, strength, geometry, assembly, taper, release,
membership or handoff rules. Record full-horizon physical congestion,
recovery, safety, delay, merge discharge, upstream storage, treatment dose,
coverage, member changes, taper and release telemetry. Treatment-period safety
and service costs count even though the sustained-congestion endpoint is
interpreted post-release.

The common absolute endpoint is 1080 s. An earlier endpoint is accepted only
when the runner records normal termination with both zero live vehicles and
zero expected vehicles; the empty remainder through 1080 then contributes no
congestion, delay, discharge or storage. A physical shortening is material
only when the total duration of runs that individually last at least 60 s is
reduced by at least 10 s (and by more than exact-repeat variation). The
unchanged treatment must realize 20/20 command seconds and full occupied-lane
coverage; the generic historical 80% dose threshold is not used here.
For both prevention and shortening, materiality additionally requires at
least a 10 s reduction in raw congestion-predicate exposure and in the longest
contiguous run, preventing a 60 s versus 59 s threshold artifact. A
service-inferior treatment is reported as a separate service failure and does
not count as safe nonbeneficial diversity. Every admitted treatment must pass
the frozen safety and service guardrails before expanded data generation.

Classify each requested treatment by the first failing stage:

1. infeasible formation;
2. failed realized dose;
3. safety failure;
4. service failure;
5. safe but ineffective/nonbeneficial treatment;
6. safe beneficial treatment beyond exact repeat noise.

Expanded counterfactual generation is considered only if multiple independent
seeds are safe and beneficial beyond repeat noise while multiple safe
nonbeneficial outcomes remain. No predictor, extra seed, or validation/final
partition is authorized by this protocol.
