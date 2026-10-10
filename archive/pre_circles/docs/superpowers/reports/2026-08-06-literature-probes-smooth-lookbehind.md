# Literature probes: smooth execution & looking-behind reference

Date: 2026-08-06
User steer: research online whether there is a way for a CAV to take action
without inducing TTC (HDV–HDV) events. Two mechanisms surfaced by the
literature were implemented as eval-time-only overrides on the champion
(iter_90 + soft mask + sign gate) and probed at dense p10 over 12 held-out
seeds, benchmarked against the champion's own profile on the same seeds.

## Mechanisms tested

- **Smooth execution** (Stern et al. 2017, ring-road wave dissipation): all
  speed changes executed as a slow ramp (0.4 m/s² cap vs the mode's 1.0,
  with EMA smoothing) instead of a step. Hypothesis: wave injection scales
  with the *rate* of CAV speed change.
- **Looking-behind reference** (Wang 2020 / 2023, LCC "looking-behind"
  topology): the executed speed target is the EMA of the upstream band
  (150–200 m behind) mean speed instead of the mode's own target. The CAV
  becomes an upstream wave-absorber. Hypothesis: the harm happens where the
  CAV cannot see (upstream), so upstream information is the blind spot.

## Results (12 seeds, dense p10, 1.5 s ruler)

| Config | d_ttc | better/worse | d_gap | d_loss |
|---|---|---|---|---|
| champion (sign gate) | **−3.42** | **9/3** | −3.25 | −4.01 |
| smooth-exec | −1.08 | 8/4 | −3.28 | −2.27 |
| look-behind | +0.67 | 5/7 | −1.87 | −1.34 |

Both probes lose to the champion on every leg. Worse, they destroy the
champion's largest TTC wins: seed101 −10 → +5/+6, seed112 −12 → −1/−4,
seed104 −6 → −1/+6.

## Verdict: mechanism falsified

If execution abruptness were the injection source, smoothing would reduce
TTC-worse seeds. It instead reduced the champion's TTC-*improvement* seeds —
because the champion's TTC wins come from *active downstream metering*
(holding the merge calm), and a slower/upstream-following execution
neutered exactly that metering while the HDV–HDV wave harm persisted. The
same holds for looking-behind: upstream information did not make the CAV a
better absorber here.

Conclusion: the TTC cost is not a function of execution abruptness or
information topology — it is intrinsic to active metering in this dense
merge corridor. This is the fourth independent falsification (reward
levers, p20 training, medium training, execution mechanics) of a
mechanism-bound coupling, further anchoring the finding that the coupling
is a property of the scenario + the *act* of metering, removable only by
changing the scenario (penetration) or not metering.

## Artifacts

- Infra (default OFF, reusable): `env/actions.py` (`smooth_exec`,
  `look_behind_exec`, `_exec_cmd`, `_behind_band_mean_speed`), `env/core.py`
  mode_config wiring.
- Tests: `tests/test_smooth_exec.py` (8 tests).
- Drivers: `_eval_champion_iter90_{smooth,lookbehind}15_dt05_smoke12.py`.
- Evals: `runs/eval/champion_iter90_{smooth,lookbehind}15_dt05_smoke12`.