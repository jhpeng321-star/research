# HDV driver population switch (owner-approved 2026-10-14)

`population = "legacy3" | "textbook"`. Default everywhere stays `legacy3`, so old code and data
reproduce unchanged. `textbook` has to be requested explicitly:

| layer | how to pick it |
|---|---|
| `traffic_shepherd.scenarios.request_schedule / build_scenario` | `population="textbook"` |
| `traffic_shepherd.c4.env.CavSumoEnv` | `CavSumoEnv(..., population="textbook")` (`cav_vtype` is then ignored) |
| `python -m traffic_shepherd.c4.collect` | `--population textbook` (recorded in `COLLECT_SUMMARY.json`) |
| scenario manifest (`manifest.json["scenario"]`) | `driver_population`, `driver_population_version`, `driver_truth_file`, `driver_truth_sha256`, `cav_vtype="cav_typical"` (these keys are added only in textbook mode) |

## Textbook population (`traffic_shepherd/c4/population.py`, stdlib only)

The IDM centre values come from the Treiber & Kesting (2013) highway calibration (*Traffic Flow
Dynamics*, Table 11.2): v0 = speed limit, T = 1.2 s, a = 1.2 m/s², b = 2.0 m/s², s0 = 2 m,
delta = 4. Each HDV gets its own vType `tb.<vehicle_id>` with these draws:

- `speedFactor` ~ N(1, 0.1), truncated to [0.8, 1.2]. Desired speed = route limit × speedFactor (mainline 31.29 m/s, ramp 22.22 m/s). `speedDev` stays 0.
- `tau` (T): lognormal, median 1.2 s, 10–90 % range 0.9–1.6 s, truncated to [0.7, 2.0].
- `accel` (a): lognormal, median 1.2, 10–90 % range 0.8–1.8, truncated to [0.6, 2.5].
- `decel` (b): lognormal, median 2.0, 10–90 % range 1.5–2.8, truncated to [1.2, 3.5].
- `minGap` (s0): N(2.0, 0.3), truncated to [1.2, 3.0].
- Fixed values: `delta` 4, `emergencyDecel` 9, `sigma` 0.
- (ln a, ln 1/T) is bivariate normal with rho 0.4 before truncation, so aggressive drivers both accelerate harder and follow closer. Measured on 5,000 draws: corr = 0.355 after truncation.
- LC2013 draws: `lcCooperative` U[0.3, 1], `lcAssertive` U[0.5, 1.5], `lcSpeedGain` U[0.5, 1.5], `impatience` U[0, 0.6]. This is the same `impatience` attribute the legacy vTypes use, not `lcImpatience`. `lcStrategic` = 1.
- Truncation uses joint rejection resampling for the (a, T) pair, from the same stream.
- Sampled on 5,000 draws: medians are 1.00 / 1.20 / 1.21 / 2.01 / 2.00, and the 10–90 % ranges are T 0.92–1.59, a 0.82–1.77, b 1.50–2.68.
- Each vehicle's draws are seeded by `textbook_driver_v1:{seed}:{vehicle_id}`. This stream is independent of the departure, CAV-identity and legacy type streams, so departures and CAV identities match legacy exactly (tested).

### CAVs

Every CAV uses one vType, `cav_typical`, set to the population medians (owner rule: a CAV has no
driver type). The values are in `TEXTBOOK_CAV_PARAMS`:

- speedFactor 1.0, T 1.2, a 1.2, b 2.0, s0 2.0, delta 4
- lcCooperative 0.65, lcAssertive 1.0, lcSpeedGain 1.0, impatience 0.3

### Truth table

Each scenario writes `scenario/hetdata_truth.csv`. It has one row per vehicle with the columns
`vehicle_id, driver_class (textbook | cav_typical), vtype_id, is_cav` plus the 11
`wms.data.driver_truth.PARAM_ORDER` columns. The values equal the exact values SUMO receives
(rounded to 4 decimals). `load_driver_sidecar` and `dense_truth` (`truth_path`) find it in the
episode `scenario/` directory.

**This table is for evaluation only. Never use it as a model input or in a loss.**

### Exports for the integrator (branch c4/infrav5)

```python
from traffic_shepherd.c4.population import (
    TEXTBOOK_POPULATION_MEDIANS,  # {"accel":1.2,"time_headway":1.2,"decel":2.0,"min_gap":2.0,"speed_factor":1.0,"delta":4.0}
    TEXTBOOK_CAV_PARAMS,          # PARAM_ORDER keys: speed_factor, tau_s, min_gap_m, accel_mps2, decel_mps2, delta, lc_*, impatience
    TEXTBOOK_CAV_VTYPE,           # "cav_typical"
    sample_driver,                # (seed, vehicle_id) -> PARAM_ORDER dict
    POPULATIONS, check_population)
```

- The WM driver prior (`loss.driver_prior`) should use `TEXTBOOK_POPULATION_MEDIANS`. Its keys are the wms driver-encoder names; v0 = limit × speed_factor.
- The CAV execution cap `idm_normal` in textbook mode should use `TEXTBOOK_CAV_PARAMS`: accel 1.2, decel 2.0, tau 1.2, minGap 2.0, speedFactor 1.0, delta 4. It keeps the legacy normal_car values (2.6 / 4.5 / 1.1 / 2.5) in legacy3 mode.

### Acceleration caps

- The direct command cap stays `direct_accel_max_mps2` = 3.0 and the policy cap stays 2.6.
- Under speedMode 27, bit 1 is set: SUMO caps each CAV at its vType accel. For `cav_typical` that is **1.2 m/s²**, so in textbook mode SUMO binds before either cap. Higher commands are clipped to 1.2.
- Bit 2 is not set, so b = 2.0 never limits CAV braking. Braking down to emergencyDecel 9 is possible.

## Checks

- **SUMO 1.27.1 IDM ignores `sigma`.** FCD trajectories (600 s, legacy fixture at 2300 veh/h) were byte-identical with sigma 0.2/0.4/0.6 vs sigma 0, and with `--seed 1` vs `--seed 2`. The textbook population is also identical across SUMO seeds.
- **Legacy3 is bit-identical.** `routes_sha256` / `requests_sha256` are pinned to values produced by the pre-change code (1b249be). The manifest key set and the scenario file list are unchanged.

## SUMO smoke

Setup:
- Run ID C4-POPFIX-SMOKE-20261014, laptop, 3 processes, local output `runs/active/C4-POPFIX-SMOKE-20261014/` (gitignored).
- Each episode: warmup 300 s plus a 300 s window. Two dev seeds per case: 4107/4108 at p0.2/m2700, and 4109/4110 at p0.4/m2400.
- Policies: native and generic_random (`gr_mode` speed, no branches), run through `collect.run_job`.
- Script: `tools/c4_population_smoke.py`.

All metrics are window means:
- jam = vehicle-steps with v < 8 m/s
- exit = mainline_exit crossings, i.e. throughput past the merge, per 300 s
- entry = mainline entry crossings
- wait = source wait in vehicle-seconds

| policy | case | population | exit | entry | wait (veh·s) | mean v (m/s) | jam | min TTC (s) |
|---|---|---|---|---|---|---|---|---|
| native | m2700 p0.2 | legacy3 | 235.0 | 220.0 | 1,922 | 22.57 | 0.148 | 1.31 |
| native | m2700 p0.2 | textbook | 192.0 | 160.0 | 24,719 | 19.31 | 0.228 | 1.46 |
| native | m2400 p0.4 | legacy3 | 227.5 | 194.5 | 194 | 24.12 | 0.118 | 1.36 |
| native | m2400 p0.4 | textbook | 191.0 | 163.5 | 14,437 | 20.26 | 0.192 | 1.61 |
| generic_random | m2700 p0.2 | legacy3 | 230.0 | 207.0 | 3,711 | 20.69 | 0.082 | 1.26 |
| generic_random | m2700 p0.2 | textbook | 190.5 | 159.0 | 24,881 | 20.03 | 0.117 | 1.44 |
| generic_random | m2400 p0.4 | legacy3 | 223.0 | 190.5 | 883 | 20.98 | 0.049 | 1.47 |
| generic_random | m2400 p0.4 | textbook | 189.5 | 161.0 | 14,856 | 18.86 | 0.138 | 1.80 |

Results:

- **Safety:** 0 collisions and 0 teleports in all 16 episodes.
- **Congestion (owner's expectation confirmed):** with textbook drivers, throughput past the merge falls 17 % (190.8 vs 228.9 per 300 s). Mean speed falls from 22.1 to 19.6 m/s. Jam fraction rises from 0.099 to 0.169.
- **Source wait grows about 12×**, because the mainline demand is no longer served.
- **CPU cost:**
  - Textbook: 0.091 CPU-s (python + SUMO) per simulated second, vs 0.082 for legacy3 (+11 %), measured with 3 concurrent processes.
  - The routes file is about 0.5 MB (about 1.4 k vTypes), and building the scenario takes about 0.8 s per episode.

### Why the textbook population saturates

These are pure-SUMO diagnostics on seed 4108 (CAVs native, no TraCI); scripts were in agent
scratch only.

- **Saturation demand.** The textbook network capacity is about 2,330 veh/h in total (about 1,930 mainline plus 400 ramp). The source queue starts growing at mainline demand of about 2,100–2,300 veh/h. At m2400–m2700, the C4 demand levels, it grows without bound (110 vehicles waiting at t = 1200 s at m2500). Legacy3 is still unsaturated at m2500.
- **The cause is the low maximum acceleration a.** Over the 300–600 s window at m2700, vehicles inserted:
  - textbook: 186
  - textbook with decel 4.5: 221
  - textbook with accel 2.6: 240 (= legacy3)
  - textbook with tau 1.1: 186
- **The ramp is not what limits it.** With the ramp removed, textbook mainline insertion is still capped at about 1,930 veh/h (161 per 300 s), vs 215 per 300 s for legacy3. So the bottleneck is the queue-discharge capacity of IDM with a ≈ 1.2.

**Implication for collection:** at the current demands, textbook episodes are oversaturated, and source wait will dominate the cost. Either accept this, or add lower demands (the development family allows 1900–2800).
