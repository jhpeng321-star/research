# P3 rollout2 G1 EQUIVALENCE — zero-residual / arrivals-off vs partA_v2g

**Piece:** P3 ROLLOUT2 (`output/wd01/wm_n1/rollout2/**`, branch
`zcode/wmn1p3`, base `zcode/wmn1base2` 99ad833).
**Gate:** G1 `zero-residual` of `PREFLIGHT_GATES_IDEA_A.md`
(zcode/wd01 8e702ed). **Tool:** `rollout2/equivalence.py`
(JSON report: `rollout2/equivalence_report.json`, regenerable).

## REAL J4 TRAIN result (second-pass review, 2026-10-01) — G1 PASS

Run in this workspace against the read-only data root `E:/research-dense`
(`WD_FORBID_D=1`; only `physics_steps.jsonl` of the two TRAIN runs was
opened — never HOLDOUT/CAL/chunk-5, `evaluate_summary`,
`dispatcher_rows`, `insertions`; zero SUMO):

```
$env:WD_FORBID_D='1'; python rollout2/equivalence.py --data-root E:/research-dense --n-steps 240
G1 PASS max_pos_err=0.000e+00 max_speed_err=0.000e+00 (tol 1e-06) source=j4_train_inventory
```

Full stdout (both window reports): `rollout2/g1_j4_train_stdout.txt`.

| window (TRAIN, J4 walls-pilot chunk-4) | run | window t0 | n_veh (CAV) | n_steps | max abs position err | max abs speed err | compared pairs | LCs exercised |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `w8659m2700p10t420` | `w8659m2700p10t420__w-lo-early` | 450 s | 30 (3) | 240 (60 s) | **0.0** | **0.0** | 1903 | 10 (3 merge, 3 disc-left, 4 disc-right) |
| `w8660m2700p10t300` | `w8660m2700p10t300__w-lo-early` | 330 s | 37 (3) | 240 (60 s) | **0.0** | **0.0** | 2964 | 18 (5 merge, 8 disc-left, 5 disc-right) |

Aggregate over both windows: max position error **0.0**, max speed error
**0.0** over 4867 compared vehicle-tick pairs (tolerance 1e-6), zero
collisions. Window tables were built with the registered eval_a2
machinery from each run's own physics rows; CAV identity from the runs'
`cav_ids`. Splits come from the J4 chunk-4 plan
(`E:/research-dense/output/campaign_walls_v1/results_plan/WORLD_PLAN.json`,
`splits` baked into the wm_n1 inventory parquet at build time by
`inventory/inventory_lib.py` PLAN_SPECS `walls_pilot_chunk4`); both
selected worlds are TRAIN in that plan. Engine-vs-engine equivalence is
exact — the port is math-identical on real traffic (merges, discretionary
LCs, platoons, free-road and end-guard regimes all exercised).

**Second-pass fix (review fix):** the previous selector required
`arm == "native_base"`, which excludes every J4 walls-pilot run (their
arms are the wall arms `W-*`), so the "J4 first" preference was dead
code and real J4 windows could never be opened (the report below fell
back to synthetic). The selector now accepts any complete non-smoke arm
in the J4 package (the zero-residual window table is arm-independent),
keeps `native_base` only for the generic chunk-4 fallback, dedupes
windows by `(world, t0_s)`, runs `--n-windows` (default 2) and — when no
data root has usable runs — fails loudly instead of disguising itself as
a synthetic pass (the fallback now requires explicit `--synthetic`).

## Earlier result (first pass, data-less workspace, 2026-10-01)

| run | source | window | n_veh | n_steps | max abs position err | max abs speed err | tolerance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G1 primary | `synthetic_fallback` (labelled; see below) | t0=30 s, 120 s | 8 (1 CAV) | 240 (60 s) | **0.0** | **0.0** | 1e-6 |
| G1 + replayed CAV commands | `synthetic_fallback` | t0=30 s | 8 (2 CAV) | 240 | **0.0** | **0.0** | 1e-6 |
| explicit table path | `explicit_table` | t0=45 s | 8 | 60 | **0.0** | **0.0** | 1e-6 |

The comparison is per vehicle per tick against the READ-ONLY scalar
`partA_v2g.engine` (lane, s, v; positions and speeds within 1e-6 — the
port is exact, the observed error is exactly 0.0). Lane changes were
exercised (2 mandatory merges, 2–3 discretionary), plus the accel-lane
end guard and road-end exits (only pre-exit ticks are compared; the
scalar engine removes exited vehicles from its snapshots).

**Why `synthetic_fallback`:** this workspace carries no campaign
physics logs (raw run evidence was deleted by MIG-04 and `runs/` does
not exist here), so no J4 TRAIN window can be opened locally. The
fallback is a deterministic, seeded synthetic J4-world scene built on
the frozen `params_nominal.json` geometry — clearly labelled in the
report (`source=synthetic_fallback`, "NOT a data window"). The
registered J4 TRAIN check runs where the data lives:

```powershell
# in the data-equipped environment (run data under <root>/runs/...):
$env:WD_FORBID_D='1'
python rollout2/equivalence.py --data-root <root> --n-steps 240
# window selection: TRAIN split FIRST; J4 walls-pilot package
# (campaign_walls_pilot_chunk4, any complete non-smoke arm) before
# generic chunk-4 native_base; 2 distinct TRAIN windows by default.
# HOLDOUT/CAL/chunk-5 are never selected or opened. Without usable run
# data the tool now FAILS loudly; the synthetic scene requires explicit
# --synthetic. Alternative: feed a pre-built window table:
python rollout2/equivalence.py --vehicles-json table.json
```

The window table for the real gate is built with the registered eval_a2
machinery (`partA_v2g.eval_a.NativeLog` + `tracked_state`, read-only),
identical to `REPORT_PA2G_EVAL_A2.md`.

## Integrator switch (INTERFACES_V2 Mismatch 5)

`partA_v2g/engine.py:661` integrates **ballistically on the new speed**
(`s_new = s + v_new*dt`, the measured SUMO form); the Idea A default D1
rule (frozen in `rollout2.validity.integrate_d1`) is the **trapezoid on
speed** (`v_new = max(0, v + a*dt); s_new = s + 0.5*(v + v_new)*dt`).
The per-step difference is

```
s_d1 - s_v2g = 0.5 * (v - v_new) * dt
```

(<= 0 when accelerating, >= 0 when braking; bounded by 0.375 m per step
at the a-clip and dt = 0.5). `Rollout2Config.integrator` selects the
rule: `"d1"` (Idea A default, with the same-lane gap projection) and
`"v2g"` (like-for-like G1 mode: ballistic integration, gap repair
skipped — the v2g engine never repairs overlaps, it only counts them).
Everything else (IDM/MOBIL/merge rules, the CAV execution rule, the
accel-lane end guard, collision counting, exits) is identical in both
modes and exactly mirrors `partA_v2g.engine.step`.

Measured on the labelled fallback scene (residual 0, arrivals off,
240 ticks): max single-step |Δs| = **0.296 m** at the first tick
(theoretical bound 0.375 m); over the 120 s window the feedback through
car-following gaps accumulates to mean |Δs| = **0.49 m**, max
|Δs| = **2.46 m** across present vehicle-steps. This is the documented,
expected divergence between the two integrators — it is why G1 compares
in `"v2g"` mode.

## Faithfulness contract (pinned by tests)

`rollout2/tests/test_engine_parity.py` pins, on synthetic scenes
covering free-road, merge-zone, discretionary-LC, dense and end-clamp
regimes:

* one-step batched == scalar v2g **exactly** (0.0 error, all rows);
* multi-step zero-residual/arrivals-off equivalence in `"v2g"` mode
  (120 ticks) — 0.0 error, identical collision and anomaly counts;
* replayed CAV commands into both engines (the
  `ctrl.execution_speed` port, engine.py:649-658) — 0.0 error;
* the `d1`/`v2g` integrator difference equals the documented formula;
* safety gates are never residual-adjustable (the P2 residual only
  shifts MOBIL incentive margins, on HDV slots only);
* teacher-forced mode integrates FROM the recorded CAV state each tick;
* K-sample rollouts are deterministic under a seed.

`WD_FORBID_D=1` pytest (43 passed):

```
$env:WD_FORBID_D='1'; python -m pytest rollout2/tests -q
..........................................                        [100%]
43 passed in 12.03s
```
