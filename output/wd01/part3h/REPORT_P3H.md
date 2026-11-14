# WD-01 Part 3H — gate v2 criterion (c) + M=5 ensemble eval

**HOLDOUT NOT ACCESSED.**  4 validation roots × t0 {120,150,...,480} =
52 windows × 3 variants (L0/L1/L3) × M=5 members = 780 member-window
rollouts.  Raw metrics saved incrementally (2288 rows in
results/raw/raw_metrics.jsonl).

## Gate v2 verdict at H=60

| criterion | threshold | L0 | L1 | L3 | verdict |
| --- | --- | --- | --- | --- | --- |
| (a) collisions/1000 | ≤ 1.0 | 0.0 ✅ | 0.008 ✅ | 0.008 ✅ | **PASS** |
| (b) LC within ±20% | truth 23.9 | 0 ❌ | 29.3 (22.4%) ❌ | 29.3 (22.4%) ❌ | **FAIL** |
| (c) delay band coverage ≥80% | — | n/a | 0% ❌ | 0% ❌ | **FAIL** |
| (c) TTC<3 band coverage ≥80% | — | n/a | 55.8% ❌ | 55.8% ❌ | **FAIL** |
| (c) CRPS(delay) < L0 MAE(1502) | < 1502 | n/a | — | 318.5 ✅ | **PASS** |
| (d) speed RMSE ≤ 5.22 | ≤ 5.22 | 4.98 ✅ | 4.69 ✅ | 4.69 ✅ | **PASS** |

**GATE v2: FAIL** — (a) and (d) pass, but (b) LC is 22.4% over (barely
above the ±20% threshold), and (c) delay band coverage is 0% because
all members produce very similar delay proxies (IDM-based emulators are
physically constrained).

## Old gate verdict at H=60

| criterion | threshold | L1 | L3 | verdict |
| --- | --- | --- | --- | --- |
| speed RMSE < 4.97 | < 4.97 | 4.69 ✅ | 4.69 ✅ | **PASS** |
| collisions ≤ 1/1000 | ≤ 1 | 0.008 ✅ | 0.008 ✅ | **PASS** |
| LC within ±20% | ±20% | 22.4% ❌ | 22.4% ❌ | **FAIL** |
| delay min-max coverage ≥50% | ≥ 50% | 100% ✅ | 100% ✅ | **PASS** |

**OLD GATE: 3/4 criteria pass; LC count is the only failure.**

## Coverage detail at H=60

| variant | delay 10-90% coverage | delay min-max coverage | TTC<3 10-90% coverage | TTC<3 min-max coverage |
| --- | --- | --- | --- | --- |
| L1 | 0% | 100% | 56% | 100% |
| L3 | 0% | 100% | 56% | 100% |

The 0% delay 10-90% coverage is because all 5 members produce very
similar delay proxies (they share the same physics and only differ in
CRN noise streams).  The delay proxy is a SUM of per-vehicle times, so
it concentrates tightly.  The min-max coverage is 100% (trivially, any
value is in the min-max range).  This is a **diversity problem**: the
members need more diverse vType distributions or LC histories to
produce meaningful spread.

## CRPS(delay) comparison

- L3 ensemble CRPS mean = 318.5 s
- L0 deterministic delay MAE = 1502.0 s
- CRPS << MAE → the ensemble CRPS passes (the ensemble is more accurate
  than a point forecast from L0).

## Key findings

1. **Collisions are essentially zero** (0.008/1000) for all structured
   rungs — the IDM prior + dawdle noise + LC hazard produce physically
   plausible trajectories.
2. **Speed RMSE 4.69 m/s at H=60** passes the old gate (< 4.97) and
   gate v2 (d) (≤ 5.22).
3. **LC count 29.3 vs truth 23.9** (22.4% over) — marginally outside
   the ±20% gate.  The LC hazard is slightly over-firing but in the
   right ballpark.
4. **Delay proxy bias is positive** (ensemble drives slower than
   truth), with L0 MAE = 1502 s and L3 ensemble CRPS = 318.5 s.
5. **Member spread is minimal** for delay proxy (all members share the
   same physics), making the 10-90% band coverage criterion moot —
   the gate v2 (c) criterion is not achievable for delay proxy with
   IDM-based emulators that share the same physics.
