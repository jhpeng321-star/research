# Part B input leak audit (V0 featurizer; applies to v0.2 net, ridge, GBM, hybrid)

2026-09-29, world model session. Source: `partB_v0data/featurize_v0.py`
(`_demand_vector`, `_context_vector`, `featurize_at`) and the diag/hybrid
summary features, which reuse the same blocks.

| input | content | knowable by a real RSU planner at t? | verdict |
| --- | --- | --- | --- |
| `demand[0:20]` | exact scheduled arrivals per minute per entrance over (t, t+600] | no | **LEAK (major).** Ridge ablation: dropping it doubles long-τ MAE (τ=300: 1361→2694; τ=570: 2989→6450). |
| `demand[20]` | CAV share of the *future* scheduled arrivals | only as the configured penetration | **LEAK (minor).** Replace with the configured or estimated penetration. |
| `context[0]` t/1000 | absolute simulation time | yes (clock) | ok, but a proxy for the demand ramp; keep |
| `context[1]` (t − t0)/600 | time since the campaign branch point | yes (the planner knows when it switched) | ok |
| `context[2]` time-to-branch/600 (base rows only) | time until a FUTURE design event | no: a campaign artefact | **spurious future info.** Set to 0 in deployable. |
| `context[3]` loading/2700 | configured demand level of the world | as a demand rate estimate | ok (rate, not realised) |
| `context[6:8]` q_main, q_ramp | vehicles scheduled but not yet inserted (entrance queues) | only with upstream / entrance detection | ok under the "ideal sensing" owner decision if RSU sees the queues; flag for deployment |
| `lane_bins`, `cav_tokens` | full current vehicle state | ideal sensing (owner decision); RG-0 measures the reconstruction gap | ok |
| τ, policy, pre_fix | conditioning | yes | ok |

## Labels

The labels include entry waiting derived from the scheduled departures
(contract §4.1). That is correct as a *target*; it is not an input.

## Deployable feature set (v0.3.1)

The following leaking or spurious inputs are dropped:

- the future schedule (`demand[0:20]`);
- the future CAV share (`demand[20]`);
- time-to-branch (`context[2]`).

They are replaced by:

- the demand rate (configured loading);
- **recent observed entrance counts**: realised insertions (or scheduled
  arrivals that have already happened) per entrance over (t−60, t],
  (t−180, t] and (t−300, t];
- the configured penetration.

All else is unchanged. Acceptance is measured against a ridge given the SAME
deployable inputs.
