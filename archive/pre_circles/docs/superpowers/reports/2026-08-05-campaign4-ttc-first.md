# Campaign 4: TTC-first reward pivot

Date: 2026-08-05
Base: champion `shepherd_fair_safe_v10c_softmask/iter_90` + step mask +
sign gate = **28/50 domination, 26/50 win-rule, 14 TTC-worse**.
User steer: meet TTC hard first, fairness second; delay improvement wanted
but only needs to avoid large regression. All runs: warm from iter_90,
+30k at lr 3e-5 (the only non-drifting warm recipe), corridor_ttc_penalty
OFF (signed 0-for-3 as a training term), full 50-seed eval under both
rules. Discrete champion stack only (hybrid shelved by user steer).

## Verdicts

| Run | Reward change | Dom | Win | TTC-worse | Notes |
|-----|--------------|-----|-----|-----------|-------|
| champion | — | 28 | 26 | 14 | baseline |
| v1 | safety 1.25→3.0, delay_budget 1.10→0.5, flow 1.0→0.6 | 0 | 0 | 50 | SIGNED FAIL: ego-TTC avoidance → more/earlier braking → more HDV-HDV waves. Flow held champion-level (d_gap −2.44, d_loss −2.01) but TTC +14.5 mean on all seeds |
| v2 | stability 0.25→1.5, delay_budget 0.5, flow 0.6 | 0 | 0 | 50 | SIGNED FAIL: flow BETTER than champion (d_gap −2.62, d_loss −2.79) but TTC +13.6 on all seeds. Two reward levers, two TTC failures — TTC is not reward-addressable |
| v3 | wave_damp mode (research_v2 flow_stabilization port, opt-in 6th mode) | — | — | — | SIGNED DEAD: could not be sampled at any incentive (0.0 at 0 / cost-free / +0.08 / +0.3 — trained fair_balance preference unbeatable); v5 (fair_balance disabled) forced 97.5% usage but harmful on all legs (TTC +17, gap +8.4, loss +8.5 means). Dominant = harmful, rare = never sampled. The v2-era wave damper does not transfer to dense p10 |

## Campaign-4 conclusion

Two reward levers (safety, stability) both moved TTC the *wrong* way while
improving flow — the corridor TTC harm is not reward-addressable. The one
mechanism-level tool tried (wave_damp, ported from the old lineage) fails
at dense p10 both by sampling (the trained fair_balance preference never
yields samples, even against mask + cost-free + elevated use-bonus) and by
effect (dominant usage harms all three legs). No TTC-first lever found;
the champion `iter_90` (28/50, 26/50, 14 TTC-worse) stands. The honest
remaining interpretation, from the win-map: the TTC/flow coupling inside
this corridor at this density is the binding constraint of the *scenario*,
and every CAV-reachable lever — reward, mask, gate, action space, or tool —
trades on it rather than moving it.

## The research_v2 finding (with cautions)

The v2-era lineage had a dedicated `flow_stabilization` mode the dense
redesign dropped. Their models posted large TTC cuts — but measured
against **All-HDV** (no CAVs at all), at **medium density**, so the
numbers are NOT comparable to the dense no_op ruler (user caution,
verified: no no_op evals exist in the old projects). The mechanism
stands independent of the baseline question; the port gets its own
honest test at the dense no_op-paired 1.5 s ruler.

## Analysis so far

v1's failure isolates the hard part of the TTC-first pivot: the eval's
TTC ruler counts **corridor** events (HDV-HDV waves), but every
CAV-reachable reward term is **ego**-centric. Raising ego safety makes the
CAV avoid proximity by braking — which *creates* the waves the ruler
counts. The TTC lever must target the wave itself, not the CAV's
exposure to it.

Candidate directions (pre-registered):
- **v2 (running)**: penalize the CAV's own speed discontinuities
  (jerk/stability ×6) — the injection source.
- **v3 (if v2 fails)**: a baseline-relative TTC term (`ttc_win`: bonus
  when the live corridor TTC count is under the *paired no-op's* count
  for that seed — the eval's own ruler, with the baseline solving the
  "chaos is naturally high" problem that made the absolute
  corridor_ttc_penalty hackable).
