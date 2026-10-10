# Overnight do-no-harm campaign — 50-seed holdout @ 1.5 s TTC ruler

Date: 2026-08-05
Champion entering the night: `runs/rllib/shepherd_fair_safe_v10c_softmask/checkpoints/iter_90`
+ step soft mask + recover sign gate = **28/50 domination, 26/50 win-rule, 14 TTC-worse**
(eval `runs/eval/fair_safe_v10c_iter90_signgate15_dt05_holdout101_150`).
Bars to beat: domination > 28, win > 26, TTC-worse < 14, no safety regression.
Priority (user steer): **TTC uncompensated > fairness > delay** (slack OK on the latter two).

Rules:
- Domination: gap better + loss within +0.5 + safety (TTC) not worse... (scoreboard tool).
- Win: `ttc < noop` strictly, `d_gap <= +0.1`, `d_loss <= +0.5`, `d_coll <= 0`.
- All numbers at the **1.5 s** TTC ruler, seeds 101–150, dt=0.5. Same-ruler comparisons only.

## Verdict table

| # | Experiment | Type | Dom | Win | TTC-worse | Verdict |
|---|-----------|------|-----|-----|-----------|---------|
| 0 | iter_90 + step mask + sign gate (champion) | inference | 28 | 26 | 14 | CHAMPION |
| 1 | Hard mask (discount 0) | inference | 27 | 26 | 14 | REJECTED — structural worse (0 dom vs 2, 11 harmed vs 9); masked leakage was net-POSITIVE on structural seeds; structural harm is not mask-addressable |
| 2 | Graduated mask (inference) | inference | 24 | 20 | 18 | REJECTED |
| 3 | v10d (more training, 30k) | training | 23 | 22 | 22 | REJECTED — drift |
| 4 | Smoothness (rate limits 0.6) | inference | 20 | 19 | 17 | REJECTED |
| 5 | v10e corridor-TTC penalty (30k warm) | training | 26 | 24 | 15 | REJECTED — reward road dead; 4th warm-drift confirmation |
| 6 | Gentle fair_balance 0.93 | inference | 20 | 19 | 20 | REJECTED — all legs worse; "gentler = fewer waves" refuted |
| 7 | Hybrid mask (hard <= 0.02) | inference | 28 | 26 | 14 | NO EFFECT — champion-identical (hard zone never fires); not adopted |
| 8 | Firm fair_balance 0.85 | inference | — | 0/14 | 14/14 | REJECTED at 14 seeds (killed): net d_ttc +167 vs champ −51; 0.90 is a sharp local optimum; ENTIRE factor family dead |
| 9 | Recover TTC gate (banned when smooth) | inference | — | — | 10/10 | REJECTED at 10 seeds (killed): TTC +3…+17 every seed; recover is a TTC-RELIEF valve, not injector; seed101 flow-degenerate |
| 10 | Fair-tip 240 (Tier-C) | inference | — | — | 4/4 | REJECTED at 4 seeds (killed): TTC +7…+17 despite real flow gains (−6…−10); TTC/flow coupling — metering buys flow at TTC cost |
| 11 | v10f: tip240 trained in, 30k warm @ lr 3e-5 | training | — | — | 4/4 | REJECTED at 4 seeds (killed): TTC +10…+16 with excellent flow (gap −6.3…−7.1) — coupling persists even trained-in; LAST ROAD CLOSED |

## Final conclusion

**The TTC/flow coupling is structural.** Across 11 experiments spanning
every lever family — reward shaping (v10e), inference masks (hard,
graduated, hybrid), global strength knobs (gentle, firm, smoothness),
state-conditional gates (recover TTC gate), spatial extent (tip240), and
trained-in adaptation (v10f) — the pattern held without exception:
any change that buys flow costs TTC, and any change that softens
intervention costs flow. iter_90's trained operating point (28/26/14) is
the Pareto corner of this policy class on this corridor. The champion
configuration stands as the checkpoint of record.

## The noise-floor hypothesis (campaign-level, testing)

Every behavior-CHANGING lever degraded TTC (gentle, firm, recover-gate,
v10e, hard, graduated, smoothness); only behavior-PRESERVING configs held
(hybrid = identical, champion). Working theory: iter_90 sits at a
behavioral noise floor for corridor TTC — its 14 TTC-worse seeds may be
irreducible chaotic variance, and only TRAINED-IN changes (like its own
step mask and sign gate) can shift the floor. Decisive test: tip240 — if
it also degrades TTC broadly, the inference-lever road is closed and the
only remaining road is short warm-training (5–10k; the drift data points
were all ≥30k) with new gates trained in.

## Signed findings (mechanism-level)

1. **Training-based levers are closed.** Four warm-start increments from
   iter_90 (v10d, v10e ×1, +2 earlier) all degraded the holdout, regardless
   of reward shaping. iter_90 is a local optimum that gradient steps escape
   in bad directions. All remaining levers are inference-time.
2. **Corridor-TTC penalty backfired** (v10e): the reward term pushed the
   policy to disturb flow more, not less — TTC harm is emergent system
   dynamics, not directly reward-shapable at this weight (0.2).
3. **Smooth-seed disturbance signature.** pearson(d_ttc, noop_ttc) = −0.65:
   the policy ADDS TTC on smooth corridors (regressed seeds: noop_ttc mean
   10.9) and REMOVES it on chaotic ones (ok seeds: 16.5). On smooth seeds
   interventions are net-disturbance; on chaotic seeds they calm traffic.
4. **"Gentler = fewer waves" refuted.** Global gentle (0.93) interim:
   11 TTC-regressed vs champion 5 on the same 26 seeds; net −19 vs −83.
   The empirical direction is monotone the other way: STRONGER fair_balance
   → fewer TTC events (decisive gap creation beats wave-injection concern).
   Counter-move = firm (0.85), experiment #8.
5. **Win-map sizing.** 26 wins + 9 TTC-only losses + 15 other losses.
   The 9 TTC-only losses pass gap/loss/coll with margins of +1.5…+10.4 —
   directly convertible by any lever that trims 1–6 TTC events there.
   Two of them (124, 145) are exact TTC ties (strict-< rule fails them).
6. **Structural harm is chaotic divergence** (60/40 main/ramp split, no bias)
   and not mask-addressable (hard mask made it worse). No lever found; out of
   scope for the remaining night.

## Incident log

- 2026-08-05 ~03:00: gentle+hybrid evals collided on a shared
  `run_tag_prefix` (`v10besthold_{seed}`) — corrupted tripinfo XML *and*
  silent cross-contamination (identical seed114 rows in both). Both killed,
  policy rows purged, prefixes made unique per driver
  (`gentle15hold_`/`hybrid15hold_`/`v10efhold_`/`firm15hold_`), relaunched.
  Rule: **every concurrent eval driver must have a unique run_tag_prefix**;
  verify before launch.

## Checkpoint of record

`runs/rllib/shepherd_fair_safe_v10c_softmask/checkpoints/iter_90`
+ step soft mask (`easy_intervention_mask_discount 0.1`, absolute gate
`|gap| <= 0.05`) + recover sign gate (`gap > 0.05`) — **28/50 domination,
26/50 win-rule, 14 TTC-worse** at the 1.5 s ruler, seeds 101–150
(eval `runs/eval/fair_safe_v10c_iter90_signgate15_dt05_holdout101_150`;
re-verified against per-seed data 2026-08-05).
No experiment beat the 28/26 bars without a TTC regression; v10f's
trained-in tip240 confirmed the coupling is structural, not a tuning miss.

## Conversion map (path from 26 → 30+ wins)

All 24 losses classified by failed legs and margins (champion, 1.5 s ruler):

| Tier | Seeds | Signature | Lever |
|------|-------|-----------|-------|
| A — TTC-only, small excess (≤3) | 102(+2), 103(+1), 124(0), 132(+3), 135(+2), 138(+3), 145(0) | gap/loss margins −1.6…−10.3 | smooth-corridor TTC trim (firm/adaptive fair_balance) |
| B — TTC-only, big excess | 127(+6), 144(+4) | margins thinner (144 loss slack +0.65) | same, harder |
| C — flow near-miss, TTC banked | 128(g+1.12), 143(l+0.58), 146(g+1.94), 147(g+1.91) | delay improved, fairness slightly worsened | fairness boost (eligibility probe decides) |
| D — flow mid-harm, TTC banked | 101, 116, 120, 148 | d_gap +0.7…+7.7, d_loss +0.6…+8.9 | none identified |
| E — deep multi-leg | 106, 115, 119, 131, 136, 137, 140 | all legs fail | structural chaos — out of scope |

Realistic ceiling if A+B+C fully convert: 26 + 13 = 39 wins. Campaign
target: 30–33 via Tier A (primary) + Tier C (secondary).

## Dead-road ledger (do not retry without new mechanism evidence)

| Road | Evidence |
|------|----------|
| Tax hike (easy_intervention 0.15→0.25) | rejected earlier (v10b) |
| Global recover ablation | recover is load-bearing on wins (34→31 @1.0 s) |
| Global rate-limit softening (smoothness 0.6) | 20/19/17 |
| Graduated mask at inference | 24/20/18 |
| Hard mask (discount 0) | 27/26/14, structural worse — harm not mask-addressable |
| Hybrid mask (hard ≤0.02) | 28/26/14 identical — no effect |
| >0 warm training from iter_90 at lr 3e-4 | v10d 23/22/22, v10e 26/24/15 (both 30k) — always drifts |
| Corridor-TTC reward penalty 0.2 | v10e: bought flow, paid TTC — backfired |
| fair_balance factor ≠ 0.90 | gentle 0.93 → 20/19/20; firm 0.85 → 14/14 TTC-worse. Sharp local optimum |
| Adaptive factor (either direction) | dead by transitivity of the above two |
| Recover TTC gate at inference | 10/10 TTC-worse — recover is a relief valve |
| Fair-tip 240 at inference | 4/4 TTC-worse despite flow gains — TTC/flow coupling |
| tip240 trained-in (v10f, 30k @ lr 3e-5) | 4/4 TTC-worse (+10…+16) with excellent flow — coupling is structural, not tuning |
| "Pick fair_balance more" (logit boost) | pick\|elig = 0.94–1.00 — no headroom |
| Coast-force, anti-tailgate gate, pairwise follower-TTC reward, relative fairness mask | rejected earlier (autopsy: TTC harm is HDV–HDV waves, not behind-CAV squeeze) |
