# Campaign 3: hybrid (mode, magnitude) action space

Date: 2026-08-05
Champion to beat: `shepherd_fair_safe_v10c_softmask/iter_90` + step mask +
sign gate = **28/50 domination, 26/50 win-rule, 14 TTC-worse** (1.5 s ruler).

## Design evidence (autopsy first, per user steer)

Action-wake probe (champion, 6 seeds, ~2400 interventions logged): 28% of
fair_balance firings leave a +1 corridor-TTC wake, 72% none. **Triple
null**: neither follower gap (96.9m vs 100.3m), corridor state, ego speed
(14.6 vs 13.9 m/s), lane position (311 vs 325m), leader geometry,
downstream speed, edge, nor episode time separates the two groups. The
wake is stochastic/systemic — no per-step observable can trigger
context-dependent execution. Therefore the magnitude dial cannot be
rule-based; it must be LEARNED, and the reward must price gentleness so
PPO can find where strength pays.

## Architecture (v3, final form)

Composite distribution per CAV: **Categorical(5 modes) × Gaussian(dial)**,
sampled model-side. The eligibility mask (identical to the champion's)
applies to the mode logits before sampling (`logits += log(mask)`) — no
env-side argmax, no gradient mismatch. The dial (raw [-1,1] → [0,1]) maps
to per-mode execution strength: fair_balance factor [0.85, 1.0], recover
push [0, 2.0], protect [0.85, 1.0], vacate soft [0.90, 1.0]; dial < 0.05
releases control. Intervention costs scale by the dial with a 0.3 floor.
Both levels learned jointly by one network; the champion's gates (sign
gate, eligibility, mask semantics) are preserved untouched.

## Version history (each a real experiment)

| Ver | Design | Outcome |
|-----|--------|---------|
| v1 | Gaussian scores + env argmax; mag-scaled costs, no floor | iter_25 smoke: universal harm (gap +8…+24). Flaw: dial init 0.5 → interventions pay half cost → coast pressure halved |
| v2 | + cost floor 0.3, gentle dial init −0.7 | iter_25 smoke: universal harm again. Deeper flaw found: env-side argmax breaks PPO gradient (executed action ~ independent of logged action) |
| v3 | composite Categorical×Gaussian (the fix) | Learned to coast perfectly (12/12 do-no-harm, dL/dG = 0.00) but **dial collapsed to ~0** (raw mean −0.999): a perfect abstainer. Never harms, never helps, TTC +13 from faint leakages. Bottom-up gradient can't open the dial (opening pays cost+TTC now, benefit delayed) |
| v4 | top-down dial init (+0.5, start open) | iter_25 smoke: **real flow** (8/12 pass, 9/12 loss wins, dL −0.87, dG −1.80 — first from-scratch flow). iter_50: degraded (5/12, dL −0.21) — third peak-then-drift arc. Full 50-seed eval of iter_25 peak: [pending] |

## Findings

1. **Mode selection is learnable; the dial's learning signal is weak.**
   The composite architecture fixes selection (v3's clean coast, v4's real
   metering). But a dial step changes a slowdown by a few percent whose
   corridor effect takes 10–30 s and is buried in chaotic variance — the
   PPO advantage for the dial is near-zero-mean noise. v3 shows the
   collapse (to 0), v4 shows the open-dial behavior it enables.
2. **Every fresh run peaks at ~25% and degrades** — B, B-nopen, and now
   v4 (8/12 → 5/12 pass). The champion's TTC-efficiency at convergence is
   not reproduced by any fresh trajectory tonight.
3. **The coupling held even with a working hybrid**: v4 buys real flow at
   TTC +13.6/seed at its peak — fatal under the win rule (TTC strictly
   better required).

## Verdict

**The hybrid does not beat the champion.** v4's best checkpoint (iter_25
peak) on the full 50-seed holdout: **0/50 domination, 0/50 win-rule,
50/50 TTC-worse** — real flow gains (dL/dG negative on a third of the
seeds at champion-comparable levels) rendered worthless by TTC regression
on every seed. v3 (bottom-up) produced a harmless but useless abstainer;
v4 (top-down) produced real metering with fatal TTC cost. The composite
Categorical×Gaussian architecture itself is correct and learnable
(selection works, dial responds to init direction) — but no variant
escaped the TTC/flow coupling tonight. The remaining untried dial fixes
(discretized strengths, frozen-selection dial training, auxiliary
short-horizon signal) are documented for future work.

## Checkpoint of record

Unchanged: `shepherd_fair_safe_v10c_softmask/iter_90` + step mask + sign
gate (28/50 domination, 26/50 win-rule, 14 TTC-worse).
