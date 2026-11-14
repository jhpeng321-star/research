# Fixed-P30 Limited-Domain Screen Results

**Date:** 2026-08-30
**Status:** development screen complete on already-opened development data;
**no learned activation supported**; fixed 5–10% rule frozen as the strongest
honest decentralized damping arm (pending prospective acceptance on untouched
seeds). No RL training authorized (the frozen value gates failed).
**Protocol:** `docs/IDM_FIXED_P30_LIMITED_DOMAIN_PROTOCOL.md`
**Artifact:** `runs/eval/idm_limited_domain_p30_screen_v1/summary.json`
(tool `tools/screen_idm_limited_domain_p30.py`, 3 focused tests)
**Dataset:** `runs/eval/idm_paired_option_dataset_l3_six_pen_v1`
(178 paired P30 decisions; this screen uses the 42 in-domain records)

## 1. Domain dataset (5–10% penetration, demands 1900/2000/2100 veh/h)

- Records: 42 (penetration 0.05: 19, 0.10: 23; demand 1900: 12, 2000: 15,
  2100: 15; train seeds 4121–4128: 27; one-time held 4129–4132: 15).
- Frozen classification counts: helpful 20, harmful 11, neutral 11,
  constraint-failed 2. No decision was discarded (all 42 retained; 0
  no-decision in the source for these cells).
- Held 15 in-domain rows: 7 helpful, 3 harmful, 5 neutral; the fixed
  domain rule (always activate within 5–10%) achieves held mean paired reward
  **+0.04343**, harmful fraction 20%.

## 2. Screen A — outcome classification (diagnostic)

Helpful versus harmful among qualified activations (helpful 7 / harmful 3 on
held; train qualified 19). Logistic regression, alpha chosen from training
seed-grouped OOF only, held opened once.

| family | held AUROC | held bal-acc | held AP | held n | train OOF AUROC |
|---|---|---|---|---|---|
| option_context_12 | 0.667 | 0.714 | 0.863 | 10 | 0.167 |
| zone_controller_option_37 | 0.762 | 0.786 | 0.913 | 10 | 0.410 |
| physical_134 | 0.286 | 0.571 | 0.680 | 10 | 0.372 |
| physical_plus_option_146 | 0.429 | 0.571 | 0.738 | 10 | 0.308 |

Diagnostic verdict: the held AUROC/balanced-accuracy values that nominally
reach the 0.70/0.65 targets are **not credible evidence**. The held sample has
only 7 positive / 3 negative rows (n = 10); the training grouped-out-of-fold
AUROC for the same family is ≈ 0.41 (at or below chance), so the held result
is small-sample selection noise rather than a learned signal. AUROC is
diagnostic evidence only under the frozen protocol and does not authorize
training.

## 3. Screen B — conditional expected option value (primary authorization question)

Ridge regression on causal, penetration-free features, alpha + threshold
selected on seed-grouped training OOF (seeds 4121–4128), held opened once
(4129–4132). Penetration is the operating-domain admission rule outside the
actor, so all in-domain records are governed by the domain rule first.

| family | selected (alpha, thr) | held coverage | held mean reward | held helpful/harmful/neutral | harmful frac | passes all gates |
|---|---|---|---|---|---|---|
| option_context_12 | (10.0, −0.0062) | 66.7% | +0.0429 | 5/1/4 | 10% | **no** (not above always-activate; margin −0.0005 vs rule) |
| zone_controller_option_37 | (100.0, −0.0128) | 46.7% | +0.0345 | 3/1/3 | 14% | **no** (below rule by −0.0089) |
| physical_134 | (1.0, 0.1572) | 26.7% | −0.0013 | 2/2/0 | 50% | **no** |
| physical_plus_option_146 | (10.0, 0.0486) | 40.0% | +0.0111 | 3/2/1 | 33% | **no** |

Per-demand held results (option_context_12, best family): 1900 +0.0197,
2000 −0.0001, 2100 +0.1092 — i.e., even the best family is not positive in
more than one profitable demand cell (it is negative in 2000 veh/h), and its
overall mean (+0.0429) is below the fixed rule (+0.04343) by −0.0005, far
below the frozen +0.01 predeclared margin.

## 4. Frozen authorization conclusion

No causal deployable observation model beats the fixed 5–10% penetration rule
on held development groups. The frozen gates (activation ≥ 20%, positive
conditional mean reward, helpful > harmful, harmful ≤ 25%, value above never
activate / always activate / the rule by ≥ +0.01, positive in more than one
demand cell, no added safety/service violations) are **not all met by any
family** for two reasons: (a) nothing exceeds the rule at all; (b) the best
family fails the multi-cell positivity requirement at 2000 veh/h. The only
family with no constraint failure among its activated held rows
(option_context_12) still loses to the rule.

**Decision per the frozen protocol failure path (§9): do not force PPO. The
fixed 5–10% penetration rule is frozen as the strongest honest decentralized
damping arm. Learned activation was not supported by causal observations in
the supported domain.** A future observation/value screen would require
substantial new causal evidence (more development decisions or a changed
observation contract), which is out of scope without a new versioned protocol.

## 5. What was not done

No reward audit was needed (the value gate failed first per the protocol
ordering); no training smoke or campaign launched; no prospective seed
consumed; no threshold changed after viewing held results
(selection was frozen on training OOF before opening 4129–4132); no
classification label changed; no harmful or neutral case discarded. All
screen results are development-only.
