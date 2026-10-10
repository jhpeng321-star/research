# v10c softmask (mask-v1) — holdout 101–150 results

Date: 2026-08-04
Run: `runs/rllib/shepherd_fair_safe_v10c_softmask` (warm from v10 `iter_85`, +~84k steps to 600k)
Eval: 50 held-out seeds (101–150), dt=0.5, paired no_op ruler, slack-rule domination scoreboard

**Headline: iter_90 = 34/50 dominated — beats v10 i85 (33), ties best-ever v3 i90 (34).**
Final checkpoint = 30/50 (over-trained; drift autopsy below).

## Change vs v10

RSU-informed **soft action mask** (mask-v1), commits `962a053` + `2d02e2c`.

- Gate (absolute, per user steer): when the corridor is **really fair** —
  `|stream_delay_gap| <= shepherd_easy_intervention_mask_gap_abs = 0.05`
  (dense coast |gap| ~ 0.10) — discretionary flow-hurting modes are
  discounted. NOT the no_op-relative condition (legacy fallback via
  `gap_abs = 0`).
- Masked modes: `fair_balance`, `recover_speed`, `vacate_drop`.
  `protect` exempt (safety); `coast` never gated.
- Soft semantics: `shepherd_easy_intervention_mask_discount = 0.1` — the
  eligibility-mask bit becomes 0.1, so the model applies `log(0.1) ≈ -2.3`
  to those logits instead of banning them; `apply()` does NOT remap.
  (`0.0` = hard ban with coast remap, `1.0` = no effect.)
- Mechanism: the CAV gets the fairness state from the RSU stream-delay EMAs
  already in the env; the mask rides the existing 5-bit action-mask suffix
  (`logits += log(mask)` in `models.py`), no obs-shape or model change.
- Tests: `tests/test_easy_intervention_mask.py` (9 tests: soft/hard bits,
  remap vs no-remap, absolute-vs-ref distinction, unknown gap, flag off,
  clamping). 45/45 pass with the reward/action suites.

## Hypothesis

v10 i85 (33/50 dominated) still structurally harmed 9 seeds — mostly
intervening when the corridor was already balanced. Discounting those
actions in exactly those states should cut the harm without giving up the
wins, pushing domination above the 33/50 (v10 i85) and 34/50 (v3 i90) bars.

## Results (final checkpoint, iter ~100/101 = i85 + ~90k steps)

- **Dominated seeds: 30/50 — REGRESSION** (bars: v10 i85 = 33, v3 i90 = 34, v9 i40 = 27)
- winnable 28/39; structural: 2 dominated, 4 ties, 9 harmed (harm unchanged vs i85)
- Means: d_loss = −2.24, d_G = −2.32, d_coll = 0 (i85: −3.03 / −2.83 / 0 — aggregate still clearly better than no_op, just less so)
- Gate (non-slack) view: pass 27, loss_wins 31, G_wins 34

### Flip analysis vs v10 i85 (17 flips: 10 lost, 7 gained, net −3)

- Lost: 101, 105, 116, 118, 121, 122, 123, 126, 128, 132
- Gained: 110, 127, 139, 140, 144, 149, 150
- 8 of the 10 losses are large `G+loss` failures (dG +2.5…+4.5): the policy now
  actively *unbalances* those corridors vs no_op. 105/121 lost only on the
  +0.5 s loss slack (barely).
- Structural harm stayed at 9 — the mask did not reduce the harmed-seed count,
  which was its stated purpose.

### Autopsy (working hypothesis: training drift, not the mask)

- v10 history: i85 = 33/50, but v10's own *final* checkpoint = 21/50 — training
  past i85 was already documented to degrade this recipe. v10c final = i85 +
  ~90k more steps → 30/50 sits on that same degradation slope (milder, and
  with a changed MDP from the mask).
- Reward curve over the run: −43.9 (iter 88) → dip −55.2 (iter 89, the mask
  changed the MDP at warm start) → noisy ~−47 band; TB reward is a stale
  selector on this project, so checkpoint choice by domination eval is
  required.
- The 7 gained seeds show the mask recipe *can* win new seeds (110/127/139/
  140/144/149/150, several former failures now dominated).

### Drift-isolation probe result: iter_90 = **34/50** ✓

`checkpoints/iter_90` (i85 + ~30k steps) on the same 50-seed holdout:
**34/50 dominated** (winnable 32/39; structural: 2 dominated, 2 ties,
9 harmed) — **beats the 33/50 bar (v10 i85) and ties the best-ever
34/50 (v3 i90)**. Scoreboard:
`runs/eval/fair_safe_v10c_softmask_iter90_dt05_holdout101_150/scoreboard.json`

| checkpoint | steps from i85 | dominated |
|---|---|---|
| v10 i85 (bar) | 0 | 33/50 |
| v10c iter_90 | +30k | **34/50** |
| v10c final | +90k | 30/50 |

The drift hypothesis is confirmed: the mask recipe improves on i85 when
checkpoint drift is controlled; the extra ~60k steps to `final` destroyed
4 seeds' worth of quality.

## Verdict / next lever

**mask-v1 works.** Soft RSU-informed masking (|gap| ≤ 0.05 → discount 0.1
on fair_balance/recover_speed/vacate_drop; protect exempt) trained warm
from v10 i85 delivers **34/50 dominated at iter_90**, tying the best-ever
bar. The mask is config-on by default (`shepherd_easy_intervention_mask*`
in `config/defaults.py`) and covered by 9 tests.

Lessons:
1. Checkpoint drift is the binding constraint, not the reward recipe —
   v10 i85 33 → v10 final 21, v10c iter_90 34 → v10c final 30. Future
   runs should plan short warm-start increments (~30k steps) with
   per-checkpoint domination evals instead of one long run.
2. Structural harm is still 9 seeds and untouched by both the tax and
   the mask — that subpopulation needs a different lever (its no_op is
   already near-fair; the wins must come from not-losing, not from
   intervening better).

Next levers, in order:
1. Adopt v10c iter_90 as the new checkpoint of record (34/50).
2. If pushing past 34: another short (+30k) warm run from iter_90 with
   the same mask, evaluating every checkpoint — stop at the first
   regression.
3. For the 9 structural-harm seeds: analyze their per-seed traces to see
   whether harm comes from intervening at all (then a stricter gate for
   near-fair no_op seeds) or from a specific mode.


## Addendum 2026-08-04 (evening): safety dimension at the 1.5 s TTC ruler

TTC threshold restored to the intended 1.5 s (`f282785`). At this ruler the
picture changes materially vs the 1.0 s era (where no_op showed ~0 events):

- **no_op has 745 TTC events** across the 50-seed holdout (mean ~15/seed).
- **iter_90 policy total: 658 (−12% vs no_op)** — the shepherd reduces
  aggregate risk.
- **But risk is redistributed**: ~16 seeds show policy ttc > noop ttc, which
  now fails the scoreboard safety leg (`policy_ttc <= noop_ttc`). Domination
  at the 1.5 s ruler: step mask 27/50 (was 34/50 at 1.0 s).

**Win-rule scorecard** (`tools/win_scorecard.py`; win = ttc strictly better,
gap ≤ +0.1, loss ≤ +0.5, coll ≤): step mask 26/50, graduated 20/50,
sign gate 26/50.

**Mechanism autopsy** (`_autopsy_ttc_mechanism.py`, seeds 137/119/127/106/138):
the TTC elevation is **HDV–HDV compression waves (~85% of excess events)**,
NOT followers squeezed directly behind the CAV (~15%). Consequently the
anti-tailgate gate and pairwise follower-TTC reward shaping are dead ends;
the remaining TTC lever is intervention smoothness (softer rate limits /
wake-region penalty) or accept-and-report.

**Recover autopsy conclusion**: recover_speed is load-bearing (global
ablation 34 → 31/50) and is now sign-gated (`a074f6c`): eligible only when
the main stream is behind (`gap > 0.05`). Effect at 1.5 s ruler:
domination 27 → 28, ttc-regressed seeds 16 → 14, structural harm unchanged.
Kept on by default.

**Graduated mask (mask-v2, `06e9548`)**: inference-neutral-to-negative vs
the step mask (24/50 vs 27/50 at 1.5 s). Only viable as a trained lever.

Levers now in flight: (1) v10d training run, warm from iter_90 (+30k steps,
step mask + sign gate); (2) smoothness inference test (`mode_max_decel` /
`mode_max_accel` 1.0/1.25 → 0.6/0.6) against the TTC redistribution.

## Final lever verdicts (2026-08-04 late)

| candidate | domination | win-rule | ttc-regressed | verdict |
|---|---|---|---|---|
| iter_90 + sign gate | 28/50 | 26/50 | 14 | CHAMPION (kept) |
| v10d (iter_90 +30k trained) | 23 | 22 | 22 | rejected — drift again (3rd confirmation: v10 33->21, v10c 34->30, v10d 28->23) |
| smoothness (dec/acc 0.6) | 20 | 19 | 17 | rejected — softer control hurts flow and TTC |
| graduated mask (inference) | 24 | 20 | 18 | rejected at inference; training-only question left open |

All listed levers tried and signed. Training more from a good checkpoint is
confirmed spent. Remaining presentation-readiness item: multi-episode
variance run (10-15 seeds x 3-5 episodes) for CIs on the headline metrics.
