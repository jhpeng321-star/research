# v10: Learned Abstention for Per-Seed Domination — Design

Date: 2026-08-04
Status: approved by user (2026-08-04)
Supersedes/extends: `docs/PLAN_V9_IMPROVE_PASS.md` Phase 4

## 1. Objective

Maximize **per-seed strict domination** of the trained policy vs the paired `no_op`
baseline on holdout seeds 101–150 (dt=0.5, p=0.10, dense 3500 veh/h):

> A seed is dominated iff policy `fair_gap` < no_op `fair_gap` − 0.1s
> (strict G margin) AND policy `loss` ≤ no_op `loss` + 0.5s (slack) AND
> collisions ≤ AND ttc ≤.

The user approved this slack version of the rule (0.1s G margin, 0.5s loss
slack) on 2026-08-04, replacing the original eps-strict rule.

Bars to beat (reproduced from existing per-seed JSONs, under the slack rule):

| Checkpoint | Domination (slack rule) |
|---|---|
| v9 iter_40 | 27/50 |
| v3 iter_90 | 34/50 |
| v8 iter_150 | 33/50 |

Stretch target: 39/50 (all winnable seeds). On the 11 structural seeds
(106, 117, 119, 120, 129, 136, 137, 143, 146, 147, 148 — no_op G 0.27–5.0s,
already near-fair) harm-free ties are the accepted ceiling; strict wins there
are noise-level by construction.

## 2. Key findings this design is based on

- Delay never binds when G improves: on all three checkpoints, every seed with
  G improved also has loss ≤ no_op. Safety never binds (coll=0, ttc=0 for both
  policy and no_op on all 50 seeds). **The single problem is losing G on easy
  seeds.**
- Fail mode: policy coasts ~97% everywhere; sparse (~1%) `fair_balance` /
  `protect` actions fire on easy seeds and only add harm (e.g. seed 148:
  no_op G=0.27, dG=+1.09 from ~2% interventions).
- External gates/masks are disproven by experiment: Phase-1 relative gate +
  fairness mask peeked 10/20; v9b/v9c retraining with gates regressed. v8's
  action-mask approach is rejected by the user on principle: the policy must
  see fairness through the merge-zone obs (v9 121-D, 12-D zone features) and
  decide on its own.
- Train-selection mismatch: `best/` stamped by iter 1–5 TB reward does not
  track the scorecard. Selection must use the domination scoreboard only.
- Where the policy wins, wins are large: median G reduction ≈ 4.8s
  (~65–70% of no_op gap), max ~17s. Preserving these wins is as important as
  fixing the easy-seed losses.

## 3. Non-goals

- No inference-time action mask or hard gate on the deployed policy.
- No warm-start from v3/v8 (117-D obs, incompatible with v9's 121-D).
- No changes to the delay do-no-harm term (one-sided delay_pen 1.10) — it works.
- No demand/scenario changes; hyper 4300 veh/h remains out of scope.

## 4. Components

### 4.1 Domination scoreboard tool (`tools/domination_scoreboard.py`)

Single source of truth for selection and reporting.

- Input: an eval dir with per-seed policy JSONs + paired no_op JSONs
  (layout of `runs/eval/fair_safe_*_dt05_holdout101_150/`), or a peek dir.
- Output: strict-domination count, per-seed table (dG, dLoss, dColl, dTTC,
  dominate/tie/fail), split of fail causes (G-only / loss-only / both),
  structural-seed breakdown (winnable 39 vs structural 11).
- Must reproduce the known bars 27/34/33 (slack rule, approved 2026-08-04)
  on existing eval dirs as its acceptance test.
- Note: trust per-seed JSONs / summary.json, not eval.log tails
  (known 27-vs-28 discrepancy on v9 i40).

### 4.2 Reward v10 (`env/rewards.py`, weights in `config/defaults.py`)

v9 reward plus two additions; all other v9 weights unchanged
(safe 1.25, fair 1.20, delay_pen 1.10, delay_win 0.30, eff 0.35, stab 0.25,
ctrl tax 0.25, inelig/mask 0.5/0.2, mode_use_bonus 0.08):

1. **fair_win term** (weight 0.20, per PLAN_V9 Phase 4): pays only for
   *strict* reduction of |G| below the paired no_op ref
   (`(G_ref − |G|)+ / κ`, clipped to [0,1]) — the mirror image of the
   existing excess-gap penalty. No payment for ties.
2. **easy-state intervention tax** (weight 0.15, tunable 0.10–0.25): when
   |G| ≤ `stream_delay_gap_ref` (nothing to win), any non-coast action incurs
   a fixed penalty. This is a *soft reward signal*, not a mask — the policy
   remains free to act and learns that abstention dominates on easy states.
   The Phase-1 relative gate / fairness mask stays disabled for v10 training.

### 4.3 Training curriculum — easy-seed oversampling (`train.py` / paired pool)

- Oversample low-G_ref / easy paired-pool episodes so easy states are common
  enough in training for abstention to be learnable (currently rare → ~1%
  rogue interventions survive). Target: easy-pool episodes ≈ 30–40% of
  sampled episodes (exact mechanism follows the existing paired-pool sampling
  code; ratio exposed as a config knob).
- Warm-start from v9 iter_40 weights (121-D compatible).
- Peek protocol unchanged: eval seeds 101–120 every ~10 iters; promote to
  full holdout (101–150) only when peek strict-domination improves over the
  v9-i40 peek baseline (11/20). Checkpoint selection by scoreboard only,
  never by TB reward.

### 4.4 Evaluation & reporting

- Full-holdout eval of the selected checkpoint with
  `tools/domination_scoreboard.py`; compare vs 27/34/33 bars (slack rule).
- Report: strict domination count, per-seed table, fail-cause split,
  structural-seed behavior (tie vs harm), G-win magnitudes preserved
  (median dG on winnable seeds should stay ≈ −4s or better).
- If the run regresses below 27/50: deliverable is an autopsy (per-seed
  delta table + action-mix probe like `runs/probe/v9_mode_mix_pass_fail/`)
  and the next lever, not another blind retrain.

## 5. Process

- Initialize git in `D:\research-dense` before any code change; commit the
  current state as the baseline (existing `.gitignore` already excludes
  `runs/`). Each subsequent change lands as its own commit so experiment
  lineage is reproducible.

## 6. Testing

- Scoreboard: reproduces 27/34/33 (slack rule) on the three existing eval dirs.
- Reward v10: unit tests in the existing `tests/` style —
  fair_win pays only on strict improvement below ref; intervention tax fires
  only when |G| ≤ ref and action ≠ coast; zero otherwise; v9 terms unchanged
  when new weights are 0 (backward-compat default).
- Oversampling: sampled-pool distribution test (easy ratio within tolerance).
- Full `python -m pytest -q` green before training starts (conda env
  `rl_driving`).

## 7. Risks

- **Reward hacking / coast collapse**: fair_win + tax could push the policy
  to coast everywhere (ties beat losses). Mitigation: fair_win weight (0.20)
  keeps winning attractive; delay_win 0.30 unchanged; monitor win-magnitude
  preservation in the scoreboard report, not just the count.
- **Oversampling shifts the training distribution** away from hard seeds and
  erases existing wins (v9d autopsy: "early coast-force wiped medium-hard
  wins"). Mitigation: warm-start from v9 i40, cap easy ratio at 30–40%,
  peek-gate promotion.
- **Structural seeds stay unwinnable**: accepted; reported separately.
