# REGISTRATION_WM_N1, AMENDMENT 6 (DRAFT, NOT FROZEN): a state-only global token for the correction head

**Status:** DRAFT, 2026-10-02, world model session. It freezes only on the
owner's OK. Until then, the code path is behind the flag
`state_global_token`, default **OFF**, and the registered A3 behaviour is
unchanged.

- Amends `REGISTRATION_WM_N1.md` (`666233aa…`) and A1–A5. A5 is
  `7788c270…`.
- **UNCHANGED:**
  - the §4 targets and decision rule;
  - the chunk-5 TEST (sealed);
  - HOLDOUT (sealed);
  - the CAL split `bc4e32f2…`;
  - the variants and tie order;
  - the A4 tuning knobs;
  - K_max = 192.

## A6.1 Finding that motivates it

A3.1 specifies that long-horizon outcomes "come from the rollout PLUS a
strategy-independent correction head fed a summary of the CAVs' planned
motion". The implementation reviewed on 2026-10-02 (commit 649be00) does not
do that:

- In both training (`train/idea_a.py` ~1307) and evaluation
  (`evalh/idea_a.py` ~352), the correction head receives `global_token =
  zeros`.
- So delay, TTC < 3 s and hard-brake are predicted from the 16-feature CAV
  motion summary alone. No traffic-state input reaches the outcome head, and
  the multi-step rollout's horizon (20 steps = 10 s) is far shorter than the
  outcome horizons (τ up to 570 s).
- The outcome prediction is therefore close to state-blind. That
  contradicts the "rollout PLUS correction" composition and would weaken
  every paired-contrast and incident (INC1) result.

## A6.2 Change

The correction head's `global_token [B, d_model]` comes from a small
**state encoder** applied to the t0 state only:

- **Inputs (exhaustive):**
  1. the t0 `veh_kin` slots `[K = 192, 7]` (s_down, the three lane one-hots,
     v, a, length), with the slot mask;
  2. the entrance counter features of the t0 tick (the same `counter`
     channel the arrival head already uses).
- **Not inputs:**
  - no `is_cav` flag;
  - no CAV plan or `cav_plan_ch` (that information already enters through
    the motion summary);
  - no strategy, action, policy, arm, wall-label or outcome channel.
- **Architecture:** masked mean and max pooling over the slots, then an MLP,
  concatenated with an MLP of the counters, then a linear layer to d_model.
- The label-leak test (G3) is extended to assert that the token is
  bit-identical under any perturbation of label/action/plan channels and of
  `is_cav`.
- The same encoder feeds training, CAL calibration and evaluation, the
  `wm_api` scorer and the INC1 predictor. The obstacle of INC1 enters only as
  a stopped vehicle in `veh_kin`.

## A6.3 What does not change

- The losses and λ's: target normalization is an implementation fix, not
  part of A6.
- The contrast definition.
- The HDV residual, rollout and arrival heads.
- The ensemble design, seeds, data, splits and the A4 improvement loop.

## A6.4 Evaluation and disclosure

- **The flag:** OFF reproduces the old zeros path byte-identically. This is
  tested.
- **Member 1 and A4 tuning:** member 1 trains with the flag ON only if A6 is
  frozen before m1 starts. Otherwise m1 trains with it OFF, and A6 enters as
  an A4-loop knob only after a separate owner decision.
- The A6 freeze, or its rejection, is recorded in the WM-N1 ledger entry.
