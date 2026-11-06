# FULLPOL: how the clean5 policy selects and learns cooperative moves

**Owner directive.** The policy must be able to accelerate, brake and change lane, and must be
able to use and learn cooperative moves.
**Owner framing.** Learn WHEN to cooperate: conditions, not moments.

**Branches.**
- `c4/fullpol-5a` (e36ece92): clean5a only, without LCEXEC/N9 code. N4 merges this one.
- `c4/fullpol`: clean5a plus clean5b (LCEXEC merged).

All keys are default OFF, and when OFF the outputs are byte-identical; a subprocess golden is
tested against the pre-FULLPOL sources.

## 1. Action coverage

| Level | Mechanism |
|---|---|
| Per tick | Frozen MAPPO plus the cooperation-layer residual (F1). Gives any acceleration or braking within the CAV cap of 1.2 m/s^2 and speedMode 27. |
| Decision | The existing slot-choice trigger picks one slate entry. |

Slate entries by run:
- **clean5a** (`clean5_ev`): N2 entries only:
  - none;
  - solo hold: a 1-member `group_hold` (`SoloHold`), with N2's exact law;
  - 2-member hold;
  - abreast_wall: wall_g/wall_h, plus its V2V initiate/join slots.
  The G2 pair path is bypassed (N11 F10).
- **clean5b** (`clean5_ev.families += free_program` plus `lc_families`): LCEXEC's per-CAV
  `free_program_initiate/join` options. Each carries a slot template of primitives:
  - longitudinal kind base/native/speed/track/match/accel;
  - a persistent lane request to any adjacent existing lane, including the closing lane;
  - own tau and lcMode;
  - offsets, holds and releases.
  A one-slot template is a single-CAV primitive.

## 2. Candidate generation (clean5b): `fullpol_programs:propose` (LCEXEC generator hook)

Templates come from `fullpol_templates.py`, one draw per class in random order. There are four
classes:
- **verified:** the arm's own templates from `lc_families.template_from_program`. This is the
  only B/R difference.
- **structured:** abreast_wall, closing_pair, solo_hold, wall_fill, rolling_wall,
  gap_open_merge, lane_redistribute.
- **random_r:** S-LCMOVES `lcm_R_v1`.
- **random_prim:** uniform over the primitive kinds.

**Constraints.**
- Every value comes from a registration (`REGISTERED` table with sources).
- Join-rule ds stays within the V2V range.
- Infeasible draws return None (N11 F8) and contribute nothing.
- Inputs are the own row and own sensing only. HDVs are geometry only, never members.
- No regime or config label is read; tests inject hidden fields to check this.
- A coverage test asserts that every registered grid value, every primitive kind and every lane
  type is drawn: left, right, into-closing and out-of-closing.

## 3. Head: the slot-choice trigger is kept (no new architecture)

The trigger already scores each candidate with a shared MLP over [MAPPO hidden | candidate
descriptor | context], which makes it a pointer over a sampled action set.

**Descriptor.** FULLPOL supplies the descriptor through LCEXEC's hook
`fullpol_head:program_descriptor`. Its zero-initialised columns are:
- 6 template slots x 37 features: primitives, lane target one-hot, into-closing, rule, releases,
  join lane/ds, and an own-slot flag;
- 9 group features: initiate/join, slots, duration, members, gap, and the own-lane one-hot.

It contains no ids, provenance or labels.

**No continuous refinement head.** Random draws inside the registered grids cover the space, and
pace, not expressivity, is the measured bottleneck.

## 4. Learning signals

**EV-BC** (the pivot, MAIN rules). `fullpol_learn.guidance_rows` re-runs the registered
rg_analyze paired seed bootstrap (10k resamples, RNG 20261106) on S-REGIME (native) and
S-REGIMEU0 (MAPPO u0) results.
- **Sign:** gain = J(base) − J(strategy).
- **Baseline:** per cell, u0 rows are used when present; otherwise native rows. The two are
  never mixed within a cell.
- **Target:** at every recorded decision row, the target is the highest-lb95 qualifying family
  (lb95 > 0, parity list) that is OFFERED in that slate, else none.
- **Loss:** −log P(set of equivalent slots).
- **Parity list:** the solo labels only, wall_{g,h}_solo{own SUMO lane}, so targets are
  lane-specific. Walls and pairs stay offered.
- **Regime label:** the episode cell_key, used offline only. A test checks that it reaches no
  slate, input or logit.
- **R arm:** has no rows, so it gets no terms.

**Exact-program BC** (clean5b). `program_protocol_contexts`:
- the initiator imitates `free_program_initiate` at t0;
- each joiner imitates `free_program_join` when its own receive step offers the reconstructed
  session at its true slot;
- contexts include sensed HDVs;
- there is no projection to legacy boxes (LEARNAUDIT 1).

**PPO** on own choices is unchanged. Program/solo decision rows are not "executed", so they keep
their PPO ratio.

**G4.**
- N2 pairs and walls are credited through G4LD (a solo is not a joint start).
- Program sessions have a G4LD adapter (rows and check). Its macro fails closed until G4 branch
  execution of programs exists.

**KL.** The 0.045 cap is unchanged. It is now measured as a mean per CAV decision row (24080ea2).
- `clean5_ev.ppo_kl='row_mean'` applies the same rule to trigger PPO. The inherited row SUM
  admitted only about 3e-4 nats per CAV per update.
- EV-BC steps = 60 (`ev_bc_steps`; MAIN registered).

**Report-only receipts.**
- `report_bc_kl`;
- `report_kl_unlabelled`;
- `report_valsplit_auc`: seed-hash folds, shadow fit, held-back target NLL and AUC;
- `fullpol_ppo_kl`.

## 5. Learning budget (fixture-measured, report only)

**KL geometry.** Going from uniform over 9 slots to 0.8 on the target is 1.11 nats (Fisher-Rao
distance 1.53). At most 0.3 per capped update gives at least 6 updates.

**EV-BC pace fixture** (`python -m tools.coopmwm.fullpol_pace_fixture`):

| Steps / lr | Result |
|---|---|
| 20 / 3e-4 | P(target) 0.32 at u6, 0.52 at u12, 0.80 at u24; uses about 10% of the cap |
| 60 / 3e-4 | 0.8 reached by u9 |

MAIN registered 60 steps.

**PPO discovery.** An always-on solo2 at D3200_p60 (SD about 26.7k veh-s per episode) needs
about 8 paired episodes at the episode level. Per-decision credit is far more diluted, so EV-BC
carries the "when", and PPO and G4 refine.

**Generalisation.** LEARNABILITY found per-moment WHEN at chance. The EV targets condition only
on the cell regime (inferable from traffic observables) and the own lane, and
`report_valsplit_auc` measures this every update.
