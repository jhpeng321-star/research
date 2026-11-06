# FULLPOL design: learning arbitrary CAV programs (clean5 policy side)

## 0. clean5a: speed-only EV mode (MAIN 19:3x; first deliverable, no LCEXEC dependency)

**Key and default.** Key `clean5_ev` (`fullpol_ev.py`). It is default OFF, and when OFF the
outputs are byte-identical.

**Slate.** The slate contains N2 families only:
- `abreast_wall`, with the wall_g and wall_h variants, plus its N2 V2V initiate/join slots
  when lane_drop v2v is on;
- `group_hold`, which becomes a runtime family with two F2 slots:
  - `group_hold:solo`: a 1-member `SoloHold` using the wall_g/wall_h parameters, matching the
    S-REGIME solo strategies wall_{g,h}_solo{L};
  - `group_hold:pair`: 2 members, using N2 R_SETS;
- none.

All legacy and W1 families are filtered out.

**Execution.** Execution is N2's closed-loop hold law (`SoloHold` is a 1-member
`LaneDropMove`). The exact executor is therefore N2's (N5 PASS).

**EV-BC.**
1. `guidance_rows` re-runs the registered rg_analyze paired seed bootstrap (10,000 resamples,
   RNG 20261106) on REGIME_RESULTS.
2. It flips the sign: gain = J(base) − J(strategy), so positive means lower delay.
3. Per (regime, family), the MAPPO-relative row (S-REGIMEU0) is preferred over the native one.
4. Families are filtered to the parity list (wall_g, wall_h, wall_{g,h}_solo{0,1,2}).
5. At every recorded decision row, the target is the highest-lb95 qualifying family (lb95 > 0)
   among those offered in that slate, else none.
6. The CE is −log P(set of equivalent slots).

**Learning rules.**
- The KL cap stays at 0.045. The step count is `imitation_epochs` (MAIN: 20).
- PPO on the team reward and G4LD credit (pairs; a solo is not a joint start) are unchanged.
- EV credit is available but OFF in the recipe (`fullpol_ev.recommended`).

**Regime label.** The regime is the episode cell, an offline label only. It is tested never to
reach the slate, descriptors, context or logits.

**R arm.** R has no R guidance rows, so it gets no terms (it learns on its own).

**Report-only receipts.** BC KL (on supervised contexts), KL on unlabelled contexts, and
seed-split validation (held-back target NLL; AUC of act vs none when both occur).

**Builder.** `python -m tools.coopmwm.fullpol_learn guidance --native REGIME_RESULTS.jsonl
[--mappo REGIMEU0.jsonl] --provenance ... --out CLEAN5A_GUIDANCE.json`. It refuses to overwrite
an existing output.

**Caveats.**
- The S-REGIME solo estimates come from wall-geometry triggers. clean5a offers solos whenever
  the ego is in the registered zone (`solo_geometry='zone'`; `'wall'` is available for strict
  matching).
- The regime is the cell only. A per-window closing-speed tercile labeller is an open issue.

Owner directive: the policy must be able to accelerate, brake and change lane, i.e. take any
move SUMO allows, and must be able to use and learn the cooperative moves. LCEXEC owns
*execution*: the primitive interface, `free_program`, the converter and the per-tick labels.
FULLPOL owns *selection and learning*. Everything is default OFF. The config key `fullpol` is
absent or False by default, and in that case every output is byte-identical.

## 0b. clean5b: free_program with lane changes (LCEXEC per-CAV initiate/join contract, ~21:3x)

**Ownership.**
- LCEXEC owns:
  - the option/protocol (`lc_families`: `free_program_initiate` / `free_program_join`
    ProgramOption, V2V relay, cap <= 6);
  - the executor and the env writer;
  - the converter.
- FULLPOL owns:
  - the template generator `fullpol_programs:propose` (`fullpol_templates.py`);
  - the trigger descriptor `fullpol_head.template_descriptor`;
  - the learning signals.

**Templates.** Templates are slot programs (`@0` = initiator) with per-slot join rules
{lane, ds}, with |ds| within the V2V range. There are four classes:
- **verified:** the arm's own `template_from_program` rows, used as-is;
- **structured:**
  - abreast_wall (wall_g / wall_h);
  - closing_pair (pair_g / pair_h / pair_t);
  - solo hold, labelled `wall_{g,h}_solo{SUMO lane of the initiator}`;
  - wall_fill (lane change into the empty lane, closing lane included);
  - gap_open_merge (merger or yielder role);
  - lane_redistribute;
- **lcm_R_v1:** random joiner lanes, so the lane targets stay adjacent;
- **primitive coverage.**

**Validation.** Every template passes `lc_program.FreeProgram` validation and
`check_registered`.

**Descriptor.** The descriptor has 6 template slots x 37 features plus 6 group features:
- per slot: primitives, lane target one-hot, into-closing, rule, releases, join-rule
  lane/ds, and an own-slot flag;
- group: initiate/join, slot count, duration, member count and gap.
It uses absolute WM lanes (a single registered map frame) and contains no ids or labels.

**Superseded.** The earlier FULLPOL group-level `free_program` option (fullpol_programs
FreeProgramOption + `free_program_runtime` hook) is superseded by LCEXEC's per-CAV protocol. It
stays default OFF.

## 1. What the policy can do (coverage argument)

| Level | Mechanism | Covers |
|---|---|---|
| Per tick (1 s) | Existing cooperation-layer residual on frozen MAPPO (F1: full strength). Bounded by the CAV cap of 1.2 m/s^2 and the speedMode-27 safe speed. | Any acceleration or braking SUMO accepts |
| Decision (registered 60 s plus V2V events) | Trigger selects one slate entry. A `free_program` entry is a per-member program of LCEXEC primitives: lon `base/native/speed/track/match/accel`, a persistent lane request to an adjacent existing lane (including the closing lane), own `tau`, `lcMode` 512/513, start/arming/give-up, and registered releases. | Lane changes, holds and speed targets for 1..4 CAVs, timed by the program's own arming position, not by the decision cadence |

A one-member program is a primitive ("change into lane L once past s, give up at s_g", "hold
v_w for h s"). A multi-member program is a joint move. Old families (wall, closing_pair, etc.)
stay available unchanged.

## 2. Candidate generation (`fullpol_programs.py`; additive hook in `feasible_proposals.candidates`)

- **Pool family `free_program`:** joins the F2 balanced slate as one more family (`slate: add`).
  It draws per slot from three source classes, in rotation:
  - (a) **verified-adapted:** the arm's own converted verified programs (B = verified, R = random
    procedure), re-bound to the ego's acquirable CAVs by role signature (WM lane relative to the
    closing lane, front-to-back order, spread). Infeasible bindings return None.
  - (b) **structured:** closing_pair, abreast_wall, group_hold, lane_balance, early_merge and
    zipper as programs.
  - (c) **random:** group size 1..cap from acquirable partners. Each member gets a random lon
    kind and parameters, a lane request to any adjacent existing lane (closing lane included),
    an arming zone and a give-up, tau, lcMode, and a hold.
- **Ranges:** every continuous value comes from `lc_program.REGISTRY` (cited sources) or an
  existing N2 registration (zone 400..1800 m before the drop; give-up 300 m before the drop).
  There are no new numbers.
- **Coverage:** a coverage test asserts every registered grid value and every lane-target type
  is drawn.
- **None-guard (N11 F8):** every pool/redraw/adapt may return None, and the caller drops it.
- **Optional `offered`:** an offered program (practice "offer, don't force") takes slot 1 and
  the rest stays balanced.
- **Observables only:** inputs are the ego's own rows, sensed HDVs (as geometry) and acquirable
  CAV ids. Members are observable CAVs only; HDVs are never members.

## 3. Policy head (keep slot choice; `fullpol_head.py`)

The existing trigger already scores each candidate with a shared MLP over [MAPPO hidden |
candidate descriptor | context]. That makes it a pointer over a sampled action set, so it can
express any program, provided the descriptor encodes the program.

- **Change:** a zero-initialised extension block (same mechanism as W1/N2) of about 135 columns.
  - Per partner slot (ego first, then V2V order) there are 33 columns: in-program flag, lon
    one-hot, v_w/dmax/amax/accel/track-offset/hold/start fractions of the registered ranges,
    tau flag and value, lcMode 512, lane request (left/right relative to own lane,
    into-closing, out-of-closing, arm/give-up distance-to-drop fractions, rule one-hot), and
    release bits.
  - There are 3 group columns.
  - The descriptor is egocentric and relational, with no ids and no source class, so programs
    are judged by content. The existing logits are unchanged until training.
- **No continuous refinement head in clean5.** Random draws inside the registered ranges
  already cover the continuous space over time. The scorer is smooth in the descriptor. A
  Gaussian refinement would add PPO variance where pace, not expressivity, is the measured
  bottleneck (LEARNAUDIT 5). This can be revisited if coverage receipts show a gap.

## 4. Learning signals (`fullpol_learn.py`)

1. **BC.**
   - Every converted verified program yields trigger CE contexts:
     - the initiator at t0, with the exact program inserted in slot 1;
     - each other member at its own activation tick, from LCEXEC's `lcexec-tick-label-v1`
       `event=activate`, with the remaining program.
   - The context is rebuilt from the causal observer, including the sensed HDV rows
     (LEARNAUDIT/BOOST-PRE: tables lacked HDVs).
   - There is no projection onto legacy boxes, which fixes LEARNAUDIT 1.
   - Optional negatives: the arm's own `*_NEGATIVES` rows become "none" targets with the
     negative program in slot 1. These teach *when*, not only *what*.
   - Equal mass per arm.
2. **Onset term / offer mode.**
   - When a practice start offers instead of forcing, the ego's own choice at t0 is a real PPO
     sample, plus G4LD's onset CE toward slot 1.
   - I reuse G4LD's term and its weight, and only provide the free_program slot.
3. **PPO on own choices.** The per-CAV trigger PPO is unchanged. free_program rows serialise
   through `to_dict`/`from_dict`. The program runs on the executor commitment (not
   `behaviour.executed`), so its decision row keeps its PPO ratio.
4. **G4 credit.**
   - I register a G4LD `ADAPTERS['free_program']` row/check pair. Members are the program
     members; jobs are group, base and each solo, where solo = that member's MemberProgram only.
   - Credit goes to the initiator's decision, plus the joiners' own decisions for V2V sessions.
   - The estimator, sampling and caps are G4LD's (unchanged).

## 4b. Regime-level expected-value guidance (owner pivot 18:21; LEARNABILITY 18:05)

LEARNABILITY found that per-moment WHEN is not predictable from t0 observables (seed-CV AUC
0.36-0.41). The clean5 target is therefore expected value by regime. I build two consumers of
the S-REGIME output. Both are default OFF and B = R.

**Input format** (`regime_guidance`): one row per regime key and program family/class, with
mean, lb95, n, the sweep id and the source sha. The regime key is an offline training label
(the cell, or the registered tercile of an observable proxy). It is never a policy input. The
policy still sees only its own sensing, V2V and RSU blocks.

**(i) EV-BC.** At every decision context of a training episode in regime r, the CE target is:

| Condition | Target |
|---|---|
| A family with lb95 > 0 exists | the best such family (by lb95) |
| Otherwise | none |

- lb95 > 0 is the existing registered admission convention, not a new threshold.
- The slate offers that family's program, adapted to the ego's acquirable CAVs.
- If the family is infeasible here, there is no label at this context.

**(ii) EV-credit.** An additive advantage for the selected program's family equals the
regime's mean gain, scaled by the same UNIT and clip as G4. It acts as a low-variance
substitute for the near-chaotic per-sample return. It never replaces the measured SUMO return.

**Symmetry:**
- B uses the B-procedure sweep rows.
- R uses rows from the matched-effort random procedure (LCMOVES R / lcm_R_v1) for the same
  regimes.
- If R has none, R gets no EV terms. Nothing is padded.

**Random class = registered procedures.** The random generator in section 2(c) is exactly
LCMOVES' random-program procedure lcm_R_v1:

| Knob | Values |
|---|---|
| lane target | uniform over {none, adjacent lanes including closing} |
| LC offset and start offset | {0, 20, 40, 60} s |
| v_w | {15, 18, 22} |
| hold | {90, 120, 180} |
| dmax | {1.0, 1.5} |
| tau | {none, 0.6, 2.0} |
| members | 1..4 acquirable CAVs (1 = single-CAV primitive; 1..4 is the tracks_spec arity) |

The structured class covers wall_fill, rolling_wall, gap_open_merge and lane_redistribute
(LCMOVES B families) when LCEXEC exposes their programs.

## 5. B/R symmetry

- **Shared:** generator, descriptor, losses, weights, KL rules and RNG streams.
- **Different:** only the template/BC source. B uses verified rows; R uses its re-verified
  random-procedure rows (PLANGEN2_R / NEWMAP_R, R_random).
- **Mass and counts:**
  - BC mass is equal per arm.
  - Context counts are reported. They are never padded.
  - An optional `bc_matched_count` uses min(nB, nR) per update with least-used rotation.
- **Tests:** a test asserts identical code paths and identical supervision for equal inputs.

## 6. Learning-speed audit and proposals (NOT changed; registered values stay)

- **KL cap.** The cap is 0.045 per update. Going from uniform over 9 slots to 0.8 on one slot
  is about 1.1 nats, and the Fisher-Rao distance is 1.53. The ideal minimum is about 5 capped
  updates; the observed number is 12 or more (clean2 CE 2.13 -> 1.21 at u6 -> 0.39 at u12).
  - **Proposal P1:** keep the 0.045 value but measure the BC KL on ordinary (unlabelled)
    on-policy contexts (`bc_kl_scope: ordinary`; default `all` = current behaviour).
  - This protects behaviour where there are no labels and lets labelled contexts concentrate
    in 1-3 updates.
- **Generalisation** (AUC 0.495 on held-out seeds vs 0.92 on train). The fixes are:
  - HDV-inclusive contexts;
  - egocentric descriptors;
  - negatives;
  - a seed-split validation AUC receipt (`fullpol_audit.valsplit_auc`, TRAIN seeds only),
    reported every update.
- **Budget estimate** (FULLPOL_READY): BC on about 80 B / 21 R rows can memorise. Generalising
  needs more distinct verified situations. PPO discovery at about 8 offers per update is
  marginal over 24 updates. G4 at 8 starts per arm for the whole run is too sparse to matter;
  roughly 50 per arm would be needed.
