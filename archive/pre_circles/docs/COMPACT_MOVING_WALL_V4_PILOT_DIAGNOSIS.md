# Compact moving-wall v4 pilot diagnosis

## Scientific verdict

The frozen v4 compact moving-wall treatment is **not ready for predictor
training**.  Its exact causal implementation is now verified, but the treatment
itself fails the predeclared dose, safety, outcome-diversity, and
collapse-prevention gates.

This is a treatment failure, not a predictor failure.  Training a larger model
on these rows would primarily learn formation/delivery failures and would not
support the paper's collapse-prevention claim.

## Frozen treatment and causal verification

- Control: the frozen decentralized damping checkpoint.
- Treatment: identical damping plus one trajectory-coherent CAV per occupied
  pre-merge lane, with a 50 m cross-lane span cap.
- Recruitment band: 200--1000 m upstream of the merge.
- Treatment duration: 20 controller steps = 20 s.  SUMO integrates at 0.5 s,
  while the damping/wall controller updates at 1.0 s.
- Assembly: at most 10 s, 2 m/s member-speed tolerance, and 1 m/s^2 maximum
  assembly deceleration.
- Holding: 10 m/s nominal wall target, 0.3 m/s^2 maximum commanded
  deceleration, and release back to identical damping.

The corrected exact-twin verification passes on development seeds 504 and 505:

`runs/probe/counterfactual_compact_exact_twin_v3_corrected_s504_505_step180_rerun.json`

Both seeds pass damping replay identity, predecision identity, in-band override
isolation, identical out-of-band damping, assembly-to-hold completion, and
release back to damping.  The artifact verdict is intentionally limited to
`causal_contract_passed_predictor_not_yet_tested`.

## Frozen pilot

Artifact:

`runs/probe/counterfactual_compact_headroom_frozen_v4_hyper_p20_s601_620`

The requested cohort was seeds 601--620 at decision steps 120, 180, and 240.
The run was stopped after 13 atomically completed seeds because the 90% safety
gate had become mathematically unattainable.  The final artifact records 13/20
completed seeds, the seven unrun seeds, and the early-stop reason.  These seeds
remain protocol-development/headroom seeds and must not be used for model
evaluation.

### Corrected strict results

- 39 paired cases; 36 legacy-eligible cases.
- 13 activations were declared feasible and started treatment.
- 7/13 activations satisfied the strict compact dose contract: **53.8%**, below
  the required 80%.
- Three activations explicitly aborted after 10 s of assembly because member
  speeds did not converge.
- Of the ten remaining full-duration activations, three spent less than 80% of
  their override within the 50 m span.
- Only 2/7 strict completed treatments were safety-admissible: **28.6%**, below
  the required 90%.
- Safety/service benefit occurred on one independent strict-completion seed;
  nonbenefit occurred on five seeds.
- There were no excess collisions or teleports.
- There were **zero strict collapse-prevention cases** and seven strict
  nonprevention cases across six seeds.
- The legacy composite headroom summary passes, while the corrected treatment
  gate fails.  This discrepancy demonstrates why the legacy label must not be
  used to authorize predictor training.

The treatment-dose denominator includes every activation that began assembly.
Assembly failures cannot be removed by subsequently marking their causal row
ineligible.

## Failure anatomy

### 1. Assembly prediction omits live traffic interactions

The three assembly timeouts began with approximately 4.1--7.2 m/s cross-lane
member-speed spread.  The bounded kinematic projection predicted feasible
formation, but leaders, followers, and lane-specific traffic prevented live
convergence.  A purely open-loop common speed target is therefore not an
adequate formation controller.

### 2. Identical member targets do not preserve a cross-lane wall

The controller sends all wall members the same nominal speed target.  It has no
relative-position feedback to correct cross-lane drift after surrounding
traffic perturbs one member.  The only apparent collapse-prevention case
(seed 607, decision 180) reached a realized span of about 159 m and remained
within 50 m for only 30% of the treatment.  Counting this case would silently
change the treatment from a compact wall to a dispersed coordinated slowdown.

### 3. The fixed 10 m/s target is not traffic-state matched

The fixed target is far below typical activation speeds, so HOLDING is usually
a continued speed-reduction command rather than travel at a formed wall speed.
Strict completed wall members lost roughly 5.7--12.3 m/s over the 20 s window,
partly due to surrounding traffic.  Five of seven strict completed treatments
failed the downstream safety/service oracle.

### 4. The runtime safety shield is too local

The shield watches collisions, teleports, wall CAV hard braking, and the
immediate follower of each wall CAV.  Excess hard-brake and low-TTC effects can
occur elsewhere in the induced shock region, so a treatment may complete while
failing the outcome-level safety budget.

### 5. The apparent effect is conditional but presently too sparse

Exploratory development-only comparison suggests the two safe strict cases
occurred with a more pronounced upstream-to-downstream speed differential and
lower queue/spillback than the five unsafe cases.  With only seven strict
completions and two safe cases, this is a hypothesis for a future physics gate,
not evidence for a calibrated predictor.

## Literature-guided v5 recommendation

The recommended next treatment is a **safety-shielded adaptive compact moving
bottleneck**, while keeping frozen decentralized damping as the control arm.

1. Add a cross-lane formation servo.  Use bounded per-member speed offsets
   around a common metering speed to regulate relative longitudinal position,
   rather than giving every member an identical open-loop target.
2. Replace the fixed 10 m/s target with a bounded, state-dependent speed
   deficit computed from predicted bottleneck inflow/discharge.  The overlay
   should be the least additional intervention above damping that achieves the
   requested inflow reduction.
3. Project front and rear safety for every member and apply a barrier-like
   command clip that minimally changes the nominal wall command.  Extend the
   runtime monitor to hard-brake/TTC onsets in the complete wall influence zone,
   not only immediate followers.
4. Freeze a stricter pre-activation formation margin (position and speed) and
   retain full occupied-lane coverage and the 50 m realized span as the primary
   treatment.  A 100--150 m distributed controller may be studied only as a
   separately named ablation; it must not be used to rescue the compact-wall
   result post hoc.
5. Reuse seeds 601--613 only for bounded v5 development.  After exact-twin
   verification, run a fresh 12--20-seed headroom gate on a new excluded
   cohort.  Do not train benefit, collapse, safety, or service predictors until
   the revised treatment passes.

This recommendation is consistent with primary moving-bottleneck work that
uses platoon speed and formation/configuration as control inputs and computes
them from predicted traffic state, rather than imposing a fixed speed:

- M. Cicic et al., *Coordinating Vehicle Platoons for Highway Bottleneck
  Decongestion and Throughput Improvement*, IEEE T-ITS, 2022,
  https://doi.org/10.1109/TITS.2021.3088775
- M. Cicic et al., *Platoon-actuated variable area mainstream traffic control
  for bottleneck decongestion*, European Journal of Control, 2022,
  https://doi.org/10.1016/j.ejcon.2022.100687
- J. Ma et al., *Speed Harmonization for Partially Connected and Automated
  Traffic*, IEEE T-ITS, 2023,
  https://doi.org/10.1109/TITS.2023.3287336
- C. Zhao, H. Yu, and T. G. Molnar, *Safety-critical traffic control by
  connected automated vehicles*, Transportation Research Part C, 2023,
  https://doi.org/10.1016/j.trc.2023.104230

## Paper framing after this result

The counterfactual-escalation premise remains viable, but the current v4 wall
cannot support a paper claim that the predictor works or that collapse is
prevented.  The strongest honest framing is:

> Decentralized damping is the always-on nominal controller.  A coordinated
> moving bottleneck is an aggressive, state-dependent escalation whose
> feasibility, safety, and conditional benefit must be separated explicitly.

If v5 produces reliable dose and safety but still no collapse-prevention
diversity, the paper should pivot its primary endpoint to safety/service or
recovery, with collapse prevention reported as a negative result.  More seeds
or a more complex predictor cannot repair absent treatment headroom.
