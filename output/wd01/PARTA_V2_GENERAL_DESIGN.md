# Part A v2 GENERAL: design note (owner decision 2026-09-29, ~15:00)

World model session, 2026-09-29. This supersedes the "v2 exact" retarget in
NOTE_DETERMINISM_REASSESSMENT.md §1 and the "LC2013-exact" direction in
DESIGN_PARTA_STRUCTURAL.md, **as Part A's product**.

**Owner's reasoning:** an exact SUMO replica would not match reality. Part A
must be a general rule-based predictor that tracks every car it is given and
uses ONLY what the reconstructor provides.

## 1. Contract

### Input

Exactly one reconstructor sample:

- the vehicles in the reconstructed zone;
- for each vehicle: id, lane, along-lane position s, speed v, plus whatever
  per-vehicle attributes the recon outputs (e.g. acceleration, length, the
  CAV flag and its commanded action).

The planner also supplies the commanded action per CAV.

Forbidden inputs:

- oracle or true vType / driver type;
- a sampled vType;
- future-arrival sampling or schedules;
- inserting vehicles the recon did not provide;
- entrance queues that are not in the recon sample;
- SUMO internals (RNG, seed, sigma, departure schedule, speedMode tricks).

### Dynamics

Generic textbook rules with ONE nominal parameter set, shared by all HDVs:

- **Car-following:** IDM (Treiber) with published defaults. The
  proposal is in §2; the owner confirms it.
- **Lane changes:** MOBIL for discretionary changes. A mandatory merge rule
  on the acceleration lane: urgency rises toward the lane end, a safe-gap
  test is applied, and a stop at the lane end if no gap is found.
- **CAVs:** they follow the commanded action through the existing controller
  emulators (partA_ctrl: native / damping / PI external-U; the wall-v2 group
  law) and the execution rule.
- **Not allowed:** SUMO-exact behaviour, and per-type tuning to SUMO.
- **Later option:** per-car parameters estimated online ONLY from that car's
  own observed history (seen cars), never from truth.

### Tracking

- Every input vehicle keeps its identity and full state (lane, s, v, a) at
  every step up to the horizon H, or until it leaves the modelled road.
- No vehicle is created. If a leader is missing because it has not arrived
  yet, that is treated as free road. The consequence is quantified (§4).

### Output

- Per-vehicle trajectories (id → [t, lane, s, v, a]) at 0.5 s steps.
- The Part A metrics computed over the TRACKED vehicles:
  - window delay;
  - TTC<3 exposure;
  - min TTC;
  - hard-brake seconds;
  - collisions;
  - lane changes by type.

### Uncertainty

- It comes ONLY from the spread across the K recon samples.
- Given one sample, the dynamics are deterministic: no dawdle and no random
  lane changes.
- M "members" become the K recon samples. They are not parameter draws.

### Arrivals

- Arrivals after t are not Part A's job. Part B and the horizon choice handle
  them.
- Part A reports the horizon at which untracked arrivals start to matter
  (§4).

## 2. Proposed nominal parameters (owner to confirm)

The parameters are published textbook values. They are NOT SUMO's per-type
values.

**IDM (Treiber & Kesting, highway defaults):**

| parameter | value |
| --- | --- |
| desired speed v0 | lane speed limit (31.29 m/s main, 22.22 m/s ramp) |
| time headway T | 1.5 s |
| max acceleration a | 1.0 m/s² |
| comfortable deceleration b | 1.5 m/s² |
| jam distance s0 | 2.0 m |
| exponent δ | 4 |
| vehicle length | 5 m, unless the recon provides it |

The integration is ballistic, at dt = 0.5 s.

**MOBIL:**

| parameter | value |
| --- | --- |
| politeness p | 0.2 |
| threshold Δa_th | 0.1 m/s² |
| safe braking b_safe | 4 m/s² |
| keep-right bias | 0.3 m/s² |

**Merge rule:**

- Mandatory within the zone [378.66, 454.10].
- The incentive is replaced by urgency (1 − d_remaining/L_zone).
- Safety is tested with b_safe, relaxed linearly toward 6 m/s² at the lane
  end.
- If no gap is found, the vehicle stops at the lane end (the obstacle rule).

**Collisions:** they are counted when the gap is < 0 after an update. The
model never teleports.

## 3. Reuse

d03703e2 ("v2 exact", running, NOT cancelled) is reviewed when it lands:

- **Kept:** its IDM + lane-change engine, its geometry/merge code and its
  unit tests, as the RULE ENGINE (parameterised).
- **Reclassified:** its oracle-vType exactness harness becomes a unit check
  of the equations only. It is NOT Part A's acceptance gate.

Part A v2 general then plugs in:

- the nominal parameter set;
- the recon-sample input adapter (the reconstructor's RG-0 module: recon
  sample → vehicle table);
- tracking + metrics;
- the CAV action path.

## 4. Evaluation (later; the scored runs need the owner's data-generation go)

The generic rules are scored against SUMO truth, reported honestly as MODEL
error and not tuned away, in two settings:

- (a) from the TRUE initial state (the logged state restricted to the recon
  zone);
- (b) from the recon samples.

Per horizon H ∈ {10, 30, 60, 120} s:

- the delay / TTC / hard-brake / LC errors;
- trajectory RMSE on tracked vehicles;
- the "missing arrivals" curve: the truth metric over (i) the tracked
  vehicles only vs (ii) all vehicles, which shows the H at which untracked
  arrivals dominate.

Setting (a) can run offline on existing logs with NO SUMO. Only new
recon-sample truth would need data generation.

## 5. Consequences for existing plans (need owner/main decisions)

- **Gate v2 and C2L1.**
  - Gate v2's distributional coverage (c) was designed for stochastic
    ensembles.
  - With TRUE state, Part A v2 general is deterministic. Its coverage is
    defined only over recon samples.
  - The C2L1 registration (amended: v1 vs v2 exact) no longer tests the
    product.
  - **Proposal:** HOLD C2L1 unrun, and write a new registration for Part A v2
    general once the recon-sample path exists.
- **Predict contract.**
  - The delay term D_A becomes a tracked-vehicles-only quantity, plus the
    Part B remainder for arrivals and untracked vehicles.
  - This needs a contract v2.1 note: define D_A over the recon-provided
    vehicles, and add `n_tracked` and `untracked_arrivals_expected` fields.
- **Part B coupling.**
  - Part B's value target must cover whatever Part A does not track: arrivals
    and vehicles outside the zone.
  - The deployable v0.3.1 inputs are already compatible (observed counts, no
    future schedule).

## 6. Open questions for the owner (via main discussion)

1. **Nominal parameters:** Treiber highway defaults (above), or SUMO's
   documented defaults (IDM accel 2.6, decel 4.5, tau 1.0, minGap 2.5)?
   Either way, one set for all vehicles?
2. **Recon output schema:** which per-vehicle fields, K, the zone extent, and
   whether entrance queues or ramp vehicles are included. Is acceleration
   provided? This decides the adapter.
3. **Vehicles near the zone edge:**
   - If a vehicle's leader is outside the recon zone, it is treated as free
     road. OK?
   - If a vehicle leaves the zone: stop tracking it, or keep simulating it on
     the modelled road to the horizon? The proposal is to keep it while it is
     on the modelled road.
4. **C2L1:** hold it, and re-register for v2 general? That is my
   recommendation.
5. **Evaluation (a):** is it allowed now? It is offline on existing logs with
   no SUMO; it restricts the logged truth state to the recon zone.

## 7. Answers (main discussion, under the owner's delegation, 2026-09-29; also recorded in tmp/zcode_review/OWNER_DECISIONS_20260928.md)

- **Q1: parameters.**
  - IDM uses Treiber highway defaults: T 1.5 s, a 1.0 m/s², b 1.5 m/s²,
    s0 2 m, δ 4, v0 = the lane limit.
  - MOBIL uses Treiber defaults: politeness p ≈ 0.5, b_safe 4 m/s²,
    Δa_th 0.1 m/s².
  - ONE set for all vehicles, kept in a single swappable parameter file.
  - NOT SUMO defaults (the goal is independence from the simulator).
  - The sensitivity to T and a is a later analysis.
- **Q2: recon schema.**
  - Now: r0/SCHEMA_V1.md. When it lands: r1/CORPUS_SCHEMA.md "Part-A input
    view".
  - The view has NO vType of any kind (test-enforced). length and minGap are
    nominal for sampled vehicles.
  - K and the zone extent come from the reconstructor.
- **Q3: zone edges.**
  - A leader outside the zone is treated as free road. Those vehicles are
    FLAGGED and COUNTED per prediction as `n_free_road_assumed`, because this
    is an optimistic bias that must be measured.
  - Vehicles leaving the zone keep being simulated while they are on the
    modelled road.
- **Q4: C2L1.** HOLD it unrun, and re-register it for v2 general once the
  recon-sample path exists.
- **Q5: evaluation (a).** It may run now, offline on existing logs, TRAIN
  only, with zero SUMO. It is registered as an ANALYSIS before running
  (REGISTRATION_PA2G_EVAL_A.md). No CAL or HOLDOUT.

## 8. Recon Part-A view: agreed fields (reconstructor follow-up b9cf7e2d, pending)

- **Sample header:** t_s, k, K, sample_weight (1/K when unweighted).
- **Zone extent:** a per-lane [s_min, s_max], the union of CAV sensing and RSU
  coverage at t.
- **Per-CAV controller block**, passed through from the recorded dispatcher
  rows:
  - action ∈ {native, damping, pi, wall};
  - wall_id, phase, time_in_phase, wall_speed, locked;
  - controller_state {damping_target_speed_mps, pi_running_target_mps}.
- **Conventions (read carefully).**
  - Use **s_down_m** (increasing downstream, = −d) and **lane_phys**
    (0 main_left, 1 main_right, 2 accel; a tested mapping from the SUMO
    lane_id and the r0 groups).
  - NEVER use r0's d (positive upstream) or the group index g0–g3 directly.
    In g0–g3, SUMO _0 is the rightmost lane, and g3 is the ramp.
