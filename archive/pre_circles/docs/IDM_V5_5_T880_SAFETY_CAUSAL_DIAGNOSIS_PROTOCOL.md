# IDM v5.5 t880 wall safety-causality diagnosis

Status: frozen diagnosis-only protocol. This is not a treatment repair, RL
training run, predictor dataset, or locked evaluation.

## Question

The prior t880 pilot produced safety failures under an otherwise dose-complete
compact wall. This protocol distinguishes four explanations without changing
the frozen treatment: replay/software inconsistency, direct first-second
assembly, the hold/release lifecycle, or metastable amplification.

## Frozen source and cohort

- analytical IDM damping controller and v5.4 compact-moving wall profile;
- 2700 veh/h mainline, 400 veh/h ramp, 30% CAV penetration;
- activation at absolute SUMO time 880 s, planned release at 900 s, common
  observation horizon 1080 s;
- Gate-F admitted development seeds 4401, 4402, and 4404 only;
- previously inspected seeds remain development-only and no validation/final
  seeds are opened.

## Arms

Every variant is paired with a freshly replayed damping-only control arm using
the same seed, reset, decision step, base policy, and horizon.

1. `sham`: command the frozen 20 s wall request and run the complete member
   latching/lifecycle path, but replace the wall actuator with a zero-effect
   actuator. It must issue no speed or tau TraCI commands. Vehicle trajectories
   should be identical to damping control; command/RSU metadata are excluded
   from the physical identity digest.
2. `assembly_only`: use the same frozen compact geometry and member selection,
   apply exactly the first one-second assembly command, then send the damping
   command and observe through 1080 s. HOLD and taper are not reached.

No equation, geometry, member rule, safety threshold, reward, seed, or outcome
definition is changed by this protocol.

## Required diagnostics and gates

Each case records full controller audits, wall override steps and IDs, state
transitions, common-horizon status, congestion outcomes, storage, and vehicle-
level hard-brake onset attribution. The machine gates are:

- sham physical trajectory identity is exact for every seed;
- assembly-only has its first override at decision+1 and no HOLD state;
- control and diagnosis share the complete 1080 s horizon;
- hard-brake events are reported as shared, treatment-only, or control-only,
  with phase, vehicle, lane, leader/follower, TTC and wall-membership context.

If sham identity fails, stop and classify software/replay. If assembly-only
reproduces the safety pattern, direct formation/assembly is implicated. If it
does not, but the original full pulse does, the hold/release path is implicated.
If neither ablation separates the outcome, report metastable amplification or
insufficient evidence; do not add seeds or tune the treatment in this run.

## Execution

The runner is `experiments/idm/run_idm_v5_5_t880_safety_causal_diagnosis.py`.
It launches 12 SUMO branches (three seeds × two variants × paired control) with
up to six workers and writes atomic artifacts under
`runs/probe/idm_v5_5_t880_safety_causal_diagnosis/`.

Predictor training remains explicitly unauthorized until a separately approved
treatment/headroom protocol passes.

