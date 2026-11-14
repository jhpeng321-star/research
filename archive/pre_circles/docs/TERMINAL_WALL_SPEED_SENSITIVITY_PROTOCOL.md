# Terminal Wall-Speed Sensitivity Protocol (v5.2 Development Audit)

## Scientific question

Does the current adaptive wall fail to change congestion outcomes because its
terminal speed is restricted to at most a 20% reduction, even when formation
and treatment duration are successfully delivered?

This experiment changes terminal wall speed, not braking rate. The treatment
formation and servo deceleration bounds remain frozen at 1.0 m/s².

## Development cohort

Only previously inspected decisions are used. They are excluded from every
later prospective pilot, model-development split, and locked test:

- seed 504, decision step 180;
- seed 606, decision step 180;
- seed 607, decision step 180;
- seed 609, decision step 120;
- seed 612, decision step 120.

The existing generator enforces at least four development seeds. Therefore,
each speed level runs the five excluded seeds at both steps 120 and 180. The
five decisions above are the frozen primary comparison; the other five grid
rows are retained and labeled as auxiliary excluded-case diagnostics.

## Treatment variants

The adaptive maximum reference-speed reduction is varied over:

- mild: 10%;
- current: 20%;
- strong: 30%.

A 40% variant is not run automatically. It may be considered only if the 30%
variant is safety-admissible yet does not produce meaningful physical effects.

Everything else is frozen: decentralized damping checkpoint and control arm,
hyper demand, Krauss mixture, 20% CAV penetration, 200–1000 m recruitment band,
full occupied-lane coverage, trajectory-coherent compact selection, 50 m span
cap, 50 m release boundary, 20 s treatment, 3 s wall headway, front/rear TTC
projection, influence-zone monitoring, and return to identical damping.

## Diagnostics

Each arm must retain replay and predecision identity. The audit records:

- requested common and member-projected terminal speeds;
- realized member speed during HOLDING;
- last-five-step terminal speed error and fraction within 1 m/s;
- treatment duration, lane coverage, and compact geometry;
- safety projection clips, collisions, teleports, hard braking, and low TTC;
- merge discharge, delay, and sustained congestion/collapse.

## Development interpretation

This five-case audit is diagnostic and cannot verify a predictor or support a
generalization claim.

Evidence that terminal speed is a limiting action dimension requires:

1. the stronger variant delivers a materially lower realized terminal speed;
2. at least four of five cases at a candidate level retain strict dose and
   compact geometry;
3. at least four of five strict cases are safety-admissible in this development
   audit (the later fresh pilot keeps its stricter 90% criterion);
4. the stronger terminal speed creates meaningful treatment-effect changes,
   rather than only more braking or delay;
5. beneficial and nonbeneficial outcomes remain present across cases.

If one level consistently dominates, freeze it and retain a binary
damping-versus-wall gate. If different states prefer different safe levels,
the next scientific treatment can be a multi-action gate over damping, mild,
and strong walls. If lower terminal speeds only worsen safety, redesign
activation timing or release rather than increasing braking authority.

## Command

~~~powershell
python experiments/counterfactual/run_counterfactual_terminal_speed_sensitivity.py --workers 2
~~~
