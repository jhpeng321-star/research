# Reference list for the traffic-shepherd research

Date: 2026-08-06
Purpose: curated bibliography mapped to the project's central findings, for
the write-up and for the "CAV metering induces HDV–HDV TTC" mechanism line.
Collected via arXiv API + targeted web search (search quota exhausted mid-run;
coverage is partial but the load-bearing references for each claim are here).

## TTC threshold / surrogate safety (justifies the 1.5 s ruler)

- **SSAM default conflict threshold = 1.5 s TTC** (FHWA Surrogate Safety
  Assessment Model). Summary w/ SSAM defaults: MDPI, *Driving Automation
  Systems Penetration and Traffic Safety* (2024) — https://www.mdpi.com/2412-3811/9/12/234
- **"Typically 1–5 s, with 1.5 s commonly used"; high-risk tier = 1.5 s,
  medium = 3 s, low = 5 s** — US DOT / National Transportation Library study —
  https://rosap.ntl.bts.gov/view/dot/79428
- **Research range 1.5–3.0 s** (Hayward 1971/1972; Migletz 1985; Van der
  Horst 1990; etc.) — National Academies Press guide —
  https://nap.nationalacademies.org/read/27281/chapter/4
- **SIMONE: <3 s uncomfortable, <1.5 s dangerous; 1.5 s ≈ human
  perception–reaction** — iDREAMS D2.1 —
  https://idreamsproject.eu/wp-content/uploads/2020/09/iDREAMS_814761_D2.1_28022020_Final.pdf

## Wave damping by a single CAV (the "gradual metering" claim, Stern)

- **Stern, R. et al. (2017). Dissipation of stop-and-go waves via control of
  autonomous vehicles: Field experiments.** arXiv:1705.01693. A single AV
  with slow sinusoidal speed modulation dissolves ring-road waves. This is
  the basis of the user's gradual-metering hypothesis — and the regime
  (stable ring flow) in which it holds.

## Anticipatory / long-approach metering and upstream reference (LCC)

- **Wang, J., Zheng, Y., et al. (2020). Leading Cruise Control in Mixed
  Traffic Flow: System Modeling, Controllability, and String Stability.**
  arXiv:2012.04313 (also arXiv:2007.11753). One CAV with upstream info can
  in principle stabilize arbitrarily long platoons.
- **Wang, J. et al. (2023). Information Flow Topology in Mixed Traffic: A
  Comparative Study between "Looking Ahead" and "Looking Behind".**
  arXiv:2309.01625. Upstream-referenced control beats ahead-referenced for
  damping upstream-propagating waves.
- **DeeP-LCC: Data-EnablEd Predictive Leading Cruise Control in Mixed
  Traffic Flow.** arXiv:2203.10639. Experimental CAV wave dissipation.

## Penetration as the stability lever (supports the p20 result)

- **Stability of multi-population traffic flows.** arXiv:2201.00381. A
  critical penetration fraction of "stable" (automated) vehicles exists
  above which mixed traffic stabilizes — the theoretical frame for the
  empirical p10→p20 gain.
- **Traffic Control via Connected and Automated Vehicles: An Open-Road Field
  Experiment with 100 CAVs (CIRCLES).** arXiv:2402.17043. Real-world
  smoothing with ~2% CAVs on an open road — the field validation that
  penetration-based flow smoothing works beyond simulation.

## Merging is the hard case (why the coupling is stubborn here)

- **Effect of adaptive cruise control systems on mixed traffic flow near an
  on-ramp** (2005). Simulated ACC near an on-ramp can *destabilize* — merges
  add lane-changing perturbations that longitudinal smoothing alone does not
  remove. Journal (not on arXiv).
- **L2 / L∞ stability analysis of heterogeneous traffic** (2016). CAVs
  should "absorb not propagate" perturbations; framework for CAV-as-absorber
  objectives. Journal (not on arXiv).

## Cooperative merging / ramp control (adjacent tools, not yet tried here)

- **Decentralized Cooperative Merging of Platoons of Connected and Automated
  Vehicles at Highway On-Ramps.** arXiv:2002.04826.
- **Evaluation of Connected Vehicle Identification-Aware Mixed Traffic
  Freeway Cooperative Merging.** arXiv:2405.12464.

## RL for mixed-autonomy traffic (RL AVs as flow actuators)

- Vinitsky, E. et al. (2018). *Benchmarks for reinforcement learning in
  mixed-autonomy traffic* (Flow). [arXiv id to confirm on next search.]
- Orosz et al., Connected Cruise Control series (e.g. arXiv:2210.04397,
  arXiv:2309.00074, arXiv:2409.06884) — CBF-based safety for CCC.

## How this maps to the project's findings

| Project finding | Anchoring reference |
|---|---|
| 1.5 s ruler is the standard critical-TTC threshold | SSAM / NAP / iDREAMS |
| Gradual single-CAV metering does NOT reduce HDV-HDV TTC at dense merge (probe falsified) | Stern works only in the stable-ring regime; merge (on-ramp ACC) case destabilizes |
| Upstream/looking-behind reference does NOT help at dense p10 (probe falsified) | LCC theory assumes the CAV can act on the platoon; at 10% with a short merge zone the disturbance is geometric |
| Penetration raise (p20) improves the scenario and holds policy transfer | multi-population stability; CIRCLES 100-CAV field test |
| Merge corridor is the worst case for longitudinal smoothing | on-ramp ACC destabilization (2005) |

Note: coverage is partial (search quota). Suggested next literature passes for
the write-up: (1) Gunter/Bullo ring-road string-stability bounds, (2) a
recent mixed-traffic merge-control RL survey, (3) V2X advisory-speed /
"HDV-assist" methods where humans receive the slowdown instead of braking —
the one mechanism-class the project has not modeled.

## Second pass (2026-08-06, OpenAlex) — new angles for the write-up

- **Adaptive control with moving actuators at motorway bottlenecks with CAVs**
  (2023), DOI 10.1016/j.trc.2023.104319 — CAV-as-moving-bottleneck control
  theory, directly frames our metering mechanism.
- **Penetration effect of CAVs on cooperative on-ramp merging** (2019),
  DOI 10.1049/iet-its.2019.0488 — how the fraction of *cooperative* CAVs
  changes merge outcomes; supports the placement/penetration levers.
- **Multi-agent RL for cooperative lane changing in mixed traffic** (2022),
  DOI 10.1007/s43684-022-00023-5 — the one genuinely NEW action dimension:
  coordinated *lane-choice* smoothing (current policy is longitudinal-only).
- **Robustness and Adaptability of RL-based Cooperative Autonomous Driving in
  Mixed-autonomy Traffic** (2022), arXiv:2202.00881 — why RL mixed-traffic
  policies are basin-fragile (explains the fresh-retrain drift).
- **Eco-Driving of General Mixed Platoons With CAVs and HDVs** (2022),
  DOI 10.1109/tiv.2022.3224679 — low-penetration cooperative driving.
