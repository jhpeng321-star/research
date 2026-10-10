# Literature scan: untried levers for the CAV-meterg TTC coupling

Date: 2026-08-06
Purpose: find NEW levers for the core problem — CAV metering at a dense merge
induces HDV–HDV TTC events (the coupling is falsified across reward,
calibration, density, execution, and mechanism axes). All candidate levers
below are **infrastructure/RSU-level**, chosen because they avoid the
mainline CAV changing speed — the proven harm source.

Search method: OpenAlex + arXiv APIs (the WebSearch quota was exhausted);
queries across speed-advisory, cooperative-merging, ramp-metering, MPC
trajectory, CACC-topology, and placement mechanism classes.

## The key reframe

Everything we tried made the **mainline CAV** absorb the ramp disturbance
(by changing its own speed) — and that speed change *is* the wave source.
The literature's merge-specific strategies instead shift the disturbance
handling to (a) the **ramp** or (b) **all drivers together**. Three untried
lever classes follow.

## Lever 1 — Ramp metering (infrastructure; strongest candidate)

Meter the **ramp inflow** with an RSU-controlled signal (classic ALINEA),
so the mainline CAV does not need to create gaps by slowing. The disturbance
source is limited before it reaches the mainline.
- Ref: *Variable Speed Limit and Ramp Metering for Mixed Traffic Flows: A
  Review and Open Questions* (2021), DOI 10.3390/app11062574.
- Ref: *Integrated optimal control strategies for freeway traffic mixed with
  CAVs: a model-based RL approach* (2021), DOI 10.1016/j.trc.2021.102987
  (RL at the infrastructure level, not the vehicle level).
- Why it's new: the RSU becomes an **actuator** (meters the ramp) instead of
  just informing the CAV. The mainline CAV can coast or meter far less →
  fewer waves.
- Risk: metering the ramp raises ramp delay, which may worsen the fairness
  gap (|main − ramp| delay) — acceptable under the TTC-first priority but
  must be measured.
- **Testable in SUMO:** a signalized ramp controller is straightforward.

## Lever 2 — VSL speed harmonization upstream (infrastructure)

Post a gradually-descending variable speed limit **upstream**, so every
human driver decelerates together instead of braking on the CAV's taillights.
Implements the user's "HDVs should not be doing any changes" directly — the
slowdown is requested of *everyone* by one signal, not imposed by one CAV.
- Ref: *Variable speed limit: a microscopic analysis in a connected vehicle
  environment* (2015), DOI 10.1016/j.trc.2015.07.014.
- Testable in SUMO: dynamic edge speed-limit changes on the upstream mainline.

## Lever 3 — Cooperative / slot-based merge (hardest, most invasive)

Coordinate the **merging vehicle** to slot into a mainline gap so the
mainline CAV never decelerates. Requires controlling/negotiating the merging
vehicle's trajectory, which the current env does not model (HDV merge is
free-flow).
- Ref: *Merging control strategies of connected and autonomous vehicles at
  freeway on-ramps: a comprehensive review* (2022), DOI 10.1108/jicv-02-2022-0005.
- Ref: *Flow-level coordination of CAVs in multilane freeway ramp merging
  areas* (2022), DOI 10.1016/j.multra.2022.100005.
- Testable only with a cooperative-merge model; lowest priority for a probe.

## Already covered / not applicable (for completeness)

- **CACC / dynamic information-flow topology** (e.g., *Lp string stability*,
  DOI 10.1109/tcst.2013.2258346): assumes platoon structure and high CAV
  density; our isolated p10 CAVs don't have the topology to exploit it.
- **Dedicated CAV lanes** (DOI 10.1016/j.trc.2020.102664): geometric
  segregation; heavy, not a near-term probe.

## Recommendation

Implement **Lever 1 (ramp metering)** first as an eval-time-only probe on the
champion: RSU meters the ramp inflow; the mainline CAV policy is unchanged.
Read: if TTC-worse drops toward/below the p10 bar (14) while delay holds,
the reframe is validated — the harm was the CAV absorbing ramp inflow, and
the RSU can take that job over. If ramp metering instead just shifts delay to
the ramp without cutting TTC, then even a pre-metered ramp cannot remove the
coupling and the penetration result remains the headline.
