# IDM Decentralized Ranked-Neighbor Validation Protocol

Status: frozen before opening seeds 4121--4132. This is development-only and
does not establish a trained policy.

## Development diagnosis being tested

The unchanged fixed-p30 action produced a clean near-critical operating point
at 2000 veh/h: 33/33 adequate and safety/service-admissible treatments, with
14 helpful, 13 neutral and 6 harmful outcomes. Mean qualified effects were
positive at all three recurrent risk onsets. In contrast, 2100 and 2200 retain
insufficient or unreliable decentralized authority and remain escalation
headroom rather than a reason to brake harder.

The current 134-field actor observation failed the frozen lower-loading gate
(LOSO AUROC 0.676, balanced accuracy 0.641). Post-hoc diagnosis found that the
ordered first three nearest-neighbor slots alone were the smallest plausible
family: pooled lower-loading AUROC 0.85 and balanced accuracy 0.78; within the
2000 cell AUROC 0.70 and balanced accuracy 0.71. The existing trust-weighted
DeepSets mean erases this distance-rank structure.

## Frozen hypothesis and prohibited shortcuts

The only proposed representation is an order-preserving MLP over the first
three nearest-neighbor slots (15 raw fields: signed relative distance, relative
speed, relative lane, turn intent and trust weight for each slot). Existing
ego, zone, pooled-neighbor and controller branches may remain available, but
no new field may be selected after this validation opens.

Configured demand, seed, branch identity, recurrence index, realized dose and
future outcomes are prohibited inputs. The policy must infer its operating
domain from causal local traffic state. The treatment, p30 reference, 20-second
duration, safety projection, reward references and service constraints remain
unchanged.

## Fresh development cells

- Seeds: 4121--4132, permanently excluded from checkpoint acceptance.
- Demands: 1900, 2000, 2100 and 2200 veh/h; ramp remains 400 veh/h.
- First, second and third persistent `risk_onset`, when present.
- One runway-ranked admissible CAV.
- Exact abstain, abstain replay and fixed-p30 activation branches.
- 60-second outcome horizon and the unchanged 0.02 effect deadband.

Seeds 4201--4240 remain unopened. No missing landmark may be replaced.

## Pass/fail gates

Integrity and treatment:

- exact branch/discovery identity and abstain replay;
- complete horizons, zero collisions and zero teleports;
- at least 80% adequate dose and 90% safety/service admissibility;
- at least 24 qualified nonneutral decisions;
- helpful and harmful outcomes in at least three seeds each.

Representation:

- pooled seed-grouped LOSO AUROC at least 0.70 and balanced accuracy at least
  0.65 for the frozen 15 ranked-neighbor fields;
- within the 2000 cell, AUROC at least 0.65 and balanced accuracy at least 0.60;
- demand-only diagnosis is reported but never supplied to the actor.

Near-critical action domain:

- at 2000, mean qualified safety effect at least +0.02;
- no more than 35% of qualified 2000 cases harmful;
- 2000 tenth-percentile effect no worse than -0.25 reference units;
- every 2000 treatment remains inside the frozen per-case safety/service bounds.

Reward readiness is separate from representation readiness. Before RL, the
paired cooperative reward must retain effect correlation at least 0.80 and
correct sign for at least 95% of substantial effects (`abs(effect) >= 0.06`).
Small/neutral cases require conservative supervised abstention; they cannot be
used to justify a long RL run.

Passing authorizes implementation and unit testing of the ranked-neighbor
encoder, supervised conservative initialization, and a 2000-step smoke. It
does not authorize 10,000 steps, checkpoint promotion or any wall/predictor
claim. Failure requires action/credit redesign, not additional feature mining
on these outcomes.
