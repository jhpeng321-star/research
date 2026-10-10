# IDM v5.5.1 adaptive hold/release no-go diagnosis

Status: development-only no-go, 2026-09-01.

The v5.5.1 four-case mechanism repair is rejected under its predeclared gate.
It is not valid to add seeds, train the predictor, tune the thresholds, or
present v5.5.1 as a safe wall controller.

## Failed gates

1. **Safety:** 4401 and 4402 still exceed the frozen hard-braking budget.
2. **Lifecycle integrity:** 4404 receives a hard safety abort during its
   adaptive taper, so it is not a completed policy even though its aggregate
   safety/service comparison happens to be admissible.
3. **Mechanism repair:** soft release does not cover the 4401 downstream
   pathway, does not make the physically beneficial 4402 safe, and itself is
   unsafe in 4404.

The treatment retains material authority and full lane coverage, so this is
not a hidden zero-dose failure. The failure is also not a replay mismatch:
predecision identity, damping replay/action-path checks, common horizon, and
exact damping handoff pass for the completed cases.

## Interpretation and defensible fallback ranking

1. **Decentralized-authority-boundary paper — recommended now.** It supports
   the honest scientific result: decentralized P30 damping has bounded
   authority; a coordinated wall can create physical benefit, but this compact
   late-overload implementation is not safely deployable across the tested
   development cases. It fits the observed safe 4405 shortening and unsafe
   4402 prevention without overstating a predictor.
2. **Review-only revised formation geometry.** A wall with a different spatial
   formation/release geometry could be a separate intervention, but it must
   receive new scientific approval, a frozen feasibility/dose contract, and a
   fresh mechanism gate. It cannot be silently called v5.5.1.
3. **Review-only state-responsive timing/duration study.** Literature on
   moving bottlenecks and jam absorption supports state-dependent authority,
   but it changes the treatment definition; it is not a threshold tweak.
4. **Safety/service-only paper.** Viable only if the scope clearly states that
   hard-brake proxies and service responses are simulation safety indicators,
   not real-world crash-risk claims.

The literature supports the mechanism premise but not a universal local
follower threshold: Čičić et al. show moving-bottleneck/platoon efficacy is
state-, severity-, and actuator-availability-dependent; Han and Ahn show merge
breakdown is stochastic and conditioned on headways, demand and merging.
Neither source validates this implementation as safe, calibrated, or generally
predictable.

No v5.5.2 design is implemented or authorized by this diagnosis.
