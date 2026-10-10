# Paper skeleton — learned metastable corridor control under IDM (IEEE IV draft)

> Historical wall-first manuscript. Its claims and numbers are not accepted
> evidence for the restored standalone-trained-damping project. Preserve this
> text as earlier work; do not import its results into the current paper without
> a separate provenance and applicability audit. Current methods are drafted in
> `TRAINED_DAMPING_METHODS_DRAFT_2026-09-05.md`; the active route is
> `TRAINED_DAMPING_TO_PAPER_ROUTE_2026-09-05.md`.

## Working title
**"Learning to hold the lower branch: a moving-bottleneck wall against metastable
breakdown under realistic car-following (IDM)"**

## Abstract (draft, fill exact numbers from the firmed table)
Freeway merge corridors exhibit stochastic breakdown near capacity — collapse is often
preventable but not by smooth, low-penetration CAV damping alone. We study a corridor-level
"wall": a moving bottleneck formed by CAVs at 20% penetration, throttling throughput at the
merge throat via headway-as-throttle (follow-gap tau). Under Krauss car-following the wall
is superfluous (breakdown is rare and collapse kinematically cheap). Under IDM — string
unstable, capacity-drop-producing, the standard CF model in the capacity literature —
breakdown becomes probabilistic at 2000-2200 veh/h and certain at 2400+, and collapse
carries a real delay premium (+2.3-3.4 s/veh). A single-agent RSU supervisor learned with
RL (health-signal observation, potential-based reward shaping) engages the wall: it cuts
the collapse probability from 0.86 to 0.08 at 2200 (p<0.001 vs the hand-tuned rule), from
1.00 to 0.46 at 2400 at zero measurable delay cost (22.0 vs 22.6 s, n=50), and reduces
safety-critical events 3.5×. Under persisting breakdown (2700) the wall's managed
discharge still saves 4.4 s/veh (p=0.005). We measure the delay-protection frontier
completely — follow-gap, wall speed, band length, penetration, and the damping-composition
negative control — and show the learned wall dominates the hand-tuned frontier.

## Section outline with measured numbers

### 1. Introduction
- The problem: stochastic breakdown at merge bottlenecks (capacity drop, hysteresis —
  Banks 1991, Hall & Agyemang-Duah 1991, Cassidy & Bertini 1999, Persaud 1998).
- Prior control: ALINEA ramp metering (Papageorgiou 1991), VSL — both queue-forming,
  delay-expensive. Our contrast: move the bottleneck INSIDE the corridor as a CAV wall.
- Claim: a learned engagement policy generalizes where hand-tuned rules degrade, and the
  gap between rule and learned IS the point.

### 2. The wall mechanism
- Headway-as-throttle: throughput = v/(v·tau + minGap); tau is the throttle, speed is the
  delay dial. Duty-cycle = convoy transit / cycle period; at 20% pen the wall exists a
  fraction of the time — sufficient because breakdown onset is slow (minutes).
- Gentleness: rate clip 0.3 m/s², tau ramp — the wall must not nucleate the wave it
  prevents (hard-brake audit: 0 steps).
- The leak (Krauss): isolated damping can't throttle because HDV followers keep their own
  tau — why the champion (damping) fails; the wall overrides headways directly.

### 3. The IDM regime (the pivot — main chapter)
- Critical-density shift: no-op collapse probability vs demand — Krauss: 0.36 @2400,
  0.76 @2600, 0.90 @2700; IDM: 1.00 @2400+ (n=50). IDM breaks down ~10-15% earlier.
- Collapse premium: ~0 under Krauss (−0.8 to +0.8 s); +2.3-3.4 s at 2400-IDM;
  −4.4 to −6.2 s recoverable under the wall at 2700 (n=50, p=0.005 / p=0.0001).

### 4. Results — the workability map (all headline cells n=50)
**Table: demand × arm (P · delay · TTC):**
- 2000: no_op 0.40·11.6·0.6 | trigger 0.12·11.9·0.2 | sup_v43 0.06·11.8·0.3 | sup_idm 0.02·12.6·0.2
- 2200: no_op 0.86·14.5·1.1 | trigger 0.48·14.9·0.6 | **sup_v43 0.08·15.5·0.4 (p<0.001 vs rule)** | sup_idm 0.08·17.1·0.4
- 2400: no_op 1.00·22.6·2.8 | trigger 0.85·19.6·0.8 | tau-4 rule 0.48·20.6 | sup_v43 0.46·22.0·0.8 | sup_idm 0.34·25.3·0.8 | VSL-10 1.00·67.3·2.5
- 2700 (p=0.2): no_op 1.00·47.9·4.1 | trigger 1.00·41.7 | sup_v43 0.96·43.5 (delay p=0.005)

**R2 — penetration surface (2700):** p0.4: sup_v43 0.55@45.0 DOMINATES trigger 0.75@51.5;
p0.6: sup_idm 0.05@106.3 (ceiling), sup_v43 0.20@54.5, trigger 0.15@60.9.

**R3 — the tau law:** follow-gap 3→4: rule protection 0.85→0.48 at +1.0 s (2400);
speed and band length secondary (w8 flat; band 200-300 m buys delay by surrendering
protection: 0.46→0.90).

### 5. Negative controls (one figure each)
- Damping champion + wall (both physics): P=1.00 — antagonistic roles; motivates the
  single-agent RSU architecture with role separation.
- VSL-10 (uniform speed limit): P=1.00 at 67.3 s — 3× the wall's delay; blind speed
  control forms the very queue the wall avoids.
- Assembly (ewall): delay-optimal (36.6 s @2700) but protection-free — placement matters
  for protection, not for discharge management.

### 6. Honesty / limitations
- Delay claim scoped: wins vs no control at 2700 (p=0.005), ties at 2400, +0.6-1.0 s in
  the band; unconditional delay superiority is frontier-forbidden (the cheap trigger's
  19.6 s IS its 0.85 collapse).
- Learned-vs-rule at 2000: parity (rule tightens to 0.12 at n=50).
- Chain: SUMO TraCI; IDM default params; single corridor geometry; generalization across
  maps/demand shapes untested beyond the probed ladder.

## Figure list
1. Mechanism schematic: headway-as-throttle + duty cycle.
2. Breakdown probability vs demand: Krauss vs IDM (critical-density shift).
3. The band: no_op vs trigger vs sup at 2000/2200 (P + delay + TTC).
4. The 2400 frontier: P vs delay scatter of ALL arms (frontier + cheat region + learned).
5. Penetration surface at 2700 (P vs delay vs pen for trigger/sup_v43/sup_idm).
6. Negative controls: VSL time series + stack ablation.
7. (optional) Hysteresis: (density, flow) or (delay, health) two-branch scatter.

## The claim set (final wording)
- C1 (mechanism): a moving-bottleneck formed by 20% CAVs can throttle a merge throat at
  zero hard-brake risk (audited).
- C2 (regime): under IDM, breakdown is stochastic in 2000-2200 and certain at 2400+; the
  collapse premium is real (+2.3-3.4 s), unlike under smooth car-following.
- C3 (learned engagement): a health-observing RSU policy generalizes where hand-tuned
  triggers degrade — 0.08 vs 0.48 collapse at 2200 (p<0.001), 0.46 vs 1.00 at 2400 at
  zero delay cost (n=50).
- C4 (safety): TTC events cut 3.5-10× vs no control.
- C5 (economics): managed discharge under persisting breakdown saves 4.4-6.2 s/veh
  (p≤0.005, n=50) — the wall pays where collapse is expensive.
- C6 (frontier, measured not assumed): every mechanism knob trades delay for protection;
  the learned model sits below the hand-tuned frontier (4.9 s cheaper than the long-band
  rule at matched protection, p<0.001).
