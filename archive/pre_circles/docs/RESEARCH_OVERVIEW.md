# Research Overview — Metastable Health-Gated CAV Wall Control

Workspace: `D:\research-dense` (SUMO + TraCI + Ray RLlib; conda env `rl_driving`).
Date of this document: 2026-08-24. Prepared as a self-contained orientation for a
reviewer agent; every number below is measured from row files in `runs/` unless
marked "pending" or "stopped".

---

## 1. The idea in one paragraph

A freeway merge corridor exhibits stochastic ("metastable") breakdown near capacity.
A corridor-side RSU observes a **health signal h** — an AND-composite of
density / speed / variance conditions from the merge zone, gated to the free-flow
regime — and uses it (with hysteresis: arm h<0.25, release h>0.45) to engage a
**wall**: a moving bottleneck formed by 20%-penetration CAVs (headway-as-throttle;
wall speed 10 m/s, follow-gap tau=3 s, gentle 0.3 m/s² envelope, duty-cycled convoy
formation). Separately, a **damping layer** (a per-CAV RL champion, modes
coast/recover/guard) is intended for the healthy/low-density regime. The measured
program characterizes both layers across two car-following physics models
(Krauss = smooth; IDM = string-unstable), eight densities (2000-2900 veh/h), three
penetrations (20/40/60%), and five control families (none, damping, ALINEA ramp
metering, VSL, the wall family).

---

## 2. Definitions and mechanics (used everywhere below)

- **Health h** ∈ [0,1]: AND of (rho above threshold) ∧ (speed below threshold) ∧
  (variance above threshold), each gated to the free-flow regime; EMA-smoothed.
  Collapse metric: **sustained collapse** = h < 0.15 for ≥ 60 consecutive steps
  after step 100 (episode = 1800 steps ≈ 30 min).
- **Wall**: moving bottleneck at distance-to-merge band 200-400 m (800 m variant),
  wall_speed 10 m/s, wall_tau 3.0 s (set via TraCI setTau on in-band CAVs),
  gentle rate clip 0.3 m/s² (audited: 0 hard-brake steps under Krauss).
  Throughput law: flow = v/(v·tau + minGap); tau is the throttle, speed the delay dial.
- **Duty cycle**: protection tracks the fraction of time a wall exists
  (convoy transit / cycle period); explains why 20% CAVs suffice at low density
  and why higher penetration rescues protection at high density.
- **The leak**: isolated CAV braking cannot throttle an HDV stream (HDVs keep their
  own tau) — hence the wall overrides headways directly; also why per-CAV damping
  is weak when the corridor is HDV-dominated.
- **Row files**: every episode is one JSON row (`schema:2`) under
  `runs/probe/metastable_feasibility/stage10_{tag}` with fields: variant, seed,
  loss (mean per-vehicle delay, s), ttc_events, sustained_collapse, spill_min_m,
  first_collapse_step, outflow_coll/free, inflow_mean, fair_gap, h stat, etc.
  no-op rows live in `runs/eval/noop_cache_{tag}`. Tags: `light_m{demand}r400`
  = Krauss long-map; `light_m{demand}r400idm` = IDM; `_penNN_` = penetration scope.

---

## 3. Results — Krauss (smooth physics; the paper's demonstration physics)

All headline cells n=200 (except flagged). P = collapse probability; loss = mean delay.

| demand | no control | damping champion | ALINEA (o=0.15) | wall trigger | RSU+wall (sup_v43) |
|---|---|---|---|---|---|
| 2400 | 0.36 · 11.0 s | 0.25 · 10.9 s (n=20) | 0.40 · 16.2 s (n=20) | 0.02 · 12.3 s | **0.03** · 12.1 s |
| 2600 | 0.76 · 13.3 s | 0.70 · 13.5 s (n=20) | — | 0.10 · 15.7 s | **0.05** · 16.5 s |
| 2700 | 0.90 · 14.5 s | 0.90 · 14.5 s (n=200) | 0.95 · 19.0 s (n=20) | 0.21 · 18.2 s | **0.11** · 19.6 s |
| 2800 / 2900 | 0.94 / 1.00 (no control only) | | | | |

TTC (2700): no control 3.8, damping 4.1, trigger 1.9, RSU+wall 2.9.
Wall+damping stack (2700, concurrent): P=0.94 · 14.4 s — the stack destroys the
wall's protection (damping and wall are antagonistic; exclusivity is a design rule).

**Two-layer arc (Krauss)**: the damping champion protects directionally at the
onset densities (collapse −0.06..−0.11, delay parity; n=20 cells NOT significant)
and saturates to *exactly no control* at 2700 (n=200). The health-gated wall
carries the high-density regime (0.90→0.11, n=200). ALINEA — canonical ramp
metering — protects nothing on Krauss and costs +4.5-5.2 s (the collapse is
mainline-driven, not ramp-driven). Under Krauss the hand-tuned trigger already
achieves 0.02-0.21: the *learned* engagement's decisive advantage is an IDM result
(see §4), not a Krauss result.

---

## 4. Results — IDM (string-unstable physics; the stakes/economics chapter)

No control collapses ~always from 2400 up (ladder: 2000: 0.40, 2200: 0.86,
2250+: 1.00). The **collapse premium** (delay cost of a collapsed episode) is
+2.3-3.4 s at 2400-IDM vs ~0 under Krauss — the anchor for the paper's economics.

| demand | no control | wall trigger | tau-4 rule | RSU+wall | RSU (IDM-trained) |
|---|---|---|---|---|---|
| 2000 (n=50) | 0.40 · 11.6 s | 0.12 · 11.9 s | 0.05 · 12.1 s (n=20) | **0.06** · 11.8 s | 0.02 · 12.6 s |
| 2200 (n=50) | 0.86 · 14.5 s | 0.48 · 14.9 s | 0.25 · 15.0 s (n=20) | **0.08** · 15.5 s | 0.08 · 17.1 s |
| 2300 (n=50) | 0.90 · 17.0 s | 0.64 · 17.2 s | — | **0.22** · 18.0 s | 0.30 · 21.9 s |
| 2400 (n=50) | 1.00 · 22.6 s | 0.82 · 19.6 s | 0.48 · 20.6 s | **0.46** · 22.0 s | 0.34 · 25.3 s |
| 2600 (n=20) | 1.00 · 36.7 s | 1.00 · 32.7 s | — | 0.70 · **31.3 s (−5.4, p=0.006)** | — |
| 2700 (n=50) | 1.00 · 47.9 s | 1.00 · 41.7 s | — | 0.96 · **43.5 s (−4.4, p=0.005)** | — |

Fine ladder (n=20): 2250: no 1.00@15.9 / sup 0.10@16.5; 2350: no 1.00@19.6 / sup
0.20@19.7. VSL-10 @2400: P=1.00 · 67.3 s. Ramp gate (all-stop on health) @2400:
P=1.00 · 147.2 s; @2700: P=1.00 · 180.7 s. ALINEA @2400-IDM: P=0.95 · 28.1 s
(all o_targets 0.10/0.15/0.20 clamp identically). Wall+champion stack @2400-IDM:
P=1.00 · 20.8 s.

**Key IDM statements**:
- The learned supervisor beats the hand-tuned trigger decisively at the band
  (2200: 0.08 vs 0.48, p<0.001; delay +0.6 s) and dominates at 2700-p0.4
  (0.55@45.0 vs trigger 0.75@51.5).
- Protection degrades with demand (0.06 → 0.46 → 0.96) while **delay flips to
  significant wins** at 2600/2700 (−5.4/−4.4 s; paired Wilcoxon p≤0.011) — the
  managed-discharge channel: the wall orders the discharge of a persisting
  breakdown (throughput unchanged; spill contained ~400-600 m closer to the throat).
- **Tau law**: follow-gap 3.0→4.0: rule protection 0.82→0.48 @2400 at +1.0 s;
  wall speed and band length are secondary (w8: 0.82; w12: 0.90; short band
  b300 buys delay by surrendering protection: 0.46→0.90). Gap-dial (tau-4 rung
  imposed on trained policies) is inert-to-harmful: the gap must be learned, not
  imposed.
- The 2300 cell's earlier +1.7 s delay anomaly was noise (n=50: +1.0 s).

**Penetration surface @2700-IDM (n=20/cell)**:

| pen | no control | trigger | sup_v43 | sup_idm |
|---|---|---|---|---|
| 0.2 | 1.00 · 47.9 s | 1.00 · 41.7 s | 0.96 · 43.5 s | (0 rows) |
| 0.4 | 1.00 · 45.3 s | 0.75 · 51.5 s | **0.55** · 45.0 s | 0.50 · 76.7 s |
| 0.6 | 1.00 · 42.5 s | **0.15** · 60.9 s | 0.20 · 54.5 s | **0.05** · 106.3 s |

Penetration buys protection at an increasing delay price ("the surface is the
frontier in penetration space"). Under smooth physics, penetration ALONE reduces
uncontrolled collapse (Krauss 2700: P 0.98→0.72 from 5%→50% CAV — a different
baseline population, see caveats); under IDM it does not (P=1.00 at all pens).

---

## 5. Negative results (each measured; the paper's boundaries)

1. **Damping under IDM is inert or harmful** (the whole campaign):
   - trained champion alone: null at 2000 (0.47≈0.40), 2200 (0.87≈0.86), 2400
     (1.00, all axes n.s.);
   - uniform scripted damping: harmful at 2400 (28.0-28.8 s vs 22.6, p<0.01),
     null at 2200;
   - health-triggered scripted damping: null at n=50 (15.0 s/1.14 vs 14.5/1.1 —
     the n=20 TTC "trend" was noise);
   - wave-filter cruise (I-24/Stern shape, own-EMA and leader-EMA): catastrophic
     continuous (P=1.00, 2-2.6× delay), parity gated;
   - spatial-average replica (Martínez multi-leader + healthy-gated polarity):
     the mechanism was first mis-probed (trigger 1.01 could never fire; import
     crash on the gated branch), fixed, then null when engaged.
   - Literature context: damping works in rings (<5%, Stern et al.) and locally
     on open highways (I-24/MegaVanderTest); open merge corridors with 20% CAV,
     local observations, and a demand transient violate its conditions.
2. **Ramp metering family fails**: ALINEA (P≥0.95, +4.5-5.5 s, both physics),
   VSL-10 (1.00 @ 67.3 s), binary ramp gate (1.00 @ 147-181 s) — because at
   2400+ the mainline stream alone is overcritical (verified: gating the ramp to
   zero leaves P=1.00).
3. **Structural shortage (stall vehicle)**: P=1.00 in ALL arms with the wall
   active from t=0 (2000-2700-stall); sup_v43 adds +23 s at 2700-stall. A
   structural capacity loss cannot be throttled away — upstream control only
   relocates the inevitable queue.
4. **Damping + wall composition (stack)**: P→1.00 in both physics — exclusivity
   rule: the two layers never run concurrently.

---

## 6. The frontier (why delay cannot "win" everywhere)

E[delay] = healthy-floor + engagement-cost − premium × prevented-share.
At 2400-IDM the throttle cost (~2.3 s on healthy episodes) ≈ the collapse premium
(2.3-3.4 s) × share — revenue-neutral by construction; unconditional delay wins
vs the collapse-cheat trigger (19.6 s at P=0.82) are frontier-forbidden (its delay
advantage IS its collapse rate). Every swept knob (speed, tau, band length,
penetration, engagement timing, physics) trades delay for protection along the
same measured frontier; the learned policies sit below the hand-tuned frontier
(4.9 s cheaper than the long-band rule at matched protection, p<0.001).

---

## 7. Assets

**Trained policies** (checkpoints under `runs/rllib/`):
- `wall_supervisor_v44/checkpoints/best` → evaluated as **sup_v43** (Krauss-trained;
  rungs = 200-800 m band, w10 t3 / w12 t3). Model of record.
- `wall_supervisor_idm/checkpoints/best` → **sup_idm** (IDM-native; protection
  specialist: 0.34@2400, 0.02@2000, ceiling 0.05@2700-p0.6 at +delay).
- `shepherd_pruned_v1_dense_mpen/checkpoints/best` → the pruned damping champion
  (coast/recover/guard).
- `wall_supervisor_econ` — sup_econ (delay-weighted fine-tune, delay_norm 40 +
  collapse_bonus 400) — **training stopped mid-run (~iter 36/50, ~288k steps);
  never evaluated. Restartable with `train_supervisor.py --resume-from ...`.**
- `champ_idm_wave` — wave-gated damping retrain — **stopped at boot; never trained.**
- Demo/closed: BC models, collapse-risk gate (symptom predictor, evaluated
  negative), smoothed/wave-filter policies (probe-only, `policies/inflow_control.py`:
  SmoothCruisePolicy, SpatialCruisePolicy, RampGatePolicy, AlineaPolicy, GatePolicy,
  WallHoldPolicy, VSLPolicy, RampGatePolicy+).

**Probe driver**: `_probe_metastable_feasibility.py --stage 10 --scene D
--tag light_mD{r400|r400idm} --variants a,b,c --steps 1800 --workers 4
--map-root D:/research/highway_merge_shepherd_long --pen 0.2 --seeds 101-120`

**Variants**: no_op, w10_t30_h025 (+_b300, _lb800), w10_t40_h025, w8_t30_h025,
w12_t25_h025, ewall_12_t25_h025, sup_v43, sup_idm, sup_v43_t40, sup_idm_t40,
vsl_10, ramp_gate, ramp_gate_wall, alinea_o10/15/20, rl_pruned3, wall_plus_champ,
sdamp_g02/g06/trig, smooth_cont/h/v2/v2_cont, champ_idm/stack_idm/champ_wave/
stack_wave (the last four need their checkpoints; champ_idm/champ_wave paths are
empty until trained).

**Demand variants** (long map `D:\research\highway_merge_shepherd_long\_light\`):
Krauss: light_m2400/2500/2600/2700/2800/2900r400; IDM: light_m2000/2200/2250/
2300/2350/2400/2600/2700r400idm; stall variants at 2000-2700; penetration via
probe `--pen`.

---

## 8. Charts (professor set)

`runs/probe/charts_professor/`:
- `krauss_two_layer_4panel.png` + `krauss_2700_two_layer.png` — Krauss two-layer
  story (Panel 1: onset map; Panel 2: 2700 flagship; Panel 3: protection across
  onset; Panel 4: h-hysteresis decision rule).
- `krauss_density_two_layer.png` — Krauss-only: no control/damping/ALINEA?/wall
  vs density (built before ALINEA existed; ALINEA bars can be added).
- `fig_idm_ladder.png` — IDM P/delay/TTC vs demand.
- `fig_idm_pen_surface.png` — 2700-IDM penetration surface.
- `fig_krauss_pen.png` — penetration contrast (Krauss decays, IDM flat).
- `simple_rsu_damping_vs_noop.png` — three-arm bar chart, two physics.
Chart generators: `_charts_krauss_two_layer.py`, `_charts_density_pen.py`,
`_charts_simple_rsu_damping.py`, `_charts_krauss_density.py`.

---

## 9. Paper framing (agreed direction)

Main claim: **the corridor's metastable health signal h is the control variable
for engaging a CAV moving-bottleneck wall; the wall protects where everything
else measured fails.** Structure:
1. Introduction: literature damping (Stern etc.) only cures shockwaves at low
   density — measured confirmation on our champion (0.36→0.25 @2400 Krauss,
   parity; saturates to no-control at 2700); the gap at high density.
2. The health factor (definition, hysteresis, why a physical state variable).
3. The wall actuator (headway-as-throttle, gentleness, duty cycle, zero hard-brakes).
4. The low-density layer (damping; Krauss evidence, IDM inertness as boundary).
5. Results: the Krauss comparator table (no control/damping/ALINEA/wall×2);
   the IDM stakes (collapse premium +2.3-3.4 s vs ~0; delay parity @2400,
   wins −4.4/−5.4 @2600/2700 p≤0.011); the ladder; the penetration surface.
6. Boundaries: stall, demand≥capacity, metering family, stack — all measured.
Related-work anchors (literature audit, 2026-08-24): moving-bottleneck control
(Čičić/Jin/Johansson 2019; Vishnoi et al. 2023; Garavello/Goatin/Liard/Piccoli
2019/2024; Ramadan/Seibold 2017), capacity-drop control (Cao/Nilsson/Coogan 2021;
Yu et al. 2019), damping (Stern 2018; Martínez 2023; I-24 2023), metering
(Papageorgiou ALINEA 1991), smooth-model under-production of capacity drop.

NOVELTY positioning: not theory-novel; novel as a systems package — composite
health-gated hysteresis + microscopically feasible gentle duty-cycled bottleneck
+ learned engagement over a mechanistic actuator + the complete measured
regime/economics/boundary map. Avoid "first moving-bottleneck metering" (false).
See `docs/IDM_PAPER_SKELETON.md` (v1 skeleton) and `docs/expC_runbook.md`,
`docs/metastable_merge_control.md`, `docs/expC_metastable_health_mask.md`.

---

## 10. Reproducing key numbers (verification path)

- Krauss cells: `runs/probe/metastable_feasibility/stage10_light_m24/26/2700r400_long`
  (sup_v43, w10_t30_h025, rl_pruned3: n=200; no_op from the noop cache).
- IDM ladder: `stage10_light_m{2000..2700}r400idm_long` + noop caches.
- Penetration: `stage10_light_m2700r400idm_pen40/60_long`.
- ALINEA/ramp-gate/VSL: `stage10_light_m2400r400idm_long` (vsl_10, ramp_gate,
  alinea_o*); Krauss ALINEA: `stage10_light_m{2400,2700}r400_long\alinea_o15.*`.
- Collapse metric + row schema: `policies/metastable_rule.py` (HealthRecordingWrapper),
  `_probe_metastable_feasibility.py` (row writer, `--stage 10` variants).
- Tests: `python -m pytest -q` (401/401 green at last run).

---

## 11. Caveats (state of the evidence)

- Headline cells n=50; fine ladder points (2250/2350/2600) and the entire
  penetration surface n=20; Krauss damping cells at 2400/2600 n=20 (directional,
  NOT significant). sup_idm@2700-p0.2 has no rows.
- Krauss 2600/2700 delay with the wall is HIGHER than no control (19.6 vs 14.5 s
  @2700) — protection trades delay under Krauss; the delay WINS exist only under
  IDM at 2600/2700. Any claim must keep the physics straight.
- Penetration changes the baseline population (CAV vTypes differ from HDVs).
- Single map (long corridor), single 30-min demand shape, single merge geometry;
  "collapse" is the h<0.15×60-step operationalization.
- sup_econ and champ_wave were stopped mid-run (see §7) — their verdicts are
  not part of the evidence.
- The ALINEA implementation's occupancy proxy and the IDM o_target sweep clamp
  identically (2000-IDM o10/15/20 all → P=0.95@28.1 s); the Krauss ALINEA cell
  used o=0.15 only.