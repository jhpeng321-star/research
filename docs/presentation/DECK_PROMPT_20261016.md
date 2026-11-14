# Prompt: build the C4 / WM-S showcase deck (2026-10-16)

Make a short, simple slide deck (PDF, about 10–12 pages) for a research showcase. Audience: professor and
labmates. Big picture only. Use big fonts, one figure or table or at most 4 bullets per page, no dense
paragraphs, and a one-line sign convention under every table. Three parts, clearly labelled:

1. The research and how it works
2. Current progress of the models
3. The contributions we want to make

Use only the facts and numbers below. Do not invent numbers. Where a value says **PENDING**, leave a clearly
marked placeholder ("to be filled tonight").

---

## Part 1: The research and how it works

### The problem
- A highway on-ramp merge, simulated in SUMO.
- A small share of connected automated vehicles (CAVs, 5–60 %) drive among human-driven vehicles (HDVs).
- Question: can the CAVs reduce everyone's delay by how they accelerate and change lanes?
- Traffic is heavy: demand 1900–2700 veh/h, while the road saturates near ~1,930 veh/h on the mainline.

### The idea
- Learn a **world model** of mixed traffic from SUMO data.
- Train the CAV policy **inside the world model** (Dreamer-style: gradients flow through imagined rollouts).
- **Check in real SUMO** and feed those episodes back to refresh the world model (the "Dreamer loop").
- Constraints:
  - no hand-designed controllers inside learning; the policy outputs acceleration and lane decisions directly;
  - human driver behaviour is learned from prediction error only (no driver-type labels).

### How the world model (WM-S) works (one simple diagram)
Every 0.5 s, for every vehicle:
- **Driver settings:** each human driver's settings (desired speed, headway, gaps, comfortable accel/brake) are
  inferred from that driver's observed history by a learned encoder, trained only from prediction error.
- **Car-following:** IDM physics with those settings.
- **Lane changes:** learned decisions (overtake / return / ramp merge heads). Physics only blocks unsafe changes,
  using a safe-gap check of the same form as SUMO's lane-change model. **No MOBIL** in the model we report.
- **Merge-zone rules:** ramp cars wait at the lane end until the gap is safe; mainline cars yield.
- **Correction:** a small gated learned correction on top.
- **CAVs:** their commands are an input, so the model can evaluate a controller it never saw in training.
- Rolled forward step by step for 10–120 s.

### How the policy learns: the Dreamer loop (one cycle diagram)
1. The policy trains in imagination inside the world model. One policy is shared by all CAVs; each CAV sees its
   neighbours plus a shared roadside signal. The reward is the delay of all vehicles.
2. Every new policy checkpoint drives in real SUMO: 50-episode batches on CPU servers.
3. The world model is refreshed on all policy episodes so far. A refresh is accepted only if its validation error
   stays within 1.3× the previous model's.
4. The policy continues on the accepted world model. Repeat.

### Cooperation is in the objective
The trainer runs extra "what-if" versions of each imagined scene:
- each CAV alone (D_i);
- a group of neighbouring CAVs (D_G);
- all CAVs (D_all).

Cooperation gain C_G = D_G − Σ D_i is rewarded. Weights: all 1.0, group 1.0, individual 0.5, cooperation 0.5,
team 0.5, with a guard against gaming.

---

## Part 2: Current progress

### (a) World model accuracy on unseen walls
Setup: an unseen "moving wall" controller, never in training; 36 new worlds; 139 states.
Metric: median HDV position error (m); lower is better.

| Model | 10 s | 30 s | 60 s | 120 s | Blow-ups |
|---|---|---|---|---|---|
| **v5m (v5 + merge rules)** | **3.36** | **12.34** | 31.14 | **89.13** | **5/139** |
| v5 | 3.64 | 12.93 | 27.85 | 92.20 | 25/139 |
| v6m | 4.78 | 14.15 | 38.58 | 103.72 | 19/139 |

- v5m was chosen as the loop's starting world model. It is significantly better than v6m at 10 s and 60 s, and
  has 5× fewer blow-ups than v5.
- Predicting the wall's effect (wall vs no wall from the same state): v5m gets the sign right in 99 % of states
  at 30 s and 93 % at 60 s.

### (b) The Dreamer loop works
Run: started 14:41, stopped 21:00. Four world-model refreshes were accepted. About 1,300 policy episodes were
run in SUMO. **The world model we report is refresh 4** (trained on 750 policy episodes); the final policy is the
one trained on it.

**Learning curve (main figure).** Policy minus native, system time per vehicle in SUMO (s/veh), per 50-episode
batch, 900 s episodes; lower is better, 0 = native. Each point is ±2–3. Mark batch 9, where the first
loop-refreshed model took over.

| Batch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gap (s/veh) | +95.6 | +74.3 | +66.0 | +45.5 | +31.9 | +24.8 | +26.2 | +24.1 | +22.4 | +20.6 | +18.4 | +15.6 |

| Batch | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gap (s/veh) | +10.7 | +14.2 | +12.1 | +9.8 | +10.4 | +9.9 | +9.6 | +10.8 | +9.7 | +8.5 | +4.3 | +7.4 |

- Before the loop took effect, the gap had plateaued at about +24.
- With the loop it fell to about +5 to +8.
- The largest gain is at 40 % CAVs, where the gap went from +41 to about +13.

**World model refreshes.** Validation error on the original data (mean position error, m), before → after:

| Refresh | 30 s | 60 s | 120 s |
|---|---|---|---|
| 1 | 38.2 → 37.3 | 96.8 → 95.2 | 161.2 → 154.3 |
| 2 | 37.3 → 37.2 | 95.2 → 94.7 | 154.3 → 155.1 |
| 3 | 37.2 → 35.2 | 94.7 → 91.6 | 155.1 → 150.7 |
| 4 (reported) | 35.2 → 36.3 | 91.6 → 92.7 | 150.7 → 154.3 |

Learning the policy's own traffic did not cost accuracy on the original data.

### (c) Final SUMO evaluation
Setup: 42 seeds (964000–964041), demands 1900–2700, CAV penetration 5–60 %, 600 s episodes.
Metric: change in system time per vehicle vs native (s), paired on identical traffic; mean [95 % CI]; negative
is better.

| Controller | All | Light (1900) | Heavy (2400–2700) |
|---|---|---|---|
| MAPPO (trained directly in SUMO) | −1.51 [−1.87, −1.16] | −1.58 | −1.55 |
| PI controller | −1.16 [−1.39, −0.93] | −0.77 | −1.40 |
| Always max acceleration | −0.40 [−0.64, −0.16] | +0.07 | −0.72 |
| v5 Dreamer policy (no loop, last night) | +1.20 [+0.91, +1.49] | +1.29 | +1.05 |
| CIRCLES damping | +2.09 [+1.81, +2.37] | +2.09 | +1.90 |
| Dreamer, before loop (v5m) | **PENDING** | **PENDING** | **PENDING** |
| Dreamer, after loop (refresh 4) | **PENDING** | **PENDING** | **PENDING** |

Note: in this driver population PI helps and CIRCLES damping does not.

### (d) Does cooperation emerge?
Method: measured in SUMO by branching each decision point into what-if runs.
Metric: team delay saved vs native (veh-s); positive is better. C_G = group benefit minus the sum of individual
benefits.

| Policy | Horizon | D_all (all CAVs) | C_G (cooperation gain) |
|---|---|---|---|
| MAPPO | 60 s | −11.2 [−24.3, +1.8] | +6.1 [−2.7, +15.4] |
| MAPPO | 300 s | +428.6 [+24.0, +847.5] | −200.9 [−863.8, +500.5] |
| Dreamer after loop | 60 s | **PENDING** | **PENDING** |
| Dreamer after loop | 300 s | **PENDING** | **PENDING** |

- MAPPO helps significantly at 300 s, but its gain is individual: no significant cooperation.
- Cooperation is rewarded in the objective but has not yet been shown significantly by any method.

---

## Part 3: Contributions we want to make (one line + status each)

| Contribution | Status |
|---|---|
| **C1** Zero-shot prediction of unseen controllers (the world model predicts controllers it never saw) | Shown on PI and moving walls (green) |
| **C2** Per-driver identification (driver settings learned from prediction error, no labels) | Partial (blue) |
| **C3** Fragility analysis | Deferred (grey) |
| **C4** Learned cooperative multi-CAV control inside the world model, trained with the Dreamer loop | In progress (amber): loop working, policy improving fast in SUMO, still behind native overall; cooperation not yet significant |

### Next steps (4 bullets)
- Keep running the loop: the policy was still improving at the stop.
- Larger cooperation signal: more what-if scenes per update (memory-limited today).
- Close the gap at high penetration (40–60 % CAVs).
- Longer-horizon world-model accuracy (120 s) and the merge-zone wait times.
