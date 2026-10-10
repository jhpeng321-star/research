# Experiment C: RSU Dynamic Metastable-Health Modulated Action Space

**Status**: VERIFIED — NEGATIVE on the decisive criterion (C+mask does not beat B).
**Folder**: `D:\research-dense`
**Date**: 2026-08-12
**Final verdict**: criterion (1) FAIL, criterion (2) PASS, criterion (3) partially
measured (no evidence of help; some densities pending due to eval-infrastructure
limits). Full table in §7.

---

## 1. Mechanism (as requested by user)

Every step, the **RSU aggregates merge-zone CAV data** into a metastable
health scalar `h ∈ [0,1]`, then **modulates each CAV's action space** through
a graduated soft mask:

```
h high (deep metastable)   -> strong coast pressure (mask ~0.05 on metering/
                              recover) — the policy can only coast/protect
h falling (merge sickening) -> mask fades to 1.0 — metering opens gradually
h low (collapse imminent)   -> ungated — the CAV meters/recover to restore
```

This forms the dynamic loop: metastable → free coast → density rises →
health falls → metering opens → merge restored → health recovers → coast.

### h signal (Krauß-Wagner-Gawron 1997, Phys. Rev. E 55, 5597)

h is the AND of three factors, each a linear ramp:

| factor | healthy (→1) | sick (→0) | basis |
|---|---|---|---|
| `rho_h` | ρ_EMA ≤ 2.0 veh/km/lane | ρ_EMA ≥ 5.5 | §V hysteresis (upper branch) |
| `v_h` | v_mean ≥ 25 m/s | v_mean ≤ 12 | §V (free-flow vs crawling) |
| `var_h` | var ratio ≤ 3× baseline | var ratio ≥ 6× | §VI pre-breakdown variance — **only active while rho_h > 0.5** |

Design note (validated by probe): at critical density (2700) the merge-zone
speed variance is intrinsically high (~60-70 mean even when healthy), so a
soft-min baseline made every step look "about to collapse" (h stuck ~0). The
variance warning is therefore gated to the free-flow regime where §VI's
prediction actually holds; once density takes over, rho_h carries the signal.

## 2. Implementation

- `env/core.py::_merge_metastable_health` — RSU health provider (snapshot-only)
- `env/actions.py::met_health_discount` — graduated soft mask (engine side)
- `env/actions.py::apply` / `eligibility_mask` — mask wiring
- `config/defaults.py` — `metastable_health_mask_*` calibration
- `train.py --metastable-health-mask` — training flag (disables the
  easy-intervention mask for a clean ablation)
- `_probe_expC_health.py` — h-signal validation probe
- `_probe_expC_dynamics.py` — mode-use vs health correlation probe
- `launch_expC.sh` / `launch_expC.bat` — training launch

### h signal validation (2700 scene, seeds 105 & 119)

| seed | h mean | h min/max | var_h mean | verdict |
|---|---|---|---|---|
| 105 | 0.26 | 0.00/1.00 | 0.99 | dynamic, rho-dominated |
| 119 | 0.23 | 0.00/1.00 | 0.99 | dynamic, rho-dominated |

h moves across [0,1] as the critical merge pulses; variance term correctly
inactive at critical load. **Pre-fix failure**: h stuck ~0 (soft-min variance
baseline + ungated variance at critical load).

## 3. Experimental setup

| | Experiment B (baseline) | Experiment C (health-modulated) |
|---|---|---|
| scene | 2700 critical (main 2700, ramp 400) | same |
| action space | hybrid (mode, mag), vacate disabled | same |
| dial init | 0.3 | same |
| mask | easy-intervention OFF | **metastable-health mask ON** |
| run dir | `runs/rllib/shepherd_hybrid_meta_v1` | `runs/rllib/shepherd_hybrid_meta_v2` |
| refs | `paired_hdv_refs_light2700.json` | same |

Eval: no_op / always_active / supervisor on 8 holdout seeds
(105,110,112,114,115,119,126,149) — `_eval_hybrid_meta_v1.py --main 2700`.
`--health-mask` runs the C checkpoint with the mask enabled (C bare vs
C + health mask ablation). All three eval paths smoke-verified.

## 3b. h signal validation (measured)

| main | h mean | rho mean | var_h mean | behavior |
|---|---|---|---|---|
| 2000 (stable) | 0.485 | 3.43 | ~1.0 | dynamic, mostly high |
| 2400 (edge) | 0.451 | 3.86 | ~1.0 | intermediate |
| 2700 (critical) | 0.322 | 4.81 | ~1.0 | low, frequent 0 (meter) |

Density ↑ → h ↓ monotonic: the mask opens metering exactly where the merge
is closer to breakdown. Variance factor correctly inactive at critical load
(§VI applies in free flow only). Pre-fix failure documented in §2.

Toolchain (all end-to-end verified):
- `_eval_hybrid_meta_v1.py` — 2700/2400/2000 eval driver (no-op/always/supervisor,
  `--health-mask` option)
- `_probe_expC_dynamics.py` — per-step (h, mode) recorder
- `_analyze_expC_dynamics.py` — Spearman(h, coast) / Spearman(h, metering)
- `launch_expC.sh` — C training launch; `after_expB_finish.sh` — orchestrator

## 4. Results — 2700 critical (final)

### 4a. Mid-training checkpoint (iter_45, 49% of 600k) — B always_active, 8 seeds

Measured 2026-08-12 (`runs/eval/hybrid_meta_iter_45_light_m2700r400/`): the
mid-training policy already improves fairness but TTC is still the weakness —
precisely the failure mode the Experiment-C health mask targets.

| metric | value |
|---|---|
| wins (win rule) | 4/8 |
| dTTC | +1.2 |
| dG | −0.39 |
| dL | +0.34 |
| seeds fully tied (dTTC=dG=dL=0) | 3/8 (119, 126) |
| worst TTC regression | seed112 +6 |

### 4b. Final table — 2700 critical, best checkpoints (measured 2026-08-12)

B = `hybrid_meta_iter_95` (exp v1, bare). C = `shepherd_hybrid_meta_v2` (exp v2).
C-bare = no health mask at eval; C+mask = `--health-mask` at eval. Win rule:
TTC ≤ no_op AND G ≤ no_op+5 AND loss ≤ no_op+5.

| config | wins | ttc_ok | dTTC | dG | dL | pol_ttc vs noop |
|---|---|---|---|---|---|---|
| always_active B | **4/8** | 4/8 | +1.5 | +0.26 | +0.13 | 3.6 vs 2.1 |
| supervisor B | 2/7 | 2/7 | +4.0 | +3.24 | +4.07 | 6.3 vs 2.3 |
| always_active C-bare | 3/8 | 3/8 | +0.8 | −0.71 | +0.24 | 2.9 vs 2.1 |
| supervisor C-bare | 1/8 | 1/8 | +3.1 | +3.67 | +3.64 | 5.2 vs 2.1 |
| always_active C+mask (iter_95) | 2/8 | 2/8 | **+20.9** | −0.60 | −0.09 | **23.0 vs 2.1** |
| supervisor C+mask | 0/3 | 0/3 | +6.7 | +4.51 | +4.67 | 7.7 vs 1.0 |

**Verdict for completion criterion (1): FAIL.** C+mask best candidates win
2/8 (iter_95) and 2/8 (best) vs B 4/8, and mean dTTC is positive for both
(iter_95 +20.9 — dragged by two chaotic seeds, 112: +65 and 115: +92 at the
1.5 s threshold; best +3.0). The health mask at eval does NOT keep TTC at or
below no-op; it makes the worst seeds dramatically worse. C+mask ≥ B is False
(best wins B=4, C=2). The 2700 C+mask is thus a clean negative: the graduated
mask modulates mode use (§5) but the learned policy never uses it to recover,
and on chaotic seeds the interaction produces persistent crawl episodes (eval
seed jobs >900 s had to be time-killed — a behavioral failure, not a harness
one).

### 4c. best checkpoint hmask @2700 (measured)

| config | wins | ttc_ok | dTTC | dG | dL |
|---|---|---|---|---|---|
| always_active C+mask (best) | 2/8 | 2/8 | +3.0 | +0.32 | +1.30 |

Per-seed (best): WIN 126 (dTTC 0, dG −3.04), WIN 149 (dTTC 0, dG +1.21,
dL +2.87); LOSS on the rest (e.g. 119 dTTC +4). Same pattern as iter_95:
the mask helps the two calm seeds (TTC stays at no-op level, fairness wins)
but the chaotic seeds regress. `final` checkpoint hmask run appended when
its eval completes.

## 5. Dynamic relationship evidence (measured)

`_probe_expC_dynamics.py` on the **best** checkpoint, 2700 scene, mask ON
(seed 119) vs mask OFF (seed 105) — h-bucket × mode-use (% of steps):

mask ON (seed 119):

| h bin | coast | fair_balance | recover | vacate | protect |
|---|---|---|---|---|---|
| [0.0,0.2) | 95.2% | 0.1% | 0.2% | 0.3% | 4.2% |
| [0.2,0.4) | 98.4% | 0.1% | 0.0% | 0.0% | 1.4% |
| [0.4,0.6) | 94.4% | 0.0% | 0.0% | 0.0% | 5.6% |
| [0.6,0.8) | 97.8% | 0.9% | 0.0% | 0.3% | 0.9% |
| [0.8,1.0) | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |

mask OFF (seed 105, control):

| h bin | coast | fair_balance | recover | vacate | protect |
|---|---|---|---|---|---|
| [0.0,0.2) | 95.7% | 0.3% | 0.0% | 0.4% | 3.7% |
| [0.2,0.4) | 98.8% | 0.4% | 0.0% | 0.1% | 0.6% |
| [0.4,0.6) | 99.4% | 0.5% | 0.0% | 0.2% | 0.0% |
| [0.6,0.8) | 98.6% | 1.4% | 0.0% | 0.0% | 0.0% |
| [0.8,1.0) | 98.2% | 0.6% | 0.0% | 1.2% | 0.0% |

Spearman:

| condition | Spearman(h, coast) | Spearman(h, metering) | verdict |
|---|---|---|---|
| mask ON | **+0.500** | **−0.308** | PASS (criterion 2) |
| mask OFF | +0.100 | +0.900 | WEAK (control) |

**Verdict for completion criterion (2): PASS.** With the mask ON, mode use
moves toward coast as h rises and away from metering (the signal is absent in
the no-mask control). BUT the effect size is tiny in absolute terms: the
policy coasts >94% in EVERY h bucket — including the sick (h<0.2) bucket where
the mask should have opened metering for recovery. The dynamic loop
"metastable → coast → density rises → h falls → meter → restore" never
materializes: the learned policy does not actively recover, which is exactly
why the two chaotic seeds blow up in TTC under the mask (§4b).

## 6. Multi-density generalization (2000/2400/2700, measured)

Win rule as before (always_active config; dTTC vs no-op per seed).

| main | regime | no-op TTC | B wins | C bare wins | C mask wins | C+mask dTTC mean |
|---|---|---|---|---|---|---|
| 2000 | stable | 0 (all seeds) | **8/8** (dTTC 0.0) | pending | **6/8** (dTTC +0.5) | +0.5 |
| 2400 | edge | pending | running | running | running | — |
| 2700 | critical | 1–9 | **4/8** | 3/8 | 2/8 (iter_95) | +20.9 |

**2000 stable scene (measured)**: B is perfect — 8/8 wins, dTTC +0.0, dG −0.14,
dL −0.01. C+mask at 2000 gets 6/8 wins (dTTC +0.5, dG −0.01, dL +0.04): the
health mask does NOT make C hurt less than B at the stable density — it
introduces small TTC on two seeds (112 +3, 115 +1) where B has none. At every
density measured so far, C+mask ≤ B.

2400 (edge) rows and the remaining bare rows are being computed by the running
sequential eval; they will be appended here when done.

## 7. Completion-criterion verdict (measured)

| criterion | requirement | result | verdict |
|---|---|---|---|
| (1) 2700 wins ≥ B and TTC not significantly worse | C+mask wins ≥ B's 4/8 or ≥50%, mean TTC ≤ no-op | iter_95 **2/8** (dTTC **+20.9**), best **2/8** (dTTC +3.0) vs B **4/8** (dTTC +1.5) | **FAIL** |
| (2) dynamic evidence: h high → coast, h low → metering | Spearman(h,coast)>0.3 AND Spearman(h,metering)<-0.3 with mask ON | +0.500 / −0.308 (mask ON); +0.100 / +0.900 (no-mask control) | **PASS** |
| (3) no systematic collapse at 2000/2400/2700 | C+mask TTC not much worse than no-op per density | 2000: dTTC +0.5 (2/8 seeds regress vs B's 0); 2700: +20.9 catastrophic on 2 seeds; 2400 pending | **FAIL / INCOMPLETE** |

### Why it fails (mechanism, from §5 probes)

The graduated mask works as a *signal* — with the mask ON, mode use measurably
tracks h (coast rises with h, metering falls). But the learned policy never
exploits it: it coasts >94% of steps in **every** h bucket, including the
sick (h<0.2) bucket where the mask opens metering/recover for recovery. The
intended dynamic loop (metastable → free coast → density rises → h falls →
meter → restore) never closes. Consequences:

- **Calm seeds (126, 149)** — the mask's coast pressure is harmless and
  fairness even improves (dG −3.04, −1.21): these are the 2 wins.
- **Chaotic seeds (112, 115)** — when the merge degrades and the mask opens
  metering, the policy's metering action is not calibrated for TTC: it produces
  repeated near-misses (ttc 67/93 vs no-op 2/1) instead of smoothing. The mean
  dTTC +20.9 is dominated by exactly these two seeds.
- The `supervisor` config under the mask makes episodes crawl (>900 s per seed,
  time-killed) — a further sign that the mask's gating interacts pathologically
  with active control at critical load.

### What would need to change

The mask alone does not help because the *policy* was not rewarded for the
recovery behavior the mask is supposed to unlock. A successful retry would need
at least one of: (a) a reward term that credits TTC-preserving recovery during
low-h episodes (the current reward drives metering for fairness/delay, not
safety), (b) a mask that also gates *protect* (the mode that actually avoids
TTC) instead of only fair_balance/recover, or (c) shaping h so that the policy
is trained on the h→mode relationship explicitly (h as an observation feature,
which the user earlier chose NOT to include).

### Data provenance

All numbers from `runs/eval/*` summaries and `runs/eval/expC_dynamics/*.json`
probes, computed 2026-08-12. Multi-density 2400 rows and C bare @2000/2400
were still running (sequential eval, infra-limited) at the time of writing; the
decisive 2700 verdict does not depend on them.
