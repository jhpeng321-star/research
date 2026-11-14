# IDM P30 Damping Authority Map — Development Results

**Date:** 2026-08-30  
**Status:** development-only continuous diagnosis complete; no new SUMO run,
no prospective seed, no wall treatment, and no controller change. This is not
prospective acceptance of the frozen damping arm.

**Artifact:**
`runs/eval/idm_damping_authority_map_development_v1/summary.json`  
**Analyzer:** `tools/analyze_idm_damping_authority_map.py`  
**Focused tests:** `tests/test_idm_damping_authority_map.py`

## 1. Scientific question and contract

This analysis asks how much physical effect the unchanged P30 decentralized
action delivers across already opened traffic states, and how much absolute
safety/service burden remains afterward. It does **not** infer a binary
critical density.

Two concepts remain deliberately separate:

1. `response_class` is the existing paired label: helpful when the
   reference-scaled severe-braking/acceleration-energy effect is greater than
   +0.02 and every frozen constraint passes; neutral inside the ±0.02
   deadband with constraints passing; harmful below -0.02 or on any frozen
   constraint failure.
2. Absolute residual metrics are continuous: post-treatment severe braking,
   acceleration energy, TTC exposure, collapse duration, delay, arrivals and
   merge discharge. The demand-specific references are used only as scales,
   never as approved safety thresholds.

Accordingly, this artifact defines neither a `safe_but_insufficient` label nor
a configured-demand/penetration switch. A beneficial action can still leave
substantial residual burden, while a neutral paired effect can occur in a
state that resolves without higher-authority control.

## 2. Reused evidence and integrity

The balanced core is the existing recurrence-3 grid:

- seeds 4121–4132;
- mainline demands 1900, 2000 and 2100 veh/h;
- CAV penetrations 5%, 10%, 15%, 20%, 25% and 30%;
- exact abstain, abstain-replay and `activation_speed_ceiling_v1` branches;
- 178 paired opportunities from 216 declared seed-cells.

The map also reports the already opened 10%/2200 recurrence-3 diagnostic as a
clearly unbalanced supplement: four paired opportunities from 12 declared
seed-cells. The combined artifact therefore contains 182 paired opportunities
from 228 declared seed-cells and records all 46 no-decision cells.

All 19 source contracts match the frozen action, recurrence, candidate,
20-step dose, 60-step horizon and reward-reference identity. Every branch
matches its discovery state, abstain branches contain zero control commands,
and the maximum abstain-versus-replay difference across the audited branch
metrics is exactly 0.0. Only development seeds 4121–4132 occur; no seed 4201
or above was read.

## 3. Balanced-grid response and continuous residuals

The table uses the balanced 1900–2100 grid only. `Burden` is the mean absolute
post-treatment equal-weight severe/energy quantity in reference units. It is
a continuous comparison scale, not a pass/fail threshold.

| CAV penetration | Paired cases | Helpful / neutral / harmful | Adequate dose | Constraints pass | Mean paired effect | Mean absolute burden |
|---:|---:|---:|---:|---:|---:|---:|
| 5%  | 19 | 7 / 7 / 5  | 100.0% | 100.0% | +0.0276 | 1.765 |
| 10% | 23 | 13 / 4 / 6 | 95.7%  | 91.3%  | +0.0875 | 1.711 |
| 15% | 30 | 12 / 8 / 10 | 96.7% | 83.3% | +0.0608 | 1.784 |
| 20% | 34 | 11 / 12 / 11 | 97.1% | 94.1% | +0.0384 | 1.718 |
| 25% | 36 | 11 / 4 / 21 | 88.9% | 75.0% | -0.0298 | 1.996 |
| 30% | 36 | 13 / 9 / 14 | 91.7% | 88.9% | +0.0242 | 1.700 |

The response is plainly not monotone in configured penetration. The 25% cell
is the weakest aggregate stratum, while 30% partially recovers. This directly
rejects treating penetration—or configured demand—as a universal critical
threshold.

Within the frozen rule's development evidence domain (5–10%, 1900–2100), the
42 paired cases contain 20 helpful, 11 neutral and 11 harmful responses;
41/42 deliver adequate dose and 40/42 pass all frozen constraints. Mean paired
effect is +0.0604. Mean absolute safety burden changes from 1.7959 under the
identical abstain branch to 1.7355 under P30. Mean delay-rate change is
-0.00013 s/vehicle and mean merge-discharge change is -0.024 vehicles over the
common horizon. These are development averages with heterogeneous per-seed
outcomes, not prospective controller evidence.

The supplemental 10%/2200 slice is deliberately not pooled into balanced-grid
claims. Its four cases split 2 helpful / 2 harmful despite 100% dose and
constraint compliance. Mean paired effect is +0.0234, while mean absolute
burden is 2.211 and severe braking averages 0.475 events per step. The small,
unbalanced sample is consistent with increasing residual traffic burden but
is not enough to establish a boundary by itself.

## 4. Interpretation for escalation

The evidence supports a control-authority narrative rather than a critical
density narrative:

- P30 commands are usually delivered and often improve the paired safety
  composite, so the mechanism has real low-authority effect.
- Harmful and neutral responses remain in every penetration region, and the
  aggregate pattern is nonmonotone.
- Absolute residual burden can remain high even when the paired effect is
  positive.
- Therefore an eventual escalation decision must use realized traffic and
  wall-feasibility state, then predict the paired benefit of adding the wall.
  It must not learn a hard configured-density threshold from this map.

Rows outside 5–10%/1900–2100 are marked
`out_of_domain_diagnostic`. The frozen rule abstains outside
its admitted domain; these P30 branches diagnose why extending the low-level
rule is unsupported and are not actions taken by the frozen deployed rule.

## 5. Contamination and next gate

Seeds 4101–4106 remain reference development seeds, and 4107–4132 remain
opened development seeds. None may support an independent prospective claim.
Seeds 4201–4220 remain reserved for the separately frozen rule-arm acceptance
test; 4221–4240 remain confirmation seeds; 5001–5060 remain final paper seeds.

The authority map does not authorize the moving wall or predictor. The next
scientific order remains:

1. prospectively accept or reject the unchanged rule arm on 4201–4220;
2. freeze the exact control-arm artifact if it passes;
3. freeze one wall geometry and a separate dose/safety protocol;
4. run only the development wall-dose/headroom pilot before generating any
   predictor-scale data.

## 6. Reproduction command

```powershell
$env:PYTHONPATH = "D:\research-dense"
python tools/analyze_idm_damping_authority_map.py `
  --grid-summary runs/eval/idm_paired_option_penetration_grid_l3_v1/grid_summary.json `
  --inputs runs/eval/idm_decentralized_p30_ranked_neighbor_validation_m2200_l3_v1 `
  --refs runs/eval/idm_decentralized_lower_reward_refs_v1.json `
  --output runs/eval/idm_damping_authority_map_development_v1
```

This command reads stored JSON artifacts only. It does not start SUMO.
