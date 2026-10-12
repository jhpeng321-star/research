# WM-S C4 Gate 1 — action fidelity (CPU, existing data)

## Question
On paired controller arms that start from the **same saved SUMO state**, does WM-S rank controllers (native / damping / PI / wall) the way SUMO does at **10 / 30 / 60 s**? Verdict wording target: *WM-S ranks controllers like SUMO at 10/30 s* — **supported / not supported / mixed**.

## Data
- **39 paired groups** (world @ branch time t0), TRAIN split only, from `pack_v3` + the re-aligned `pack_damping` snap (damping arm); penetrations (%% CAV): 10% x12, 20% x13, 40% x14; demands 2400-2700 veh/h.
- **Arms** per group where paired: `native`, `base` (CIRCLES damping), `pi`, `w-lo-early/late`, `w-mid-early/late` (wall v2 laws).
- **Pairing rule**: an arm is kept only if its t0 frame equals the native t0 frame within tolerance (same vehicle ids and lanes, max |ds| <= 1 m, max |dv| <= 1 m/s); arms branched from a different physical state (e.g. an already-engaged early wall) are dropped for that group (`groups_gate1.json`). Matched arm-group counts: native 39, w-*-late 39+39, base 17, pi 16, w-*-early 12+12.
- **Informative-pair rule (data finding)**: inside the 60 s window most wall branches are still trajectory-identical to native (the wall assembles later than the window), so their paired differences are vacuous. A pair (or a group-ranking arm) enters horizon h only if the two arms' recorded SUMO outcomes differ at h. Informative pair counts per horizon and pair type: {"10|base-native": 14, "10|base-pi": 11, "10|base-w-lo-early": 6, "10|base-w-lo-late": 14, "10|base-w-mid-early": 6, "10|base-w-mid-late": 14, "10|native-pi": 9, "10|pi-w-lo-early": 4, "10|pi-w-lo-late": 9, "10|pi-w-mid-early": 4, "10|pi-w-mid-late": 9, "30|base-native": 17, "30|base-pi": 14, "30|base-w-lo-early": 9, "30|base-w-lo-late": 17, "30|base-w-mid-early": 9, "30|base-w-mid-late": 17, "30|native-pi": 12, "30|native-w-lo-early": 1, "30|pi-w-lo-early": 6, "30|pi-w-lo-late": 12, "30|pi-w-mid-early": 6, "30|pi-w-mid-late": 12, "30|w-lo-early-w-lo-late": 1, "30|w-lo-early-w-mid-early": 1, "30|w-lo-early-w-mid-late": 1, "60|base-native": 17, "60|base-pi": 14, "60|base-w-lo-early": 9, "60|base-w-lo-late": 17, "60|base-w-mid-early": 9, "60|base-w-mid-late": 17, "60|native-pi": 15, "60|native-w-lo-early": 3, "60|pi-w-lo-early": 8, "60|pi-w-lo-late": 15, "60|pi-w-mid-early": 8, "60|pi-w-mid-late": 15, "60|w-lo-early-w-lo-late": 3, "60|w-lo-early-w-mid-early": 3, "60|w-lo-early-w-mid-late": 3}.
- **SUMO outcome** per arm: computed from the recorded arm trajectory (60 s); **WM outcome** per arm: one WM-S rollout from the identical t0 state with CAVs teacher-forced to that arm's recorded CAV trajectory (HDVs predicted, entries off, 0.5 s steps).
- **Models**: WM-S R5b best (`WM-S-R5B-20261012/box/gpuA/r5b_main_a1/wm_s_best.pt`, trained WITHOUT PI and wall arms), WM-S C1 (`data_cache/ckpt_c1fast/wm_s.pt`; PI/wall unseen too); references: **physics-only** (R5b network with the learned residual and lane correction gated to zero, population-default driver parameters = IDM/MOBIL default settings) and a **constant predictor** (always native).
- **Seen vs unseen**: both arms in {native, base} = seen in WM training; any arm in {pi, wall x4} = unseen (excluded from both models' training corpora, `lpwo_v3`).
- **Outcomes at 10/30/60 s** (HDVs of the t0 roster): cumulative delay over the merge zone 0-75.44 m (vehicle-seconds, free speed 27.78 m/s), mean HDV speed in the zone, HDV hard-brake events (runs of a < -3 m/s^2), throughput past the merge (t0 vehicles crossing s = 75.44). Metrics are oriented lower-is-better (speed/throughput negated) before differencing.
- **Statistics**: pairwise sign agreement, Pearson and Spearman of paired differences (A-B, WM vs SUMO); Kendall tau-b of the arm ranking per group; regret of the WM-chosen arm (SUMO outcome of the WM pick minus SUMO best, oriented units). World-cluster bootstrap 95% CIs (n=1000, seed 0).

## Verdict
**mixed** for *WM-S ranks controllers like SUMO at 10/30 s* (primary metric HDV zone delay, R5b, pooled pairs).

| horizon | sign agreement [95% CI] | Kendall tau [95% CI] | passes |
|---|---|---|---|
| 10 s | 0.467-0.936 (pt 0.690) | 0.214-0.800 (pt 0.523) | inconclusive |
| 30 s | 0.417-0.772 (pt 0.578) | -0.053-0.561 (pt 0.246) | inconclusive |

## Main table — HDV zone delay (paired differences, pooled pairs)
### 10 s
| model | n pairs (worlds) | sign agreement [CI] | Pearson [CI] | Spearman [CI] | Kendall tau [CI] | regret [CI] | always-native regret |
|---|---|---|---|---|---|---|---|
| WM-S R5b best | 100 (15) | 0.690 [0.467, 0.936] | 0.97 [0.33, 1.00] | 0.44 [-0.03, 0.87] | 0.523 [0.214, 0.800] | 0.1 [0.0, 0.3] | 8.8 |
| WM-S C1 | 100 (15) | 0.570 [0.356, 0.810] | 0.92 [-0.08, 0.98] | 0.28 [-0.29, 0.77] | 0.370 [-0.008, 0.751] | 0.4 [0.0, 0.9] | 8.8 |
| physics-only | 100 (15) | 0.710 [0.538, 0.897] | 0.78 [0.07, 0.99] | 0.32 [-0.19, 0.81] | 0.499 [0.178, 0.798] | 0.3 [0.0, 0.8] | 8.8 |
| constant (native) | - | - | - | - | - | 8.8 [0.0, 24.2] | 8.8 |

### 30 s
| model | n pairs (worlds) | sign agreement [CI] | Pearson [CI] | Spearman [CI] | Kendall tau [CI] | regret [CI] | always-native regret |
|---|---|---|---|---|---|---|---|
| WM-S R5b best | 135 (19) | 0.578 [0.417, 0.772] | 0.31 [-0.02, 0.70] | 0.22 [-0.20, 0.63] | 0.246 [-0.053, 0.561] | 70.5 [5.0, 156.3] | 97.3 |
| WM-S C1 | 135 (19) | 0.615 [0.462, 0.781] | 0.30 [-0.02, 0.73] | 0.31 [-0.00, 0.59] | 0.246 [-0.088, 0.579] | 88.4 [7.0, 184.3] | 97.3 |
| physics-only | 135 (19) | 0.748 [0.533, 0.942] | 0.06 [-0.41, 0.71] | 0.43 [-0.12, 0.91] | 0.561 [0.211, 0.860] | 81.0 [0.0, 199.5] | 97.3 |
| constant (native) | - | - | - | - | - | 97.3 [6.3, 202.5] | 97.3 |

### 60 s
| model | n pairs (worlds) | sign agreement [CI] | Pearson [CI] | Spearman [CI] | Kendall tau [CI] | regret [CI] | always-native regret |
|---|---|---|---|---|---|---|---|
| WM-S R5b best | 156 (21) | 0.635 [0.510, 0.777] | 0.35 [-0.04, 0.68] | 0.26 [-0.09, 0.59] | 0.404 [0.095, 0.697] | 245.2 [60.7, 465.3] | 453.3 |
| WM-S C1 | 156 (21) | 0.506 [0.371, 0.654] | 0.14 [-0.30, 0.52] | 0.05 [-0.33, 0.41] | 0.167 [-0.118, 0.460] | 307.8 [84.5, 571.5] | 453.3 |
| physics-only | 156 (21) | 0.590 [0.384, 0.797] | 0.33 [-0.14, 0.70] | 0.32 [-0.16, 0.76] | 0.254 [-0.127, 0.587] | 162.6 [24.1, 364.5] | 453.3 |
| constant (native) | - | - | - | - | - | 453.3 [179.6, 765.3] | 453.3 |

## Seen vs unseen arms (HDV zone delay, R5b)
| horizon | scope | n pairs | sign agreement [CI] | Spearman [CI] |
|---|---|---|---|---|
| 10 s | seen | 14 | 0.786 [0.571, 1.000] | 0.64 [0.05, 0.91] |
| 10 s | unseen | 86 | 0.674 [0.448, 0.937] | 0.40 [-0.05, 0.87] |
| 30 s | seen | 17 | 0.647 [0.412, 0.882] | 0.26 [-0.39, 0.76] |
| 30 s | unseen | 118 | 0.568 [0.405, 0.768] | 0.21 [-0.22, 0.62] |
| 60 s | seen | 17 | 0.588 [0.353, 0.824] | -0.01 [-0.61, 0.58] |
| 60 s | unseen | 139 | 0.640 [0.516, 0.775] | 0.29 [-0.07, 0.61] |

## All metrics — sign agreement (pooled pairs) and regret, R5b vs references
| horizon | metric | R5b sign [CI] | C1 sign [CI] | physics sign [CI] | R5b regret | C1 regret | physics regret | const regret |
|---|---|---|---|---|---|---|---|---|
| 10 s | `hdv_zone_delay_s` | 0.690 [0.467, 0.936] | 0.570 [0.356, 0.810] | 0.710 [0.538, 0.897] | 0.1 | 0.4 | 0.3 | 8.8 |
| 10 s | `hdv_zone_mean_speed_mps` | 0.760 [0.568, 0.951] | 0.740 [0.557, 0.925] | 0.740 [0.541, 0.924] | 0.0 | 0.0 | 0.0 | 0.0 |
| 10 s | `hdv_hard_brake_events` | 0.600 [0.368, 0.838] | 0.390 [0.232, 0.545] | 0.770 [0.518, 1.000] | 0.0 | 0.1 | 0.0 | 0.0 |
| 10 s | `throughput_past_merge` | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.0 | 0.0 | 0.0 | 0.0 |
| 30 s | `hdv_zone_delay_s` | 0.578 [0.417, 0.772] | 0.615 [0.462, 0.781] | 0.748 [0.533, 0.942] | 70.5 | 88.4 | 81.0 | 97.3 |
| 30 s | `hdv_zone_mean_speed_mps` | 0.689 [0.519, 0.851] | 0.467 [0.283, 0.669] | 0.585 [0.364, 0.800] | 0.2 | 0.2 | 0.2 | 0.2 |
| 30 s | `hdv_hard_brake_events` | 0.430 [0.246, 0.596] | 0.319 [0.153, 0.500] | 0.304 [0.180, 0.450] | 0.9 | 0.8 | 0.7 | 0.1 |
| 30 s | `throughput_past_merge` | 0.689 [0.479, 0.846] | 0.778 [0.625, 0.898] | 0.704 [0.524, 0.867] | 0.4 | 0.4 | 0.3 | 0.4 |
| 60 s | `hdv_zone_delay_s` | 0.635 [0.510, 0.777] | 0.506 [0.371, 0.654] | 0.590 [0.384, 0.797] | 245.2 | 307.8 | 162.6 | 453.3 |
| 60 s | `hdv_zone_mean_speed_mps` | 0.494 [0.376, 0.632] | 0.359 [0.212, 0.522] | 0.654 [0.469, 0.826] | 0.6 | 0.6 | 0.3 | 0.6 |
| 60 s | `hdv_hard_brake_events` | 0.327 [0.192, 0.459] | 0.455 [0.274, 0.624] | 0.333 [0.179, 0.492] | 1.3 | 0.7 | 1.0 | 0.4 |
| 60 s | `throughput_past_merge` | 0.532 [0.393, 0.697] | 0.436 [0.256, 0.634] | 0.564 [0.416, 0.715] | 0.7 | 0.6 | 0.5 | 1.0 |

Regret is in oriented units (delay: vehicle-seconds; brakes: events; speed/throughput: negated m/s / vehicles). Constant predictor regret = regret of always picking native.

## SUMO vs WM per-arm means (HDV zone delay 30 s, vehicle-seconds, matched groups)
| arm | n | SUMO | R5b | C1 | physics |
|---|---|---|---|---|---|
| base | 17 | 3898 | 3335 | 3421 | 3016 |
| native | 39 | 3861 | 3561 | 3559 | 3385 |
| pi | 16 | 3625 | 3120 | 3315 | 2874 |
| w-lo-early | 12 | 4569 | 4239 | 4374 | 3955 |
| w-lo-late | 39 | 3861 | 3561 | 3559 | 3385 |
| w-mid-early | 12 | 4560 | 4167 | 4262 | 3862 |
| w-mid-late | 39 | 3861 | 3561 | 3559 | 3385 |

(Vehicle-seconds of HDV zone delay; lower = less delay = better for every column. The WM columns are the model's own predicted delay.)

## Caveats
- **Wall evidence inside the window is thin**: the saved wall branches assemble after the 60 s window in most groups (w-lo/mid-late are trajectory-identical to native throughout; w-*-early diverge only near 60 s in a few groups), so Gate 1 mainly tests native vs damping vs PI. Wall-action fidelity needs branches saved at assembly-minus-10..30 s or a longer window (next action).
- Small n after the informative-pair filter (100/135/156 pairs on 15/19/21 worlds at 10/30/60 s); CIs are wide.
- Exact ties (WM delta 0 and SUMO delta 0, e.g. equal integer throughput or equal hard-brake counts) count as agreements once the pair is informative.
- The physics-only reference is as good as or better than both learned models on sign agreement and Kendall tau at 10-30 s: on these pairs the learned residual does not yet add ranking fidelity over its own physics base.
- WM rollouts see no post-t0 entrants (entries off) and under-predict the delay level (~8% lower than SUMO); only paired differences carry the Gate-1 signal.

## Reproduce
```
cd E:/agent-work/c4-gate1
set PYTHONPATH=src
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe runs/active/WM-S-C4-GATE1-20261012/build_states.py
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe runs/active/WM-S-C4-GATE1-20261012/run_rollouts.py --workers 4
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe runs/active/WM-S-C4-GATE1-20261012/analyze.py
```

Inputs (read-only): `E:/research-dense/data_cache/pack_v3`, `pack_damping`, `sidecars_packv3_v2/identities`, `identities_damping`, `ckpt_c1fast/wm_s.pt`, `runs/active/WM-S-R5B-20261012/box/gpuA/r5b_main_a1/wm_s_best.pt`. TRAIN split only; no sealed/CAL/holdout/chunk5/J4B/INC1 path is read; no SUMO, no GPU, no training. `states_gate1.json` (148 MB) is regenerable and not committed.
