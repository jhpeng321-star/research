# WM_QUAL_REPORT - fine-tuned wz3to2 WM (WM-E ft_r1 phase B final wm_s.pt, sha 6da9d333)

Verdict: **FAIL** (integrity failures: none). Gate spec sha b013c0da9e9d; metrics sha 8a61fef86d39.
Model: backend `wm`, checkpoint /root/autodl-fs/wzqual/ckpt_ft/wm_s.pt (sha 6da9d3337aad), physics checkpoint (embedded / physics.json / TRAIN_RECEIPT.json) (physics sha None); samples K=4.

## Primary criteria

| criterion | value | pass rule | status | gap to threshold |
|---|---|---|---|---|
| C0_role_coverage | 0.911 (roles hold, lane_to, merge_early, pulse, zip_merge, zip_yield) | >= 0.80; hold + lane_to | PASS |  |
| D1_breakdown_rate | WM 0.669 vs real 0.823; diff -0.153 [-0.227, -0.058] | abs(diff) <= 0.10 pooled, <= 0.20 per cell (n>=12) | FAIL | +0.053 |
| D2_onset | W1 31.7 s (real events 102) | <= 45 s | PASS | -13.308 |
| D3_queue | rel err -0.152; W1/SD 0.201 | abs(rel) <= 0.25 and W1/SD <= 0.50 | PASS | -0.098 |
| D4_throughput | rel err -0.008 | abs(rel) <= 0.08 | PASS | -0.072 |
| D5_zone_crps | 120 s WM 1.674 vs keep-speed 5.561; 300 s WM 2.829 vs keep-speed 8.916 | WM <= keep-speed at 120 and 300 s | PASS |  |
| A1_within_moment_spearman_T | 0.140 [0.056, 0.183] (moments 105) | >= 0.30, CI low > 0 | FAIL | +0.160 |
| A2_pooled_spearman_T | 0.111 [-0.021, 0.214] (merge map 0.103) | >= 0.30, CI low > 0.10 | FAIL | +0.189 |
| A3_precision_positive_T | 0.654 of 433 WM-positive (base rate 0.597; merge map 0.407) | >= 0.697 | FAIL | +0.044 |
| A4_precision_at_k_S | 0.600 at k=10 of 240 (base 0.579) | >= 0.700 | FAIL | +0.100 |
| A5_regret | WM/random regret 0.838; WM gain 635.4 [108.380, 1256.014] veh-s; acted 0.817 | ratio <= 0.50 and gain CI low > 0 | FAIL | +0.338 |
| A6_beats_keep_speed | WM 0.111 vs keep-speed -0.037 | WM > keep-speed | PASS | -0.148 |

## Distributional detail

Starts 222 (not broken 124, broken at start 98); rule boundary 7.67 m/s; backend predicts entrants: True.

| cell | n | real P(bd<=300 s) | WM | diff |
|---|---|---|---|---|
| D2400_p20 | 6 | 0.833 | 0.333 | -0.500 |
| D2400_p40 | 8 | 0.625 | 0.406 | -0.219 |
| D2400_p60 | 9 | 0.667 | 0.500 | -0.167 |
| D2800_p20 | 10 | 0.900 | 0.600 | -0.300 |
| D2800_p40 | 12 | 0.750 | 0.542 | -0.208 |
| D2800_p60 | 10 | 0.600 | 0.425 | -0.175 |
| D3200_p10 | 4 | 1.000 | 1.000 | 0.000 |
| D3200_p20 | 11 | 0.909 | 0.909 | 0.000 |
| D3200_p30 | 5 | 1.000 | 0.800 | -0.200 |
| D3200_p40 | 14 | 1.000 | 0.768 | -0.232 |
| D3200_p60 | 35 | 0.829 | 0.793 | -0.036 |

Breakdown by 60/120/300 s - real {'120': 0.4435483870967742, '300': 0.8225806451612904, '60': 0.3225806451612903}, WM {'120': 0.4899193548387097, '300': 0.6693548387096774, '60': 0.28225806451612906}, keep-speed {'120': 0.024193548387096774, '300': 0.03225806451612903, '60': 0.008064516129032258}.
Queue: rel err -0.152, closing-lane -0.530, extent 0.176. Throughput rel err WM -0.008, keep-speed 0.213.
Secondary: Brier(300) WM 0.236 vs keep-speed 0.790; AUC 0.733; recovery of broken starts {'diff': 0.17602040816326525, 'n': 98, 'truth': 0.41836734693877553, 'wm': 0.5943877551020408}.
Lane-change rates per veh-h {'truth': {'0->1': 1.9235308561787618, '1->0': 4.9068888765673675, '1->2': 0.0009505176360626397, '2->1': 6.496154364064101}, 'wm': {'1->0': 0.7661306772431996, '2->1': 6.553833108975786}}.

## Action response detail

Candidates scored 844; pooled Spearman S -0.089; strict precision@k (S lower bound > 0) {'admitted_in_top': 2, 'admitted_total': 29, 'base_rate': 0.4166666666666667, 'k': 10, 'n': 240, 'precision': 0.5}.
Keep-speed reference: {'pooled_spearman_T': -0.03711456191113356, 'precision_at_k_S': {'admitted_in_top': 0, 'admitted_total': 29, 'base_rate': 0.5791666666666667, 'k': 10, 'n': 240, 'precision': 0.3}, 'precision_positive_T': {'agreement': 0.46919431279620855, 'base_rate': 0.5971563981042654, 'n': 844, 'n_pred_pos': 212, 'precision': 0.6320754716981132}, 'regret': {'acted_frac': 0.4084507042253521, 'n_moments': 142, 'none_regret': 3190.702160140845, 'random_regret': 3050.5053147649223, 'wm_gain': -119.58192323943659, 'wm_regret': 3310.284083380282}, 'within_moment_spearman_T': -0.1319754423132596}.

| family | n | Spearman T | precision WM-positive |
|---|---|---|---|
| R_random | 303 | 0.067 | 0.678 |
| abreast_wall | 182 | 0.182 | 0.780 |
| closing_pair | 278 | 0.096 | 0.569 |
| lane_balance | 27 | -0.332 | 0.478 |
| vsl_rows | 32 | 0.205 | 0.556 |
| zipper | 22 | 0.126 | 0.600 |

## Secondary car-level errors (pooled per shard)

```
{
 "note": "per-shard pooled summaries (merge = list)",
 "per_shard": [
  {
   "car_keep_speed": {
    "0.5": {
     "cav": {
      "ds_mean": 0.04938456434623939,
      "ds_median": 0.00973346943336395,
      "ds_p90": 0.14661580585626569,
      "dv_median": 0.019466938866477435,
      "dv_p90": 0.2932316117130324,
      "lane_agree": 1.0,
      "n": 64,
      "presence_agree": 1.0
     },
     "hdv": {
      "ds_mean": 0.04253734927596737,
      "ds_median": 0.010707050761116221,
      "ds_p90": 0.12859562609164604,
      "dv_median": 0.021414101522334222,
      "dv_p90": 0.2571912521832983,
      "lane_agree": 0.996,
      "n": 250,
      "presence_agree": 1.0
     }
    },
    "120.0": {
     "cav": {
      "ds_mean": 840.8675374590647,
      "ds_median": 993.631763947997,
      "ds_p90": 1372.7248681157548,
      "dv_median": 23.827101011186976,
      "dv_p90": 27.363557684006192,
      "lane_agree": 0.7777777777777778,
      "n": 9,
      "presence_agree": 0.7538461538461538
     },
     "hdv": {
      "ds_mean": 706.423762450179,
      "ds_median": 797.0914179877543,
      "ds_p90": 1120.16342859887,
      "dv_median": 18.82892538330881,
      "dv_p90": 26.469003732261157,
      "lane_agree": 0.7,
      "n": 40,
      "presence_agree": 0.7182539682539683
     }
    },
    "300.0": {
     "cav": {
      "ds_mean": null,
      "ds_median": null,
      "ds_p90": null,
      "dv_median": null,
      "dv_p90": null,
      "lane_agree": null,
      "n": 0,
      "presence_agree": 0.8461538461538461
     },
     "hdv": {
      "ds_mean": null,
      "ds_median": null,
      "ds_p90": null,
      "dv_median": null,
      "dv_p90": null,
      "lane_agree": null,
      "n": 0,
      "presence_agree": 0.8333333333333334
     }
    },
    "60.0": {
     "cav": {
      "ds_mean": 299.1640409702597,
      "ds_median": 144.6748545235797,
      "ds_p90": 805.5902404411853,
      "dv_median": 6.403586126268067,
      "dv_p90": 25.07666934514082,
      "lane_agree": 0.8292682926829268,
      "n": 41,
      "presence_agree": 0.9538461538461539
     },
     "hdv": {
      "ds_mean": 296.4602059572139,
      "ds_median": 83.13507753264787,
      "ds_p90": 921.5073250579283,
      "dv_median": 3.008559845900397,
      "dv_p90": 25.367289802503638,
      "lane_agree": 0.807909604519774,
      "n": 177,
      "presence_agree": 0.9722222222222222
     }
    }
   },
   "car_wm": {
    "0.5": {
     "cav": {
      "ds_mean": 0.06204347493447315,
      "ds_median": 0.029486127555742314,
      "ds_p90": 0.17996982347290214,
      "dv_median": 0.08584573533940387,
      "dv_p90": 0.6663486537840626,
      "lane_agree": 1.0,
      "n": 256,
      "presence_agree": 1.0
     },
     "hdv": {
      "ds_mean": 0.053474105150473406,
      "ds_median": 0.02860194239769953,
      "ds_p90": 0.16725850314924173,
      "dv_median": 0.08806141586685357,
      "dv_p90": 0.40523126030369866,
      "lane_agree": 0.988,
      "n": 1000,
      "presence_agree": 1.0
     }
    },
    "120.0": {
     "cav": {
      "ds_mean": 420.6931438741876,
      "ds_median": 241.05967107471884,
      "ds_p90": 1084.2642193433487,
      "dv_median": 5.329241671142766,
      "dv_p90": 23.56582201332135,
      "lane_agree": 0.7333333333333333,
      "n": 60,
      "presence_agree": 0.9538461538461539
     },
     "hdv": {
      "ds_mean": 348.1980427198387,
      "ds_median": 269.8914862684742,
      "ds_p90": 778.7064715406534,
      "dv_median": 7.357861476763205,
      "dv_p90": 16.66962048804608,
      "lane_agree": 0.7246376811594203,
      "n": 276,
      "presence_agree": 0.8958333333333334
     }
    },
    "300.0": {
     "cav": {
      "ds_mean": null,
      "ds_median": null,
      "ds_p90": null,
      "dv_median": null,
      "dv_p90": null,
      "lane_agree": null,
      "n": 0,
      "presence_agree": 1.0
     },
     "hdv": {
      "ds_mean": null,
      "ds_median": null,
      "ds_p90": null,
      "dv_median": null,
      "dv_p90": null,
      "lane_agree": null,
      "n": 0,
      "presence_agree": 0.996031746031746
     }
    },
    "60.0": {
     "cav": {
      "ds_mean": 123.63729529135158,
      "ds_median": 80.2383135078253,
      "ds_p90": 289.73444032726485,
      "dv_median": 3.5510532157361894,
      "dv_p90": 10.146552386039971,
      "lane_agree": 0.8414634146341463,
      "n": 164,
      "presence_agree": 0.9769230769230769
     },
     "hdv": {
      "ds_mean": 107.57341172671092,
      "ds_median": 70.81529189931939,
      "ds_p90": 224.72356354305958,
      "dv_median": 2.601673490707953,
      "dv_p90": 9.513637578555528,
      "lane_agree": 0.826271186440678,
      "n": 708,
      "presence_agree": 0.9861111111111112
     }
    }
   },
   "traffic_wm": {
    "J_cohort_120_err": 363.9021755675415,
    "J_cohort_120_true": 2863.9086087106766,
    "J_cohort_300_err": 336.7072140731237,
    "J_cohort_300_true": 3605.2636036404692,
    "J_cohort_60_err": 90.54342103032856,
    "J_cohort_60_true": 1642.821209923699,
    "downstream_density_mae": 2.5895215576005453,
    "downstream_speed_mae": 2.9262486094540665,
    "drop_density_mae": 10.212654320987655,
    "drop_speed_mae": 7.459954722756503,
    "upstream_density_mae": 2.250142985222284,
    "upstream_speed_mae": 2.9518871458335596,
    "zone_speed_mae": 7.654485145263368
   }
  },
  {
   "car_keep_speed": {
    "0.5": {
     "cav": {
      "ds_mean": 0.028669126410472912,
      "ds_median": 0.006854739178152158,
      "ds_p90": 0.0735351498706905,
      "dv_median": 0.01370947835626879,
      "dv_p90": 0.14707029974133692,
      "lane_agree": 0.9908256880733946,
      "n": 109,
      "presence_agree": 1.0
     },
     "hdv": {
      "ds_mean": 0.03055802210986231,
      "ds_median": 0.005802523342936183,
      "ds_p90": 0.05088653577467994,
      "dv_median": 0.0116050466864408,
      "dv_p90": 0.1070016523282078,
      "lane_agree": 0.9813664596273292,
      "n": 161,
      "presence_agree": 1.0
     }
    },
    "120.0": {
     "cav": {
      "ds_mean":
```
