# Oldhist (history-conditioned WM-S) vs C1 structured WM-S

Checkpoint: `WM-S-OLDHIST-20261007/box/full/wm_s.pt` (sha256 6966c985..., `--rollout-mode driver_params`). C1 v2 sets re-materialised with the pre-t0 history window (`_out/c1_states_v2hist/`), entrance counters present, entries **OFF**, workers 4, horizon 120 s. Scored arm: PI for `u_same`/`u_new`, native for `seen`. Reference for the paired bootstrap: `c1struct`, 1000 resamples over worlds.

## 1. HDV |ds| (m), per-state median across states

| model | set | arm | 10 s | 30 s | 60 s | 120 s | blow-up rate @60/120 |
|---|---|---|---|---|---|---|---|
| oldhist | u_same | pi | 4.65 | 18.89 | 48.98 | 241.04 | 0.400 |
| oldhist | u_new | pi | 4.32 | 12.93 | 51.73 | 111.93 | 0.421 |
| oldhist | seen | native | 4.15 | 17.24 | 52.45 | 165.29 | 0.425 |
| c1struct | u_same | pi | 2.35 | 6.69 | 16.94 | 69.74 | 0.023 |
| c1struct | u_new | pi | 2.23 | 5.65 | 15.43 | 39.50 | 0.053 |
| c1struct | seen | native | 2.17 | 8.19 | 28.50 | 146.80 | 0.261 |
| blackbox_traj | u_same | pi | 2.92 | 10.79 | 23.79 | 156.48 | 0.031 |
| blackbox_traj | u_new | pi | 2.39 | 10.17 | 22.40 | 173.30 | 0.000 |
| blackbox_traj | seen | native | 2.65 | 13.27 | 32.31 | 196.17 | 0.327 |
| blackbox_id | u_same | pi | 2.37 | 9.88 | 28.17 | 94.48 | 0.123 |
| blackbox_id | u_new | pi | 2.04 | 10.69 | 23.59 | 83.70 | 0.105 |
| blackbox_id | seen | native | 2.43 | 14.08 | 47.71 | 217.45 | 0.438 |

## 2. Paired difference oldhist - c1struct (median HDV |ds|, 95% CI over worlds)

Positive = oldhist worse than C1 struct.

| set (arm) | 10 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|
| u_same (pi) | 2.2 [1.7, 2.7] | 11.5 [8.8, 13.7] | 25.6 [16.6, 31.9] | 96.5 [16.3, 193.2] |
| u_new (pi) | 2.1 [0.9, 3.4] | 8.9 [4.6, 17.8] | 30.4 [12.0, 37.5] | 13.7 [-8.2, 195.3] |
| seen (native) | 2.0 [1.2, 3.4] | 12.5 [5.3, 16.4] | 27.1 [13.4, 50.2] | 40.8 [-6.0, 352.6] |

## 3. Split by t0 near-merge mean HDV speed (`vnear`: free >20, congested 10-20, jam <10 m/s)

| set | class | n | oldhist | c1struct | bb_traj | bb_id | oldhist - c1struct (median) |
|---|---|---|---|---|---|---|---|
| u_same | free | 27 | 60.79 | 22.01 | 23.60 | 28.43 | +29.62 |
| u_same | congested | 77 | 50.19 | 18.17 | 25.17 | 27.45 | +29.01 |
| u_same | jam | 36 | 18.21 | 11.47 | 22.04 | 29.53 | +9.73 |
| u_new | free | 5 | 57.47 | 17.52 | 17.55 | 19.35 | +32.62 |
| u_new | congested | 11 | 53.13 | 15.59 | 25.06 | 34.07 | +34.60 |
| u_new | jam | 5 | 14.55 | 8.13 | 19.39 | 21.31 | +4.56 |
| seen | free | 7 | 45.99 | 19.32 | 18.51 | 36.04 | +26.68 |
| seen | congested | 14 | 56.80 | 14.28 | 25.03 | 28.98 | +44.90 |
| seen | jam | 5 | 28.65 | 17.38 | 18.54 | 38.77 | +11.27 |

## 4. Recovered per-vehicle driver params vs the three vTypes

- States encoded: 318; stored pre-t0 ticks per state: min 2, median 2, max 2; valid pre-t0 steps per slot ~1 (the encoder was trained on a 10-step window).
- Typed vehicles (offline `error_decomp.recover_types` labels): 32750; nearest-vType accuracy 0.6937 (normal-class base rate).
- Confusion (true -> predicted nearest vType):
  - `aggressive_car`: normal_car=3389
  - `normal_car`: normal_car=22718
  - `cautious_car`: normal_car=6643
- Recovered means `tau/min_gap/accel/decel` = 1.106 / 2.547 / 2.517 / 4.415 m-s (vType centres: aggressive 0.95/1.5/3.0/4.5, normal 1.10/2.5/2.6/4.5, cautious 1.50/3.5/2.0/3.0): the encoder collapses onto the normal/default driver for essentially every vehicle.

## Verdict

The history-conditioned `driver_params` model is **substantially worse** than the C1 structured model on every set and horizon. Per-state median HDV |ds|@60 s: oldhist 42.4 / 52.5 / 52.5 m on u_same / u_new / seen versus c1struct 16.2 / 17.1 / 28.5 m; the paired oldhist-c1struct difference is +25.6 m [+16.6, +32.0] on u_same (PI arm) and +27.1 m [+13.4, +50.3] on seen (native), each CI excluding 0. The blow-up rate rises from ~2-5% (c1struct, unseen) to 40-42% (oldhist).

The cause is not the history mechanism but its **absence at evaluation time**. Each C1 pack stores a decision window that begins ~1 tick before t0 (`run.tick_start = t0 - 0.5 s`); there is no pre-t0 trajectory in the packs. Re-materialising therefore yields a ~2-tick history with ~1 valid step per slot, whereas the oldhist encoder was trained on a 10-step window (the trainer synthesised history by shifting t0 forward with `--start-shift-prob 0.75` inside the stored window). Fed an almost-empty history, the encoder returns near-default (`normal`) parameters for every vehicle: nearest-vType accuracy 0.694 is exactly the normal-class base rate and every vehicle, including truly aggressive/cautious ones, is mapped to normal. oldhist thus behaves as a default-parameter physics model rather than the trained history-conditioned model.

Consequently the registration hypothesis - that pre-t0 driver identification helps most in **free** flow - is **not tested** by this comparison. The congestion split shows oldhist degrading most in free and congested states (+29 to +45 m at 60 s) and least in jam (+5 to +11 m), which is consistent with the default-parameter fallback mattering most where the unknown preferred speed dominates, not with history helping. A valid history-conditioned test requires either a pack that stores the H-step pre-t0 window at the registered t0, or an evaluation at a shifted t0 with the same window geometry as training.

Caveats: offline CPU scoring, no training and no SUMO; one checkpoint; entries OFF; bootstrap 1000, seed 0; 11-13 u_same / 2 u_new states are truncated before 120 s and excluded from that horizon and from the blow-up denominator. These are model-vs-model point estimates, not efficacy evidence.
