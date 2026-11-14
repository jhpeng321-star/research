# RUN1_EVAL - first trained WM-S model vs baselines (TRAIN, 20 states, offline; no SUMO/GPU)
Status: complete. Delay = normalized seconds sum((1-v/v_free)*dt). Arm native; |ds| m, |dv| m/s; per-state median/p90 aggregated across 20 states.
Checkpoint wm_s.pt (sha256 97d8f998...); physics replayed from TRAIN_RECEIPT (open_road, sumo_fidelity, sumo_idm). Horizons 5/10/30/60/120/180/300 s.

## Car |ds| median / p90 (native) over horizons 5,10,30,60,120,180,300
- keep_speed hdv 0.8 2.6 16.8 73.4 1403.7 2259.7 960.7 / 17.7 63.2 362.2 858.7 2425.9 2643.8 1678.5 | cav 5.0 11.4 33.7 134.8 1746.3 2448.6 1813.6 / 18.9 63.1 310.0 589.7 2150.3 2680.4 1813.6
- pure_physics hdv 2.9 8.2 28.6 60.1 976.7 2113.9 n/a / 9.6 28.0 128.0 371.0 1207.8 2388.4 n/a | cav 3.9 10.4 41.7 72.3 1049.4 2165.2 n/a / 12.0 30.2 106.1 109.0 1307.4 2276.6 n/a
- wms_run1 hdv 0.9 2.4 6.2 17.4 157.2 239.1 n/a / 4.8 12.2 40.7 105.6 323.5 524.9 n/a | cav 0.0 0.0 0.0 0.0 0.0 0.0 n/a / 0.0 0.0 0.0 0.0 0.0 0.0 n/a
- wms_stage10 hdv 0.9 1.5 164.9 927.4 1830.9 1702.5 n/a / 4.7 10.0 237.4 1023.8 1915.4 1775.3 n/a | cav 0.0 0.0 0.0 0.0 0.0 0.0 n/a / 0.0 0.0 0.0 0.0 0.0 0.0 n/a
- wms_stage30 hdv 0.8 1.5 4.0 66.5 198.3 326.7 n/a / 4.9 11.8 39.4 159.3 586.2 1229.3 n/a | cav 0.0 0.0 0.0 0.0 0.0 0.0 n/a / 0.0 0.0 0.0 0.0 0.0 0.0 n/a
- wms_stage60 hdv 1.0 1.9 6.4 18.6 208.9 262.6 n/a / 4.9 11.6 43.1 68.6 385.1 304.7 n/a | cav 0.0 0.0 0.0 0.0 0.0 0.0 n/a / 0.0 0.0 0.0 0.0 0.0 0.0 n/a
- wms_stage120 hdv 0.9 2.4 6.2 17.4 157.2 239.1 n/a / 4.8 12.2 40.7 105.6 323.5 524.9 n/a | cav 0.0 0.0 0.0 0.0 0.0 0.0 n/a / 0.0 0.0 0.0 0.0 0.0 0.0 n/a

## Car |dv| median / p90 (native) over horizons 5,10,30,60,120,180,300
- keep_speed hdv 0.2 0.4 0.9 2.0 18.7 16.4 12.1 / 5.5 10.5 20.3 24.3 26.3 21.7 14.1 | cav 1.2 1.4 1.3 2.9 24.0 11.7 20.5 / 6.2 10.3 14.6 22.6 25.1 15.3 20.5
- pure_physics hdv 0.9 1.1 1.2 1.5 22.4 17.1 n/a / 3.0 4.4 8.2 16.0 24.2 23.8 n/a | cav 1.2 1.6 1.5 1.4 22.8 13.2 n/a / 3.5 4.3 8.1 5.0 24.5 13.2 n/a
- wms_run1 hdv 0.7 0.4 0.4 0.8 1.3 2.4 n/a / 2.1 1.7 3.6 6.4 4.1 3.5 n/a | cav 0.0 0.0 0.0 0.0 0.0 0.0 n/a / 0.0 0.0 0.0 0.0 0.0 0.0 n/a
- wms_stage10 hdv 0.3 0.5 21.3 26.0 1.2 1.4 n/a / 1.4 1.6 27.3 27.9 2.3 2.2 n/a | cav 0.0 0.0 0.0 0.0 0.0 0.0 n/a / 0.0 0.0 0.0 0.0 0.0 0.0 n/a
- wms_stage30 hdv 0.4 0.4 0.5 3.4 3.9 6.8 n/a / 1.9 1.8 2.1 10.0 13.2 14.1 n/a | cav 0.0 0.0 0.0 0.0 0.0 0.0 n/a / 0.0 0.0 0.0 0.0 0.0 0.0 n/a
- wms_stage60 hdv 0.5 0.4 0.4 0.9 1.6 1.3 n/a / 2.0 1.7 2.2 4.9 5.2 3.7 n/a | cav 0.0 0.0 0.0 0.0 0.0 0.0 n/a / 0.0 0.0 0.0 0.0 0.0 0.0 n/a
- wms_stage120 hdv 0.7 0.4 0.4 0.8 1.3 2.4 n/a / 2.1 1.7 3.6 6.4 4.1 3.5 n/a | cav 0.0 0.0 0.0 0.0 0.0 0.0 n/a / 0.0 0.0 0.0 0.0 0.0 0.0 n/a

## Lane agreement (HDV; CAV) and matched fraction (HDV), per horizon
- keep_speed lane hdv[0.97 0.96 0.93 0.92 0.88 0.93 1.00] cav[0.96 0.94 0.84 0.89 0.86 0.81 1.00] matched hdv[0.96 0.91 0.75 0.51 0.12 0.04 0.00]
- pure_physics lane hdv[0.92 0.91 0.88 0.86 0.80 0.68 n/a] cav[0.85 0.84 0.74 0.76 0.76 0.67 n/a] matched hdv[0.95 0.91 0.75 0.51 0.12 0.04 0.00]
- wms_run1 lane hdv[0.98 0.97 0.93 0.85 0.77 0.88 n/a] cav[1.00 1.00 1.00 1.00 1.00 1.00 n/a] matched hdv[0.91 0.82 0.59 0.38 0.06 0.01 0.00]

## Traffic per section, 30 s bins (recorded->predicted, mean over bins/states; speed/stopped/queue/delay)
- upstream keep_speed[speed 23.02->13.06 stopped 1.21->0.91 queue 0.00->0.00 delay 532.70->139.81] | pure_physics[speed 23.02->26.51 stopped 1.21->0.02 queue 0.00->0.00 delay 532.70->38.40] | wms_run1[speed 23.02->15.49 stopped 1.21->0.69 queue 0.00->0.00 delay 532.70->94.07]
- merge keep_speed[speed 9.17->7.83 stopped 1.27->1.07 queue 8.04->8.49 delay 192.58->62.30] | pure_physics[speed 9.17->22.25 stopped 1.27->0.00 queue 8.04->0.01 delay 192.58->12.01] | wms_run1[speed 9.17->15.16 stopped 1.27->0.13 queue 8.04->0.55 delay 192.58->19.53]
- downstream keep_speed[speed 25.68->24.22 stopped 0.00->0.03 queue 0.00->0.00 delay 38.19->342.60] | pure_physics[speed 25.68->26.70 stopped 0.00->0.00 queue 0.00->0.00 delay 38.19->59.17] | wms_run1[speed 25.68->11.34 stopped 0.00->0.51 queue 0.00->0.00 delay 38.19->29.31]

## Paired PI-native (merge, predicted vs recorded): sign agreement / rank corr
- keep_speed mean_speed 28/176 rho=n/a (-0.14->0.00); total_delay 36/200 rho=n/a (1.21->0.00)
- pure_physics mean_speed 26/80 rho=-0.17 (-0.16->-0.00); total_delay 44/200 rho=-0.07 (1.21->0.01)
- wms_run1 mean_speed 56/86 rho=0.11 (-0.04->-0.25); total_delay 82/200 rho=0.19 (1.21->0.20)

## Verdict (wms_run1)
wms_run1 beats median |ds| of keep_speed+pure_physics up to 180 s (per-horizon Y/Y/Y/Y/Y/Y/?); beats both on |dv| at 5/7 horizons.
Plot: `median_ds_run1.svg` (median |ds| all vehicles vs horizon, all specs). Per-state detail in each run's EVAL.json.
