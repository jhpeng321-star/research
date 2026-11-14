# OLDHIST recovered driver parameters: original (~1-tick) vs shifted (10-step) history

Checkpoint `_out/shift_ckpt/oldhist/wm_s.pt` (sha256 6966c985..., rollout_mode driver_params), same physics. Offline labels are the deterministic scenario replay (`recover_types`); nearest-vType is over `(time_headway, min_gap, accel, decel)` vs the three vType centres.

| history | valid pre-t0 steps (median) | typed vehicles | nearest-vType accuracy | normal base rate |
|---|---|---|---|---|
| original (~1 tick) | 1 | 32750 | 0.6937 | 0.6937 |
| shifted (real 10-step) | 10 | 31429 | 0.4242 | 0.6945 |

Paired on the 16381 (world, vehicle) identities recovered in both runs: original accuracy 0.6921 (all mapped to normal), shifted accuracy 0.4353.

Predicted-class counts (paired subset): original {"aggressive_car": 0, "normal_car": 16381, "cautious_car": 0}; shifted {"aggressive_car": 4073, "normal_car": 4601, "cautious_car": 7707}.

Recovered means (tau / min_gap / accel / decel):

| history | time_headway | min_gap | accel | decel |
|---|---|---|---|---|
| original | 1.141 | 2.647 | 2.407 | 4.115 |
| shifted | 1.175 | 2.859 | 2.067 | 3.785 |

Shifted confusion (true -> nearest vType):

| true / pred | aggressive | normal | cautious |
|---|---|---|---|
| aggressive_car | 1538 | 830 | 881 |
| normal_car | 6238 | 6878 | 8712 |
| cautious_car | 50 | 1386 | 4916 |

## Interpretation

The original states reproduce the published result exactly (n_typed 32750, accuracy 0.6936793893129771, confusion all -> normal): with ~1 valid history step the encoder collapses to the physics default (normal) driver. With the real 10-step history the encoder responds to the observed motion and its recovered parameters spread across all three vType classes (paired predicted counts aggressive/normal/cautious = 4073/4601/7707 vs 0/16381/0 before). However the recovered parameters are not calibrated to the SUMO vType labels: nearest-vType accuracy falls to 0.435, below the 0.692 normal-class base rate, because the encoder was trained by trajectory loss alone (no driver-parameter sidecar supervision). Real history therefore changes the recovered parameters substantially but does not by itself yield correct vType identification; the hybrid run needs the WM-S2 driver-parameter supervision for calibrated recovery.
