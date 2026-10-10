# RUN1_EVAL_FAIR - fair run-1 comparison (TRAIN, 20 states, offline; no SUMO/GPU)
Status: complete. Delay unchanged; car metrics native arm, |ds| m, |dv| m/s. Checkpoint wm_s.pt sha256 97d8f998....
Fairness: CAVs in every spec replay the recorded arm (teacher-forced, max CAV |ds|/|dv| deviation across keep_speed_tf/pure_physics_tf/wms_run1 = 0.000122); car metrics use the COMMON matched set (present in recorded and matched by all 5 specs).
States: 20; horizons 5/10/30/60/120/180/300 s; baseline variants keep_speed_tf and pure_physics_tf added so no spec gains a CAV-error-0 advantage.

## Matched fraction (recorded-present vehicles matched; median over states), horizons 5,10,30,60,120,180,300
- keep_speed [0.96 0.92 0.76 0.53 0.14 0.02 0.00]
- keep_speed_tf [0.96 0.91 0.73 0.50 0.12 0.01 0.00]
- pure_physics [0.95 0.91 0.75 0.53 0.14 0.02 0.00]
- pure_physics_tf [0.95 0.90 0.73 0.50 0.12 0.01 0.00]
- wms_run1 [0.91 0.83 0.59 0.38 0.03 0.00 0.00]
- COMMON (all 5 specs) [0.91 0.83 0.59 0.38 0.03 0.00 0.00]; common set size median [93 86 60 39 4 0 0]; states nonempty [20 20 20 20 14 6 0]

## Car |ds| median on COMMON set, per stratum (horizons 5,10,30,60,120,180,300)
- keep_speed hdv 0.7 2.4 13.7 56.7 1341.2 1830.6 n/a | cav 4.4 11.3 25.5 132.0 1548.5 2198.7 n/a
- keep_speed_tf hdv 0.7 2.4 13.7 56.7 1341.2 1830.6 n/a | cav 0.0 0.0 0.0 0.0 0.0 0.0 n/a
- pure_physics hdv 2.8 8.0 23.7 46.4 694.6 2069.5 n/a | cav 4.2 10.4 31.6 55.8 1054.0 2298.5 n/a
- pure_physics_tf hdv 2.8 8.0 23.7 46.4 694.6 2069.5 n/a | cav 0.0 0.0 0.0 0.0 0.0 0.0 n/a
- wms_run1 hdv 0.9 2.4 6.2 17.4 157.2 239.1 n/a | cav 0.0 0.0 0.0 0.0 0.0 0.0 n/a

## Car |dv| median on COMMON set, per stratum (horizons 5,10,30,60,120,180,300)
- keep_speed hdv 0.2 0.3 0.7 1.6 20.8 13.9 n/a | cav 1.2 1.2 1.2 3.2 24.7 12.9 n/a
- keep_speed_tf hdv 0.2 0.3 0.7 1.6 20.8 13.9 n/a | cav 0.0 0.0 0.0 0.0 0.0 0.0 n/a
- pure_physics hdv 0.9 1.1 0.9 1.1 23.3 23.9 n/a | cav 1.2 1.6 1.0 1.3 23.8 23.4 n/a
- pure_physics_tf hdv 0.9 1.1 0.9 1.1 23.3 23.9 n/a | cav 0.0 0.0 0.0 0.0 0.0 0.0 n/a
- wms_run1 hdv 0.7 0.4 0.4 0.9 1.3 2.4 n/a | cav 0.0 0.0 0.0 0.0 0.0 0.0 n/a

## Why wms matches fewer vehicles
wms presence is a cumulative mask (wms.model.rollout ANDs each step's teacher presence), and its teacher marks junction-lane (lane<0) or missing recorded rows absent; once dropped a vehicle never returns. keep_speed keeps every t0 vehicle present. So transient junction/exit or entry events permanently remove vehicles from wms (and it has no entry model), lowering its matched fraction.

## 300 s column
Cannot be computed: at 300 s the 5-spec common matched set is empty in all 20 states (median size 0). No t0 vehicle remains at t0+300 in the recorded run for 19/20 states (network residence < 300 s) and the predictors add no entries; wms/pure_physics match none at 300 s, keep_speed alone retains a t0 vehicle in 1/20 states, so the common set stays empty.

## Verdict (wms_run1, on the common set)
With identical CAV motion and the common matched vehicle set, wms_run1 has the lowest median |ds| among all four baselines (keep_speed, keep_speed_tf, pure_physics, pure_physics_tf) at 30-180 s (per-horizon n/n/Y/Y/Y/Y/?); it is not lowest at 5-10 s. Support thins: the common set is nonempty in only 14/20 states at 120 s and 6/20 at 180 s, and 300 s is not computable (empty common set in all states).
