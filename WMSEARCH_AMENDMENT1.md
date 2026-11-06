
## S-WMSEARCH AMENDMENT 1 (2026-11-06 21:10 +08; owner direction via the coordinator ~20:45; BEFORE any wm_search physics)
- Replaces the moment-level design of the 20:20 registration (WMSEARCH_REGISTRATION.md sha256 43035ac1): the WM search
  is now STRATEGY-LEVEL and condition-aware, and the incumbent to beat is solo2. solo2 is S-REGIME/S-REGIMEU0
  wall_h_solo2: wall trigger geometry, only the SUMO-lane-2 (left) member holds 15 m/s for 90 s, dmax 1.5, and the
  cooldown is 90 s. On MAPPO u0 it gives -24% at D3200 and -14% at D2800_p60, and it hurts at D2400.
- Phase cap: wm_search <= 280,000 sim-s. Runs only after WM-E's final checkpoint PASSES WM_QUAL_GATE; if it does not
  pass, nothing runs and MAIN decides.
- Strategy space (`tools/wzprop/strategy_search.py`; all values come from the registered lcm_plan ranges):
  - trigger: wall geometry, or per-lane rank picks in a registered zone with a spread of <= 200/300 m;
  - 1-6 roles, each with a lane, a rank, v_w 15/18/22, dmax 1.0/1.5, hold 90/120/180 s, onset 0/20/40/60 s
    (stagger), and an optional adjacent lc_target;
  - cooldown and one slot.
  The same select/program functions run in WM imagination and in SUMO (`strategy_hook.StrategyCore`). The SUMO
  runner is the S-REGIMEU0 plant: hold law requested through the single writer, lanes through the G5 lane path.
- WM search, zero SUMO, GPU after WM-E finishes:
  - per demand cell, start states are WM-D's TRAIN tapes (observable: CAV self, CAV-detected, RSU counters);
  - one activation per state, H = 300 s, CRN-paired with the base and solo2;
  - score = world-cluster LCB of the mean gain vs solo2, minus 0.5 x disagreement/sqrt(n);
  - MAP-Elites archive over (n CAVs, lanes, lane changes, speed bin, stagger, zone, trigger);
  - evolution from seeds (solo2 and its variants, REGIME rules, lane-2 hold combinations) plus random strategies.
  Output: the best candidate beating solo2 per cell.
- Arms for SUMO verification, matched effort (4 strategies each):
  - WM = the top WM-LCB strategies from distinct archive cells (solo2 excluded);
  - structured = fixed, pre-registered: solo2_v18_h90, solo2_v15_h120, l2hold_l1gap, l2hold_stagger;
  - random = 4 seeded random strategies from the same space (seed 20261107).
- SUMO verification (`tools/wzprop/wmsearch_u0.py`):
  - base MAPPO u0, the S-REGIMEU0 harness, the same u0 checkpoint (sha checked against S-REGIMEU0 PLAN.json);
  - cell D3200_p60; paired seeds 989770-989785 (16 seeds; REGIME sweep TRAIN-type range; unused by S-REGIME
    989700-989723/989899, S-REGIMEU0 989850-989865, S-LOWSHARE-U0 989820-989849);
  - per seed: u0, solo2 and the 12 arm strategies, 1,200 s each (warm-up 300 s);
  - 16 x 14 x 1,200 = 268,800 sim-s, within the cap;
  - host cpuV, nice 19 with CUDA_VISIBLE_DEVICES empty (S-REGIMEU0 dispatcher); own PIDs only.
- Primary result:
  - per strategy, the paired diff J(strategy) - J(solo2) over [300, 1200) s, with a 10,000-draw seed bootstrap 95% CI
    (RNG 20261107; negative = better);
  - per arm, the number of strategies whose CI upper bound is below 0, and the best strategy.
  - Secondary: vs u0; safety (collisions, teleports, TTC, hard braking); activations.
  - A GROUP strategy beating solo2 with CI < 0 would be the WM-discovery result.
  - No other gate.
- Known limit: the WM's base in imagination is the WM-native CAV law (planapi models non-member CAVs as native
  drivers), not MAPPO u0. So the WM ranks strategies under a native-like base, and SUMO on u0 decides.
