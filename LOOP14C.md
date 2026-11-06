# LOOP14C: CAV-side fixes for the policy loop (S-CAVSIDE, 2026-10-25)

Branch `c4/loop14c` (from loop14 1785c999). Basis: `c4-pulsep/PULSEP.md`. All switches are off by default; off is bit-identical to loop14.

## Switches

- **(1) `policy_ramp_merge`** (`[policy14]`): policy CAVs are never auto-merged.
  - A ramp CAV merges only by its own lane-head `left`, which is already in the action space.
  - The merge goes through the gap mask, the consequence mask and the optional 13E `secure_gap_hook`.
- **(2) `native_ramp_merge="wm_law"`**: any CAV that is still auto-merged stays on the ramp in the bridge.
  - It is handed to the WM through `cav.WM_MERGE_KEY`.
  - `_apply_merge_pivot` then decides it like an HDV: first passage, 13E secure-gap veto on its learned settings, overlap veto, lane-end stop.
  - No constants are added. A non-first-passage WM raises.
- **(3) `counterfactual.native_cav="head"`**: the existing learned native-CAV head.
- `configs/c4_policy14_loop14c.toml` = (2) + (3), with (1) off. It is not registered.

**Parity caveat.** In SUMO, policy CAVs run laneChangeMode 513, so LC2013's strategic ramp merge stays on; "stay" sends no request. SUMO therefore matches (1) off + (2) on. Turning (1) on also needs a SUMO executor change (strategic bits off for policy ramp CAVs), which I did not make. SUMO also defers ramp requests within v·1 s + 10 m of the edge end; the WM mask does not.

## R1 usage

- **Where it enters the loop:** `dreamer.imagine` (`native_fn`) drives the native CAVs of counterfactual and entrance-zone rows. Under policy14, R1 reaches the actor only through the sampled-marginal lane advantage (flip rows, weight 0.25). It also enters the report-only TTC `native_relative` reference and `native_relative_gain`.
- **Already from real SUMO:** the critic, real returns (`loop14_returns`) and `c4_pairs`.
- **The qualification tools on `c4/actqual`** use the twin for every CAV; I left them unchanged.
- **The learned head** (`configs/native_cav_head_v5.json`) was fitted by `tools/native_cav_fit.py` on v5_native TRAIN-val CAV trajectories: 58 worlds, 451k rows, stop RMSE 0.20. It is longitudinal only; refit with `--data-root` set to converted native-arm episodes.
- **With (3) on, the loop no longer uses the IDM twin.**

## Verification

- **Bit identity, flags off:** a seeded Policy14 imagination (all, base, group and flip rows; deterministic and sampled; synthetic hybrid WM with the 13E veto) matches the loop14 tip on 55/55 tensors (`tools/loop14_checks/loop14c_bitid.py`).
- **`tests/unit/test_loop14c.py` (13 tests):**
  - (1) the CAV waits, merges exactly when it asks, and the mask and gs refusals hold.
  - (2) the native ramp CAV matches the HDV ramp car's lane, `merge_pivot_c` and commit over 10 steps (strict/hybrid × accept/veto).
  - FD gradients through the delegated pivot path and through the head.
- **Suites:** 447 passed (loop14, policy14, merge-pivot and related). Flags-on `Policy14Trainer.update()` stays finite.

## Offline check (EXPLORATORY; the registered gate is unchanged)

- **Model:** 13E stage-A `wm_s.pt` (md5 dbe4d52b).
- **Data:** the 12 S-ACTQUAL TRAIN states (copied before the cpuN reclaim).
- **Method:** the registered `qual_wm_pulse` / `wm_coop_k` (test A) and `score_units` with `TOLERANCES_v2`, 6 samples per variant.
- **Scratch:** `E:/agent-work/c4-loop14c-scratch/qual`; scripts and JSON are in `tools/loop14_checks/loop14c_qual/`.

The table gives the sign rate as set1 / set2.

| unit (n) | off | wm_law (2) | (2)+head |
|---|---|---|---|
| pulse_p/follower@5 (7) | .29/.29 | .71/.57 | .71/**.86 PASS** |
| pulse_m/follower@5 (17) | .71 FAIL/.76 | **.82/.88 PASS** | .76/.76 PASS |
| pulse_m/follower@10 (18) | .83/.83 | .89/.89 | .78/.83 |
| gap_open/follower@10 (6) | .50/.33 | .33/.17 FAIL | .50/.17 FAIL |
| pulse_p target Δv sign@2 s | 1/7, 0/7 | 1/7, 0/7 | **6/7, 6/7** |

- **Ramp-merge timing (R2):** with off, the ramp target merges at 1 s in 4 targets where SUMO waits at least 9 s or more than 30 s. With wm_law, every one of those samples waits at least 8 s.
- **Conclusions:**
  - (2) fixes R2 timing and moves pulse_m and pulse_p toward SUMO.
  - (3) fixes the pulse_p reference (R1).
  - gap_open is unchanged: n=6, and the ramp-car Δmerge is −1 to −4 s in the WM vs +7.3 s in SUMO.
