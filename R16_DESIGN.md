# S-RAMPSPEED (R16): learned ramp-car speed adaptation (D2)

Branch `c4/v10r16` (from c4/v10r14b `7adf5663`). Build and offline inference only; **nothing trained or launched**.

## Defect and term

The WM ramp car follows only its own lane, drives to the end and merges from standstill; SUMO's LC2013 adapts speed to lane 1 and merges while moving.

`model.ramp_speed_adapt_learnable` (default off) applies to learned (HDV) slots on lane 2 inside the zone (`0 <= s <= 75.44`) that have a lane-1 leader L (the pivot's convention; the bumper gap g may be negative):

    a_v = max(IDM(v, v_L, g; v0, kappa*T, s0, a, b), -b)     ego's learned settings
    w   = sigmoid(omega) * sigmoid((D - d_end) / s0)          d_end = 75.44 - s
    a  <- a + w * min(a_v - a, 0)                             before the emergency envelope

**Scalars and init.**
- Three learned shared scalars (D, omega, kappa); the gate width is the learned s0, as in R13Y. No hand-set constants.
- Init D = 65 m, strength 0.4, kappa = 2.0: a grid fit of the 1 s-ahead speed on 894,620 TRAIN v5_native ramp rows with the 13E b2 learned population centre (RMSE 0.636 -> 0.470 m/s; `evidence_r16/init_fit.json`).

**Gradients and scope.**
- Pathwise in state, settings and scalars; neighbour choice is discrete.
- CAVs are excluded by the learned mask.

**Interactions.**
- The first-passage law, gs veto, overlap veto, reservation and lane-end stop are unchanged; the term only changes the acceleration fed to the pivot geometry.
- It is deceleration-only, so the driver_b envelope (applied after it) never conflicts.
- Yield and courtesy act on lane-1 followers, a disjoint set of slots.

**Acceptance hazard?** Only via approach speed. The per-second merge hazard at an open gap (fixed diagnostic law), 45-75 m before the end, was SUMO .15 vs R14 .04-.08; R16 gives .10-.12 with the same acceptance law. What remains is positioning, not acceptance:
- open-gap share is .12 vs SUMO .21;
- WM merges sit tight behind the leader (gap 3 vs 6 m) with large follower gaps (7 vs 3 m), which points to follower-side cooperation.

## Offline (13E b2 FINAL `952ec059`, R14 physics, CRN, TRAIN panel, gpuR CPU)

`R16_OFFLINE_REPORT.json`. Pooled values; brackets are state-bootstrap 95% CIs of R16-R14.

| 120 s, 48 states | SUMO | R14 | R16 | R16-R14 |
|---|---|---|---|---|
| merge dist before end (m, median) | 31.2 | 8.7 | 16.9 | +8.2 [3.6,11.7] |
| late (<15 m) share | .31 | .66 | .48 | -.18 [-.22,-.12] |
| standstill share | .34 | .68 | .58 | -.10 [-.16,-.05] |
| first seen -> merge (s) | 35 | 33 | 33 | 0 [-3,4] |
| merge-zone delay (veh-s) | 479 | 536 | 469 | -66 [-93,-42] |
| ramp delay (veh-s) | 497 | 487 | 543 | +55 [17,91] |
| lane-1 standing | 2.27 | 3.30 | 2.08 | -1.2 [-1.7,-0.8] |
| TTC<1.5 (pair-s) | 4.3 | 36.5 | 33.8 | -2.7 [-5.2,-0.4] |

300 s, 24 states (R14 -> R16 [CI], SUMO): merge dist 7.4 -> 11.9 [+0.5,+11.9] (38.6); late .70 -> .53 [-.23,-.12] (.22); standstill .72 -> .62 [-.15,-.06] (.25); first seen -> merge 40 -> 48 s [+1,+14] (47); ramp delay +157 [-151,+432]; lane-1 stops -.05 [-.09,-.01].

**Exploratory qualification units (cpuX).** Registered tools, 12 TRAIN states, 6 samples; the registered gate is untouched (`evidence_r16/R16_QUALCHECK.json`). Sign rate, set1/set2, R14 -> R16:
- pulse_m@5: .71 FAIL/.76 -> **.76/.88 PASS**
- gap_open@10: .67/.67 -> .67/.67 (INCONCLUSIVE, n=6)
- gap_open ramp-car Δmerge: -1.2/-1.7 -> -2.5/-0.9 s (SUMO +7.3)
- pulse_p: unchanged (R1 harness)

**Reading.** At the init, R16 closes about half of the merge-position gap, fixes merge-zone delay and the lane-1 column, and lowers TTC; it does **not** fix the gap_open sign, and ramp delay rises. Recalibration is needed.

## Verification

- **Flags-off bit identity vs the r14b tip:** PASS, 6 cases, 1222 tensors (`tools/r16_checks/bit_identity.py`).
- **Tests:** `test_r16_ramp_speed_adapt.py` (21: wiring, no-op selection, formula, scalar/state/setting FD, 3-step rollout FD, 10-seed CRN synthetic approach through the real pivot) and `test_r16_gate.py` (9) pass; R13E/R13Y/R14 suites 457 passed.

## Training plan (gpuB; not launched)

- **A** (`configs/c4_wm_v10round16.toml`): R14 A recipe (30 x 120 s) + ramp scalars at lr x1000; init = 13E b2 FINAL, as R14 A, so R16 A vs R14 A isolates the term (3 new tensors at the TRAIN init).
- **B**: R14 B schedule, scalars frozen. No core-frozen variant: the init alone leaves gap_open and ramp delay wrong.
- **Gate** `wms.train.round16_gate` (root `R16_ROOT`, default `/root/autodl-tmp/c4/v5`) reuses the 13E strict verdicts (trainable set, frozen bytes, recipe identity) + ramp checks (A trained, B frozen); preflight = configs, data, init sha, offline re-analysis. No R14 stop-rule waiters (gpuR-only states).
- **Run** (`repo_r16_<sha8>` on gpuB): `bash scripts/round16_gpuB.sh --smoke`, then `--chain`; ~4 CPU threads, `WMS_GPU_MEM_FRACTION=0.40`.
- **Next:** follower-side positioning; ramp entry speed.
