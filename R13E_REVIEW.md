# R13E_REVIEW (S-R13EREV, 2026-10-25): c4/v10r13e @ 8c7ff0a1

Read-only review plus local CPU tests. Nothing launched; gpuR was only listed (read-only).

**Verdict: GO-WITH-CHANGES.** The code is clean and off-parity holds. Three items block smoke/launch.

## Tests (rl_driving, PYTHONPATH=src;tests)
- R13E emergency + secure_gap + offline: 98 passed. Plan: 63 passed, but only with `--basetemp` inside the worktree. With the default basetemp it gives 45 errors and 3 failures, which are environmental. Total: 161 passed, matching the design's 160+1.
- Regressions (r13c_contract, r12c_physics_only, r12_base, r12_checks, r13a, r13f, r10b): 90 passed, 2 skipped (R12_SLOW).
- The flags-off bit-identity test against the actual 13C tip passed. Only intended paths differ from c4/v10r13c: physics/emergency, accel, config, merge_pivot, wms, configs, gate, tests and tools.

## Blocking
**B1. The stop rule will very likely stop stage A from the flag alone.** The inherited 13C rule compares the flags-on candidate's rollout hard-brake share (a<-3) with **flags-off** S-R12C, and stops at a ratio above 2. d11_run5 takes physics from each checkpoint, and the arm-set flags-off override is applied only to run_onestep. R12C_ROLLOUT gives S-R12C 0.1% versus 1.0% for the E proxy (SUMO 0.2%). A also has a single stage, so the "midrun" check (`wm_s_stage120.pt`) is really end-of-run. A STOP there also blocks B. Fix: make the reference arm S-R12C with the same flags on, so the comparison isolates training. Alternatively, register the flags-only ratio from the offline arms and a SUMO-referenced bound before smoke.

**B2. The emergency envelope caps only the IDM baseline.** Residual and courtesy are added afterwards (`follower_courtesy_total`), so in stage B, where both are trained, the final acceleration can exceed the envelope. Overlap-freedom is then lost exactly when it matters. SUMO applies vsafe to the final speed. Fix: apply `safe_accel` once to the final pre-clamp acceleration, the way R13Y applies its min-cap. Stage A is unaffected because its emissions are off.

**B3. The secure gap is not GAPACC `gs`, and the gate checks no values.** R13E requires `s0_f + v_f*T_f + relu(...)`. GAPACC `factworker.install_gate` used `T_F*v_F + relu(...)`, with **no s0**, and it accepted non-finite need. Adding s0 is SUMO-faithful (LC compares gap−minGap with secureGap) but stricter than the evidence. gs already rejected 26% of SUMO's own insertions and moved stuck +2.5 [−0.2, +5.8], so the starvation risk is higher and unmeasured. The emergency envelope is also not the evidence's E (which was `emergency_decel_mps2=9`). All GAPACC/R12CROLL numbers are therefore proxies. `check_offline` validates integrity and pairing only. Fix: either drop s0 to match gs, or keep it and register numeric criteria before smoke for the emergency+gs arm versus off and SUMO (merges/run, stuck, ramp wait, a<-3 share, post-insertion hb). Note: untracked `tools/r13e_checks/gs_equivalence.py` exists from another agent and was not reviewed.

## Verified (non-blocking)
- **Emergency physics.**
  - Constants: B = −accel_min_mps2 = 9 (plant clamp, also SUMO's default emergencyDecel). dt = 0.5. The smoothing width is sqrt(eps)·B, which is numerical only. s0 is learned.
  - Recursive feasibility: I checked it analytically for the clipped-trapezoid plant, for both cases u ≥ B·dt and u < B·dt. It holds provided leaders brake no harder than B.
  - Softmin never exceeds the true min, so the envelope cannot create an overlap.
  - Fidelity: it uses B for both cars and no headway τ. That makes it a last-resort envelope, more permissive than SUMO's vsafe, and the design says so.
  - Gradients: probed on 200k random fp32 states. No NaN. |da/dgap| and |da/ds0| ≤ 8 s⁻² (2/dt²), |da/dv| ≤ 4, p99 ≤ 2.3. No stiffness blow-up.
- **Veto.** Uses per-slot learned T, b and s0, gathered by neighbour index. The overlap veto is unchanged. Recorded replay and courtesy intent use the ungated boundary, so no recorded merge gets zero mass and there are no NaNs (tested). The surrogate scale is the existing tau, frozen in both stages. No new tensors.
- **Stage A.** Trainable set is exactly R12C's `EXPECTED_TRAINABLE_TENSORS`. Init is as-is. Differences from R12C are init, the flags, courtesy loss 0/contrast off (frozen head), and prior 0. `check_configs` enforces all of this.
- **Stage B.** `check_configs` enforces resolved equality to 13C plus the two flags and paths. The A→B init applies the 13C transforms once (courtesy zero, no residual reinit). A FINAL keeps R12C's residual/courtesy bytes. `check_init` enforces this.
- **gpuR.** `/root/autodl-tmp/c4/v5`, `/root/miniconda3/bin/python`, the R12C sha (c76c338d…, matches) and r9 stage300 all exist. No r13e artifacts yet. `check_source` enforces `repo_r13e_<sha8>`. noclobber, mkdir locks and prior-artifact refusal are present. The pidfd-only signalling and the stop-reason self-test are inherited from 13C.
- **Owner rules.** Driver settings are learned only. No timers, forced merges, padding or likelihood floors. Overlap veto intact. `merge_pivot_state_st=true`. Integrator unchanged (no Euler).

## Notes
- In stage A the veto surrogate sends merge-outcome gradient into the learned T, b and s0. Watch car-following BASE.
- CAV neighbours use encoder-inferred T and b.
- The gate has no GPU-busy guard. 13C production (PID 527550, 4 h in) still holds gpuR, so launch only after it finishes.
- `.round13e_gate.lock` is never released. A re-smoke needs a manual, recorded operator step.
- **R13Y** (`c4/v10r13y`): **not merge-ready**. It is uncommitted work-in-progress on eb75010c: 6 modified files, untracked `merge_yield.py` and 2 tests. It overlaps wms.py, legacy_flat.py and train.py. After merging: re-run combined off-parity, test the yield cap together with the emergency envelope and secure gap (`merge_end_blocked`), re-prepare `repo_r13e_<sha8>`, and redo the offline study on the merged code.
