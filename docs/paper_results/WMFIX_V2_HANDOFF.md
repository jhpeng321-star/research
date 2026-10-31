# PILOT v2 handoff (Claude -> GLM executor), 2026-10-31 ~15:20 +08

## Done (gpuA /root/autodl-tmp/c4/wmfixmerge20261031 = R; all own work)
- Data (change 1): wmdata views rsynced cpuQ->gpuA /root/autodl-tmp/c4/wmdata20261031/views,
  2443 files 0 mismatches vs SHA256SUM.json (R/v2/WMDATA_VIEWS_VERIFY.json). merge_corpus ->
  R/v2/corpus (12 FIT + 40 wmdata worlds, 236 episodes, FIT mappo excluded); prepare ->
  R/v2/TRAIN_PREPARED_V2.json sha 84751dc7... (PREP_SHA256.txt, PREP_SUMMARY.json: source
  scale 53.98, not the FIT-only eps floor). HARNESS_CONTRACT_V2.json e67850f3... (new runtime).
- Code staged + hash-matched to c4/wmfixmerge (wmfixc merge, merge_corpus, feasibility flags,
  pilot_v2 7cc913be with mini/full concurrency fix, panel_eval_v2). Backup of the replaced
  staged files: /root/autodl-tmp/c4/wmfix_v2_prestage_backup/pre_v2_tree.tar.
- Infra for wmdata arms: R/_in/pi_deps (frozen controller sources, 69 files = manifest
  b3c031ad) and R/_in/ortdeps (onnxruntime symlinks). PYTHONPATH must be src:.:R/_in/ortdeps.

## FD feasibility (evidence: docs/paper_results/WMFIX_V2_FEAS.md)
- FIT-12 preview FEASIBILITY_SMOOTH_FIT12.json (152e7a03): FAIL clearly bad (FD 5.7e17 vs
  analytic 0.0126; loss 0.045 -> 1.3e14 after one Adam step) = eps source scale on FIT-only
  data. Surrogate level is also ~30x low (79.6 vs exact 2375.5 veh-s) on every binding.
- REGISTERED binding R/v2/FEASIBILITY_V2c.json (PID 189176, log v2/feas_v2c.log): STATUS_AT_HANDOFF

## Running PIDs (own; never kill anything else - S-WMCALIB has 12 GPU procs on gpuA)
- 189176 feasibility V2c; 188758 panel_eval_v2 smoke (FIT group 40 = w973002 with v1
  step_000000; parity target from v1 PANEL_step0_r0 group 0: 120 s 449.627 / 300 s 302.891)
  -> R/v2/smoke/SHARD_fit973002_v1step0.json.
- 191834 launcher v2/launch_pilot_v2.sh: waits for FEASIBILITY_V2c; launches pilot_v2 only if
  PASS + launch_ready + finite_difference_ok, else writes v2/PILOT_V2_NOT_LAUNCHED.json.
  Pilot out R/v2/pilot (RUN.json has the python PID), log v2/pilot_v2.log, 21570 s cap.

## Next steps
1. Check smoke parity (exact match to v1 values = same eval path) and FEASIBILITY_V2c.
2. If NOT_LAUNCHED: report to MAIN (do not retune). Repair candidates in WMFIX_V2_FEAS.md
   (capacity-limited surrogate; or v2 data-only with smoothing OFF) need MAIN approval.
3. If launched: poll v2/pilot (update_*.json, SHARD_mini_*.json, PILOT_V2.json). PASS ->
   pilot continues the curriculum itself; then run the staged sharded verdict
   (WMFIX_LAUNCH_READY.md section "MAIN ADDENDUM 12:45") with the FINAL checkpoint + R/v2/pilot/
   candidate_config.json. Record WMFIX_PILOT_V2.md + REGISTRATION.md lines.
4. Each panel subprocess re-loads 236 episodes (~20 min under load): mini panels lag
   training by ~30 min, so early stop is weak; caching the Runtime is a cheap speed-up.
