# PILOT v2 handoff (Claude -> GLM executor), 2026-10-31 ~15:20 +08

## SUPERSEDING MAIN17:05 recovery / Sol17:17 gpuA clock
- Old dispatcher197030 FINISHED/exit1 after exact parity PASS and4 updates;
  step0 panel failed because WMDATA CTRL0 entrant feedback was unbound.
  Infrastructure BLOCKED, not a measured gate FAIL. Old outputs preserved.
- Repair committed55733a60; local29 CPU PASS / staged31 CPU PASS (including
  direct-row reproduction and exact step4/Adam resume). See WMFIX_PILOT_V2DATA.md.
- **ONE LIVE dispatcher gpuA222535/start881844640**, PPID1/SID222535,
  `tools.wmfixmerge.recover_data_pilot --execute`, waiting>=14336MiB.
  Authoritative NEW `R/v2data/recovery_1705/{DISPATCH,STATUS}.json`, EVENTS.jsonl.
  At17:17 WAIT_GPU_MEMORY11485MiB. Never start a second dispatcher.
- UPDATE17:20:12: launched after15337MiB free; timeout223221/python223222,
  remaining20528s cap. Corpus initializing; read NEW STATUS/logs before action.
- RECOVERY.json sha056c295c441bff291ec3548a47be2b1c0dffd00c085e53a51e0dd5f59bca0d10;
  prepared84751dc7/baseb8797242 unchanged; new contract3cc03fed changes only
  Runtime hash vs parente67850f3; controllerscd47e8ad binds recorded CTRL0 +
  speed-MAPPO identities. Only Runtime/policy binding/hash-list + driver changed.
- New pilot will resume exact step4/Adam into `recovery_1705/pilot`, retaining
 1041.774795s already spent (remaining timeout20528s). No parity/update repetition.
  Full52 panels0/10/20/30, same gate/exclusions/lr/curriculum/CRN. Each full-panel
  receipt is published immediately upon all four shards finishing.
- Monitor NEW pilot FULL_PANEL_STEP10.json / PILOT_V2.json. PASS auto-continues;
  FAIL stops. Verdict remains staged, never automatically launches. No peer touched.
- UPDATE17:45:12: step10 saved8bbd3a6d (finite/frozen kept); all14 inherited
  files byte-identical. Three step0 shards run; step10 PANEL is still pending.
  Checkpoint completion is NOT a full-panel result or pilot PASS.

## SUPERSEDING MAIN 15:25 v2-DATA / Sol resume 15:34 gpuA clock
- MAIN approved DATA ONLY, smoothing OFF, amended v1 30-step pilot gate. Code:
  81ddfb5d + 6a53549c; dispatch evidence e5b52ebb. See WMFIX_PILOT_V2DATA.md.
- **ONE LIVE DISPATCHER: gpuA PID197030/start881175520**, PPID1/SID197030,
  `tools.wmfixmerge.launch_data_pilot --execute`, root R (below). Do NOT duplicate.
- R/v2data/DISPATCH.json is exclusive; STATUS.json + EVENTS.jsonl are authoritative.
  Last probe 15:34: WAIT_GPU_MEMORY, 11239 MiB free <14336 required. No pilot yet.
- Old parity PID188758 exited WITHOUT a receipt. Dispatcher will recover it into
  NEW v2data/smoke after memory frees, demand exact 449.62744140625/302.890625
  (original v1 group0), then create NEW v2data/pilot and launch automatically.
- Explicit --registered-data-pilot route pins prepared84751dc7/contracte67850f3;
  default smoothing+FD route unchanged. 16 local CPU tests /11 staged tests PASS.
  Eight MAIN15:00 TRAIN selection seeds are excluded from fitting (44 fit/52 gate).
- Full52 panels at0/10/20/30; unchanged pilot.decide(), SD0, finite gradients,
  frozen-bitwise required. PASS continues v1 curriculum within21570s; FAIL stops.
- Remaining: monitor PARITY.json, pilot/PILOT_V2.json; report decision immediately
  to REGISTRATION.md and WMFIX_PILOT_V2DATA.md. On PASS wait FINAL.json, then run
  the staged sharded verdict in WMFIX_LAUNCH_READY.md with train-dir **v2data/pilot**
  and NEW verdict output; retain C1's documented missing-cohort/runner blocker as
  a blocker, never a qualified result. These post-pilot verdict jobs are NOT
  automatically dispatched by the current launcher.
- No peer PIDs signalled, no SUMO, no push. Original smoothed-v2 failure below is
  historical; do NOT invoke v2/launch_pilot_v2.sh for this authorization.

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
- REGISTERED binding R/v2/FEASIBILITY_V2c.json (16bf32cd, exited 15:12): **FAIL** - FD -0.213/
  +3.09/-0.525 (eps 1e-3/1e-4/1e-5) vs analytic -0.0108; one step does not reduce loss.
  Smoothing only covers the source column (~2% of the gradient); road/upstream stay
  hard-event dominated. Launcher therefore did NOT start the pilot (PILOT_V2_NOT_LAUNCHED.json).

## Running PIDs (own; never kill anything else - S-WMCALIB has 12 GPU procs on gpuA)
- 189176 feasibility V2c (exited); 191834 launcher (exited, not launched); 188758 panel_eval_v2 smoke (FIT group 40 = w973002 with v1
  step_000000; parity target from v1 PANEL_step0_r0 group 0: 120 s 449.627 / 300 s 302.891)
  -> R/v2/smoke/SHARD_fit973002_v1step0.json.
- If MAIN approves a pilot later: `bash v2/launch_pilot_v2.sh` pattern (pilot out R/v2/pilot
  must be new/empty; 21570 s cap; pilot_v2 refuses any non-PASS feasibility receipt).

## Next steps
1. Check smoke parity (exact match to v1 values = same eval path) and FEASIBILITY_V2c.
2. NOT_LAUNCHED (current state): MAIN decides; do not retune gates. Repair candidates in WMFIX_V2_FEAS.md
   (capacity-limited surrogate; or v2 data-only with smoothing OFF) need MAIN approval.
3. If launched: poll v2/pilot (update_*.json, SHARD_mini_*.json, PILOT_V2.json). PASS ->
   pilot continues the curriculum itself; then run the staged sharded verdict
   (WMFIX_LAUNCH_READY.md section "MAIN ADDENDUM 12:45") with the FINAL checkpoint + R/v2/pilot/
   candidate_config.json. Record WMFIX_PILOT_V2.md + REGISTRATION.md lines.
4. Each panel subprocess re-loads 236 episodes (~20 min under load): mini panels lag
   training by ~30 min, so early stop is weak; caching the Runtime is a cheap speed-up.
