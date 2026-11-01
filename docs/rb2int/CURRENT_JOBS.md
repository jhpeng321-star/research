# Resume here — INT owned development jobs, sourceb2df7458

Read this and HANDOFF.md before any launch. **FINAL EVALUATION IS RUNNING.**
Do not launch either supervisor again.

## Active final ownership (highest priority)

- gpuA supervisor473053/start889857083/boot a81555d7-176e-4ed8-bc7d-896fc89cc914.
- gpuR supervisor924719/start4214495421/boot d19e6898-efd0-4288-a901-d63b21f11163.
- Root on BOTH hosts: `/root/autodl-tmp/c4/rb2int20261101/final_v5_v1`.
- Control source d0df3d66e80a45e1e03298a7e590429ab7f67adf; model source remainsb2df7458.
  ReleaseV2 SHA6579ece632f1368aa8ea80c25c612297da3d5e57a44a91793e4ea4643d0e01df;
  planSHA185c565df34e4150c90e4b12d55391ab29772c58ab84eff7c0dc3e50ff1caa80.
- gpuA first children473062(v5/969436),473068(LP/969436),473081(v4/969590),
  473094(v5/969590). gpuR924722/924725/924733(WM2/969590,969595,969599).
  Child PIDs change only as the fixed world queue advances; inspect PROCESS/EXIT
  receipts and exact incarnation, never kill by name or redispatch a claim.
- Latest status02: both supervisors alive, no failed worlds; gpuA4future files,
  gpuR still beginning first futures. Future files are not complete K8 worlds.
- Read: `python -B -m tools.rb2int.final_ops status --host gpuA --control /root/autodl-tmp/c4/rb2int20261101/phys_code_d0df3d66e80a --receipt <NEW_LOCAL_JSON>` (gpuR analog).
-216worldjobs/1,728new future tasks,128original baseline reserved tasks reused;
  v5/LP pairs stay on the same host.4gpuA/3gpuR own5GiB workers/nice19/singlethread.
  No new world starts after Nov2 09:00; running own worlds finish, all attrition retained.

## Active read-only collector/delivery continuation

- Local9660/create_filetime134379946028423282, `runs/rb2int_finish_v2`.
  One20min collector, no inference dispatch. First successful poll16:16:43:
  7/216complete K8 worlds, committed receipts. Checks and reuses the same cache.
- V1 local46500 EXITED at a CRLF whitespace-check false positive after collection;
  exact bytes/caches/STOP retained. Scoped Git attribute correction passes; no
  hook/check bypass or forecast replay. Do not restart V1.
- Watch log/PROCESS/START_ONCE/STOP/COMPLETE before any collector action. V2 will
  score/publish/commit only when full216worldjobs exist; incomplete deadline is
  explicit, never relabeled full. Watched analysis/source or progress-report edits
  cause a safe stop; preserve edits and resume explicitly if needed.

Model stage is immutable and remains valid across documentation/scoring commits:

`gpuA:/root/autodl-tmp/c4/rb2int20261101/phys_code_b2df7458a41f`

Archive SHA83cfe685d21c975a7776eb06817e31beaee9d152a32f4ca5f26af647f4128d1c.
Use this existing stage; do not restage/replay merely because Git HEAD advances.

## Completed; never relaunch

- v5 closed60: own446051 EXITED, two development worlds PASS,42.9256s.
  Recipe SHA023fa74a4d3059115ec0a5f28d39030a76137d871db5798eb1d23a13fc1e32c5.
- LP closed60: own446341 EXITED, two development worlds PASS,38.7158s.
  Recipe SHA95802e9bc6d4661cd86dd738db0853cb0ce6055ebcf23194943a5c745f945c15.
- Both preserve exact WM2 encoder/checkpoint, causal motion/LC conditioning,
  finite/conserved cohort and learned-correction guard. Completion snapshots in
  `SMOKE_CLOSED_{V5,LP}_B2DF7458_COMPLETE.json`.
- v5 fivearm600 own446421 EXITED/PASS: dev969420/969421, K1/allfivearms/H600;
  task times220.56/232.72s. `CANARY_V5_B2DF7458_STATUS03.json` saves the complete
  `DEV_CANARY_PASS.json` and identity-bound task hashes. Never rerun this canary.
- LP fivearm600 own449208 EXITED/PASS: same two dev worlds/allfivearms/H600/K1,
 218.12/215.52s. `CANARY_LP_B2DF7458_STATUS01.json` saves its exact pass/hashes.

## No remaining INT smoke process

- Completed LP fivearm600: own449208/start_ticks889033367/boot
  a81555d7-176e-4ed8-bc7d-896fc89cc914. Root:
  `/root/autodl-tmp/c4/rb2int20261101/dev_b2df7458a41f_fivearm600_lp_cuda`.
  Dev969420/969421, K1/allfivearms/H600, strict5GiB CUDA/nice19/oneCPU thread.
  Earlier coarse-load/memory admission failures preceded claim/Popen. Launch
  succeeded only after v5 completed/released memory; exact PROCESS receipt saved.
  It has now completed; do not duplicate it.
- Read: `python -B -m tools.rb2int.smoke_ops status --host gpuA --runtime /root/autodl-tmp/c4/rb2int20261101/phys_code_b2df7458a41f --recipe wm_phys_v5_lp.toml --kind fivearm600 --device cuda`.
- Completion requires `tasks/DEV_CANARY_PASS.json`, not merely a COMPLETE marker
  or process exit. This K1 development canary is never counted as final K8.

## Baseline protocol verification active

- WM2 bridge own gpuR909809/start4213959815/boot
  d19e6898-efd0-4288-a901-d63b21f11163, root
  `/root/autodl-tmp/c4/rb2int20261101/bridge_378d7be5c182_wm2`.
  Fixed dev969420/future0/all5arms/H600, EXITED/PASS150.70s. Verification:
  BRIDGE_WM2_378D7BE5_PASS.json, max absolute clock/speed error0 against saved
  original forecast,15region fields/45event decisions equal. Never replay it.
- Control stage378d7be5 exists on BOTH hosts with archive
  SHA531494803bb989e331e7679092fdd7c354c23718a4698aac21d1b1c69da7ac54.
  Model/teacher source remainsb2df7458. Do not confuse these two roles.
- v4 bridge461214 EXITED/PASS236.71s, max output error0; exact completion/parity
  receipts `BRIDGE_V4_378D7BE5_{STATUS01,PASS}.json`. First admission had stopped
  before claim on raw memory accounting; read-only correction counted only clean
  inactive file cache. No original or successful validation task was repeated.
- Final releaseV2/plan are DEPLOYED/RUNNING as above. See FINAL_EXECUTION_DESIGN.md.
  Do not use the preserved V1 release or launch either root a second time.

## Not launched

- LP fivearm600 initial admissions were rejected on host load/effective memory,
  **before root/claim/Popen**. A prospective launcher correction now uses
  half-physical-host headroom plus measured effective cgroup quota (<80% after
  adding one thread), retaining memory/disk/GPU caps. The one successful LP launch
  followed actual memory release; no forecast or successful claim was repeated.
- Final supervisors are RUNNING. The scoring/collection continuation is still
  being prepared; do not invent a second inference dispatcher. Preserve completed
  exporter904864, transfers and all baseline/kernel/model smokes.

## Exposure and uncertainty

Staged named-artifact audit V2: SHA5b2bf147b63127fc5efafa60776add8309457b9aac2246eaaa77a2fca4d936a8.
All declared active parameter fit/selection worlds are dev-only; checkpoint hash
verified and406 source/artifact files bound. No final outcome opened by this audit.
Earlier audit retained; V2 adds prohibited-path guards without changing artifacts.
Historical aggregate exposure remains disclosed; no untouched-data claim.

Rate-distance CI implementation now conservatively propagates joint signed-bias
bands through abs/subtraction (zero-bias kink); see RATE_CI_IMPLEMENTATION.md.
New audit/CI/launcher checks20PASS; all-four-source integration135PASS.

FULL-v5 and FULL-v5-LP **READY for engineering/teacher-quality rerun**, source/recipe/smoke bound
in FREEZE_ENGINEERING_V5.md and HANDOFF.md. Distributional qualification remains
pending. Both integrated five-arm receipts are now PASS; no final forecast exists.
Recheck worker handoffs for later dev evidence without duplicating their jobs.
