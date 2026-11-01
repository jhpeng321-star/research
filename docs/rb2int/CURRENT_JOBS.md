# Resume here — INT owned development jobs, sourceb2df7458

Read this and HANDOFF.md before any launch. Final evaluation has **not** started.
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
- v4 bridge first admission stopped before root/claim/Popen on a misleading raw
  cgroup-free-memory check. Read-only correction accounts for clean inactive file
  cache; see MEMORY_ACCOUNTING.md. No v4 bridge PID yet; next launch is one-use.

## Not launched

- LP fivearm600 initial admissions were rejected on host load/effective memory,
  **before root/claim/Popen**. A prospective launcher correction now uses
  half-physical-host headroom plus measured effective cgroup quota (<80% after
  adding one thread), retaining memory/disk/GPU caps. The one successful LP launch
  followed actual memory release; no forecast or successful claim was repeated.
- No final panel/collector/dispatcher is running for INT. Final causal inputs are
  prepared, transferred and exact-prefix verified; kernel bridges and exclusive
  final dispatch remain pending. Preserve the completed export904864 and transfers.

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
