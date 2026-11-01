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

## Running; status only

- v5 fivearm600: own446421/start_ticks888936830/boot
  a81555d7-176e-4ed8-bc7d-896fc89cc914. Root:
  `/root/autodl-tmp/c4/rb2int20261101/dev_b2df7458a41f_fivearm600_v5_cuda`.
  Dev969420/969421, K1/allfivearms/H600, strict5GiB CUDA/nice19/oneCPU thread.
  Last saved status02: alive, zero complete tasks; no error log. Subsequent exact
  process audit:183elapsed/183CPU seconds,1.94GB RSS; GPU process1,170MiB. This
  is active computation, not evidence of a hang. Do not duplicate or stop it.
- Read: `python -B -m tools.rb2int.smoke_ops status --host gpuA --runtime /root/autodl-tmp/c4/rb2int20261101/phys_code_b2df7458a41f --recipe wm_phys_v5.toml --kind fivearm600 --device cuda`.
- Completion requires `tasks/DEV_CANARY_PASS.json`, not merely a COMPLETE marker
  or process exit. This K1 development canary is never counted as final K8.

## Not launched

- LP fivearm600 initial admission was rejected on a coarse host-load20 guard,
  **before root/claim/Popen**. Read-only admission snapshot confirms root absent.
  No LP full-canary PID/task exists. A prospective launcher correction now uses
  half-physical-host headroom plus measured effective cgroup quota (<80% after
  adding one thread), retaining memory/disk/GPU caps. Recheck admission and launch
  once; this is no scientific retry. Existing v5 job is unchanged.
- No final panel/collector/dispatcher is running for INT. Final input preparation,
  baseline-source/observation bridge and exclusive final dispatch remain pending.

## Exposure and uncertainty

Staged named-artifact audit V2: SHA5b2bf147b63127fc5efafa60776add8309457b9aac2246eaaa77a2fca4d936a8.
All declared active parameter fit/selection worlds are dev-only; checkpoint hash
verified and406 source/artifact files bound. No final outcome opened by this audit.
Earlier audit retained; V2 adds prohibited-path guards without changing artifacts.
Historical aggregate exposure remains disclosed; no untouched-data claim.

Rate-distance CI implementation now conservatively propagates joint signed-bias
bands through abs/subtraction (zero-bias kink); see RATE_CI_IMPLEMENTATION.md.
New audit/CI/launcher checks20PASS; all-four-source integration135PASS.

FULL-v5/LP are **NOT READY for teacher handoff** until successful integrated
five-arm smoke and immutable release binding. Recheck worker handoffs for later
development evidence without duplicating their owned running work.
