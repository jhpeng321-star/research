STATUS: COMPLETED — strong pure-learned black-box HDV baseline implemented, tested, real-data CPU dry run passed under the 6 GB memory rule, and committed.

# BB_TRAJ — WM-N1 strong black-box trajectory baseline (C1 leave-controller-out)

## What it is
`output/wd01/wm_n1/wms/blackbox_traj.py` — `BlackBoxTrajModel`, a purely learned
next-state model with **no physics base** (no IDM/MOBIL). HDV acceleration and
lane logits come entirely from a neural network (`_LearnedDynamics`) over the
same neighbour kinematics (`hdv.net._KinBlock` attention) and rolling motion
history as WM-S. HDV next states are integrated only through the shared dt/cmap
step, never through IDM. CAVs are teacher-forced from the recorded controller
trajectory (`state["cav_teacher"]`), exactly as WM-S, so the controller's own
motion is known and only the HDV mechanism is learned. This isolates the value of
the physics-structured HDV model on unseen controllers.

Selected by `python -m wms.cli ... --blackbox-traj` (default off); default WM-S
behaviour and `residual_net` outputs are unchanged.

## Params / parity
- BlackBoxTrajModel total params: **772,275** (identical count to WmSModel 772,275).
- Learned dynamics trunk params: 749,151. `physics_step` is None (no pluggable
  physics); `physics_accel()` / `mobil_merge_logits()` raise `NotImplementedError`.
- `physics` is retained only for dt, accel clamp and gap-history geometry.

## Real-data CPU dry run (READ-ONLY TRAIN)
Command (cwd `output/wd01/wm_n1`; OMP/MKL threads 4), guard runner `_out/bb_mem_probe.py`:
```
WM_N1_TRAIN_OUTCOMES=D:/rd-staging/wmn1_feat/labels_cov_20261004T072826Z
WM_N1_TRAIN_IDENTITIES=E:/tmp/wmn1_train_runs_20261002/sidecars3_20261003T0005CN/identities
python -m wms.cli dry-run --pack D:/rd-staging/wmn1_feat/pack_v3 \
  --identities <identity dir> --out .../runs/bb_traj_dry_20261006d \
  --threads 4 --batch-size 4 --unroll-s 10 --blackbox-traj
```
Result `status: passed` (cwd `raw`): 2 optimizer steps, batch 4, 10 s unroll (20
steps), threads 4, num-workers 0; `n_records=2153`, TRAIN, `split=TRAIN`.
- update 1: loss 20.857 (pos 18.870, spd 0.840, lane 1.147); finite=true.
- update 2: loss 61.789 (pos 57.090, spd 3.667, lane 1.032); finite=true.
- both: valid_targets=5981, valid_hdv_slots_t0=313, g(0)=0.0, gate monotone, gate_last=0.1277.
- step time: **~10.8 s fwd+bwd per 20-step window = 540 ms/sim-step** at batch 4 (updates 11.3 s / 9.7 s).
- receipt: `output/wd01/wm_n1/wms/runs/bb_traj_dry_20261006d/TRAIN_RECEIPT.json`.

## Memory (rule: every python process < 6 GB)
Sampled subprocess tree peak on the dry run: **RSS 914.6 MB, USS 838.4 MB,
private commit 1218.8 MB** (in-receipt peak working set 914.9 MB) — well under 6 GB.
`_out/bb_mem_probe.py` hard-aborts on USS > 6000 MB; the run completed un-aborted.
Data path is lazy/read-only: `IdeaACorpus`/`MemmapCorpus` mmap `.npy` packs
(`mmap_mode="r"`), one window is materialised per indexed access
(`from_real_packs` builds a lazy loader; no whole-pack / all-run load).
Important: without intervention the raw process peaked at ~6.2 GB because the
K=192 pairwise-geometry attention-bias MLP (`[B,K,K,ffn]`) is retained across the
20-step truncated-BPTT graph. The black-box blocks are therefore wrapped in
`torch.utils.checkpoint` (recompute in backward), which is numerically identical
(dropout=0, verified by `test_checkpoint_preserves_outputs`) and drops the peak to
< 1 GB. This changes only the black-box model's memory/time profile, not its math
or capacity; WM-S defaults are untouched.

## Tests
`python -m pytest -q output/wd01/wm_n1/wms/tests/test_blackbox_traj.py output/wd01/wm_n1/wms/tests/test_wms.py`
-> **20 passed** (9 black-box + 11 WM-S).
Black-box contract tests: output shapes; k=0 is purely learned (no physics term);
gradients flow to trunk/temporal/gate; CAV slots teacher-forced and HDV slots
learned (rollout + single step); no leakage (kinematics-only, no CAV-flag/action
channel); no physics base reachable; check400-compatible trainer batch; and
checkpoint-preserves-outputs.
`ruff check` on the changed files: all checks passed.

## Files
- `output/wd01/wm_n1/wms/blackbox_traj.py` (new)
- `output/wd01/wm_n1/wms/tests/test_blackbox_traj.py` (new)
- `output/wd01/wm_n1/wms/cli.py` (`--blackbox-traj` flag, default off)
- `output/wd01/wm_n1/wms/__init__.py` (export)
- `_out/bb_mem_probe.py` (memory-bounded runner), `_out/BB_TRAJ.md` (this report)
