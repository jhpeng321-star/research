# S-C4LAUNCH — MAIN 2026-10-28 15:00

## Step 1: prospective authority recorded

Appended MAIN15:00 decisions to the authoritative
`E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md` before GPU work.
The same decisions are committed in `tools/c4prep/evidence/MAIN_1500.json`.
gpuA `arm/loop/c4prep/MAIN_1500.json` SHA256
`2b0c43a7f8157e8402df98932359582e6138d054fefc965b566f5ddcfb83598e`
and append-only `LOOP14_STATE.json.main_launch_amendments` now bind the launch.
Peer WM2 boundary, stop and budget fields were preserved. No GPU job started.

The anchor attempt uses C0/C1 fitting and C2 world-disjoint validation only,
active horizons10/60/90, budget3600s including a measured pilot-rate projection.
Anchor-OFF with q10/K8 ON is the authorized fallback, not a fitted-scale claim.
WM3 uses MAIN's inherited WM2 trusted90s authority, not an invented WM3 PASS.

## Step 2: bounded replay selected the authorized fallback

Runtime5a86c667 launched own coordinator350365/start855123141. Imported C0–C2
preflight passed; genuine replay completed7 K8 rows including90s in319.31s.
Weighted completion projection10217.86s exceeds3600s; it stopped normally,
exit0, no signal. AnchorOFF, fit-free q10/K8ON; no fitted-scale claim.
Full immutable receipt `arm/loop/c4prep/launch_fit_v1/FIT_DECISION.json`;
compact local `tools/c4prep/evidence/LAUNCH_FIT.json`.

Launch resolution implements explicit inherited-horizon authority with original
WM2 gate hash retained; it does not generate WM3 verdicts. Exact60% four-cell
sampling supersedes the inherited rejection tilt, only for training draws.
MAIN policy uses registered full weights8.9/9.9/8.9/10.3/all1 from the first
update, guarded real individual/coop8.9/8.9. FD/free-road probes are alsoOFF.
Candidate-A isolated locking permits the separately owned WM gate to finish.

## Real dry-run attempt1 retained, failed before a completed update

Source89f0119f, coordinator353727/start855201015, worker353733 disappeared
without RUN_SUMMARY or updates.jsonl. Only initial native-best and long-credit
receipts exist. Last observed RSS84,660,244,480bytes approached the container's
96,636,764,160byte limit; sampled GPU peak5,026,873,344bytes remained below0.45.
The cgroup reports0 OOM-kills, so the exact termination cause is unconfirmed.
No signal sent, no policy phase launched. Original runtime/output retained.

Operational memory refinement preserves microbatch1/segment8/both host
offloads and all objectives/draws: repeated persistent WM/actor parameter
storage snapshots now share one host copy per tensor version/view geometry.
Transient activations keep original spill semantics; source storage remains
owned by the trainer, preventing pointer reuse. A versioned cache cannot reuse
snapshots after in-place mutations. CUDA gradient parity and existing exact
LONGMEM tests are required before a fresh immutable two-update attempt.
