# S-COOPNATIVE — historical DESIGN READY; superseded before training

> Recovery 2026-11-04: current registry still records the later MAIN supersession;
> native-start requires explicit reactivation, fresh seeds and budget reconciliation.
> [Current recovery receipt](coopnative/RESUME_20261104_NATIVE_RECHECK.md). No new simulation launched.

> Recovery 2026-11-02: this heading and quote describe the historical native-start
> build. MAIN superseded it at 2026-11-01 16:22 with S-COOPMWM; the proposed TRAIN
> seeds have since been consumed by that successor. Native-start launch is blocked
> pending explicit reactivation, fresh seeds and budget reconciliation. See
> [the recovery receipt](coopnative/RESUME_20261102.md). No new simulation was launched.

Build commit `8368ce1a`, branch `c4/coopnative`, parent `7f2a96c3`.
**New SUMO: 0 sim-s. No training/evaluation/timing simulation or dispatcher started.**
Design: [COOPNATIVE_DESIGN.md](COOPNATIVE_DESIGN.md), 61 lines, SHA256
`1866111aa1b7e5f6b9b521e913c3f002b054db86949e9eb0c5633f59985ab989`.

## Implemented and verified
- Fresh native actor/observable central critic; PPO, 4D RSU broadcast, exact correlated
  shared-component exploration; 900s task and 400s effective GAE credit horizon.
- Every road CAV, ramp included: held speed/time-gap targets, W10s; exact-zero output
  releases to native for the whole window. No MAPPO actor/weights/fallback in learning.
- Reused COOPFREE tracking, original safety and CommandOwnership; default COOPFREE
  zone behavior retained, optional road-wide scope for this learner.
- Comparator-only MAPPO and genuine frozen CTRL0 adapters; full team+source J,
  paired whole-world bootstrap, unchanged COOPBEHAV detectors and evaluation-only C_G twins.
- Exclusive immutable charges/checkpoints, complete optimizer/RNG resume, approval gate
  before worker physics; failures/outstanding charges stop, no automatic retry/refund.
- Final CPU **15/15 PASS, 0 skips, 1.270s, SUMO0** on existing cpuQ environment,
  with `subprocess.Popen` prohibited: zero/native CommandOwnership parity, road-wide
  hold/expiry and real safety, exact comparator-J accounting, leakage/local
  noninterference, joint likelihood/gradient, delayed credit, exact Adam/RNG resume,
  dual seed guards, fake-plant orchestration, paired statistics and detector ABI.
- Receipt: `coopnative/CPU_coopnative20261101_cpu5.json`, binding all staged sources.
  Earlier CPU2 failed a mutable fake-observation fixture; corrected fixture and all
  historical receipts/stages retained. No real simulator ran during any build.
- Fresh TRAIN986000–986071 + independent TRAIN-cell checks986072–986083: registry
  range/identity scan and safe-name exposure audits PASS on cpuQ/gpuA/gpuR, zero hits;
  both frozen guards pass all108 admitted TRAIN/check/dev worlds. Re-audit before dispatch.
- All24 S-DISTILL dev MAPPO episode RESULT bytes verified against original receipt hashes
  on gpuA; exact984000–984023 panel and paired safety metrics bound in `coopnative/MAPPO_DEV.json`.

## Comparator training provenance and equal-budget rule
`coopnative/COMPARATOR.json` preserves production2 RUN_SUMMARY and one existing CPU episode.
Verified actor SHA d9042cc5; summary SHA8b25ee1e. Iter169 =169updates ×46episodes ×600s:
**4,664,400 controlled env-steps/sim-s; 6,996,600 total sim-s including300s warmup**;
7,774episodes/zero failures, gamma.99/lambda.95, four PPO epochs.
Training reward was per-physics `team_reward`, summed per control tick, weights
delay1/throughput.5/smooth.05/hard-brake1/fairness1; alpha0 at169. Scoring uses road+source J.
The comparator inherited an earlier actor at source iteration170 (SHAaecd7d70);
that earlier training is additional, not silently included in the reported169-update budget.
Our proposed **entire-run cap is4,664,400 total sim-s**, stricter than its total budget.
Training alone:4,147,200 total/3,110,400 controlled sim-s,144fresh on-policy updates.
Publish learning curves vs cumulative TRAIN sim-s AND all charged sim-s; dev never tunes the actor.

## Budget request — total cap4,664,400 sim-s, not yet approved
- TRAIN:4,147,200 =144×24×1200, all12cells/update, no exact twins in gradients.
- Independent TRAIN-cell checks:288,000 =18×12×1200 learned +24×1200 native/MAPPO controls.
- DEV:115,200 =2×24×1200 learned (u8/final) +48×1200 native/genuine CTRL0; reused MAPPO costs0.
- Exact detected-behaviour C_G twins:96,000 maximum; complete groups, no positive-only selection.
- Zero-update physical native canary:1,200; its native control is reused from the fixed dev arm.
- Unallocated:16,800. No retries/refunds or automatic transfer of unused phase reservations.

## CPU throughput and ETA quote — estimates, no new SUMO timing
Historical ONE CPU episode:900sim-s/15.895786wall-s =203,828sim-s/hour/worker on cpuQ.
New CPU-only sparse synthetic episode overhead:cpuQ.286s/gpuA.327s/gpuR.518s;
PPO24×90frames/four epochs extrapolates5.49/8.72/14.55s respectively (not traffic measurements).
Using the historical one-episode rate, CPU scaling ratios and >2× shared-load/size allowance:
- cpuQ: **90,000sim-s/hour/worker ×2 =180,000/hour** initially.
- gpuA: **70,000/hour ×1CPU worker**, optional only after WMPLAN/C1 priority/headroom release.
- gpuR: **50,000/hour ×1CPU worker**, same optional qualification; no GPU allocation.
Optional combined capacity300,000/hour; initial production runner supports cpuQ only.
Cross-host dispatch requires a separately frozen CPU/parity binding before using optional capacity.
All workers nice19/single-thread; WMPLAN first, C1SWITCH2 then COOPFREE; resource gates retained.
First u8 check costs332,400 total sim-s including controls/canary: **2–3h after admission**.
First full24-world dev verdict adds28,800: **about2–3h after admission**; no early victory claim.
Final dev+discovery: **28–36h cpuQ-only**, or **16–24h** with qualified optional helpers.
These are active-allocation estimates; time yielded to higher-priority work adds to wall ETA.

## Approval/resume handoff
Waiting for MAIN to append the exact approval phrase in the shared REGISTRATION.md.
Its16:05 entry records an owner-approved6M ceiling; the exact required gate phrase is absent.
Before any launch: fresh source-bound exposure/CPU receipts, immutable binding of actual
approval line/cap/runtime/CTRL0/cache, registered dispatch identity; physical native canary first.
No BINDING/APPROVAL/DISPATCH or training supervisor exists for this task yet.
After approval, publish every8-update learning check here; keep learning/discovery evidence separate.
