# S-RB-DECIDE handoff

<!-- wmdecide-refresh:start -->
## Latest automated saved-output credibility refresh

- Collected/scored 2026-11-01T00:39:15.179717+00:00; start at `docs/paper_results/wmdecide/refresh_20261101_003716Z/INDEX.md`.
- All scores use validated complete K8 worlds; per-candidate matched WM2
  sections use exactly the candidate's available common worlds.
- Read-only collection; CPU-only analysis; no inference/dispatch/process control.
- Diagnostic only; EVAL/HDV60 adoption gate remains authoritative.
- wm2: DIAGNOSTIC_ONLY; 20 scored worlds.
- full_v1: DIAGNOSTIC_ONLY; 9 scored worlds.
- full_v3: DIAGNOSTIC_ONLY; 10 scored worlds.
- a2: DIAGNOSTIC_ONLY; 12 scored worlds.
- b2: DIAGNOSTIC_ONLY; 11 scored worlds.
- c2: UNAVAILABLE_OR_INVALID; 0 scored worlds.
<!-- wmdecide-refresh:end -->
Branch `c4/rb-decide`, base `f0db3c82`; owner WM-PHYS registration line5683.

## Current integration routing — fresh-session reconciliation ~05:55 +08

- Own collector PID4984 positively matched to its persistent claim/start time;
  remains live,1200s refresh,09:00+08 deadline. No replacement needed.
- Latest evidence is always the marked block above; historical snapshots below
  retain their original counts. Receipt: `docs/paper_results/wmdecide/RESUME_0555.md`.
- INT already has picker/credibility `e8302db3`. FULL-v4/source22cc32f6 belongs
  to INT's existing noon collector; use its local immutable shards/publication.
  DECIDE's live collector covers WM2/FULL-v1/FULL-v3/A2/B2/C2 only.
- For FULL-v4, the generic credibility CLI needs no change: score local shards
  once and match WM2 with identical explicit `--worlds`; reuse completed reports.
- Prior18 picker/credibility +4 snapshot +7 refresh CPU tests remain applicable;
  this resume changes documentation only. No simulation/source changes.

## Collector recovery — 05:21 +08

- First own collector38724 exited after producing a valid05:18 publication;
  stopped before Git staging on Windows GBK decoding of a UTF-8 diff. Its
  claim/STOP/raw outputs are retained; publication committed as `59562ee3`.
- Explicit UTF-8 Git decoding fixes the failure. A replacement uses a NEW claim
  and `--previous` pointing at the completed `REFRESH.json`, preserving/reusing
  all hash-verified finished scores. Seven refresh tests PASS/15.396s including
  UTF-8 decoding and the previous-cache owner binding. See recovery receipt.

## Bounded continuation prepared — 05:14 +08

- `tools/wmdecide/refresh.py` collects/scored saved output JSON only, every20min
  through an explicit09:00 deadline. Per-candidate matched-WM2 sections avoid
  shrinking all comparisons to the smallest variant's panel. Identical complete
  ensembles reuse hash-verified scores; incomplete/new futures are never inferred.
- Persistent exclusive claim, ownPID/create-time receipt, single-thread CPU/load
  guard, terminal STOP/no automatic restart; no inference, process signalling or
  producer/finalizer interaction. Updates one handoff block, commits exact new
  publication and handoff only; stops for foreign worktree/staged/branch changes.
-6 refresh CPU tests PASS, including actual evaluator→matched publication/cache,
  partial-K blocking, exclusive ownership, deadline and preservation guards.
  Launch/first-cycle receipt will identify the single local collector owner.

## Latest publication — 05:05 +08 saved-output refresh

- Start at `docs/paper_results/wmdecide/INTERIM_0505.md`. Portable sections:
  `available_0505/INDEX.md` (WM2 full20; FULL-v1/A2 six worlds; FULL-v3 three)
  and `matched3_0505/INDEX.md` (all four EXACT969432/433/434).
- Physics variants improve absolute-delay CRPS vs WM2 at120/300/600s on the
  common3-world pilot, but all worsen300s all-pair preference Brier. Absolute
  interval coverage remains low; confidence is not qualified by improved means.
- FULL-v3 selected forecast93.1/93.1/98.8% vs observed strict wins5/9,5/9,9/10.
  Cohorts vary by model; these are original-state diagnostic calls, not policy wins.
- B2:16 saved tasks,0 complete K8 worlds; explicit blocked report retained.
  C2: no saved output JSON yet. No inference/dispatch/retry/process action.
- Hash-bound snapshots and raw reports preserved under `E:/Temp/opencode/`;
  exact paths, portable source manifests and analysis limits are in the interim.
  Owner00:44 extension remains in force; newer complete outputs can be published
  to NEW directories with the snapshot→credibility→publish commands.

## Continuation — 05:02 +08

- Resume receipt: `docs/paper_results/wmdecide/RESUME_0502.md`; prior commits,
  matched04:00 evidence and producer/finalizer ownership preserved.
- Finished the previously uncommitted `tools/wmdecide/snapshot.py`: read-only
  SSH collection of explicitly registered evaluator output JSON into NEW local
  directories, with transfer/source hashes and strict producer/path admission.
  No downloaded code execution, remote writes, inference or process control.
- Existing18 decision/credibility tests passed;4 additional snapshot tests pass,
  including generated inventory syntax, host binding, exact bytes, malformed
  member/path/hash rejection before output and immutable destination admission.
  Ruff PASS. Next publication will use newly collected complete-K8 outputs.

## Latest publication — matched 04:00 snapshot, scored ~05:00 +08

- **Credibility sections now contain real complete-K8 evidence**, for WM2 and
  WM-PHYS-FULL-v1 on identical worlds969432/433/434/436. Start at
  `docs/paper_results/wmdecide/INTERIM_0400.md`; portable per-model sections/three
  SVGs are in `docs/paper_results/wmdecide/matched_0400/`. This supersedes the old
  blocked progress receipt, without replacing or rerunning any inference.
- FULL-v1 delay CRPS improves at120/300/600s, but still loses absolute-delay cell
  climatology and range coverage is10/40/25%. Selected0.8-rule strict wins are
  **4/13,6/12,10/11** vs mean forecast94.2/96.9/95.5%; all-pair preference Brier
  worsens at300s. These are four-world diagnostics, not qualification/policy wins.
- Matched cells are singleton strata: credibility CIs explicitly unavailable.
  No finite-K or same-state calibration claim; HDV60 adoption gate still required.
- Code improvement `e8302db3`: strict `--worlds ID ...`, selected-switch Brier/
  reliability, visible coverage and picker switch-rate/expected/recorded savings.
  New `python -B -m tools.wmdecide.publish --reports REPORT... --matched --out NEW`
  exports hash-verified portable sections/figures and pooled receipts without scoring.
- **18 CPU tests PASS /4.582s** including actual EVAL-validator→report→publication,
  reverse selected directions/ties, label-blind selection, abstention, exact world
  admission, immutable exports, tamper rejection and singleton-CI guard.
- For later A2/B2/C2/FULL-v3: run the credibility CLI separately per immutable
  identity; use the same explicit scored worlds on WM2 for matched comparisons.
  Preserve all raw audit/FILES receipts and publish to a NEW version directory.

## Fresh-session continuation — 2026-11-01 04:44 +08

- Preserved clean HEAD `cbf042f7` and all three implementation/verification commits.
  INT already integrated `16dff9ad` and independently passed the 15 CPU tests.
- New evidence is available: EVAL's latest fetched WM2 shards now contain complete
  K8 worlds; INT retains immutable local `runs/rbint_evidence/{wm2_0400,full_v1_0400}`
  snapshots and a four-world paired comparison. Previous zero-complete-world receipt
  below is historical, not the current evidence status.
- Improved CLI adds `--worlds ID ...` for strict matched-world scoring and a third
  reliability diagram / Brier comparison for the actual selected 0.8-confidence
  switches, including reverse directions and ties. No-switch cases stay in switch
  rate / saving summaries, not calibration. `CREDIBILITY.md` now exposes coverage.
  **18 CPU tests PASS /5.889s**; next score the immutable local snapshots.
  Existing EVAL/INT/RUNA/RUNB inference and finalizers retain ownership.
- Owner 00:44 extension applies: improve until ~09:00, integrator freeze ~09:30.
  Local 04:43 resource check: 22 logical CPUs, 0% load; NVIDIA driver unavailable.
  Analysis will be CPU-only with one BLAS/OMP thread.

## READY FOR INTEGRATION

Implementation commits: `df4fa61e` (paired picker), `16dff9ad` (credibility).
Copy/merge `tools/wmdecide/` into the integrated runtime and use the CLI below.

- `tools/wmdecide/decision.py`: `Future`, `summarize`, `decide`; NumPy/CPU only.
- Shared draw IDs align all candidate actions; missing/duplicate/failed futures
  reject. Full-team expected delay, predictive spread, event probability, paired
  strict-win probability and saving distribution are returned as JSON-safe dicts.
- Fixed owner confidence `SWITCH_CONFIDENCE=0.8`; choose lowest-mean eligible
  candidate only if it also improves expected delay; otherwise keep incumbent.
- No shared model-file changes or default switches. S-C1SWITCH2 can import API.
- Credibility CLI READY: `python -B -m tools.wmdecide credibility --eval-root
  E:/agent-work/c4-rb-eval --inputs SHARD0 SHARD1 SHARD2 SHARD3 --out NEW_REPORT`.
  Run separately for WM2 and each WM-PHYS variant; output parent must exist.
  `--allow-partial` scores only complete K8 worlds, explicitly diagnostic.
- Integrator: include each output's `CREDIBILITY.md` in the morning report and
  retain adjacent `reliability_breakdown.svg`/`reliability_preference.svg` links.
  `summary.csv`/`credibility.json` carry action/cell/horizon/split breakdowns;
  `FILES.json` and `audit.json` bind inputs, source code and outputs.
- Metrics: empirical/fair CRPS for absolute and paired team delay vs same-arm/
  same-pair pooled+cell climatology; breakdown and pair-preference Brier/reliability;
  50/80/90% central and range coverage. Strict paired ties count as non-wins.
- EVAL integration reuses `read_predictions` and `real_observables.json`, preserving
  validation, K8 identity, BIC, ledger and development-only world-excluded baselines.
- CIs: 10000 pointwise stratified whole-world bootstrap resamples, seed20261031;
  saved futures fixed, no finite-K/model-error claim or simultaneous qualification.
  Existing EVAL adoption/non-regression gate remains authoritative.
- Verification: combined **15 CPU tests PASS, 0 skips (10.68s)**, independent
  CRPS/CDF/rank checks, leakage/missing-climate/tie guards, real-validator→report,
  SVG XML/coordinates, output hashes, public picker CLI and overwrite rejection.
  Ruff and `git diff --check` PASS. See `tools/wmdecide/VERIFICATION.md`.
- Fetched WM2 evidence checked with actual CLI: 11 tasks validated, **0 complete
  K8 worlds**, so **BLOCKED_NO_COMPLETE_K8_WORLD**, no partial-ensemble scores.
  Receipt: `E:/Temp/opencode/wmdecide-wm2-progress-58cb40f9745e47e1ac1eb43919a495f5/`.
- Scientific credibility scores await completed saved K8 worlds and local candidate
  shard paths. Existing EVAL/RUNA/RUNB producers retain their inference ownership.
  S-RB-INT should run the same command once per model when outputs land, then
  include each `CREDIBILITY.md` and its figures in the morning report.
- No simulation/training/remote dispatch or peer-process interaction performed.
