# S-SUMOES-RUN — final result, 2026-10-31

**Stopped under the registered four-consecutive-rejection rule after updates 0–3.
No accepted update and no learned improvement. The final 137-scalar head is exactly
zero, hence CTRL0-equivalent. Charged SUMO time = actual time = 47,700 sim-s of
the 650,000 sim-s cap; 602,300 sim-s remain unspent.**

Worktree `E:/agent-work/c4-sumoes`, branch `c4/sumoes`. Owner authority:
REGISTRATION 2026-10-31 14:25 variant (b), MAIN14:40 amendment, fresh-session resume.
The prior work was retained. No jobs, baseline, canary, or update were duplicated.

## Learning curve: real-SUMO panel J600

Metric: mean E3 team delay, `cumulative_600.team_delay_vs`, veh-s, **lower is better**.
The fixed eight-world TRAIN acceptance panel is 969536–969543. Retained-head curve:

- **Baseline:** 46,958.76242850634; zero head, eight panel worlds. Canary on969536:
  CTRL0 and ours-zero both 9,555.490908879332, identical initial-state SHA.
  Nine episodes total including the extra CTRL0 canary; 8,100 sim-s.
- **Update 0, historical Adam/base-relative implementation:** proposed panel mean
  46,958.762615873544, delta **+0.0001873672**; rejected and bitwise rolled back.
  Retained mean **46,958.76242850634**. Three real-SUMO antithetic fitness differences
  `Jplus-Jminus`, averaged over six worlds: +1,246.0138751665684,
  +4,748.034020225207, +3,182.2239642540594 veh-s. Increment39,600 sim-s;
  cumulative47,700. Earlier receipt: screen~75s, SUMO~161s.
- **Update 1, MAIN14:40:** all eight directions rejected by WM screen; no candidate
  or SUMO dispatch. Retained mean **46,958.76242850634** (cached baseline value).
  Increment0, cumulative47,700 sim-s. Screen102.881977s; driver0.005689s.
- **Update 2, MAIN14:40:** all eight directions rejected by WM screen; no candidate
  or SUMO dispatch. Retained mean **46,958.76242850634** (cached).
  Increment0, cumulative47,700 sim-s. Screen102.079500s; driver0.004121s.
- **Update 3, MAIN14:40:** all eight directions rejected by WM screen; no candidate
  or SUMO dispatch. Retained mean **46,958.76242850634** (cached).
  Increment0, cumulative47,700 sim-s. Screen101.419477s; driver0.004179s.

Thus the retained-head learning curve is flat. Updates1–3 are **screen-only no-step
outcomes**, not new panel measurements or real-SUMO candidate rejections. There were
no real-SUMO fitness differences to report for these three updates.

Machine-readable curve, episode hashes, screen decisions and reconciliation:
[`FINAL_RECONCILIATION.json`](../../tools/sumoes/evidence/FINAL_RECONCILIATION.json).
Per-update receipts: [`RUN.md`](../../tools/sumoes/RUN.md).

## MAIN14:40 implementation and audit correction

Committed before the resumed screens as `beb08d64`; CPU tests32PASS on cpuQ's
existing environment and RuffPASS. Source files were SHA-matched between local and
cpuQ, with bundle SHA
`ca4b69610e95dfae751c828d218235cc330e79932f5a422800df83708b228c2e`.

**A — normalized ES step.** For the WM-kept directions, average real-SUMO paired
differences over six gradient worlds, then propose
`theta + sigma * mean(-sign(Jplus-Jminus) * d/||d||)`.
Zero differences cast zero votes; the mean is not renormalized. Historical Adam
fields remain readable but no longer take steps. Eight-world acceptance and exact
rollback are unchanged. Sigma remained0.07621979740797888 throughout; no accepted
steps or rejection-based sigma halving occurred before the early stop.

**B — matched cap-relative law and E3 C1 trunk.** Code inspection corrected an
important historical description: update0 SUMO's helper was named
`caprel_commands`, but its formula added residuals to the **uncapped native base**,
as did the old WM runtime. The earlier claim that SUMO already used true
cap-relative composition was incorrect. That update is preserved as historical.

The resumed paths share `tools/sumoes/caprel.py`: compute executed CTRL0 from
native base and mode27 ceiling, add the residual relative to CTRL0, apply the
registered bounds and final ceiling. SUMO uses live API geometry/cap queries;
WM uses the frozen inferred geometry/cap. Zero or no-effect residuals preserve
the original native-base SUMO write exactly. Both paths use C1 trunk SHA64f70af4;
the543-file E3 source, CTRL0 wrapper and observer remain frozen. The corrected-law
nonzero-residual SUMO path was CPU-tested but **never dispatched**, because all
resumed screen directions failed. A new real-SUMO zero canary was not repeated.

**C — braking-side confident gain only.** On the eight existing TRAIN starts
969401/402/404/405/407/408/410/411, each screen used WM2+learned mix,120physics steps
(60s), paired common futures, theta and both perturbation sides. Braking orientation
was determined by larger total negative-residual mass, without using fitness to
choose a side. Gain = `J60(theta)-J60(braking side)`; qualify only if median>0 AND
q10>0; rank by median, keep at most3. Each screen ran136 WM rollouts on24nice19 CPU
processes. All24 direction medians and all24 q10 values were negative.

- Update1 median gain range: **−26.692400 to −3.960595 veh-s**; kept0/8.
- Update2 median gain range: **−24.396633 to −0.065428 veh-s**; kept0/8.
- Update3 median gain range: **−26.034745 to −4.115202 veh-s**; kept0/8.

Exact values and all per-start costs/braking masses are preserved in
`tools/sumoes/evidence/screen_u01.json` through `screen_u03.json`, bound to their
request SHA. Legacy `gains90_descriptive`/J90 fields sum the available60s bins;
they are not independent90s measurements and are not used for selection or claims.

## Stop, budget and mechanical checks

- Historical update0 was a panel rejection; updates1–3 were screen rejections.
  The pre-resumption implementation receipt explicitly counted screen no-step
  outcomes as rejected opportunities for the four-consecutive rule, includingu0.
- Afteru3, run.py emitted `scientific_stop` with reason
  `four_consecutive_rejected_updates_including_screen_no_step` before update4.
  There is no update4 request, screen or SUMO job. Updates4–15 were not attempted.
- The separate >2/3 rejections over updates0–9 sigma rule was never reached.
- Durable accounting reconciles **53 unique precharged jobs,53 completed results**:
  9baseline/canary +44update0 episodes, each900s. Actual=charged=47,700; no refunds.
- All53 results: **0collisions,0teleports,0dropped vehicle steps,0controller lane calls**.
  Fourteen distinct SUMO TRAIN worlds; `initial_state_sha256` agrees for every
  repeated seed, within perturbation pairs and against the stored state.
- No live task Python jobs at final reconciliation; no process signals sent.
  No evaluation-seed jobs, push, deletion or new venv.

Tool-scope disclosure: the initial content-search call was addressed to the named
REGISTRATION.md file, but the tool also returned matches from its parent run tree,
including an unrelated prior-world timestamp line and file names. No returned
outcomes were used and no additional seeds were allocated from those matches.
Subsequent registration access used exact-path Read and patch operations. This
unexpected search expansion is recorded rather than claiming a perfectly scoped
search history.

## Final head for S-EVAL-A

Portable artifact: [`tools/sumoes/evidence/FINAL_HEAD.json`](../../tools/sumoes/evidence/FINAL_HEAD.json).
On cpuQ: `~/sumoes20261031/FINAL_HEAD.json`.

- **Head SHA256:** `226f2fad46e9e695b1fc277750a1bfdf5c403f17b38984efde3a16773e5fc255`.
  Definition: SHA256 of `json.dumps([float(v) for v in theta])` encoded as UTF-8,
  exactly the existing `theta_sha` protocol. All137values are0.0.
- Export-file SHA256 (runtime LF bytes):
  `99982b566a7214e6a70e732052e4b9e70acb7523abddc24cf792e6756269502a`.
- Frozen E3 C1 trunk SHA256:
  `64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c`.
- Final state SHA256:
  `6adafbfd8b286f27966127d65688d4e05f45344ae347de95c0e07b74f3c5c64e`.
- Accounting ledger SHA256:
  `5cc54a332ad9778a68b04126f414fe17583f0c1a7fc98918df8b594a7a09fe64`.

Set only `residual_head.weight`(136) and `residual_head.bias`(1) on that frozen
trunk using the existing head loader and the registered wrapper. This artifact
is the search's actual final head, not a selected nonzero candidate. It reproduces
CTRL0, so the run does **not** support a claim of learned improvement over CTRL0,
MAPPO or native. S-EVAL-A was not dispatched by this task.

## Commit/evidence trail

- Prior work: 5888da6a allocation,4df1f496 evaluator,a3ac4762 canary/baseline,
  b6fa4676 update0/handoff.
- Resumption: beb08d64 amendment implementation; f912e85e update1;
  f21ed909 update2;8149db98 update3/stop. Each update was entered in both RUN.md
  and the shared REGISTRATION before proceeding.
- Original local evidence snapshots were retained. Final evidence includes
  `u01_state.json`/`u02_state.json`/`u03_state.json`, `MAIN1440_RUN_LOG.jsonl`,
  `MAIN1440_IMPLEMENTATION.json`, `FINAL_HEAD.json`, `FINAL_RECONCILIATION.json`.
- Reconciliation/export utility: `tools/sumoes/remote/finalize.py`, no SUMO calls,
  exclusive final artifact writes. Reported statistics come from these receipts.
