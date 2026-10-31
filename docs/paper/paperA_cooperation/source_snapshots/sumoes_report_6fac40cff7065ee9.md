# S-SUMOES-RUN — latest Track 3B result, 2026-10-31

**COMPLETE: Track3B stopped scientifically after four consecutive rejected REAL
updates (u0–u3), with zero accepted updates and no demonstrated improvement over
MAPPO. The final137-scalar head is exactly zero/MAPPO-equivalent.**

Combined charge **171,900 = actual171,900sim-s**,191completed/precharged episodes;
**478,100sim-s unspent** under the650,000cap. All safety/lane counters are0 and all
repeated-world initial-state hashes agree. No task processes remained at final
reconciliation. Updates4–15 and S-EVAL-A were not dispatched.

## Authorized continuation and method

MAIN15:54 explicitly lifted the prior logging-failure stop under the owner's
Track3B approval. Fresh-session status/log/receipts were read first; the repaired
scientific driver SHA matched
`8946bbe5d5982b9905f243c7ded98089d708447ffc840e3c8227a2428490097a`.
Own STOP and prior remote final-head bytes were retained as `RETAINED_PRE1554_*`.
The passing canary was not repeated. Baseline and every completed update were
logged in RUN.md and REGISTRATION.md and committed before the next update.

- Frozen E3 runtime manifest `b3c031ad`, original MAPPO wrapper/native lanes,
  MAPPO iter169 `d9042cc5`, C1 feature trunk `64f70af4`;137-scalar head only.
- Shared SUMO/WM law: bounded MAPPO safety-filtered action plus C1 residual,
  followed by unchanged MAPPO speedMode27/safe-speed execution limits.
- Each update:8directions ranked by WM2+mix60s paired rollouts, top2 always sent
  to real SUMO,6fresh gradient worlds, both antithetic sides;24probe episodes.
  Normalized sign-weighted mean-unit-direction step;8candidate-panel episodes.
  Strictly improved panel mean required, otherwise exact rollback.
- CPU checks were already recorded before this continuation:32original checks,
  9Track3B checks including the exact logging regression, and a4tick WM zero-head
  bitwise bridge check. Scientific source was unchanged during this continuation.
  New operational launch/checkpoint helpers passed Ruff and their real receipt
  checks; they do not change scientific selection or fitness.

## Track3B learning curve — measured proposals and retained head

Metric: E3 `cumulative_600.team_delay_vs`, team delay in veh-s, **lower is better**;
300s native warmup +600s control. Fixed TRAIN acceptance panel969536–969543,
8worlds. This is an adaptive training panel, not a final-evaluation estimate.

- **MAPPO/zero-head baseline:** mean **46,495.41967697356**;8episodes,7,200sim-s,
  cumulative56,700. Wall41.157s. Baseline measured once after the saved canary.
- **u0:** proposed **48,097.82292018744**, delta+1,602.4032432138847 (**+3.446%**),
  REJECTED; retained46,495.41967697356. WM kept[3,5]; real mean
  Jplus−Jminus=[−1516.8667999108632,+486.1790572007497]. Cumulative85,500sim-s;
  WM137.474s +driver1236.274s, including the original CPU load-gate wait.
- **u1:** proposed **47,089.45982430789**, delta+594.04014733433 (**+1.278%**),
  REJECTED; retained46,495.41967697356. WM kept[0,1]; real mean
  Jplus−Jminus=[−1800.0451346238453,−337.0464619398114]. Cumulative114,300sim-s;
  WM164.461s +driver251.502s.
- **u2:** proposed **46,972.46190802667**, delta+477.04223105311394 (**+1.026%**),
  REJECTED; retained46,495.41967697356. WM kept[2,3]; real mean
  Jplus−Jminus=[+52.56731164455414,+1312.9760504563649]. Cumulative143,100sim-s;
  WM140.226s +driver150.884s.
- **u3:** proposed **47,737.15188534471**, delta+1,241.7322083711551 (**+2.671%**),
  REJECTED; retained46,495.41967697356. WM kept[4,5]; real mean
  Jplus−Jminus=[−2128.826236526173,+335.5787608623502]. Cumulative171,900sim-s;
  WM95.945s +driver90.446s. Four-real-rejection stop confirmed by the driver
  before any u4 request, screen or SUMO allocation.

Sigma stayed **0.07621993672711026** throughout. Proposed step norms were
0.055460042574/0.056930753851/0.052872797314/0.048260164083. Neither the accepted-step
nor update9 rejection-based sigma halving was reached. The retained-head curve is
flat at46,495.41967697356: rollback values are cached measurements of the identical
zero head, not extra panel reruns. Some individual worlds improved, but none of
the four proposed means improved. This limited search does not establish that
every direction in the residual family is non-improving.

Exact per-world probe/panel J600, acting shares and positive/negative/zero residual
command counts are recorded per update in [`RUN.md`](../../tools/sumoes/RUN.md)
and `tools/sumoes/evidence/MAIN1554_u00.json` through `MAIN1554_u03.json`.
Sign counts are bounded composed-command minus MAPPO action **before the physics
cap**; they are not measurements of counterfactual physical acceleration.

## Final accounting and mechanical reconciliation

- Historical CTRL0 experiment47,700 +Track3B canary1,800 +MAPPO baseline7,200
  +4×28,800updates = **171,900sim-s**. This continuation added122,400sim-s.
- **191unique charges =191completed results**, each900sim-s;32distinct SUMO
  TRAIN worlds. Every completed result passed collision/teleport/dropped/lane
  checks and every repeated-world initial_state_sha256 matched, including pairs.
- No simulation failed or was retried during continuation. Intermittent SSH
  monitoring failures did not restart detached jobs. The load gate and peer
  workloads were preserved; no process signals were sent.
- Final read-only verification confirms137zeros, exact rollback,4rejections,
  2real directions/update, budget increments, unchanged repaired driver/original
  design hashes and absent u4 request. One initial post-run verification assumed
  an amendment JSON at the remote root; that path was absent and the check was
  corrected before writing the verification receipt. It caused no dispatch.
- Final state SHA:
  `7db8798b87000695aafc42d440d2e479c6a18afb0e55ec1f67aabdc0fe1315e8`.
- Combined ledger SHA:
  `0b0f00a5a39ccaf57992f7031bcf6dab292c5ea53c3c17707e4edbf31aff0735`.
- Evidence: [`MAIN1554_final.json`](../../tools/sumoes/evidence/MAIN1554_final.json),
  [`MAIN1554_FINAL_VERIFICATION.json`](../../tools/sumoes/evidence/MAIN1554_FINAL_VERIFICATION.json),
  `3B_FINAL_MAIN1554_STATE.json`, `accounting_3B_MAIN1554.jsonl`,
  `RUN_LOG_3B_MAIN1554.jsonl`. Historical stopped-result artifacts remain intact.

## Final137-scalar head / S-EVAL-A handoff

Latest artifact:
[`FINAL_HEAD_3B_MAIN1554.json`](../../tools/sumoes/evidence/FINAL_HEAD_3B_MAIN1554.json).
Remote export: `~/sumoes20261031/FINAL_HEAD_3B.json`.

- **Canonical head SHA256:**
  `226f2fad46e9e695b1fc277750a1bfdf5c403f17b38984efde3a16773e5fc255`.
  Protocol: SHA256 of UTF-8 `json.dumps([float(v) for v in theta])`.
- Export-file SHA256 (remote LF bytes):
  `9ae3a09f8ad56f039f0f940a4336044518e0c455bf8228ef8891fed649c29245`.
- Required MAPPO base SHA:
  `d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca`.
- Required C1 feature-trunk SHA:
  `64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c`.

Load only the136head weights and1bias with the Track3B `mappo_residual` wrapper.
The zero-scalar SHA also appears in the historical CTRL0 experiment; the base
and wrapper determine the controller semantics. **This final artifact is
MAPPO-equivalent and not an eligible learned improvement candidate** (explicit
in the final verification receipt). No new learned MAPPO gain or cooperation
claim is supported, and S-EVAL-A was not launched.

Continuation commits: `a8fec3c0` reconciliation, `dc537ba6` baseline,
`5da8eb72` u0, `b765f916` u1, `d88757c2` u2, `de9a5e0b` u3/scientific stop.
No push. Scope/tool-search disclosure remains recorded in RUN.md and the
historical notes below; no outcomes from the unexpected parent-tree matches
were used for allocation, selection or inference.

## Historical post-canary operational stop (superseded by MAIN15:54)

### Track 3B implementation and canary

Authority: MAIN15:45 owner-approved switch, REGISTRATION.md; implementation
commit `a49d980f`, canary/stop `f04bb907`, post-stop logging repair `eaabecff`.
Prospective amendment config `tools/sumoes/design_3b.json` SHA
`9851cc62086086f8fa1796f5233fb4321425a63b385717d599126a722c740ead`.

- Base: speed-only MAPPO iter169, native lanes, export SHA
  `d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca`.
  Original byte-identical MAPPO E3 wrapper/CLI, speedMode27 and safety executor.
- Residual: unchanged137-scalar C1 head/features (`64f70af4` trunk), initially0;
  compose MAPPO safety-filtered action plus residual within MAPPO stored limits,
  followed by the original MAPPO physics cap. SUMO and WM share this action law.
- WM implementation ranks8 directions by their better-side median predicted60s
  gain and always passes top2; no screen veto. Normalized sign/unit-direction
  sigma step and strict8-world mean-panel improvement/bitwise rollback implemented.
- CPU verification:32 original regressions PASS;9 new Track3B checks PASS after
  repair; WM zero-head direct-MAPPO bridge bitwise equal over4physics ticks;
  both residual signs finite in WM; Ruff PASS. The full WM ranking production
  rollouts were not launched before the operational stop.
- Real SUMO canary, TRAINseed969440,300s warmup+600s control per arm:
  **plain MAPPO J600=10,157.678294306275 == zero-head3B J600 bitwise**.
  Both initial-state SHA
  `bb3ce9aa2eb64143a5d6e599abd2651e984f578eb18bf83e64e2b9045572698a`.
  Episode walls25.792354/34.550413s;0collisions/teleports/dropped/lane calls.
- 3B acting share1604/1617=0.9919604205318491; bounded residual-command signs
  negative0/positive0/zero1604. These are composed-command differences before
  the final physics cap, not counterfactual physical-acceleration measurements.

### Operational stop, learning-curve availability and budget

Both canary workers completed successfully and the paired PASS receipt was
durably saved to `state_3b.json`. The next call `log('canary_done', **receipt)`
raised `TypeError: log() got multiple values for argument 'kind'`, because the
receipt includes `kind`. This is a driver failure, not an episode safety failure
or scientific rejection. The explicit mechanical-failure rule stopped further
dispatch. Repair separates the event name from receipt.kind; a CPU regression
executes the exact failing call and verifies both fields. There was no automatic
continuation, retry, refund or repeated canary. Shared remote STOP is installed.

**There is no Track3B learning curve or measured MAPPO panel mean yet.** Updates0,
accepted0; the final head remains allzeros/MAPPO-equivalent. The canary is one
world, not a substitute for the8-world panel or an improvement estimate.

- Historical charge47,700 plus canary1,800 = **49,500charged=actual sim-s**.
- **55unique precharged jobs=55completed results;600,500sim-s remain** under650,000.
- All55results have0safety/lane counters; all repeated-world initial hashes agree.
- No live task workers/drivers at final reconciliation; no signals or evaluation
  dispatch. Prior source/state/evidence retained; original faulty source archived
  in remote `3b_step1.tar`, error log unchanged.

Fresh-session tool-scope disclosure: the initial grep addressed REGISTRATION.md
but also returned unrelated parent-tree timestamp/path matches. Those outcomes
were not used; subsequent registration access used exact-path Read/patch.

### Track 3B final head / continuation handoff

Artifact [`FINAL_HEAD_3B.json`](../../tools/sumoes/evidence/FINAL_HEAD_3B.json)
is explicitly marked **not an eligible learned candidate**. A zero head is
MAPPO-equivalent under the3B loader, unlike the CTRL0-equivalent historical artifact.
The base checkpoint, trunk and law must accompany the scalar-head SHA.

- Canonical137-scalar head SHA:
  `226f2fad46e9e695b1fc277750a1bfdf5c403f17b38984efde3a16773e5fc255`.
- 3B export file SHA:
  `0e54f112cbacb6b558879c51c1605df5ca0065a26728993b839b6c722122c442`.
- Final3B state SHA:
  `4d766c47d725583544469a1723da808addac143d664c1f6f12556e1f972eaefa`.
- Combined ledger SHA:
  `6934f50cf2abff16d9fb834de2ce3fda4affd240e2a414b3586a1bfaaa956867`.
- Receipts: `3B_CANARY_STATE.json`, `3B_CANARY_ERROR.txt`,
  `3B_LOGGING_REPAIR.json`, `accounting_3B.jsonl`,
  [`FINAL_RECONCILIATION_3B.json`](../../tools/sumoes/evidence/FINAL_RECONCILIATION_3B.json).

Owner continuation is needed after this mechanical stop. The saved canary is
complete; a future authorized continuation should begin at the eight-world
MAPPO baseline, retaining both canary charges and original stopped-run receipts.
S-EVAL-A was not launched. No learned improvement is demonstrated by Track3B.

---

# Historical CTRL0-base final result (retained)

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
