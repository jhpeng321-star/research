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

Attempt2 source1f5cc6aa passed58CPU checks and the separate strict CUDA
copy-parity check. Coordinator362850/start855400127, worker362856 again
disappeared before a durable update; last GPU sample5,026,873,344bytes at
08:10:34UTC. Status SSH intermittently closes; no peer workload was touched.
Parameter-copy sharing alone did not make the full-K replay viable.

The next execution refinement keeps the complete original K8 forward,
quantile/cooperation/safety reductions and their global VJP, but replays one
future graph at a time. Other futures retain detached original forward values;
each active future still differentiates through full-K coupled reductions.
Parameter VJPs sum before the single original optimizer step. Exact random-tape
boundaries and final RNG state are retained. This bounds graph memory across
futures while keeping microbatch1/segment8/both offloadsON. Require existing
LONGMEM gradient/optimizer/RNG proofs plus a production residual-actor K8/q10
case before another real dry attempt. No full policy phase has launched.

## Current boundary: BLOCKED_C4LAUNCH_MEMORY_AND_SSH

Last successful status08:13:49UTC confirms worker362856 not alive and no
completed dry updates. Subsequent status, diagnostic and staging connections
close; verbose fresh non-multiplexed SSH confirms TCP connects but the provider
closes before key exchange. No credential/source/process workaround was used.
Staging a19deb5e failed at the initial source query BEFORE installation, so the
streamed-future repair remains local, committed and **unverified/unactivated**.
`tools/c4prep/evidence/LAUNCH_BLOCKED.json` binds the boundary. No full launch,
no C4A_RUNNING claim, no automatic retry, no SUMO/new charge, no signals/push.

### Resume only after gpuA transport recovers

1. `python -B tools/c4prep/stage.py` — immutable NEW source revision.
2. `python -B tools/c4prep/launch_ops.py checks` — must pass complete LONGMEM,
   production K8/q10 gradient/optimizer/RNG equivalence and CUDA copy proofs.
3. `python -B tools/c4prep/launch_ops.py prepare`, then `... launch_ops.py dry`.
   Preserve previous attempts; require actual two-update finite/nonzero-module
   gradient/credit/memory PASS, and commit the measured receipt before step4.
4. `python -B tools/c4prep/launch_ops.py start` — appends final anchor/config/
   dry-run bindings to loop JSON BEFORE detaching A, with unchanged gates.
5. Poll `... launch_ops.py status` at>=60s. Only after>=10 finite updates and
   successful gate/gradient/residual/credit-gap/memory evidence use
   `... launch_ops.py verify` to record C4A_RUNNING and preserve peer status.

Final transport receipt: one transient successful status at08:25:58UTC again
confirmed source1f5cc6aa/worker362856 exited/0updates. Subsequent staging failed;
commitd2ca68fb reduces staging to one fresh non-multiplexed source-only SSH
transaction, but both bounded attempts still closed. The streamed VJP remains
unverified and its activation is not confirmed. No background retry is armed.
Final compact snapshot: `tools/c4prep/evidence/LAUNCH_BLOCKED_FINAL.json`.

## MAIN17:00 recovery — one attempt, deadline 2026-10-29 06:00 UTC+8

Recovered clean HEAD2b2a5dbf and read registration/receipts before any dispatch.
The measured anchorOFF/q10-K8ON decision stands; no fitting repeated.
Initial status and later diagnostic SSH calls closed; >=300s refusal backoff
and >=120s remote polling apply. No new remote operation is confirmed.

Local CPU verification now uses an isolated Python3.13/torch CPU environment.
The previously unrun K8 fixture omitted a native pair for its third start;
corrected the synthetic fixture to pair every start as production requires.
Full production residual group/trunk training, nonzero synthetic trained head,
entrantON case, optimizer/target state and whole RNG-state parity are covered.
Complete requested CPU suite:60PASS/2CUDA-only skips in109.54s. These are
not GPU qualification or real training updates. CUDA full-actor K8 remains pending.
Qualification now detaches into source-bound CPU/CUDA stages with durable logs
and PID/results, removing the brittle35s SSH-bound qualification timeout.

Recovered one successful batched SSH at09:12:45UTC: MAIN17:00 appended to
loop JSON (authority SHA8835c2a201b388f20eff5c0ed184c70b1d4ea848098b3daabe2ea035c126e46b).
Both prior workers are dead. Second CHILD_EXITS records exit-9/SIGKILL;
worker log has no traceback, RUN_SUMMARY absent. Coordinator's traceback is
the later missing-summary read, not the worker's cause. cgroup current18.72GB,
max96.64GB, high140722, oom0/oom_kill0; killer identity remains unconfirmed.
Next source stage was refused; >=300s backoff resumed, no GPU job dispatched.

Operational monitoring now records own-worker hostRSS/high-water/cgroup pressure
with GPU samples and retains failed receipts when a summary is missing. The
detached coordinator enforces MAIN17:00's deadline against its own unreaped child,
and self-records C4A_RUNNING only after>=10finite updates plus the existing
phase-relative signal gate and full gradient/memory verification. It cannot
auto-launch policy after the dry run; the measured dry receipt must first be
committed locally. Live JSONL readers ignore unfinished last writes while still
rejecting complete corruption. Operational unit checks21PASS; fatal-error RuffPASS.

Transport continues intermittent: next immutable staging attempt again closed,
no qualification dispatch confirmed. `stage.py --checks` now batches immutable
source installation and detached qualification in that same SSH transaction,
with an existing source-bound PID receipt preventing duplicate dispatch. Both
installer modes compile and preserve native-head inclusion; operational suite
22PASS. No auto dry/policy launch is attached to staging.

Recovery09:31:04UTC confirms pointer still1f5cc6aa, no new qualifier PID;
the staging timeout did not activate the repair or dispatch GPU work.
Full-resume inspection found the independent C1-config overwrite defect noted
by S-HOLD-1 commit96a60687. Adapted its atomic registered-floor restoration for
A, additionally retaining the explicitly requested entrantON/deterministicCRN
flags whenever the anti-exploit boundary is requested. No checkpoint weights,
Adam state or settings changed. New historical-resume regressionPASS; both
installer-mode testsPASS (3total). Full local repeat timed out under120s and is
not claimed a new complete PASS; detached remote qualification remains required.
Staging uses fast source-only gzip and a pre-SSH bundle receipt to distinguish
local archive preparation from transport; no datasets/checkpoints are bundled.

The next full-source bundle (4,979,707bytes) prepared successfully but SSH closed.
Added receipt-bound delta staging from the confirmed immutable1f5cc6aa runtime:
local Git blobs determine unchanged bytes without a remote source probe, and
every installed file is SHA-verified against the new full manifest. Previous
runtime is read-only; qualification dispatch remains one-shot/source-bound.
All four full/delta × stage-only/stage+checks installer proofsPASS; verbose SSH
will identify whether any next failure is pre-authentication or during transfer.

Detached local complete CPU qualification on789d03d3 succeeded:63PASS,
2CUDA-only skips,149.92s; exact core VJP and resume repairs now fully CPU-verified.
Receipt `tools/c4prep/evidence/RESUME_CPU_FINAL.json`. Next delta179,967bytes
failed BEFORE SSH key exchange (TCP connects, no authentication or transfer),
so payload size is not the refusal cause and no qualifier was dispatched.
One local receipt-bound transport monitor may retry the same immutable source
at>=300s after refusal and poll detached qualification at>=120s, stopping on
qualification result/error or MAIN17:00's deadline. It cannot dispatch dry or
policy; no peer process signalling or SUMO path exists.

Source dde58943 staged successfully (1217SHA-verified files,12transferred)
at09:46:04UTC and detached qualifier389499/start856097866 completed normally
at09:46:56UTC, CPU stageFAIL. The qualifier's CPU/tmp parent did not exist;
pytest's basetemp setup failed. Original logs/results retained, CUDA not run.
Operational fix creates each CPU/CUDA stage parent before pytest. No dry or
policy was dispatched; the local transport monitor terminated on this failure.

## GPU-qualified recovery source cf1e9eda — two real dry updates next

Fresh immutable cf1e9eda staged1217verified files (1delta). Own qualifier
390414/start856121033 completed09:51:19UTC:65CPU testsPASS/2CUDA-only skips,
then2CUDA testsPASS (production full residual actor K8/q10/entrantON exact
gradient/update/optimizer/whole RNG parity and strict saved-copy parity).
No qualification timeout, no lingering qualifier; full receipts/logs retained.
Compact `tools/c4prep/evidence/RESUME_GPU_QUALIFIED.json` committed before dry.
Streamed full-K VJP repair is now GPU-verified/activated in that runtime.
No real dry-update PASS or full phase claimed yet; MAIN17:00 deadline stands.

Real two-update dry run detached09:53:46UTC, own coordinator391354/start856144075,
worker391362; isolated resolved_A_MAIN1500_cf1e9eda5f8f. Preparation verified
actual registered/effective individual/coop8.9/8.9, anchorOFF/q10-K8ON/90s.
Initial09:56:07UTC worker alive, hostRSS1.81GB, GPU0.81GB;0updates yet, still
initial scoring. Deadline/host+GPU self-monitoring active. Dispatch receipt
`tools/c4prep/evidence/RESUME_DRY_DISPATCH.json`; no full phase dispatched.

## Measured dry CUDA OOM — reuse completed B operational proof

Worker391362 exited1 with0actual updates by10:17:04UTC. This time the retained
RUN_SUMMARY reports CUDA allocatorOOM (324MiB request,21.32GiB allowance,
20.79GiB allocated). Host high-water14,162,716KiB; sampledGPUpeak23,290,970,112
bytes exceeded the strict process cap. Original output/failed dry receipt kept.
Receipt `tools/c4prep/evidence/RESUME_DRY_CUDA_OOM.json`; no full phase.

Fresh S-HOLD-1 delivery8ddc33c9 already proves two actual90s/K8 updates using
the same streamed core at segment8 and registered read-only component-gradient
diagnostic period25 (9afba5bc exact loss/gradient/Adam/RNG proof;12.47GB peak
reserved). Reuse that completed operational repair: explicit env period25;
actual per-module optimizer gradients remain logged every update. Segment1 was
considered locally but is NOT activated; microbatch1/segment8/both offloads stay.
New A production microbatch1/K8 proof compares auditsON vs period25 on full
actor/entrantON, including optimizer/RNG; remote qualification required before
fresh dry. Local operational tests25PASS; long new parity proof still pending.

Fixed receipt accounting discovered before launch: trainer and CLI both append
metrics to updates.jsonl, so verification/deadline/rate now use only UNIQUE
timestamped CLI completions. Ten writes cannot be counted as ten updates.
Added owned memory execution trace and preserved exception traceback/effective
configuration for diagnosis. Existing optional `ssh -J cpuP gpuA` relay from B's
handoff is supported via C4_SSH_JUMP without key/config or peer-job changes.
MEMORY_EXECUTION.json must be appended to registration/loop state before retry.

New immutable8e751843 qualified10:39:16UTC:66CPU testsPASS/2CUDA-only skips,
then2CUDA testsPASS. This includes production full-actor K8/q10/entrantON,
microbatch1, auditON baseline versus period25 streamed replay: loss/gradients/
updated parameters/Adam/target/RNG parity and nonzero actual module gradients.
MEMORY_EXECUTION prospective loop amendment confirmed, SHA6fc435fb536ca519f655442e4fdb8486ffb3da8845bcb226aa0577de26213756.
Existing cpuP SSH relay worked; qualifier403033/start856406773 exited normally.
Receipt RESUME_AUDIT_CUDA_QUALIFIED.json committed before next actual dry run.

Fresh two-real-update dry coordinator403793/start856426714 detached10:40:52UTC
on8e751843 after qualified proof and loop amendment. Registered8.9/8.9 weights
and all original scientific settings resolved successfully. Existing failed runs
untouched, no phase dispatch. Receipt RESUME_DRY_AUDIT_DISPATCH.json; read-only
watcher37624 uses the existing cpuP relay, >=150s polls/>=300s refusal backoff.
Own execution-memory trace tails are now exposed by the batched status transport.

First ACTUAL unique dry update181 completed (timestamp1793185543.097754):
finite actorloss0.03010/critic0, groupgrad712.30, own-trunk1405.98/report274.76/
attention751.33, mean|residual|0.055965m/s2, mean−q10 credit346.3602,
effective individual/coop8.9/8.9,90s. GPU reserved12,201,230,336bytes and sampled
process12,650,020,864bytes, host~13.76GB. Second scratch update running.
Receipt RESUME_FIRST_REAL_UPDATE.json; this is1of2, not a dry PASS/full launch.

Last confirmed11:18:35UTC: first unique update181 finite, second update's
future5 starting, host13.8GB/GPU12.65GB stable. Three subsequent cpuP-relay
status refusals at>=300s spacing and a direct refusal leave completion UNKNOWN.
Stopped only local read-only watcher37624 after verifying its exact creation
time/command; remote compute untouched. TCP DNS to1.1.1.1 is still intercepted
to Clash fake172.19.0.17; independent DNS-over-HTTPS returns106.120.183.117.
Optional per-command C4_SSH_HOSTNAME override retains the ORIGINAL pinned host
alias and strict host-key checking. This does not edit keys/global routing or
peer processes. Route test remains pending after a fresh>=300s backoff.

## REAL TWO-UPDATE PASS — source8e751843, before full phase

Public-IP pinned route also refused; after>=300s the existing cpuP relay
recovered11:55:59UTC. Original dry worker403814 had completed normally by
11:23:20UTC, exit0, actual unique181/182 BOTH90s, all finite/nonzero module
gradients and all effective flags/credit/memory checksPASS. No redispatch.
ConfigSHA31bbbff88ebbc3dbca441a06092a5828609b055c407343a1037ea87863c8d996,
summarySHA2fc09d05c347f14b51834b347d114d475835442c22c93f6030d4f59707b4ad80.
Mean|residual|0.057507m/s2, mean−q10 gap195.7121, group/trunk minima2.2300/
4.5096, reserved12.203GB/process12.652GB, host high-water13,494,068KiB.
Exact effective8.9/8.9 and60%focus retained. Interval rate3.404updates/h;
150-update projection is~44h including this slow exact K8 execution, not a
completed phase claim. RESUME_REAL_DRY_PASS.json committed BEFORE detached
launch, which must resume ORIGINAL64f70af4/C1 (never the scratch actor).

## FULL A DETACHED — awaiting ten UNIQUE finite phase updates

Dry PASS committed31290088 before start. Full150-update A coordinator428847/
start856893514 detached11:58:40UTC, ORIGINAL C1SHA64f70af4, config31bbbff8,
source8e751843. Final fit/config/dry/memory bindings appended to loop JSON
BEFORE dispatch; no scientific settings/gates/budget fields changed. All prior
attempts and scratch outputs retained. RESUME_POLICY_DISPATCH.json records this
dispatch. Own source-bound remote coordinator samples memory, enforces06:00,
and sets AUTONOMY C4A_RUNNING only after>=10UNIQUE finite updates plus required
module/credit/memory evidence and unchanged phase signal gate PASS. No premature
C4A_RUNNING claim. Read-only local watcher24280 polls>=300s through cpuP with
Windows child consoles suppressed; no extra dispatch or SUMO is armed.
