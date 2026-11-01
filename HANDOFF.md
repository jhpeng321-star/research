# S-C1SWITCH2 integration handoff

Authoritative design: `docs/paper_results/C1SWITCH2_DESIGN.md` and `tools/c1switch2/design.py`.
Protocol `S-C1SWITCH2/v2`: **16 fresh main worlds 969136..969151**, canary969152,
six controllers in order `native,ctrl0,mappo,pi,circles,wall`, decisions300..880/20s.
Use `docs/paper_results/c1switch2/PANEL_v2.json` (16 worlds, unchanged identities).
SHA256 `7c780c25237dde73a657937ab577b8e48f71766817f4363f3298982e90144827`.
It exports seed,penetration,demand, common measurement indices and registered design SHA.
It replaces the unregistered preliminary12-world proposal seen by S-SELMAPPO.

## Evaluation ownership / selector peers
- **C1SWITCH2 owns all evaluation physics**, including both RL arms, once final weights exist.
- S-SELMAPPO and S-SELMAPPO2 should supply final checkpoint/config/source SHA256 receipts;
  keep their prospective evaluation episodes unstarted and reuse this task's paired receipts.
- This avoids duplicate comparator episodes and pays for the J900 tail here.
- Each main episode runs native300,30x20s choices, then last choice held300s to t1200.
  Primary J600 is t300..900; secondary J900 is t300..1200, with NO additional decisions.
- Selector ABI: frozen `select(anonymous_payload)->controller_name` for fleet PPO;
  per-CAV returns `{self_reported_cav_id:controller_name}` for exactly the current live CAVs.
- Per-CAV replay must rebuild its actual allocation/memory/entrant default rule causally;
  S-SELMAPPO2 supplies its validated grouped execution bridge. It is not simulated by fleet actions.
- Training observations/reward remain those registered by each selector task. No online updates here.

## WALL ABI / S-WALLSW
Integrated S-WALLSW through `1b0581d0` (handoff `07fb123e`); frozen v2.1/v2.2 law.
The v2 backend injects a wall facade with:
- `create_sumo()->controller`, whose `start(now_s=...)`, `stop(now_s=...)`, `command(env)->(command,kwargs)`
  obey the shared `Controller('wall',inputs)` lifecycle proposed in the selector handoff.
- `capture_memory(controller,env)->state`: only causal controller-owned group/command state,
  CAV self IDs only; no simulator HDV identities, hidden settings or future requests.
- `validate_memory(state,own_ids,now)->same_state`: exact field/self-ID/time validation;
  accepts `None` for inactive WALL. Predictor RPC preserves this memory in its anonymous key.
- `validate_lane_commands(counts)`: attest wall-specific commands; original five keep v7's zero-lane gate.
- Engine imaginations must use the same assembly/release state and retain it for incumbent WALL;
  stop/reset ownership when leaving WALL; fresh start when re-entering it.

## Runtime / WM-BB engine gates
`runner.py` loads hash-bound `runtime`, `predictors`, and `selectors` entrypoints from BINDING.json.
Runtime factory `(binding)->runtime`; `runtime.create_backend(path,world,ledger,stage,id,arm=... )`.
Fleet backends can reuse `tools.c1switch2.backend.Backend(...,wall=facade)` after frozen v7 setup.
The per-CAV runtime must supply the validated selector bridge with the same backend methods.
All construction happens AFTER current registration approval and integration admission.

`inference.Predictor` accepts attested `imagine_all(payload,K=8,horizon_s=60)` six-row batching,
or isolated `imagine(...)` per controller. `engine.Engine` implements real frozen WM2/BB feedback.
The six candidates get identical anonymous CRN seeds, isolated controller memory and true feedback.
MAIN00:29 permits a registered COMMON ROAD-ONLY prediction objective for both WM and BB.
Applied explicitly: entries OFF; mix configured but inactive; no source cost prediction.
Full-team J600/J900 and full-team branch oracle remain the evaluation objectives.
`selectors.py` bridges the fleet peer's FrozenFleetSelector and per-CAV save_checkpoint ABI;
`runtime.py` wires the per-CAV owner's make_mixed_fleet and shared frozen executor.
`SELECTOR_BINDINGS_PENDING.json` documents missing final169 weights: stubs raise, never pick.
Supply final checkpoint + immutable freeze/source/config receipts, not intermediate checkpoints.
Grouped execution/main admission and both-GPU parity remain gates.
Both real frozen checkpoints now PASS the CPU K8/60s six-way feedback probe on gpuR;
receipts `MODEL_CPU_{wm,bb}_steps120_c1switch2_run_7e448ccdafc0c1c3.json` preserve provenance.
Per S-SELMAPPO2's `SELMAPPO2_HANDOFF.md`, per-CAV features synchronize from live
backend contexts before EACH pick, including entrant/re-entry resets and retirement.
No WM-PHYS activation without its independently hashed S-RB-EVAL qualification receipt.

## Current execution state
**MAIN12:14 continuation now REGISTERED + immutable CPU PREPARED:** bothGPUroot
`/root/c1switch2_gpu_bb_cont_74807c59d8cd942c`, realBB full60s/K8/sixway and
incumbentWALL CPUprobes PASS, identicalsamplesSHA37852b51 on bothcards.
No newGPU/physics; oldCPUstage reused unmodified. gpuR rootfree55869440bytes
at latestresourcecheck, bothcards memoryPASS and load13.08PASS, diskBLOCKED.
Postnoon supervisor built: resource observations60s, ownedprogress+commits20min,
bounded24h observation. Only first unstarted gpuRfleet and MAIN12:14BB attempt
may be admitted after strictresources; durable per-scope attemptclaims forbid
automatic retries even if startup errors. Originalobserver25348EXITED required.
GPU accounting now includes BOTH original BB startup and new continuation;
final total staysUNKNOWN while any serviceLIVE, finalized debit always retained.
CPU22/22/RuffPASS incl liveallocation accounting and no-retry/diskrefusal tests.
**MAIN12:14 BB continuation built/CPU-tested, NOT dispatched:** exact plan
74807c59d8cd942ca0dbc39427a7b1264880e2ac5ce2e3d9c72c0ee80e565110.
Preserve original BB roots/STOP/parity/final receipts and zero-physics CPU stage.
New immutable GPU roots use the SAME frozen engine binding/checkpoints/objective;
external allocation metadata subtracts exact oldgpuA21.381694240495563 and
oldgpuR21.642283007502556s from original5400s/card: remaining5378.618305759504
and5378.357716992497s, aggregate10756.976022752002s. No cap increase/refund.
One-use exception requires exact EXITED predecessor/final receipt hashes,
original16-world CPU stage stillunclaimed/empty, current authority+registry,
real both-GPU parity and CPU gates. gpuRdisk>=2GiB/load<20 and bothGPUfree>=8GiB
checked BEFORE claim/allocation. ThreeBB+onefleet workers<=4taskgpuRslots.
CPU89PASS with existingMiniconda (defaultPython lacks torch), targeted28PASS,
RuffPASS. Prospectively register exact plan, then `bbcontinue prepare`; only
after resources pass, `bbcontinue start`. Original WM episodes remain terminal.
**12:15 fresh-session reconciliation1793506521:** cpuQ fleet1152412 remains
LIVE/noSTOP, all12local episodes/360decisions complete and original oracles
progressing (34closedjobs,28680charged/28080closedactual). Original cpuQ WM
oracle1163082 COMPLETE24/24,14640actual; original gpuR12/12 COMPLETE6000actual.
No completed work replayed. Aggregate196560charged/195960closedactual,
GPU27729.12296916172s retained. cpuQ uptime6.028days/free6580211712bytes,
ownoutputs334586200bytes. Peer per-CAV606968 stillLIVE/final169absent.
MAIN12:14 registered BB startup continuation exception observed; original
43.02397724799812GPU-s MUST remain charged. Continuation not yet built/claimed.
gpuR fleet remains UNCLAIMED: fresh load12.963 passes but rootfree104337408
and datafree2032021504bytes fail2GiB gate; wait for MAIN archival, no dispatch.
**12:00 final foreground supervision receipt1793505658:** cpuQ1152412 LIVE/noSTOP,
fleet8/12local episodes complete/292decisions,11240charged/9600closedactual;
cpuQ originalWMoracle1163082 LIVE/noSTOP,8/24branches closed,
4960charged/4560closedactual. gpuR original12oracles COMPLETE6000actual.
All112fixed/random134400+canary4680 preserved. Task169440charged/
167400closedactual at1793505654; openjobs excluded from closedactual. GPU final
27729.12296916172s=7.7025341581h; base468380sim-s/19h unchanged.
cpuQfree4958310400bytes/ownoutputs313255286bytes. gpuR fleet4world partition
stillUNCLAIMED: latest root143589376/data2032021504bytes free and load42.67>20,
so no dispatch. BB remains terminal0physics/43.02397724799812GPU-s; explicit
MAIN continuation exception absent. Per-CAV606968LIVE/final169absent.
MAIN11:48 FINAL WM-PHYS ruleFAIL now definitive: optionalPHYS OFF, keep WM2.
Original observer25348 completed its noon deadline normally at1793505605 and
EXITED; no replacement/duplicate watcher. Running science remains under its
existing immutable supervisors/hardcaps. FullpairedWM/RL primaryCI unavailable.
Resume via `python -B -m tools.c1switch2.reconcile`; do not redispatch cpuQ fleet
or either oracle scope. For gpuR fleet's FIRST dispatch only after CPU/disk gates:
`python -B -m tools.c1switch2.selectorops dispatch --host gpuR --workers 1`.
**11:55 original recovered-cpuQ oracle continuation now LIVE:** own1163082,
`/home/ubuntu/c1switch2_run_oracles_0e790a668b7ec5d8`, exactly24 original
UNSTARTED branches/14640sim-s cap/GPU0. Registered plan and immutable original
source/gate/predecessor-ledger proof checked before exclusive claim; no episodes
or GPU services restarted. FleetcpuQ1152412 remains independent under its own cap.
Receipt `CPUQ_WM_ORACLE_DISPATCH.json`; observer checks use exact owned roots.
**11:55 recovered-cpuQ ORIGINAL oracle-only scope prepared (no compute):** all
four durably completed WM episodes have24 original six-way specs and ZERO prior
branch charges/jobs. Exact14640sim-s/GPU0; retain original5460incl both failures,
combinedcpuQWM20100<=original63480. Same registered measured indices/prefixes;
no failed/new WM episode, inference restart, replacement world, or completed replay.
PlanSHA0e790a668b7ec5d8140510203f997067b0b301cb90a72db0a81b3b89157f3a86;
one nice19/load28/disk2GiB-yieldcpuQworker, original byteverified runtime/gates,
one-use claim and stop-on-child-error. CPU5/5/RuffPASS; prospective registration
and final source/binding verification precede launch. OriginalgpuR12oraclesDONE.
**11:52 accounted heartbeat:** fleet cpuQ1152412 LIVE/noSTOP,2episodes complete,
103decisions,4160charged/2400closedactual; gpuR partition stillUNCLAIMED/diskblocked.
Task aggregate157400charged/155640closedactual at1793505157, exact totalGPU
27729.12296916172s (WM27686.09899191372+BB43.02397724799812). Basecap468380
unchanged; unspent reservations do NOT authorize terminal-arm retries. Updated
`STATUS_CURRENT.json` supersedes old UNKNOWNcpuQ/unusedBB/finalFleetPending state.
Reconciler now preserves unknown exact totals if a host is unreachable and excludes
open-job physics from closedactual; accounting regressions2/2/RuffPASS.
cpuQ housekeeping increased free space to5034971136bytes; ownoutputs302384354bytes.
**11:50 fleet CPU evaluation LIVE / disk gate yielded:** owncpuQ1152412,
two workers at original `e79775aa1f380c25`,1340charged/37decisions/noSTOP at
1793505028. OriginalgpuR four-world partition `6e152d9a0b7bf7a4` UNCLAIMED:
resource refusal BEFORE claim/physics, rootfree143589376bytes/data2032021504bytes
(both<2GiB). No deletion or relocation. MAIN notice `_sol/C1SWITCH2_1150_DISK_AND_BB_ALERT.md`.
cpuQfree2972311552bytes/taskoutputs301123940bytes, per-CAV606968LIVE,
observer25348LIVE. Reconciliation receipt1793505031; original caps unchanged.
**11:47 original WM oracle scope COMPLETE:** own885469 EXITED/noSTOP, all12
branches closed, charged=actual6000sim-s/GPU0. Never redispatch. Original WM
total now14160sim-s (5460cpuQ+2700gpuR+6000continuation), six completed episodes.
**Final-fleet immutable admission ready:** cpuQ root `c1switch2_run_e79775aa1f380c25`
and gpuR root `c1switch2_run_6e152d9a0b7bf7a4`, EACH147CPU PASS/1host-onlyskip,
bothguards17/17, real final169 hash/source validation and CPU-policy probePASS.
CPU probe initially omitted frozen-runtime setup/nice19/SUMO executable PATH;
all three refusals preceded policy/physics/claims. Reused successful cpuQ stage
and its existing preflight; installed same runtime environment as real episodes,
no repeated preflight/jobs/charges. gpuR copy contains only pinned final checkpoint,
frozenreceipt and declared Python sources. Selector dispatch stillUNCLAIMED.
Fresh read-only monitor scans now include selector roots and WM-oracle completion;
original live watcher25348 remains untouched. CPU regression6/6/RuffPASS.
**Final-fleet evaluation bridge built, not yet dispatched:** delivered genuine
final169 SHAbe0bb66d is now accepted by selector-only immutable CPU stages.
Same original12cpuQ/4gpuR worlds, full1200s episodes+registered six-way full-prefix
oracles, original79800sim-s/GPU0. `selectorops` validates frozen checkpoint/source
receipt, CPU-loads the actual policy before any claim/physics, keeps all common
scientific gates and rejects duplicate same-arm claims. No training restart.
Local84CPU PASS/RuffPASS. Prospective registration and immutable host preflight
are still required before dispatch; per-CAV remains pending its own final169.
**11:44 diagnosed BB admission defect, CPU fix only:** `registry_hits` counted
the exact own06:00 durableSTOP job reference as a conflicting allocation.
Narrow correction removes ONLY that known job token in that exact incident;
other seed reservations on the same line remain conflicts. BB startup now
checks current registry BEFORE allocating GPUs. CPU77/77 PASS, no restart or
new scientific dispatch; terminal BB claim/spent43.02397724799812GPU-s retained.
**11:40 fresh-session reconciliation:** cpuQ uptime6.003days, NOT rebooted.
Original WM supervisors649272/763520 both EXITED with durable STOP; cpuQ now
reconciled5460charged=closedactual (4episodes/123decisions), gpuR2700 (2/60).
All112fixed/random episodes134400 and mechanics4680 remain complete. No replay.
Original gpuR oracle continuation885469 LIVE/noSTOP,6/12branches complete,
2840charged/2280closedactual at1793504329;6000cap/GPU0 unchanged.
BB startup before this resume passed real both-GPU parity, then stopped BEFORE
physics on registry audit false-positive: the task's own06:00 failure report
mentions world969147. GPU services432280/885526 EXITED, retain exact spent
21.381694240495563+21.642283007502556=43.02397724799812GPU-s; BBphysics0.
Do not retry its terminal one-use dispatch or reset/refund its allocation.
Fleet final169 checkpoint receipt now delivered as `SELMAPPO_FINAL.json` (peer
training645235 EXITED); per-CAV606968 remains LIVE, final169 receipt absent.
Observer25348 LIVE throughnoon. cpuQ free3,049,521,152bytes; our task roots total
296,200,608bytes. Read-only evidence `RESUME_RECONCILIATION_1793504400.json`.
**11:35 fresh-session continuation:** original registered twelve UNSTARTED WM
oracle branches now dispatched once on gpuR, ownPID885469, root
`/root/c1switch2_run_oracles_8d6ceccb69058bb0`;6000sim-s cap/GPU0.
Time/load/current-registration, predecessor ledger, immutable supervisor and CPU
binding checks passed. No original WM episodes or completed CPU arms replayed.
Receipt: `WM_ORACLE_CONTINUATION_DISPATCH.json`. Existing observer25348 remains
live; unrelated selector handoff changes in this worktree are preserved.
**Additional CPU-only original-WM oracle continuation (after11:30 only):**
gpuR stopped-stage proof confirms2700charged, no oracle branch ever charged,
and exactly12 registered specs for completed worlds969139/j1 and969143/j13.
Finish these UNSTARTED six-way sets for6000sim-s/GPU0; retain old2700 incl failure,
combinedgpuRWM8700<=16320. No new WM pick or GPU allocation, no cpuQ-world replay.
Plan8d6ceccb69058bb0c753f57c7558f8a129e608951c280d5cd123e0a109c4926a.
`wmoracles.py` copies the original byte-verified runtime/BINDING/GATES/specs,
adds a one-use12-job supervisor, preserves every predecessor artifact and stops
on any child error. Original measured states/prefix costs remain380x6+620x6.
One oracle worker plus THREE relocatedBB workers respect the task's4gpuR cap.
CPU26/26/Ruff PASS; source/plan committed before prospective registration/stage.
Original-WM oracle stage now REGISTERED and CPU binding verifiedPASS:
`/root/c1switch2_run_oracles_8d6ceccb69058bb0`,106original source hashes and
unchanged original BINDING/GATES; no jobs/ledger/GPU. First metadata check used
the wrong cwd and stopped before any job, corrected check uses pytools as worker
dispatch already does. At/after11:30 ONLY: `python -B -m tools.c1switch2.wmoracles start`.
Plan/stage/CPU receipts are WM_ORACLE_CONTINUATION_{PLAN,STAGE,CPU}.json.
**08:xx MAIN07:07 unused-BB relocation prepared in source, NO compute:**
prospective plan d5037b81825d4788131b49c675934ee71e6007da91fec60012de9ecdbc75abde
moves ONLY unstarted BB's same16worlds to<=4gpuRCPUworkers AFTER11:30, with
original79800sim-s/3GPUh (bothcards) and all scientific/one-use/parity gates.
Runner/ledger accept only this exact registered arm/host/manifest; no WM replay.
Relay now requires16episodes before releasing GPUs. Local96CPU/Ruff PASS;
defaultPython has no torch, so existing Miniconda was used, no environment created.
See docs/paper_results/c1switch2/BB_RELOCATION_DESIGN.md and bbrelocate.py.
Immutable gpuR CPU preflight and prospective registration precede any allocation.
**Unused BB placement now REGISTERED + CPU-VERIFIED:** immutable gpuR root
`/root/c1switch2_run_bbd7ba8b48ed5829`, binding54848c3e7059ef2691878116db7c76dbead02962d42e1f281aae4ad8514b5802;
143CPU PASS/1host-onlyskip, BOTH guards17/17, no GPU/SUMO. All16BBworlds and
79800total cap verified. `BB_EXECUTION_BOUND.json` is the ONE prepared config.
At/after11:30 ONLY: `python -B -m tools.c1switch2.bbrelocate start`; it refuses
early time/prior claims and starts only with CPU/GPU resources, then real parity.
Do not run old follower/prepare or duplicate this binding. Scientific start remains
UNCLAIMED, no automatic science retry. Original cpuQ WM work remains UNKNOWN.
**06:00 terminal WM GPU receipts reconciled:** BOTH services EXITED after
ordinary COMPLETE release; gpuA13845.159507265314s + gpuR13840.939484648407s
=27686.09899191372GPU-s (7.6905830533h), all validation/idle included, no refund.
BB remains0physics/0GPU and observer36376 is deliberately PARKED while cpuQ
is offline; do not call its expected missing-process alert an unknown crash.
cpuQ WM current charge/actual remainUNKNOWN. Verified total lower bounds only:
146620charged/145380closedactual (CPU134400+canary4680+historicalcpuQWM4840/3600
+finalgpuRWM2700/2700). Six WM episodes confirmed by finalgpuR2 plus last relay
cpuQ4; full panel/oracles/primary WM-vs-RL are INCOMPLETE, no primary efficacy CI.
Receipts GPU_RELEASE_wm_1793484020 + GPU_PROGRESS_{gpuA1793484027,gpuR1793484032};
STATUS_CURRENT now distinguishes unknown totals from these historical lower bounds.
Terminal WM relay31696 creation/command identity verified and stopped by exactPID
AFTER both GPU_FINISHED receipts, avoiding pointless rapid SSH polling. Its
RELAY_wm/COMPLETE.json records failed=true/episodes_done=false and verified proof;
no original claim/log/lock was removed. Only read-only20min observer25348 remains
active locally for this task; no inference/science redispatch is armed.
CPU-only transport correction for future unused BB work: `relay_cycle` now
isolates each CPU/GPU/route observation. An offline first CPU is UNKNOWN and
cannot hide another host's durableSTOP/global release or starve its immutable
RPC routes. No route fallback, scientific retry or engine/controller change.
Two outage/STOP regressions plus waiting/observer gates:10/10CPU, Ruff PASS.
This code is not a WM restart and has not launched any BB allocation.
**05:59 WM global-stop release required:** gpuR supervisor763520 EXITED with
durableSTOP `wm__s969147`; exact worker traceback is inference unavailable timeout.
Its2completeepisodes/60decisions and2700charged=actual are preserved, including
300s failed native prefix. cpuQ remains unreachable/UNKNOWN. Ordinary relay
policy releases BOTH GPUs on ANY owned CPU STOP, but its first-host SSH failure
prevented reaching this check. `gpuops release-after-cpu-stop` applies exactly
that existing release via hash-bound engine and owned PID/root checks; no reset.
BB waiter36376 creation/command identity verified and stopped by exactPID while
still waiting, BEFORE WM release, preserving BB's unused3GPUh/79800sim-s until
host access returns. Original claims/logs retained; no science replay or restart.
**05:38 shared cpuQ SSH outage:** direct connection is closed before login;
one approved-host jump attempt also failed. MAIN's shared registration reports
the same cpuQ outage in S-COOPFREE05:28. Current cpuQ WM status is UNKNOWN;
do not turn missing observations into zero counts or infer successful closure.
Immutable completed CPU totals remain112episodes/134400actual, plus4680canary.
05:37 heartbeat749d9740 retained transport alert; its printed28CPU/2WM counts
cover ONLY reachablegpuR, not a regression/deletion of the other host's results.
Both owned WM GPU services remain LIVE/noSTOP,98/90responses at that snapshot;
relay31696 is LIVE but last successful full status1793481610 is stale while it
retries cpuQ transport. No scientific restart, replay, refund or cap reset.
Observer25348 and corrected BB waiter36376 LIVE; BB remains0GPU/physics.
Pending: restore host access, reconcile exact ownSTOP/open jobs, retain any
failed reservations; optionalPHYS still unqualified after MAIN05:27 retraction.
**05:13 BB waiting observer failure diagnosed:** originalWindows43404 exited
on `RuntimeError: Connection closed by 172.19.0.17 port 11000` while polling WM
release, BEFORE entering its science try-block. Original START/WAIT/log and
closed exclusive lock retained. BOTH exact BB GPU roots still have no DISPATCH,
GPU_STARTED or physics; zero allocation spent. CPU/WM jobs remain untouched.
`modelops.wait_for_release` now retains transient transport errors and repeats
only read-only resource observations. `resume-wait-start` permits ONE explicit
correction after verifying exited43404/config hashes/original error/noSTOP/no
bound BB execution/no GPU claim or BB physics claim on either GPU host.
Original BB caps/gates and science no-retry behavior remain enforced. Regression
CPU7/7 and Ruff PASS; corrected observer dispatch receipt will follow.
**05:15 corrected read-only BB follower36376 LIVE**, one-use claim SHA256
`bd690a89f2d65526a1769a3a7fed708662a47cdade98899123d0b179bb1abf56`.
Original43404 is terminal and must not be monitored as the current follower.
Proof/START/dispatch are `BB_FOLLOW/WAIT_RESUME_{ONCE,STARTED}.json` and
`BB_FOLLOW_WAIT_RESUME_DISPATCH.json`; error and original receipts preserved.
This continuation was disclosed in shared REGISTRATION before its launch.
Read-only monitor follow-up now resolves the current BB follower from its
correction receipt and alerts on missing local relay/follower processes. Its
own observer-only replacement path refuses a still-live predecessor and keeps
all first-run claims/receipts; no science process is restarted. CPU8/8/Ruff PASS.
Observer-only replacement complete05:17: old27200 identity/creation-time verified
and stopped by exactPID; new25348 LIVE through original noon deadline, first
heartbeat8da4445e: WM4episodes/181decisions,6920charged/4800closedactual,zeroalerts.
Current BB observer36376 is verifiedLIVE. All science PIDs/allocations unchanged.
Observer replacement receipts: WATCH_MAIN0427/{RESUME_ONCE,RESUME_DISPATCH,RUN_RESUME_ONCE}.json.
**04:29 +08 resume verified (Unix1793478517..1793478575): all112 CPU-panel
episodes COMPLETE, charged=actual134400sim-s (cpuQ84/100800; gpuR28/33600),
both supervisors EXITED successfully/noSTOP. NEVER redispatch either CPU panel.**
WM supervisors649272/763520 live/noSTOP:3complete episodes,126validated decisions,
5220charged/3600closedactual; open-job actual excluded. Both own GPU services
342449/761968 live, noSTOP/FINISHED; elapsed clocks remain charged under original caps.
Local relay31696 and BB follower43404 alive; BB WAIT fresh at1793478527,
both WM allocations unfinished, so BB still0physics/0GPU. No duplicate dispatch.
Total at these receipts:144300charged/142680closedactual including4680canary;
GPU aggregate elapsed approximately4.66h, finalmonotonic charge pending.
Peer final169 checkpoints remain undelivered; follow peer observers, never evaluate
intermediate weights. Next owned-job heartbeat due about04:49, then every20min to noon.
Receipts: CPU_PROGRESS_{cpuQ_1793478517,gpuR_1793478525},
MODEL_PROGRESS_wm_{cpuQ_1793478535,gpuR_1793478539},
GPU_PROGRESS_{gpuA_1793478570,gpuR_1793478575}.json.
`tools.c1switch2.watch` adds read-only20min heartbeats through12:00+08:00,
with explicit-root/PID identity checks, accounting deltas, STOP/stall alerts,
current task registration and final-checkpoint receipt presence. Each heartbeat
appends a short line here and commits ONLY its own explicit evidence paths;
an already-staged index is preserved and blocks its commit. It cannot dispatch,
restart, signal or retrain scientific jobs. CPU4/4 and Ruff PASS before launch.
Watcher ownWindows27200 is LIVE, first04:32 heartbeat committed9da788a4/zeroalerts;
next04:52, then20min cadence and final12:00 check. Receipts/owned-PID claim:
`tools/c1switch2/evidence/WATCH_MAIN0427/{DISPATCH,RUN_ONCE,STATUS}.json`.
Never duplicate this watcher; each check remains read-only with respect to science.
Latest resume state: `tools/c1switch2/evidence/STATUS_CURRENT.json` plus task-only `operations status`.
Mechanics-only canary PASS: owncpuQ PID467748 EXITED, root
`/home/ubuntu/c1switch2_run_4ad4ece01d5a9ac1`; charged=actual4680 sim-s, GPU0. Never redispatch it.
Both native and full30x20s six-way replay pairs matched bitwise; all6 alternatives passed.
Original receipt bytes exported as `CANARY_c1switch2_run_4ad4ece01d5a9ac1.json`, SHA256
`2cdfd15288ac7dc657ed35aa86117ec429698d022fedef3600f0a27bceadb55d`.
Accounting plus immutable BINDING/GATES/STAGED/PREFLIGHT archived beside it.
MAIN01:48 CPU subset is now running: cpuQ PID604784, gpuR PID759845.
cpuQ `/home/ubuntu/c1switch2_run_eb0eda956a815f14`, four workers,100800sim-s.
gpuR `/root/c1switch2_run_716d39f08986c84f`, one worker,33600sim-s.
Exactly native/CTRL0/MAPPO/PI/CIRCLES/WALL/random;112episodes total,134400sim-s,GPU0.
Both hosts pass130CPU/1skip. Never duplicate their write-once CPU dispatch claims.
Use `operations cpu-collect --host HOST --root ROOT`; retained main accounting is incremental.
WM picker RUNNING: cpuQ649272 (2workers) + gpuR763520 (1worker), model CPU roots
`/home/ubuntu/c1switch2_run_292bfffe94b13ed8` and `/root/c1switch2_run_292bfffe94b13ed8`.
WM services gpuA342449/gpuR761968 share root `/root/c1switch2_gpu_wm_95614dbfc754e322`;
hard28800GPU-s/card includes loading/validation/idle, never restart/top-up.
Real six-way/K8/full60s/fresh+continued-WALL GPU parity PASS; both model CPU stages133/1skip.
Local Windows relay31696 is LIVE; `tools/c1switch2/evidence/RELAY_wm/STATUS.json`.
Validated main RPC responses and first two20s decisions now recorded; no efficacy result yet.
One-use BB follower43404 waits for BOTH owned WM services to end and free>=8GiB
(resource-only context margin), then BB parity/CPU gates and original79800sim-s/3GPUh arm.
See BB_FOLLOW/WAIT.json, BB_EXECUTION.json; never launch a duplicate follower/service/arm.
BB frozen checkpoint/physics hashes and both-host CPU K8/60s/all6/continued-WALL probes PASS.
Priority admission left peers untouched. Optional WM-PHYS remains independently gated.
Latest fleet peer delivery: `SELMAPPO_MAIN0208_RUNNING.md`: owncpuQ645235 continuation
running, no final169. Per-CAV owncpuQ606968 continuation likewise has no supplied final169.
Do not substitute checkpoint000 or restart peer work; peers acknowledge our evaluation ownership.
Initial PID446935 exited before setup/physics with zero charge; all receipts retained.
Repair1065f89d was explicitly registered and passed pristine-process CPU construction before dispatch.
The 23:48 owner approval and MAIN01:48 independent CPU authorization are present;
WALL integrated, final trained selectors pending.
`tools/c1switch2/wall.py` binds the real peer lifecycle and self-ID-only memory;
missing members remain absent and abort atomically on continuation. No simulator slots cross RPC.
Latest model cpuQ AND gpuR CPU verification: 133 passed/1 host-only skip each.
Tests include real wall memory parity, six-row synthetic WM/BB mechanics and actual peer
feature/model interfaces with clearly synthetic final169 test weights (not trained/evaluated arms).
`python -B -m tools.c1switch2.gate --polls 1` reports approval without authorizing an unbound runtime.
CPU preparation/exposure receipts are in `tools/c1switch2/evidence/`; prior versions are preserved.
Old cpuQ `/home/ubuntu/c1switch2_run_2d2400c953ef2545` is a retained zero-physics STOP.
gpuR `/root/c1switch2_run_7e448ccdafc0c1c3` is CPU-only; no GPU services or main jobs armed.
New CPU-only snapshots `c1switch2_cpu_6b84f28df5d4` on both hosts pass127/1skip,
including the actual peer grouped-WALL retirement bridge and bound two-GPU RPC factories.
`operations.py status` inspects task-only dispatch receipts; canary is a write-once
cpuQ claim using the original4680 sim-s reservation, never an automatic retry.
The initial local exposure regex reported timestamp/negative-evidence false positives, fixed prospectively.
An exact-file Grep call expanded into the parent tree; incidental numerical matches were not used.
All subsequent registration reads/audits use exact checked paths; filename scans prune before traversal.

- 2026-11-01T04:32:28+08:00 cpu:episodes=112,decisions=3360,charged=134400,closedactual=134400; wm:episodes=3,decisions=131,charged=5320,closedactual=3600; alerts=0; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793478748.json`.

- 2026-11-01T04:52:28+08:00 cpu:episodes=112,decisions=3360,charged=134400,closedactual=134400; wm:episodes=3,decisions=157,charged=5840,closedactual=3600; alerts=0; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793479948.json`.

- 2026-11-01T05:12:28+08:00 cpu:episodes=112,decisions=3360,charged=134400,closedactual=134400; wm:episodes=4,decisions=175,charged=6800,closedactual=4800; alerts=0; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793481148.json`.

- 2026-11-01T05:17:51+08:00 cpu:episodes=112,decisions=3360,charged=134400,closedactual=134400; wm:episodes=4,decisions=181,charged=6920,closedactual=4800; alerts=0; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793481471.json`.

- 2026-11-01T05:37:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2400; alerts=1; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793482671.json`.

- 2026-11-01T05:57:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2700; alerts=2; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793483871.json`.

- 2026-11-01T06:17:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2700; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793485071.json`.

- 2026-11-01T06:37:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2700; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793486271.json`.

- 2026-11-01T06:57:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2700; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793487471.json`.

- 2026-11-01T07:17:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2700; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793488671.json`.

- 2026-11-01T07:37:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2700; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793489871.json`.

- 2026-11-01T07:57:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2700; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793491071.json`.

- 2026-11-01T08:17:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2700; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793492271.json`.

- 2026-11-01T08:37:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2700; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793493471.json`.

- 2026-11-01T08:57:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2700; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793494671.json`.

- 2026-11-01T09:17:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2700; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793495871.json`.

- 2026-11-01T09:37:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2700; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793497071.json`.

- 2026-11-01T09:57:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2700; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793498271.json`.

- 2026-11-01T10:17:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2700; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793499471.json`.

- 2026-11-01T10:37:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2700; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793500671.json`.

- 2026-11-01T10:57:51+08:00 cpu:episodes=28,decisions=840,charged=33600,closedactual=33600; wm:episodes=2,decisions=60,charged=2700,closedactual=2700; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793501871.json`.

- 2026-11-01T11:37:25+08:00 cpu:episodes=112,decisions=3360,charged=134400,closedactual=134400; wm:episodes=6,decisions=183,charged=8160,closedactual=8160; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793504245.json`.

- 2026-11-01T11:57:25+08:00 cpu:episodes=112,decisions=3360,charged=134400,closedactual=134400; wm:episodes=6,decisions=183,charged=8160,closedactual=8160; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793505445.json`.

- 2026-11-01T12:00:00+08:00 cpu:episodes=112,decisions=3360,charged=134400,closedactual=134400; wm:episodes=6,decisions=183,charged=8160,closedactual=8160; alerts=3; receipt `tools/c1switch2/evidence/WATCH_MAIN0427/PROGRESS_1793505600.json`.
