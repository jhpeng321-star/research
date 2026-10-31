# Fleet PPO selector — original stop preserved; MAIN continuation acknowledged

## Preserved attempt
Design `7e4def77`, initial implementation `a5dafd3b`, real WALL integration/registration
`76d546e5`. Shared registration binds design SHA256
`992c05601f03d1d9caad0421b041dbeaa05fa8621ae7e3020be42e3c9f5b41d0`, exact7,012,800sim-s.
The approved first dispatch was made only after CPU28/28 and real frozen checkpoint/WALL
integration PASS. Own cpuQ PID284174 exited; no running selector job, no physics or training.

Original root `/home/ubuntu/selmappo20261031/build_bc4c3aae72073204` remains immutable.
`training/ledger.jsonl`: first mechanics world START900, ENDactual0.0; source guard refused
before scenario construction/SUMO start. **900reserved remains charged; actual0.**
Ledger SHA256 `f94a2488879d79fb433e15a57e6d99710b62147d108eb6c2b19e772feda84253`.
`STOP.json`, original log, integration receipt, dispatch and exclusive task dispatch lock
are preserved. Local complete receipts: `docs/paper_results/selmappo/DISPATCH_*.json`,
`STOP_*.json`, `STAGE_*.json`. Production0, checkpoints0, no efficacy claim.

## Diagnosed repair
Patching `c4.seeds.check_case` alone missed `c4.env.check_case`, captured at import.
The source scenario module similarly captures `require_development_case`.
`tools/selmappo/guards.py` now scopes BOTH definitions and BOTH captured aliases,
admits only exact episode seed/demand/exposed role, restores all four on every exit.
Frozen-runtime CPU integration directly exercises these actual aliases with SUMO
start/step intercepted;29/29CPU tests +real checkpoint/WALL/alias gate PASS.

Repair stage `/home/ubuntu/selmappo20261031/build_914fecba3ffdf4cc`;
binding `914fecba3ffdf4cc975000a86f90db0317e26645b810f572fcfaa0160c03c1d2`.
**STAGED ONLY. No retry is armed or authorised.** Shared per-CAV hooks and block API
were also restored at S-SELMAPPO2's request; see `SELMAPPO2_RESPONSE.md`.

## MAIN acknowledgement 00:29 (2026-11-01)
Original design stops on any failed simulation and has no retry/refund reserve.
MAIN explicitly accepted SAME-CAP continuation: retain failed900, do NOT
repeat its seed, run only the already allocated unused second mechanics900, then
production6,996,600 and evaluation14,400 =>7,012,800charged (7,011,900actual if all finish).
Implementation inherits the exact hash-bound old ledger, refuses the failed seed, and
uses a separate exclusive continuation lock; the original dispatch/lock are preserved.
Only the unused second mechanics episode runs; any further failure stops without retry.
An additional successful replacement smoke without another amendment would exceed the
conservative cap by900. Budget advance approval alone does not authorise that retry.
Final169 evaluation still requires the registered16world S-C1SWITCH2 manifest.

## Second mechanics terminal stop (preserved)
Continuation commit `c082b233` bound source `999584a8cfacaf81`, CPU30/30 and real
frozen-controller/WALL/seed-alias gate PASS. Own cpuQ PID357398 ran ONLY seed18007999;
the original failed18007998 START/END was inherited unchanged, not simulated again.
Second episode reached900 actual sim-s, then failed in terminal bookkeeping:
the frozen `CavSumoEnv.step` closes `env.api` before returning `info['done']=True`.
Our wrapper queried `env.now`, then its WALL cleanup repeated the same closed-clock query.
This is NOT a successful mechanics result: final reward/safety receipt was not completed.
The ledger nevertheless closed correctly: **1,800charged /900actual**, zero production,
zero checkpoints; ownPID357398 exited. Both dispatch locks and all sources/logs remain.
Ledger SHA256 `be3bcd873b401cce95b14a9659f33bcc8de38823eb8a6ca5a830119ca13347ce`.
Receipts: `docs/paper_results/selmappo/{DISPATCH,STOP}_build_999584a8cfacaf81.json`.

CPU repair uses returned terminal `time_s`, skips unused terminal sensor collection,
keeps the final reward interval, and makes cleanup idempotent. CPU31/31 PASS incl a
600-tick closed-connection/final-reward/cleanup regression; real frozen controller,
WALL and seed-alias integration PASS, with SUMO start/step intercepted.
Immutable stage `build_68394a9568862929`, full binding
`68394a95688629292c307a20861c409d264a3c88104851b719caa5917d44e7a1`.
**CPU ONLY, NOT DISPATCHED.** No mechanics allocation remains; neither smoke seed may
run again. No automatic second continuation is implemented or armed.

MAIN acknowledgement needed under the registered stop rule: accept the terminal-only
mechanics failure after900actual and authorize production-only continuation on the
CPU-tested fix, retaining both smoke charges; no repeated/replacement mechanics.
Same upper cap:1,800 historical +6,996,600production +14,400reserved eval =7,012,800.
Alternatively MAIN must specify a budget/protocol amendment for another real mechanics gate.
C1SWITCH2's registered evaluation ownership is now confirmed: keep our14,400 reservation
unstarted and reuse its paired16world J600/J900 receipts; only final169 is eligible.

## MAIN acknowledgement 01:08 (2026-11-01; current)
REGISTRATION.md now explicitly acknowledges production-only continuation within7,012,800:
retain BOTH mechanics charges, run NO additional mechanics episode, then production only.
Fresh-session reconciliation: git HEAD0422aa41; own second trainer357398 is absent on cpuQ;
original/second logs and1,800charged/900actual ledger match the preserved stop above.
No fleet trainer appeared in cpuQ's process inventory at01:09+08:00.
Implement a separate one-use production lock and exact second-ledger inheritance before
launch. Both smoke seeds remain permanently consumed. No continuation has yet launched.
MAIN's priority rule: nice19, single-thread/CUDAhidden, pause new dispatch above0.8xcores;
cooperation and WM evaluation have priority, then C1SWITCH2, then selector training.
Evaluation stays peer-owned on registered PANEL_v2.json (16worlds969136..969151,
SHA2567c780c25237dde73a657937ab577b8e48f71766817f4363f3298982e90144827).

Production-only implementation now passes34/34CPU on cpuQ, including real optimizer,
exact-ledger/no-smoke replay, load admission, and remote-template construction checks;
real frozen checkpoints/WALL lifecycle/seed-alias integration also PASS with no physics.
Local32PASS/2torch-unavailableSKIP, RuffPASS. Stage`build_b3d13ccfae77de71`, fullbinding
`b3d13ccfae77de71cd8ff70d2ee444de514104855652968aa70965a54aae4aba`.
Dispatch revalidates ALL source hashes, both old locks/exitedPIDs and exact stopped
ledger/log, then acquires exclusiveFLEET_PRODUCTION_ONCE; no automatic retries.
No new SUMO at this CPU-only checkpoint. Receipt`STAGE_b3d13ccfae77de71.json`.

## Production-only dispatch (preserved)
After implementation92969232 and registration-receiptae38d65e, one explicit production
launch claimed`FLEET_PRODUCTION_ONCE.json`; cpuQ own trainer503239 is live.
Dispatch receipt`DISPATCH_build_b3d13ccfae77de71.json`; source remains immutable.
The new ledger inherited both old START/END pairs unchanged and started production
at18000000; first poll records10production reservations and a resource-priority pause.
No mechanics rerun, completed training update, final checkpoint or evaluation claim.
Original STOPs/locks/ledgers remain historical receipts; do not interpret them as
permission for another retry. Any new failure still stops with all charges retained.

## Production transport STOP (current; 2026-11-01 ~01:18+08:00)
Own503239 exited, no task descendants remain. First wave reserved41worlds18000000..18000040
before PyTorch tensor IPC failed (`received0items of ancdata`/BrokenProcessPool).
Each trajectory returned30separateold_logp tensors;32complete returns retain~960storageFDs
against cpuQ's softFDlimit1024. This is a transport failure, not a measured policy outcome.
33worlds have complete900sresult/actual receipts (32supervisor-closed +18000032);
8killed workers have no terminal actual:18000028,18000034..18000040.
**38,700charged retained =1,800historical +41x900; actual30,600..37,800, exact unknown.**
Ledger9open reservations is unchanged; one has independently proven900actual, eight do not.
LedgerSHA`497e0c292a1acb757f4bd733fa4a60b9ad7e2853e6b16d16e8fe50c6c570ec88`;
STOP_build_b3d13ccfae77de71.json binds all33result/actual receipts, log/failure/source hashes.
No completed PPO update, only initial000SHAbdd5af7f40971528d8ed6aefa8fba7353597a3d989234f9341eb2789dc3f8adc;
no final169/evaluation/learning-success claim. All locks and immutable sources remain.
No worker trajectories were persisted by the stopped implementation: even successful
physics cannot be silently replayed or treated as recoverable PPO training records.
CPU-only repair: inline serialized bytes in BOTH directions (no tensor storageFDs on
process queues), retain each full trajectory before its return; stress46x30CPU transport
and real optimizer before proposing another immutable stage. No new physics authorized.
MAIN acknowledgement/protocol amendment needed: same-cap proposal is skip ALL41consumed
worlds (including successful ones), first update ONLY5unused worlds18000041..18000045,
then original168x46updates18000046..18007773; final169only. Retain38,700historical
+7733x900future +14,400unstarted evaluation =7,012,800; no refund/replacements/mechanics.
This reduces gradient-bearing episodes by41 and must be disclosed/approved explicitly.

Transport repair is now CPU35/35 PASS on cpuQ (46episodes x30decisions through a real
spawn pool, bitwise model/logp preservation, bounded parentFDs, real worker wire schema,
durable trajectory/hash/no-overwrite and real1380-framePPO update). Genuine frozen
checkpoint/WALL lifecycle/seed aliases integration alsoPASS with SUMO intercepted.
Local32PASS/3torch-unavailableSKIP/RuffPASS. CPU-only immutable stage
`build_59a85b1215ade170`, binding
`59a85b1215ade17045f65ea36f22f3f3c418aa3521e9d62094d9190ca5eae123`;
receipt`STAGE_59a85b1215ade170.json`, newSUMO0, NOT DISPATCHED.
This fixes transport but does not implement/arm another continuation; current consumed
worlds/schedule/accounting require the specific MAIN amendment above first.

## MAIN acknowledgement 01:48 (fresh-session reconciliation)
REGISTRATION.md explicitly accepts skipping all41 consumed worlds, update1 on the
5 unused worlds18000041..18000045, then168x46, final169only, same7,012,800 cap.
Fresh remote inventory confirms all three prior fleet trainer PIDs absent and only
the original three exclusive locks present. cpuQ load2.38/32cores at inventory;
no continuation dispatched. Implement a fourth one-use lock and immutable inherited
history, retaining unknown actual bounds rather than inventing terminal debits.
The two untracked SELMAPPO2 request files are peer work and are preserved.

## MAIN01:48 dispatch STOP: history-reader schema failure (current)
Registered implementation `e6c24e73`/resource-wait `678b4548`; exact owned-history audit
repair `4200a7b8`. Pre-lock attempt on a6b stage ran no supervisor/reservation/physics.
Corrected immutable build `3351504cece84e4b` passed38/38CPU and genuine integration,
current registry/initial policy/RNG gates. Exclusive fourth lock was claimed ONCE.
cpuQ ownPID606965 then exited; no owned descendants remain. Disk had recovered to
2,601,025,536bytes at launch, so normal per-episode admission admitted the first5 worlds.

All five reservations18000041..18000045 failed in `PrechargedEpisode.reserve`:
the per-episode scan assumed every ledger row had `id`; the new HISTORY row does not.
All five tracebacks are at `runtime.py` budget.reserve BEFORE CavSumoEnv constructor,
reset or SUMO start; all five episode directories are empty. **New actual0, retained
new charge4,500; total43,200charged /30,600..37,800historical actual bounds.**
New ledger retains5open reservations and the byte-identical original open ledger.
New ledgerSHA `7ae8d3b572fa4ab22298b9f0d161790afc7de3bd0dd13ded8804134fcf36f0d4`.
Stop receipt binds all failures/log/initial000 (unchanged original hash);0updates,
no final169, no evaluation. All sources/stages/locks/charges remain, no retries/refunds.

CPU-only correction reads `row.get('id')`; regression now explicitly exercises the real
PrechargedEpisode reserve/finish against inherited HISTORY plus a fresh live reservation.
The previous inheritance test had exercised only supervisor Budget, missing this boundary.
No continuation implementation/automatic launch is authorized after this new stop.
MAIN must amend the now-exhausted first5 schedule; proposal in `MAIN_CONTINUATION_REQUEST.md`.

Repair CPU38/38 (0skips) and real frozen integration/current-registry/initial policy-RNG
gate PASS, newSUMO0. Immutable CPU-only stage `build_36c71820cd9e03d8`, full binding
`36c71820cd9e03d85efb894f899eb33e94ad6e5ad084a06106a03ef9c558c55e`.
NOT DISPATCHED; existing fourth task lock prevents relaunch. Source observations,
controllers, hand-over, reward and PPO optimizer are unchanged by this row-reader fix.
