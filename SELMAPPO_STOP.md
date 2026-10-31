# Fleet PPO selector — stopped before physics; repair staged

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

## MAIN decision needed
Original design stops on any failed simulation and has no retry/refund reserve.
Proposed SAME-CAP continuation (not implemented/dispatched): retain failed900, do NOT
repeat its seed, run only the already allocated unused second mechanics900, then
production6,996,600 and evaluation14,400 =>7,012,800charged (7,011,900actual if all finish).
MAIN must explicitly accept one successful mechanics episode and acknowledge continuation.
An additional successful replacement smoke without another amendment would exceed the
conservative cap by900. Budget advance approval alone does not authorise that retry.
Final169 evaluation still requires the registered16world S-C1SWITCH2 manifest.
