# PLANFIX preserve live WM work under the unchanged pace guard — 2026-11-05

Owner constraint: never slow clean2, own PIDs only. At23:38 the actual latest
R13 interval1184.009515s exceeded the prospectively bound1167.460120s maximum;
A14/B13 remained within their original limits. This is telemetry, not a causal
attribution to the one-core WM process.

At23:41:16 operator independently bound exact own PID364064/start8210497,
boot2c19986e-dd5e-4de9-95ca-3212b1ea0568, UID/argv/source binding, opened a Linux
pidfd and SIGSTOPped ONLY that WM incarnation. Kernel stateT verified. No
training process was signalled and no source/model/bank/ledger was changed.
Current partial full-gate computation/RNG/model remains in RAM in that SAME PID;
no completed or partial computation needs to be restarted.

Prospectively stage ONE separate CPU-resident resource supervisor, no model or
simulator process. Its only signal target is the above exact own WM incarnation,
guarded by pidfd/start/boot/UID/argv. It shares affinity47/nice19, polls public
receipts every60s, at most21600s. It may resume ONLY after a NEW completed
interval for each previously over-limit arm is within the SAME original limits,
with other current intervals in bounds, original clean2 owners/config active,
no clean2/own STOP or FAULT. It never changes/re-estimates the baseline. It pauses
only this own process on another over-limit/health condition. Requests/results
are fsynced to own append-only control events; source and controller owner are
SHA-bound in root AUTHORITY before controller dispatch. No name-based signal,
training-control change, model restart/duplicate, gate adoption, numerical
source/criterion change, new search or SUMO route is authorized.

Original CPUv6 still requires all8same-process exact gate receipts before its
ONE search. Gate elapsed wall times include any resource pause; report them as
such, not as uninterrupted throughput. Source/model/future seeds/populations/K/
horizons and physics admission criteria remain unchanged. If source/identity
fails, worker exits or a failure receipt appears, the controller stops without
retry/replacement. It cannot expand or admit plans or apply a bank switch.

Phase33,600/600,000/protected566,400 unchanged. Read-only23:38 actual global
charged2765040/5000000; disjoint allocation bound including full protected phase
remains5000000. Prepared switch remains disabled, actual update indexUNSET for
MAIN/owner; no native verification is released before actual speed/gate proofs
and the separate CPU SUMO amendment.
