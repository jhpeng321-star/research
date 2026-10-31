# MAIN01:48 exact continuation registration (before dispatch)

Implementation `e6c24e73`, resource-wait supervisor `678b4548`.
Immutable cpuQ stage `build_a6b8109d8bd55712`, binding
`a6b8109d8bd557126be5b3fc54872d6fd67ad81b789cb37f0a7bbb5dd8a6da09`.
CPU37/37, real frozen checkpoint/WALL/alias integration and actual initial policy/RNG
parity passed with SUMO intercepted. Ruff and diff checks passed.

Shared REGISTRATION.md now records the MAIN01:48-authorized exact continuation:
38,700 historical + 7,733 x 900 future production + 14,400 unstarted peer-owned
evaluation reservation = **7,012,800 sim-s**. All41 consumed production seeds and both
mechanics seeds are skipped; first update uses5 unused worlds, then168x46, final169only.
Thus41 fewer episodes contribute gradients. Unknown historical actual stays bounded
30,600..37,800; original open ledger is preserved byte-for-byte, without invented ENDs.

cpuQ has CPU/memory headroom but free disk2,119,102,464bytes is below2GiB. MAIN capacity
notice is in the shared registry. A single exclusive resource-wait supervisor can be
started, with no physics/reservation while admission fails. Every episode retains the
same load/memory/disk checks before precharge; nice19, CUDAhidden, <=12single-thread
workers, no retries, 24h total wall stop. Prior three locks/PIDs/stopped receipts checked.
Evaluation remains owned by C1SWITCH2. No final checkpoint exists yet.

Pre-dispatch registration audit caught historical *owned disclosures* as conflicts:
the original audit exempted only the first design line. This stopped BEFORE the
new exclusive lock, supervisor, reservation or physics. Remote inventory confirms
only the three old locks remain. Exact full-line owned-history exemptions now retain
conflict detection for any changed/extended line; no broad task-prefix exemption.
The actual shared registry audit is now also part of the frozen CPU integration gate.

Corrected immutable stage for dispatch: `build_3351504cece84e4b`, binding
`3351504cece84e4b5c289689dd65350dfe784dd0dfbbb3c93fa16a681df59235`.
CPU38/38 and real integration/registry/initial-model-RNG gates PASS; new SUMO0.
All preceding stages remain immutable and undispatched. Budget/schedule unchanged.
