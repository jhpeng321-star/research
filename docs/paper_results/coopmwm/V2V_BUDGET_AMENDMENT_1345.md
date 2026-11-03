# V2V quote amendment — MAIN13:43 / 13:45

Supersedes the **start-point and workload assumptions** of
`V2V_BUDGET_PREPARATION_1332.json`; preserves that historical receipt unchanged.
All V2V arms now start from the same original MAPPO u0 with zero-init V2V,
then train updates 1–24. Maximum additional authority is 3,000,000 sim-s.

The metadata-only quote helper has six passing local CPU tests. Inputs are
the existing public TRAIN u10 lesson/cooperation reports, SHA-pinned in
`V2V_BUDGET_SCENARIOS_1345_V2.json` (receipt SHA256
`f9b406aa0cc846b6fbced1c7ed72b04833839850982720508dcdae08d05e82e8`).
No outcome-based training choice, simulation, inference or reservation.

- Three-arm fixed training plus u8/u24 episode tests: **2,764,800 sim-s**.
- With the current diagnostic slates and conservative fresh reference branches:
  **5,268,000 sim-s**, above the ceiling. MAIN's A/B-only fallback is required.
- A/B-only fixed training plus the same episode tests: **1,843,200 sim-s**.
- A/B-only with current slates and fresh references: **3,085,920 sim-s**,
  still **85,920** above the ceiling.
- A/B-only with **qualified reuse of the same existing baseline branches**:
  **2,921,760 sim-s**, within the ceiling by **78,240 sim-s**. The implied cap
  would be **17,421,760 sim-s** only after that exact allocation is registered.
  This is conditional: state/window/prefix/CRN and baseline policy identity must
  match by SHA. If that gate fails, stop before spending; no assumed adoption.

Diagnostic-scope assumptions are explicit: A is evaluated on B's frozen states
but never imitates B; lessons every update; isolated group/solo/background checks
with one reference (or its qualified adoption). These figures quote episode-level
u8/u24 tests. Adding final A/B paired branches would add **691,200 sim-s** and
exceed the ceiling; that additional V2V test scope needs MAIN's decision. The
original non-V2V paired-branch allocation and plan remain in force.

**Not a full-run upper bound or a launch certificate.** This projects the current
cooperative slate across 24 updates. Future bank growth and any new verification,
confirmation, failed/replacement or MAPPO-reference-failure spending need guarded
phase room before physics. Neither a phase transfer nor a cap raise has been
performed by this quote step. The runner owner/integrator must resolve baseline
reuse and the actual diagnostic slate before registration and durable launch.
