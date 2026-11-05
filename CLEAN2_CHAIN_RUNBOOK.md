# cpuV clean2 detached launch chain

Owner authority: MAIN 15:23 `restart`, MAIN 18:04 immediate priority, and
MAIN 18:19 laptop-independent chain. Clean1 stays stopped at A9/B9/R9.

## Exact launch source

- Runtime commit: `285663a103d18fd33e9137db14fba33b62423a23`.
- Source: `/home/ubuntu/coopmwm20261105_clean2_src_285663a1`.
- All 538 packaged files are SHA-verified, including `ops/clean2_chain.py`.
- Full native CPU qualification control:
  `/home/ubuntu/cpuv_setup_20261105/clean2_tests_285663a1`.
- Chain control:
  `/home/ubuntu/cpuv_setup_20261105/clean2_chain_285663a1`.
- Experiment root: `/home/ubuntu/coopmwm20261105_clean2`.

## Admission and execution

The one-use dispatcher checks the completed exact-source Linux qualification,
every source SHA, absence of an existing clean2 root and absence of a previous
chain dispatch. It starts one `start_new_session=True` process with nice19 and
the qualified CPU/MKL environment. There is no SSH command inside the chain.

The chain performs zero-physics preparation and actual-bank preflight, then the
existing end-to-end smoke, existing single full-window safety gate, and detached
A/B/R u0 training owners plus the P3 monitor. Final configuration and sole ledger
must be prepared before smoke to preserve the existing exact-config receipts and
append-only accounting. Training is never started before both passes.

Every smoke check remains in `launch_qualification.smoke` and
`clean2_receipts.validate_smoke`: actual physical command changes, proposal and
infeasible-rate receipts, usable verified practice/trigger breadth, measured
close-call reward rows, both actual F5 report quantities, remaining registered
mechanisms, augmentation non-use, and complete replicated dry grids. The gate
requires complete grading and exactly zero collisions/teleports. The chain
also verifies all smoke/fix/reporting/gate SHA and configuration bindings.

Any exception or failed receipt writes a FAILED transition and durable STOP.
There is **no automatic retry**, environment void, stop clearance, partial-launch
retry, or adoption of an already attempted qualification. Preserve evidence and
diagnose before any further authorized action. Never rerun the dispatcher to
resolve uncertainty: inspect its authoritative dispatch and status first.

## Receipts and supervision

`CHAIN_STATUS.json` is atomically updated in control and experiment root. Every
transition also has an immutable `STATUS_NNN_STEP_STATE.json`. Control contains
`CHAIN_DISPATCH.json`, `CHAIN_STARTED.json`, `AUTHORITY.json`, `CHAIN.log`,
`smoke.log`, `gate.log`, and subprocess exit receipts. Once preparation completes,
`clean2/chain` links to this control, making all logs accessible from the root.

`LAUNCHED.json` binds commit, configuration/runtime SHA, smoke/gate, root, start
time and each role's PID/start_ticks. A partial launch writes evidence and STOP.
The chain stays alive as a public-receipt supervisor after dispatch, checks owner
identities and safety/budget, and writes five-line `clean2/CLEAN2_STATUS.md`
summaries plus immutable `MONITORED_uNNN.json` receipts. P3 continues all original
learning receipts, u2/u4 probes, and N5 u6/u12/u24 tests. Any hard safety event
pauses; A retains its full original rule and B/R composites remain report-only.

The remote summaries survive laptop departure. After reconnection synchronize
them to `E:/research-dense/runs/active/C4-V7FT-20261017/CLEAN1_STATUS.md`, prefixed
`CLEAN2`, and append the launch registration from `LAUNCHED.json`.

## Budget and optional bank handoff

`CLEAN2_BUDGET.json` is computed before physics using all clean1 ledger rows,
960 parity, and the entire unspent approved plan_fix reservation. Existing
plan_fix rows are counted exactly once. Complete native branch manifests must
fit the disjoint remainder within 5,000,000 sim-s. Smoke hard cap is 20,000.
The chain never shortens a window or changes the registered u24 synergy criterion.

Optional handoff is bound prospectively to
`clean1/plan_fix/PLAN_FIX_READY.json`, even while absent; missing means current
bank only. Accepted F3 readers discover it when published and still require real
native verification. Launch never waits for plan_fix or variety collection.

## Remote access

Use the accepted F6 `Transport` from WSL Ubuntu-22.04, same uid/shared
`/tmp/c4-cpuv-1000/master.sock`, serialized bounded admission. No direct native
Windows SSH, forced master closure, daemon configuration change or command
replay. See `CLEAN1_LAUNCH.md` for stale-reservation operator recovery; retain
shared master, socket and last-attempt state and prove all session processes
exited before archiving a stale reservation. Signal only explicitly owned PIDs.
