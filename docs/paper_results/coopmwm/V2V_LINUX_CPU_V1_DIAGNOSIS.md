# V2V Linux qualification V1 — retained failure, environment correction only

Immutable source `/home/ubuntu/coopmwm20261103_v2vcpu1`, owned PID3871752,
completed CPU-only in120.267s:711 passed,52 subtests passed,1 failed,1 inherited
deselection,0 skips,0 sim-s. All V2V observation/model/bank/end-to-end/runner tests
passed. Release is **not qualified** by this failed receipt.

Failure: `test_nullfix_cpu.py::Gate::test_missing_net_leaves_speeds_unresolvable`.
The launcher bound `nullfix_geometry.install(..., net_path=FROZEN_NETWORK)` for the
test process. That deliberately makes lane speeds resolvable despite the test's
empty `RESEARCH_DENSE_ROOT`; it contradicts this negative fixture's setup.
The live-qualified C1 CPU launcher installs only the additive geometry helper,
allowing the fixture to supply/miss its own network. This is a qualification
environment error, not a V2V/plant/numeric change.

V2 uses a **new immutable source root**, same merged code and complete test list,
and the inherited helper-only install. No failure is excluded or test weakened;
the complete suite is rerun once. Original V1 dispatch/log/result/source retained.
No live training/GPU root or binding changes; no SUMO/inference or reservation.
