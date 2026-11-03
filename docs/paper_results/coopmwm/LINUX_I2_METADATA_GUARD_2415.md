# MAIN00:00 pipelined Linux qualification — I2 result and harness correction

Pinned candidate:`ac8e1f318b70b1a9faf7b99c13589696f76601e2` (launch-final).
Every explicit overlay was read from that commit's git blobs, not mutable WIP.
Immutable source:`/home/ubuntu/coopmwm20261103_intentcpu_i2`.
Owner952101/start73630658 is complete; never redispatch I2.

Result: **932 passed +52 subtests passed,2 failed**, one inherited real-SUMO
test deselected,0 skips,139.323s,zero simulation. The two failures are the
inherited blanket subprocess guard rejecting newly added CPU metadata tests:

- cooperation-search CLI `preflight --sha` against a temporary descriptor bank;
- gate identity test's temporary git fixture init/add/commit/rev-parse/diff.

No model, policy, plan, simulator or optimizer failure was observed in those
two cases; the operations were refused before dispatch. Keep the immutable
failure evidence. A narrow harness allowlist now admits only these exact
metadata command shapes, with temporary paths, no shell/executable overrides,
and known git verbs. All physics/inference/trainer commands remain blocked.
Three local boundary tests PASS. This touches only ls_cpu_test and its tests,
not any integrator-owned learner/audit-fix module.

Next: qualify the corrected harness in a NEW generation, then run only the
final source delta when READY lands. The final audit and real-bank gate remain
mandatory. No gate execution or V2V launch was dispatched.

Prospective gate inputs are prepared through `v2v_runner.prepare_intent` in
`/home/ubuntu/coopmwm20261104_gate_inputs1`:48-wide causal intent,per_seed_v1,
sources JSON and common-u0 SHA. They are explicitly not launch-authorised;
final runtime/source pin must be finalised before the real gate. Receipt
`GATE_INPUTS_PREPARED_1793722209028900300.json`.
