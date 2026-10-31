# MAIN23:04 recovery checkpoint

- Audit commit: `5681dfa1`; CPU budget-admission commit: `d3d5c59f`.
- Registered design SHA256:
  `d9567959e929a196d19c64d6acdf9617598b021252a381bdfd0977e791e6867d`
  (73 lines).
- Shared `REGISTRATION.md` includes both `S-C1SWITCH MAIN23:04 INTERRUPTION
  RECONCILED / SAME-PLAN RESTART BUDGET-BLOCKED` and `S-C1SWITCH RECOVERY
  ADMISSION CPU PASS / AWAIT MAIN DECISION` receipts.
- Exact existing debit227310sim-s;10complete episodes+231complete branches.
- Standard recovery projection501960sim-s: Step047010,
  WM188010including superseded12450, BB180540, random14400 and five fixed
  controllers14400each. This is a diagnostic projection, not an approval.
- BB GPU remaining0 with12uncommitted picks; WM remaining4738.271059s with
  143uncommitted picks. Original per-model hard caps remain in force.
- CPU tests12PASS. Frozen v7 scientific source diff from pre-resume commit
  `b7ed5cff` is empty. No new SUMO/GPU compute or scientific redispatch.
- All audited remote own workers/services absent; local relay13556 exited via
  cooperative STOP and retained closed-lock receipt.
- **Pending:** MAIN budget/protocol decision before an operational recovery
  implementation and dispatch. Current preflight is explicitly non-dispatching.
