# Fleet PPO controller-selector training status

Observed UTC: 2026-11-01T12:34:34.486780+00:00
- cpuQ supervisor: 645235; alive=False; nice=None.
- Genuine completed PPO updates: 169/169; final169 eligible=True.
- New production: 7728 complete, 0 outstanding; failures=0.
- Charged: 6,998,400 sim-s, including43,200historical; exact total cap7,012,800.
- Actual sim-s bounds: 6,985,800..6,993,000 (old unknown actuals retained).
- MAIN02:08 schedule23+23+167x46; 46fewer gradient episodes than original; final169only.
- Every reservation yields above0.8cores or below8GiB RAM/2GiB disk; nice19, one thread, CUDA hidden.
- Training observations are not efficacy results. Evaluation remains exclusively C1SWITCH2-owned.
- Local14,400evaluation reservation stays UNSTARTED; peer has the verified frozen final169.
- Immutable build: `build_9bb7869d8f590818`; binding `9bb7869d8f590818c46980a16a63be00f72e00d3491dec95f903649dfa7bf790`.
- Receipts: `docs/paper_results/selmappo/`; protocol/design and previous stops: `HANDOFF.md`.

The original five-minute observer completed its final169 handoff and exited normally.
Fresh-session reconciliation verified completion. The existing noon supervisor published
its final heartbeat and exited at its deadline; both local PIDs are now absent.
No fleet training processes remain, and no restart is required.

## Evaluation handoff reconciliation
- C1SWITCH2 receipt verified Unix1793536225.8118775 records fleet evaluation COMPLETE:
  16worlds/480decisions and96six-way oracle branches across the original host partitions.
- Evaluation charged=closed actual79,800sim-s, owned by the peer's evaluation budget.
- Shared final169 handoff remains byte-identical; genuine final checkpoint and all immutable
  staged source hashes, update schedule, ledger sets and finite policy validation PASS.
- WM/BB/per-CAV comparison is still incomplete in the peer's status; no primary efficacy
  result is inferred here. Evaluation analysis remains C1SWITCH2-owned.
- Fresh reconciliation: `selmappo/RESUME_20261101_2034.json`; fresh training receipt committed369a294e.
