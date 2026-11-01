# Fleet PPO controller-selector training status

Observed UTC: 2026-11-01T03:39:13.320648+00:00
- cpuQ supervisor: 645235; alive=False; nice=None.
- Genuine completed PPO updates: 169/169; final169 eligible=True.
- New production: 7728 complete, 0 outstanding; failures=0.
- Charged: 6,998,400 sim-s, including43,200historical; exact total cap7,012,800.
- Actual sim-s bounds: 6,985,800..6,993,000 (old unknown actuals retained).
- MAIN02:08 schedule23+23+167x46; 46fewer gradient episodes than original; final169only.
- Every reservation yields above0.8cores or below8GiB RAM/2GiB disk; nice19, one thread, CUDA hidden.
- Training observations are not efficacy results. Evaluation remains exclusively C1SWITCH2-owned.
- Local14,400evaluation reservation stays UNSTARTED; peer receives frozen final169 only.
- Immutable build: `build_9bb7869d8f590818`; binding `9bb7869d8f590818c46980a16a63be00f72e00d3491dec95f903649dfa7bf790`.
- Receipts: `docs/paper_results/selmappo/`; protocol/design and previous stops: `HANDOFF.md`.

The original five-minute observer completed its final169 handoff and exited normally.
Fresh-session reconciliation verified completion; the existing noon supervisor remains
active. No fleet training processes remain, and no restart is required.
