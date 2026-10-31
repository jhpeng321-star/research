# Fleet PPO controller-selector training status

Observed UTC: 2026-10-31T20:25:21.432108+00:00
- cpuQ supervisor: 645235; alive=True; nice=19.
- Genuine completed PPO updates: 22/169; final169 eligible=False.
- New production: 976 complete, 9 outstanding; failures=0.
- Charged: 929,700 sim-s, including43,200historical; exact total cap7,012,800.
- Actual sim-s bounds: 909,000..924,300 (old unknown actuals retained).
- MAIN02:08 schedule23+23+167x46; 46fewer gradient episodes than original; final169only.
- Every reservation yields above0.8cores or below8GiB RAM/2GiB disk; nice19, one thread, CUDA hidden.
- Training observations are not efficacy results. Evaluation remains exclusively C1SWITCH2-owned.
- Local14,400evaluation reservation stays UNSTARTED; peer receives frozen final169 only.
- Immutable build: `build_9bb7869d8f590818`; binding `9bb7869d8f590818c46980a16a63be00f72e00d3491dec95f903649dfa7bf790`.
- Receipts: `docs/paper_results/selmappo/`; protocol/design and previous stops: `HANDOFF.md`.

This status is maintained by one read-only five-minute observer. It cannot launch, retry,
resume or stop training/evaluation and does not signal any process.
