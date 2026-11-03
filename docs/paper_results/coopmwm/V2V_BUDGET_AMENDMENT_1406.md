# MAIN14:06 three-arm quote — current-bank metadata checked

The latest authority retains A-v2v/B-v2v/R-v2v from the common original MAPPO
u0 through u24. It supersedes the older A/B-only budget fallback.

## Exact current-slate workload

Receipt `V2V_BUDGET_SCENARIOS_1406_LIVE_V5.json`, SHA256
`795b482d175b02fd59ea553c939af5d6fb4cd497c69925897d0dc98a4d97b884`:

- Training: 3 ×24 ×24 ×1200 = **2,073,600 sim-s**.
- u8 episode tests, all three arms: **345,600**.
- u24 episode tests, all three arms: **345,600**.
- B-v2v matched lesson at u6/u12/u18/u24, frozen eight-entry slate: **21,840**.
- B-v2v isolated all/solo/background plus conservatively fresh reference at those
  same four updates, all five currently admitted cooperative states: **81,720**.
- R-v2v isolated all/solo/background plus conservatively fresh reference at
  u12/u24, all eleven currently admitted cooperative states: **103,480**.
- A-v2v extra diagnostics and V2V final paired branches: **0**.

**Total: 2,971,840 sim-s**; proposed total cap **17,471,840**, below17.5M by
**28,160**. All three training arms and both episode tests fit. No fallback
drop or baseline-reuse qualification is needed to fit this current slate.
The conditional SHA-qualified reference-reuse scenario is2,941,160; it is
retained as conditional and is not assumed in the selected quote.

## Integrity and scope

- Live read-only TRAIN metadata checks both complete cooperative sets against
  confirmation IDs and current BANK_QUALITY counts: B5/R11, missing IDs0/0.
  Fixed lesson-list SHAs and each cooperative verification receipt SHA are pinned.
- The direct u11 diagnostic-report quote refused incomplete cooperative metadata;
  failed states are never silently omitted. The live quote instead uses their
  original qualified bank decision metadata, not incomplete diagnostic outputs.
- Every diagnostic branch is charged through its full prefix end, not just its
  post-intervention window. Fresh references are charged once per state/update.
- Bank growth is not bounded by this projection. At each scheduled diagnostic,
  the launcher must prequote the complete then-current slate against its shared
  phase and register any permitted transfer before physics. Neither a partial
  branch set nor an implicit ceiling increase is authorised by this quote.
- Legacy1345 scenarios and preparation receipts remain unchanged. V3 and V4
  snapshots are retained; V5 adds explicit current-bank count completeness checks.

Ten CPU tests PASS,0skips,0.002s. Live metadata check invokes no SUMO/model
inference and writes nothing remote. No budget allocation, cap amendment,
simulation reservation or V2V dispatch occurred here. The existing runner owner
must register the exact scoped allocation/SHA, qualify Linux/rehearsal gates,
and dispatch its one durable watcher under MAIN13:45/14:06.
