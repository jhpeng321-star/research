# T2C r2 fix-round receipts

## Step 1 — recovery and contract inspection

Recovered HEAD: `347b2e9a` on `c4/rerun-tracks`. Initial status: deleted
`T2C_READY.json`, untracked `REVIEW_T2C_r2.md` and `T2C_READY.blocked2.json`.
These review artifacts and all previous work are preserved. Read the full shared
spec, r2 independent review, previous readiness and r1 fix receipts first.
No matching T2C pytest/runtime process was running; no dispatch was started.

The single r2 blocker is confirmed against in-memory-readable track-base source:
root authority and ledger admission require the Track 2 criterion unconditionally;
ledger CAP no longer subtracts the legacy superseded charge; preflight requires
Track 2 quote/config/registration/criterion even for legacy preparation.

Fix boundary: explicit default-false `track2` on root/ledger/charge/preflight,
configuration-driven selection in `bind_ledger`, and explicit ON forwarding from
the Track 2 runtime/CLI. Legacy preflight retains its approval/hash/staging receipt
contract, while ON retains all complete-registration and authority checks.
