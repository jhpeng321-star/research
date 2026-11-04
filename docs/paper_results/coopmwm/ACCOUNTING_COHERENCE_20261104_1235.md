# Accounting coherence check — Nov4, 12:35 UTC+8

Read-only evidence: `ACCOUNTING_COHERENCE_1793766839760940300.json`.

The active bank bindinga91884b6 reflects MAIN's registered840,000sim-s transfer:
bank_verification_ext4,940,000allocated/4,278,910charged/661,090remaining;
sumo_verification1,960,000allocated/1,911,880charged/48,120remaining. The immutable
V2V runtime06e88ecd retains the previous4.1M/2.8M values. This produces a false
negative bank balance in a monitor that uses only the runtime's old phase table.
Total12,995,410charged/cap17,471,840, including811,640retained superseded.

MAIN/practice operator notified: the811,640superseded allocation is fully spent,
although it is not a CHARGES.jsonl row. `v2v_live.accounting` subtracts that fixed
retained expense from the global ledger cap. Its apparent ledger-only balance
must not be treated as practice funding. Both current bindings retain the full
allocation. Any practice funding reconciliation remains with MAIN/the delegate.

No allocation/runtime change, worker dispatch, new simulation or signal by this
check. The future frozen-lesson phase gap remains13,200sim-s (u12 still fits).

## Reporting repair verified
The existing `_sol/coopmwm_public_progress.py` read-only helper now takes phase
allocations from the registered live bank binding, verifies its amendment against
AUTHORITY.md, records the binding SHA and runtime phase delta, and explicitly
counts the fixed811,640retained expense once. It refuses an underfunded retained
allocation or cap mismatch. It does not install an accounting function or mutate
the production runtime. Live verification:
`PUBLIC_PROGRESS_1793766958836422500.json` reports13,080,120charged and **no
negative live phases**, with amendment276799ab recorded. Monitor4 remains healthy.

## Subsequent registry resolution (read at12:41)
MAIN's13:10registry correction explicitly preserves the spent superseded charge
and names`paired_branch` as the345,600sim-s practice donor. The launch operator's
13:05receipt confirms the erroneous donor was never applied. Practice deployment
is still operator-owned; no new runtime was adopted by this check. After that
runtime changes, compare both its phase table and the bank binding again before
reporting balances: today's helper verification covers runtime06e88ecd and bank
bindinga91884b6, not an unobserved future practice authority.
