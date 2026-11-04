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
