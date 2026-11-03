# BANK_FILTER_PREVIEW — statistical bank filter (FIX 2, `bank_filter='statistical'`)

Read-only preview of the statistical bank filter against the live verified-plan bank.
Written by the c4/bank-stat-filter workstream; **no live artifact was modified** — everything
below was produced by reading published receipts only.

## The rule being previewed

Implemented in `tools/coopmwm/bank_filter.py`, applied by `replay_bank.bank_pairs(..., bank_filter='statistical')`
(new binding key `bank_filter`, **default off = byte-identical old behaviour**):

1. **Imitation admission**: an admitted entry feeds imitation only if its replicated
   advantage is significantly > 0 —
   - aggregate gate: every recorded replicate advantage (first seed, confirmation
     replication) must be > 0 (the existing rule, re-checked);
   - primary significance: paired per-vehicle road-speed-deficit differences of the two
     branches (policy continues vs chosen plan) are reconstructed from the recorded
     `observable.jsonl.gz` traces over each replicate's window, pooled across the replicate
     seeds; the one-sided 95% bootstrap lower bound of the mean (10000 resamples, 5th
     percentile, `numpy.default_rng(20261101)`) must be > 0;
   - documented fallback (only when no replicate has a readable trace): the same one-sided
     bound over the >= 2 replicate aggregates must be > 0 (with two replicates this equals
     "every replicate > 0"). No hand-set behavioural thresholds anywhere.
2. **Cooperative label**: cooperative only if the synergy is significantly > 0 by the same
   method — group-minus-best-solo paired per-vehicle bound > 0 (fallback: recorded aggregate
   gain > 0 when no trace pair is readable), plus, where every solo branch completed, the
   recorded group-minus-sum-of-solos synergy `C_G = group − Σ solo` (the registered credit
   contrast, `tools/coopfree/credit.py`) must be > 0. A physical sum-of-solos branch does not
   exist, so that contrast is only observable as an aggregate (disclosed in the report).
3. **Reporting**: kept/dropped entries with reasons and bounds under
   `summary['bank_filter']`; the 50% cooperative oversampling then applies to the
   **filtered** cooperative set (weights unchanged).

## Local reachability of the live bank's per-entry data

| source | reachable? | contents |
|---|---|---|
| `E:/research-dense/docs/paper_results/coopmwm` | **absent** | directory does not exist on this host |
| `E:/agent-work/c4-coopnative/docs/paper_results/coopmwm/warm_live` | receipts only | `B_UPDATE_001..003` only (pre-confirmation `weights` format, no `entries` list); **`B_UPDATE_016` is not published** |
| `.../coopmwm/B_REPLAY_LS_LAYOUT_1793661280011983300.json` | **yes** | a real per-entry bank summary captured by `replay_ops.layout_check` against the live run root (63 decisions, 8 admitted u3-label entries) |
| branch `observable.jsonl.gz` traces, `confirmations/`, `confirm_replication/`, `confirm_solo/` | **no** | never published; they stay on the live cpuQ run root |

`BANK_QUALITY.json`, the confirmation records and the branch traces are therefore **not
reachable locally**, so the full per-vehicle rule cannot be executed here. What **can** be
applied read-only is the aggregate portion of the rule, using the published layout snapshot:
each entry's replication advantage is recoverable from the two published aggregates as
`repl = 2·mean_replicated_advantage − grade`.

## Aggregate-fallback preview on the published live snapshot (8 admitted entries)

Derived from `B_REPLAY_LS_LAYOUT_1793661280011983300.json` (u3 LS-COLLECT labels, live root):

| entry | grade (veh-s) | repl (derived, veh-s) | aggregate 95% LB | aggregate-fallback verdict | recorded label | recorded gain (veh-s) |
|---|---:|---:|---:|---|---|---:|
| 986016 t0300 (g3) | +522.037 | +343.462 | +343.462 | keep | non-coop | −38.000 |
| 986019 t0300 (g4) | +0.252 | +0.0016 | +0.0016 | keep (fallback only) | non-coop | −0.011 |
| 986048 t0300 (g2) | +958.678 | +60.656 | +60.656 | keep | **cooperative** | +958.678 |
| 986052 t0300 (g2) | +0.0138 | +0.0063 | +0.0063 | keep (fallback only) | **cooperative** | **+0.0005** |
| 986066 t0300 (g6) | +5580.842 | +1066.809 | +1066.809 | keep | **cooperative** | +4430.099 |
| 986048 t0600 (g2) | +0.000068 | +0.0000177 | +0.0000177 | keep (fallback only) | non-coop | −0.0000033 |
| 986056 t0300 (g3) | +0.0000111 | +0.0000099 | +0.0000099 | keep (fallback only) | non-coop | −0.0000022 |
| 986057 t0300 (g6) | +0.0134 | +0.0609 | +0.0134 | keep | non-coop | −0.00016 |

Reading (honest scope): with **aggregates only**, every entry passes the documented fallback
(both replicates > 0, two-point bootstrap bound > 0) — the fallback is deliberately weak
because it is all the data supports. The four near-zero rows (986019, 986052, 986048 t0600,
986056) are precisely the dilution the filter targets: under the **primary per-vehicle rule**
they are expected to drop once their branch traces are supplied (a +1e-5..+1e-2 veh-s
aggregate is invisible in per-vehicle paired deficits, so the one-sided bound cannot stay
above 0) — expected from the rule, not proven here. The recorded `cooperative=True` label of
**986052 (+0.0005 veh-s gain)** is exactly the ~zero-synergy case the statistical label
removes (bound ≤ 0 and/or `C_G ≤ 0`); 986048 t0300 and 986066 have large recorded gains and
are expected to survive with their traces. The same treatment applies to the later u8-era
entry (u008, +40 veh-s) and the six ~0-grade entries named in the problem statement:
`B_UPDATE_016` itself is not published locally, so its 12 entries could not be enumerated
here.

## What the live agent must pass for the full rule

Run read-only on the live root (`bank_pairs` writes nothing when `upto_unix` is used for
preview; the filter only reads):

1. per entry, the first-seed decision receipt (`replay_bank/<arm>/sumo_verification/
   verifications/<vid>.json`): `window`, `branches[0]` (policy) and
   `branches[chosen+1]` (chosen) names, `grade_veh_s`;
2. the two first-seed branch traces
   `replay_bank/<arm>/sumo_verification/branches/<vid>/<branch>/observable.jsonl.gz`;
3. the confirmation record (`replay_bank/<arm>/confirmations/<vid>.json`):
   `replication.verification_id`, `replication.window`,
   `replication_advantage_veh_s`, `cooperation` block (`cooperation_gain_veh_s`,
   `best_solo_member`, `solo[*].verification_id`, `solo_advantages_veh_s`, `complete`);
4. the replication branch receipt + traces under
   `replay_bank/<arm>/confirm_replication/{verifications,branches}`;
5. the best-solo branch receipt + traces under
   `replay_bank/<arm>/confirm_solo/{verifications,branches}`.

With those, `replay_bank.bank_pairs(root, arm, seed, bank_filter='statistical')` reproduces
the preview above with the per-vehicle bounds filled in and emits the kept/dropped table
(entry id, decision, reason, bound, `n_paired`, cooperative old/new) under
`summary['bank_filter']`. Missing traces degrade entry-by-entry to the documented aggregate
fallback (never imputed), so a partial pass is still informative.

## Configuration / call sites

- binding key: `bank_filter` in `WARM_*_RUNTIME_BINDING` (values: absent/`None`/`False` = off;
  `'statistical'` = the registered rule; anything else raises `ValueError`).
- `replay_bank.bank_pairs(..., bank_filter=...)` — trainer-side admission + cooperative label
  (`tools/coopmwm/replay_bank.py`).
- `warm_runner.run_arm` — plumbs `b.get('bank_filter')` into the B/R update bank pairs call
  (`tools/coopmwm/warm_runner.py`, the producer of `B_UPDATE_*.json` `exact_imitation`).
- CPU tests: `tools/coopmwm/test_bank_filter_cpu.py` (7 tests: zero-gain dropped, noisy
  positive dropped, clear positive kept, ~zero synergy not cooperative, label driven by
  traces not gain, unreadable-trace degradation, off = byte-identical).
