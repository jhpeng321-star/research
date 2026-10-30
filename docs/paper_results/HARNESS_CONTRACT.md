# HARNESS_CONTRACT — WMFIX source-clock and panel comparability (2026-10-31)

Reviewer blocker (WMFIX_REVIEW.md claim 4): the recorded labels sum the LEFT
boundary of the post-service entrance-queue samples, q_0..q_{n-1}
(tools/wmrequal/labels.py:141, S-GAPCHECK 0.5 s rectangle), while both runtime
samplers appended `pending` only AFTER each step's insertion and summed all n
samples, i.e. q_1..q_n. Every predicted source/J column therefore carried a
one-tick shift of exactly `.5*(q_n - q_0)` per row, `.5*Δq_n` per paired
contrast (paired starts share Q0 by construction, prepare.py:65).

## Contract (binding for the WMFIX verdict and every future scorer change)

1. The queue sample series is q_0 (pre-rollout Q0, seeded at construction of the
   tracked entries) followed by one post-service sample per rollout step, q_1..q_n.
2. The source rectangle at horizon h (n = 2h ticks) is `.5 * sum(q_0..q_{n-1})`
   — the labels' left boundary. The final post-service sample q_n is dropped.
3. `values[h]['queue']` stays the labels' end-of-horizon sample q_n.
4. `HARNESS_CONTRACT.json` (schema `wmfix_harness_contract/1`, emitted by
   `python -m tools.wmfixmerge.harness_contract`) records the digests of
   labels/score/native_lanes/runtime, the measured WMREQUAL3 one-tick shift
   (below), `source_clock_match` (digests equal the tree) and `residual_remains`
   (the zero-head retarget check, addendum 03:55 step 4). feasibility.py and
   verdict.py hard-require this receipt before any GPU launch/verdict.

## Fix

- tools/wmfixmerge/runtime.py `TrackedEntries.__init__` seeds `[pending.copy()]`
  (Q0) and the rollout source sums `queues[:steps]`.
- tools/wmrequal/score.py `Entries.__init__` seeds the Q0 sample; the source
  column becomes `.5*queue[:, :-1].sum(-1)`; the reported end-of-horizon queue
  moves from index n-1 to index n (same q_n as the labels).
- Both keep flag-OFF paths untouched; the admission/ordinal fix (c11faf69) is
  independent of the clock fix.

## Measured comparison vs WMREQUAL3

- Synthetic, exact (tests/unit/test_wmfix_contract.py): on controlled conserve-ON
  dynamics with Q0=[2,1], +[1,0] due/tick and blocked service (q_k=[2+k,1]),
  the fixed runtime rectangle equals the label rectangle `.5*Σ q_0..q_{n-1}`
  and the pre-fix convention differs by exactly `.5*(q_n - q_0)`; under exact
  service the tracked series stays at Q0 and the rectangle equals the labels'.
- Recorded validation panel: `harness_contract.py --real REAL.json` measures,
  per horizon, `.5*(values[h].queue - queue0_total)` per row and
  `.5*(queue_arm - queue_native)` per contrast (mean/median/max |.| veh-s, plus
  mean |source| for scale). This quantifies how much of WMREQUAL3.md's WM-side
  source/J error was the clock shift. The numbers land in
  `HARNESS_CONTRACT.json` (`wmrequal3_one_tick_shift`) at the GPU run root;
  Panel A of the WMFIX verdict re-scores BOTH WM2 and the candidate under the
  fixed clock, so the A comparison is internally consistent, and the measured
  shift table is the bridge back to the WMREQUAL3.md numbers.

## Commands

```
python -m tools.wmfixmerge.harness_contract \
  --corrected-report <WMREQUAL3.md> --prepared <TRAIN_PREPARED.json> \
  --real <REAL.json> --residual-remains {0,1} \
  --out /root/autodl-tmp/c4/wmfixmerge20261031/HARNESS_CONTRACT.json
```
`--residual-remains 1` only after the zero-head A-panel run leaves the upstream
residual; `0` records the retarget amendment (do not train the head).
