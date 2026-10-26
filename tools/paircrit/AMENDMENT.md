# S-PAIRCRIT-1 cycle-boundary amendment (proposal only; NOT installed)

At the next **unstarted policy phase**, after an accepted native-residual policy
has a completed, matched300s POLICY/B TRAIN cohort, install c4/paircrit source from
parent6e458c8b. Record source/config/registration/input hashes before dispatch.
Do not import the rejected cycle0 actor or label its outcomes as the new actor's.

## 1. Paired critic

Let `C` be the completed conversion receipt belonging to `state.policy.sha256`.
Use existing artifacts only:

```text
python -m wms.control.paircrit_export \
  --source C.pairs --policy-returns C.returns \
  --native-returns C.native_returns --output NEW_PAIRING_JSON
```

`C.*` are bound receipt values, not directory guesses. Require the source policy
SHA in the real-return export to equal the accepted `state.policy.sha256`.
Initialize `state.real_returns=C.returns`,
`state.native_real_returns=C.native_returns`, and
`state.pairing_receipt=NEW_PAIRING_JSON`. The exporter opens only numeric converted
arrays, export receipts and observable summaries; it allocates no simulation.

Create a boundary-owned resolved policy TOML derived from
`configs/c4_policy14_paircrit.toml`; apply exactly:

```toml
[paircrit]
enabled = true
warmup_updates = 100

[longcritic]
enabled = false

[real_share]
enabled = false

[breakdown]
enabled = false
```

Set loop `policy.config` to that resolved TOML and loop `policy.paircrit=true`,
`policy.longcritic=false`, `policy.breakdown=false`; retain `policy.native_init=true`
and existing frozen-panel/horizon/reliability arguments. `policy_cmd` supplies the
current real/native exports, pairing receipt and exact source-policy SHA. Migrate
an absolute checkpoint by restoring the accepted actor first, then creating the
zero-initialized three-output advantage head and its new critic optimizer; keep
the actor optimizer when no new feature head is attached. Warm the paired critic
100 updates on fitting worlds, hard-copy target state, and never fit development
worlds. Later cycles resume paired weights and rebind/warm against their new cohort.

Retain gamma1, authoritative source-plus-road J, current action/lane ownership,
native-relative progress/acceptance vetoes, same accepted WM, same actor updates,
30–60s exploration persistence, charged seconds/seeds, failed-attempt accounting,
36,933 primary/8,000 combined-long/210,000 total limits, time limits and overlap.
Paired overlap starts **after conversion**, with both arms/proof hash-bound, and
joins before check/promotion. It retains the independent frozen accepted WM.
WM actor imagination remains≤60s; unmeasured120/300s difference assistance is zero.

## 2. Breakdown head — blocked by current measurements

Current `breakdown.pt` is **not qualified**; do not use it for deployment. It is a
diagnostic, frozen-feature prototype. Rule/constants and the failed calibration
must remain in the record. No manual behavioural threshold or per-result rule
adjustment is permitted.

For a later boundary with adequate existing real outcomes and independently
qualified calibration, set raw policy `[breakdown].enabled=true`,
`checkpoint=QUALIFIED_FEATURE_HEAD`,
`gate_checkpoint=QUALIFIED_EXTERNAL_CANDIDATE_HEAD`, and loop
`policy.breakdown=true`. The feature head is frozen inside the actor. The separate
gate head must bind to the selected **complete candidate checkpoint SHA**, use the
same registered rule, and qualify on worlds disjoint from fitting and temperature
calibration. External qualification avoids a checkpoint/calibration hash cycle.

Qualification rule: independent-world simultaneous upper95 bootstrap of the
maximum six-output mean Brier difference versus fitting-world event prevalence
must be≤0;2000 draws, seed20261026, at least3 scoring worlds/output. This is forecast
noninferiority, not conditional-future or efficacy certification. Event-specific
WM reliability is rule/WM-SHA-bound lower95 Brier skill, zero when unavailable;
K≥2 paired sample frequencies retain censor masks. Delay reliability cannot be
transferred to event probabilities.

Deployment must additionally have policy-minus-native predicted probability≤0 at
all three horizons and all16 frozen starts, verified by the receipt before adoption
and collection. Missing/unqualified/stale candidate calibration refuses adoption;
the original critic-free native J veto remains independently necessary. The current
two-outcome test panel cannot establish probability-difference predictability.

This amendment does not authorize more SUMO or change the main loop's budget.
