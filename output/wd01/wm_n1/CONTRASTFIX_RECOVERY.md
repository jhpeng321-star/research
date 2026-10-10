# CONTRASTFIX_RECOVERY — abandoned `contrastfix` at `0aa3cbb` (2026-10-02)

Recovery of the abandoned `contrastfix` branch at `0aa3cbb` on
`zcode/contrastfix2`.  Two independent defects are fixed: (A) the paired
CONTRAST loss silently returned a fake `0.0` whenever a sibling pair shared no
outcome tau or no valid HDV slot, and (B) the evaluation outcome token was built
without the t0 entrance counter the trainer uses, breaking A6 train/eval token
parity.  No real data, SUMO, fitting or training was executed.

## Follow-up (2026-10-02, contrastfix2 resume — DRAFT, pending reviewed guard)

The follow-up hardens (B) into full ORIGINAL-t0 parity and is **code-only**:
the new/updated tests have NOT been executed; the previous broad-suite runs
below are superseded (see the disclosure in "Tests").  Changes versus the
first contrastfix commit:

* `rollout2/state.py` (ownership explicitly expanded): `Rollouts` gains three
  OPTIONAL channels — `t0_kin [B,K,7]`, `t0_mask [B,K]`, `t0_counter
  [B,2,30]` — defaulting to `None` so direct constructions keep the legacy
  path.
* `rollout2/engine.py` freezes the ORIGINAL t0 channels from the P3 input
  world BEFORE the sample loop, i.e. before any controller step or teacher
  mutation: `_t0_kin(world)` builds the exact kinematic view `(s_down, 3x
  lane one-hot, v, a, length)` and `t0_mask` clones `world.present`.  The
  trajectory tensors cannot recover this state (`accel[0]` is the first
  FUTURE applied acceleration; the snapshots carry no length).
* `evalh/idea_a.py`: an A6-ON checkpoint (encoder present) refuses missing
  t0 channels — kinematics, mask or counter — in BOTH the real attribute
  path and the injected mapping path (fail closed).  The trajectory
  fallback (first-future acceleration, 5 m default length, ones mask) and
  the zero counter are never substituted.  An A6-OFF checkpoint keeps the
  legacy zero-token path.

## Follow-up 2 (2026-10-02, verified HDV identity alignment — code-only DRAFT)

Third pass on the same branch, still **code-only** (no tests, CLI, imports or
data reads executed; the new tests below are drafted, audited code-only and
NOT run).  Defect fixed: HDV identity alignment.  FORMAT slots are
persistent/reused run-local positions, so "same slot number across sibling
arms" is NOT an identity: the same vehicle can sit at different slot indices
in the two arms and the same index can hold different vehicles.  The first
fix's HDV support (slot intersection) is therefore invalid for a real pack.

Sidecar contract (exact, drafted independently in the TRAIN identity
workspace; source read only, never imported at module import time — the lazy
import happens at first use and a loader injection exists for synthetic
tests): `load_identity_sidecar(pack_dir, sidecar_dir).ids_for_item(original
GLOBAL manifest position) -> int64[K]` with `-1` = absent slot, plus
`.receipt_sha256`; the sidecar directory comes from env
`WM_N1_TRAIN_IDENTITIES`.

Changes (owned files only):

* `train/contrast_support.py`: `support_ids`, `common_verified_ids`,
  `align_hdv_ids`; `shared_support`/`has_support` are common-ID based when
  BOTH sides carry verified IDs; the slot intersection survives only as the
  documented fallback for ID-less legacy synthetic batches.  Shared
  outcome-tau eligibility is unchanged.
* `train/idea_a.py`: the canonical batch gains optional `vehicle_ids [B,K]`
  int64 (all-or-none collate; loss-slot alignment ONLY — never a network
  input; no loss weight or math change).  `IdeaACorpus` resolves every
  item's ORIGINAL GLOBAL manifest position from the pack manifest itself
  (`MemmapCorpus._items` is the split-filtered list, so a dataset position
  is a TRAIN position and unusable — dataset.py was not edited), lazily
  loads the sidecar through the exact interface, validates each row
  fail-closed (missing/unknown/duplicated identity, or an unverified
  HDV-valid slot, raise `IdeaAConfigError`; never silently disable the HDV
  sub-term, never guess), attaches `vehicle_ids` to TRAIN windows, adds
  `ids` to `support_masks` and exposes `verified_identities()`.  CAL
  corpora never load identities (TRAIN-only provenance; CAL is not read
  here).  `HeldOutTrainView.verified_identities()` delegates — kept
  quick-tune/held-out windows share the corpus's global-position-keyed
  rows.
* HDV contrast sub-term: `_aligned_hdv_pinball` gathers each arm's
  predicted AND realised response at ITS OWN slot index per common verified
  ID (differing A/B indices) BEFORE the frozen pinball primitive; quantile
  levels, weights and the loss math are unchanged; IDs never enter the
  network.  The legacy slot-intersection path remains only for batches
  without IDs.
* `IdeaATrainer.prepare_training_source(source, windows=None)` is now the
  ONE strict fit-preparation entry, used by `fit`: (1) TRAIN-split gate,
  (2) Amendment 6 seal, (2b) quick-tune TRAIN narrowing when configured,
  (3) the verified-identity gate (an actual packed fit fails closed on any
  missing/unknown/duplicated identity BEFORE the sampler/optimizer), (4)
  the ID-aware eligible sibling pool with its empty-pool refusal, (5) the
  TRAIN-only `fit_outcome_norm` — all before any batch or optimizer exists.
  The timing harness must call this helper instead of any duplicated
  preparation or legacy fallback.  `TRAIN_RECEIPT.json` records the
  identity block (mode, counts, sidecar `receipt_sha256`).

Synthetic explicit windows may carry explicit verified fixture IDs; no
identity is ever guessed for the actual pack.

## Follow-up 3 (2026-10-02, canonical preparation binding + canonical pack alias — code-only DRAFT)

Fourth pass on the same branch, still **code-only** (no tests, CLI, imports
or data reads executed).  It repairs the two independent acceptance blockers
raised by the coordinator review
(`E:/research-dense/tmp/zcode_review/WMN1_NEXT_SOURCE_REPAIRS_20261002.md`,
owner tree outside this workspace copy; read source-only this pass — the
earlier "findings file NOT FOUND" note below is thereby superseded: the file
exists at the owner tree and was not present inside this workspace copy).

1. Canonical preparation binding (`train/idea_a.py`): the previous
   `prepare_training_source` draft returned the pool/normalization/identity
   record but did NOT bind it — `fit` bound the values afterwards, while a
   strict timing caller that only ran the helper then read the bound
   attributes got the constructor's identity `OutcomeNorm` and the
   UNFILTERED legacy pair fallback (`_pairs_for`), silently diverging the
   sampler and normalization between timing and fit.  The helper now binds
   the EXACT prepared objects (`self._contrast_pair_pool = pool`,
   `self._identity_receipt = identity`, `self.outcome_norm = outcome_norm`)
   BEFORE it returns, once, for fit and timing alike; `fit` consumes the
   same returned record without refitting (its duplicate binding lines were
   removed).  `IdeaATrainer.__init__` now initializes
   `_optimizer`/`_scheduler` to `None` (the draft gate node asserts their
   absence directly; previously only `getattr(..., None)` paths existed).
   The output record contract is unchanged (same keys, same order of gates).
2. Canonical pack alias allowance (`train/idea_a.py`): the identity loader's
   strict path checks reject any symlink/reparse ancestor, but a deployed
   GPU pack location is an INTENTIONAL frozen alias
   (`$WORK/pack` -> `/root/autodl-tmp/gpu1/wm3/upload`).  New OPTIONAL env
   `WM_N1_CANONICAL_PACK` — set by root ONLY to the separately
   coordinator-validated REAL canonical target (after the raw SHA verify).
   ONLY when it is set: strict `Path.resolve(strict=True)` equality between
   the pack alias and the env target is checked BEFORE the loader reads
   anything; a mismatch raises `IdeaAConfigError` and the loader is never
   called; on match the LITERAL canonical target (never a re-resolved or
   rewritten path) is handed to the identity loader, whose own strict checks
   then prove the target itself is link-free.  The manifest/INDEX-vs-receipt
   binding and the never-open-a-recorded-source contract stay in the loader.
   With the env unset there is NO automatic resolve fallback: the literal
   `pack_dir` is passed unchanged, so an alias still refuses until root
   publishes the validated target.  `corpus.pack_dir` and the new identity
   receipt fields `pack_alias` (original literal alias) and
   `canonical_pack` (validated literal target, or None) sit alongside the
   sidecar `receipt_sha256` in `verified_identities()`/`TRAIN_RECEIPT.json`
   for audit.  No frozen alias, pack bytes or registration was touched.

New draft nodes in `train/tests/test_contrastfix_recovery.py` (NOT executed):
canonical-binding sentinels (`test_prepare_training_source_binds_exact_pool_norm_identity`,
`test_helper_only_caller_gets_fit_state_not_legacy_fallback`) and the
canonical-pack fixtures (`_dir_alias` symlink/junction helper;
`test_canonical_pack_env_passes_literal_target_to_loader`,
`test_canonical_pack_mismatch_refuses_before_loader`,
`test_canonical_pack_env_absent_passes_literal_alias_no_fallback` — the
fixture skips only when the host supports neither symlinks nor junctions).

## Follow-up 4 (2026-10-02, guarded-run repairs: scalar-target aligned pinball + strict-TRAIN-sidecar identity fixtures — code-only DRAFT)

Fifth pass on the same branch; the NEW nodes are again **code-only** (no
pytest, import, CLI or fixture execution).  Evidence base: the root bounded
synthetic guard was executed and accepted, and the same recovery file ran
against the B integration source (b9b87f3) as **35 passed / 4 failed**
(retained synthetic-only log:
`E:/tmp/wmn1_guard_runs_20261002/wmn1_contrast_synthetic1_20261002_090635_618075_5844/guarded_run.log`).
The four failures had two root causes, both repaired here:

1. PROD broadcast bug (`train/idea_a.py` `_aligned_hdv_pinball`): the
   ID-aligned term handed the frozen `model.losses.pinball` BOTH the
   predicted contrast and the target as `[nIDs, 19]`; the frozen primitive
   `unsqueeze(-1)`s its TARGET (contract: `target` broadcasts to
   `[..., 1]`), so every matched-ID count outside `{1, 19}` raised
   `RuntimeError: The size of tensor a (19) must match the size of tensor b
   (nIDs)` — and `nIDs == 19` would have broadcast SILENTLY into a wrong
   `[19, 19, 19]` cross-quantile mean.  The fix is one argument: the
   realised per-matched-ID difference `ra_targ - rb_targ` is passed at its
   contracted SCALAR shape.  The frozen primitive, quantile levels, weights,
   common-ID alignment and differentiable math are unchanged (gradients
   still flow only through the prediction; the realised target is data).
2. Synthetic identity fixtures vs the strict B label gate
   (`train/dataset.py` `_mixed_split_identity_pack` /
   `_resolve_outcome_source`, read SOURCE-only in the sibling integration
   worktree and deliberately NOT edited here): the old
   `_write_identity_pack` INDEX runs carried only `tick_start`/`tick_end`,
   so a TRAIN open of the mixed TRAIN+CALIBRATION pack refused with
   `CorpusRefusal ... incomplete per-run identity (shard_dir /
   n_outcome_rows)` BEFORE any label read — and with complete INDEX
   identity a TRAIN open additionally requires the `WM_N1_TRAIN_OUTCOMES`
   sidecar (no synthetic/legacy mixed-pack fallback exists).  The fixture
   now (a) writes the corpus-writer per-run identity (`key`, `chunk`,
   `world`, `arm`, `split`, `t0_s`, `shard_dir`, `n_outcome_rows`) in
   manifest order so `plan_sidecar`'s derived global row ranges reconcile
   with every item's `outcome_rows`; (b) stages a synthetic per-run shard
   tree BESIDE the pack (`<pack>_shards/<chunk>/<world>/<arm>/outcomes.parquet`,
   value-identical slices of the packed synthetic table, mirroring the
   production shard layout); and (c) builds the TRAIN-only outcome sidecar
   through the PRODUCTION `train.label_isolation.build_sidecar` (real
   receipt, real hashes — never a hand-written receipt), so
   `load_train_outcomes` re-validates it exactly like a staging build.  The
   three TRAIN-reading tests point `WM_N1_TRAIN_OUTCOMES` at
   `<pack>_train_outcomes` via monkeypatch.  Preserved: the CAL-first
   global-position interleave (`_runs_cal_first`, CAL at GLOBAL position
   0), every pre-existing assertion (masks, IDs, receipt), the
   CALIBRATION corpus's authorized mixed-table read, and production label
   isolation (nothing relaxed; no `train/dataset.py` or labels byte touched
   in this worktree; the alias fixtures keep skip-on-unsupported-host).

New regression nodes (guard-pending): the `_aligned_pinball_reference`
frozen-contract helper plus
`test_hdv_contrast_multiple_permuted_ids_scalar_per_id_target` (ncommon=3
!= NQ, fully disjoint permuted slots, diagnostics pinned),
`test_hdv_contrast_at_ncommon_nq_collision_stays_scalar_target` (ncommon=19
== NQ: the old wide target broadcast silently; the loss must equal the
scalar-target reference exactly) and
`test_hdv_contrast_gradient_flows_through_matched_id_term` (ncommon=2:
backward reaches every HDV parameter, finite, nonzero norm).

Disclosure: for this pass only a static `ruff check` on the two owned
Python files (clean) and a `py_compile` syntax check were executed — no
pytest, import, CLI, fixture or data access.  The synthetic-only guard log
above was the sole read outside the owned source, as authorized.

## Follow-up 5 (2026-10-02, guarded sidecar JSON-finalization delegate — code-only DRAFT)

Sixth pass; a single narrow fixture fix in the solely-owned
`train/tests/test_contrastfix_recovery.py` (no production byte touched, no
pytest/import/py_compile/CLI/data execution).  Root cause: Follow-up 4's
`_build_train_outcome_sidecar` invokes the PRODUCTION
`train.label_isolation.build_sidecar`, whose receipt JSON finalization goes
through `LI._write_json_atomic` (tmp file + `os.replace`) — an audit event
the strict guard denies, so the guarded acceptance of the Follow-up-4
nodes could never pass as written.  Fix (delegate as in B1287389): a narrow
autouse fixture `_plain_sidecar_json_finalization` patches ONLY
`train.label_isolation._write_json_atomic` for this file, swapping the
atomic finalization for an EXCLUSIVE fresh-target write (`open(target,
"xb")`, never overwritten) with the IDENTICAL body
`json.dumps(payload, indent=1, sort_keys=True).encode("utf-8")` and the
identical Path return.  Preserved: production `plan_sidecar` planning,
coverage reconciliation, shard reads and every isolation revalidation in
`build_sidecar` execute unpatched; the loader spy assertions (`seen == {}`,
recorded-literal receipt checks) are untouched; production
`train.label_isolation` and the guard itself are not altered.  Where the B
module is absent (this draft tree) the fixture is inert and sidecar nodes
keep failing at their own lazy import, exactly as before.  The real
pinball fix is accepted at source level; runtime acceptance stays held
until the coordinator's guard runs the nodes.

## A. `response_contrast` was zero on real canonical sibling batches

### Cause

Same `(world, t0)` / native-policy does **not** imply that two sibling arms share
a realised outcome horizon or a valid HDV slot.  The old
`IdeaATrainer.paired_response_contrast` (pre-fix `train/idea_a.py` ~1587) formed
every policy-native pair, then:

* outcome sub-term: `pair_mask = omask[a] & omask[b]`; when the two arms' taus
  were disjoint the frozen `model.losses.contrast_loss` computed
  `(loss * m).sum() / (m.sum()*NQ).clamp(min=1)`, i.e. `0.0` for an all-false
  mask;
* HDV sub-term: `_masked_mean(pb, pmask)` divides by `denom.clamp(min=1.0)`,
  again `0.0` for an all-false mask;
* the two sub-terms were then averaged `sum(terms)/len(terms)`, so one supported
  sub-term was **halved** by an unsupported fake-zero sub-term.

The previous coordinator measured 11 of 40 laptop smoke steps reporting
`response_contrast` exactly `0.0`; the loss was training on a mask artifact, not
on the contrast.

### Fix

New module `train/contrast_support.py` (pure tensor/mask inspection, no I/O):

* `item_support_masks` (`train/contrast_support.py:50`) extracts
  `outcome_mask` (taus) and `target_mask & ~is_cav` (HDV slots) from a v1
  `MemmapCorpus` item or a canonical Idea A window.
* `build_eligible_pair_pool` (`train/contrast_support.py:131`) keeps only
  `train.dataset.contrast_pairs` candidates whose two support vectors intersect
  in a tau or a slot.

`train/idea_a.py`:

* `IdeaACorpus.support_masks` (`train/idea_a.py:810`) is a cheap fast path that
  reads only `target_mask`/`veh_step`/the packed outcome rows.
* `IdeaATrainer._contrast_pool` / `_pairs_for` (`train/idea_a.py:1787`, `1801`)
  compute the pool once and hand the **same** list to `epoch_batches` and the
  `PairAwareBatchSampler`; both iterators call `_pairs_for`, so the seeded
  pair-aware order is identical in-process and through the DataLoader.
* `fit` (`train/idea_a.py:2023`) precomputes the pool; on a real run
  (`_require_all_terms`) an empty pool raises before step one.
* `paired_response_contrast` (`train/idea_a.py:1610`) re-derives the support from
  the canonical batch at loss time, appends a sub-term only for the pairs with
  real support, excludes an unsupported sub-term entirely (no dilution), and
  raises `IdeaAConfigError` with its counts when neither sub-term has support.
  The registered pinball/`contrast_loss` math and `K=192` are unchanged (only
  the pair selection / denominator honesty changed); no epsilon floor or
  fabricated label is introduced.
* Per-step diagnostics are recorded as `contrast_support`
  (`pairs`, `outcome_pairs`, `outcome_taus`, `hdv_pairs`, `hdv_slots`,
  `outcome_subterm`, `hdv_subterm`) into `train_log.jsonl` (`train/idea_a.py:1942`),
  the stdout line (`1955`) and `TRAIN_RECEIPT.json` (`2103`).

Unpaired items are **not** dropped: `train.dataset.epoch_batches` still permutes
every position; only pair partners are appended, so unpaired examples keep
training the other losses.

## B. A6 eval/train token parity (ORIGINAL t0 kin, mask, counter)

### Cause

Training built the correction-head token with
`StateGlobalToken(kin, kin_mask, counter)` (`IdeaATrainer.global_token`), while
evaluation called `enc(kin, mask, None)` (`evalh/idea_a.py:401` pre-fix).  With
trained `counter_mlp` weights the eval token differed from the training token
for the identical t0 state; the A6 registration lists the t0 entrance counters
as an input.  The first fix carried only the counter; the trajectory-derived
kinematics still approximated t0 with the first FUTURE acceleration, a 5 m
default length and an ones mask, so the token was still not the training token
whenever the t0 acceleration/length deviated.

### Fix (follow-up state)

* `rollout2/state.py`: `Rollouts` carries explicit optional `t0_kin`/
  `t0_mask`/`t0_counter` channels (see the follow-up section above).
* `rollout2/engine.py:run` freezes all three from the input world before any
  step/teacher mutation and returns them on every rollout.
* `evalh/idea_a.py` resolves the channels through one helper (`_t0_channels`,
  attribute and injected-mapping forms) and, for an A6-ON checkpoint,
  requires them: `_kin_from_rollout(required=True)` raises
  `IdeaANotLoadedError` when `t0_kin`/`t0_mask` are absent or partial, and
  `_state_global_token` raises when the counter channel is absent or `None`.
  No action, strategy, label or future-response channel enters the token.

The engine change is essential: the ORIGINAL t0 state travels with the
rollout, so the outcome path (`_outcome(rows, rollouts)`) can rebuild the
identical training token without a side-channel cache.

## Tests — SUPERSEDED counts; new tests PENDING reviewed guard

**Disclosure (2026-10-02, corrected — supersedes the earlier no-CAL-read
claims):** the previously reported broad suites
(`python -m pytest evalh/tests rollout2/tests wm_api/tests -q`, 190 passed,
and the `train/tests` 150-pass run) executed synthetic CAL-split test modules
inside `evalh/tests` (e.g. the CAL-split and `run_eval_cal_half` tests).  The
correction to the earlier wording: a committed real `CAL_SPLIT_A4`
world-NAME load is KNOWN to have occurred — this was not merely synthetic
metadata handling; and no OS-level complete audit of every file handle or
read exists, so it must NOT be asserted that no CAL/HOLDOUT outcome file was
ever opened.  Only this much is evidenced: the earlier blanket
"no real CAL/HOLDOUT outcome file was opened" claim is unproven and is
withdrawn; what is known is the committed CAL_SPLIT_A4 world-name load plus
the CAL-split metadata exposure through those code paths.  Reference:
coordinator review note `tmp/zcode_review/WMN1_CAL_SPLIT_EXPOSURE_20261002.md`
(owner tree; not present in this workspace copy).  Those broad suites must
NOT be re-run until the coordinator's split-safe reader/guard is integrated
and the CAL prohibition is clarified.

**Current status (this draft):** the identity-alignment follow-up above is
committed as a source draft, Follow-up 3 (canonical preparation binding +
canonical pack alias) is drafted in the same code-only mode, and
Follow-up 4 repairs the two root causes behind the guarded B-integration
run's 35 passed / 4 failed result (scalar-target aligned pinball; strict-
TRAIN-sidecar identity fixtures, production `build_sidecar`).  The
updated/new synthetic tests were audited
code-only and have NOT been executed; execution awaits the coordinator's
reviewed guard and explicit node authorization, with `WD_FORBID_D=1`.  The
coordination findings file
`E:/research-dense/tmp/zcode_review/WMN1_NEXT_SOURCE_REPAIRS_20261002.md`
was supplied by the coordinator at the owner tree (outside this workspace
copy) and was read source-only for Follow-up 3; it is still NOT present
under this workspace root.

Historical counts from the FIRST contrastfix commit (recorded for the ledger,
superseded as current verification): `test_contrastfix_recovery.py` +
`test_contrastfix_token_parity.py` 17 passed; scoped suites 150 / 190 / 46
passed; ruff clean on the touched files.

### Intended EXACT future test selections (code-only audit; not run)

Single command, after the coordinator authorizes the guard and these nodes:

```
WD_FORBID_D=1 python -m pytest \
    train/tests/test_contrastfix_recovery.py \
    evalh/tests/test_contrastfix_token_parity.py \
    "rollout2/tests/test_rollout2_interface.py::test_engine_freezes_original_t0_channels_before_mutation" \
    -q

WD_FORBID_D=1 python -m ruff check train/contrast_support.py train/idea_a.py \
    train/tests/test_contrastfix_recovery.py evalh/idea_a.py \
    evalh/tests/test_contrastfix_token_parity.py rollout2/engine.py \
    rollout2/state.py rollout2/tests/test_rollout2_interface.py
```

Node inventory (42 tests in `test_contrastfix_recovery.py` + 1 new engine
test; eval/rollout nodes owned by the separate evalcorr task):

* `train/tests/test_contrastfix_recovery.py` — 42 tests: the 12 first-fix
  contracts (eligible support, sampling-pool parity, fail-closed counts,
  dilution guard; one updated to satisfy the new identity gate), 22 NEW
  verified-identity contracts: ID helper semantics
  (`common_verified_ids`/`align_hdv_ids`), pool HDV eligibility by permuted
  common IDs (NOT slot overlap), loss-time swapped-slot alignment with an
  exact pinball expectation, same-slot different-VID exclusion, validity/
  non-CAV/absent-ID masks, corpus identity rows keyed by the GLOBAL manifest
  position (CAL run occupies position 0), `support_masks` IDs, CAL corpora
  carry no identities, missing/duplicated/unknown/absent-row fail-closed
  gates, `verified_identities` receipt, unset-env refusal,
  source-without-accessor refusal, identity gate ordering BEFORE the pool
  refusal, unverified-status refusal, `prepare_training_source` record
  before the optimizer, explicit-windows mode, and an end-to-end synthetic
  fit recording the verified identity receipt — plus 5 Follow-up-3 nodes:
  the canonical binding sentinels (exact pool/norm/identity objects bound
  before return; a helper-only caller reads the fit state, not the legacy
  unfiltered fallback; no optimizer/sample before binding; unchanged record
  key contract) and the canonical-pack alias fixtures (literal canonical
  target passed on a validated match; mismatch refuses before the loader is
  called; env absent/blank passes the literal alias with NO resolve
  fallback, receipt `canonical_pack` None and `pack_alias` literal) — plus
  3 Follow-up-4 nodes: the `_aligned_pinball_reference` frozen-contract
  helper, the multi-ID permuted-slot scalar-target equality (ncommon=3),
  the ncommon=19 == NQ silent-broadcast regression and the two-ID
  differentiability/backward node; the `_identity_pack` fixture itself now
  emits the corpus-writer per-run INDEX identity, a synthetic per-run shard
  tree and a production-built TRAIN-only outcome sidecar
  (`train.label_isolation.build_sidecar`), with the three TRAIN-reading
  tests pointing `WM_N1_TRAIN_OUTCOMES` at it (all pre-existing assertions,
  the CAL-first global-position interleave and the CALIBRATION
  mixed-table read unchanged).
* `evalh/tests/test_contrastfix_token_parity.py` — REWRITTEN (evalcorr task),
  9 tests: 1. train/eval token equality with the counter (attribute +
  mapping paths); 2. token uses NONZERO t0 acceleration, never the first
  FUTURE step; 3. token uses VARIED lengths, never the 5 m default;
  4. same t0 state + different future plans => SAME token;
  5. missing/partial t0 kin/mask channels fail closed (attribute + mapping);
  6. missing/None counter channel fails closed (attribute + mapping);
  7. token sensitivity to the counter value;
  8. A6-OFF checkpoint keeps the legacy zero token;
  9. `_rollout_counter` reports channel presence, not value.
* `rollout2/tests/test_rollout2_interface.py::test_engine_freezes_original_t0_channels_before_mutation`
  — NEW (evalcorr task): the engine returns `t0_kin`/`t0_mask`/`t0_counter`
  captured from the input world before mutation, with the exact kinematic
  column order, dtype-follows-world, detached-clone counter semantics.

Pending (NOT in this task's files; held until their owners land them):

* evalcorr task: `evalh`/`rollout2`/`wm_api` files above, `M1_READY.md`.
* B worker: `train/dataset.py` `ensure_pair` must honor the ELIGIBLE pool
  (its `pairs=None` fallback rebuilds the UNFILTERED `contrast_pairs` list;
  missing guard in dataset.py itself, tests held with it).
* Timing worker: must call `IdeaATrainer.prepare_training_source` (strict
  public helper) instead of any duplicated preparation or legacy fallback;
  the helper now BINDS the prepared pool/norm/identity before returning
  (Follow-up 3), so the timing caller must consume that bound state and
  reject an older trainer missing the method (no raw-pair fallback).
* Identity module owner (TRAIN identity sidecar): the strict loader must
  keep rejecting link ancestors on the path it is handed, verify the raw
  manifest+INDEX binding against the sidecar receipt and never open a
  recorded source path; with `WM_N1_CANONICAL_PACK` set it now receives the
  validated literal canonical target (link-free by its own strict checks),
  while the alias literal stays in `pack_alias`/`pack_dir` for the handoff
  gate.  Root publishes the env only after its own SHA verify.
* Coordinator: split-safe reader/guard + node authorization.

Explicitly NOT selected (contain CAL-split metadata tests or unrelated
suites; forbidden until the guard lands): `evalh/tests` whole directory
(includes `test_cal_split.py`, `test_run_eval_cal_half.py`), `rollout2/tests`
whole directory beyond the single node above, and any real-pack, 40/50-step,
SUMO or GPU command.

wm_api needs no new test: its outcome head builds kin from the
`StateSnapshot` t0 state directly (always the real t0) and its existing
`wm_api/tests/test_state_token.py` refusals are unchanged by this work.

## Limitations / honest scope

- Two-stage history: the FIRST contrastfix commit ran only code review +
  synthetic CPU tests (counts above, now superseded); the 2026-10-02
  follow-ups (t0-parity and this identity pass) ran NO tests at all
  (code-only drafts).  **No claim of a 40- or 50-step real pass.**  The
  registered 50-step smoke and the real-member timing gate are documented in
  `M1_READY.md` and remain the coordinator's allocation.
- The eligible pool is a sampling hint; the loss re-validates support every
  step (now by common verified ID when the batch carries `vehicle_ids`), so
  a batch assembled by an external path cannot smuggle in an unsupported
  pair.
- `IdeaACorpus.support_masks` uses the frozen `MemmapCorpus._array` /
  `outcome_target` read-only access.  `train/dataset.py` was not edited; the
  new helper is called from the `idea_a` iterator with the existing
  `epoch_batches(pairs=...)` parameter.  `ensure_pair`'s own unfiltered
  fallback inside dataset.py is the B worker's pending guard.
- The eligible pool is O(items) once per member fit for sources without the
  cheap `support_masks` fast path; it reads only TRAIN masks and never
  CAL/HOLDOUT/TEST.  Identity rows are validated once per global manifest
  position and cached; the corpus pickling drops that cache (re-validated
  lazily in workers), the loaded sidecar pickles as compact bytes.
- `rollout2/state.py` was edited in the FIRST follow-up (ownership explicitly
  expanded by the coordinator): the three new `Rollouts` fields are optional
  with `None` defaults, so code that constructs `Rollouts` directly (tests)
  does not carry t0 channels and stays on the legacy path; an A6-ON
  evaluation of such a rollout now fails closed instead of approximating.
  This identity pass did not touch `rollout2` (evalcorr-owned).
- Verified identity gates apply to every windows=None (packed-corpus) fit,
  quick-tune included; explicit synthetic windows record their fixture-ID
  coverage instead and are never identity-guessed.
- The follow-up tests are drafted, audited code-only and NOT executed; the
  coordinator's reviewed guard and explicit node authorization are required
  first (see "Tests" above).  Follow-up 3 ran ONLY the static `ruff check`
  on the three owned Python files (all clean); no test, import, CLI or
  data access was executed in any follow-up pass.
