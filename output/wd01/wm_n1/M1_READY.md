# M1_READY — A6 binding + contrastfix recovery, member-1 readiness (2026-10-02)

**Status:** A6 (state-only global token, `REGISTRATION_WM_N1_AMENDMENT_6_STATE_TOKEN.md`)
is bound into every approval, into training, and into every outcome-producing
evaluation path, and **contrastfix** repairs the paired CONTRAST loss and the
train/eval token parity (see `CONTRASTFIX_RECOVERY.md`).  **2026-10-02
follow-up (draft):** ORIGINAL-t0 kin/mask/counter parity is implemented
code-only; the follow-up tests are PENDING the coordinator's reviewed guard,
and the earlier broad-suite verification counts are SUPERSEDED — the previous
suites exposed CAL split metadata (disclosure in `CONTRASTFIX_RECOVERY.md`).
No SUMO, no real pack, zero drive-D access (`WD_FORBID_D=1`).

- **Contrastfix commit:** on `zcode/contrastfix2` (this commit; see
  `CONTRASTFIX_RECOVERY.md` for the changed files and test counts).
- **Pre-contrastfix ship commit (A6 binding):** `545684a` on `zcode/a6bind`.
- **WIP base (previous executor):** `5381f4f`; pre-A6 merge `bc6a51b`.

## Frozen APPROVAL identities (LF-sha256, CRLF-safe; re-hashed against the files)

Every member / held-out approval now additionally binds `amendment_6_lf_sha256`,
and the gate re-hashes the frozen A6 file (refusing a missing or tampered copy):

| record | LF-sha256 |
| --- | --- |
| `REGISTRATION_WM_N1.md` | `666233aa1d1e466750d6d944b9102c84a2f32850ebeb4a97c9f3c9b56dd93246` |
| `REGISTRATION_WM_N1_AMENDMENT_1.md` (member-1-first) | `decb96562ebfb549b56406d980be9c78bf4a377bd50c5eadeea08aac16e16c57` |
| `REGISTRATION_WM_N1_AMENDMENT_2_J4_DISCLOSURE.md` | `56fa99a04fdd38e37a518dfc65972bc5e2e6d227210372a75258bf28249c83ed` |
| `REGISTRATION_WM_N1_AMENDMENT_3_IDEA_A.md` | `a597d9861dcc859cbcb9c10b65069a81031732516ec12a0296fd68aff604eef6` |
| `REGISTRATION_WM_N1_AMENDMENT_4_M1_TUNING.md` | `d392b30030f2b04ddb5db25c681cf92bca80fe1fc4e77d234303f7bbb810822f` |
| `CAL_SPLIT_A4.json` | `bc4e32f2b6efcf0e0165cb175a0b3ffaec0aaa6f419076463f9242ad90820d46` |
| `REGISTRATION_WM_N1_AMENDMENT_5_KMAX_RULE.md` | `7788c270f35a0620f0bfd71a61d871f741b6bbbffa144ba61438e28b73d13ae7` |
| `REGISTRATION_WM_N1_AMENDMENT_6_STATE_TOKEN.md` | `28dbc184b0178d5f74f681e0701771725fc8abc80e534e1087f0a9e49835c948` |
| `TIMING_GATE.json` | raw-bytes sha256 of the written gate (per run, not a constant) |

The single A6 gate lives in `evalh.seals.require_frozen_amendment_6()`:
`bundle.approval._bind_amendment_6`, `bundle.multibox.require_config_approval` and
the trainer all call it / the same constant.

## What is now enforced

1. **Approvals** — `bundle.approval.require_approval` (member-1 / members-2-5),
   `bundle.approval.require_a_m2_5` (`APPROVAL_A_M2_5`), and
   `bundle.multibox.require_config_approval` (every `APPROVAL.json`, including the
   `2box/c_m1only` boxA/boxB configs) require a `amendment_6_lf_sha256` equal to
   the freeze **and** re-hash the file CRLF-safely.  `write_config_approval` and
   `write_approval_a_m2_5` emit the field.
2. **Training** — `state_global_token` defaults ON.  `IdeaATrainer.fit` calls
   `a6_binding()` first, which refuses (before touching data) unless A6 is
   present and matches; the bundle executor `bundle.e2e.run_config_job` goes
   through the same `fit`.  OFF happens only via the explicit CLI flag
   `python -m train.idea_a ... --no-state-global-token`.  `TRAIN_RECEIPT.json` and
   the member checkpoint record `state_global_token` **and**
   `amendment_6_lf_sha256`.
3. **Outcome paths** — `evalh.idea_a.IdeaAVariantPredictor._outcome` (also used by
   `adapter.predictor.IdeaABundlePredictor`), and `wm_api.engine.outcome_head`
   build the token from the t0 state with the checkpoint's `StateGlobalToken`
   weights and refuse to score a state-token checkpoint with an absent/empty
   state (never a zero token).  `adapter.predictor.load_idea_a_bundle` mounts
   `state_enc` from its manifest.  `inc1eval` computes trajectory metrics from
   rollouts and does **not** consume the correction-head outcome, so it needs no
   A6 change.
4. **Contrastfix (this commit + 2026-10-02 follow-up draft)** —
   - The paired CONTRAST loss only counts sibling pairs that share a realised
     outcome tau or a valid HDV slot (`train/contrast_support.py`); a sub-term
     with no support is excluded, never averaged in as a fake `0.0`, and a batch
     with no support fails closed with its counts.
   - The eligible TRAIN pair pool is precomputed from the real mask arrays and
     drives the **same** seeded pair-aware order in-process and through the
     DataLoader; unpaired items keep training the other losses.
   - The P3 engine freezes the ORIGINAL t0 channels (`t0_kin`/`t0_mask`/
     `t0_counter`, optional fields on `Rollouts`) from the input world BEFORE
     any controller/teacher step, and every A6-ON evaluation path builds the
     token from exactly those channels — an A6-ON checkpoint REFUSES missing
     t0 kinematics, mask or counter channels, in the real attribute path and
     the injected mapping path (no first-future acceleration / 5 m default /
     ones mask / zero counter substitute).  A6-OFF keeps the legacy zero
     token.  See `CONTRASTFIX_RECOVERY.md`.

## Exact Linux GPU runbook (pack_v3; run from `output/wd01/wm_n1`)

`$WORK` = box-local run directory.  The upload lives at
`$WORK/upload/manifest.json`; the queue's `convert+verify` stage materialises the
registered v3 pack at **`pack_source=$WORK/pack`** (`k_max=192`) before the
real-member timing gate and any training.  A6 (`state_global_token`) is ON by
default — **never** pass `--no-state-global-token` for the registered run.  This
runbook is **stop-before-approval**: it does not author or write any APPROVAL.

**Hard ordering (coordinator, 2026-10-02):** FULL training (member-1 and
members 2-5) runs only AFTER (a) the 50-step bounded smoke is green on the
converted pack AND (b) the ALL5 conditions hold — the timing gate reports
`pack_source=$WORK/pack`, `k_max=192`, A6 ON and **all five** registered loss
terms.  No training section may be executed ahead of those gates.

### 0. Environment (both boxes)

```bash
# envprefix: the RD GPU env is NOT the box default python; export it first
export PATH=/root/autodl-tmp/gpu1/env/rd_gpu/bin:$PATH

# TRAIN-only sidecar env REQUIRED: jobs that train (member-1, members 2-5,
# the 50-step smoke) must run inside the training sidecar environment; eval /
# timing-gate-only jobs run in the base RD env.  Never mix them in one shell.
python -m bundle.envcheck --device cuda --format v3
# pre-flight is synthetic/CPU; it does not open the CAL split or real data
python -m preflight all --data-root /path/to/research-dense   # expect 6/6 PASS
```

### 1. GPU-A (member-1) — real timing gate, HARD STOP

```bash
python -m bundle --manifest "$WORK/upload/manifest.json" --work-dir "$WORK" \
    --approval "$WORK/APPROVAL.json" --stop-after timing-gate --device cuda
# -> $WORK/TIMING_GATE.json + logs/timing-gate.log
# The gate must report pack_source=$WORK/pack, k_max=192, A6 ON and all five
# registered loss terms (the ALL5 condition).  Read the verdict and STOP: the
# owner APPROVAL binds the gate's raw sha256; do not author it here.
```

### 2. 50-step bounded smoke (after the timing gate, BEFORE any training)

```bash
# 50 steps x 1 epoch on the real converted pack; A6 ON by default.
python -m train.idea_a --member 1 --seed 101 --corpus "$WORK/pack" \
    --out "$WORK/smoke50" --device cuda --num-workers 8 \
    --epochs 1 --steps-per-epoch 50 --k-max 192 --log-every 1
# Inspect $WORK/smoke50/train_log.jsonl: every step must carry a
# "contrast_support" block with non-zero outcome/HDV pair support for the
# steps that report response_contrast; a zero-support step must raise, not
# log response_contrast=0.  This is a bounded smoke, NOT efficacy evidence.
# Smoke green + ALL5 above => full training sections below MAY proceed.
```

### 3. GPU-A member-1 (only after owner approval AND smoke + ALL5; NOT authored here)

```bash
# member-1 only (boxA.json stop_after = member-1); TRAIN-only sidecar env
python -m bundle.multibox run-box-config \
    --config bundle/box_configs/2box/c_m1only/boxA.json --work-dir "$WORK"
```

### 4. GPU-B — timing gate ONLY until the CAL prohibition is clarified

```bash
python -m bundle --manifest "$WORK/upload/manifest.json" --work-dir "$WORK" \
    --approval "$WORK/APPROVAL.json" --stop-after timing-gate --device cuda
# gpuB remains TIMING-ONLY (no training, no eval) until the coordinator
# clarifies the CAL prohibition / split-safe reader; heldout-i-j4b
# approval/training is NOT authorised.  STOP HERE for gpuB.
```

### 5. Standalone registered training (members 2-5) — last, after A_M2_5 approval

```bash
python -m bundle --manifest "$WORK/upload/manifest.json" --work-dir "$WORK" \
    --approval "$WORK/APPROVAL.json" \
    --a-m2-5-approval "$WORK/APPROVAL_A_M2_5.json" \
    --winning-config "$WORK/WINNING_CONFIG.json" \
    --stop-after members-2-5 --device cuda
```

Notes for the box:
- `bundle` / `bundle.multibox` have **no** `--num-workers` flag; the real timing
  gate resolves workers host-scaled (`num_workers=None` -> `min(8, cpu-2)`,
  recorded in the gate).  Member training and the standalone
  `python -m train.idea_a --num-workers 8` control the DataLoader directly.
- `run-box-config` honours the config's `stop_after` (`member-1` for boxA,
  `heldout-i-j4b` for boxB), so member-1-first is structural.
- The contrastfix eligible-pair pool is computed once per `fit` from the real
  TRAIN masks; the in-process sampler and the DataLoader sampler receive the
  same list, so their seeded orders are identical (parity unit-tested).
- Run `bundle`/`train.idea_a` under a filesystem guard that forbids the CAL
  split and any sealed path; the pack conversion/verify is the only real-data
  entry.

## Verification status (SUPERSEDED — no re-run authorized)

**2026-10-02 disclosure:** the broad suites below executed synthetic
CAL-split test modules in `evalh/tests` and exposed CAL split METADATA; the
earlier "no calibration read" claim is superseded (full disclosure in
`CONTRASTFIX_RECOVERY.md`).  These counts are historical records of the FIRST
contrastfix commit only and are NOT current verification.  The 2026-10-02
follow-up (ORIGINAL-t0 channels + fail-closed refusals) is committed as a
source draft; its tests are pending the coordinator's reviewed guard and
explicit node list (see `CONTRASTFIX_RECOVERY.md` "Tests").

```
# historical, first contrastfix commit only — DO NOT RE-RUN in this form
python -m pytest train/tests -q                        # 150 passed (superseded)
python -m pytest evalh/tests rollout2/tests wm_api/tests -q   # 190 (superseded)
python -m pytest corr/tests model/tests -q             # 46 passed (superseded)
python -m ruff check <contrastfix files>               # passed (superseded)
python -m preflight all --data-root E:/research-dense  # 6/6 (superseded)
```

The broader A6-binding suite from `545684a` (643 passed) is likewise a
historical record.  The 50-step real laptop/GPU smoke is **not** part of any
verification and requires the coordinator's allocation; the intended exact
future test selections are listed in `CONTRASTFIX_RECOVERY.md`.

New contrastfix tests: `train/tests/test_contrastfix_recovery.py` (12 tests,
unchanged: ineligible semantic pair, mismatched tau, no HDV support, partial
eligible pool, switching `__native` is not a reference, all-ineligible
fail-closed, no dilution by a zero term, fit-time fail-closed, epoch-order
parity, in-process vs DataLoader parity, unpaired items retained) and
`evalh/tests/test_contrastfix_token_parity.py` (REWRITTEN draft, 9 tests:
train/eval token equality with the counter, NONZERO t0 acceleration not the
first future step, VARIED lengths not the 5 m default, same t0 + different
future plans => same token, missing kin/mask fail-closed, missing counter
fail-closed, counter sensitivity, A6-OFF legacy zero token, counter
channel-presence reporting) plus the new
`rollout2/tests/test_rollout2_interface.py::test_engine_freezes_original_t0_channels_before_mutation`
(engine returns the ORIGINAL t0 channels captured before mutation).

New/updated A6 tests from the binding commit: `bundle/tests/test_approval.py`,
`bundle/tests/test_multibox.py`, `bundle/tests/test_winning_config.py`
(A_M2_5 writer + tamper), `bundle/tests/test_e2e_a6.py`, `train/tests/test_m1fix2.py`,
`evalh/tests/test_idea_a_variant.py`, `evalh/tests/test_idea_a_adapter.py`,
`wm_api/tests/test_state_token.py`.

## Honest remaining notes

- `wm_api` builds the A6 token from the `StateSnapshot` t0 state directly;
  `evalh` builds it from the rollout's EXPLICIT ORIGINAL t0 channels
  (`t0_kin`/`t0_mask`/`t0_counter`, frozen by `rollout2.engine` before any
  step).  An A6-ON evaluation of a rollout without those channels fails
  closed (both attribute and mapping forms); an A6-OFF checkpoint keeps the
  legacy zero-token path.
- The A6 §A6.2 "extend G3 to the token" invariant is covered by unit tests
  (`corr/tests/test_corr_interface.py::test_state_global_token_ignores_label_channels`
  and `train/tests/test_m1fix2.py::test_state_global_token_is_state_only_and_label_invariant`);
  the `preflight` G3 gate itself still targets the P2 HDV schema.
- `bundle/README.md`'s example `APPROVAL.json` predates A4/A6; the authoritative
  required fields are `_REQUIRED_FIELDS` in `bundle/approval.py` (registration,
  A1, A4, CAL split and A6 LF-shas + the gate raw sha).
- The contrastfix eligibility pool is precomputed with `source[i]` on sources
  that expose no cheap `support_masks` (the `IdeaACorpus` fast path reads only
  `target_mask`/`veh_step`/packed outcome rows).  It is O(items) once per member
  fit; it never reads CAL/HOLDOUT/TEST.
