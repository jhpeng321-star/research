# M1 INTEGRATION (zcode/m1integ @ 1878348) — K_max on the Idea A path + pack_v3 boxes

**Status:** integration complete (2026-10-01). Branch `zcode/m1integ`
(base merge `1878348`); cwd `output/wd01/wm_n1`. No featurization, no real
packs, no SUMO, no D: reads/writes. Synthetic tmp packs only.

This document records what changed in the integration of the member-1 training
path (`zcode/wd01` + `zcode/kfix192` + `zcode/m1realbatch2`), the test evidence,
and the exact GPU-box command sequence.

## 1. What changed

### 1.1 `k_max` threaded through `IdeaACorpus` (amendment 5, K=192 v3)

The merge kept both config fields (`k_max`, `allow_incomplete_losses`) and the
`IdeaACorpus` call sites, but `k_max` was not forwarded to the frozen
`MemmapCorpus`. Fixed:

* `train/idea_a.py` — `IdeaACorpus.__init__` gains keyword-only `k_max=None`;
  it forwards to `MemmapCorpus(..., k_max=k_max)` and re-exposes
  `self.k_max` / `self.format`. `None` takes the pack's K; an explicit 128
  (v2) / 192 (v3) makes a pack/model K mismatch refuse **on the Idea A path**
  (the same `CorpusRefusal` `MemmapCorpus` raises).
* Call sites now thread it: `IdeaATrainer.evaluate_cal` (CAL view),
  `corpus_overfit_probe` (G4), `bundle/timing._idea_a_trainer` (timing/G4/G5),
  `bundle/e2e.run_config_job`, and `train.idea_a.main`.
* CLI: `python -m train.idea_a` gains `--k-max {128,192}` (default None →
  pack's K) and `--steps-per-epoch` (bounded smoke/test override).

### 1.2 Bundle: timing gate + envcheck default to v3 (pack_v3)

* `bundle/timing.py` — the Idea A corpus measurement builds its synthetic pack
  at `DEFAULT_GATE_K_MAX = 192` and opens the corpus with the matching explicit
  `k_max`; `run_timing_gate(..., format=, k_max=)` records `format` / `k_max`.
  The registered member-1 / held-out (i) gate therefore runs on the K=192
  real-batch path.
* `bundle/envcheck.py` — `run_selfcheck` / `--format` default to v3
  (K_max=192); `--format v2` is the frozen K=128 estimate.
* `bundle/multibox.py` — `build_m1_only_configs()` (only the `c_m1only` plan)
  sets `corpus_source = D:/rd-staging/wmn1_feat/pack_v3`, `format = "v3"`,
  `k_max = 192`, and an `envcheck_argv` with `--format v3` on both the box files
  and every config entry. The `2box/{a,b,c}`, `3box`, `4box` plans stay on
  pack_v2. The c_m1only JSON was **regenerated with the generator**
  (`bundle.multibox.write_m1_only_configs`), not hand-edited.

### 1.3 Approval binding: amendment 5 required (CRLF-safe)

`bundle/multibox.py` per-config `APPROVAL.json` (schema
`wm_n1_multibox_approval/1`, used by the `c_m1only` boxA/boxB configs) now
requires the frozen Amendment 5 LF-sha
`amendment_5_lf_sha256 = 7788c270f35a0620f0bfd71a61d871f741b6bbbffa144ba61438e28b73d13ae7`
(`REGISTRATION_WM_N1_AMENDMENT_5_KMAX_RULE.md`). Added to `DEFAULT_RECORDS`,
`write_config_approval`, and `require_config_approval` (declared value ==
frozen constant AND re-hashed file equality, LF-normalised so a CRLF checkout
hashes identically). The member-1 `bundle.approval` binding is unchanged.

### 1.4 Docs

`M1_FIRST_PLAN.md` and `bundle/box_configs/README.md`: pack_v3, `--format v3`
envcheck, the timing-gate STOP, and the amendment-5 approval requirement.

## 2. Evidence

### 2.1 `main()`-path fits (exact GPU-box argv), all five A3.2 losses active

Synthetic packs (real `corpus.tests.synth` → P9 channels → `corpus.pack`), run
through `train.idea_a.main` with `--epochs 1 --steps-per-epoch 2 --k-max K`.
Real `TRAIN_RECEIPT.json` excerpt (pasted from stdout):

```
K=192 rc=0 active_loss_terms=['arrival', 'correction_crps', 'residual_nll',
  'response_contrast', 'rollout'] allow_incomplete_losses=False steps=2
  losses=[36.68999481201172, 29.721805572509766]
K=128 rc=0 active_loss_terms=['arrival', 'correction_crps', 'residual_nll',
  'response_contrast', 'rollout'] allow_incomplete_losses=False steps=2
  losses=[35.80455780029297, 28.675230026245117]
```

Both are `allow_incomplete_losses=False` (fail-closed) and produce every
registered term. A K mismatch (`k_max=128` against a K=192 pack) refuses with
`CorpusRefusal: K_max mismatch` through `main()`; the cumulative rollout mask
and the run-boundary window are re-checked at K=192
(`train/tests/test_idea_a_kmain.py`).

### 2.2 Test counts (WD_FORBID_D=1, CPU, no SUMO)

| suite | result |
| --- | --- |
| features + corpus + model + combine + hdv + adapter + wm_api + evalh | 399 passed |
| train | 109 passed |
| bundle | 150 passed |
| rollout2 | 43 passed |
| inc1 (inc1eval; `PYTHONPATH` wm_n1 before wd01, see note) | 31 passed |
| preflight | 35 passed |
| arrival / corr / events / inventory / ctrl | 9 / 12 / 9 / 21 / 29 passed |

`python -m preflight all --data-root E:/research-dense` → **6/6 PASS**
(`preflight/reports/PREFLIGHT_SUMMARY.json`).

Pre-existing, out-of-scope: `census_kmax_tools/test_kcensus.py` is
**2 failed / 3 passed** at the task base. Its K=192 residual relies on
monkeypatching `features.constants.K_MAX`, but `features.featurizer.parse_run`
captured the default `k_max: int = C.K_MAX` at import (a `zcode/kfix192`
change), so `census_lib`'s residual stays 128. This integration does not touch
`census_kmax_tools` or `features/`, and these tests are not in the enumerated
acceptance suites; the A5 census result is frozen evidence and was not
modified.

Note: `inc1/tests/conftest.py` inserts `output/wd01` at `sys.path[0]` when
`WD_FORBID_D=1`, shadowing the `wm_n1/inc1` package with `wd01/inc1`. Running
with `PYTHONPATH` = `wm_n1;wd01` (so the membership checks skip the re-insert)
resolves it; this is a pre-existing harness quirk, not a code failure.

## 3. Exact GPU-box command sequence (pack_v3)

`$WORK` is the box-local run directory; both boxes are provisioned with the
pack_v3 upload produced by the amendment-5 featurization (out of scope here;
zero featurization on this CPU box).

### GPU-A — Idea A member 1 alone, then STOP

```
python -m bundle.envcheck --device cuda --format v3
python -m bundle.multibox run-box-config \
    --config bundle/box_configs/2box/c_m1only/boxA.json --work-dir $WORK
```

or, through the queue (same stages, explicit stop):

```
python -m bundle --manifest $WORK/upload/manifest.json --work-dir $WORK \
    --approval $WORK/approvals/APPROVAL.json \
    --stop-after member-1 --device cuda
```

Stages: `verify-upload -> convert+verify -> timing-gate (pack_v3, K=192) ->
approval -> member-1`, then STOP. The box waits at the timing gate until the
owner `APPROVAL.json` binds `$WORK/TIMING_GATE.json`'s sha; members 2-5 are not
queued.

### GPU-B — held-out (i) `J4 -> J4B` only, then STOP

```
python -m bundle.envcheck --device cuda --format v3
python -m bundle.multibox run-box-config \
    --config bundle/box_configs/2box/c_m1only/boxB.json --work-dir $WORK
```

Equivalent single-member argv:

```
python -m train.idea_a --member 1 --seed 101 --corpus $WORK/pack \
    --out $WORK/members/heldout-i-j4b --device cuda --ensemble-k 1 --k-max 192
python -m evalh.idea_a heldout-config --test i \
    --out $WORK/eval/heldout-i-j4b \
    --predictor adapter.predictor:IdeaABundlePredictor \
    --reference evalh.climatology:ClimatologyPredictor
```
