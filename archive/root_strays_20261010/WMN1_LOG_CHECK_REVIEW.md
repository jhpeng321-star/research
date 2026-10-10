# WMN1_LOG_CHECK_REVIEW.md

Reviewer-facing notes for `tools/wmn1_log_check.py` (standalone stdlib
integrity checker for a raw IdeaATrainer `train_log.jsonl`) and its draft
tests `tools/tests/test_wmn1_log_check.py`.

Status: **CODE ONLY deliverable.** Static syntax/lint checks were run on the
owned sources only. The draft tests were **NOT executed** by the authoring
agent (no pytest/unittest, no module import as execution); root will run the
exact synthetic checks later under the accepted guard. This document and the
tool write **no approval**: the checker reports raw-file format integrity
only.

## Schema source (read from trainer source text, never from run logs)

The exact per-step record is `IdeaATrainer._log_step` in
`output/wd01/wm_n1/train/idea_a.py` (lines 2224-2251), appended to
`<out>/train_log.jsonl` opened in `"w"` mode at fit start (line 2395):

- `step`: `int`, incremented **before** logging, so a fresh fit writes
  exactly `1..N` in order (`self.step = 0` at line 1563, `self.step += 1`
  before `_log_step` at lines 2416-2417).
- `total`: `float` (loss of the step; `None` only if `self.losses` were
  empty, which never happens on a real fit step - the checker rejects
  non-finite/null `total`).
- `terms`: dict of the five required loss terms `residual_nll`, `rollout`,
  `arrival`, `correction_crps`, `response_contrast`
  (`REQUIRED_LOSS_TERMS`, idea_a.py:330; produced by `loss_terms`,
  idea_a.py:1985-2010), each a plain float (`float(v.detach())`).
- `contrast_support`: dict or `null` (presence/type checked only).
- `s_per_step`: `float` step seconds.
- `cuda_mem_bytes` / `cuda_peak_bytes`: non-negative `int` when the actual
  device is cuda **and** CUDA is available, else `null`
  (idea_a.py:2232-2237).
- Lines are `json.dumps(record, sort_keys=True)` + `\n`.

## What the checker validates (fail-closed, raw every-step)

- Exact raw record count equals `--expected-steps` (40 or 50); steps are
  exactly `1..N` in order, so duplicates, out-of-order, missing and extra
  steps all reject.
- All five required terms present in every record, finite (NaN/Infinity
  reject, including the non-standard `NaN`/`Infinity` JSON literals that
  `json.loads` accepts), and **non-zero every step**.
- `response_contrast` strictly positive every step.
- **Negative finite `residual_nll` is legitimate and accepted** (excess NLL
  form).
- Booleans rejected for every numeric field (`True`/`False` are JSON
  booleans, not numbers); `step` must be a plain integer.
- Malformed lines (invalid JSON, blank lines, non-object records), missing
  keys/numerics, extra unknown top-level keys, truncated last record, and
  empty files reject. No repair, no floors, no substitution of summaries
  for per-step validation.
- Device: the log carries **no literal device string**; the actual device is
  signaled by `cuda_mem_bytes`/`cuda_peak_bytes`. `--expected-device cpu`
  requires both fields `null` in **every** record; `--expected-device cuda`
  requires both fields non-null non-negative integers in **every** record.
  Any deviation (e.g. one cuda record inside a cpu log) is a mixed/actual
  device rejection.

## Report content (evidence, not approval)

The report (stdout JSON, or exclusive `--out` JSON) contains: the raw file's
SHA-256 (streamed bytes) and byte count, record count, per-term
first/last/min/max/mean plus a simple least-squares slope/trend label and a
negative-value count, the full failure list, and the requested mode. Trends
are evidence for the ROOT decision **only**; they implement no automatic
stability threshold and no acceptance gate. The tool never writes an
approval verdict.

## CLI

```
python tools/wmn1_log_check.py --log <explicit path to train_log.jsonl> \
    --expected-steps {40|50} --expected-device {cpu|cuda} [--out <report.json>]
```

- The log path must be given explicitly; there is no default path and no
  search.
- `--out` refuses to overwrite an existing file (exit code 2, file left
  byte-identical); without `--out` the JSON report goes to stdout.
- Exit codes: `0` integrity pass, `1` integrity fail, `2` usage/IO error
  (missing file, existing `--out`, invalid arguments).
- Stdlib only (`argparse`, `hashlib`, `json`, `math`, `pathlib`); no project
  imports, no data execution, no GPU access.

## Non-goals

- No repair/flooring of values, no acceptance or approval decision, no
  model-quality judgement, no reading of HOLDOUT/CAL/sealed or real run
  artifacts, no trainer or architecture modification.
- `contrast_support` is validated for presence/type only; deep semantic
  validation is intentionally out of scope for a raw-log integrity pass.

## Draft tests

`tools/tests/test_wmn1_log_check.py` holds pytest-style synthetic tests
covering: valid CPU40/CUDA50 logs (negative `residual_nll` accepted), wrong
count, extra step, duplicate/out-of-order/zero-start steps, boolean term and
boolean step, NaN/Infinity, zero term, non-positive `response_contrast`,
missing term, malformed/truncated/blank/empty logs, null `total`, negative
`s_per_step`, extra unknown key, mixed device, cuda-vs-null fields, mode
mismatch, SHA-256 correctness, exclusive `--out` (overwrite refused),
missing log file, and required/choice-validated CLI arguments. They import
only the checker module and the stdlib; all fixtures are synthetic records
in the exact schema. **Not run by the authoring agent.**

## Correction log

- 2026-10-02 (test fixture only): `test_missing_numeric_total_null_rejected`
  called `_record(1, total=None)`, but the helper treats a `None` `total`
  argument as "compute the sum", so the fixture silently serialized a valid
  float and never exercised a JSON `null` `total`. Fixed by building a valid
  record and setting `record["total"] = None` explicitly before writing, so
  `json.dumps` emits a literal `null`. The test assertion and
  `tools/wmn1_log_check.py` are unchanged. Runtime pytest remains with root
  under the accepted guard; only static checks were run on the changed
  sources.
