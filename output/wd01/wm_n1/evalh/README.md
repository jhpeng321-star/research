# evalh — WM-N1 evaluation harness and GPU timing gate

Design `WM_N1_DESIGN.md` sections 6-7; registration `REGISTRATION_WM_N1.md`
section 4.  CPU-only, zero SUMO, synthetic tests.  The model package (e1) is
built in parallel, so the harness codes against the abstract
`evalh.protocol.Predictor` protocol and ships a trivial climatology reference.

## Layout

| file | role |
| --- | --- |
| `protocol.py` | `Predictor` protocol, `OutcomeRow`, `RolloutRow`, `RolloutBatch`, the 19 quantile levels |
| `climatology.py` | trivial reference `ClimatologyPredictor` (per-(tau, policy) empirical quantiles, persistence rollout) |
| `predictors.py` | predictor loader by `module:attr` import path |
| `sources.py` | `EvalSource` protocol + deterministic `SyntheticSource` |
| `metrics.py` | the registration metrics and the frozen decision rule |
| `seals.py` | CAL read-once, chunk-5 TEST gate, HOLDOUT refusal |
| `run_eval.py` | `--split {train_oof,cal,final_test}` CLI |
| `timing_gate.py` | the 1-hour GPU timing gate |
| `tests/` | synthetic CPU tests |

## Metrics (each cites the registration line)

`metrics.py` implements, and cites in its docstrings:

* paired dV CRPS with **comonotone** quantile pairing (`q_a - q_b` at the same
  levels) — REGISTRATION_WM_N1.md:82-84, design:201-204;
* sign accuracy on `|dY| > TRAIN-median` pairs, with a **world-bootstrap 90 %**
  CI of the difference — REGISTRATION_WM_N1.md:83-84,92;
* level CRPS / MAE — REGISTRATION_WM_N1.md:85-86,92;
* 10-90 coverage and PIT — REGISTRATION_WM_N1.md:92-96, design:199-200;
* short-horizon position / speed RMSE — REGISTRATION_WM_N1.md:87,
  design:195-198;
* collision and TTC<3 error per 1000 veh-steps — REGISTRATION_WM_N1.md:88,90.

`build_delta_pairs` / `delta_metrics` compare the planner variant against the
hybrid **on identical windows**; `level_crps_pair` produces the outcome-CRPS
comparison; `rollout_metrics` produces the H10-H120 s/v RMSE and the validity
counters.

## Decision rule (verbatim, REGISTRATION_WM_N1.md:79-88)

`metrics.decide()` returns `WM-N1_BEATS_HYBRID` iff **all four** hold, else
`HYBRID_STAYS`:

1. dV CRPS is lower than T's for >= 2 of the gating contrasts
   {damping-native, PI-native, wall-native (if available)} at >= 2 of the tau
   {60, 120, 300}.  The world-bootstrap 90 % CI of the difference must
   exclude 0.
2. Outcome CRPS is not worse than T's by more than 5 % at any tau.
3. H10/H30 s_RMSE is not worse than Part A v2g by more than 10 %.
4. Collisions <= T + 0.1 per 1000 veh-steps.

## Seals

`seals.py` enforces the owner constraints:

* **CAL** is read at most once per marker directory (`SEAL_CAL_READ.json`);
  a second CAL read is refused.
* **chunk-5 TEST** needs `--final-test` **and** both frozen LF-normalised
  sha256 values, and is read at most once (`SEAL_FINAL_TEST_READ.json`):
  * `REGISTRATION_CHUNK5_TEST.md` =
    `f8c7d760baedeb359c501cd90a3e2a599fb0ddfef390ea555cf304c38214b31e`
  * `REGISTRATION_WM_N1.md` =
    `666233aa1d1e466750d6d944b9102c84a2f32850ebeb4a97c9f3c9b56dd93246`
* **HOLDOUT** is always refused via the canonical
  `E:/research-dense/tools/holdout_guard.py`.

Hashing is **LF-normalised** (`\r\n` -> `\n`, lone `\r` -> `\n`), so a CRLF
checkout and an LF checkout of a frozen registration hash identically.
`--dry-run` checks the seals without consuming a read-once marker.

## Running

```
python -m evalh.run_eval --split train_oof --out RUN \
    --predictor pkg.mod:Variant --reference pkg.mod:Hybrid
python -m evalh.run_eval --split cal --out RUN ...
python -m evalh.run_eval --split final_test --final-test --out RUN \
    --chunk5-sha f8c7... --registration-sha 6662...
python -m evalh.timing_gate --out TIMING_GATE.json --device cpu --tiny
```

`run_eval` writes the per-row predictions (`rows.csv`) **before** any
aggregation, then `REPORT.json` and `REPORT.md`, then prints the verdict.

## Timing gate

`timing_gate.py` measures training-step and rollout-sample throughput (plus a
loss-decrease check) and extrapolates the full plan against the design's
budget: `BUDGET_HOURS_MAX = 50` GPU-h (design:226), a 1-hour gate limit
(design:230), `+30 %` rollout fine-tuning and `+2 GPU-h` evaluation
(design:227).  It writes `TIMING_GATE.json` with a `passed` verdict and the
citation.  `--device cpu --tiny` is the CPU test path.

## Tests

```
python -m pytest output/wd01/wm_n1/evalh/tests -q     # with WD_FORBID_D=1
```

Covers: CRPS / coverage / bootstrap-CI sanity, the decision-rule truth table
(each item alone flips the verdict), the seal refusals (wrong sha, CRLF copy
accepted via LF normalisation, second read refused, HOLDOUT refused), an
end-to-end `run_eval` on synthetic data, and a `--tiny` CPU timing-gate run.
No SUMO, no training, no real-data evaluation.
