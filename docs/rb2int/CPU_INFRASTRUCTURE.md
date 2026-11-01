# Extended-panel infrastructure verification — 2026-11-01

Command (existing Miniconda interpreter, local CPU):

`python -B -m pytest -q tests/test_rb2int_panel.py tests/test_rb2int_inputs.py tests/test_rb2int_uncertainty.py tests/test_rb2int_gate_bootstrap.py tests/test_rb2int_predictions.py tests/test_rb2int_comparison.py tests/test_rb2int_runtime_isolation.py tools/rbeval/test_core.py tools/rbeval/test_score.py tests/test_rbint_compare.py tests/test_rbint_endpoint_protocol.py tests/test_rbint_source.py`

**56 PASS / 1 explicit skip / 24.39s.** Skip: historical GPU canary receipt is not
fetched in this new worktree (`tools/rbeval/test_score.py:73`). No GPU/SUMO/new
inference, training or job dispatch was involved. New tool/test Ruff check PASS.

Covered: metadata-only panel selection and four singleton disclosures; no FIT
world in final roster; immutable registration/release/input hashes; explicit
development-bridge versus final execution purpose; missing source-counter
rejection; numeric VID versus packed-slot identity; causal motion history;
frozen-runtime recipe precedence in a fresh subprocess; fixed-stratum counts;
unscored same-cell peer regression; finite-K variation cannot rescue singleton
factual cells; complete K8 and duplicate-copy guards; delay ledger consistency;
all-arm world aggregation; nonlinear group rate-distance; point versus supported
switching status; existing detector/source/endpoint-protocol checks.

This verifies the evaluation infrastructure. Candidate integration, exact worker
artifact exposure clearance, staged real-runtime smoke/parity, inference and the
four-model final analysis remain pending.
