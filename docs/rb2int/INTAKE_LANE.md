# LANE engineering intake (candidate, not final freeze)

Source snapshot `6b4bfbd5` includes `73a5685b` design, `a29116e9` implementation,
and staged-dependency/smoke gate `6b4bfbd5`. Merged into INT after source/diff and
HANDOFF review. Only HANDOFF.md conflicted; both INT status and the imported LANE
snapshot are preserved, with peer process ownership labeled.

- Source-backed nondecision lookahead freeze and actionStepLength/tauRemainder
  semantics; no invented generic wish decay. SUMO1.27.1 source audit retained.
- Explicit fitted-LC candidate recipe `configs/wm_phys_rb2_lane.toml`; base recipe
  remains OFF. Persistent LC population, separately causal-conditioned by own
  observed identities after IDM; sampled-profile60s replay initializes wishes.
- Frozen LC population SHA256
  `b903fa49fe1f3f1a15428ae48de5c8f86863aa5c5821dbb0acc8275c77d4483f`.
  `PACKAGE.json` declares only969400–969431 fit, no population refit and no
  transplanted saved-ID posterior. Exact artifact hashes and provenance retained.
- Worker reports only development diagnosis, no WMDATA2 or reserved outcomes
  used for round2 selection. Final artifact-by-artifact exposure clearance remains
  a release prerequisite alongside the other workers.
- Native FIT969400 preliminary smoke reduces proposals547→K8mean31.375 and
  improves precision2.74→11.16%, but recall50→11.67%. These are worker's conditional
  development diagnostics, not free-run/full-model qualification. Its disjoint
  native32/nonnative12 CPU diagnostics are still running; do not duplicate them.

INT CPU command:

`python -B -m pytest -q tests/test_lane_continuation.py tests/test_lc2013.py tests/test_lc2013_snapshot.py tests/test_lc2013_motion_points.py tests/test_lc2013_model.py tests/test_lc2013_geometry.py tests/test_lc2013_encoder.py tests/test_lc2013_validation.py tests/test_rbint_recipe.py tests/test_rbint_motion_posterior.py tests/test_rbint_lane_fast.py`

**92 PASS / 1 CUDA-capacity skip / 30.49s.** First attempt was terminated by the
30s terminal timeout after82 visible passes; process audit confirmed it had exited
before this completed verification. No inference/peer job was restarted.

Remaining lane limits: incomplete serial negotiation, internal critical leaders,
zipper unsafe-link veto and nondefault nonlegacy acceptance. The fitted support
was estimated with an earlier source kernel/pooled IDM. FULL-v5/LP still NOT READY.
