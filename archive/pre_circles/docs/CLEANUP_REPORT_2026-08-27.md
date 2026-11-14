# Repository cleanup report — 2026-08-27

## Outcome

The repository was upgraded from a mixed historical experiment workspace into
an IDM/counterfactual-focused codebase. The cleanup preserved all current source,
tests, scenarios, manuscript material, and `runs/` evidence. The final regression
result is 531 passing tests.

## Removed

- 199 already-deleted tracked one-off scripts whose copies in `archive/` were
  verified byte-for-byte against Git blobs;
- the 2.1 GB duplicate `archive/`, including two very large old phase logs;
- three unused local virtual environments (`.rllibvenv`, `.rllib310venv`, and
  `.mlvenv`), totaling about 2.6 GB;
- retired Krauss/ExpC/metastable `_probe_*`, `_eval_*`, `_analyze_*`, phase,
  supervisor-training, and shell/batch entry points at repository root;
- approximately 47 MB of root training/probe logs and completion sentinels;
- unused medium/fair-safe tool chains for behavior cloning, overnight baking,
  old checkpoint-talk extraction, baseline PDF generation, and medium map work;
- old root report PDFs, recovered poster/render scratch, TeX intermediates,
  empty submission staging, and agent-plan scratch.

The material removed in this pass was sent to the Windows Recycle Bin and is
recoverable until the bin is emptied. Formerly tracked files also remain in Git
history.

## Preserved

- every current modified file in `config/`, `env/`, `models.py`, `policies/`,
  `tests/`, `tools/build_longer_map.py`, and `train.py`;
- all new IDM residual, wall-supervisor, counterfactual, scenario, audit, and test
  files;
- all `runs/` checkpoints and evaluation/probe data (about 7.6 GB);
- current IEEE IV manuscript, presentation, and shareable output files;
- detailed historical Markdown reports and design records.

## Structural changes

- `README.md` and `AGENTS.md` now describe the IDM-native project rather than the
  retired Krauss phase pipeline.
- Reusable IDM provenance, seed partitions, and metrics moved from root into
  `idm/contract.py` and `idm/pilot_metrics.py`.
- The counterfactual handoff moved into `docs/`.
- `docs/INDEX.md`, `docs/LEGACY_RESEARCH_RECORD.md`, and
  `docs/ARTIFACT_MANIFEST.md` now separate current contracts, historical evidence,
  and run artifacts.
- `.gitignore` now excludes local environments, caches, transient logs, render
  scratch, and LaTeX intermediates.
- `requirements.txt` now includes the active predictor dependency
  `scikit-learn`; `requirements-dev.txt` contains the test dependency.

## Final hardening pass

- Active paper-specific commands moved from repository root into
  `experiments/idm/` and `experiments/counterfactual/`.
- All maintained Python files were formatted with Ruff.
- A frozen Ruff correctness gate now enforces syntax, import, undefined-name,
  duplicate-definition, and unused-code checks through `pyproject.toml`.
- Repeated atomic JSON writers were consolidated into `experiments/io.py` and
  covered by a replacement/cleanup regression test.
- Stale manual test runners containing references to removed tests were deleted;
  Pytest remains the only test runner.
- Generated reports, PDFs, ZIPs, and PPTX binaries are preserved locally but
  excluded from source control.
- `CODE_ARCHITECTURE.md` now freezes module ownership and dependency direction.

## Code-retention audit

Top-level symbols in the large retained modules (`env/actions.py`,
`env/features.py`, `env/rewards.py`, `policies/inflow_control.py`, and
`policies/metastable_rule.py`) were checked against active code and tests. No
unreferenced top-level mechanism was removed from these shared libraries. This
protects the current damping and wall implementations while eliminating their
obsolete one-off launchers.

## Verification

- Baseline before cleanup: 531 passed.
- Targeted IDM package/driver tests after restructuring: 14 passed.
- Maintained CLI help smoke: `train.py`, `experiments/idm/run_idm_analytical_pilot.py`, and
  `experiments/counterfactual/generate_counterfactual_escalation_headroom.py` passed.
- Final repository suite: 531 passed.
- Final hardening suite: 532 passed after adding the shared-writer regression test.
- Ruff format check and lint check: passed.
- `git diff --check`: passed.
- Frozen IDM manifest imports and resolves the correct repository/scenario paths.

## Known local issue

The pre-existing `.pytest_cache` directory has a Windows ACL that denies access
even to the active process. It could not be moved to the Recycle Bin and causes
a harmless Pytest cache warning. It is ignored by Git and does not affect test
execution or results. Repairing its ownership may require an elevated Windows
session; no elevation was attempted during this cleanup.
