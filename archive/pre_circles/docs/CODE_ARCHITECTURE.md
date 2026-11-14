# Code architecture and ownership

The repository root contains only general training/evaluation entry points and
project metadata. Scientific experiment drivers live under `experiments/`; shared
behavior belongs in importable packages.

## Runtime layers

1. `config/` owns immutable names, scenario defaults, and configuration merging.
2. `env/` owns SUMO lifecycle, observations, rewards, action execution, snapshots,
   the IDM residual engine, and the wall supervisor state machine.
3. `policies/` owns evaluation-time controllers. A policy may choose actions, but
   must not redefine environment metrics or experiment labels.
4. `counterfactual/` owns paired-replay contracts, treatment/outcome definitions,
   feasibility, readiness, and predictor models. It contains no CLI orchestration.
5. `idm/` owns the frozen IDM provenance/seed contract and reusable pilot metrics.
6. `experiments/idm/` and `experiments/counterfactual/` compose those packages
   into resumable commands and artifact summaries.
7. `tools/` contains maintained diagnostics and builders that are not paper
   treatments. Invoke them as modules where possible: `python -m tools.<name>`.
8. `tests/` verifies component behavior and cross-layer scientific contracts.

## Dependency direction

```text
config
  ↓
env ← policies
  ↓      ↓
idm   counterfactual
   \    /
 experiments
```

Lower layers must not import experiment drivers. A narrowly reusable helper
discovered in multiple drivers should move into `idm/`, `counterfactual/`, or
`experiments/io.py` depending on whether it represents science or orchestration.

## Root entry points

- `train.py` — RLlib training.
- `evaluate_policy.py` — policy evaluation in SUMO.
- `evaluate_sumo_baseline.py` — uncontrolled SUMO baseline evaluation.
- `evaluation.py` — reusable offline metric calculations.
- `models.py` — RLlib/Torch model definitions.

New paper-specific commands must not be added to root.

## Quality gates

```powershell
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
```

Ruff’s frozen baseline covers syntax, undefined names, duplicate definitions,
unused imports/variables, and core style errors. Expanding it to more opinionated
rules requires a separate reviewed change so scientific code is not silently
rewritten.

## Artifact rules

- `runs/` contains ignored experiment state and evidence.
- `output/`, generated paper PDFs/zips, and rendered PPTX files are ignored.
- Source Markdown, TeX, BibTeX, figures, scenario XML, and deck builders are
  versioned.
- Every paper-scale artifact must record seeds, schema/contract version, treatment
  parameters, and development/validation/final-test role.
