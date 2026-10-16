"""Score configured models on configured state sets and compare with the previous best.

This is the driver behind ``python -m wms score`` and ``python -m wms eval`` (WP-I glue
between :mod:`wms.config.eval` and the per-state scorer / comparison of this package):

* every logical name (checkpoint, state file) goes through :mod:`wms.data.registry`;
* a set may be restricted to a deterministic subset (``EvalSetConfig.subset``);
* each model is rolled out once per state and arm (CAVs teacher-forced), reduced to the compact
  per-state result of :func:`wms.eval.score.evaluate_state` and kept per set;
* every non-reference model is compared with the single previous best through
  :func:`wms.eval.compare.compare`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from wms.data.registry import require_present

from .compare import DEFAULT_HORIZONS_S, ModelResult, compare
from .errors import ComparisonError, EvalError
from .predictor import RolloutFlags, WmsPredictor
from .score import ScoreSpec, score_states
from .subsets import select_states

RESULT_SCHEMA = "wms_score_result/1"


@dataclass(frozen=True)
class CheckpointPredictorFactory:
    """Picklable factory of ``predict_states`` for one checkpoint (one per worker process)."""

    path: str
    entries: bool = False
    entries_v2: bool = False
    diag: tuple[str, ...] = ()

    def __call__(self):
        diag = set(self.diag)
        flags = RolloutFlags(
            entries=self.entries, entries_v2=self.entries_v2,
            **{name: name in diag for name in DIAG_SWITCHES})
        return WmsPredictor.from_checkpoint(self.path, flags).predict_states


#: Round-5 inference-time attribution switches (config ``eval.flags.diag_*``).
DIAG_SWITCHES = ("diag_lane_gate_shared", "diag_lane_cap_off", "diag_encoder_population",
                 "diag_recorded_lanes")


def rollout_flags(config: Any) -> dict[str, bool]:
    """Predictor switches from ``EvalConfig.flags`` (projection / merge_cal come from the checkpoint)."""
    flags = getattr(config, "flags", None)
    out = {"entries": bool(getattr(flags, "entries", False)), "entries_v2": False}
    out.update({name: bool(getattr(flags, name, False)) for name in DIAG_SWITCHES})
    return out


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_model(checkpoint: Path, set_records: Mapping[str, Sequence[dict]], spec: ScoreSpec, *,
                entries: bool = False, workers: int = 1,
                diag: Sequence[str] = ()) -> dict[str, list[dict]]:
    """Per-state results of one checkpoint on every set (state order preserved)."""
    factory = CheckpointPredictorFactory(str(checkpoint), entries=entries,
                                         diag=tuple(diag))
    return {name: score_states(records, factory, spec, workers=workers)
            for name, records in set_records.items()}


def load_set_records(config: Any, sets: Sequence[str]) -> dict[str, list[dict]]:
    """State records of every requested set (subset applied), loaded once and shared by models."""
    records: dict[str, list[dict]] = {}
    for name in sets:
        spec = config.set_specs.get(name)
        if spec is None or not spec.states:
            raise EvalError(f"eval set {name!r} has no `states` file in eval.set_specs")
        records[name] = list(select_states(require_present(spec.states), spec.subset))
        if not records[name]:
            raise EvalError(f"eval set {name!r} selected no states")
    return records


def resolve_models(models: Mapping[str, str], previous_best: str) -> dict[str, str]:
    """``{model name: logical checkpoint}`` including the reference under its own name."""
    resolved = dict(models)
    resolved.setdefault(previous_best, previous_best)
    return resolved


def build_result(models: Mapping[str, str], sets: Sequence[str], *, config: Any) -> dict:
    """Score ``models`` on ``sets`` and compare each with ``config.previous_best``.

    ``models`` maps a display name to a logical checkpoint name (or an existing path).  The
    reference (``config.previous_best``) is scored too, under its own name, unless listed.
    """
    models = resolve_models(models, config.previous_best)
    reference = config.previous_best
    horizons = tuple(float(h) for h in config.horizons_s)
    spec = ScoreSpec(horizons_s=horizons, predict_horizon_s=max(horizons))
    records = load_set_records(config, sets)
    flags = rollout_flags(config)
    scored: dict[str, dict] = {}
    for name, logical in models.items():
        path = Path(require_present(logical))
        diag = tuple(name for name in DIAG_SWITCHES if flags[name])
        states = score_model(path, records, spec, entries=flags["entries"],
                             workers=int(config.workers), diag=diag)
        scored[name] = {"checkpoint": logical, "path": str(path), "sha256": sha256_of(path),
                        "states_by_set": states}
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA, "name": config.name, "sets": list(sets),
        "horizons_s": list(horizons), "previous_best": reference,
        "subsets": {s: config.set_specs[s].subset for s in sets},
        "n_states": {s: len(records[s]) for s in sets}, "models": scored, "comparisons": {}}
    ref = ModelResult(reference, scored[reference]["states_by_set"], scored[reference]["sha256"])
    compare_horizons = tuple(h for h in DEFAULT_HORIZONS_S if h in horizons)
    if not compare_horizons:
        raise ComparisonError("eval.horizons_s must include 60 and/or 120 s to compare")
    for name, block in scored.items():
        if name == reference:
            continue
        candidate = ModelResult(name, block["states_by_set"], block["sha256"])
        result["comparisons"][name] = compare(candidate, ref, sets=tuple(sets),
                                              horizons=compare_horizons)
    return result
