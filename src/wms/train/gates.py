"""The single smoke gate (replaces check400, gate_run2, gate_s2 and gate_stages).

A gate is a short real-data training (:func:`run_gate`) followed by a fail-closed check
of its ``train_log.jsonl`` (:func:`check_log`).  One parametric :class:`GateSpec` covers
what the four generations checked separately:

* base contract of every step: finite flag, monotone gate with ``g(0) == 0``, non-empty HDV
  target mask, every numeric field finite, optional exact record counts / unroll length;
* every curriculum stage present (and, for the multi-stage gates, with exactly the
  expected number of steps);
* every enabled loss term present and finite at every step, plus optional metric
  families (traffic / entrants / projection / conservation) seen somewhere in the log;
* no term dominating the update (``dominance`` x position) and no term growing faster than
  ``growth`` between consecutive stages (per-car terms get the position term's own horizon
  growth as baseline; the traffic term is exempt while its ramp opens);
* gradient-norm shares for every required term at every stage (``loss_share.jsonl``);
* optional driver-parameter bounds and a wall-clock projection of the explicit full budget.

Gates must run on the real data path: :func:`run_gate` refuses synthetic array datasets
unless ``allow_fixture`` is passed (unit tests only).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from wms.core.errors import WmsError

from .curriculum import project_hours
from .settings import TrainSettings

#: Encoder driver-parameter value bounds (hard clamps in the driver encoder).
DRIVER_PARAM_BOUNDS = {
    "v0": (5.0, 60.0), "time_headway": (0.2, 4.0), "min_gap": (0.5, 10.0),
    "accel": (0.3, 5.0), "decel": (0.5, 10.0), "politeness": (0.0, 1.0),
    "mobil_threshold": (0.0, 2.0), "merge_gap": (2.0, 30.0), "merge_urgency": (0.1, 2.0)}
#: Metric families (substring of a logged key) the multi-stage S2 smoke must contain.
S2_FAMILIES = ("traffic", "entrant", "projection", "conservation")
#: Per-car terms whose stage growth is normalised by the position term's horizon growth.
PER_CAR_TERMS = ("position", "speed", "lane", "asym_speed", "decel_weight", "propagation")


class GateFailed(WmsError, RuntimeError):
    """A smoke gate did not pass (carries the failure list)."""


@dataclass(frozen=True)
class GateSpec:
    """Parameters of one gate flavour."""

    stages_s: tuple[float, ...] = (10.0, 30.0, 60.0, 120.0)
    smoke_stage_steps: tuple[int, ...] = (6, 6, 6, 6)
    expected_records: Optional[int] = None
    steps_per_stage: Optional[int] = 6
    unroll_steps: Optional[int] = None
    families: tuple[str, ...] = ()
    driver_params: bool = False
    stage_analysis: bool = True
    require_loss_share: bool = True
    dominance: float = 100.0
    growth: float = 10.0
    floor: float = 1.0e-6
    ramp_end_s: float = 1.0


GATE_PRESETS: dict[str, GateSpec] = {
    # the A11-style 400-step single-stage contract
    "400": GateSpec(stages_s=(10.0,), smoke_stage_steps=(400,), expected_records=400,
                    steps_per_stage=None, unroll_steps=20, stage_analysis=False,
                    require_loss_share=False),
    # the R3 chain smoke: four stages x six steps with every enabled term
    "stages": GateSpec(),
    # the WM-S2 five-stage smoke with the run-2 metric families and encoder bounds
    "s2": GateSpec(stages_s=(10.0, 30.0, 60.0, 120.0, 300.0), smoke_stage_steps=(4,) * 5,
                   steps_per_stage=4, families=S2_FAMILIES, driver_params=True,
                   require_loss_share=False, ramp_end_s=120.0),
}


@dataclass
class GateResult:
    """Outcome of a gate: failures are collected, never raised, until :meth:`require`."""

    passed: bool
    failures: list[str]
    records: int = 0
    per_stage: dict[float, dict[str, float]] = field(default_factory=dict)
    loss_share: dict[float, dict[str, float]] = field(default_factory=dict)
    projection: Optional[dict] = None
    terms: tuple[str, ...] = ()

    def require(self) -> "GateResult":
        if not self.passed:
            raise GateFailed("GATE FAIL " + "; ".join(self.failures))
        return self


def required_terms(settings: TrainSettings, *, long_horizon: bool = True) -> tuple[str, ...]:
    """Loss terms an enabled configuration must log at every step."""
    terms = ["position", "speed", "lane"]
    if settings.traffic_loss:
        terms.append("traffic")
    from .losses.braking import BrakingParams
    terms.extend(BrakingParams(
        asym_alpha=settings.asym_speed_alpha, decel_weight=settings.decel_weight,
        prop_weight=settings.propagation_weight).log_terms)
    if settings.stability_weight > 0:
        terms.append("stability")
    if settings.conservation_weight > 0:
        terms.append("conservation")
    if long_horizon:
        terms.extend(name for name, weight in (("residual_mag", settings.residual_mag_weight),
                                               ("lane_rate", settings.lane_rate_weight),
                                               ("flip_rate", settings.flip_rate_weight))
                     if weight > 0.0)
    if settings.aggregate_weight > 0.0:
        terms.extend(name for name, weight in (("agg_field", settings.aggregate_field_weight),
                                               ("agg_reward", settings.aggregate_reward_weight),
                                               ("agg_flow", settings.aggregate_flow_weight))
                     if weight > 0.0)
    return tuple(dict.fromkeys(terms))


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _check_step(row: dict, spec: GateSpec, terms: Sequence[str], failures: list[str]) -> None:
    step = row.get("step")
    if row.get("finite") is not True:
        failures.append(f"finite flag at step {step}")
    if row.get("gate_monotone") is not True:
        failures.append(f"gate not monotone at step {step}")
    if abs(float(row.get("gate_g0", 1.0))) > 0.0:
        failures.append(f"g(0) != 0 at step {step}")
    if int(row.get("valid_targets", 0)) < 1:
        failures.append(f"empty HDV target mask at step {step}")
    curve = [float(x) for x in row.get("gate_curve", [])]
    if not curve or curve[0] != 0.0 or curve[-1] < 0.0 or any(b < a for a, b in zip(curve, curve[1:])):
        failures.append(f"g(k) not monotone/non-negative at step {step}")
    if spec.unroll_steps is not None and row.get("unroll_steps") != spec.unroll_steps:
        failures.append(f"unroll_steps {row.get('unroll_steps')} != {spec.unroll_steps} at step {step}")
    for key, value in row.items():
        if _numeric(value) and not math.isfinite(float(value)):
            failures.append(f"nonfinite {key} at step {step}")
    for term in (*terms, "seconds"):
        if not _numeric(row.get(term)):
            failures.append(f"required term {term!r} absent/non-numeric at step {step}")


def _check_driver_params(rows: Sequence[dict], failures: list[str]) -> None:
    for row in rows:
        step, stats = row.get("step"), row.get("driver_param_stats")
        if not stats:
            failures.append(f"driver_param_stats missing at step {step}")
            continue
        for name, (lo, hi) in DRIVER_PARAM_BOUNDS.items():
            entry = stats.get(name)
            if entry is None:
                failures.append(f"driver param {name} missing at step {step}")
                continue
            for key in ("min", "median", "max"):
                value = float(entry[key])
                if not math.isfinite(value) or value < lo - 1.0e-6 or value > hi + 1.0e-6:
                    failures.append(f"driver param {name}.{key}={value} outside [{lo}, {hi}] "
                                    f"at step {step}")


def _stage_means(by_stage: Mapping[float, list[dict]], stages: Sequence[float],
                 terms: Sequence[str], failures: list[str]) -> dict[float, dict[str, float]]:
    per_stage: dict[float, dict[str, float]] = {}
    for stage_s in stages:
        block = by_stage.get(round(stage_s, 3))
        if not block:
            continue
        means = {}
        for term in terms:
            values = [float(r[term]) for r in block if _numeric(r.get(term))]
            if values:
                means[term] = sum(values) / len(values)
        per_stage[stage_s] = means
    return per_stage


def _check_stage_dynamics(per_stage: Mapping[float, Mapping[str, float]], stages: Sequence[float],
                          terms: Sequence[str], spec: GateSpec, failures: list[str]) -> None:
    """No dominating term; bounded stage-to-stage growth (horizon-normalised for per-car terms)."""
    for stage_s, means in per_stage.items():
        pos = means.get("position")
        if pos is None:
            continue
        denom = max(abs(pos), spec.floor)
        for term, value in means.items():
            if term != "position" and abs(value) > spec.dominance * denom:
                failures.append(f"stage {stage_s}: {term}={value:.6g} exceeds "
                                f"{spec.dominance:g}x position={pos:.6g}")
    for term in terms:
        for prev_s, cur_s in zip(stages, stages[1:]):
            prev, cur = per_stage.get(prev_s), per_stage.get(cur_s)
            if not prev or not cur or term not in prev or term not in cur:
                continue
            if term == "traffic" and cur_s <= spec.ramp_end_s:
                continue  # intentional horizon ramp, not a blow-up
            threshold = spec.growth
            if term in PER_CAR_TERMS and prev.get("position", 0.0) > spec.floor \
                    and "position" in cur:
                threshold = spec.growth * max(1.0, abs(cur["position"]) / abs(prev["position"]))
            if prev[term] > spec.floor and cur[term] > threshold * prev[term]:
                failures.append(f"{term} grew {cur[term] / prev[term]:.1f}x from stage {prev_s} "
                                f"to {cur_s} (> {threshold:g}x)")


def _check_loss_shares(path: Path, stages: Sequence[float], terms: Sequence[str],
                       failures: list[str]) -> dict[float, dict[str, float]]:
    shares: dict[float, dict[str, float]] = {}
    for name in ("loss_share.jsonl", "val_log.jsonl"):
        candidate = path.with_name(name)
        if candidate.is_file():
            for row in _read_rows(candidate):
                if "loss_share" in row and "stage_s" in row and row["loss_share"]:
                    shares[float(row["stage_s"])] = row["loss_share"]
    for stage_s in stages:
        share = shares.get(float(stage_s))
        if share is None:
            failures.append(f"loss share missing for stage {stage_s}")
            continue
        for term in terms:
            value = share.get(term)
            if not _numeric(value) or not math.isfinite(value) or value < 0:
                failures.append(f"invalid/missing loss-share {term!r} at stage {stage_s}")
    return shares


def check_log(log_path: Path, spec: GateSpec, terms: Sequence[str], *,
              full_stage_steps: Optional[Sequence[int]] = None,
              max_hours: Optional[float] = None) -> GateResult:
    """Evaluate a smoke ``train_log.jsonl`` against ``spec`` (no exception on failure)."""
    log_path = Path(log_path)
    rows = _read_rows(log_path)
    failures: list[str] = []
    if not rows:
        return GateResult(False, ["empty train log"], terms=tuple(terms))
    if spec.expected_records is not None and len(rows) != spec.expected_records:
        failures.append(f"records {len(rows)} != {spec.expected_records}")
    by_stage: dict[float, list[dict]] = {}
    for row in rows:
        _check_step(row, spec, terms, failures)
        if "stage_s" not in row:
            failures.append("row without stage_s")
        else:
            by_stage.setdefault(round(float(row["stage_s"]), 3), []).append(row)
    stages = [float(s) for s in spec.stages_s]
    missing = [s for s in stages if round(s, 3) not in by_stage]
    if missing:
        failures.append(f"curriculum stages absent from log: {missing}")
    if spec.steps_per_stage is not None:
        for stage, block in by_stage.items():
            if len(block) != spec.steps_per_stage:
                failures.append(f"stage {stage:g}: {len(block)} records != {spec.steps_per_stage}")
    seen = {key.lower() for row in rows for key, value in row.items() if _numeric(value)}
    for family in spec.families:
        if not any(family in key for key in seen):
            failures.append(f"no {family!r} term in the smoke log")
    if spec.driver_params:
        _check_driver_params(rows, failures)
    per_stage = _stage_means(by_stage, stages, terms, failures)
    if spec.stage_analysis:
        _check_stage_dynamics(per_stage, stages, terms, spec, failures)
    shares = (_check_loss_shares(log_path, stages, [t for t in terms if t != "seconds"], failures)
              if spec.require_loss_share else {})
    projection = None
    if full_stage_steps is not None and max_hours is not None:
        try:
            projection = project_hours(log_path, full_stage_steps, stages_s=stages)
        except ValueError as exc:
            failures.append(f"projection: {exc}")
        else:
            if not math.isfinite(projection["projected_hours"]) \
                    or projection["projected_hours"] > float(max_hours):
                failures.append(f"projected {projection['projected_hours']:.3f} h > {max_hours} h cap")
    return GateResult(not failures, failures, len(rows), per_stage, shares, projection, tuple(terms))


def is_synthetic_dataset(dataset: Any) -> bool:
    """True for in-memory array fixtures (``receipt['mode'] == 'arrays'``)."""
    return getattr(dataset, "receipt", {}).get("mode") == "arrays"


def run_gate(name: str, cfg: Any, out_dir: Path, *, model: Any, dataset: Any,
             val_dataset: Any = None, allow_fixture: bool = False,
             max_hours: Optional[float] = None) -> GateResult:
    """Smoke-train ``cfg`` for the preset's few steps per stage and check the resulting log.

    ``cfg`` supplies the model/loss/data recipe; the curriculum and step counts come from
    the preset.  The explicit full ``stage_steps`` of ``cfg`` (when given) are projected to
    wall-clock hours from the measured step times and refused above ``max_hours``.
    """
    from .trainer import WmSTrainer

    if name not in GATE_PRESETS:
        raise ValueError(f"unknown gate {name!r}; choose one of {sorted(GATE_PRESETS)}")
    if is_synthetic_dataset(dataset) and not allow_fixture:
        raise GateFailed("gates must run on the real data path; a synthetic array dataset was "
                         "given (pass allow_fixture=True in unit tests only)")
    spec = GATE_PRESETS[name]
    settings = TrainSettings.from_config(cfg)
    flat = cfg.to_legacy_flat()
    flat.update(unroll_curriculum_s=list(spec.stages_s), stage_steps=list(spec.smoke_stage_steps),
                loss_share_logging=True)
    smoke_cfg = type(cfg).from_legacy_flat(flat)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise GateFailed(f"gate output exists: {out_dir}; choose a fresh directory")
    WmSTrainer(smoke_cfg, model, dataset, val_dataset=val_dataset, out_dir=out_dir).fit()
    terms = required_terms(settings)
    return check_log(out_dir / "train_log.jsonl", spec, terms,
                     full_stage_steps=settings.stage_steps, max_hours=max_hours)
