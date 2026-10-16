"""Evaluation configuration: state sets, horizons, compared models and rollout flags."""
from __future__ import annotations

from dataclasses import field
from typing import Optional

from .schema import config_dataclass, is_finite, require

#: The only sets the previous-best comparison ranks.
RANKED_SETS = ("wall_same", "wall_pilot", "seen_controls")


@config_dataclass
class RolloutFlags:
    """How a checkpoint is rolled out when scored.

    The ``diag_*`` switches are round-5 inference-time attribution diagnostics
    (default OFF; scoring is unchanged when they are off).
    """

    entries: bool = False
    projection: bool = False
    merge_cal: bool = True
    rollout_mode: str = "residual"
    diag_lane_gate_shared: bool = False
    diag_lane_cap_off: bool = False
    diag_encoder_population: bool = False
    diag_recorded_lanes: bool = False

    def __post_init__(self) -> None:
        require(self.rollout_mode in ("residual", "hybrid"),
                "eval.flags.rollout_mode must be residual or hybrid "
                "(driver_params scoring was REJECTED, change-log #15)")


SUBSET_KINDS = ("first", "first_per_world")


@config_dataclass
class EvalSetConfig:
    """Where one named evaluation state set comes from (logical data names).

    ``subset`` optionally restricts a materialised state file to a deterministic subset:
    ``first:N`` (the first N states by ``(world, t0_s, key)``) or ``first_per_world:N`` (the
    earliest state of each of the first N worlds; ``first_per_world:20`` is the 20-state quick
    diagnostic subset).
    """

    pack: Optional[str] = None
    windows: Optional[str] = None
    identities: Optional[str] = None
    states: Optional[str] = None
    subset: Optional[str] = None

    def __post_init__(self) -> None:
        if self.subset is not None:
            kind, _, count = self.subset.partition(":")
            require(kind in SUBSET_KINDS and count.isdigit() and int(count) >= 1,
                    f"eval.set_specs.*.subset must look like first:N or first_per_world:N, "
                    f"got {self.subset!r}")


@config_dataclass
class EvalConfig:
    """Complete evaluation configuration (schema version 2)."""

    schema_version: int = 2
    name: str = "wms-eval"
    sets: tuple[str, ...] = RANKED_SETS
    set_specs: dict[str, EvalSetConfig] = field(default_factory=dict)
    horizons_s: tuple[float, ...] = (10.0, 30.0, 60.0, 120.0)
    models: dict[str, str] = field(default_factory=dict)
    previous_best: str = "ckpt_c1fast"
    flags: RolloutFlags = field(default_factory=RolloutFlags)
    workers: int = 1
    out_dir: str = "runs/active/wms_eval"

    def __post_init__(self) -> None:
        require(self.schema_version == 2, f"unsupported schema_version {self.schema_version}")
        require(bool(self.sets), "eval.sets must not be empty")
        require(bool(self.horizons_s) and all(is_finite(h) and h > 0 for h in self.horizons_s),
                "eval.horizons_s must be positive numbers")
        unknown = sorted(set(self.set_specs) - set(self.sets))
        require(not unknown, f"eval.set_specs for sets not listed in eval.sets: {unknown}")
        require(self.workers >= 1, "eval.workers must be positive")
