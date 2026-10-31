"""Load the single adopted baseline/scenario configuration."""
from dataclasses import dataclass
import json
from pathlib import Path

from .research_assets import resolve_research_path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "baselines" / "circles" / "adapter.json"


@dataclass(frozen=True)
class AdapterConfig:
    model_path: Path
    model_sha256: str
    control_dt_s: float
    physics_dt_s: float
    sensor_range_m: float
    acceleration_min_mps2: float
    acceleration_max_mps2: float
    speed_mode: int
    network_path: Path
    scenario_template_path: Path
    episode_seconds: int
    demand_end_seconds: int
    ramp_vph: int
    edge_ids: dict


def load_config(path: Path | None = None) -> AdapterConfig:
    path = Path(path or CONFIG_PATH).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.pop("schema", None) != "circles_direct_adapter/1":
        raise ValueError("unsupported adapter schema")
    if data.pop("working_adopted", None) is not True or data.pop("scientifically_accepted", None) is not False:
        raise ValueError("adoption status must not imply scientific acceptance")
    cfg = AdapterConfig(**{**data,
        "model_path": (path.parent / data["model_path"]).resolve(),
        "network_path": (ROOT / resolve_research_path(data["network_path"])).resolve(),
        "scenario_template_path": (ROOT / data["scenario_template_path"]).resolve(),
    })
    # These are the adopted transfer contract, not tunable command defaults.
    if (cfg.control_dt_s, cfg.physics_dt_s, cfg.sensor_range_m,
        cfg.acceleration_min_mps2, cfg.acceleration_max_mps2, cfg.speed_mode) != (1.0, .5, 100., -3., 1.5, 31):
        raise ValueError("configuration differs from the adopted direct interface")
    if len(cfg.model_sha256) != 64 or any(c not in "0123456789abcdef" for c in cfg.model_sha256):
        raise ValueError("invalid model digest")
    if not 0 < cfg.demand_end_seconds < cfg.episode_seconds or cfg.ramp_vph <= 0:
        raise ValueError("invalid scenario horizon or demand")
    return cfg
