"""S-SUMOES frozen design config: exact workload, budget and seed block.

Every number here is part of the registered design (docs/paper_results/
SUMOES_DESIGN.md). The shipped TOML keeps ``enabled = false``; only the
registered run task may flip it after the exposure audit and ACK.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

LOOP_TRAIN_FIRST = 966_000
LOOP_TRAIN_LAST = 969_999
WMDATA_BLOCK = range(969_400, 969_440)
PROHIBITED_SEED_RANGES = ((964_000, 964_199), (962_000, 962_199), (977_000, 977_999))
PENETRATIONS = (0.05, 0.1, 0.2, 0.4)
DEMANDS = (1900, 2400, 2700)
CELL_CYCLE = tuple((p, d) for p in PENETRATIONS for d in DEMANDS)
# Variant (b) panel: 4 penetrations x the two demand corners (as the S-WMDATA corner cells).
PANEL_CELLS = tuple((p, d) for p in PENETRATIONS for d in (1900, 2700))


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SumoEsConfig:
    enabled: bool = False
    warmup_s: int = 300
    control_s: int = 600
    episode_charge_sim_s: int = 900
    pairs_per_update: int = 3
    gradient_worlds_per_update: int = 6
    panel_worlds: int = 8
    wm_screen_directions: int = 8
    wm_screen_horizon_s: int = 60
    wm_screen_starts: int = 8
    sigma_target_residual_rms_mps2: float = 0.10
    sigma_min_mps2: float = 0.02
    sigma_halve_every_accepted: int = 4
    adam_lr: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    step_rule: str = 'MAIN1440_sign_weighted_mean_unit'
    max_updates: int = 16
    cap_sim_s: int = 650_000
    seed_block_first: int = 969_440
    seed_block_last: int = 969_543
    direction_seed: int = 7_000_031

    def __post_init__(self) -> None:
        if self.step_rule != 'MAIN1440_sign_weighted_mean_unit':
            raise ConfigError('only the registered MAIN1440 normalized ES step is active')
        if self.episode_charge_sim_s != self.warmup_s + self.control_s:
            raise ConfigError("episode charge must equal warmup + control seconds")
        for name in ("pairs_per_update", "gradient_worlds_per_update", "panel_worlds",
                     "max_updates", "wm_screen_directions", "wm_screen_horizon_s",
                     "wm_screen_starts", "cap_sim_s"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive")
        if self.wm_screen_directions < self.pairs_per_update:
            raise ConfigError("wm_screen_directions must be >= pairs_per_update")
        if not 0.0 < self.sigma_min_mps2 <= self.sigma_target_residual_rms_mps2:
            raise ConfigError("sigma_min must be in (0, target]")
        if self.sigma_halve_every_accepted <= 0 or self.adam_lr <= 0:
            raise ConfigError("sigma schedule and adam_lr must be positive")
        if not (LOOP_TRAIN_FIRST <= self.seed_block_first <= self.seed_block_last
                <= LOOP_TRAIN_LAST):
            raise ConfigError("seed block must lie inside the C4_LOOP_TRAIN block")
        needed = self.gradient_seed_count() + self.panel_worlds
        available = self.seed_block_last - self.seed_block_first + 1
        if available != needed:
            raise ConfigError(f"seed block has {available} seeds; design needs {needed}")
        seeds = self.all_seeds()
        if any(seed in WMDATA_BLOCK for seed in seeds):
            raise ConfigError("seed block overlaps the registered S-WMDATA sub-block")
        for lo, hi in PROHIBITED_SEED_RANGES:
            if any(lo <= seed <= hi for seed in seeds):
                raise ConfigError(f"seed block intersects prohibited range {lo}-{hi}")
        planned = self.planned_sim_s()
        if planned > self.cap_sim_s:
            raise ConfigError(f"planned {planned} sim-s exceeds cap {self.cap_sim_s}")

    def gradient_seed_count(self) -> int:
        return self.max_updates * self.gradient_worlds_per_update

    def all_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed_block_first, self.seed_block_last + 1))

    def gradient_seeds(self, update: int) -> tuple[int, ...]:
        if not 0 <= update < self.max_updates:
            raise ConfigError(f"update {update} outside 0..{self.max_updates - 1}")
        start = update * self.gradient_worlds_per_update
        return self.all_seeds()[start:start + self.gradient_worlds_per_update]

    def panel_seeds(self) -> tuple[int, ...]:
        return self.all_seeds()[-self.panel_worlds:]

    def cell_for(self, seed: int) -> tuple[float, int]:
        """Variant (b) mapping. Gradient world j of update u -> CELL_CYCLE[(2j + u%2) % 12]
        (W=6: every update spans all 4 penetrations and all 3 demands; two consecutive
        updates cover all 12 cells). Panel world i -> PANEL_CELLS[i % 8]."""
        if seed not in self.all_seeds():
            raise ConfigError(f"seed {seed} outside the registered block")
        panel = self.panel_seeds()
        if seed in panel:
            return PANEL_CELLS[panel.index(seed) % len(PANEL_CELLS)]
        position = seed - self.seed_block_first
        update, world = divmod(position, self.gradient_worlds_per_update)
        return CELL_CYCLE[(2 * world + update % 2) % len(CELL_CYCLE)]

    def episodes_per_update(self) -> int:
        return 2 * self.pairs_per_update * self.gradient_worlds_per_update + self.panel_worlds

    def planned_episodes(self) -> int:
        return self.panel_worlds + self.max_updates * self.episodes_per_update()

    def planned_sim_s(self) -> int:
        return self.planned_episodes() * self.episode_charge_sim_s

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "SumoEsConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    @classmethod
    def load(cls, path: str | Path) -> "SumoEsConfig":
        from wms.core import toml

        text = Path(path).read_text(encoding="utf-8")
        section = toml.loads(text).get("sumoes", {})
        if not isinstance(section, dict):
            raise ConfigError("config must contain a [sumoes] table")
        return cls.from_dict(section)
