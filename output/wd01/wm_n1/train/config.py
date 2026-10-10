"""WM-N1 training configuration.

Defaults are copied from ``WM_N1_DESIGN.md`` section 3 (losses, data units,
uncertainty, selection) and section 7 (15 epochs x 5 members), and from
``model.net.WmN1Config`` for the architecture.  Keys are the config contract;
a YAML or JSON file may override any subset.

Interface note (deviation D1, see ``train/README.md``): the stub wrote the
CAL-selection flag as ``caliber``; the frozen ``INTERFACES.md`` Config-keys
table names it ``calibrate_on_cal``.  This implementation uses
``calibrate_on_cal``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

N_MEMBERS = 5
# Independent member seeds; recorded BEFORE training (registration A1).
DEFAULT_SEEDS = (101, 202, 303, 404, 505)

# Member-1-first training order (REGISTRATION_WM_N1_AMENDMENT_1.md A1).
MEMBER_ORDER = (1, 2, 3, 4, 5)


class ConfigError(ValueError):
    """A config file violates the WM-N1 config contract."""


@dataclass
class TrainConfig:
    """Training + architecture config (defaults = WM_N1_DESIGN.md section 3)."""

    # --- architecture (model.net.WmN1Config) ---
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ffn: int = 768
    n_mix: int = 3
    radius_m: float = 150.0
    bias_hidden: int = 64
    counter_channels: int = 128
    dropout: float = 0.0
    # --- schedule (design section 7 / timing_gate.MemberConfig) ---
    n_members: int = N_MEMBERS
    seeds: tuple = DEFAULT_SEEDS
    epochs: int = 15
    steps_per_epoch: int = 2000
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    # --- losses (design section 3) ---
    lambda_contrast: float = 1.0            # design:115, lambda_c = 1
    w_step: float = 1.0
    w_lane: float = 1.0
    w_arrival: float = 1.0
    w_outcome: float = 1.0
    w_collision: float = 1.0
    # --- rollout fine-tuning (design section 3b/7) ---
    rollout_horizon_steps: int = 20
    rollout_fraction: float = 0.30
    scheduled_sampling: float = 0.0
    # --- selection (design:121-122: CAL early stopping, world-grouped) ---
    early_stop_patience: int = 3
    calibrate_on_cal: bool = False          # CAL is diagnostics only here
    # --- io ---
    device: str = "cuda"
    out_dir: str = "runs/wm_n1"
    resume: str | None = None

    # ------------------------------------------------------------- helpers --
    def seed_for_member(self, member: int) -> int:
        """Registered seed for ensemble ``member`` (1-based, amendment A1)."""
        if not 1 <= int(member) <= len(self.seeds):
            raise ConfigError(
                f"member {member} has no registered seed in {self.seeds!r}")
        return int(self.seeds[int(member) - 1])

    def model_config(self):
        """Build the matching :class:`model.net.WmN1Config`."""
        from model.net import WmN1Config

        return WmN1Config(
            d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_layers,
            ffn=self.ffn, n_mix=self.n_mix, radius_m=self.radius_m,
            bias_hidden=self.bias_hidden, counter_channels=self.counter_channels,
            dropout=self.dropout)

    def loss_weights(self) -> dict:
        """Loss weights handed to :func:`model.losses.total_loss`."""
        return dict(step=self.w_step, lane=self.w_lane, arrival=self.w_arrival,
                    outcome=self.w_outcome, collision=self.w_collision,
                    contrast=self.lambda_contrast)

    # ------------------------------------------------------------- io ------
    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        """Build from a mapping; unknown keys are a contract error."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(
                f"unknown TrainConfig key(s): {unknown}; known keys: "
                f"{sorted(known)}")
        kwargs = {}
        for key, value in data.items():
            if key == "seeds":
                value = tuple(int(x) for x in value)
            kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path) -> "TrainConfig":
        """Load YAML/JSON overrides; missing keys keep the defaults."""
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(raw) or {}
        else:
            data = json.loads(raw) if raw.strip() else {}
        if not isinstance(data, dict):
            raise ConfigError(f"config {p} must be a mapping, got {type(data)}")
        return cls.from_dict(data)

    def save(self, path) -> Path:
        """Write the resolved config as JSON (used for the run receipt)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
        return p

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seeds"] = [int(s) for s in self.seeds]
        return d
