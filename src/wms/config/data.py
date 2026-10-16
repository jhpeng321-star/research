"""Data-side configuration: packs, sidecars, window lists, switching mix, history, jam,
collected C4 episodes."""
from __future__ import annotations

import math
from dataclasses import field
from typing import Literal, Optional

from .schema import config_dataclass, is_finite, require

#: CAV driving during training rollouts (``teacher`` replays the recorded CAV).
CAV_DRIVE_MODES = ("teacher", "closed_loop")
#: Physics step in seconds; history strides must be multiples of it.
SIM_DT_S = 0.5


@config_dataclass
class SidecarConfig:
    """Per-pack read-only sidecars, keyed by the pack role used in ``DataConfig.packs``."""

    identities: dict[str, str] = field(default_factory=dict)
    overlays: dict[str, str] = field(default_factory=dict)
    outcomes: Optional[str] = None
    driver_sidecar: Optional[str] = None


@config_dataclass
class WindowConfig:
    """Window lists (logical names) selecting TRAIN and monitoring windows."""

    train: Optional[str] = None
    val: Optional[str] = None
    exclude: tuple[str, ...] = ()
    leave_pi_wall_out: bool = False
    val_holdout_frac: float = 0.0
    val_holdout_seed: int = 0

    def __post_init__(self) -> None:
        require(0.0 <= self.val_holdout_frac < 1.0, "data.windows.val_holdout_frac must be in [0,1)")
        if self.val_holdout_frac > 0.0:
            require(self.val is None,
                    "data.windows.val_holdout_frac and data.windows.val are mutually exclusive")
            require(self.train is not None,
                    "data.windows.val_holdout_frac requires data.windows.train")


@config_dataclass
class SwitchingConfig:
    """Switching / direct-output TRAIN pack mixed with the original packs."""

    pack: Optional[str] = None
    identities: Optional[str] = None
    driver_params: Optional[str] = None
    windows_train: Optional[str] = None
    windows_val: Optional[str] = None
    timeline_index: Optional[str] = None
    zero_shot_heldout: Optional[str] = None
    heldout_family_policy: Literal["drop", "refuse", "release"] = "drop"
    heldout_max_drop_frac: float = 0.85
    mix_ratio: float = 0.5
    event_oversample: float = 2.0
    val_frac: float = 0.1
    val_seed: int = 0
    window_fraction: float = 0.0

    @property
    def enabled(self) -> bool:
        """True when a switching pack is configured."""
        return self.pack is not None

    def __post_init__(self) -> None:
        needed = {"identities": self.identities, "windows_train": self.windows_train,
                  "timeline_index": self.timeline_index,
                  "zero_shot_heldout": (self.zero_shot_heldout
                                        if self.heldout_family_policy != "release" else "")}
        if self.pack is None:
            stray = sorted(k for k, v in needed.items() if v is not None)
            require(not stray, f"data.switching.{stray} given without data.switching.pack")
        else:
            missing = sorted(k for k, v in needed.items() if v is None)
            require(not missing, f"data.switching.pack requires data.switching.{missing}")
        require(0.0 < self.mix_ratio < 1.0, "data.switching.mix_ratio must be in (0,1)")
        require(is_finite(self.event_oversample) and self.event_oversample >= 1.0,
                "data.switching.event_oversample must be >= 1")
        require(0.0 <= self.heldout_max_drop_frac <= 1.0,
                "data.switching.heldout_max_drop_frac must be in [0,1]")
        require(0.0 <= self.val_frac < 1.0, "data.switching.val_frac must be in [0,1)")
        require(0.0 <= self.window_fraction <= 1.0, "data.switching.window_fraction must be in [0,1]")


@config_dataclass
class JamConfig:
    """Oversampling of windows containing a jam (1 = off)."""

    oversample: float = 1.0
    speed_mps: float = 5.0
    cache_dir: Optional[str] = None
    #: INFRAV5: a window is a jam window when MORE than this fraction of its recorded HDV
    #: observations are below ``speed_mps``.  0 (default) = legacy rule: any single HDV
    #: observation below ``speed_mps`` (flags ~97 % of the native pack windows -- ramp
    #: mergers stopping at the end of the acceleration lane -- so oversampling is ~a no-op).
    min_fraction: float = 0.0

    def __post_init__(self) -> None:
        require(is_finite(self.oversample) and self.oversample >= 1.0,
                "data.jam.oversample must be >= 1")
        require(is_finite(self.min_fraction) and 0.0 <= self.min_fraction < 1.0,
                "data.jam.min_fraction must be in [0, 1)")
        require(is_finite(self.speed_mps) and self.speed_mps > 0.0,
                "data.jam.speed_mps must be positive")


@config_dataclass
class C4EpisodesConfig:
    """Collected C4 SUMO episodes (Dreamer data loop) mixed into the training data.

    ``path`` is a logical data name (or absolute path) of a source directory written by
    ``python -m wms.data.c4_episodes``; ``mix_ratio`` is the sampling share of its windows
    (the other sources keep their relative shares).  Windows start every
    ``window_stride_s`` seconds from ``start_s`` (default: each episode's warmup end).
    ``use_val`` appends the source's validation episodes to the monitoring set.
    """

    path: Optional[str] = None
    mix_ratio: float = 0.3
    window_stride_s: float = 10.0
    start_s: Optional[float] = None
    use_val: bool = True

    @property
    def enabled(self) -> bool:
        """True when a collected-episode source is configured."""
        return self.path is not None

    def __post_init__(self) -> None:
        require(0.0 < self.mix_ratio < 1.0, "data.c4_episodes.mix_ratio must be in (0,1)")
        stride = self.window_stride_s
        require(is_finite(stride) and stride >= 1.0 and abs(stride - round(stride)) < 1e-9,
                "data.c4_episodes.window_stride_s must be a whole number of seconds >= 1")
        require(self.start_s is None or (is_finite(self.start_s) and self.start_s >= 0.0),
                "data.c4_episodes.start_s must be non-negative")


@config_dataclass
class C4SourceConfig:
    """One more converted C4 episode source (PAIRLOSS, PAIRLOSS_NOTES.md), e.g. GENRAND.

    ``name`` labels the source in the mix receipt; ``path`` is a logical data name (env
    ``WMS_DATA_<NAME>``, e.g. ``genrand_plain`` -> ``WMS_DATA_GENRAND_PLAIN``) or an absolute
    path written by ``python -m wms.data.c4_episodes``.  ``mix_ratio`` is the sampling share of
    its ordinary windows (0 = none; allowed for a ``paired`` source only).  ``paired`` marks a
    paired-branch source: its ``pair_key`` groups feed the paired batches of
    ``loss.paired_diff`` (one window per arm at the branch time).  Worlds keep ONE split side
    across every C4 source (validation wins on disagreement).
    """

    name: str = ""
    path: Optional[str] = None
    mix_ratio: float = 0.0
    window_stride_s: float = 10.0
    start_s: Optional[float] = None
    use_val: bool = True
    paired: bool = False

    def __post_init__(self) -> None:
        require(bool(self.name) and self.name.replace("_", "").isalnum(),
                "data.c4_sources[].name must be a non-empty identifier")
        require(self.path is not None, f"data.c4_sources[{self.name}].path is required")
        require(is_finite(self.mix_ratio) and 0.0 <= self.mix_ratio < 1.0,
                f"data.c4_sources[{self.name}].mix_ratio must be in [0,1)")
        require(self.mix_ratio > 0.0 or self.paired,
                f"data.c4_sources[{self.name}].mix_ratio must be > 0 for a non-paired source")
        stride = self.window_stride_s
        require(is_finite(stride) and stride >= 1.0 and abs(stride - round(stride)) < 1e-9,
                f"data.c4_sources[{self.name}].window_stride_s must be whole seconds >= 1")
        require(self.start_s is None or (is_finite(self.start_s) and self.start_s >= 0.0),
                f"data.c4_sources[{self.name}].start_s must be non-negative")


def history_spec(history_s: Optional[float], history_stride_s: float = 1.0,
                 dt_s: float = SIM_DT_S) -> tuple[int, int, int]:
    """Native window ticks, stride ticks and sampled ticks (same arithmetic as the legacy code)."""
    if history_s is None:
        return 10, 1, 10
    for name, value in (("history_s", history_s), ("history_stride_s", history_stride_s),
                        ("dt_s", dt_s)):
        require(is_finite(value) and value > 0, f"{name} must be finite and positive")
    stride = history_stride_s / dt_s
    require(stride >= 1 and abs(stride - round(stride)) <= 1e-6,
            "history_stride_s must be a positive multiple of dt_s")
    native = int(math.floor(history_s / dt_s + 1e-8)) + 1
    stride_ticks = int(round(stride))
    return native, stride_ticks, (native - 1) // stride_ticks + 1


@config_dataclass
class DataConfig:
    """Everything that selects and perturbs training data (logical names, never local paths)."""

    packs: dict[str, str] = field(default_factory=dict)
    sidecars: SidecarConfig = field(default_factory=SidecarConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    switching: SwitchingConfig = field(default_factory=SwitchingConfig)
    history_s: Optional[float] = None
    history_stride_s: float = 1.0
    start_shift_prob: float = 0.0
    state_noise_std_m: float = 0.0
    state_noise_std_mps: float = 0.0
    jam: JamConfig = field(default_factory=JamConfig)
    c4_episodes: C4EpisodesConfig = field(default_factory=C4EpisodesConfig)
    #: PAIRLOSS: further converted C4 episode sources (GENRAND plain / paired branches), mixed
    #: in after ``c4_episodes`` with their own sampling shares (empty = off, unchanged).
    c4_sources: tuple[C4SourceConfig, ...] = ()
    entries_v2: bool = False
    entry_pad: int = 8
    #: C4 WM v2 (WMLOSS_NOTES.md): size the entrant padding per batch from the observed
    #: entry rate x unroll horizon (1.5x + 16, at least ``entry_pad``, at most
    #: ``entry_pad_max``) and keep every blocked arrival pending (uncapped backlog), as the
    #: C4 imagination does (``wms.control.dreamer.with_entry_slots``).  Off = legacy fixed pad.
    entry_pad_auto: bool = False
    entry_pad_max: int = 256
    #: ENTRYFIX (ENTRYFIX_NOTES.md): insertion rule of the entries-v2 process
    #: (``wms.model.entries.EntryConfig.insert_rule``).  "v2" = legacy, bit-identical;
    #: "sumo_like" = SUMO departSpeed="max"/departLane="best"/departPos="base" insertion;
    #: "sumo_like_v2" = V6ENTRY: additionally SUMO's saturated-entrance semantics (test at
    #: the end of the insertion step, depart at the stop speed behind the lane's rearmost
    #: vehicle; ENTRY_DIAG.md).  Default "v2" keeps every existing run bit-identical.
    entry_insert_rule: Literal["v2", "sumo_like", "sumo_like_v2"] = "v2"
    cav_drive: Literal["teacher", "closed_loop"] = "teacher"
    #: JUNCTIONFIX (JUNCTIONFIX_NOTES.md, :mod:`wms.data.junction_fix`): junction-transit rows of
    #: the pack windows.  "off" = legacy (lane clipped to 0, frozen s); "ffill" = evaluation
    #: convention (previous lane kept, s interpolated, still loss-excluded).
    junction_fix: Literal["off", "ffill"] = "off"
    #: DATAHYG (DATAHYG_NOTES.md, :mod:`wms.data.controller_hygiene`): no designed controllers in
    #: the training windows.  "off" = legacy; "strict" = drop direct-mix windows with a damping
    #: CAV, refuse dspeed windows over merge_aimed / gap_opening segments, relabel dspeed CAVs.
    controller_hygiene: Literal["off", "strict"] = "off"
    #: V5DATA: name of the ``c4_sources`` entry that REPLACES the legacy pack data as the base
    #: TRAIN / validation source (e.g. ``"native_v5"``).  None (default) = legacy assembly,
    #: unchanged.  When set, the inherited ``packs`` / ``sidecars`` / ``windows`` /
    #: ``switching`` tables are cleared (an ``extends`` child cannot delete inherited keys) and
    #: every window comes from ``c4_sources``: the base source's ``mix_ratio`` must equal
    #: 1 - the other sources' ratios, and validation is the sources' held-out world split.
    c4_base: Optional[str] = None

    @property
    def effective_start_shift_prob(self) -> float:
        """Start-shift probability actually applied: 0 when PI/wall windows are left out."""
        return 0.0 if self.windows.leave_pi_wall_out else self.start_shift_prob

    def __post_init__(self) -> None:
        history_spec(self.history_s, self.history_stride_s)
        require(0.0 <= self.start_shift_prob <= 1.0, "data.start_shift_prob must be in [0,1]")
        require(self.state_noise_std_m >= 0.0 and self.state_noise_std_mps >= 0.0,
                "state noise stds must be non-negative")
        require(self.entry_pad >= 0, "data.entry_pad must be non-negative")
        require(self.entry_pad_max >= 1, "data.entry_pad_max must be positive")
        extra = sorted(set(self.sidecars.identities) - set(self.packs))
        require(not extra, f"data.sidecars.identities for unknown pack roles {extra}")
        extra = sorted(set(self.sidecars.overlays) - set(self.packs))
        require(not extra, f"data.sidecars.overlays for unknown pack roles {extra}")
        names = [src.name for src in self.c4_sources]
        require(len(set(names)) == len(names) and "c4_episodes" not in names,
                f"data.c4_sources names must be unique and not 'c4_episodes' (got {names})")
        if self.c4_base is not None:
            self._clear_legacy_data(names)
        total = sum(src.mix_ratio for src in self.c4_sources if src.name != self.c4_base) + (
            self.c4_episodes.mix_ratio if self.c4_episodes.enabled else 0.0)
        require(total < 1.0, "the C4 source mix ratios must sum to < 1")
        if self.packs:
            missing = sorted(set(self.packs) - set(self.sidecars.identities))
            require(not missing, f"data.sidecars.identities missing for pack roles {missing}")

    def _clear_legacy_data(self, names: list[str]) -> None:
        """V5DATA ``c4_base``: check the base source and drop the inherited legacy pack data."""
        base = self.c4_base
        require(base in names, f"data.c4_base {base!r} must name a data.c4_sources entry ({names})")
        src = self.c4_sources[names.index(base)]
        require(not src.paired, f"data.c4_base {base!r} must not be a paired source")
        require(src.use_val, f"data.c4_base {base!r} must keep use_val = true (validation split)")
        require(not self.c4_episodes.enabled, "data.c4_base and data.c4_episodes are exclusive")
        others = sum(s.mix_ratio for s in self.c4_sources if s.name != base)
        require(abs(src.mix_ratio + others - 1.0) < 1e-6,
                f"data.c4_base {base!r}: mix_ratio {src.mix_ratio} must equal 1 - {others:g} "
                "(the other c4_sources' ratios)")
        for name, value in (("packs", {}), ("sidecars", SidecarConfig()),
                            ("windows", WindowConfig()), ("switching", SwitchingConfig())):
            object.__setattr__(self, name, value)
