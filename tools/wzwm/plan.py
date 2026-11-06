"""Phase plan of the wz3to2 fine-tune: configs, ``--set`` overrides, profiles and resume trimming.

Phases run as separate ``python -m wms train`` invocations chained by ``model.init_from`` - the
same way R16 stage A -> stage B and the loop fine-tunes were run:

* ``A`` ``configs/c4_wm_wz3to2_ft_a.toml``: init = INIT_WZ.pt (WM2 + geometry re-init);
* ``B`` ``configs/c4_wm_wz3to2_ft_b.toml``: init = phase-A final ``wm_s.pt``.

Resume: a phase interrupted after ``done`` optimizer steps restarts from its newest weights
checkpoint (``wm_s_latest.pt`` written every ``WMS_LATEST_EVERY`` steps, or a stage checkpoint)
with the per-stage lists trimmed to the remaining steps.  Adam moments and sampler state restart
(recorded in the attempt receipt); weights, horizons, learning rates and batch sizes continue.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

PHASES = ("A", "B")
CONFIGS = {"A": "configs/c4_wm_wz3to2_ft_a.toml", "B": "configs/c4_wm_wz3to2_ft_b.toml"}
STAGE_LISTS = ("unroll_s", "stage_steps", "truncation_steps_per_stage")
OPTIM_LISTS = ("lr_per_stage", "batch_size_per_stage")
#: Profiles only shorten stage step counts (never a horizon): 'night' keeps every R16 stage but
#: cuts the 300 s stage of phase B from 60 to 24 steps so A+B fit one night on the 4090D.
#: 'fast' (results first, owner 2026-11-06 "ASAP") keeps the same stages and horizons with roughly
#: half the steps and 8 validation windows in phase B.
PROFILES = {
    "full": {},
    "night": {"B": {"curriculum.stage_steps": [40, 24]}},
    "fast": {"A": {"curriculum.stage_steps": [16]},
             "B": {"curriculum.stage_steps": [24, 8], "selection.val_max_windows": 8}},
    "smoke": {"A": {"curriculum.stage_steps": [2], "selection.val_max_windows": 2}},
}


@dataclass
class Source:
    name: str
    path: str
    mix_ratio: float
    paired: bool = False
    window_stride_s: float = 10.0

    def toml(self) -> str:
        return (f'{{name = {json.dumps(self.name)}, path = {json.dumps(self.path)}, '
                f'mix_ratio = {self.mix_ratio!r}, window_stride_s = {self.window_stride_s!r}, '
                f'use_val = true, paired = {"true" if self.paired else "false"}}}')


@dataclass
class PhaseSpec:
    phase: str
    config: str
    overrides: list = field(default_factory=list)


def parse_source(text: str) -> tuple[str, str, Optional[float]]:
    """``NAME=PATH[@MIX]`` -> (name, path, mix or None)."""
    name, sep, rest = text.partition("=")
    if not sep or not name or not rest:
        raise ValueError(f"source {text!r} must look like NAME=PATH[@MIX]")
    path, at, mix = rest.rpartition("@") if "@" in rest else (rest, "", "")
    return name.strip(), path.strip(), (float(mix) if at else None)


def mix_sources(specs: Sequence[tuple[str, str, Optional[float]]], train_episodes: Mapping[str, int],
                *, base: Optional[str] = None) -> tuple[list[Source], str]:
    """Sources with mix ratios; the base (first, or ``base``) gets ``1 - others``.

    Missing ratios are proportional to TRAIN episodes (the R14 D3 rule: a window of every source
    is drawn equally often).  At least two sources are required (``data.c4_base`` needs every
    ratio in [0, 1))."""
    if len(specs) < 2:
        raise ValueError("bind at least two sources (data.c4_base needs ratios < 1); "
                         "use run.py --split-by-arm for a single converted source")
    names = [s[0] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate source names {names}")
    base = base or names[0]
    if base not in names:
        raise ValueError(f"base {base!r} not among {names}")
    total = sum(int(train_episodes.get(n, 0)) for n in names)
    ratios = {}
    for name, _path, mix in specs:
        if mix is None:
            if total <= 0:
                raise ValueError("cannot derive mix ratios without TRAIN episodes")
            mix = int(train_episodes.get(name, 0)) / total
        ratios[name] = float(mix)
    others = sum(r for n, r in ratios.items() if n != base)
    if not 0.0 < others < 1.0:
        raise ValueError(f"non-base mix ratios sum to {others}, need (0, 1)")
    ratios[base] = 1.0 - others
    out = [Source(n, p, round(ratios[n], 6)) for n, p, _ in specs]
    fix = 1.0 - sum(s.mix_ratio for s in out if s.name != base)
    for s in out:
        if s.name == base:
            s.mix_ratio = fix
    return out, base


def phase_overrides(phase: str, *, init_from: str, sources: Sequence[Source], base: str,
                    device: str = "cuda", profile: str = "night", extra: Sequence[str] = ()) -> list[str]:
    """The ``--set`` list of one phase (data binding, init, device, profile)."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    sets = [f"model.init_from={json.dumps(str(init_from))}",
            f"runtime.device={json.dumps(device)}",
            f"data.c4_base={json.dumps(base)}",
            "data.c4_sources=[" + ", ".join(s.toml() for s in sources) + "]"]
    if any(s.paired for s in sources):
        # the WM2 values (loop recipe) when a paired-branch source is bound
        sets += ["loss.paired_diff.weight=1.0", "loss.paired_diff.fraction=0.25"]
    for key, value in PROFILES[profile].get(phase, {}).items():
        sets.append(f"{key}={json.dumps(value)}")
    sets.extend(extra)
    return sets


def remaining_lists(curriculum: Mapping, optim: Mapping, done: int) -> dict:
    """Per-stage lists of the steps left after ``done`` optimizer steps (resume).

    Returns ``{"curriculum.unroll_s": [...], ..., "optim.lr_per_stage": [...]}`` for the
    remaining stages (the current one shortened); ``{}`` when nothing remains."""
    steps = [int(s) for s in curriculum["stage_steps"]]
    if done < 0:
        raise ValueError("done < 0")
    if done >= sum(steps):
        return {}
    i, left = 0, int(done)
    while left >= steps[i]:
        left -= steps[i]
        i += 1
    out = {}
    for key in STAGE_LISTS:
        values = list(curriculum[key])
        if len(values) != len(steps):
            raise ValueError(f"curriculum.{key} length {len(values)} != {len(steps)} stages")
        out[f"curriculum.{key}"] = values[i:]
    out["curriculum.stage_steps"] = [steps[i] - left] + steps[i + 1:]
    for key in OPTIM_LISTS:
        values = optim.get(key)
        if values is not None:
            out[f"optim.{key}"] = list(values)[i:]
    return out


def resume_overrides(curriculum: Mapping, optim: Mapping, done: int) -> list[str]:
    return [f"{k}={json.dumps(v)}" for k, v in remaining_lists(curriculum, optim, done).items()]


def total_steps(curriculum: Mapping) -> int:
    return int(sum(int(s) for s in curriculum["stage_steps"]))
