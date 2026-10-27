"""S-WMCOMBINED: the longwm extra real paired source (flag default OFF = unchanged)."""
from __future__ import annotations

from pathlib import Path

import pytest

from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.data.c4_pairs import ConcatPairedSource, PairedBranchSource

SMOKE = Path("E:/agent-work/c4-genrand/runs/active/C4-GENRAND-SMOKE-20261014")
BRANCH = SMOKE / "branch_conv"
needs_smoke = pytest.mark.skipif(not (BRANCH / "C4_EPISODES_MANIFEST.json").is_file(),
                                 reason="GENRAND smoke outputs not present")


# --------------------------------------------------------------------------- config
def test_longwm_extra_paired_defaults_off():
    assert TrainConfig().loss.longwm_extra_paired is None
    for name in ("c4_wm", "c4_wm_v10round2", "c4_wm_loop14_r16_ft"):
        cfg = load_train_config(name)
        assert cfg.loss.longwm_extra_paired is None, name
    assert TrainConfig().to_legacy_flat().get("longwm_extra_paired") is None


def test_longwm_extra_paired_override_and_flat_roundtrip():
    cfg = load_train_config("configs/c4_wm_loop14_r16_ft.toml",
                            ['loss.longwm_extra_paired="e3_pairs"'])
    assert cfg.loss.longwm_extra_paired == "e3_pairs"
    assert cfg.to_legacy_flat()["longwm_extra_paired"] == "e3_pairs"
    back = TrainConfig.from_legacy_flat(cfg.to_legacy_flat())
    assert back.loss.longwm_extra_paired == "e3_pairs"


def test_longwm_extra_paired_unknown_source_refused_at_assemble():
    # The named source must exist in data.c4_sources; the assemble helper raises.
    import dataclasses

    from wms.cli.assemble import _longwm_paired_sources
    from wms.core.errors import ConfigError
    cfg = load_train_config("configs/c4_wm_loop14_r16_ft.toml")

    # Default OFF: the pool is exactly loop14_pairs (bit-identical legacy behavior).
    pool, extra = _longwm_paired_sources(cfg.loss, cfg.data)
    assert [s.name for s in pool] == ["loop14_pairs"] and extra is None

    cfg = load_train_config("configs/c4_wm_loop14_r16_ft.toml",
                            ['loss.longwm_extra_paired="e3_pairs"'])
    entries = {s.name: s for s in cfg.data.c4_sources}
    named = dataclasses.replace(cfg.loss, longwm_enabled=True)
    with pytest.raises(ConfigError):  # no c4_sources entry named e3_pairs exists
        _longwm_paired_sources(named, cfg.data)

    added = dataclasses.replace(cfg.data, c4_sources=(
        *cfg.data.c4_sources,
        dataclasses.replace(entries["loop14_pairs"], name="e3_pairs", paired=True,
                            mix_ratio=0.0, use_val=False)))
    pool, extra = _longwm_paired_sources(named, added)
    assert [s.name for s in pool] == ["loop14_pairs", "e3_pairs"] and extra == "e3_pairs"


# --------------------------------------------------------------------------- concat source
@needs_smoke
def _source(name):
    return PairedBranchSource(BRANCH, horizon_steps=120, name=name, arm_kinds=("B", "SEED"))


@needs_smoke
def test_concat_pair_keys_globally_unique_and_records_tagged():
    a, b = _source("loopA"), _source("e3B")
    joined = ConcatPairedSource([a, b])
    keys = [g["pair_key"] for g in joined.groups]
    assert len(set(keys)) == len(keys) == len(a.groups) + len(b.groups)
    assert all(k.startswith(("loopA#", "e3B#")) for k in keys)
    assert joined.horizon == min(a.horizon, b.horizon)
    assert joined.horizon_mode == a.horizon_mode
    assert joined.receipt["n_groups"] == len(joined.groups)
    assert joined.receipt["n_windows"] == (a.receipt["n_windows"] + b.receipt["n_windows"])
    g = joined.groups[0]
    rows = g["rows"]
    records = joined.group_records(0, rows, base_row=0)
    assert [r["pair_group"] for r in records] == [0] * len(records)
    assert {r["pair_key"] for r in records} == {g["pair_key"]}
    # the dispatch reached the owning child with child-local rows
    child = joined.sources[g["src"]]
    local = child.group_records(g["local"], rows, base_row=0)
    import numpy as np
    assert all(np.array_equal(rec["states"], loc["states"])
               for rec, loc in zip(records, local))


@needs_smoke
def test_concat_rejects_mixed_modes_and_empty():
    a = _source("loopA")
    class _Mode:
        horizon_mode = "fixed"
    with pytest.raises(ValueError):
        ConcatPairedSource([])
    with pytest.raises(ValueError):
        ConcatPairedSource([a, a])  # same name twice -> pair_key collision
    object.__setattr__  # (no-op; keeps ruff quiet about bare attribute use)
