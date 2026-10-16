"""V5DATA: ``data.c4_base`` - train on converted C4 sources only (no legacy packs).

1. config: default off (legacy configs unchanged), the v5 config clears every inherited legacy
   table, validation of the switch;
2. ``mix_c4_sources``: shares, jam oversampling inside every source, validation naming;
3. ``build_datasets`` with ``c4_base``: no pack name is ever resolved, base source first,
   world split shared across sources, paired batches still wrapped;
4. (optional) the same on the V5DATA smoke conversions when present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from wms.config.load import load_train_config
from wms.config.schema import from_mapping, to_mapping
from wms.config.train import TrainConfig
from wms.core.errors import ConfigError

REPO = Path(__file__).resolve().parents[2]
V5 = REPO / "configs" / "c4_wm_v5.toml"
V5_BASE = REPO / "configs" / "c4_wm_v5_base.toml"
SCRATCH = Path("runs/scratch/v5data")


# ============================================================================ 1. config
def test_default_off_and_legacy_configs_keep_their_packs():
    assert TrainConfig().data.c4_base is None
    for name in ("c4_wm.toml", "c4_wm_v5_base.toml", "c4_wm_v4.toml"):
        cfg = load_train_config(REPO / "configs" / name)
        assert cfg.data.c4_base is None
        assert set(cfg.data.packs) == {"pack_v3", "dspeed"}
        assert cfg.data.switching.enabled and cfg.data.windows.train is not None
    flat = load_train_config(V5_BASE).to_legacy_flat()
    assert "c4_base" not in flat and flat["window_list"] is not None
    mapping = to_mapping(load_train_config(V5_BASE))["data"]
    assert mapping["c4_base"] is None
    assert mapping["packs"] == {"pack_v3": "pack_v3", "dspeed": "pack_dspeed"}


def test_v5_config_trains_on_v5_sources_only():
    from wms.cli.assemble import declared_data_names

    cfg = load_train_config(V5)
    data = cfg.data
    assert data.c4_base == "native_v5"
    assert data.packs == {} and not data.switching.enabled
    assert data.sidecars.identities == {} and data.sidecars.outcomes is None
    assert data.windows.train is None and data.windows.val is None and data.windows.exclude == ()
    assert {s.name: (s.path, s.mix_ratio, s.paired, s.use_val) for s in data.c4_sources} == {
        "native_v5": ("v5_native", 0.65, False, True),
        "genrand_plain": ("genrand_plain", 0.30, False, True),
        "genrand_branch": ("genrand_branch", 0.05, True, True)}
    assert declared_data_names(cfg) == {"c4_sources[native_v5]": "v5_native",
                                        "c4_sources[genrand_plain]": "genrand_plain",
                                        "c4_sources[genrand_branch]": "genrand_branch"}
    assert data.jam.oversample == 2.0 and data.jam.min_fraction == 0.3
    assert cfg.loss.paired_diff.fraction == 0.25 and cfg.loss.driver_prior.target == "textbook"
    assert cfg.selection.val_max_windows == 512 and cfg.curriculum.selection == "multi_horizon_v2"
    assert cfg.selection.select_by_long_horizon
    cfg.validate_for_training()             # validation comes from the c4 sources' split
    # registry env names the box exports
    from wms.data.registry import env_var_name
    assert [env_var_name(s.path) for s in data.c4_sources] == [
        "WMS_DATA_V5_NATIVE", "WMS_DATA_GENRAND_PLAIN", "WMS_DATA_GENRAND_BRANCH"]


@pytest.mark.parametrize("override, message", [
    (["data.c4_base=missing"], "must name a data.c4_sources entry"),
    (["data.c4_base=genrand_branch"], "must not be a paired source"),
    (["data.c4_sources=[{name='native_v5', path='v5_native', mix_ratio=0.6}, "
      "{name='genrand_plain', path='genrand_plain', mix_ratio=0.25}]"], "must equal 1 -"),
    (["data.c4_sources=[{name='native_v5', path='v5_native', mix_ratio=0.75, use_val=false}, "
      "{name='genrand_plain', path='genrand_plain', mix_ratio=0.25}]"], "use_val"),
    (["data.c4_episodes.path=x", "data.c4_episodes.mix_ratio=0.1"], "exclusive"),
])
def test_switch_validation(override, message):
    with pytest.raises(ConfigError, match=message):
        load_train_config(V5, override)


def test_switch_round_trips():
    cfg = load_train_config(V5)
    again = from_mapping(TrainConfig, to_mapping(cfg))
    assert to_mapping(again) == to_mapping(cfg)


# ============================================================================ 2. mix
class _Windows(torch.utils.data.Dataset):
    """Windows whose min HDV speed is 2 m/s (jam) or 20 m/s."""

    def __init__(self, jam_flags, tag=""):
        self.flags = list(jam_flags)
        self.tag = tag
        self.receipt = {"n_records": len(self.flags), "tag": tag}

    def __len__(self):
        return len(self.flags)

    def __getitem__(self, i):
        states = np.zeros((5, 3, 7), np.float32)
        states[..., 4] = 20.0
        if self.flags[i]:
            states[:, 1:, 4] = 2.0
        return {"states": states, "present": np.ones((5, 3), bool),
                "is_cav": np.array([[True, False, False]] * 5), "tag": self.tag}


def _src(flags, val=None, tag=""):
    return {"train": _Windows(flags, tag), "train_events": None, "receipt": {"tag": tag},
            "val": None if val is None else _Windows(val, tag + "_val")}


@pytest.mark.parametrize("factor", [1.0, 2.0])
def test_mix_c4_sources_shares_jam_and_val(factor):
    from wms.data.c4_pairs import mix_c4_sources

    native = [True] + [False] * 3                  # 25 % jam
    plain = [True] * 2 + [False] * 8               # 20 % jam
    train, val = mix_c4_sources(
        [("native_v5", _src(native, val=[False, False]), 0.70),
         ("genrand_plain", _src(plain, val=[True]), 0.25),
         ("genrand_branch", _src([False, True]), 0.05)],
        jam_oversample=factor, jam_speed_mps=5.0, jam_min_fraction=0.3)
    assert [s.name for s in train.sources] == ["native_v5", "genrand_plain", "genrand_branch"]
    assert train.shares == pytest.approx([0.70, 0.25, 0.05])
    w = np.asarray(train.stage_weights(60.0))
    for k, src in enumerate(train.sources):
        lo = train.offsets[k]
        ws = w[lo:lo + len(src.dataset)]
        assert ws.sum() == pytest.approx(train.shares[k])
        jam = np.asarray(src.dataset.flags)
        if factor == 1.0:
            assert src.jam is None and np.allclose(ws, ws[0])
        else:
            assert np.array_equal(src.jam, jam)
            assert ws[jam][0] == pytest.approx(factor * ws[~jam][0])
    assert set(train.receipt["c4_sources"]) == {"native_v5", "genrand_plain", "genrand_branch"}
    assert [s.name for s in val.sources] == ["native_v5_val", "genrand_plain_val"]
    assert len(val) == 3


def test_mix_c4_sources_without_val_or_train():
    from wms.data.c4_pairs import mix_c4_sources
    from wms.data.mixdata import MixDataRefusal

    train, val = mix_c4_sources([("native_v5", _src([False, True]), 0.8),
                                 ("genrand_plain", _src([False]), 0.2)])
    assert val is None and len(train) == 3
    with pytest.raises(MixDataRefusal):
        mix_c4_sources([("native_v5", {"train": None, "val": None}, 1.0)])


# ============================================================================ 3. assembly
def _fake_sources(monkeypatch):
    """Patch the C4 loaders: three sources sharing world ``w1`` (val in genrand_plain)."""
    from wms.data import c4_episodes, c4_pairs

    manifests = {
        "v5_native": [("w0", "train"), ("w1", "train"), ("w2", "val")],
        "genrand_plain": [("w1", "val"), ("w3", "train")],
        "genrand_branch": [("w4", "train")],
    }
    calls = {}

    def read_source_manifest(root):
        name = Path(root).name
        return {"k_slots": 3, "episodes": [{"world": w, "split_role": r}
                                           for w, r in manifests[name]]}

    def build_c4_datasets(root, **kw):
        name = Path(root).name
        calls[name] = kw
        roles = kw["role_of"]
        n_train = sum(roles[w] == "train" for w, _ in manifests[name])
        n_val = sum(roles[w] == "val" for w, _ in manifests[name])
        return _src([False, True] * n_train, val=[False] * n_val if n_val else None, tag=name)

    class _Pairs:
        def __init__(self, root, *, name, **kw):
            calls["paired"] = dict(kw, name=name)
            self.receipt = {"n_groups": 1, "n_windows": 2}
            self.horizon = 120
            self.groups = [{"rows": [0, 1]}]

    monkeypatch.setattr(c4_episodes, "read_source_manifest", read_source_manifest)
    monkeypatch.setattr(c4_episodes, "build_c4_datasets", build_c4_datasets)
    monkeypatch.setattr(c4_pairs, "PairedBranchSource", _Pairs)
    return calls


def test_build_datasets_reads_no_pack_and_mixes_v5_sources(monkeypatch, tmp_path):
    from wms.cli.assemble import build_datasets
    from wms.data.c4_pairs import PairedBatchDataset

    calls = _fake_sources(monkeypatch)
    asked = []

    def resolver(name):
        asked.append(name)
        return tmp_path / name

    cfg = load_train_config(V5, ["runtime.device=cpu"])
    train, val, resolved = build_datasets(cfg, tmp_path, resolver)
    assert sorted(asked) == ["genrand_branch", "genrand_plain", "v5_native"]
    assert set(resolved) == set(asked)
    assert isinstance(train, PairedBatchDataset) and train.fraction == 0.25
    inner = train.inner
    assert [s.name for s in inner.sources] == ["native_v5", "genrand_plain", "genrand_branch"]
    assert inner.shares == pytest.approx([0.65, 0.30, 0.05])
    assert inner.jam_oversample == 2.0 and all(s.jam is not None for s in inner.sources)
    assert inner.receipt["c4_base"] == "native_v5"
    assert inner.receipt["c4_world_split"]["overridden_to_val"] == ["w1"]
    # w1 is validation everywhere: native_v5 keeps only w0 for TRAIN, w1 + w2 for validation
    assert len(inner.sources[0].dataset) == 2
    assert [s.name for s in val.sources] == ["native_v5_val", "genrand_plain_val"]
    assert len(val.sources[0].dataset) == 2
    for name in ("v5_native", "genrand_plain", "genrand_branch"):
        assert calls[name]["k_slots"] == 3 and calls[name]["heldout"] is None
        assert calls[name]["junction_fix"] == "ffill" and calls[name]["command_known"]
    assert calls["paired"]["name"] == "genrand_branch" and calls["paired"]["k_slots"] == 3
    assert "stage_sampling" in inner.receipt


def test_order_of_sources_unchanged_when_off():
    cfg = load_train_config(V5_BASE)
    base = cfg.data.c4_base
    order = sorted(cfg.data.c4_sources, key=lambda src: src.name != base)
    assert [s.name for s in order] == [s.name for s in cfg.data.c4_sources]


# ============================================================================ 4. smoke data
_CONV = {n: SCRATCH / f"conv_{n}" for n in ("native", "generic_random", "branch")}
needs_scratch = pytest.mark.skipif(
    not all((p / "C4_EPISODES_MANIFEST.json").is_file() for p in _CONV.values()),
    reason="V5DATA smoke conversions not present")


@needs_scratch
def test_build_datasets_on_v5_smoke_conversions(tmp_path):
    from wms.cli.assemble import build_datasets

    paths = {"v5_native": _CONV["native"], "genrand_plain": _CONV["generic_random"],
             "genrand_branch": _CONV["branch"]}
    cfg = load_train_config(V5, ["runtime.device=cpu"])
    train, val, _ = build_datasets(cfg, tmp_path, lambda n: paths[n])
    names = [s.name for s in train.inner.sources]
    assert names == ["native_v5", "genrand_plain", "genrand_branch"]
    rec = train.inner[0]
    assert rec["meta"]["source"] == "c4_episodes" and rec["states"].shape[0] == 241
    assert {s.name for s in val.sources} >= {"native_v5_val", "genrand_plain_val"}
    assert train.inner.receipt["c4_base"] == "native_v5"
