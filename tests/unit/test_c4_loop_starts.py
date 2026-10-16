"""C4 Dreamer loop: the policy dreams from the states it visits in SUMO.

``[data] start_source_weights`` (configs/c4_policy_v6_loop.toml) draws, per penetration, the start
source by weight and then a uniform window of it; without the key the sampler is unchanged.  A loop
resume (WMS_LOOP_REFRESH=1) tolerates the changed window set.
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from tests.unit.test_c4_policyv5 import REPO, _fake_sources, _no_jam_scan


def _wm_three(tmp_path):
    text = f"""schema_version = 2
name = "fake-loop"
extends = "{(REPO / 'configs' / 'c4_wm_v5.toml').as_posix()}"
[[data.c4_sources]]
name = "native_v5"
path = "native_src"
mix_ratio = 0.50
window_stride_s = 10.0
use_val = true
[[data.c4_sources]]
name = "genrand_plain"
path = "plain_src"
mix_ratio = 0.25
window_stride_s = 10.0
use_val = true
[[data.c4_sources]]
name = "v6_policy"
path = "branch_src"
mix_ratio = 0.25
window_stride_s = 10.0
use_val = true
"""
    path = tmp_path / "wm_loop.toml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_loop_policy_config_resolves():
    from wms.control.train_cli import _read_config

    raw = _read_config(REPO / "configs" / "c4_policy_v6_loop.toml")
    full = _read_config(REPO / "configs" / "c4_policy_v6_full.toml")
    assert raw["data"]["wm_config"] == "configs/c4_wm_v6_refresh.toml"
    assert raw["data"]["start_sources"] == ["native_v5", "genrand_plain", "v6_policy"]
    assert raw["data"]["start_source_weights"] == {"native_v5": 0.25, "genrand_plain": 0.25, "v6_policy": 0.5}
    for section in ("trainer", "counterfactual", "lanes", "reward", "policy"):
        assert raw[section] == full[section], section
    refresh = _read_config(REPO / "configs" / "c4_wm_v6_refresh.toml")
    assert "v6_policy" in [s["name"] for s in refresh["data"]["c4_sources"]]


def test_weighted_start_sources(monkeypatch, tmp_path):
    from wms.control.train_cli import _load_starts

    _fake_sources(monkeypatch, tmp_path)
    _no_jam_scan(monkeypatch)
    model = SimpleNamespace(history_s=None, history_stride_s=1.0)
    weights = {"native_v5": 0.25, "genrand_plain": 0.25, "v6_policy": 0.5}
    data = {"wm_config": _wm_three(tmp_path), "start_sources": ["native_v5", "genrand_plain", "v6_policy"],
            "start_source_weights": weights, "val_fraction": 0.25}
    train, val = _load_starts(data, [model], seed=0)
    assert train.source_weights == weights and val.source_weights == weights
    counts = {p: Counter() for p in train.penetrations}
    draws = []
    for _ in range(4000):
        index = train._draw_index()
        meta = train.metadata[index]
        draws.append(index)
        counts[0.2 if "p20" in meta["world"] else 0.4][meta["start_source"]] += 1
    # p20 has all three sources: v6_policy ~ 0.5, the others ~ 0.25 each
    p20 = counts[0.2]
    share = p20["v6_policy"] / sum(p20.values())
    assert 0.45 < share < 0.55, p20
    assert abs(p20["native_v5"] - p20["genrand_plain"]) / sum(p20.values()) < 0.06
    # p40 has no v6_policy windows: native/plain renormalised to 0.5/0.5
    assert set(counts[0.4]) == {"native_v5", "genrand_plain"}
    # unweighted (the default) is far from that share: v6_policy has 1 world of 3 at p20
    plain, _ = _load_starts({k: v for k, v in data.items() if k != "start_source_weights"}, [model], seed=0)
    assert plain.source_weights is None and plain.source_groups is None
    p20_plain = Counter(plain.metadata[plain._draw_index()]["start_source"] for _ in range(3000))
    assert p20_plain["v6_policy"] / sum(p20_plain.values()) < 0.3


def test_default_draws_unchanged(monkeypatch, tmp_path):
    """No weights: exactly the legacy draw (uniform penetration, then uniform window)."""
    from wms.control.train_cli import _load_starts

    _fake_sources(monkeypatch, tmp_path)
    _no_jam_scan(monkeypatch)
    model = SimpleNamespace(history_s=None, history_stride_s=1.0)
    train, _ = _load_starts({"wm_config": _wm_three(tmp_path)}, [model], seed=3)
    gen = torch.Generator().manual_seed(3 + 1)
    for _ in range(200):
        group = train.groups[train.penetrations[int(torch.randint(len(train.penetrations), (), generator=gen))]]
        expected = group[int(torch.randint(len(group), (), generator=gen))]
        assert train._draw_index() == expected


def test_weight_refusals(monkeypatch, tmp_path):
    from wms.control.train_cli import _load_starts

    _fake_sources(monkeypatch, tmp_path)
    _no_jam_scan(monkeypatch)
    model = SimpleNamespace(history_s=None, history_stride_s=1.0)
    wm = _wm_three(tmp_path)
    with pytest.raises(ValueError, match="one weight per start source"):
        _load_starts({"wm_config": wm, "start_sources": ["native_v5", "v6_policy"],
                      "start_source_weights": {"native_v5": 1.0}}, [model], seed=0)
    with pytest.raises(ValueError, match="needs a WM config with data.c4_base"):
        _load_starts({"wm_config": str(REPO / "configs" / "c4_wm.toml"),
                      "start_source_weights": {"native_v5": 1.0}}, [model], seed=0)


def test_loop_resume_group_check():
    from wms.control.dreamer import _same_groups

    sampler = SimpleNamespace(groups={0.2: [1, 2]}, relabel=False, target_penetration=0.2)
    assert _same_groups(sampler, {"groups": {0.2: [1, 2]}, "relabel": False, "target_penetration": 0.2})
    assert not _same_groups(sampler, {"groups": {0.2: [1, 2, 3]}, "relabel": False, "target_penetration": 0.2})
    assert not _same_groups(None, {"groups": {}})
