"""WM-A: the WM-S map port on wz3to2 (road geometry, physics legality, entries, exits, data pipeline).

CPU, no SUMO, no GPU.  The wz3to2 road comes from N1's descriptor when c4/nm-abstract is merged
(``tools.coopmwm.map_descriptor``) and from the identical stub before that; every test that
switches the road uses :func:`wms.core.road.road_scope`, so the process road stays the original.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

torch = pytest.importorskip("torch")

from tests.wz import synth  # noqa: E402
from wms.core import road as R  # noqa: E402

SPEC = ROOT / "assets" / "research" / "newmap_candidates" / "wz3to2" / "MAP_SPEC.json"


def wz_road(k_slots=None):
    road = R.road_from_descriptor(R.stub_descriptor("wz3to2"))
    return road if k_slots is None else R.with_overrides(road, k_slots=k_slots)


@pytest.fixture(autouse=True)
def _clean_road():
    R._reset_for_tests()
    yield
    R._reset_for_tests()


# ------------------------------------------------------------------ geometry
def test_original_road_equals_the_frozen_constants():
    from wms.model import entries
    from wms.physics import batched
    from wms.physics.config import PhysicsConfig

    o = R.MERGE_ORIGINAL
    assert o.is_original and o.closing_change_restricted and not o.into_closing_allowed
    assert o.entry_s_m == entries.ENTRY_S_M
    assert o.entry_lane_speed_mps == entries.SUMO_LANE_SPEED_MPS
    assert o.road_end_m == batched.MODEL_FRAME_ROAD_END_M
    assert tuple(e[1] for e in o.lane_extents_s_m) == batched.MODEL_FRAME_LANE_LENGTHS
    p = PhysicsConfig()
    assert o.merge_zone_s_m == (p.merge_zone_s0_m, p.merge_zone_s1_m)
    assert (o.closing_lane, o.target_lane) == (p.merge_lane, p.merge_target_lane)
    assert R.current_road() is o                     # no env map -> the original


def test_wz3to2_road_from_the_descriptor():
    road = wz_road()
    assert road.name == "wz3to2" and road.kind == "lane_drop"
    assert road.lane_extents_s_m == ((-2620.56, 1379.44), (-2620.56, 1379.44), (-2620.56, 75.44))
    assert road.entry_s_m == (-2620.56,) * 3 and road.entry_edge_end_s_m == (75.44,) * 3
    assert road.road_end_m == 1379.44 and road.closing_lane_end_m == 75.44
    assert road.closing_change_zone_s_m == (-2620.56, 75.44) and road.into_closing_allowed
    assert not road.closing_change_restricted
    assert road.entry_lane_speed_mps == (31.29,) * 3 and road.k_slots == 512
    assert road.rsu_mainline_s_m == (-800.0, 225.44) and road.rsu_closing_s_m == (-378.66, 75.44)
    assert road.entrances == ((0, 1), (2,))
    assert json.loads(json.dumps(road.record()))["name"] == "wz3to2" and len(road.digest) == 64


def test_real_n1_descriptor_matches_the_stub():
    md = pytest.importorskip("tools.coopmwm.map_descriptor")
    if not hasattr(md, "load") or not SPEC.is_file():
        pytest.skip("c4/nm-abstract (N1 map descriptor + wz3to2 assets) not merged yet")
    real = R.road_from_descriptor(md.load(SPEC))
    stub = wz_road()
    skip = {"descriptor", "descriptor_sha256"}
    a = {k: v for k, v in real.record().items() if k not in skip}
    b = {k: v for k, v in stub.record().items() if k not in skip}
    assert a == b
    assert real.descriptor == "n1" and real.descriptor_sha256 == md.load(SPEC).digest


def test_selection_scope_and_binding():
    wz = wz_road()
    with R.road_scope(wz):
        assert R.current_road() is wz
        assert R.map_default(553.33, 553.33, wz.road_end_m) == 1379.44
        assert R.map_default(600.0, 553.33, wz.road_end_m) == 600.0
    assert R.current_road() is R.MERGE_ORIGINAL
    assert R.map_default(553.33, 553.33, 1379.44) == 553.33
    with pytest.raises(ValueError):                    # defaulted to the original: one map
        R.bind_road(wz)
    R._reset_for_tests()
    assert R.bind_road(wz) is wz and R.bind_road(wz) is wz and R.current_road() is wz
    with pytest.raises(ValueError):
        R.bind_road(R.MERGE_ORIGINAL)


def test_env_selects_the_map(monkeypatch, tmp_path):
    spec = tmp_path / "MAP_SPEC.json"
    spec.write_text(json.dumps({"name": "wz3to2"}))
    monkeypatch.setenv("WMS_MAP_SPEC", str(spec))
    import importlib
    md = importlib.util.find_spec("tools.coopmwm.map_descriptor")
    if md is not None:
        mod = importlib.import_module("tools.coopmwm.map_descriptor")
        if hasattr(mod, "load"):
            pytest.skip("real N1 loader validates the MAP_SPEC digest; covered above")
    assert R.current_road().name == "wz3to2"


# ------------------------------------------------------------------ physics
def _scene():
    """Three cars: lane-2 car far upstream (s=-1000), lane-2 car in the lane-end zone, lane-1 car."""
    kin = torch.zeros(1, 3, 7)
    for j, (s, lane, v) in enumerate(((-1000.0, 2, 20.0), (40.0, 2, 10.0), (-1300.0, 1, 25.0))):
        kin[0, j, 0], kin[0, j, 1 + lane], kin[0, j, 4], kin[0, j, 6] = s, 1.0, v, 5.0
    return kin, torch.ones(1, 3, dtype=torch.bool)


def test_closing_lane_change_is_legal_upstream_only_on_the_lane_drop():
    from wms.physics.config import PhysicsConfig
    from wms.physics.mobil import mobil_merge_logits, physical_merge_envelope, MASKED_LOGIT

    p = PhysicsConfig(sumo_fidelity=True, lane_gap_rule="speed", lane_rule="learned_discretionary")
    kin, present = _scene()
    merge_env = physical_merge_envelope(kin, present, p)["legal"]
    merge_logits = mobil_merge_logits(kin, present, p)
    with R.road_scope(wz_road()):
        wz_env = physical_merge_envelope(kin, present, p)["legal"]
        wz_logits = mobil_merge_logits(kin, present, p)
    assert merge_env.tolist() == [[False, True, True]]
    assert wz_env.tolist() == [[True, True, True]]
    assert float(merge_logits[0, 0, 1]) == MASKED_LOGIT and float(wz_logits[0, 0, 1]) > MASKED_LOGIT
    # the lane-end car and the mainline car are unchanged by the map switch
    assert torch.equal(merge_logits[0, 1:], wz_logits[0, 1:])


def test_exit_bound_and_entry_process_follow_the_road():
    from wms.model.entries import TorchEntryProcess
    from wms.physics.batched import road_end_exit_bound
    from wms.physics.config import PhysicsConfig

    p = PhysicsConfig()
    counter = np.zeros((2, 30), np.float32)
    assert road_end_exit_bound(p) == 553.33
    assert TorchEntryProcess(counter).entry_s == (-2645.67, -2644.22, -378.66)
    with R.road_scope(wz_road()):
        assert road_end_exit_bound(p) == 1379.44
        proc = TorchEntryProcess(counter)
        assert proc.entry_s == (-2620.56,) * 3 and proc.entry_edge_end == (75.44,) * 3
        assert tuple(proc.cfg.sumo_lane_speed_mps) == (31.29,) * 3
    assert road_end_exit_bound(PhysicsConfig(exit_at_road_end=False)) is None


# ------------------------------------------------------------------ data pipeline
def _wz_meta(road):
    return {"map": road.name, "map_descriptor_sha256": road.descriptor_sha256,
            "road_digest": road.digest}


def _write(root, geo, seeds, road=None, k=64):
    collect = root / "collect" / "episodes" / "native"
    for seed in seeds:
        synth.write_episode(collect, geo, seed=seed, k=k,
                            extra_meta=None if road is None else _wz_meta(road))
    return root / "collect"


def test_converter_writes_a_wz3to2_source_and_guards_maps_and_seeds(tmp_path):
    from wms.data import c4_episodes as C4
    from wms.data.map_guard import MapDataRefusal

    road = wz_road(k_slots=64)
    collect = _write(tmp_path / "a", synth.WZ3TO2, (989200, 989201), road)
    with R.road_scope(road):
        doc = C4.convert([collect], tmp_path / "src", val_frac=0.5, split_seed=0)
        assert doc["map"] == "wz3to2" and doc["road_digest"] == road.digest
        assert doc["k_slots"] == 64 and {e["map"] for e in doc["episodes"]} == {"wz3to2"}
        assert C4.read_source_manifest(tmp_path / "src")["map"] == "wz3to2"
        ep = C4.EpisodeArrays(tmp_path / "src" / doc["episodes"][0]["dir"])
        counter = np.asarray(ep.array("counter"))
        assert counter[-1, 1].sum() > 0          # closing-lane insertions = entrance 1
        held = _write(tmp_path / "h", synth.WZ3TO2, (989600,), road)
        with pytest.raises(MapDataRefusal):
            C4.convert([held], tmp_path / "src_h")
        merge_eps = _write(tmp_path / "m", synth.MERGE, (989210,))
        with pytest.raises(MapDataRefusal):
            C4.convert([merge_eps], tmp_path / "src_m")
    with pytest.raises(MapDataRefusal):                # wz source under the merge road
        C4.read_source_manifest(tmp_path / "src")


def test_wz3to2_windows_train_through_the_round16_model(tmp_path):
    """End to end on wz3to2: converted windows -> loop-FT model forward/backward, finite, with
    upstream closing-lane changes legal in the rollout (the first-passage law owns only the
    lane-end zone)."""
    from tests.wz import golden_merge
    from wms.cli.assemble import build_model, build_physics
    from wms.data import c4_episodes as C4
    from wms.data.collate import collate
    from wms.train.trainer import WmSTrainer

    road = wz_road(k_slots=64)
    collect = _write(tmp_path, synth.WZ3TO2, (989300, 989301), road)
    torch.manual_seed(5)
    with R.road_scope(road):
        C4.convert([collect], tmp_path / "src", val_frac=0.0)
        cfg = golden_merge.load_cfg()
        built = C4.build_c4_datasets(
            tmp_path / "src", horizon_steps=20, history_s=cfg.data.history_s,
            history_stride_s=cfg.data.history_stride_s, entries_v2=True, stride_s=10.0,
            junction_fix=cfg.data.junction_fix, command_known=True, merge_pivot_history=True)
        train = built["train"]
        batch = collate([train[0], train[len(train) - 1]])
        states = np.asarray(batch["states"])
        lane = states[..., 1:4].argmax(-1)
        assert ((lane == 2) & np.asarray(batch["present"]).astype(bool)
                & (states[..., 0] < -1000)).any()        # closing-lane cars far upstream
        model = build_model(cfg, build_physics(cfg))

        class _T(WmSTrainer):
            def _init_from_checkpoint(self, name):
                return None

        trainer = _T(cfg, model, train, out_dir=tmp_path / "run")
        rec = trainer.forward_backward(batch, unroll_steps=20, backward=True)
    assert np.isfinite(rec["loss"]) and rec["valid_targets"] > 0
