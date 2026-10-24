"""R13E offline contracts on synthetic CPU data only; no production result fixtures."""
from dataclasses import asdict, dataclass
import gzip
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("r13e_offline", ROOT / "tools/r13e_checks/offline.py")
O = importlib.util.module_from_spec(SPEC)  # noqa: E741 (offline module alias)
sys.modules[SPEC.name] = O  # fork Pool must resolve top-level worker functions by module name
SPEC.loader.exec_module(O)


@pytest.fixture(autouse=True)
def cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def workspace(monkeypatch):
    # Independent of pytest's global temp settings, including in a parent's test run.
    with tempfile.TemporaryDirectory(prefix="r13e_test_", dir=ROOT / "_out") as directory:
        path = Path(directory)
        monkeypatch.setattr(O, "ROOT", path)
        yield path


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize("token", O.FORBIDDEN)
def test_every_supplied_path_is_rejected_before_any_read(token, workspace, monkeypatch):
    def no_read(*a, **kw):
        pytest.fail("restricted supplied path must fail before a file is opened")

    monkeypatch.setattr(O, "read_json", no_read)
    monkeypatch.setattr(O, "checked_path", no_read)
    for option in ("--checkpoint", "--train-root", "--states", "--out", "--config"):
        args = ["--checkpoint", "missing.pt", "--train-root", "missing_data", "--states", "missing.json",
                "--out", "out", "--config", "config.toml"]
        args[args.index(option) + 1] = f"data/{token.upper()}/asset"
        with pytest.raises(ValueError, match="restricted"):
            O.main(args)


@pytest.mark.parametrize("seed", [964000, 964099, 964199])
def test_seed_refusal_covers_paths_world_ids_and_nested_metadata(seed, workspace):
    with pytest.raises(ValueError, match="seed"):
        O.checked_path(f"w{seed}m2700/absent")
    for value in ({"meta": {"seed": seed}}, {"world": f"w{seed}m2700p20t300"},
                  {"meta": {"simulation_seed": str(seed)}}):
        with pytest.raises(ValueError, match="seed"):
            O.audit_metadata(value)
    with pytest.raises(ValueError, match="seed"):
        O.seed_all(seed)
    O.audit_metadata({"seed": 963999, "other_seed": 964200})


def test_workspace_missing_assets_and_config_extends_are_fail_closed(workspace):
    with pytest.raises(FileNotFoundError, match="missing required asset"):
        O.checked_path("absent.pt")
    with pytest.raises(ValueError, match="inside workspace"):
        O.checked_path(workspace.parent / "outside.toml", exists=False)
    child = workspace / "child.toml"
    child.write_text('extends = "../outside.toml"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="inside workspace"):
        O.check_config(child)
    child.write_text('extends = "SEALED.toml"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="restricted"):
        O.check_config(child)


def test_resolved_target_is_guarded_before_read(workspace, monkeypatch):
    # Model resolution without creating, resolving or opening any restricted name.
    target = workspace / "holdout.json"
    link = workspace / "innocent.json"
    original = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda p, *a, **kw: target if p == link else original(p, *a, **kw))
    monkeypatch.setattr(Path, "open", lambda *a, **kw: pytest.fail("no file may be opened"))
    with pytest.raises(ValueError, match="restricted"):
        O.read_json(link)


def state(index=0, **extra):
    vehicles = [{"id": "h", "s": 0, "v": 5, "lane": 1, "is_cav": False}]
    return {"world": f"w{4100 + index}m2700p20t300", "seed": 4100 + index,
            "initial_state": {"time_s": 300, "vehicles": vehicles}, "t0_s": 300,
            "recorded": {"native": [{"time_s": 300 + t * .5, "vehicles": vehicles}
                                     for t in range(241)]}, **extra}


def test_state_train_provenance_never_inferred_from_filename(workspace):
    path = write_json(workspace / "train_states.json", [state(i) for i in range(48)])
    with pytest.raises(ValueError, match="lacks recorded TRAIN"):
        O.load_train_states(path, {})
    worlds = {state(i)["world"]: "train" for i in range(48)}
    assert len(O.load_train_states(path, worlds)) == 48
    worlds[state(0)["world"]] = "val"
    with pytest.raises(ValueError, match="validation"):
        O.load_train_states(path, worlds)
    records = [state(i, meta={"split": "TRAIN"}) for i in range(48)]
    assert len(O.load_train_states(write_json(path, records), {})) == 48
    records[0]["meta"]["split"] = "TEST"
    with pytest.raises(ValueError, match="not TRAIN"):
        O.load_train_states(write_json(path, {"split": "TRAIN", "states": records}), {})


def test_gzip_envelope_duplicate_full_horizon_and_unselected_records(workspace):
    path = workspace / "states.json.gz"
    records = [state(i) for i in range(48)]

    def save():
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump({"metadata": {"split": "TRAIN"}, "records": records}, stream)

    save()
    assert len(O.load_train_states(path, {})) == 48
    records[-1] = records[0]
    save()
    with pytest.raises(ValueError, match="duplicate"):
        O.load_train_states(path, {})
    records[-1] = state(47)
    records[0]["recorded"]["native"].pop(10)
    save()
    with pytest.raises(ValueError, match="full recorded 120 s"):
        O.load_train_states(path, {})
    records[0] = state(0)
    records.append(state(49, split="TEST"))
    save()
    with pytest.raises(ValueError, match="not TRAIN"):
        O.load_train_states(path, {})


def test_manifest_provenance_and_array_paths_checked_before_dataset_reads(workspace, monkeypatch):
    from wms.data import c4_episodes as ce

    # Empty array stubs: validation must not attempt np.load during this metadata stage.
    directory = workspace / "episode"
    directory.mkdir()
    for name in ce.ARRAYS:
        (directory / f"{name}.npy").touch()
    entry = {"world": "w4100m2700p20t300", "split_role": "train", "dir": "episode"}
    doc = {"schema": ce.SOURCE_SCHEMA, "split": "TRAIN", "episodes": [entry]}
    path = write_json(workspace / O.MANIFEST, doc)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("no arrays before provenance"))
    docs, worlds = O.train_manifests([workspace])
    assert worlds[entry["world"]] == "train" and docs[0][1] == doc
    entry["split_role"] = "val"
    write_json(path, doc)
    with pytest.raises(ValueError, match="no recorded train-role"):
        O.train_manifests([workspace])
    entry["split_role"] = "train"
    doc.pop("split")
    write_json(path, doc)
    with pytest.raises(ValueError, match="explicitly be TRAIN"):
        O.train_manifests([workspace])
    doc["split"] = "TRAIN"
    entry["dir"] = "hetdata_truth/episode"
    write_json(path, doc)
    with pytest.raises(ValueError, match="restricted"):
        O.train_manifests([workspace])


def test_real_source_loader_keeps_recorded_training_split_and_never_opens_validation_arrays(workspace, monkeypatch):
    from wms.config.train import TrainConfig
    from wms.data import c4_episodes as ce

    # Avoid leaving synthetic memmaps in the process-global LRU after this test.
    monkeypatch.setattr(ce, "_open_episode", ce._open_episode.__wrapped__)
    kin = np.zeros((241, 1, 7), np.float32)
    kin[..., 2], kin[..., 4], kin[..., 6] = 1, 5, 5
    arrays = {"veh_kin": kin, "veh_mask": np.ones((241, 1), np.uint8),
              "target_mask": np.ones((241, 1), np.uint8), "counter": np.zeros((241, 2, 30), np.float32),
              "vid": np.zeros((241, 1), np.int32), "is_cav": np.zeros((241, 1), np.uint8),
              "time_s": 300 + np.arange(241) * .5, "control_time_s": 300 + np.arange(121, dtype=float),
              "controller": np.zeros((121, 1), np.int8),
              "issued_acceleration_mps2": np.zeros((121, 1), np.float32),
              "issued_target_speed_mps": np.zeros((121, 1), np.float32)}
    entries = []
    for role, world in (("train", "w4100m2700p20t0"), ("val", "w4101m2700p20t0")):
        directory = workspace / role
        directory.mkdir()
        for name in ce.ARRAYS:
            if role == "train":
                np.save(directory / f"{name}.npy", arrays[name])
            else:
                # Opening validation arrays would fail; a metadata TRAIN claim
                # must not cause them to be silently relabeled and read.
                (directory / f"{name}.npy").touch()
        entries.append({"world": world, "episode": world, "dir": role, "arm": "native", "controller": "native",
                        "warmup_s": 0, "split_role": role})
    doc = {"schema": ce.SOURCE_SCHEMA, "split": "TRAIN", "episodes": entries}
    write_json(workspace / O.MANIFEST, doc)
    docs, _worlds = O.train_manifests([workspace])
    sets = O.train_datasets(TrainConfig.from_dict({}), docs)
    assert len(sets) == 1 and len(sets[0]) == 1
    assert sets[0]._loader.entries == [entries[0]]
    assert sets[0][0]["states"].shape == (241, 1, 7)
    assert doc["episodes"][1]["split_role"] == "val"
    # Real loader positions store a tick index, NOT seconds or time since warmup.
    assert sets[0].positions == [(0, 0)]
    record = state(world=entries[0]["world"])
    assert O.onestep_jobs(sets, [record]) == [(0, 0, f"0:{entries[0]['episode']}:0")]


@pytest.fixture
def checkpoint(workspace):
    from wms.config.train import TrainConfig
    from wms.model.checkpoint import save_checkpoint
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel
    from wms.physics.config import PhysicsConfig

    torch.manual_seed(17)
    flat = TrainConfig.from_dict({"model": {"follower_courtesy": True, "follower_courtesy_version": 2,
                                          **{key: True for key in O.EMISSIONS}}}).to_legacy_flat()
    spec = ModelSpec.from_mapping(flat)
    model = WmSModel(spec).eval()
    path = workspace / "wm_s.pt"
    save_checkpoint(path, model, flat)
    write_json(workspace / "physics.json", asdict(PhysicsConfig()))
    return path, model


def test_real_strict_loader_keeps_every_tensor_byte_and_emission(checkpoint):
    path, original = checkpoint
    before = O.file_digest(path)
    predictor, capture, digest = O.load_exact_predictor(path)
    assert digest == O.state_digest(original.state_dict())
    assert O.state_digest(predictor.model.state_dict()) == digest
    assert capture["mode_info"] == {"applied": {}}
    assert capture["hdv_exits"] == "none" and capture["entries"] == "off"
    assert all(getattr(predictor.model, k) is True for k in O.EMISSIONS)
    assert not predictor.flags.diag_recorded_lanes
    assert not predictor.model.training and not any(p.requires_grad for p in predictor.model.parameters())
    assert all(p.device.type == "cpu" for p in predictor.model.parameters())
    assert O.file_digest(path) == before


def test_free_hdv_wrapper_drops_recorded_junction_and_exit_presence_only_for_hdvs(checkpoint, monkeypatch):
    from wms.model.wms import WmSModel

    seen = {}

    def rollout(model, initial, **kwargs):
        seen.update(kwargs)
        return {"test_capture": True}

    monkeypatch.setattr(WmSModel, "rollout", rollout)
    p, capture, _digest = O.load_exact_predictor(checkpoint[0])
    initial = {"is_cav": torch.tensor([[False, True]]), "present": torch.tensor([[True, True]])}
    junction = torch.ones(1, 3, 2, dtype=torch.bool)
    present = torch.zeros(1, 3, 2, dtype=torch.bool)
    p.model.rollout(initial, junction_seq=junction, present_seq=present)
    assert not seen["junction_seq"][0, :, 0].any()  # recorded HDV transit cannot revive it
    assert seen["junction_seq"][0, :, 1].all()  # recorded CAV transit is exogenous
    assert seen["present_seq"][0, :, 0].all()  # HDV departure is model-determined
    assert not seen["present_seq"][0, :, 1].any()
    assert junction.all() and not present.any()  # input records are not mutated
    assert capture["last"] == {"test_capture": True}


@pytest.mark.parametrize("damage", ["missing_optional", "extra", "wrong_shape", "wrong_dtype", "emission"])
def test_no_partial_loading_ignored_keys_casts_or_emission_overrides(checkpoint, damage):
    path, _ = checkpoint
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if damage == "missing_optional":
        del payload["model"]["follower_courtesy.net.0.weight"]
    elif damage == "extra":
        payload["model"]["ignored.weight"] = torch.ones(1)
    elif damage == "wrong_shape":
        payload["model"]["follower_courtesy.net.0.weight"] = torch.zeros(1, 1)
    elif damage == "wrong_dtype":
        payload["model"]["follower_courtesy.net.0.weight"] = payload["model"]["follower_courtesy.net.0.weight"].double()
    else:
        payload["config"]["temporal_accel_off"] = False
    torch.save(payload, path)
    with pytest.raises((RuntimeError, ValueError)):
        O.load_exact_predictor(path)


def test_state_digest_distinguishes_signed_zero():
    assert O.state_digest({"x": torch.tensor([0.])}) != O.state_digest({"x": torch.tensor([-0.])})


def test_arm_override_replaces_all_physics_holders_preserving_other_fields(monkeypatch):
    from tools import wm_eval_common as wc

    @dataclass
    class Physics:
        emergency_braking: bool = False
        secure_gap_veto: bool = False
        merge_end_wait: bool = False
        decel: float = 2.01

    model = SimpleNamespace(physics=Physics())
    child = SimpleNamespace(physics=Physics(merge_end_wait=True, decel=3.3))
    monkeypatch.setattr(wc, "_physics_holders", lambda m: [(model, "physics"), (child, "physics")])
    old = [asdict(o.physics) for o in (model, child)]
    for arm in (*O.ARMS, "off"):
        O.set_arm(model, arm)
        for o, saved in zip((model, child), old):
            assert asdict(o.physics) == {**saved, **O.ARMS[arm]}
    with pytest.raises(ValueError, match="unknown"):
        O.set_arm(model, "other")
    model.physics = SimpleNamespace(__dataclass_fields__={})
    with pytest.raises(ValueError, match="unavailable"):
        O.set_arm(model, "emergency+gs")


def test_base_bias_mae_classes_and_paired_window_cis():
    from tools.r10b_checks.run_onestep import COLS, BASE_COLS

    cols = COLS + BASE_COLS
    a = np.zeros((12, len(cols)), np.float32)
    d = {c: i for i, c in enumerate(cols)}
    a[:, d["lane"]] = 1
    a[:, d["win"]] = np.repeat([0, 1], 6)
    a[:, d["rec"]] = -2
    a[:, d["fin"]] = np.repeat([-1, 1], 6)  # biases +1, +3, not recorded-minus-predicted
    a[:, d["v"]] = 10
    a[:, d["gapF"]] = 70
    b = a.copy()
    b[:, d["fin"]] -= .5
    inputs = {"off": a, "emergency": b, "emergency+gs": b.copy()}
    result = O.analyze_onestep(inputs, cols)
    braking = result["off"]["recorded_braking"]
    assert braking["bias"]["mean"] == 2 and braking["mae"]["mean"] == 2
    assert braking["bias"]["lo"] == 1 and braking["bias"]["hi"] == 3
    assert braking["bias"]["ci_available"] and braking["bias"]["clusters"] == 2
    assert result["off"]["free_road"]["mae"]["mean"] == 2
    assert result["off"]["all_lane1"]["mae"]["mean"] == 2
    assert not result["off"]["slow_start"]["bias"]["ci_available"]
    assert result["emergency_minus_off"]["recorded_braking"]["bias"]["mean"] == -.5
    assert result["emergency_minus_off"]["all_lane1"]["mae"]["mean"] == -.5
    assert result["emergency+gs_minus_emergency"]["all_lane1"]["bias"]["mean"] == 0
    assert O.json_safe(O.analyze_onestep(inputs, cols)) == O.json_safe(result)
    a[:, d["v"]], a[:, d["gapF"]] = 1, 6
    b[:, d["v"]], b[:, d["gapF"]] = 1, 6
    inputs["emergency+gs"] = b
    slow = O.analyze_onestep(inputs, cols)
    assert slow["off"]["slow_start"]["mae"]["mean"] == 2
    assert slow["off"]["free_road"]["bias"]["n"] == 0
    a[:, d["v"]], a[:, d["gapF"]] = 10, np.nan
    b[:, d["v"]], b[:, d["gapF"]] = 10, np.nan
    assert O.analyze_onestep(inputs, cols)["emergency"]["free_road"]["mae"]["n"] == 12


def test_insufficient_clusters_or_rows_are_null_in_strict_json():
    single = O._estimate(np.arange(10.), np.zeros(10))
    assert single["mean"] == 4.5 and not single["ci_available"]
    assert O.json_safe(single)["lo"] is None
    short = O._estimate([1, 2, 3, 4], [0, 0, 1, 1])
    assert not short["ci_available"] and O.json_safe(short)["mean"] is None
    json.dumps(O.json_safe({"single": single, "short": short}), allow_nan=False)


def test_rollout_thresholds_leader_cav_overlaps_and_absent_rows():
    # HDV at 0 follows a CAV at 8 m (3 m bumper gap); unrelated CAV at lane 0.
    pos = np.array([[0, 8, 100], [0, 8, 100], [0, 4, 100]], float)
    vel = np.array([[12, 8, 0], [10.5, 8, 0], [8.5, 8, 0]], float)
    lane = np.array([[1, 1, 0]] * 3)
    mask = np.ones_like(pos, bool)
    metrics = O.rollout_metrics(pos, vel, lane, mask, [False, True, True], np.array([5, 5, 5]))
    assert metrics["a_lt_minus3_rows"] == 1  # exactly -3 at first transition is excluded
    assert metrics["finite_hdv_transitions"] == 2
    assert metrics["a_lt_minus3_share"] == .5
    assert metrics["ttc_lt_1_5_rows"] == 2 and metrics["ttc_eligible_rows"] == 2
    assert metrics["overlap_pair_steps"] == 1 and not metrics["blowup"]
    vel[1, 0] = 10  # positive 3 m gap / 2 m/s = exactly 1.5, excluded
    assert O.rollout_metrics(pos, vel, lane, mask, [False, True, True], [5, 5, 5])["ttc_lt_1_5_rows"] == 1
    mask[2, 0] = False
    metrics = O.rollout_metrics(pos, vel, lane, mask, [False, True, True], [5, 5, 5])
    assert metrics["overlap_pair_steps"] == 0 and metrics["hdv_rows"] == 1


@pytest.mark.parametrize("channel,value", [("pos", np.nan), ("pos", 100001), ("vel", 101),
                                           ("vel", -1), ("lane", 1.2), ("lane", np.inf)])
def test_rollout_blowups_are_counted_not_silently_dropped(channel, value):
    arrays = {"pos": np.zeros((2, 1)), "vel": np.ones((2, 1)), "lane": np.ones((2, 1))}
    arrays[channel][1, 0] = value
    result = O.rollout_metrics(arrays["pos"], arrays["vel"], arrays["lane"], np.ones((2, 1), bool), [False], [5])
    assert result["blowup"] and result["invalid_hdv_rows"] == 1


def test_post_insertion_counts_hdv_inserters_and_fixed_followers_strictly_after_change():
    pos = np.tile([10., 0., 50.], (16, 1))
    speed = np.full_like(pos, 10.)
    lane = np.ones_like(pos)
    lane[0, 0] = 2  # HDV ramp insertion at t=0.5 s
    lane[-1, 0] = 0  # second insertion too late for a full post window
    speed[1:, 1] = 8  # braking DURING insertion is excluded from post metrics
    speed[2:, :2] = [8, 6]  # both brake strictly after insertion
    speed[3:, 1] = 4.5  # exactly -3 does not qualify
    mask = np.ones_like(pos, bool)
    m = O.rollout_metrics(pos, speed, lane, mask, [False, False, True], [5, 5, 5])
    assert m["merges_per_run"] == 1 and m["hdv_insertion_events"] == 2
    assert m["post_insertion_truncated_events"] == 1
    for who in ("inserter", "follower"):
        assert m[f"hdv_{who}_post_events"] == 1
        assert m[f"hdv_{who}_post_hard_brake_events"] == 1
        assert m[f"hdv_{who}_post_finite_rows"] == 12
        assert m[f"hdv_{who}_post_hard_brake_rows"] == 1
        assert m[f"hdv_{who}_post_hard_brake_rows_share"] == 1 / 12
    mask[4, 1] = False
    m = O.rollout_metrics(pos, speed, lane, mask, [False, False, True], [5, 5, 5])
    assert m["hdv_follower_post_finite_rows"] == 10  # no acceleration through absence
    # A CAV inserter never enters these HDV-inserter event denominators.
    m = O.rollout_metrics(pos, speed, lane, mask, [True, False, True], [5, 5, 5])
    assert m["hdv_insertion_events"] == 0 and m["hdv_follower_post_events"] == 0
    assert O.json_safe(m)["hdv_follower_post_hard_brake_events_share"] is None


def test_stuck_clock_threshold_zone_speed_and_reset():
    pos = np.full((122, 1), O.MERGE_END_M)
    speed, lane, mask = np.zeros_like(pos), np.full_like(pos, 2), np.ones_like(pos, bool)
    m = O.rollout_metrics(pos, speed, lane, mask, [False], [5])
    assert m["stuck_vehicle_steps"] == 1 and m["stuck_final_vehicles"] == 1
    assert m["stuck_max_wait_s"] == 60.5 and m["stuck_mean_ramp_in_zone"] == 1
    speed[61] = 1  # strict speed<1; breaks the consecutive wait
    assert O.rollout_metrics(pos, speed, lane, mask, [False], [5])["stuck_final_vehicles"] == 0
    speed[:] = 0
    pos[61] += .01  # beyond the inclusive zone boundary also resets
    assert O.rollout_metrics(pos, speed, lane, mask, [False], [5])["stuck_vehicle_steps"] == 0


def test_delay_regions_exposure_and_invalid_rows_use_shared_speed_deficit_proxy():
    pos = np.tile([-1., 0., O.MERGE_END_M, 500., 1.], (3, 1))
    speed = np.tile([0., O.V_REF / 2, 0., 0., 0.], (3, 1))
    lane, mask = np.ones_like(pos), np.ones_like(pos, bool)
    m = O.rollout_metrics(pos, speed, lane, mask, [False] * 4 + [True], [5] * 5)
    assert m["delay_upstream_veh_s"] == 1 and m["delay_merge_veh_s"] == .5
    assert m["delay_downstream_veh_s"] == 1 and m["delay_total_veh_s"] == 2.5
    assert all(m[f"exposure_{r}_veh_s"] == 1 for r in ("upstream", "merge", "downstream"))
    speed[1, 0] = np.nan
    m = O.rollout_metrics(pos, speed, lane, mask, [False] * 4 + [True], [5] * 5)
    assert m["invalid_hdv_rows"] == 1 and m["blowup"]
    assert m["delay_upstream_veh_s"] == .5 and m["exposure_upstream_veh_s"] == .5


def test_seed_replicates_resample_together_as_state_clusters():
    # Within-state seed variation cancels; row-independent resampling would have width.
    stats = O._estimate(np.tile([-10., 10.], 5), np.repeat(np.arange(5), 2))
    assert stats["n"] == 10 and stats["clusters"] == 5
    assert stats["mean"] == stats["lo"] == stats["hi"] == 0


def test_rollout_pairing_resets_all_rngs_uses_capture_last_and_frees_hdv(monkeypatch):
    from wms.eval.predictor import RolloutFlags
    import random

    calls, capture = [], {}
    model = SimpleNamespace(arm=None)
    monkeypatch.setattr(O, "set_arm", lambda m, arm: setattr(m, "arm", arm))

    class Predictor:
        flags = RolloutFlags()

        def __init__(self):
            self.model = model

        def predict_states(self, record, controllers, horizon, seed):
            draws = (random.random(), np.random.random(), torch.rand(()).item())
            calls.append((record["seed"], model.arm, self.flags.lane_seed, horizon, seed, draws))
            assert controllers["arm"] == "native" and not self.flags.diag_recorded_lanes
            # Raw tensors, not the rendered frames, drive the metrics.
            capture["last"] = {"positions": torch.arange(240.)[None, :, None],
                               "speeds": torch.ones(1, 240, 1), "lanes": torch.ones(1, 240, 1),
                               "masks": torch.ones(1, 240, 1, dtype=torch.bool)}
            return [{"deliberately": "irrelevant public frame result"}]

    result = O.run_rollouts(Predictor(), capture, [state(i) for i in range(48)])
    assert len(result["rows"]) == 96 and len(calls) == 288
    for a, b, c in zip(calls[::3], calls[1::3], calls[2::3]):
        assert [a[1], b[1], c[1]] == list(O.ARMS)
        assert a[:1] + a[2:] == b[:1] + b[2:] == c[:1] + c[2:]
        assert a[2] == a[4] and a[3] == 120
    assert len({r["seed"] for r in result["rows"]}) == 96
    stats = result["state_cluster_run_mean"]["emergency_minus_off"]["hdv_rows"]
    assert stats["mean"] == 0 and stats["clusters"] == 48 and stats["n"] == 96
    assert all(r["off"]["hdv_rows"] == 240 for r in result["rows"])
    assert result["totals"]["off"]["states"] == 48 and result["totals"]["off"]["runs"] == 96


def test_real_base_producer_is_used_with_strict_checkpoint_and_global_state_restored(checkpoint):
    from tools.r10b_checks import run_onestep as producer
    from wms.config.train import TrainConfig
    from wms.physics.config import PhysicsConfig

    path, _ = checkpoint
    p, capture, digest = O.load_exact_predictor(path)
    assert {"emergency_braking", "secure_gap_veto"} <= PhysicsConfig.__dataclass_fields__.keys()
    # Three lane-1 HDVs: a closing follower, a stopped slow-start, a free-road leader.
    kin = np.zeros((241, 3, 7), np.float32)
    kin[..., 0] = [0, 15, 100]
    kin[..., 2] = 1
    kin[..., 4] = [12, 0, 10]
    kin[..., 5] = -2
    kin[..., 6] = 5
    item = {"states": kin, "present": np.ones((241, 3), bool), "is_cav": np.zeros((241, 3), bool),
            "slot_index": np.arange(3)}

    class Dataset:
        positions = [(0, 0), (0, 2)]
        _loader = SimpleNamespace(entries=[{"episode": "synthetic_train_episode"}])

        def __getitem__(self, index):
            return item

    cfg = TrainConfig.from_legacy_flat(p.backend.config)
    original_forward = p.model.forward
    forward_calls = []

    def wrapped_forward(*args, **kwargs):
        forward_calls.append(True)
        return original_forward(*args, **kwargs)

    p.model.forward = wrapped_forward
    before = producer.G.copy()
    producer.G["test_sentinel"] = "restore me"
    try:
        packed, cols, windows = O.run_onestep(p, cfg, [Dataset()])
        assert len(windows) == 1
        assert packed["off"].shape[0] > 100
        assert all(packed[arm].shape == packed["off"].shape for arm in O.ARMS)
        assert {"gapF", "vLF", "dacc", "ddec"} <= set(cols)
        assert producer.G["test_sentinel"] == "restore me"
        assert O.state_digest(p.model.state_dict()) == digest
        assert p.model.forward is wrapped_forward and len(forward_calls) == 3 * 240
        report = O.analyze_onestep(packed, cols)
        assert report["off"]["all_lane1"]["bias"]["n"] > 5
        assert O.state_digest(p.model.state_dict()) == digest
    finally:
        producer.G.clear()
        producer.G.update(before)


def test_cli_default_and_missing_asset_never_writes_a_report(workspace):
    args = ["--checkpoint", "absent.pt", "--train-root", "train", "--states", "states.json", "--out", "out"]
    assert O.build_parser().parse_args(args).config == "c4_wm_v10round12c.toml"
    assert O.build_parser().parse_args(args).workers == 1
    assert not O.build_parser().parse_args(args).one_step_panel
    with pytest.raises(FileNotFoundError, match="missing required asset"):
        O.main(args)
    assert not (workspace / "out").exists()


def test_linux_limits_set_nice19_and_single_torch_thread(monkeypatch):
    import os
    priority = [3]
    monkeypatch.setattr(O.sys, "platform", "linux")
    monkeypatch.setattr(os, "PRIO_PROCESS", 0, raising=False)
    monkeypatch.setattr(os, "getpriority", lambda *a: priority[0], raising=False)
    monkeypatch.setattr(os, "nice", lambda increment: priority.__setitem__(0, priority[0] + increment), raising=False)
    O.cpu_limits()
    assert priority == [19] and torch.get_num_threads() == 1
    assert torch.get_num_interop_threads() == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("workers", [0, 5, -1, 1.5, True])
def test_worker_bounds_apply_to_python_api(workers):
    with pytest.raises(ValueError, match="1..4"):
        list(O._execute(lambda job: pytest.fail("must validate first"), [0], {}, workers))


@pytest.mark.parametrize("workers", ["0", "5", "-1", "1.5"])
def test_worker_bounds_apply_to_cli(workers):
    with pytest.raises(SystemExit) as exc:
        O.build_parser().parse_args(["--checkpoint", "x", "--train-root", "x", "--states", "x",
                                    "--out", "x", "--workers", workers])
    assert exc.value.code == 2


def test_parallel_requires_linux_without_falling_back_to_spawn(monkeypatch):
    monkeypatch.setattr(O.sys, "platform", "win32")
    monkeypatch.setattr(O.multiprocessing, "get_context", lambda *a: pytest.fail("no spawn fallback"))
    with pytest.raises(ValueError, match="Linux fork"):
        list(O._execute(lambda job: job, [0], {}, 2))


def panel_dataset(entries, positions, times):
    def episode(index):
        return SimpleNamespace(array=lambda name: times[index] if name == "time_s" else None)

    return SimpleNamespace(positions=positions, _loader=SimpleNamespace(entries=entries, episode=episode))


def test_explicit_panel_matches_world_and_array_time_in_record_order():
    records = [state(1), state(0)]
    entries = [{"episode": "arbitrary_name_a", "world": state(0)["world"]},
               {"episode": "arbitrary_name_b", "world": state(1)["world"]}]
    # Tick 2 means t=300, not t=1; tick 0 at t=299 must not be selected.
    ds = panel_dataset(entries, [(0, 0), (0, 2), (1, 2)], [np.array([299., 299.5, 300.])] * 2)
    assert O.onestep_jobs([ds], records) == [(0, 2, "0:arbitrary_name_b:2"), (0, 1, "0:arbitrary_name_a:1")]
    # Default is still a fixed-RNG choice of one window per episode, in episode order.
    rng = np.random.default_rng(O.RNG_SEED)
    pick = int(rng.choice([0, 1]))
    assert O.onestep_jobs([ds]) == [(0, pick, f"0:arbitrary_name_a:{pick}"), (0, 2, "0:arbitrary_name_b:2")]
    # Metadata-world is supported, but dataset world comes from the entry itself.
    r = records[0].copy()
    r["meta"] = {"world": r.pop("world")}
    assert O.onestep_jobs([ds], [r])[0][1] == 2


def test_panel_refuses_missing_ambiguous_wrong_branch_and_duplicate_matches():
    world = state()["world"]
    entry = {"episode": "a", "world": world, "arm": "native"}
    ds = panel_dataset([entry], [(0, 0)], [np.array([300.])])
    for record in (state(world="unknown"), state(episode="wrong"), state(arm="wrong"),
                   state(initial_state={"time_s": 300.5, "vehicles": []})):
        with pytest.raises(ValueError, match="exactly one TRAIN window"):
            O.onestep_jobs([ds], [record])
    with pytest.raises(ValueError, match="duplicate"):
        O.onestep_jobs([ds], [state(), state()])
    other = panel_dataset([{**entry, "episode": "b", "arm": "branch"}], [(0, 0)], [np.array([300.])])
    with pytest.raises(ValueError, match="found 2"):
        O.onestep_jobs([ds, other], [state()])
    assert O.onestep_jobs([ds, other], [state(episode="b", arm="branch")]) == [(1, 0, "1:b:0")]


def test_merge_zone_occupancy_and_speed_use_valid_hdv_rows_strict_boundaries():
    pos = np.tile([0., 75.439, 75.44, -0.01, 10., 10., 10., 10.], (3, 1))
    speed = np.tile([2., 8., 20., 20., 20., 20., 20., 20.], (3, 1))
    lanes = np.tile([0., 1., 0., 1., 2., 0., 1., 0.], (3, 1))
    mask = np.ones_like(pos, bool)
    mask[1:, 5] = False
    speed[1:, 6] = np.nan
    # Last slot is a CAV, lane 2 is a ramp HDV, slot 6 is invalid.
    cav = [False] * 7 + [True]
    m = O.rollout_metrics(pos, speed, lanes, mask, cav, [5] * 8)
    assert m["merge_zone_hdv_rows"] == 4
    assert m["merge_zone_mean_hdv_count"] == 2
    assert m["merge_zone_mean_speed_mps"] == 5
    mask[2, :2] = False  # empty frame still contributes to the time-mean denominator
    m = O.rollout_metrics(pos, speed, lanes, mask, cav, [5] * 8)
    assert m["merge_zone_hdv_rows"] == 2 and m["merge_zone_mean_hdv_count"] == 1
    assert m["merge_zone_mean_speed_mps"] == 5
    mask[1:, :2] = False
    m = O.rollout_metrics(pos, speed, lanes, mask, cav, [5] * 8)
    assert m["merge_zone_mean_hdv_count"] == 0
    assert O.json_safe(m)["merge_zone_mean_speed_mps"] is None


@pytest.fixture(params=["reversed", "fork"])
def parallel_pool(request, monkeypatch):
    """Force worst-case completion ordering on every OS; also use real fork on Linux."""
    if request.param == "fork":
        if not sys.platform.startswith("linux"):
            pytest.skip("real fork integration requires Linux")
        return
    monkeypatch.setattr(O.sys, "platform", "linux")

    class Pool:
        def __init__(self, *, processes, initializer):
            assert 1 < processes <= 4 and initializer is O.cpu_limits

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def imap_unordered(self, function, jobs, *, chunksize):
            assert chunksize == 1
            for job in reversed(jobs):
                # Job queues contain only descriptors, never datasets or closures.
                assert pickle.loads(pickle.dumps(job)) == job
                yield function(job)

    def get_context(method):
        assert method == "fork"
        return SimpleNamespace(Pool=Pool)

    monkeypatch.setattr(O.multiprocessing, "get_context", get_context)


@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_onestep_exact_rows_crn_order_and_bootstrap(monkeypatch, parallel_pool, workers):
    from tools.r10b_checks import run_onestep as producer
    import random

    model = SimpleNamespace(state_dict=lambda: {}, arm=None, forward=lambda: None)
    original_forward = model.forward
    p = SimpleNamespace(model=model, backend=SimpleNamespace(config={}))
    ds = panel_dataset([{"episode": "a"}, {"episode": "b"}], [(0, 0), (1, 0)], [])
    cols = producer.COLS + producer.BASE_COLS
    ci = {name: i for i, name in enumerate(cols)}
    monkeypatch.setattr(O, "set_arm", lambda m, arm: setattr(m, "arm", arm))

    def one_window(job):
        arm = model.arm
        arr = np.zeros((6, len(cols)), np.float32)
        arr[:, ci["lane"]], arr[:, ci["v"]], arr[:, ci["gapF"]] = 1, 10, 70
        arr[:, ci["rec"]] = -2
        # RNG-dependent fixed geometry proves the per-arm seed reset as well.
        arr[:, ci["s"]] = random.random() + np.random.random() + torch.rand(()).item()
        arr[:, ci["fin"]] = job[1] + list(O.ARMS).index(arm)
        del model.forward  # exactly the BASE producer's wrapper-removal behavior
        return job[2], {arm: (arr, {})}

    monkeypatch.setattr(producer, "one_window", one_window)
    before = producer.G.copy()
    serial = O.run_onestep(p, None, [ds])
    events = []
    parallel = O.run_onestep(p, None, [ds], workers=workers, progress=events.append)
    assert parallel[1:] == serial[1:]
    for arm in O.ARMS:
        np.testing.assert_array_equal(parallel[0][arm], serial[0][arm])
    assert O.json_safe(O.analyze_onestep(*parallel[:2])) == O.json_safe(O.analyze_onestep(*serial[:2]))
    assert len(events) == 6 and {e["state"] for e in events} == {0, 1}
    assert events[-1]["completed"] == events[-1]["total"] == 6
    assert producer.G == before and model.forward is original_forward and O._FORK == {}


@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_rollouts_exact_pairing_order_and_cluster_bootstrap(monkeypatch, parallel_pool, workers):
    from wms.eval.predictor import RolloutFlags
    import random

    capture = {}

    class Predictor:
        flags = RolloutFlags()

        def __init__(self):
            self.model = SimpleNamespace(arm=None)

        def predict_states(self, record, controllers, horizon, seed):
            assert seed == self.flags.lane_seed and horizon == 120
            assert "last" not in capture
            draw = random.random() + np.random.random() + torch.rand(()).item()
            speed = draw + list(O.ARMS).index(self.model.arm)
            capture["last"] = {"positions": torch.full((1, 240, 1), float(record["seed"] % 10)),
                               "speeds": torch.full((1, 240, 1), speed),
                               "lanes": torch.ones(1, 240, 1),
                               "masks": torch.ones(1, 240, 1, dtype=torch.bool)}

    monkeypatch.setattr(O, "set_arm", lambda m, arm: setattr(m, "arm", arm))
    records = [state(i) for i in range(5)]
    p = Predictor()
    serial = O.run_rollouts(p, capture, records)
    events = []
    parallel = O.run_rollouts(p, capture, records, workers=workers, progress=events.append)
    assert O.json_safe(parallel) == O.json_safe(serial)
    assert len(events) == 30 and events[-1]["completed"] == events[-1]["total"] == 30
    assert all(e["phase"] == "rollout" and "metrics" in e and not set(O.ARMS) & e.keys() for e in events)
    assert parallel["state_cluster_run_mean"]["emergency_minus_off"]["merge_zone_mean_speed_mps"]["mean"] == pytest.approx(1)
    assert O._FORK == {}


def test_onestep_restores_wrappers_and_globals_on_failure(monkeypatch):
    from tools.r10b_checks import run_onestep as producer

    def forward():
        return None

    model = SimpleNamespace(forward=forward)
    before = producer.G.copy()
    monkeypatch.setattr(O, "set_arm", lambda *a: None)
    monkeypatch.setattr(O, "seed_all", lambda *a: None)

    def fail(job):
        del model.forward
        raise RuntimeError("inference failed")

    monkeypatch.setattr(producer, "one_window", fail)
    context = dict(predictor=SimpleNamespace(model=model), sets=[], cfg=None, settings=None)
    with pytest.raises(RuntimeError, match="inference failed"):
        list(O._execute(O._onestep_job, [(0, "off", (0, 0, "w"))], context, 1))
    assert model.forward is forward and producer.G == before and O._FORK == {}


def test_compressed_npz_roundtrip_and_disk_budget_before_write(workspace, monkeypatch):
    packed = {arm: np.arange(24, dtype=np.float32).reshape(8, 3) for arm in O.ARMS}
    total = O.save_onestep(workspace, packed, ["a", "b", "c"])
    assert total == sum(p.stat().st_size for p in workspace.glob("*.npz")) < O.MAX_NPZ_BYTES
    for arm in O.ARMS:
        with np.load(workspace / f"onestep_{arm}.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["rows"], packed[arm])
            assert list(data["cols"]) == ["a", "b", "c"]
    out = workspace / "too_small"
    out.mkdir()
    monkeypatch.setattr(O, "MAX_NPZ_BYTES", 1)
    with pytest.raises(ValueError, match="disk budget"):
        O.save_onestep(out, packed, ["a", "b", "c"])
    assert list(out.iterdir()) == []
    # The budget is combined across arms, rather than a per-file ceiling.
    monkeypatch.setattr(O, "MAX_NPZ_BYTES", (workspace / "onestep_off.npz").stat().st_size * 2 - 1)
    with pytest.raises(ValueError, match="disk budget"):
        O.save_onestep(out, packed, ["a", "b", "c"])
    assert [p.name for p in out.iterdir()] == ["onestep_off.npz"]


def test_provenance_verifies_pinned_model_and_fingerprints_tooling(monkeypatch):
    provenance = O.source_provenance()
    assert provenance["model_src_pin_verified"]
    assert provenance["model_src_git_tree"] == O.MODEL_SRC_TREE
    assert provenance["tooling_overlay_sha256"]["tools/r13e_checks/offline.py"] == O.file_digest(Path(O.__file__))
    # A mismatch must be fatal even if checkpoint bytes still match.
    monkeypatch.setattr(O, "MODEL_SRC_TREE", "0" * 40)
    with pytest.raises(ValueError, match="model sources differ from pinned"):
        O.source_provenance()


@pytest.mark.parametrize("interrupt", [False, True])
def test_main_persists_flushed_progress_and_npz_before_rollouts(checkpoint, monkeypatch, interrupt):
    from wms.config import load

    path, _ = checkpoint
    workspace = path.parent
    out = workspace / "out"
    config = workspace / "config.toml"
    config.write_text("", encoding="utf-8")
    records = [state(i) for i in range(48)]
    write_json(workspace / "states.json", records)
    write_json(workspace / O.MANIFEST, {})
    monkeypatch.setattr(O, "cpu_limits", lambda: None)
    monkeypatch.setattr(load, "load_train_config", lambda *a: "cfg")
    monkeypatch.setattr(O, "source_provenance", lambda: {"model_revision": O.MODEL_REVISION})
    monkeypatch.setattr(O, "train_manifests", lambda *a: ([], {r["world"]: "train" for r in records}))
    monkeypatch.setattr(O, "train_datasets", lambda *a: "sets")

    def jobs(sets, panel):
        assert sets == "sets" and panel == records
        return [(0, 0, "panel_window")]

    monkeypatch.setattr(O, "onestep_jobs", jobs)

    def journal():
        return [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]

    def one_step(predictor, cfg, sets, *, workers, records, progress):
        assert workers == 1 and cfg == "cfg" and sets == "sets" and len(records) == 48
        progress({"phase": "one_step", "state": 0, "arm": "off", "rows": 1})
        assert [e["phase"] for e in journal()] == ["start", "one_step"]  # still open; flush required
        return {arm: np.zeros((1, 1), np.float32) for arm in O.ARMS}, ["win"], ["panel_window"]

    def rollouts(predictor, capture, selected, *, workers, progress):
        assert workers == 1 and selected == records
        assert len(list(out.glob("*.npz"))) == 3
        progress({"phase": "rollout", "state": 0, "replicate": 0, "arm": "off", "metrics": {"n": 240}})
        assert journal()[-1]["metrics"] == {"n": 240}
        if interrupt:
            raise RuntimeError("interrupted")
        return {"test": "rollouts"}

    monkeypatch.setattr(O, "run_onestep", one_step)
    monkeypatch.setattr(O, "analyze_onestep", lambda *a: {"test": "one_step"})
    monkeypatch.setattr(O, "run_rollouts", rollouts)
    args = ["--checkpoint", str(path), "--train-root", str(workspace), "--states", str(workspace / "states.json"),
            "--out", str(out), "--config", str(config), "--one-step-panel"]
    if interrupt:
        with pytest.raises(RuntimeError, match="interrupted"):
            O.main(args)
        assert not (out / "report.json").exists()
    else:
        assert O.main(args) == 0
        report = json.loads((out / "report.json").read_text())
        assert report["one_step_panel"] and report["workers"] == 1 and report["strict_exact_weights"]
        assert report["one_step_windows"] == ["panel_window"]
        assert report["npz_bytes"] < O.MAX_NPZ_BYTES and report["provenance"]["model_revision"] == O.MODEL_REVISION
    assert journal()[-1]["phase"] == "rollout" and len(list(out.glob("*.npz"))) == 3
