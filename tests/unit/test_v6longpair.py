"""Long paired horizons through real C4 loaders, collation and CPU truncated BPTT.

Synthetic converted episodes need no simulator or local data.  The final smoke
test additionally checks the workspace's converted real branches when present.
"""
from __future__ import annotations

import dataclasses
import gc
import json
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from wms.config.load import load_train_config
from wms.config.train import PairedDiffLossConfig, TrainConfig
from wms.core.errors import ConfigError
from wms.data.c4_episodes import MANIFEST_NAME, SOURCE_SCHEMA, _open_episode, episode_arrays
from wms.data.c4_pairs import PairedBatchDataset, PairedBranchSource
from wms.data.collate import collate
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.train.inputs import build_targets, prepare_inputs
from wms.train.losses.compose import build_loss
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

ROOT = Path(__file__).resolve().parents[2]
REAL_BRANCHES = ROOT / "_out/v6longpair/smoke_real/paired/branch/c4_episodes"
CPU = torch.device("cpu")


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def workspace_tmp():
    # Deliberately independent of pytest's system-temp default, including on Windows.
    parent = ROOT / "_out/v6longpair"
    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="unit-longpair-", dir=parent) as directory:
        yield Path(directory)
        _open_episode.cache_clear()
        gc.collect()  # release the converter reader's memory maps before Windows cleanup


def _write_source(root, groups):
    """Write the real converter's arrays from deterministic, distinct branch motions.

    Each tuple gives arm durations in seconds, including an intentionally uneven
    group when requested. All arms share 10 s of prehistory and the exact t0 state.
    """
    root.mkdir()
    entries = []
    k, t0 = 4, 10.0
    for group, durations in groups.items():
        for arm, duration in enumerate(durations):
            times = np.arange(0.0, t0 + duration + 0.25, 0.5)
            controls = np.arange(0.0, t0 + duration, 1.0)
            shape = (len(times), k)
            tau = np.maximum(times - t0, 0.0)[:, None]
            effect = arm * np.array([1.0, 0.7, 0.4, 0.0])[None, :]
            speed = np.broadcast_to(5.0 - 0.5 * effect * (1 - np.exp(-tau / 10)), shape)
            position = (np.array([-1000, -1040, -1100, -1150])[None, :]
                        + 5.0 * times[:, None]
                        - 0.5 * effect * (tau - 10 * (1 - np.exp(-tau / 10))))
            ctrl = np.zeros((len(controls), k), np.int8)
            if arm:
                ctrl[controls >= t0, 0] = 3
            raw = dict(
                present=np.ones(shape, bool),
                lane=np.broadcast_to(np.array([0, 0, 1, 1]), shape),
                vid_index=np.broadcast_to(np.arange(k, dtype=np.int32), shape),
                s_m=position, v_mps=speed,
                a_mps2=-0.05 * effect * np.exp(-tau / 10) * (tau > 0),
                is_cav=np.broadcast_to(np.array([True, False, False, False]), shape),
                time_s=times, control_time_s=controls, controller=ctrl,
                issued_acceleration_mps2=np.where(ctrl == 3, -0.05, 0),
                issued_target_speed_mps=np.zeros_like(ctrl, dtype=np.float32),
            )
            name = f"{group}_{'native' if arm == 0 else f'arm{arm}'}"
            directory = root / name
            directory.mkdir()
            for key, value in episode_arrays(raw).items():
                np.save(directory / f"{key}.npy", value, allow_pickle=False)
            entries.append(dict(
                episode=name, dir=name, world="w4107m2700p20t10", arm=name,
                pair_key=group, branch={"kind": "B" if arm == 0 else "SEED"},
                warmup_s=t0, horizon_s=duration, split_role="train",
                controllers_used=["native", "direct"], controller="direct",
            ))
    (root / MANIFEST_NAME).write_text(json.dumps(dict(
        schema=SOURCE_SCHEMA, split="TRAIN", k_slots=k, dt_s=0.5,
        allow_designed=False, n_episodes=len(entries), episodes=entries)), encoding="utf-8")
    return root


@pytest.fixture(scope="module", params=[240, 300], ids=["long240s", "long300s"])
def mixed_source(workspace_tmp, request):
    longest = request.param
    path = _write_source(workspace_tmp / f"mixed{longest}", {
        "short": (60, 60), "medium": (120, 120), "long": (longest, longest),
    })
    return path, longest


def _source(path, **kwargs):
    return PairedBranchSource(path, horizon_steps=600, history_s=10.0,
                              entries_v2=True, command_known=True, **kwargs)


def _wrapped(source, *, groups=3, fraction=1.0):
    return PairedBatchDataset(source.dataset, source, fraction=fraction, groups=groups,
                              max_arms=2, batch_size=2 * groups, seed=17)


def _assert_identical(a, b):
    """Exact bytes, dtypes, shapes and structure, rather than approximate float equality."""
    assert type(a) is type(b)
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_identical(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            _assert_identical(left, right)
    elif isinstance(a, (np.ndarray, torch.Tensor)):
        assert a.shape == b.shape and a.dtype == b.dtype
        if isinstance(a, torch.Tensor):
            a, b = a.detach().cpu().numpy(), b.detach().cpu().numpy()
        assert a.tobytes() == b.tobytes()
    else:
        assert a == b


def test_fixed_config_defaults_and_stage_roundtrips():
    default = TrainConfig()
    explicit = dataclasses.replace(default, loss=dataclasses.replace(
        default.loss, paired_diff=PairedDiffLossConfig(horizon_mode="fixed")))
    assert default.to_dict() == explicit.to_dict()
    assert default.to_legacy_flat() == explicit.to_legacy_flat()
    assert "paired_diff_horizon_mode" not in default.to_legacy_flat()
    assert TrainSettings.from_config(default).paired_diff_horizon_mode == "fixed"
    stage = dataclasses.replace(default, loss=dataclasses.replace(
        default.loss, paired_diff=PairedDiffLossConfig(horizon_mode="stage")))
    assert stage.to_legacy_flat()["paired_diff_horizon_mode"] == "stage"
    assert TrainConfig.from_dict(stage.to_dict()) == stage
    restored = TrainConfig.from_legacy_flat(stage.to_legacy_flat())
    assert restored.loss.paired_diff == stage.loss.paired_diff
    assert TrainSettings.from_config(restored).paired_diff_horizon_mode == "stage"


@pytest.mark.parametrize("mode", ["adaptive", "STAGE", ""])
def test_invalid_modes_rejected_at_config_settings_and_source(mode, workspace_tmp):
    with pytest.raises(ConfigError, match="horizon_mode"):
        PairedDiffLossConfig(horizon_mode=mode)
    with pytest.raises(ConfigError, match="horizon_mode"):
        TrainConfig.from_legacy_flat({"paired_diff_horizon_mode": mode})
    with pytest.raises(ValueError, match="horizon_mode"):
        TrainSettings(paired_diff_horizon_mode=mode)
    with pytest.raises(ValueError, match="horizon_mode"):
        _source(workspace_tmp / "unused", horizon_mode=mode)


def test_v6_preset_inherits_recipe_and_roundtrips():
    cfg = load_train_config(str(ROOT / "configs/c4_wm_v6_longpair.toml"))
    base = load_train_config(str(ROOT / "configs/c4_wm_v5.toml"))
    cfg.validate_for_training()
    assert cfg.loss.paired_diff.horizon_mode == "stage"
    assert cfg.loss.paired_diff.weight > 0
    assert dataclasses.replace(cfg.loss.paired_diff, horizon_mode="fixed") == base.loss.paired_diff
    assert cfg.model == base.model and cfg.curriculum == base.curriculum
    paired, = [source for source in cfg.data.c4_sources if source.paired]
    assert paired.path == "v6_longpair"
    assert cfg.curriculum.unroll_s[-1] >= 120
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
    flat = cfg.to_legacy_flat()
    assert TrainConfig.from_legacy_flat(flat).to_legacy_flat() == flat
    assert TrainSettings.from_config(cfg).paired_diff_horizon_mode == "stage"


def test_default_fixed_batches_and_backward_are_bit_identical(mixed_source):
    path, _ = mixed_source
    default, explicit = _source(path), _source(path, horizon_mode="fixed")
    assert default.horizon == explicit.horizon == 120
    _assert_identical(default.receipt, explicit.receipt)
    _assert_identical(default.groups, explicit.groups)
    batches = []
    for source in (default, explicit):
        wrapped = _wrapped(source)
        wrapped.set_paired_horizon(600)  # fixed remains globally capped at 60 s
        batch = collate(wrapped.__getitems__(list(range(6))))
        assert batch["states"].shape[1] == 121
        batches.append(batch)
    _assert_identical(*batches)
    runs = []
    for batch, options in zip(batches, ({}, {"paired_diff_horizon_mode": "fixed"})):
        torch.manual_seed(123)
        model = WmSModel(ModelSpec()).train()
        settings = TrainSettings(threads=1, paired_diff_weight=1.0, **options)
        runner = UnrollRunner(model, settings, build_loss(settings, physics=model.physics),
                              projection=None, device=CPU)
        record = runner.run(batch, unroll_steps=6, truncation_steps=3, backward=True)
        grads = {name: p.grad.clone() for name, p in model.named_parameters()
                 if p.grad is not None}
        assert grads and any(bool(g.ne(0).any()) for g in grads.values())
        runs.append((record, grads))
    _assert_identical(*runs)


@pytest.mark.parametrize("requested_s", [30, 60, 90, 120, 180, 240, 270, 300, 360])
def test_stage_selects_covering_groups_else_longest(mixed_source, requested_s):
    path, longest = mixed_source
    source = _source(path, horizon_mode="stage")
    wrapped = _wrapped(source)
    wrapped.set_paired_horizon(2 * requested_s)
    actual_s = min(requested_s, longest)
    available = {"short": 60, "medium": 120, "long": longest}
    expected = {key for key, seconds in available.items() if seconds >= actual_s}
    assert wrapped.paired_horizon == 2 * actual_s
    assert {source.groups[i]["pair_key"] for i in wrapped.eligible_groups} == expected
    records = wrapped.__getitems__(list(range(6)))
    assert {r["meta"]["pair_key"] for r in records} == expected
    _assert_identical(records, wrapped.__getitems__(list(range(6))))
    batch = collate(records)
    assert batch["states"].shape == (2 * len(expected), 2 * actual_s + 1, 4, 7)
    assert batch["road_continue"].shape[1] == 2 * actual_s
    assert batch["road_valid"].all() and batch["loss_valid"].all()
    assert batch["action_timeline"]["controller"].shape[1] == actual_s
    for i in range(0, len(records), 2):
        assert records[i]["meta"]["arm"].endswith("_native")
        assert batch["pair_ref"][i:i + 2].tolist() == [i, i]
        assert torch.equal(batch["states"][i, 0], batch["states"][i + 1, 0])
        assert not torch.equal(batch["cav_teacher"][i, 1:], batch["cav_teacher"][i + 1, 1:])


def test_short_arm_only_caps_its_group_and_stage_can_move_backwards(workspace_tmp):
    path = _write_source(workspace_tmp / "uneven", {
        "short": (30, 30), "uneven": (300, 60), "long": (300, 300),
    })
    source = _source(path, horizon_mode="stage")
    assert {g["pair_key"]: g["horizon_steps"] for g in source.groups} == {
        "short": 60, "uneven": 120, "long": 600,
    }
    assert source.horizon == 600
    wrapped = _wrapped(source)
    for requested, expected in ((240, {"long"}), (720, {"long"}),
                                (120, {"long", "uneven"}), (60, {"short", "uneven", "long"})):
        wrapped.set_paired_horizon(requested)
        records = wrapped.__getitems__(list(range(6)))
        assert {r["meta"]["pair_key"] for r in records} == expected
        assert {len(r["states"]) for r in records} == {min(requested, 600) + 1}
    for invalid in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            wrapped.set_paired_horizon(invalid)
    uneven = next(i for i, g in enumerate(source.groups) if g["pair_key"] == "uneven")
    with pytest.raises(ValueError, match="exceeds available"):
        source.group_records(uneven, source.groups[uneven]["rows"], base_row=0,
                             horizon_steps=240)


def test_actual_120s_batch_runs_240_steps_with_truncated_backward(mixed_source, monkeypatch):
    path, _ = mixed_source
    source = _source(path, horizon_mode="stage")
    wrapped = _wrapped(source, groups=1)
    wrapped.set_paired_horizon(240)
    batch = next(iter(DataLoader(wrapped, batch_size=2, num_workers=0, collate_fn=collate)))
    assert batch["states"].shape == (2, 241, 4, 7)
    torch.manual_seed(123)
    model = WmSModel(ModelSpec()).train()
    settings = TrainSettings(threads=1, paired_diff_weight=1.0, aggregate_weight=0.1,
                             paired_diff_horizon_mode="stage", truncation_steps=20)
    composer = build_loss(settings, physics=model.physics)
    prepared, chunks = [], []
    original_prepare, original_rollout = composer.prepare, model.rollout

    def capture_targets(targets):
        original_prepare(targets)
        prepared.append(targets)

    def capture_rollout(current, **kwargs):
        assert all(not value.requires_grad for value in current.values()
                   if isinstance(value, torch.Tensor))
        chunks.append((kwargs["k0"], kwargs["n_steps"]))
        return original_rollout(current, **kwargs)

    monkeypatch.setattr(composer, "prepare", capture_targets)
    monkeypatch.setattr(model, "rollout", capture_rollout)
    runner = UnrollRunner(model, settings, composer, projection=None, device=CPU)
    record = runner.run(batch, unroll_steps=240, backward=True)
    assert chunks == [(start, 20) for start in range(0, 240, 20)]
    targets, = prepared
    assert targets.n_steps == 240 and targets.hdv_mask.shape == (2, 240, 4)
    for road in (targets.road, targets.pair.road):
        assert road.present.shape == (2, 241, 4)
        assert road.cont.shape == (2, 240, 4)
        assert road.valid.shape == (2, 241) and road.valid.all()
    assert targets.prepared["paired_diff"]["w_agg"] == 240
    assert targets.prepared["paired_diff"]["mask"].shape == (2, 240, 4)
    assert record["paired_horizon_steps"] == 240 and record["paired_horizon_s"] == 120.0
    assert record["paired_batch"] and record["paired_arms"] == 1
    for key in ("loss", "position", "speed", "pair_pos", "pair_speed", "pair_agg",
                "agg_field", "agg_reward"):
        assert np.isfinite(record[key]) and record[key] >= 0, key
    assert record["pair_pos"] > 0 and record["pair_agg"] > 0
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(g.device.type == "cpu" and torch.isfinite(g).all() for g in grads)
    assert any(bool(g.ne(0).any()) for g in grads)


def test_optional_real_converted_120s_batch():
    if not (REAL_BRANCHES / MANIFEST_NAME).is_file():
        pytest.skip("workspace converted long-branch smoke data is not present")
    source = _source(REAL_BRANCHES, horizon_mode="stage")
    wrapped = _wrapped(source, groups=1)
    wrapped.set_paired_horizon(240)
    batch = next(iter(DataLoader(wrapped, batch_size=2, num_workers=0, collate_fn=collate)))
    assert batch["states"].shape[1] == 241
    assert batch["pair_ref"].tolist() == [0, 0]
    assert torch.equal(batch["states"][0, 0], batch["states"][1, 0])
    assert not torch.equal(batch["cav_teacher"][0, 1:], batch["cav_teacher"][1, 1:])
    model = WmSModel(ModelSpec())
    settings = TrainSettings(threads=1, paired_diff_weight=1.0, aggregate_weight=0.1,
                             paired_diff_horizon_mode="stage")
    inputs = prepare_inputs(batch, model, settings, CPU, 240)
    targets = build_targets(inputs, batch, settings, model.physics.dt_s, CPU)
    assert targets.n_steps == 240
    assert targets.road.cont.shape[1] == targets.pair.road.cont.shape[1] == 240
    assert batch["road_valid"].all()


@pytest.fixture(scope="module")
def integration_source(workspace_tmp):
    return _write_source(workspace_tmp / "integration", {
        "short": (60, 60), "medium": (120, 120), "long": (240, 240),
    })


def _fit_config(*, mode="stage", workers=0, weight=1.0, stages=(30.0, 120.0, 300.0)):
    """One cheap optimizer-call stand-in per stage; a short draw would always be 10 s."""
    return TrainConfig.from_dict({
        "loss": {"paired_diff": {"weight": weight, "horizon_mode": mode}},
        "curriculum": {
            "unroll_s": stages, "stage_steps": [1] * len(stages),
            "truncation_steps_per_stage": [5 * (i + 1) for i in range(len(stages))],
            "mixed_horizon_frac": 1.0, "mixed_horizon_s": [10.0],
            "mixed_horizon_min_stage_s": 0.0,
        },
        "optim": {"batch_size": 2},
        "runtime": {"device": "cpu", "threads": 1, "omp": 1, "num_workers": workers},
    })


@pytest.mark.parametrize("workers,weighted", [(0, False), (0, True), (1, False)],
                         ids=["plain-loader", "weighted-loader", "persistent-worker"])
def test_fit_propagates_stage_rebuilds_loader_and_bypasses_short_draw(
        integration_source, workspace_tmp, monkeypatch, workers, weighted):
    from wms.train import trainer as trainer_module

    source = _source(integration_source, horizon_mode="stage")
    # Assembly has a separate ordinary dataset/receipt, not the paired source itself.
    inner = _source(integration_source, horizon_mode="fixed").dataset
    dataset = PairedBatchDataset(inner, source, fraction=1.0, groups=1,
                                 max_arms=2, batch_size=2, seed=17)
    weight_calls = []
    if weighted:
        def stage_weights(seconds):
            weight_calls.append(seconds)
            return np.ones(len(dataset))

        monkeypatch.setattr(dataset.inner, "stage_weights", stage_weights, raising=False)
    cfg = _fit_config(workers=workers)
    # Runtime validation currently requires zero workers. Probe the trainer's actual
    # persistent-worker implementation directly with this picklable C4 dataset too.
    trainer = trainer_module.WmSTrainer(
        cfg, WmSModel(ModelSpec()), dataset,
        out_dir=workspace_tmp / f"fit-{workers}-{int(weighted)}")
    monkeypatch.delenv(trainer_module.STAGE_SNAPSHOT_ENV, raising=False)
    events, loaders, processes, observed = [], [], [], []
    original_set = PairedBatchDataset.set_paired_horizon
    original_loader = trainer._make_loader

    def track_set(self, steps):
        if self is dataset:
            # Strong references to the old loader would keep its worker snapshot alive.
            if loaders:
                assert loaders[-1]() is None
                assert all(not process.is_alive() for process in processes)
            events.append(("set", steps))
        return original_set(self, steps)

    def track_loader(sampler):
        events.append(("loader", dataset.paired_horizon))
        if weighted:
            assert isinstance(sampler, torch.utils.data.WeightedRandomSampler)
        else:
            assert sampler is None
        loader = original_loader(sampler)
        assert loader.num_workers == workers
        assert loader.persistent_workers == bool(workers)
        if workers:
            loader.timeout = 30  # bounded failure if a spawned worker cannot load the fixture
        loaders.append(weakref.ref(loader))
        return loader

    def reject_short_draw(self, stage_s):
        pytest.fail(f"stage paired batch incorrectly requested a short draw at {stage_s} s")

    def fake_train_batch(batch, *, unroll_steps, truncation_steps):
        index = len(observed)
        expected_steps = (60, 240, 600)[index]
        available_steps = (60, 240, 480)[index]
        assert unroll_steps == expected_steps
        assert truncation_steps == 5 * (index + 1)
        assert batch["states"].shape == (2, available_steps + 1, 4, 7)
        assert batch["pair_ref"].tolist() == [0, 0]
        assert batch["road_continue"].shape[1] == available_steps
        assert batch["road_valid"].all()
        eligible = {"short", "medium", "long"} if index == 0 else (
            {"medium", "long"} if index == 1 else {"long"})
        assert {source.groups[g]["pair_key"] for g in batch["pair_group"].tolist()} <= eligible
        # Exercise the runner's actual input cap at the 300 s stage / 240 s fallback.
        inputs = prepare_inputs(batch, trainer.model, trainer.settings, CPU, unroll_steps)
        assert inputs.n_steps == available_steps
        if workers:
            loader = loaders[-1]()
            worker, = loader._iterator._workers
            assert worker.is_alive()
            assert worker not in processes
            processes.append(worker)
        observed.append((unroll_steps, batch["states"].shape[1]))
        return {"loss": 0.0, "unroll_steps": inputs.n_steps}

    monkeypatch.setattr(PairedBatchDataset, "set_paired_horizon", track_set)
    monkeypatch.setattr(trainer, "_make_loader", track_loader)
    monkeypatch.setattr(trainer, "train_batch", fake_train_batch)
    monkeypatch.setattr(trainer_module.MixedHorizonSchedule, "draw", reject_short_draw)
    try:
        result = trainer.fit()
        assert observed == [(60, 61), (240, 241), (600, 481)]
        assert events == [("set", 60), ("loader", 60), ("set", 240), ("loader", 240),
                          ("set", 600), ("loader", 480)]
        assert weight_calls == ([30.0, 120.0, 300.0] if weighted else [])
        assert [row["stage_s"] for row in result.history] == [30.0, 120.0, 300.0]
        assert all("mixed_horizon_s" not in row for row in result.history)
        assert result.receipt["steps"] == 3
        assert len(loaders) == 3 and all(reference() is None for reference in loaders)
        assert len(processes) == (3 if workers else 0)
        assert all(not process.is_alive() for process in processes)
    finally:
        # Also clean up if an assertion fails while a persistent iterator is live.
        for reference in loaders:
            loader = reference()
            if loader is not None and loader._iterator is not None:
                loader._iterator._shutdown_workers()


@pytest.mark.parametrize("mode,fraction,weight", [
    ("stage", 0.0, 1.0), ("fixed", 1.0, 1.0), ("stage", 1.0, 0.0),
], ids=["ordinary-batch", "fixed-paired-batch", "disabled-pair-loss"])
def test_fit_short_draw_still_applies_outside_stage_pairs(
        integration_source, workspace_tmp, monkeypatch, mode, fraction, weight):
    from wms.train import trainer as trainer_module

    source = _source(integration_source, horizon_mode=mode)
    # Ordinary windows must stack even though the stage-mode paired source has mixed lengths.
    inner = _source(integration_source, horizon_mode="fixed").dataset
    dataset = PairedBatchDataset(inner, source, fraction=fraction, groups=1,
                                 max_arms=2, batch_size=2, seed=17)
    cfg = _fit_config(mode=mode, weight=weight, stages=(120.0,))
    trainer = trainer_module.WmSTrainer(
        cfg, WmSModel(ModelSpec()), dataset,
        out_dir=workspace_tmp / f"short-draw-{mode}-{fraction}-{weight}")
    monkeypatch.delenv(trainer_module.STAGE_SNAPSHOT_ENV, raising=False)
    draws, batches = [], []
    original_draw = trainer_module.MixedHorizonSchedule.draw

    def track_draw(self, stage_s):
        draws.append(stage_s)
        return original_draw(self, stage_s)

    def fake_train_batch(batch, *, unroll_steps, truncation_steps):
        assert ("pair_ref" in batch) == bool(fraction)
        assert unroll_steps == 20 and truncation_steps == 5
        batches.append(batch["states"].shape[1])
        return {"loss": 0.0}

    monkeypatch.setattr(trainer_module.MixedHorizonSchedule, "draw", track_draw)
    monkeypatch.setattr(trainer, "train_batch", fake_train_batch)
    result = trainer.fit()
    assert draws == [120.0]
    assert batches == ([241] if mode == "stage" and fraction else [121])
    assert result.history[0]["mixed_horizon_s"] == 10.0


@pytest.mark.parametrize("mode", ["stage", "fixed"])
def test_assembly_forwards_paired_mode_to_real_constructor(
        integration_source, workspace_tmp, monkeypatch, mode):
    from wms.cli.assemble import build_datasets
    from wms.data import c4_pairs

    cfg = load_train_config(str(ROOT / "configs/c4_wm_v6_longpair.toml"), [
        f"loss.paired_diff.horizon_mode={mode}", "loss.paired_diff.fraction=1.0",
        "loss.paired_diff.groups=1", "loss.paired_diff.max_arms=2", "optim.batch_size=2",
        "data.jam.oversample=1.0", "runtime.device=cpu", "runtime.threads=1", "runtime.omp=1",
    ])
    calls, resolved_names = [], []

    def capture_constructor(root, **kwargs):
        calls.append((Path(root), dict(kwargs)))
        return PairedBranchSource(root, **kwargs)

    def resolver(name):
        resolved_names.append(name)
        return integration_source

    monkeypatch.setattr(c4_pairs, "PairedBranchSource", capture_constructor)
    train, val, resolved = build_datasets(cfg, workspace_tmp / f"assembly-{mode}", resolver)
    (path, kwargs), = calls
    assert path == integration_source
    assert kwargs["horizon_steps"] == round(max(cfg.curriculum.unroll_s) / 0.5) == 240
    assert kwargs["name"] == "genrand_branch"
    if mode == "stage":
        assert kwargs["horizon_mode"] == "stage"
    else:
        assert "horizon_mode" not in kwargs  # preserve the legacy constructor call
    assert set(resolved_names) == set(resolved) == {source.path for source in cfg.data.c4_sources}
    assert isinstance(train, PairedBatchDataset) and train.pairs.horizon_mode == mode
    assert val is None
    for steps in (60, 240):
        train.set_paired_horizon(steps)
        batch = next(iter(DataLoader(train, batch_size=2, num_workers=0, collate_fn=collate)))
        assert batch["pair_ref"].tolist() == [0, 0]
        assert batch["states"].shape[1] == (steps + 1 if mode == "stage" else 121)
        assert batch["road_valid"].all()
