"""GPUFIX tests: device-correct rollout/timing/fit, DataLoader parity,
empty-mask fail-closed, actual-device recording and per-step logging.

CPU only, tmp dirs, synthetic packs - no SUMO, no real data, no D:.
"""
from __future__ import annotations

import json
import inspect
from contextlib import contextmanager

import pytest
import torch

import train.idea_a as ia
from model import losses
from model.rollout import rollout as model_rollout
from test_idea_a_fakes import make_windows, seeded_bundle

#: the tensor factories the brief flagged: a missing ``device=`` here is a
#: host tensor on the GPU path.  ``*_like`` inherit and are deliberately not
#: guarded; ``rand*/randperm`` are CPU index/sampling generation.
_GUARDED_FACTORIES = ("zeros", "ones", "full", "arange", "tensor",
                      "as_tensor", "empty")


@contextmanager
def strict_device_factories():
    """Arm a strict device= check on the flagged ``torch`` tensor factories.

    Calls made from inside the torch package itself (optimizer state scalars,
    etc.) are exempt: the guard targets *our* train/eval/gate code, not
    library internals.  Module construction (``nn.Parameter(torch.zeros(...))``)
    is likewise expected to build on CPU and then ``.to(device)``.
    """
    originals = {}
    try:
        for name in _GUARDED_FACTORIES:
            originals[name] = getattr(torch, name)

            def make(orig, fname):
                def wrapper(*args, **kwargs):
                    if "device" not in kwargs:
                        frame = inspect.currentframe().f_back
                        origin = frame.f_globals.get("__file__", "") or ""
                        if ("site-packages" in origin
                                or "\\torch\\" in origin or "/torch/" in origin):
                            return orig(*args, **kwargs)
                        raise AssertionError(
                            f"torch.{fname} called without device= on the "
                            f"GPU path (caller={origin}, args={args!r})")
                    return orig(*args, **kwargs)
                return wrapper

            setattr(torch, name, make(originals[name], name))
        yield
    finally:
        for name, orig in originals.items():
            setattr(torch, name, orig)


# ------------------------------------------------- (A) device-correct -------
def test_timing_forward_loss_and_rollout_have_no_host_tensors():
    from bundle import timing as T

    cfg = T.make_config(tiny=True)
    model = T._build_model(cfg, "cpu")
    gen = torch.Generator().manual_seed(0)
    batch = T.make_scene(cfg, "cpu", gen)
    targets = T.make_targets(cfg, "cpu", gen)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    with strict_device_factories():
        opt.zero_grad(set_to_none=True)
        loss = losses.total_loss(model(**batch), targets)
        loss.backward()
    model.eval()
    with strict_device_factories(), torch.no_grad():
        out = model_rollout(model, batch, n_steps=2, seed=0)
    assert out["lanes"].shape[0] == 1


def test_forward_loss_on_meta_device_has_no_host_tensors():
    from bundle import timing as T

    cfg = T.make_config(tiny=True)
    model = T._build_model(cfg, "meta")
    gen = torch.Generator().manual_seed(0)
    batch = T.make_scene(cfg, "meta", gen)
    targets = T.make_targets(cfg, "meta", gen)
    pred = model(**batch)
    step = pred["step"]
    loss = losses.gmm_accel_nll(
        targets["accel"], step["accel_mu"], step["accel_log_sigma"],
        step["accel_mix_logits"], targets["target_mask"])
    assert loss.device.type == "meta"


def test_rollout_lane_where_full_is_on_the_vehicle_device():
    from model import constants as C

    B, K = 2, 4
    veh = torch.zeros(B, K, C.VEH_F, device="meta")
    lane = torch.where(
        veh[..., C.LANE_LEFT:C.LANE_ACCEL + 1].sum(-1) > 0,
        veh[..., C.LANE_LEFT:C.LANE_ACCEL + 1].argmax(-1).long(),
        torch.full((B, K), -1, dtype=torch.long, device=veh.device))
    assert lane.device.type == "meta"


def test_timing_gate_measurement_has_no_host_tensors():
    """The gate's real measurement path (train step + rollout) is device-clean."""
    from bundle import timing as T

    cfg = T.make_config(tiny=True)
    model = T._build_model(cfg, "cpu")     # module init owns host Parameters
    with strict_device_factories():
        out = T.measure_throughput(cfg, "cpu", warmup=1, timed_steps=1,
                                   model=model)
    assert out["steps_per_sec"] > 0
    assert out["rollout_scene_steps_per_sec"] > 0


def test_one_fit_step_has_no_host_tensors(tmp_path):
    """A full fit step (forward, five losses, backward, log) is device-clean."""
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=1, batch_size=2,
                              out_dir=str(tmp_path / "out"), num_workers=0)
    tr = ia.IdeaATrainer(cfg, corpus=None, bundle=seeded_bundle(0))
    windows = make_windows(1)              # fixture scaffolding is host-side
    with strict_device_factories():
        receipt = tr.fit(windows=windows)
    assert receipt["steps"] == 1
    assert (tmp_path / "out" / "train_log.jsonl").is_file()


# ---------------------------------------------- (B) empty-mask fail-closed -
def _collapsed_batch(*, empty_residual: bool = False,
                     empty_rollout: bool = False) -> dict:
    batch = ia.collate_canonical(make_windows(1))
    if empty_residual:
        batch["target_mask"] = torch.zeros_like(batch["target_mask"])
    if empty_rollout:
        batch["present0"] = torch.zeros_like(batch["present0"])
    return batch


def test_residual_nll_empty_mask_fails_closed_on_a_real_run():
    tr = ia.IdeaATrainer(ia.IdeaATrainConfig(out_dir="unused"),
                         corpus=None, bundle=seeded_bundle(0))
    tr._require_all_terms = True
    batch = _collapsed_batch(empty_residual=True)
    with pytest.raises(ia.IdeaAConfigError):
        tr.residual_nll(batch)
    # the documented test flag (not a real run) keeps the old 0.0
    tr._require_all_terms = False
    assert float(tr.residual_nll(batch)) == 0.0


def test_rollout_empty_mask_fails_closed_on_a_real_run():
    tr = ia.IdeaATrainer(ia.IdeaATrainConfig(out_dir="unused"),
                         corpus=None, bundle=seeded_bundle(0))
    tr._require_all_terms = True
    batch = _collapsed_batch(empty_rollout=True)
    with pytest.raises(ia.IdeaAConfigError):
        tr.rollout_loss(batch)


# ------------------------------------------------- DataLoader tensor parity -
def _many_group_runs(groups: int, arms_per_group: int):
    runs = []
    for g in range(groups):
        world = f"w{g}"
        t0 = 100.0 + g
        runs.append(dict(chunk="chunk4", world=world, arm="native",
                         policy="native", split="TRAIN", t0_s=t0))
        for a in range(arms_per_group - 1):
            policy = "switching" if a % 2 else "damping"
            arm = f"switching{a + 1}" if policy == "switching" else f"base{a + 1}"
            runs.append(dict(chunk="chunk4", world=world, arm=arm,
                             policy=policy, split="TRAIN", t0_s=t0))
    return runs


def test_dataloader_worker_batches_equal_in_process_tensor_by_tensor(tmp_path):
    from bundle.e2e import write_synthetic_pack

    pack = tmp_path / "pack"
    write_synthetic_pack(pack, k=8, n_ticks=4, seed=0,
                         runs=_many_group_runs(6, 4))
    corpus0 = ia.IdeaACorpus(str(pack), split="TRAIN", pair_sampling=True,
                             horizon_steps=2)
    corpus2 = ia.IdeaACorpus(str(pack), split="TRAIN", pair_sampling=True,
                             horizon_steps=2)
    cfg0 = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=1, batch_size=4,
                               out_dir="unused", num_workers=0)
    cfg2 = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=1, batch_size=4,
                               out_dir="unused", num_workers=2)
    bundle = seeded_bundle(0)
    tr0 = ia.IdeaATrainer(cfg0, corpus0, member=1, seed=101, bundle=bundle)
    tr2 = ia.IdeaATrainer(cfg2, corpus2, member=1, seed=101, bundle=bundle)
    in_proc = tr0._iter_batches(corpus0)
    loader = tr2._iter_batches_loader(corpus2)
    for _ in range(2):
        a = next(in_proc)
        b = next(loader)
        assert sorted(a) == sorted(b)
        assert a["meta"] == b["meta"]
        for key in sorted(k for k, v in a.items() if torch.is_tensor(v)):
            assert torch.equal(a[key], b[key]), f"tensor {key!r} differs"
        # the five A3.2 terms computed from the two batches must be equal
        terms_a = tr0.loss_terms(a, require_all=False)
        terms_b = tr0.loss_terms(b, require_all=False)
        assert sorted(terms_a) == sorted(terms_b)
        for key in terms_a:
            assert torch.allclose(terms_a[key], terms_b[key], atol=0.0,
                                  rtol=0.0), f"loss term {key!r} differs"


def test_corpus_is_picklable_after_array_access(tmp_path):
    """The memmap cache must not be serialized to DataLoader workers.

    ``np.memmap`` pickles its whole array (multi-GB for a real pack); the
    corpus must drop the cache on ``__getstate__`` and re-open it lazily.
    """
    import pickle

    from bundle.e2e import write_synthetic_pack

    pack = tmp_path / "pack"
    write_synthetic_pack(pack, k=8, n_ticks=4, seed=0,
                         runs=_many_group_runs(4, 4))
    corpus = ia.IdeaACorpus(str(pack), split="TRAIN", pair_sampling=True,
                            horizon_steps=2)
    first = corpus[0]                                   # populate the caches
    assert corpus._mm and corpus._v1._mm
    state = corpus.__getstate__()
    assert state["_mm"] == {}
    assert state["_v1"].__getstate__()["_mm"] == {}
    clone = pickle.loads(pickle.dumps(corpus, protocol=4))
    again = clone[0]
    assert sorted(first) == sorted(again)
    for key in sorted(k for k, v in first.items() if torch.is_tensor(v)):
        assert torch.equal(first[key], again[key]), f"tensor {key!r} differs"


# ------------------------------------- (C) actual device + per-step logging -
def test_actual_device_recorded_and_asserted(tmp_path):
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=1, batch_size=2,
                              out_dir=str(tmp_path / "out"), num_workers=0)
    tr = ia.IdeaATrainer(cfg, corpus=None, bundle=seeded_bundle(0),
                         device="cpu")
    assert tr.config.device == "cpu"
    receipt = tr.fit(windows=make_windows(1))
    assert receipt["steps"] == 1
    sidecar = json.loads((tmp_path / "out" / "member_1.json").read_text(
        encoding="utf-8"))
    assert sidecar["config"]["device"] == "cpu"
    # a mixed-device parameter is rejected by the fit assertion
    tr_bad = ia.IdeaATrainer(ia.IdeaATrainConfig(out_dir="unused",
                                                 num_workers=0),
                             corpus=None, bundle=seeded_bundle(0),
                             device="cpu")
    tr_bad.bundle.hdv.to("meta")
    with pytest.raises(ia.IdeaAConfigError):
        tr_bad._assert_on_device(ia.collate_canonical(make_windows(1)))


def test_train_log_jsonl_written_per_step(tmp_path):
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=3, batch_size=2,
                              out_dir=str(tmp_path / "out"), num_workers=0)
    tr = ia.IdeaATrainer(cfg, corpus=None, bundle=seeded_bundle(0))
    tr.fit(windows=make_windows(3))
    log = (tmp_path / "out" / "train_log.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    assert len(log) == 3
    rows = [json.loads(line) for line in log]
    assert [r["step"] for r in rows] == [1, 2, 3]
    for r in rows:
        assert isinstance(r["total"], float)
        assert isinstance(r["s_per_step"], float)
        assert "residual_nll" in r["terms"]
        assert "rollout" in r["terms"]
        assert "cuda_mem_bytes" in r
