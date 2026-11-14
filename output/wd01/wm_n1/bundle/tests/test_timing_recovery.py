"""Focused recovery tests for the timing gate (coordinator review of 767403d).

Cheap synthetic/mocked contracts only (no real pack, no training, no CUDA,
``WD_FORBID_D=1``):

* ``measure_idea_a_corpus`` must use the SAME batching path ``fit`` uses for the
  resolved workers (DataLoader when ``resolved_num_workers() > 0``, else the
  in-process iterator) and record the actual path/workers.
* ``pack_source`` must be the EXACT supplied pack path (owner 2.3), with the
  ``real-pack``/``synthetic`` kind kept separately.
* every timing workdir is PRESERVED on success and on failure (the old
  ``shutil.rmtree`` destroyed logs/checkpoints/failure evidence), and the gate
  binds the artifact paths.
* the throughput setup must make ONE strict canonical call to the trainer's
  ``prepare_training_source(source, windows=None)`` BEFORE any batch iterator
  or optimizer is built (coordinator contrastfix contract centralizing the A6
  gate, TRAIN-only gate, eligible pair pool + fail-closed empty-pool refusal,
  and the TRAIN-only outcome normalization).  A trainer WITHOUT the canonical
  method FAILS CLOSED (no legacy/raw fallback, no duplicated inline logic);
  a preparation failure propagates and the workdir is preserved.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from train import idea_a as ia

from bundle import timing as T


class _FakeIAConfig:
    def __init__(self, num_workers):
        self.lr = 1e-3
        self._nw = int(num_workers)
        self.outcome_norm_min_scale = 1e-3

    def resolved_num_workers(self):
        return self._nw


class _FakeCorpus:
    def __init__(self, pack_dir, k_max):
        self.pack_dir = str(pack_dir)
        self.k_max = int(k_max)
        self.split = "TRAIN"
        # cheap item metadata (the canonical eligible-pool input): every item
        # carries a pair_key, like the real packed corpus items
        self._items = [{"pair_key": "w1@0", "policy": "native"},
                       {"pair_key": "w1@0", "policy": "__native"}]
        self._order = [0, 1]

    def __len__(self):
        return 0


class _FakeTrainer:
    last = None

    def __init__(self, *, raise_in_step=False, eligible_pool=None,
                 with_prepare=True):
        self.used = None
        self.calls = []
        self._last_terms = ["residual_nll", "rollout"]
        self._contrast_pair_pool = "unset"
        self.outcome_norm = None
        self.raise_in_step = bool(raise_in_step)
        self.eligible_pool = list(eligible_pool or [])
        if with_prepare:
            # the coordinator's CANONICAL trainer interface (the only
            # preparation path the timing gate may use): centralizes A6,
            # the TRAIN gate, the eligible pool + empty-pool refusal and the
            # TRAIN-only outcome normalization - mocked here, never duplicated
            self.prepare_training_source = self._prepare
        _FakeTrainer.last = self

    def _prepare(self, source, windows=None):
        self.calls.append(("prepare", source, windows))
        if not self.eligible_pool:
            # A's canonical empty-pool refusal (fail closed, like fit)
            raise ia.IdeaAConfigError(
                "the TRAIN source has no eligible sibling pair with shared "
                "outcome-tau or valid-HDV-slot support for the registered "
                "CONTRAST loss (2 item(s), 0 eligible pairs); refusing "
                "(contrastfix)")
        self._contrast_pair_pool = list(self.eligible_pool)
        self.outcome_norm = "prepared-norm"
        return list(self.eligible_pool)

    def _gen(self):
        i = 0
        while True:
            yield {"i": i}
            i += 1

    def _iter_batches(self, corpus):
        self.used = "in-process"
        self.calls.append(("sampler", "in-process"))
        return self._gen()

    def _iter_batches_loader(self, corpus):
        self.used = "dataloader"
        self.calls.append(("sampler", "dataloader"))
        return self._gen()

    def _optim_step(self, batch, optimizer):
        self.calls.append(("optim",))
        if self.raise_in_step:
            raise RuntimeError("boom-step")
        return 0.0

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(3))])


def _mock_trainer(monkeypatch, *, num_workers, raise_in_step=False,
                  eligible_pool=("pair-0",), with_prepare=True):
    created = {}

    def fake_idea_a_trainer(pack_dir, **kw):
        out_dir = Path(kw["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "TRAIN_RECEIPT.json").write_text("{}", encoding="utf-8")
        created["out_dir"] = out_dir
        corpus = _FakeCorpus(pack_dir, kw.get("k_max"))
        created["corpus"] = corpus
        trainer = _FakeTrainer(raise_in_step=raise_in_step,
                               eligible_pool=eligible_pool,
                               with_prepare=with_prepare)
        return _FakeIAConfig(num_workers), corpus, trainer

    monkeypatch.setattr(T, "_idea_a_trainer", fake_idea_a_trainer)

    def fake_make_optimizer(trainer, lr):
        _FakeTrainer.last.calls.append(("optimizer",))
        return object(), None

    monkeypatch.setattr(ia, "_make_optimizer", fake_make_optimizer)
    return created


def _cfg(num_workers):
    return T.make_config(tiny=True, num_workers=num_workers)


def test_in_process_path_when_workers_zero(tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    pack.mkdir()
    _mock_trainer(monkeypatch, num_workers=0)
    wd = tmp_path / "wd"

    out = T.measure_idea_a_corpus(_cfg(0), "cpu", warmup=1, timed_steps=2,
                                  k_max=192, pack_dir=str(pack), work_dir=wd)

    assert out["batch_path"] == "in-process"
    assert _FakeTrainer.last.used == "in-process"
    assert out["num_workers"] == 0
    assert out["pack_source"] == str(pack)
    assert out["source_kind"] == "real-pack"
    assert out["work_dir"] == str(wd)
    assert (wd / "out" / "TRAIN_RECEIPT.json").is_file()


def test_dataloader_path_when_workers_positive(tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    pack.mkdir()
    _mock_trainer(monkeypatch, num_workers=2)
    wd = tmp_path / "wd"

    out = T.measure_idea_a_corpus(_cfg(2), "cpu", warmup=1, timed_steps=2,
                                  k_max=192, pack_dir=str(pack), work_dir=wd)

    assert out["batch_path"] == "dataloader"
    assert _FakeTrainer.last.used == "dataloader"
    assert out["num_workers"] == 2
    assert out["pack_source"] == str(pack)
    assert out["source_kind"] == "real-pack"


def test_workdir_preserved_on_failure(tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    pack.mkdir()
    created = _mock_trainer(monkeypatch, num_workers=0, raise_in_step=True)
    wd = tmp_path / "wd"

    with pytest.raises(RuntimeError, match="boom-step"):
        T.measure_idea_a_corpus(_cfg(0), "cpu", warmup=1, timed_steps=2,
                                k_max=192, pack_dir=str(pack), work_dir=wd)

    # the failure workdir (log/checkpoint evidence) is NEVER removed
    assert wd.is_dir()
    assert (created["out_dir"] / "TRAIN_RECEIPT.json").is_file()


def test_synthetic_pack_source_is_actual_temp_path(tmp_path, monkeypatch):
    _mock_trainer(monkeypatch, num_workers=0)
    wd = tmp_path / "wd"

    out = T.measure_idea_a_corpus(_cfg(0), "cpu", warmup=1, timed_steps=1,
                                  k_max=192, tiny=True, work_dir=wd)

    assert out["source_kind"] == "synthetic"
    assert Path(out["pack_source"]).is_dir()
    assert Path(out["pack_source"]).name == "pack"


def test_gate_binds_and_preserves_artifact_paths(tmp_path, monkeypatch):
    gate_work = tmp_path / "gate"
    ia_wd, ia_out = gate_work / "idea_a", gate_work / "idea_a" / "out"
    loss_wd, loss_out = gate_work / "loss", gate_work / "loss" / "out"
    pack = tmp_path / "pack"
    pack.mkdir()

    monkeypatch.setattr(T, "measure_idea_a_corpus", lambda cfg, device, **kw: {
        "k_max": 192, "pack_source": str(pack), "source_kind": "real-pack",
        "batch_path": "dataloader", "num_workers": 2, "steps_per_sec": 1.0,
        "n_params": 10, "peak_memory_mb": 1.0, "peak_cuda_memory_bytes": None,
        "work_dir": str(ia_wd), "out_dir": str(ia_out)})
    monkeypatch.setattr(T, "loss_decrease_idea_a", lambda cfg, device, **kw: {
        "decreased": True, "active_loss_terms": ["residual_nll"],
        "work_dir": str(loss_wd), "out_dir": str(loss_out)})
    monkeypatch.setattr(T, "measure_throughput", lambda cfg, device, **kw: {
        "rollout_scene_steps_per_sec": 1.0, "rollout_samples_per_sec": 1.0,
        "peak_memory_mb": 1.0})
    monkeypatch.setattr(T, "loss_decrease_check",
                        lambda cfg, device, **kw: {"decreased": True})
    monkeypatch.setattr(T.T, "extrapolate",
                        lambda cfg, throughput: {"feasible": True,
                                                 "total_hours": 1.0})

    gate = T.run_timing_gate(_cfg(2), device="cpu", tiny=True,
                             pack_dir=str(pack), work_dir=gate_work,
                             warmup=1, timed_steps=1, loss_steps=2)

    assert gate["work_dir"] == str(gate_work)
    assert gate["pack_source"] == str(pack)
    assert gate["source_kind"] == "real-pack"
    assert gate["batch_path"] == "dataloader"
    assert gate["artifacts"] == {
        "work_dir": str(gate_work),
        "idea_a_work_dir": str(ia_wd), "idea_a_out_dir": str(ia_out),
        "loss_work_dir": str(loss_wd), "loss_out_dir": str(loss_out)}
    assert gate_work.is_dir()


# ------------- canonical fit preparation shared by throughput (strict) -------
def test_canonical_prepare_runs_once_before_sampler_and_optimizer(
        tmp_path, monkeypatch):
    """ONE strict ``prepare_training_source`` call, then sampler/optimizer.

    The timing gate must call the trainer's CANONICAL preparation exactly
    once, with the measured source and ``windows=None``, BEFORE any batch
    iterator or optimizer is built - and never duplicate the preparation
    logic inline (the pool/norm the canonical call bound is what the
    measurement sees).
    """
    pack = tmp_path / "pack"
    pack.mkdir()
    created = _mock_trainer(monkeypatch, num_workers=0,
                            eligible_pool=("w1@t0",))

    out = T.measure_idea_a_corpus(_cfg(0), "cpu", warmup=1, timed_steps=1,
                                  k_max=192, pack_dir=str(pack), work_dir=None)

    trainer = _FakeTrainer.last
    # exactly ONE canonical prepare call: (corpus, windows=None)
    prepares = [c for c in trainer.calls if c[0] == "prepare"]
    assert len(prepares) == 1
    assert prepares[0][1] is created["corpus"]
    assert prepares[0][2] is None
    # the canonical call bound the pool + TRAIN-only norm (no inline duplicate)
    assert trainer._contrast_pair_pool == ["w1@t0"]
    assert trainer.outcome_norm == "prepared-norm"
    # strict ordering: prepare BEFORE the batch iterator, the optimizer and
    # every optim step (warmup + timed)
    assert [c[0] for c in trainer.calls] == \
        ["prepare", "sampler", "optimizer", "optim", "optim"]
    assert trainer.calls[1] == ("sampler", "in-process")
    assert out["batch_path"] == "in-process"
    assert out["steps_per_sec"] > 0


def test_missing_canonical_prepare_fails_closed_and_preserves_workdir(
        tmp_path, monkeypatch):
    """A trainer WITHOUT ``prepare_training_source`` is refused outright.

    No legacy/raw fallback and no inline re-implementation may run: the gate
    must FAIL CLOSED (never silently measure the pre-contrastfix trainer with
    the raw unsupported pair pool), before any sampler/optimizer/step, and
    the workdir evidence must survive.
    """
    pack = tmp_path / "pack"
    pack.mkdir()
    created = _mock_trainer(monkeypatch, num_workers=0, with_prepare=False)
    wd = tmp_path / "wd"

    with pytest.raises(RuntimeError, match="prepare_training_source"):
        T.measure_idea_a_corpus(_cfg(0), "cpu", warmup=1, timed_steps=1,
                                k_max=192, pack_dir=str(pack), work_dir=wd)

    trainer = _FakeTrainer.last
    assert not hasattr(trainer, "prepare_training_source")
    # nothing after the refusal ran: no sampler, no optimizer, no optim step
    assert [c[0] for c in trainer.calls] == []
    assert created["out_dir"].is_dir()
    assert wd.is_dir()
    assert (wd / "out" / "TRAIN_RECEIPT.json").is_file()


def test_canonical_preparation_failure_propagates_and_preserves_workdir(
        tmp_path, monkeypatch):
    """A refusal raised BY the canonical preparation propagates unchanged.

    A's empty-pool ``IdeaAConfigError`` (fail closed, fit parity) must reach
    the caller with the workdir preserved and nothing measured after it.
    """
    pack = tmp_path / "pack"
    pack.mkdir()
    created = _mock_trainer(monkeypatch, num_workers=0, eligible_pool=[])
    wd = tmp_path / "wd"

    with pytest.raises(ia.IdeaAConfigError, match="0 eligible pairs"):
        T.measure_idea_a_corpus(_cfg(0), "cpu", warmup=1, timed_steps=1,
                                k_max=192, pack_dir=str(pack), work_dir=wd)

    trainer = _FakeTrainer.last
    assert [c[0] for c in trainer.calls] == ["prepare"]
    assert created["out_dir"].exists()
    assert wd.is_dir()
    assert (wd / "out" / "TRAIN_RECEIPT.json").is_file()
