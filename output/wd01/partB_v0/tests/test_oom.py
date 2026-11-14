"""OOM-fallback tests (simulated OutOfMemoryError on CPU, tiny steps).

No real training: a handful of optimizer steps on the tiny synthetic
dataset, with the step function raising the same class a CUDA OOM
raises, to prove the batch halves (floored), is logged, recorded and
survives a resume.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from partB_v0.data import V0Dataset, load_manifest
from partB_v0.train import (MIN_BATCH, MIN_TD_BATCH, MemberTrainer,
                            TrainConfig, oob_crps300, resolve_device)


def _cfg(root, out, **kw) -> TrainConfig:
    cfg = TrainConfig(data_root=str(root), out_dir=str(out), members=1,
                      batch_size=128, td_batch=64, max_steps=4,
                      probe_steps=2, seed=99, verify=False,
                      **kw)
    return cfg


def _ds(root):
    man = load_manifest(root, verify_files=False, log=lambda *_: None)
    return V0Dataset(man, log=lambda *_: None)


def test_oom_halves_batch_logs_and_continues(tiny_dataset, tmp_path,
                                             capsys):
    root, _ = tiny_dataset
    ds = _ds(root)
    cfg = _cfg(root, tmp_path / "out_oom")
    dev = resolve_device("cpu")
    tr = MemberTrainer(cfg, ds, 0, dev, log=print)
    tr.try_resume()
    real_step = tr._train_one_step
    calls = {"n": 0}

    def flaky_step(planned_total):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return real_step(planned_total)

    tr._train_one_step = flaky_step
    rep = tr.train(4, deadline=None,
                   eval_fn=lambda net: oob_crps300(net, ds, tr.oob_rows))

    # batch halved once (128->64, 64->32) and recorded
    assert cfg.batch_size == MIN_BATCH == 64
    assert cfg.td_batch == MIN_TD_BATCH == 32
    assert rep["oom_events"] == [dict(step=0, batch_size=128,
                                      td_batch=64, new_batch_size=64,
                                      new_td_batch=32)]
    assert rep["steps_done"] == 4          # completed after the retry
    out_text = capsys.readouterr().out
    assert "simulated OOM" in out_text or "CUDA OOM at step 0" in out_text
    assert "halving batch 128+64 -> 64+32" in out_text
    # ckpt meta carries the effective batch for resume
    meta = json.loads((tmp_path / "out_oom" / "ckpt" / "m0_meta.json")
                      .read_text("utf-8"))
    assert meta["batch_size"] == 64 and meta["td_batch"] == 32


def test_oom_at_floor_reraises(tiny_dataset, tmp_path):
    """Already at the batch floor: the OOM is not fixable -> re-raise."""
    root, _ = tiny_dataset
    ds = _ds(root)
    cfg = _cfg(root, tmp_path / "out_floor")
    cfg.batch_size, cfg.td_batch = MIN_BATCH, MIN_TD_BATCH
    dev = resolve_device("cpu")
    tr = MemberTrainer(cfg, ds, 0, dev, log=lambda *_: None)
    tr.try_resume()

    def oom_step(planned_total):
        raise torch.cuda.OutOfMemoryError("simulated OOM at floor")

    tr._train_one_step = oom_step
    with pytest.raises(torch.cuda.OutOfMemoryError):
        tr.train(4, deadline=None,
                 eval_fn=lambda net: oob_crps300(net, ds, tr.oob_rows))


def test_resumed_run_restores_oom_reduced_batch(tiny_dataset, tmp_path):
    """Resume after an OOM reduction must keep the reduced batch
    (resume-safe batching)."""
    root, _ = tiny_dataset
    ds = _ds(root)
    out = tmp_path / "out_resume"
    cfg = _cfg(root, out)
    dev = resolve_device("cpu")
    tr = MemberTrainer(cfg, ds, 0, dev, log=lambda *_: None)
    tr.try_resume()
    real_step = tr._train_one_step
    calls = {"n": 0}

    def flaky_step(planned_total):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return real_step(planned_total)

    tr._train_one_step = flaky_step
    tr.train(2, deadline=None,
             eval_fn=lambda net: oob_crps300(net, ds, tr.oob_rows))
    assert Path(cfg.out_dir, "ckpt", "m0_meta.json").exists()

    tr2 = MemberTrainer(cfg, ds, 0, dev, log=lambda *_: None)
    tr2.try_resume()
    assert tr2.cfg.batch_size == 64 and tr2.cfg.td_batch == 32
    assert tr2.oom_events == [dict(step=0, batch_size=128, td_batch=64,
                                   new_batch_size=64, new_td_batch=32)]
