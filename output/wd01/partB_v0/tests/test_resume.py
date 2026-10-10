"""Resume-equivalence and corrupt-checkpoint-fallback tests (CPU, tiny).

The owner rule is respected: these run a handful of optimizer steps on a
tiny synthetic dataset -- a determinism check, not training.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from partB_v0.data import load_manifest
from partB_v0.train import CheckpointDataMismatch, MemberTrainer, \
    TrainConfig, resolve_device


def _cfg(data_root, out_dir, max_steps, **kw) -> TrainConfig:
    return TrainConfig(data_root=str(data_root), out_dir=str(out_dir),
                       members=1, batch_size=32, td_batch=16,
                       max_steps=max_steps, probe_steps=0, seed=1234,
                       verify=False, resume=True, **kw)


def _member_at(ds, cfg, planned, stop_after=None):
    dev = resolve_device("cpu")
    tr = MemberTrainer(cfg, ds, 0, dev, log=lambda *_: None)
    tr.try_resume()
    from partB_v0.train import oob_crps300
    tr.train(planned, deadline=None,
             eval_fn=lambda net: oob_crps300(net, ds, tr.oob_rows),
             stop_after=stop_after)
    return tr


def _load_ds(root):
    man = load_manifest(root, verify_files=False, log=lambda *_: None)
    from partB_v0.data import V0Dataset
    return V0Dataset(man, log=lambda *_: None)


def test_resume_equals_straight_run(tiny_dataset):
    """N steps of an 8-step plan -> stop -> resume the SAME plan ==
    a straight 8-step run within 1e-6 (CPU, deterministic seed)."""
    root, _ = tiny_dataset
    ds = _load_ds(root)
    N, TOTAL = 4, 8

    straight = _member_at(ds, _cfg(root, Path(root).parent / "out_straight",
                                   TOTAL), TOTAL)
    # interrupted mid-plan: same planned_total, stopped after N steps
    part = _member_at(ds, _cfg(root, Path(root).parent / "out_part",
                               TOTAL), TOTAL, stop_after=N)
    assert part.step == N
    meta = json.loads((Path(root).parent / "out_part" / "ckpt"
                       / "m0_meta.json").read_text("utf-8"))
    assert meta["step"] == N and meta["completed"] is False

    resumed = _member_at(ds, _cfg(root, Path(root).parent / "out_part",
                                  TOTAL), TOTAL)
    assert resumed.step == TOTAL

    a = torch.nn.utils.parameters_to_vector(
        straight.net.parameters()).detach()
    b = torch.nn.utils.parameters_to_vector(
        resumed.net.parameters()).detach()
    max_diff = float((a - b).abs().max())
    print(f"resume equivalence: max |param diff| after {N}+{TOTAL - N} "
          f"steps = {max_diff:.3e}")
    assert max_diff < 1e-6

    # optimizer moments resumed too (AdamW exp_avg/exp_avg_sq)
    for (k1, v1), (k2, v2) in zip(straight.opt.state.items(),
                                  resumed.opt.state.items()):
        for name in ("exp_avg", "exp_avg_sq"):
            assert torch.allclose(v1[name], v2[name], atol=1e-6), name


def test_truncated_latest_falls_back_to_previous(tiny_dataset):
    """A truncated/corrupt latest checkpoint resumes from .prev."""
    root, _ = tiny_dataset
    ds = _load_ds(root)
    out = Path(root).parent / "out_fb"

    part = _member_at(ds, _cfg(root, out, 3), 3)
    full = _member_at(ds, _cfg(root, out, 6), 6)
    assert full.step == 6

    ckpt = out / "ckpt"
    # corrupt the LATEST triple (meta json truncated, state pt truncated)
    (ckpt / "m0_meta.json").write_text('{"format": "wd01_partB_v0/1", '
                                       '"step": 6, "trunc')
    raw = (ckpt / "m0_state.pt").read_bytes()
    (ckpt / "m0_state.pt").write_bytes(raw[:len(raw) // 3])

    dev = resolve_device("cpu")
    tr = MemberTrainer(_cfg(root, out, 9), ds, 0, dev,
                       log=print)
    restored = tr.try_resume()
    assert restored == 3, "should have fallen back to the .prev (3-step) " \
                          "checkpoint"
    meta = json.loads((ckpt / "m0_meta.prev.json").read_text("utf-8"))
    assert meta["step"] == 3
    # and it continues training from there
    from partB_v0.train import oob_crps300
    rep = tr.train(5, deadline=None,
                   eval_fn=lambda net: oob_crps300(net, ds, tr.oob_rows))
    assert rep["steps_done"] == 5


def test_checkpoint_refuses_foreign_manifest(tiny_dataset):
    root, _ = tiny_dataset
    ds = _load_ds(root)
    out = Path(root).parent / "out_foreign"
    _member_at(ds, _cfg(root, out, 3), 3)

    ckpt = out / "ckpt"
    meta = json.loads((ckpt / "m0_meta.json").read_text("utf-8"))
    meta["data_manifest_sha256"] = "0" * 64
    (ckpt / "m0_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    dev = resolve_device("cpu")
    tr = MemberTrainer(_cfg(root, out, 3), ds, 0, dev,
                       log=lambda *_: None)
    with pytest.raises(CheckpointDataMismatch, match="different data"):
        tr.try_resume()
