"""INC1 run-once gate and model-lock verification (zero simulation)."""
from __future__ import annotations

import numpy as np
import pytest

from inc1eval import metrics as M
from inc1eval.model_lock import (ModelLockError, build_lock, load_lock, write_lock,
                                 sha256_file)

ARMS = ["R0", "R1", "R2", "R3", "R4"]


def _inputs(seed=0, perfect=True, worlds=5):
    rng = np.random.default_rng(seed)
    true = rng.uniform(0.0, 100.0, (worlds, len(ARMS)))
    a = true.copy() if perfect else rng.uniform(0.0, 100.0, true.shape)
    b = true.copy() if perfect else rng.uniform(0.0, 100.0, true.shape)
    return M.MetricsInputs(
        worlds=[f"w{i}" for i in range(worlds)], arms=list(ARMS),
        delay=M.Channel(true=true, idea_a=a, v2g=b),
        ttc_lt3_s=M.Channel(true=true * 0.1, idea_a=a * 0.1, v2g=b * 0.1),
        hard_brake_s=M.Channel(true=true * 0.2, idea_a=a * 0.2, v2g=b * 0.2))


def test_perfect_prediction_scores():
    report = M.compute_report(_inputs(), n_boot=32, seed=1)
    d = report["delay"]["idea_a"]
    assert d["top1_hit"]["point"] == 1.0
    assert d["spearman"]["point"] == pytest.approx(1.0)
    assert d["regret_s_per_veh"]["point"] == pytest.approx(0.0)


def _lock_files(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    manifest = pack / "manifest.json"
    manifest.write_text('{"schema":"wm_n1_pack/1"}\n', encoding="utf-8")
    member = tmp_path / "member_1.pt"
    member.write_bytes(b"checkpoint-bytes")
    v2g = tmp_path / "params_nominal.json"
    v2g.write_text('{"dt_s":0.5}\n', encoding="utf-8")
    return member, manifest, v2g


def test_gate_requires_lock(tmp_path):
    inputs = _inputs()
    with pytest.raises(ModelLockError):
        M.run_inc1_metrics(tmp_path, inputs, members={},
                           pack_manifest=tmp_path / "nope",
                           v2g_params=tmp_path / "nope", n_boot=8)


def test_gate_verifies_lock_and_runs_once(tmp_path):
    member, manifest, v2g = _lock_files(tmp_path)
    lock = build_lock(members={"member_1.pt": member},
                      pack_manifest=manifest, code_commit="abc123",
                      v2g_params=v2g)
    write_lock(tmp_path, lock)
    assert load_lock(tmp_path)["code_commit"] == "abc123"

    report = M.run_inc1_metrics(
        tmp_path, _inputs(), members={"member_1.pt": member},
        pack_manifest=manifest, v2g_params=v2g, code_commit="abc123",
        n_boot=16)
    assert report["schema"] == "wm_n1_inc1_metrics/1"
    assert (tmp_path / M.METRICS_NAME).is_file()
    assert (tmp_path / M.DONE_NAME).is_file()
    # second run is refused
    with pytest.raises(M.EvalAlreadyDone):
        M.run_inc1_metrics(tmp_path, _inputs(),
                           members={"member_1.pt": member},
                           pack_manifest=manifest, v2g_params=v2g,
                           code_commit="abc123", n_boot=8)


def test_gate_rejects_swapped_checkpoint(tmp_path):
    member, manifest, v2g = _lock_files(tmp_path)
    lock = build_lock(members={"member_1.pt": member},
                      pack_manifest=manifest, code_commit="abc123",
                      v2g_params=v2g)
    write_lock(tmp_path, lock)
    member.write_bytes(b"swapped-checkpoint-bytes")   # mutate after locking
    with pytest.raises(ModelLockError):
        M.run_inc1_metrics(tmp_path, _inputs(),
                           members={"member_1.pt": member},
                           pack_manifest=manifest, v2g_params=v2g,
                           code_commit="abc123", n_boot=8)
    assert not (tmp_path / M.DONE_NAME).exists()


def test_assert_not_done(tmp_path):
    M.assert_not_done(tmp_path)
    M.write_done(tmp_path, {"schema": "x"})
    with pytest.raises(M.EvalAlreadyDone):
        M.assert_not_done(tmp_path)


def test_sha256_file_roundtrip(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == sha256_file(p)
