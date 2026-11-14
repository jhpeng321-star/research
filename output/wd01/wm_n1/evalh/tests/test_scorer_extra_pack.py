"""Sealed extra CAL-tune feature pack tests (synthetic pack only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from evalh.scorer import extra_pack as XP


def _write_pack(tmp_path, *, sealed="CAL_TUNE", worlds=("w1",), arms=("arm0",),
                corrupt=False, fill=False):
    root = tmp_path / "pack"
    root.mkdir(parents=True, exist_ok=True)
    runs, items = [], []
    tick_off = dec_off = 0
    for world in worlds:
        for arm in arms:
            runs.append(dict(
                key=f"chunk2/{world}__{arm}", chunk="chunk2", batch="chunk2",
                world=world, arm=arm, split="CALIBRATION", t0_s=0.0,
                n_ticks=4, n_decision=1, tick_start=tick_off, tick_end=tick_off + 4,
                dec_start=dec_off, dec_end=dec_off + 1))
            items.append(dict(
                world=world, arm=arm, chunk="chunk2", batch="chunk2",
                split="CALIBRATION", t0_s=0.0, pair_key=f"{world}@t0=0",
                policy="trajectory", tick=tick_off, plan_row=dec_off,
                outcome_rows=[]))
            tick_off += 4
            dec_off += 1
    n = tick_off
    arrays = {
        "veh_kin": (n, 3, 7), "veh_mask": (n, 3), "veh_step": (n, 3, 16),
        "slot_index": (n, 3), "counter": (n, 2, 30),
        "cav_plan_ch": (n, 3, 2), "target_mask": (n, 3), "hdv_base": (n, 3, 4),
        "accel_next": (n, 3), "lane_next": (n, 3), "wall": (n, 4, 11),
        "wall_mask": (n, 4), "plan": (dec_off, 3, 19, 8),
    }
    files = {}
    for name, shape in arrays.items():
        if fill and name == "veh_kin":
            a = np.arange(int(np.prod(shape)), dtype=np.float16).reshape(shape)
        else:
            a = np.zeros(shape, dtype=np.float16)
        np.save(root / f"{name}.npy", a)
        files[f"{name}.npy"] = hashlib.sha256(
            (root / f"{name}.npy").read_bytes()).hexdigest()
    (root / "outcomes.parquet").write_bytes(b"")
    files["outcomes.parquet"] = hashlib.sha256(b"").hexdigest()
    (root / "manifest.json").write_text(json.dumps(
        dict(schema="wm_n1_pack/1", items=items)), encoding="utf-8")
    (root / "INDEX.json").write_text(json.dumps(
        dict(schema="wm_n1_corpus_pack/2", sealed=sealed, runs=runs)),
        encoding="utf-8")
    files["manifest.json"] = hashlib.sha256(
        (root / "manifest.json").read_bytes()).hexdigest()
    files["INDEX.json"] = hashlib.sha256(
        (root / "INDEX.json").read_bytes()).hexdigest()
    lines = []
    for rel, sha in sorted(files.items()):
        lines.append(f"{('0' * 64) if corrupt else sha}  {rel}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def _pin(root):
    return hashlib.sha256((root / "SHA256SUMS").read_bytes()).hexdigest()


def test_loads_selected_trajectory_items(tmp_path):
    root = _write_pack(tmp_path, worlds=("w1", "w2"), arms=("arm0", "arm1"))
    pack = XP.load_extra_pack(root, expected_sha256=_pin(root),
                              worlds=("w1",), arms=("arm0",))
    assert len(pack.selected) == 1
    assert pack.positions == {("w1", "arm0"): XP.POSITION_BASE}
    _pos, item, run = pack.selected[0]
    assert item["_extra_pack"] is True and run["tick_start"] == 0


def test_checked_file_slices_per_run(tmp_path):
    root = _write_pack(tmp_path, worlds=("w1", "w2"), arms=("arm0",))
    pack = XP.load_extra_pack(root, expected_sha256=_pin(root),
                              worlds=("w1", "w2"), arms=("arm0",))
    _pos, item, run = pack.selected[0]
    kin = np.load(pack.checked_file(item, "veh_kin.npy"))
    assert kin.shape == (4, 3, 7)
    plan = np.load(pack.checked_file(item, "plan.npy"))
    assert plan.shape == (1, 3, 19, 8)


def test_item_offsets_are_run_local_and_slice_is_global(tmp_path):
    """A run's item carries RUN-LOCAL build offsets; the slice uses the run's
    GLOBAL pack offset (the latent off-by-run bug that refused run>0)."""
    root = _write_pack(tmp_path, worlds=("w1", "w2"), arms=("arm0",), fill=True)
    pack = XP.load_extra_pack(root, expected_sha256=_pin(root),
                              worlds=("w1", "w2"), arms=("arm0",))
    _p0, i0, r0 = pack.selected[0]
    _p1, i1, r1 = pack.selected[1]
    # build-path offsets are local to each run's slice ...
    assert (i0["tick"], i0["plan_row"]) == (0, 0)
    assert (i1["tick"], i1["plan_row"]) == (0, 0)
    # ... while the slice base is the run's global pack offset.
    assert i1["_extra_tick_offset"] == 4 and i1["_extra_dec_offset"] == 1
    assert r1["tick_start"] == 0 and r1["tick_end"] == 4
    kin1 = np.load(pack.checked_file(i1, "veh_kin.npy"))
    assert kin1.shape == (4, 3, 7)
    assert kin1.reshape(-1)[0] == np.float16(4 * 3 * 7)


def test_source_routes_extra_and_released(tmp_path):
    root = _write_pack(tmp_path)

    class _Base:
        selected = ((0, {"world": "w1", "arm": "native", "batch": "chunk2",
                         "chunk": "chunk2"}, {}),)

        def _checked_file(self, item, name):
            return Path("released") / name

    pack = XP.load_extra_pack(root, expected_sha256=_pin(root),
                              worlds=("w1",), arms=("arm0",))
    src = XP.ExtraPackSource(_Base(), pack)
    assert len(src.selected) == 2
    assert src._checked_file(src.selected[0][1], "veh_kin.npy") == \
        Path("released") / "veh_kin.npy"
    assert src._checked_file(src.selected[1][1], "veh_kin.npy").name == \
        "veh_kin.npy"


def test_refuses_bad_pin_and_seal(tmp_path):
    root = _write_pack(tmp_path)
    with pytest.raises(XP.ExtraPackRefusal):
        XP.load_extra_pack(root, expected_sha256="0" * 64, worlds=("w1",),
                           arms=("arm0",))
    bad = _write_pack(tmp_path / "b2", sealed="TRAIN")
    with pytest.raises(XP.ExtraPackRefusal):
        XP.load_extra_pack(bad, expected_sha256=_pin(bad), worlds=("w1",),
                           arms=("arm0",))


def test_refuses_corrupt_listed_file(tmp_path):
    root = _write_pack(tmp_path, corrupt=True)
    with pytest.raises(XP.ExtraPackRefusal):
        XP.load_extra_pack(root, expected_sha256=_pin(root), worlds=("w1",),
                           arms=("arm0",))


def test_refuses_no_matching_item(tmp_path):
    root = _write_pack(tmp_path, worlds=("w9",), arms=("arm0",))
    with pytest.raises(XP.ExtraPackRefusal):
        XP.load_extra_pack(root, expected_sha256=_pin(root), worlds=("w1",),
                           arms=("arm0",))
