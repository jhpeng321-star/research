import json
from pathlib import Path

import pytest

from tools.wmfixmerge.data import ARRAYS, SEEDS, digest, safe, select, verify
from tools.wmfixmerge.runtime import RecordedDemand


def corpus(root):
    rows = [dict(seed=seed, world=str(seed), pair_key=str(seed) + "@t300", warmup_s=300,
                 arm=arm, dir=f"{seed}_{arm}", branch=dict(kind="B" if arm == "native" else "POLICY"))
            for seed in SEEDS for arm in ("native", "ours_c1", "coopoff_c1")]
    (root / "C4_EPISODES_MANIFEST.json").write_text(json.dumps(dict(episodes=rows)))
    (root / "SPLIT.json").write_text(json.dumps(dict(worlds={str(s): "fit" for s in SEEDS})))
    return rows


def test_only_registered_fit_before_any_tape_open(tmp_path):
    rows = corpus(tmp_path)
    assert select(tmp_path) == rows  # episode directories deliberately do not exist
    split = tmp_path / "SPLIT.json"
    split.write_text(json.dumps(dict(worlds={str(s): "validation" for s in SEEDS})))
    with pytest.raises(ValueError, match="not FIT"):
        select(tmp_path)


@pytest.mark.parametrize("mutation", ["seed", "arm", "world", "start", "path", "reference"])
def test_adversarial_manifest_rejected_before_tape_open(tmp_path, mutation):
    rows = corpus(tmp_path)
    if mutation == "seed":
        rows[0]["seed"] = 964001
    elif mutation == "arm":
        rows[1]["arm"] = "unknown"
    elif mutation == "world":
        rows[1]["world"] = "other"
    elif mutation == "start":
        rows[0]["warmup_s"] = 301
    elif mutation == "path":
        rows[0]["dir"] = "../escape"
    else:
        rows[0]["branch"]["kind"] = "POLICY"
    (tmp_path / "C4_EPISODES_MANIFEST.json").write_text(json.dumps(dict(episodes=rows)))
    with pytest.raises(ValueError):
        select(tmp_path)


def test_hash_binding_is_complete_and_tamper_evident(tmp_path):
    rows = corpus(tmp_path)
    paths = {}
    for row in rows:
        ep = tmp_path / row["dir"]
        ep.mkdir()
        for name in ARRAYS:
            p = ep / (name + ".npy")
            p.write_bytes(b"test hash contract only")
            paths[str(p)] = digest(p)
    doc = dict(schema="wmfix_train/1", root=str(tmp_path), seeds=list(SEEDS), episodes=rows,
               manifest_sha256=digest(tmp_path / "C4_EPISODES_MANIFEST.json"),
               split_sha256=digest(tmp_path / "SPLIT.json"), files=paths)
    p = tmp_path / "prepared.json"
    p.write_text(json.dumps(doc))
    assert verify(p) == doc
    item = next(iter(paths))
    Path(item).write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        verify(p)
    del doc["files"][item]
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="hash binding"):
        verify(p)


def test_restricted_paths_reject_before_io():
    for name in ("sealed/x", "x/heldout/a", "x/cal_split/a", "x/requests.rou.xml", "x/hetdata_truth/a"):
        with pytest.raises(ValueError):
            safe(name)


def test_unmatched_mappo_is_excluded_at_metadata_boundary(tmp_path):
    rows = corpus(tmp_path)
    # No episode directory exists; its unrepresented lane-head arm must never open.
    extra = dict(rows[0], arm="mappo", dir="never_open_mappo", branch=dict(kind="POLICY"))
    (tmp_path / "C4_EPISODES_MANIFEST.json").write_text(json.dumps(dict(episodes=rows + [extra])))
    assert select(tmp_path) == rows


def test_production_loader_filters_arms_before_tape_open_and_off_is_unchanged(monkeypatch, tmp_path):
    import numpy as np
    import wms.data.c4_pairs as pairs
    rows = [dict(world="w", pair_key="w@t300", warmup_s=300, split_role="train", episode=a,
                 arm=a, dir=a, branch=dict(kind="B" if a == "native" else "POLICY"))
            for a in ("native", "ours_c1", "mappo")]
    monkeypatch.setattr(pairs, "read_source_manifest", lambda root: dict(episodes=rows))
    seen = []
    class Episode:
        def array(self, name):
            return {"time_s": np.array([300., 300.5, 301.]), "vid": np.zeros((3, 1), int),
                    "veh_kin": np.zeros((3, 1, 7))}[name]
    def open_episode(path):
        seen.append(Path(path).name)
        return Episode()
    monkeypatch.setattr(pairs, "_open_episode", open_episode)
    monkeypatch.setattr(pairs, "episode_window_ticks", lambda *args, **kw: [0])
    on = pairs.PairedBranchSource(tmp_path, horizon_steps=2, arm_names=("native", "ours_c1"))
    assert "mappo" not in seen and on.receipt["n_windows"] == 2
    seen.clear()
    default = pairs.PairedBranchSource(tmp_path, horizon_steps=2)
    explicit_off = pairs.PairedBranchSource(tmp_path, horizon_steps=2, arm_names=None)
    assert "mappo" in seen and default.receipt == explicit_off.receipt
    assert default.groups == explicit_off.groups and default.receipt["n_windows"] == 3


def test_recorded_requests_are_not_service_and_stay_arm_coupled():
    schedule = [[[2, 1], [0, 3]], [[2, 1], [0, 3]]]
    due = RecordedDemand(schedule)
    assert due(0) == due(1) == [0, 0, 1]
    assert due(1) == due(0) == [1, 1, 1]
    with pytest.raises(ValueError, match="exhausted"):
        due(0)
    with pytest.raises(ValueError):
        RecordedDemand([[[.5, -1]]])
