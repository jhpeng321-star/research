import json
from pathlib import Path

import pytest

from tools.wmfixmerge.data import ARRAYS, SEEDS, digest, safe, select, verify
from tools.wmfixmerge.runtime import RecordedDemand


def corpus(root):
    rows = [dict(seed=seed, world=str(seed), pair_key=str(seed) + "@t300", warmup_s=300,
                  arm=arm, dir=f"{seed}_{arm}", branch=dict(kind="B" if arm == "native" else "POLICY"),
                  demand_vph=2400, penetration=0.05)
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


def _tape(ep, seed):
    """Minimal ledger-free t300 tape accepted by recorded()/prepare()."""
    import numpy as np
    from tools.wmrequal.labels import departures
    T, K, t0 = 721, 16, 120
    ts = 240.0 + .5 * np.arange(T)
    vid = -np.ones((T, K), np.int64)
    mask = np.zeros((T, K), bool)
    kin = np.zeros((T, K, 7))
    counter = np.zeros((T, 2, 1), np.int64)
    vid[:, 0], mask[:, 0] = 0, True                    # persistent t0-HDV, id 0
    kin[:, 0, 1], kin[:, 0, 4], kin[:, 0, 0] = 1, 20., 10. + .02 * np.arange(T)
    counter[:, 0, 0] = 1
    nxt, life = 1, {}
    dep = [departures(seed, 2400, "mainline"), departures(seed, 400, "ramp")]
    due0 = [int(np.searchsorted(d, ts[t0], side="right")) for d in dep]
    ins = [1, 0]
    for k in range(1, 601):
        t = t0 + k
        for e in (0, 1):
            due_now = int(np.searchsorted(dep[e], ts[t], side="right"))
            if k % 8 == 0 and due_now > ins[e] and due0[e] + (due_now - ins[e]) > 0:
                slot = next(s for s in range(1, K) if not mask[t - 1, s] and vid[t, s] < 0)
                vid[t, slot], mask[t, slot] = nxt, True
                kin[t, slot, 2 if e else 1] = 1
                kin[t, slot, 4], kin[t, slot, 0] = 20., 10.
                life[nxt], ins[e] = (t, e), ins[e] + 1
                nxt += 1
        for identity, (born, e) in list(life.items()):
            if born < t:
                slot = int(np.argmax(vid[t - 1] == identity))
                if t - born <= 30:
                    vid[t, slot], mask[t, slot] = identity, True
                    kin[t, slot] = kin[t - 1, slot]
                else:
                    life.pop(identity)
        counter[t, 0, 0], counter[t, 1, 0] = ins
    for name, array in (("time_s", ts), ("vid", vid), ("veh_mask", mask), ("veh_kin", kin),
                        ("is_cav", np.zeros((T, K), bool)), ("counter", counter),
                        ("target_mask", np.zeros((T, K), bool)), ("control_time_s", ts),
                        ("controller", np.zeros((T, K), np.int64)),
                        ("issued_acceleration_mps2", np.zeros((T, K))),
                        ("issued_target_speed_mps", np.full((T, K), 20.))):
        np.save(ep / (name + ".npy"), array, allow_pickle=False)


def test_prepare_reconstructs_schedule_when_no_ledger_exists(tmp_path):
    import numpy as np
    from tools.wmfixmerge.prepare import prepare
    from tools.wmrequal.labels import departures
    rows = corpus(tmp_path)
    for row in rows:
        ep = tmp_path / row["dir"]
        ep.mkdir()
        _tape(ep, row["seed"])
    out = tmp_path / "prepared.json"
    doc = prepare(tmp_path, out)
    assert doc["ledger_checked"] is False
    assert verify(out) == doc
    label = next(r for r in doc["labels"] if r["native"])
    # Q0 came from the t300 counter/registry identity, not a request ledger.
    assert label["source_method"] == "counter_registry_conservation"
    assert len(label["due_increments"]) == 600
    # Window due totals are exactly the deterministic schedule in (300 s, 600 s].
    dep = [departures(label["seed"], 2400, "mainline"), departures(label["seed"], 400, "ramp")]
    totals = np.sum(label["due_increments"], 0)
    window = [int(np.searchsorted(d, 600.0, side="right")) -
              int(np.searchsorted(d, 300.0, side="right")) for d in dep]
    assert totals.tolist() == window
    # Paired members share the reconstructed request schedule exactly.
    groups = {}
    for r in doc["labels"]:
        groups.setdefault(r["pair_key"], []).append(r)
    assert all(len({tuple(map(tuple, x["due_increments"])) for x in g}) == 1
               for g in groups.values())
    # A stray ledger file in one episode must fail closed before any tape opens.
    stray = tmp_path / rows[0]["dir"] / "entry_backlog.npy"
    np.save(stray, np.zeros((721, 2), np.int64), allow_pickle=False)
    with pytest.raises(ValueError, match="corpus-uniform"):
        prepare(tmp_path, tmp_path / "prepared2.json")
