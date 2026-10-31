"""merge_corpus tests: registered union, FIT-mappo exclusion, role wiring."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.wmfixmerge import data, merge_corpus

CORE = ("time_s", "vid", "veh_mask", "veh_kin", "is_cav", "counter", "target_mask",
        "control_time_s", "controller", "issued_acceleration_mps2", "issued_target_speed_mps2")


def _episode(root, dir_name, seed, arm, kind, world=None):
    d = root / "episodes" / dir_name
    d.mkdir(parents=True)
    for name in CORE:
        (d / f"{name}.npy").touch()
    (d / "episode.json").touch()
    world = world or f"w{seed}m1900p05t300"
    return dict(arm=arm, arm_recorded_as="direct" if kind != "B" else "native",
                branch=dict(horizon_s=300, kind=kind, t0_s=300.0),
                controllers_used=["native"] if kind == "B" else ["native", "direct"],
                demand_vph=1900, dir=f"episodes/{dir_name}", episode=dir_name,
                horizon_s=300, k_slots=192, n_control=360, n_states=721,
                pair_key=f"{world}@t300", penetration=.05, policy_sha256="",
                seed=seed, split_role="fit", t0_index=600, t_start_s=0,
                warmup_s=300, world=world)


def _corpus(root, seeds, arms, split_role, world_seed=None):
    root.mkdir(parents=True)
    episodes = []
    for seed in seeds:
        world = f"w{world_seed or seed}m1900p05t300"
        for arm in arms:
            kind = "B" if arm == "native" else "POLICY"
            episodes.append(_episode(root, f"{world}__{arm}", seed, arm, kind, world))
    manifest = dict(schema="wms_c4_episode_source/1", k_slots=192, dt_s=.5,
                    split="TRAIN", arms={a: "x" for a in arms},
                    controller_vocab=["native", "damping", "pi-external-u", "direct"],
                    episodes=episodes, n_episodes=len(episodes))
    (root / "C4_EPISODES_MANIFEST.json").write_text(json.dumps(manifest))
    split = dict(manifest_role_map={split_role: "train"}, roles={split_role: "r"},
                 rule="t", worlds={str(s): split_role for s in seeds})
    (root / "SPLIT.json").write_text(json.dumps(split))
    return root


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    base = tmp_path_factory.mktemp("merge")
    fit = _corpus(base / "e3", data.SEEDS, ("native", "ours_c1", "coopoff_c1", "mappo"), "fit")
    wd = _corpus(base / "wm", data.WMDATA_SEEDS[:3], ("native", *data.WMDATA_ARMS),
                 "wmdata_fit")
    out = base / "corpus"
    receipt = merge_corpus.merge(fit, wd, out)
    return fit, wd, out, receipt


def test_counts_and_exclusion(merged):
    fit, wd, out, receipt = merged
    assert receipt["fit_episodes"] == 3 * len(data.SEEDS)
    assert receipt["wmdata_episodes"] == 5 * 3
    assert len(receipt["excluded_fit_diagnostic_episodes"]) == len(data.SEEDS)
    assert not (out / "fit" / "episodes" / f"w{data.SEEDS[0]}m1900p05t300__mappo").exists()


def test_select_union(merged):
    fit, wd, out, receipt = merged
    rows = data.select(out)
    assert {r["arm"] for r in rows if r["seed"] in data.SEEDS} == {"native", "ours_c1", "coopoff_c1"}
    assert {r["arm"] for r in rows if r["seed"] in data.WMDATA_SEEDS[:3]} == {"native", *data.WMDATA_ARMS}
    worlds = {r["world"] for r in rows}
    assert len(worlds) == len(data.SEEDS) + 3


def test_manifest_and_split(merged):
    fit, wd, out, receipt = merged
    manifest = json.loads((out / "C4_EPISODES_MANIFEST.json").read_text())
    assert manifest["allow_designed"] is True
    assert "mappo" in manifest["arms"]  # wmdata mappo kept, FIT rows excluded
    assert all(r["dir"].startswith(("fit/", "wmdata/")) for r in manifest["episodes"])
    split = json.loads((out / "SPLIT.json").read_text())
    assert split["worlds"][str(data.SEEDS[0])] == "fit"
    assert split["worlds"][str(data.WMDATA_SEEDS[0])] == "wmdata_fit"


def test_fail_closed(merged, tmp_path):
    fit, wd, out, receipt = merged
    bad = json.loads((fit / "C4_EPISODES_MANIFEST.json").read_text())
    bad["episodes"][0]["warmup_s"] = 240
    (tmp_path / "e3bad").mkdir()
    (tmp_path / "e3bad" / "C4_EPISODES_MANIFEST.json").write_text(json.dumps(bad))
    shutil_json = json.loads((fit / "SPLIT.json").read_text())
    (tmp_path / "e3bad" / "SPLIT.json").write_text(json.dumps(shutil_json))
    (tmp_path / "e3bad" / "episodes").mkdir()
    for row in bad["episodes"]:
        d = tmp_path / "e3bad" / row["dir"]
        d.mkdir(parents=True)
        for name in CORE:
            (d / f"{name}.npy").touch()
    with pytest.raises(ValueError):
        merge_corpus.merge(tmp_path / "e3bad", wd, tmp_path / "out2")
    with pytest.raises(ValueError):
        merge_corpus.merge(fit, wd, out)  # non-empty out refused
