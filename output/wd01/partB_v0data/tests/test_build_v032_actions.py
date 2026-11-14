"""build_v032 --with-actions tests (brief item 5).

* default OFF: every existing output stays byte-identical (linked parent
  files) / array-identical (rebuilt sidecars; npz embeds a build
  timestamp) to a second OFF build, and no action sidecar or MANIFEST
  action key appears;
* ON: v0.4 action sidecars <world>__<source>_feat_act.npz appear with
  per-decision-time action ids + unknown masks aligned to cav_mask, the
  MANIFEST records them, SHA256SUMS verifies them, and fleet arms are
  policy-verified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import build_v032 as G  # noqa: E402
from conftest_load import make_mini_v02_root  # noqa: E402

ACTIONS = ("native", "damping", "pi", "wall", "unknown")


def _inject_action_evidence(runs: Path):
    """Every synthetic arm gets switch_schedule + command rows (+ a fleet
    manifest on the base arms) so the --with-actions stage has evidence."""
    for arm in sorted(runs.iterdir()):
        if not arm.is_dir():
            continue
        run = arm / "run"
        run.mkdir(exist_ok=True)
        is_base = arm.name.endswith("__base")
        controller = "damping" if is_base else None
        schedule, commands = [], []
        for vid in ("a", "c"):
            schedule.append(dict(vehicle_id=vid, time_s=240.0,
                                 action="native" if vid == "a" else
                                 "damping"))
            commands.append(dict(kind="command", timestamp_s=240.0,
                                 vehicle_id=vid,
                                 owner="native" if vid == "a" else
                                 "damping",
                                 issued_target_speed_mps=20.0,
                                 fallback_reason=None))
        sched = dict(schema="wm44_branch_types/1", arm="switching", seed=1,
                     t0_s=240.0, horizon_s=840.0, interval_lo_s=20.0,
                     interval_hi_s=120.0, events=schedule, intervals_s=[])
        (run / "switch_schedule.json").write_text(json.dumps(
            dict(run=arm.name, arm="switching", branch_seed=1,
                 schedule=sched, schedule_sha256="0" * 64)),
            encoding="utf-8")
        with open(run / "suffix_commands.jsonl", "w",
                  encoding="utf-8") as fh:
            for r in commands:
                fh.write(json.dumps(r) + "\n")
        if controller is not None:
            (run / "manifest.json").write_text(json.dumps(
                dict(schema="traffic_shepherd_evaluation/1",
                     controller=controller)), encoding="utf-8")


def _worlds_tree(out_root: Path) -> dict:
    return {p.relative_to(out_root).as_posix(): p.read_bytes()
            for p in sorted((out_root / "worlds").rglob("*"))
            if p.is_file()}


def _sidecar_arrays(out_root: Path, man: dict) -> dict:
    out = {}
    for key, e in man["worlds"].items():
        for rel in e["files"]:
            if rel.endswith(".npz"):
                with np.load(out_root / rel, allow_pickle=False) as z:
                    out[rel] = {k: np.asarray(z[k]).tobytes()
                                for k in z.files if k != "built_utc"}
    return out


def _linked_bytes(out_root: Path, man: dict) -> dict:
    out = {}
    for key, e in man["worlds"].items():
        for rel in e["files"]:
            if rel.endswith(".parquet"):
                out[rel] = (out_root / rel).read_bytes()
    return out


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    root = tmp_path_factory.mktemp("mini_act")
    mini = make_mini_v02_root(root / "mini")
    _inject_action_evidence(mini["runs"])
    return mini


def test_with_actions_off_is_identical_and_on_adds_sidecars(
        mini, tmp_path):
    v03_root = mini["root"]
    out_off1 = tmp_path / "off1"
    out_off2 = tmp_path / "off2"
    out_on = tmp_path / "on"
    for out in (out_off1, out_off2, out_on):
        rc = G.main(["--v03-root", str(v03_root), "--out", str(out),
                     "--parent-sha", "", "--no-rebuild", "--workers", "1"]
                    + (["--with-actions"] if out is out_on else []))
        assert rc == 0

    man1 = json.loads((out_off1 / "MANIFEST.json").read_text("utf-8"))
    man2 = json.loads((out_off2 / "MANIFEST.json").read_text("utf-8"))
    manon = json.loads((out_on / "MANIFEST.json").read_text("utf-8"))
    # OFF: no action keys anywhere, outputs reproducible
    assert "with_actions" not in man1["build"]
    assert "action_tokens_v04" not in man1["feature_sets"]
    assert "with_actions" not in man2["build"]
    assert man1["build"]["n_sidecars_detector"] > 0
    assert _linked_bytes(out_off1, man1) == _linked_bytes(out_off2, man2)
    a1, a2 = _sidecar_arrays(out_off1, man1), _sidecar_arrays(out_off2,
                                                              man2)
    assert set(a1) == set(a2)
    for rel in a1:
        assert a1[rel] == a2[rel], rel
    assert not list(out_off1.rglob("*_feat_act.npz"))
    # ON: identical data + action sidecars
    assert manon["build"]["with_actions"] is True
    assert manon["build"]["n_sidecars_actions"] == \
        manon["build"]["n_sidecars_detector"]      # one act sidecar/source
    assert "action_tokens_v04" in manon["feature_sets"]
    assert _linked_bytes(out_on, manon) == _linked_bytes(out_off1, man1)
    aon = _sidecar_arrays(out_on, manon)
    for rel in a1:
        assert aon[rel] == a1[rel], rel
    n_act = len(list(out_on.rglob("*_feat_act.npz")))
    assert n_act == manon["build"]["n_sidecars_actions"] > 0
    # SHA256SUMS covers the action sidecars
    import hashlib
    lines = (out_on / "SHA256SUMS").read_text("utf-8").splitlines()
    for line in lines:
        sha, rel = line.split("  ", 1)
        assert hashlib.sha256((out_on / rel).read_bytes()).hexdigest() == sha
    act_rels = [rel for _sha, rel in (line.split("  ", 1) for line in lines)
                if rel.endswith("_feat_act.npz")]
    assert len(act_rels) == n_act


def test_action_sidecar_values_fleet_and_switching(mini, tmp_path):
    """The fleet (base) arm carries uniform damping with unknown=0; the
    switching (native) arm is unknown before the schedule's first event
    and per-CAV afterwards."""
    out = tmp_path / "on2"
    rc = G.main(["--v03-root", str(mini["root"]), "--out", str(out),
                 "--parent-sha", "", "--no-rebuild", "--workers", "1",
                 "--with-actions"])
    assert rc == 0
    man = json.loads((out / "MANIFEST.json").read_text("utf-8"))
    checked = {"fleet": 0, "switch": 0}
    for key, e in man["worlds"].items():
        rel = next(r for r in e["files"] if r.endswith("_feat_act.npz"))
        with np.load(out / rel, allow_pickle=False) as z:
            ids, unk = z["action_ids"], z["action_unknown"]
            ts = z["t"]
            cav_mask = z["cav_mask"]
            ev = json.loads(str(z["action_evidence"]))
        assert ev["owner_counts"]
        rows = cav_mask > 0
        assert (ids[~rows] == 0).all() and (unk[~rows] == 1.0).all()
        if e["source"] == "base":          # fleet arm: uniform, verified
            assert ev["fleet_policy"] == "damping"
            assert ev["fleet_check"]["ok"] is True
            assert (ids[rows] == ACTIONS.index("damping")).all()
            assert (unk[rows] == 0.0).all()
            checked["fleet"] += 1
        else:               # switching arm: per-CAV evidence from t0=240
            assert ev["fleet_policy"] is None
            before = ts < 240.0 - 1e-9
            assert (unk[rows & before[:, None]] == 1.0).all()
            assert (unk[rows & ~before[:, None]] == 0.0).all()
            got = ids[rows & ~before[:, None]]
            assert np.isin(got, [ACTIONS.index("native"),
                                 ACTIONS.index("damping")]).all()
            checked["switch"] += 1
    assert checked["fleet"] > 0 and checked["switch"] > 0


def test_action_stage_fails_loud_without_evidence(mini, tmp_path):
    """An arm with NO action evidence must stop the build loudly (never an
    all-unknown silent sidecar)."""
    runs = mini["runs"]
    arm = runs / "wsynp40am2400p40t240__native"   # a built (TRAIN) arm
    keep = []
    run = arm / "run"
    for name in ("switch_schedule.json", "suffix_commands.jsonl",
                 "manifest.json"):
        p = run / name
        if p.exists():
            keep.append((p, p.read_bytes()))
            p.unlink()
    try:
        out = tmp_path / "on3"
        with pytest.raises(SystemExit, match="no action evidence"):
            G.main(["--v03-root", str(mini["root"]), "--out", str(out),
                    "--parent-sha", "", "--no-rebuild", "--workers", "1",
                    "--with-actions"])
    finally:
        for p, b in keep:
            p.write_bytes(b)
