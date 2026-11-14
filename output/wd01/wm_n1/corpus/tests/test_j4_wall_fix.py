"""J4 member_left_network terminal-tick fix (FEAT1 blocker 1).

Unit level: a synthetic wall episode whose member leaves before the
safety_abort tick must featurize -- the terminal tick tolerates members
that already left the network, every active tick still fails loudly on a
non-CAV member.  Real-data level: the two probed TRAIN J4 wall runs
(w8661m2600p10t240__w-mid-early, w8660m2700p10t300__w-lo-late,
batch walls_pilot_chunk4, split TRAIN re-decided before any file is
opened) featurize through the real featurize-all driver to v1 + Idea A
channels, verify, and carry the amendment-2 relabel flag + counts.
"""
import json

import numpy as np
import pytest

from corpus import featurize_all as FA
from corpus import idea_a as CIA
from corpus import _util as U
from corpus.j4_wall_fix import install, installed

RUNS = ("w8661m2600p10t240__w-mid-early", "w8660m2700p10t300__w-lo-late")


def _write_wall_events(run_dir, *, members=("mainline.185", "mainline.186"),
                       request_s=300.0, assembled_s=314.0, abort_s=441.0):
    ev = run_dir / "run" / "wall_events.jsonl"
    ev.parent.mkdir(parents=True, exist_ok=True)
    wid = "w-test:wall1"
    rows = [
        {"kind": "wall_request", "wall_id": wid, "timestamp_s": request_s,
         "members": list(members), "wall_speed_mps": 4.0, "hold_s": 20.0},
        {"kind": "wall_event", "wall_id": wid, "event": "assembly_start",
         "timestamp_s": request_s, "members": list(members)},
        {"kind": "wall_event", "wall_id": wid, "event": "assembled",
         "timestamp_s": assembled_s, "members": list(members)},
        {"kind": "wall_event", "wall_id": wid, "event": "safety_abort",
         "timestamp_s": abort_s, "reason": "member_left_network"},
        {"kind": "wall_handback", "wall_id": wid, "timestamp_s": abort_s,
         "members": list(members)},
    ]
    ev.write_text("".join(json.dumps(r) + "\n" for r in rows),
                  encoding="utf-8")
    return wid


@pytest.fixture()
def patched():
    assert install() or installed()
    yield
    # the patch is process-global and benign for the rest of the suite


def test_terminal_tick_tolerates_departed_member(tmp_path, patched):
    from wall_tokens_v04 import WallTokenError

    run_dir = tmp_path / "w-test__arm"
    wid = _write_wall_events(run_dir)
    from features.actions_adapter import WallFeatureSource

    walls = WallFeatureSource.from_run_dir(run_dir)
    positions = {"mainline.185": (100.0, 1)}      # only this member remains
    # active holding tick: a roster-missing member is still a loud error
    with pytest.raises(WallTokenError):
        walls.walls_at(440.0, positions=positions,
                       cav_roster={"mainline.185"})
    # terminal tick: the departed member no longer crashes the query
    states = walls.walls_at(441.0, positions=positions,
                            cav_roster={"mainline.185"})
    assert len(states) == 1
    st = states[0]
    assert st["phase"] == "aborted" and st["wall_id"] == wid
    assert st["n_members"] == 2
    assert st["mean_s_down_m"] == pytest.approx(100.0)
    assert st["lane_span"] == 0.0
    # full roster at the terminal tick: unchanged behaviour
    states = walls.walls_at(441.0,
                            positions={"mainline.185": (100.0, 1),
                                       "mainline.186": (120.0, 0)},
                            cav_roster={"mainline.185", "mainline.186"})
    assert states[0]["mean_s_down_m"] == pytest.approx(110.0)
    assert states[0]["lane_span"] == 1.0


def test_active_wall_non_cav_member_still_loud(tmp_path, patched):
    from wall_tokens_v04 import WallTokenError

    run_dir = tmp_path / "w-test__arm"
    _write_wall_events(run_dir, request_s=100.0, assembled_s=110.0,
                       abort_s=200.0)
    from features.actions_adapter import WallFeatureSource

    walls = WallFeatureSource.from_run_dir(run_dir)
    # an HDV (never a CAV) among the members on an ACTIVE tick: loud error
    with pytest.raises(WallTokenError):
        walls.walls_at(120.0, positions={"mainline.185": (10.0, 0)},
                       cav_roster=set())
    # no roster given: unchanged (no membership check)
    assert walls.walls_at(120.0)[0]["phase"] == "holding"


def test_no_terminal_tick_no_change(tmp_path, patched):
    from features.actions_adapter import WallFeatureSource

    run_dir = tmp_path / "w-test__arm"
    _write_wall_events(run_dir)                   # abort at 441
    walls = WallFeatureSource.from_run_dir(run_dir)
    positions = {"mainline.185": (100.0, 1), "mainline.186": (120.0, 0)}
    states = walls.walls_at(320.0, positions=positions,
                            cav_roster={"mainline.185", "mainline.186"})
    assert states[0]["phase"] == "holding"
    assert states[0]["n_members"] == 2


# --------------------------------------- synthetic physics-evidence cases ----
def _write_physics(run_dir, rows):
    """rows: list of (t, vehicles_keys, cav_ids)."""
    ph = run_dir / "run" / "physics_steps.jsonl"
    ph.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for t, vehs, cav in rows:
        out.append(json.dumps(dict(
            time_s=t,
            vehicles={v: {"lane_id": "1191554636_0",
                          "lane_position_m": 10.0,
                          "speed_mps": 5.0,
                          "acceleration_mps2": 0.0,
                          "road_id": "1191554636"} for v in vehs},
            cav_ids=list(cav))) + "\n")
    ph.write_text("".join(out), encoding="utf-8")


def _member_left_network_rows(*, until=350.0, step=0.5):
    """mainline.185 always present; mainline.186 leaves after ``until``."""
    rows = []
    t = 0.0
    while t <= 460.0 + 1e-9:
        vehs = ["mainline.185"]
        if t <= until + 1e-9:
            vehs.append("mainline.186")
        rows.append((round(t, 3), vehs, list(vehs)))
        t += step
    return rows


def test_road_end_member_tolerated_at_active_tick(tmp_path, patched):
    """Member leaves the network before the recorded abort: the visible
    window keeps its event timeline; the departed member degrades geometry
    like the existing absent-at-t path (amendment-2 road-end case)."""
    from features.actions_adapter import WallFeatureSource

    run_dir = tmp_path / "w-test__arm"
    _write_wall_events(run_dir)          # request 300, assembled 314, abort 441
    _write_physics(run_dir, _member_left_network_rows(until=350.0))
    walls = WallFeatureSource.from_run_dir(run_dir)
    # first active tick after the member left the network: tolerated
    states = walls.walls_at(350.5, positions={"mainline.185": (100.0, 1)},
                            cav_roster={"mainline.185"})
    assert len(states) == 1
    st = states[0]
    assert st["phase"] == "holding" and st["wall_id"] == "w-test:wall1"
    assert st["n_members"] == 2
    assert st["mean_s_down_m"] == pytest.approx(100.0)
    # the terminal tick still shows the aborted phase (unchanged behaviour)
    states = walls.walls_at(441.0, positions={"mainline.185": (100.0, 1)},
                            cav_roster={"mainline.185"})
    assert states[0]["phase"] == "aborted"
    assert states[0]["n_members"] == 2


def test_report_gap_member_carried_at_active_tick(tmp_path, patched):
    """Member present in the network but missing from the roster (slot-pool
    drop / junction report gap): its last reported member state is carried."""
    from features.actions_adapter import WallFeatureSource

    run_dir = tmp_path / "w-test__arm"
    _write_wall_events(run_dir)
    # mainline.186 stays in `vehicles` the whole time and IS a designated
    # CAV (reported at t=0) but is missing from later reports / the slot
    # pool: its last reported member state is carried
    rows = []
    t = 0.0
    while t <= 460.0 + 1e-9:
        rows.append((round(t, 3), ["mainline.185", "mainline.186"],
                     ["mainline.185", "mainline.186"] if t == 0.0 else
                     ["mainline.185"]))
        t += 0.5
    _write_physics(run_dir, rows)
    walls = WallFeatureSource.from_run_dir(run_dir)
    states = walls.walls_at(350.5, positions={"mainline.185": (100.0, 1)},
                            cav_roster={"mainline.185"})
    assert states[0]["phase"] == "holding"
    assert states[0]["n_members"] == 2
    assert states[0]["mean_s_down_m"] == pytest.approx(100.0)


def test_genuine_non_cav_member_still_loud_with_physics(tmp_path, patched):
    """Member present in the physics row and never CAV-reported anywhere:
    a genuine non-CAV member id is a data bug and still fails loudly."""
    from wall_tokens_v04 import WallTokenError

    from features.actions_adapter import WallFeatureSource

    run_dir = tmp_path / "w-test__arm"
    _write_wall_events(run_dir, members=("mainline.185", "hdv.9"))
    rows = []
    t = 0.0
    while t <= 460.0 + 1e-9:
        rows.append((round(t, 3), ["mainline.185", "hdv.9"],
                     ["mainline.185"]))
        t += 0.5
    _write_physics(run_dir, rows)
    walls = WallFeatureSource.from_run_dir(run_dir)
    with pytest.raises(WallTokenError):
        walls.walls_at(350.5, positions={"mainline.185": (100.0, 1),
                                         "hdv.9": (90.0, 0)},
                       cav_roster={"mainline.185"})


def test_missing_member_without_physics_stays_loud(tmp_path, patched):
    """No physics evidence at all: only the terminal tick tolerates."""
    from wall_tokens_v04 import WallTokenError

    from features.actions_adapter import WallFeatureSource

    run_dir = tmp_path / "w-test__arm"
    _write_wall_events(run_dir)
    walls = WallFeatureSource.from_run_dir(run_dir)
    with pytest.raises(WallTokenError):
        walls.walls_at(350.5, positions={"mainline.185": (100.0, 1)},
                       cav_roster={"mainline.185"})


# ------------------------------------------------- probed real J4 runs ----
def _data_root():
    from features import splits as S

    return S.RD


@pytest.mark.parametrize("run_name", RUNS)
def test_probed_j4_runs_featurize_and_verify(run_name, tmp_path):
    rd = _data_root()
    inv_parquet = U.WM_N1_ROOT / "inventory" / "inventory.parquet"
    if not (rd / "runs/campaign_walls_pilot_chunk4" / run_name /
            "run" / "physics_steps.jsonl").is_file():
        pytest.skip("raw J4 runs not reachable")
    import pandas as pd

    df = pd.read_parquet(inv_parquet)
    row = df[df["run"] == run_name].iloc[0]
    job = FA._resolve_run(U.sanitize(row.to_dict()), rd, tmp_path,
                          force=False)
    # split re-decided BEFORE any run file was opened
    assert job["split"] == "TRAIN"
    assert job["batch"] == "walls_pilot_chunk4"
    # blocker 3: the pre-restore counter history base now resolves to the
    # stage-1 chunk-4 runs root
    assert job["base_history_run_dir"] == \
        str(rd / "runs/campaign_stage1_chunk4" / f"{row['world']}__base")

    rec = FA._run_one(job)
    assert rec["status"] == "written"
    assert rec["key"] == (f"walls_pilot_chunk4/{job['world']}__"
                          f"{job['arm']}")
    from features import writer as W

    # shards key by the BATCH first component + exact run-dir arm suffix
    shard_dir = W.shard_dir(tmp_path, job["batch"], job["world"], job["arm"])
    idea = U.read_json(shard_dir / CIA.IDEA_A_DONE_NAME)
    # amendment-2 relabel flag + aborted-token count recorded
    assert idea["amendment2_road_end_relabel"] is True
    assert idea["road_end_relabelled_n"] >= 1
    # the abort tick is now in the wall tensor with the aborted phase
    wall = np.load(shard_dir / "wall.npy", mmap_mode="r")
    wmask = np.load(shard_dir / "wall_mask.npy", mmap_mode="r")
    hit = np.argwhere((wmask > 0) & (wall[..., 4] > 0))
    assert len(hit) == idea["road_end_relabelled_n"]
    for name in CIA.IDEA_A_ARRAYS:
        assert (shard_dir / f"{name}.npy").is_file()
        assert name in idea["arrays"]
    # blocker 3: the counter channel now carries pre-restore history
    # (departures within the 300 s window before t0, from the stage-1
    # chunk-4 base run)
    counter = np.load(shard_dir / "counter.npy", mmap_mode="r")
    assert int(np.asarray(counter[0]).sum()) > 0
    # re-run: skip-by-sha on both stages
    rec2 = FA._run_one(job)
    assert rec2["status"] == "skipped"


def test_probed_j4_pair_verify_and_pack(tmp_path):
    rd = _data_root()
    if not all((rd / "runs/campaign_walls_pilot_chunk4" / r / "run" /
                "physics_steps.jsonl").is_file() for r in RUNS):
        pytest.skip("raw J4 runs not reachable")
    from corpus import pack as PK
    from corpus import verify as VF
    from features import cli as FCLI

    out = tmp_path
    rep = FA.featurize_all(_j4_inventory(tmp_path), rd, out, workers=1)
    assert rep["written"] == 2 and rep["skipped"] == 0
    assert FCLI.main(["verify", "--data-root", str(out)]) == 0
    iver = CIA.verify_idea_a(out)
    assert iver == dict(ok=True, n_runs=2, errors=[]), iver
    man = PK.pack_shards(out, tmp_path / "pack", workers=1)
    assert set(PK.TRAIN_ARRAYS) == set(man["arrays"])
    vrep = VF.verify_pack(tmp_path / "pack")
    assert vrep["ok"] is True and vrep["n_arrays"] == len(PK.TRAIN_ARRAYS)


def _j4_inventory(tmp_path):
    import pandas as pd

    df = pd.read_parquet(U.WM_N1_ROOT / "inventory" / "inventory.parquet")
    keep = df[df["run"].isin(RUNS)]
    assert len(keep) == 2
    p = tmp_path / "inv_j4.parquet"
    keep.to_parquet(p, index=False)
    return p
