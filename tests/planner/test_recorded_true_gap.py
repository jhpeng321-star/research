"""TRUE-GAP TTC opt-in mode of recorded._safety_from_ticks (gap #18).

PL-EXP1b amendment: SUMO lane_position_m values are FRONT-bumper
positions, so the historic gap (leader_pos - follower_pos) includes the
leader's length and biases TTC HIGH. The opt-in true-gap mode subtracts
the leader's declared length (vehicle length per type_id from the run's
scenario requests.rou.xml vType definitions; a vType without ``length``
gets the SUMO passenger default 5.0 m; an unresolvable type_id fails
closed). Default behaviour (no lengths given) is UNCHANGED.

Hand-built acceptance numbers (leader pos 50 m, length 5 m, follower pos
30 m, closing speed 5 m/s): true gap 15 m -> TTC 3.0 s; old
front-difference gap 20 m -> TTC 4.0 s. Zero SUMO; tmp_path only.
"""

import json
from pathlib import Path

import pytest

from traffic_shepherd.planner.recorded import (
    SUMO_DEFAULT_VEHICLE_LENGTH_M,
    Layout,
    RunPaths,
    TrueGapLengthError,
    _safety_from_rows,
    branch_outcome,
    true_gap_lengths_from_rou,
    vtype_lengths_from_rou,
)

ROAD = "1191554636"
LANE = "1191554636_0"


def veh(pos, speed, accel=0.0):
    return {
        "road_id": ROAD,
        "lane_id": LANE,
        "lane_position_m": pos,
        "speed_mps": speed,
        "acceleration_mps2": accel,
    }


def closing_rows():
    """One tick: leader 'l' at 50 m (10 m/s), follower 'f' at 30 m
    (15 m/s) on the same lane -> closing speed 5 m/s."""
    return [
        {
            "time_s": 100.0,
            "colliding_ids": [],
            "arrived_ids": [],
            "cav_actions": {},
            "cav_ids": [],
            "vehicles": {"l": veh(50.0, 10.0), "f": veh(30.0, 15.0)},
        }
    ]


def rou_xml(vtypes, vehicles):
    return (
        "<routes>"
        + "".join(
            f'<vType id="{tid}" carFollowModel="IDM"'
            + (f' length="{length:g}"' if length is not None else "")
            + "/>"
            for tid, length in vtypes
        )
        + "".join(
            f'<vehicle id="{vid}"'
            + (f' type="{tid}"' if tid is not None else "")
            + ' route="r" depart="1.0"/>'
            for vid, tid in vehicles
        )
        + '<route id="r" edges="e0 e1"/>'
        + "</routes>"
    )


def write_rou(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "scenario" / "requests.rou.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


LEADER_LEN_M = 5.0  # the hand-built leader's declared vType length


class TestTrueGapTtc:
    def test_true_gap_ttc_3s_vs_old_gap_4s(self, tmp_path):
        """The acceptance numbers: with the TRUE bumper-to-bumper gap the
        hand-built pair has TTC 3.0 s; the old front-difference gap gives
        4.0 s. The vType length comes from the fixture's requests.rou.xml."""
        rou = write_rou(
            tmp_path,
            rou_xml([("leader_car", LEADER_LEN_M)], [("l", "leader_car"), ("f", "leader_car")]),
        )
        lengths = true_gap_lengths_from_rou([rou])
        assert lengths == {"l": LEADER_LEN_M, "f": LEADER_LEN_M}
        layout = Layout()
        rows = closing_rows()
        true_gap = _safety_from_rows(rows, layout, true_gap_lengths=lengths)
        old_gap = _safety_from_rows(rows, layout)
        assert true_gap["min_ttc_s"] == pytest.approx(3.0)  # (50 - 5 - 30) / 5
        assert old_gap["min_ttc_s"] == pytest.approx(4.0)  # (50 - 30) / 5
        assert true_gap["has_closing_pair"] is old_gap["has_closing_pair"] is True

    def test_recorded_defaults_unchanged_without_lengths(self):
        """No lengths -> the historic front-difference TTC, byte-identical
        behaviour (the PL-02a contract)."""
        safety = _safety_from_rows(closing_rows(), Layout())
        assert safety["min_ttc_s"] == pytest.approx(4.0)
        # T-A: the PRIMARY 1.5 s exposure is computed alongside the frozen
        # 3 s one; TTC 4.0 s is below neither threshold
        assert safety["ttc_lt1p5_exposure_s"] == 0.0
        assert set(safety) == {
            "min_ttc_s",
            "has_closing_pair",
            "ttc_lt3_exposure_s",
            "ttc_lt1p5_exposure_s",
            "hard_brake_s",
            "collisions",
        }

    def test_vtype_length_attribute_and_passenger_default(self, tmp_path):
        """A vType WITH ``length`` reports it; a vType WITHOUT it gets the
        named SUMO passenger default 5.0 m; a vehicle without a ``type``
        attribute is the default passenger type."""
        assert SUMO_DEFAULT_VEHICLE_LENGTH_M == 5.0
        rou = write_rou(
            tmp_path,
            rou_xml(
                [("long_truck", 12.5), ("normal_car", None)],
                [
                    ("t1", "long_truck"),
                    ("c1", "normal_car"),
                    ("c2", "normal_car"),
                    ("anon", None),  # no type attribute -> default type
                ],
            ),
        )
        assert vtype_lengths_from_rou([rou]) == {
            "long_truck": 12.5,
            "normal_car": SUMO_DEFAULT_VEHICLE_LENGTH_M,
        }
        lengths = true_gap_lengths_from_rou([rou])
        assert lengths == {
            "t1": 12.5,
            "c1": SUMO_DEFAULT_VEHICLE_LENGTH_M,
            "c2": SUMO_DEFAULT_VEHICLE_LENGTH_M,
            "anon": SUMO_DEFAULT_VEHICLE_LENGTH_M,
        }

    def test_missing_type_id_fails_closed(self, tmp_path):
        rou = write_rou(
            tmp_path, rou_xml([("normal_car", None)], [("ghost", "undeclared_car")])
        )
        with pytest.raises(TrueGapLengthError, match="undeclared_car"):
            true_gap_lengths_from_rou([rou])

    def test_leader_without_declared_length_fails_closed_at_metric_time(self, tmp_path):
        """A leader id absent from the route files has no declared length:
        the true-gap metric REFUSES instead of imputing."""
        write_rou(tmp_path, rou_xml([("normal_car", None)], [("f", "normal_car")]))
        lengths = true_gap_lengths_from_rou(
            [tmp_path / "scenario" / "requests.rou.xml"]
        )
        with pytest.raises(TrueGapLengthError, match="no declared length"):
            _safety_from_rows(closing_rows(), Layout(), true_gap_lengths=lengths)

    def test_branch_outcome_true_gap_passthrough(self, tmp_path):
        """branch_outcome(..., true_gap_lengths=...) computes the true-gap
        min TTC; the default call keeps the old number."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "physics_steps.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in closing_rows()), encoding="utf-8"
        )
        rou = write_rou(
            tmp_path,
            rou_xml([("leader_car", LEADER_LEN_M)], [("l", "leader_car"), ("f", "leader_car")]),
        )
        run = RunPaths(
            directory=run_dir,
            arm="true_gap_fixture",
            is_base=False,
            receipt={"exit_status": "complete"},
            physics_paths=(run_dir / "physics_steps.jsonl",),
        )
        lengths = true_gap_lengths_from_rou([rou])
        outcome = branch_outcome(
            run, t0_s=99.5, window_s=0.5, layout=Layout(), departs={},
            true_gap_lengths=lengths,
        )
        assert outcome.min_ttc_s == pytest.approx(3.0)
        default_outcome = branch_outcome(
            run, t0_s=99.5, window_s=0.5, layout=Layout(), departs={}
        )
        assert default_outcome.min_ttc_s == pytest.approx(4.0)

    def test_exposure_threshold_selectable_for_dual_reporting(self, tmp_path):
        """The analyzer reports TTC<1.5 AND TTC<3.0 exposures by calling the
        safety pass once per threshold (Layout.ttc_threshold_s): a TTC 2.0
        pair is exposed at 3.0 s but not at 1.5 s."""
        rou = write_rou(
            tmp_path,
            rou_xml([("car", LEADER_LEN_M)], [("l", "car"), ("f", "car")]),
        )
        lengths = true_gap_lengths_from_rou([rou])
        # gap true = (60 - 5) - 40 = 15, closing 7.5 -> TTC 2.0 s
        rows = [
            {
                "time_s": 100.0,
                "colliding_ids": [],
                "arrived_ids": [],
                "cav_actions": {},
                "cav_ids": [],
                "vehicles": {"l": veh(60.0, 10.0), "f": veh(40.0, 17.5)},
            }
        ]
        assert _safety_from_rows(
            rows, Layout(ttc_threshold_s=3.0), true_gap_lengths=lengths
        )["ttc_lt3_exposure_s"] == pytest.approx(0.5)
        assert _safety_from_rows(
            rows, Layout(ttc_threshold_s=1.5), true_gap_lengths=lengths
        )["ttc_lt3_exposure_s"] == pytest.approx(0.0)
