"""WALLFIX: wall members that enter after t0 become teacher-forced CAV slots."""
import numpy as np
import torch

from wms.eval.backend import RolloutTools, WmsBackend
from wms.eval.predictor import RolloutFlags, WmsPredictor
from wms.eval.rollout_inputs import WallMemberInsert, wall_member_teacher


def _path(first_index=2, n=4):
    times = [0.5 * i for i in range(n + 1)]
    return {"id": "wall.1", "time_s": times,
            "s": [10.0 + i for i in range(n + 1)], "lane": [0] * (n + 1),
            "v": [5.0] * (n + 1),
            "present": [i >= first_index for i in range(n + 1)]}


def _state(extra=1):
    k = 1 + extra
    return {"s_m": torch.zeros(1, k), "v_mps": torch.zeros(1, k),
            "a_mps2": torch.zeros(1, k), "lane": torch.zeros(1, k, dtype=torch.long),
            "length_m": torch.full((1, k), 5.0), "present": torch.zeros(1, k, dtype=torch.bool),
            "is_cav": torch.zeros(1, k, dtype=torch.bool)}


def test_wall_member_teacher_starts_at_first_recorded_row():
    members = wall_member_teacher({"wall_member_paths": [_path()]}, 0.0, ["main.0"], 4)
    assert len(members) == 1
    member = members[0]
    assert member["id"] == "wall.1" and member["first_index"] == 2
    assert member["present"].tolist() == [False, False, True, True, True]
    assert not member["junction"].any()
    assert member["teacher"][2, 0] == 12.0
    # Already-present t0 members are left to the normal CAV teacher.
    assert wall_member_teacher({"wall_member_paths": [_path()]}, 0.0, ["wall.1"], 4) == []


def test_wall_member_insert_uses_reserved_slot_at_first_appearance():
    member = wall_member_teacher({"wall_member_paths": [_path()]}, 0.0, ["main.0"], 4)[0]
    process = WallMemberInsert([member], start=1)
    state = _state()
    state["present"][0, 0] = True
    mask = process.insert(state)
    assert not mask.any() and not state["present"][0, 1]
    mask = process.insert(state)
    assert mask[0, 1] and state["present"][0, 1] and state["is_cav"][0, 1]
    assert float(state["s_m"][0, 1]) == 12.0
    assert float(state["v_mps"][0, 1]) == 5.0
    assert process.summary()["inserted"] == 1


class _TeacherModel:
    """Minimal rollout that follows the real teacher/entry insertion semantics."""

    history_len = 10

    def hold_teacher(self, state):
        return state

    def rollout(self, initial, *, n_steps, cav_teacher=None, present_seq=None,
                junction_seq=None, entries=None, **kwargs):
        current = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in initial.items()}
        log = {"positions": [], "speeds": [], "lanes": [], "masks": []}
        for step in range(n_steps):
            if entries is not None:
                entries.insert(current)
            nxt_s = current["s_m"] + 0.5 * current["v_mps"]
            nxt_v = current["v_mps"].clone()
            nxt_l = current["lane"].clone()
            if cav_teacher is not None:
                teacher = cav_teacher[:, step + 1]
                cav = current["is_cav"].bool() & current["present"].bool()
                nxt_s = torch.where(cav, teacher[..., 0], nxt_s)
                nxt_l = torch.where(cav, teacher[..., 1].long(), nxt_l)
                nxt_v = torch.where(cav, teacher[..., 2], nxt_v)
            nxt_p = current["present"].clone()
            if present_seq is not None:
                nxt_p = nxt_p & present_seq[:, step + 1]
            current.update({"s_m": nxt_s, "v_mps": nxt_v, "lane": nxt_l, "present": nxt_p})
            log["positions"].append(nxt_s)
            log["speeds"].append(nxt_v)
            log["lanes"].append(nxt_l)
            log["masks"].append(nxt_p.clone())
        return {"positions": torch.stack(log["positions"], 1),
                "speeds": torch.stack(log["speeds"], 1),
                "lanes": torch.stack(log["lanes"], 1),
                "masks": torch.stack(log["masks"], 1),
                "state": {"is_cav": current["is_cav"]}}


def _tools():
    return RolloutTools(
        junction_carry_presence=lambda present, junction: (np.asarray(present),
                                                          np.zeros_like(present)),
        history_spec=None, history_step_features=None, long_history_features=None,
        action_timeline=None, controllers=("native",))


def _record(with_paths=True):
    def vehicle(i, s, v, lane=0, cav=False):
        return {"id": f"main.{i}", "s": s, "v": v, "lane": lane, "present": True,
                "is_cav": cav}

    frames = [{"time_s": 0.5 * k, "vehicles": [vehicle(0, 10.0 + 5.0 * k, 10.0),
                                              vehicle(1, 50.0 + 2.5 * k, 5.0)]}
              for k in range(5)]
    rec = {"world": "w", "key": "k", "t0_s": 0.0, "source_arm": "x",
           "initial_state": {"time_s": 0.0, "vehicles": [vehicle(0, 10.0, 10.0),
                                                         vehicle(1, 50.0, 5.0)]},
           "recorded": {"native": frames, "wall": frames}}
    if with_paths:
        rec["wall_member_paths"] = [_path()]
    return rec


def test_predictor_inserts_wall_member_only_with_switch_on():
    backend = WmsBackend(model=_TeacherModel(), config={}, tools=_tools())
    off = WmsPredictor(backend).predict_states(_record(), {"arm": "wall"}, 2.0, 0)
    assert all("wall.1" not in {v["id"] for v in frame["vehicles"]} for frame in off)

    on = WmsPredictor(backend, RolloutFlags(wall_members=True)).predict_states(
        _record(), {"arm": "wall"}, 2.0, 0)
    present = [{v["id"] for v in frame["vehicles"]} for frame in on]
    assert "wall.1" not in present[0] and "wall.1" not in present[1]
    assert "wall.1" in present[2] and "wall.1" in present[3]
    member = next(v for v in on[2]["vehicles"] if v["id"] == "wall.1")
    assert member["s"] == 12.0 and member["is_cav"]
