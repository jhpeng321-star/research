"""S-BACKLOG: paired-branch conversion exports exact entrance-waiting cost clocks.

The loop's real pairs (cycles 0-2) were converted without ``entry_backlog`` because the raw
branch tapes carry no ``scenario/requests.rou.xml`` beside them (only the BASE episode does),
so ``wms.control.long_credit`` refused to fit the CQR (LONGINT.md).  These tests pin the fix:
requests resolve through ``branch.base_episode``, the per-tick backlog is the RequestLedger's
interval waiting-time integral (fractional at mid-interval departures, never-seen requests
still waiting), and the exported clocks must reconcile to the branch summary's authoritative
J ledger with ``long_credit``'s exact tolerance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wms.core.errors import TrainDataRefusal  # noqa: E402
from wms.data import c4_episodes as C4  # noqa: E402
from wms.data.action_timeline import CONTROLLERS  # noqa: E402

K = 6
DIRECT = CONTROLLERS.index("direct")
T0 = 10.0          # world warmup (tape start = branch_t - prehistory)
BRANCH_T = 12.0
HORIZON = 4.0


def _tape_arrays():
    """13 ticks (t = 10.0 .. 16.0), 6 slots.

    Slots: 0 HDV mainline v0 (present all ticks, first seen at 3.0 in the base run);
    1 ramp vehicle v1, inserted at t = 13.0; 2 CAV mainline v2 (present all ticks);
    3-5 empty.  Request v3 (mainline, due 9.0) is NEVER inserted: still waiting."""
    t_n = int(6.0 / 0.5) + 1
    time_s = T0 + np.arange(t_n) * 0.5
    present = np.zeros((t_n, K), bool)
    vid = np.full((t_n, K), -1, np.int32)
    lane = np.full((t_n, K), -1, np.int8)
    s = np.zeros((t_n, K), np.float32)
    v = np.zeros((t_n, K), np.float32)
    a = np.zeros((t_n, K), np.float32)
    is_cav = np.zeros((t_n, K), bool)
    for t in range(t_n):
        present[t, 0], vid[t, 0], lane[t, 0], v[t, 0], s[t, 0] = True, 0, 1, 20.0, -500.0
        present[t, 2], vid[t, 2], lane[t, 2], v[t, 2], s[t, 2] = True, 2, 0, 22.0, -400.0
        is_cav[t, 2] = True
    for t in range(6, t_n):                       # v1 on the ramp from t = 13.0
        present[t, 1], vid[t, 1], lane[t, 1], v[t, 1], s[t, 1] = True, 1, 2, 15.0, -370.0
    c_n = t_n - 1
    controller = np.zeros((c_n, K), np.int8)
    acc = np.zeros((c_n, K), np.float32)
    tgt = np.zeros((c_n, K), np.float32)
    after = T0 + np.arange(c_n) * 0.5 >= BRANCH_T
    controller[after, 2] = DIRECT
    return dict(time_s=time_s, s_m=s, lane=lane, lane_carry=lane.copy(), v_mps=v, a_mps2=a,
                length_true_m=np.where(present, 5.0, 0).astype(np.float32), present=present,
                is_cav=is_cav, junction=np.zeros((t_n, K), bool), vid_index=vid,
                vehicle_ids=np.asarray([f"v{i}" for i in range(4)]),
                cav_vehicle_ids=np.asarray(["v2"]),
                entry_first_seen_s=np.asarray([3.0, 13.0, 2.0, np.nan]),
                entry_first_entrance=np.asarray([0, 1, 0, -1], np.int8),
                control_time_s=T0 + np.arange(c_n, dtype=np.float64) * 0.5, controller=controller,
                issued_acceleration_mps2=acc, issued_target_speed_mps=tgt,
                controller_vocab=np.asarray(CONTROLLERS))


def _requests_xml(base: Path):
    routes = ('<routes><vehicle id="v0" depart="2.0" route="mainline_route"/>'
              '<vehicle id="v1" depart="11.3" route="ramp_route"/>'
              '<vehicle id="v2" depart="1.0" route="mainline_route"/>'
              '<vehicle id="v3" depart="9.0" route="mainline_route"/></routes>')
    scenario = base / "scenario"
    scenario.mkdir(parents=True)
    (scenario / "requests.rou.xml").write_text(routes)


def _pair_tape(tmp_path: Path, *, kind="POLICY"):
    """A raw paired-branch tape (as ``traffic_shepherd.c4.genrand.write_branch_episode``
    writes it) plus its base episode; returns (tape dir, base dir)."""
    base = tmp_path / "collect" / "episodes" / "policy" / "s4107_m2700_p0200"
    _requests_xml(base)
    tape = tmp_path / "collect" / "branches" / "policy" / "s4107_m2700_p0200" / "t0012_policy"
    tape.mkdir(parents=True)
    meta = dict(schema="c4_sumo_episode/1", seed=4107, demand_vph=2700, penetration=0.2,
                warmup_s=BRANCH_T, horizon_s=HORIZON, controller="direct", k_slots=K,
                arm="loop14_pairs_b0012_policy", world_t0_s=T0,
                branch={"pair_key": "w4107m2700p20t10@t12", "base_world": "w4107m2700p20t10",
                        "base_episode": str(base), "branch_t_s": 12, "arm": kind, "kind": kind,
                        "horizon_s": 4, "prehistory_s": 2})
    arrays = _tape_arrays()
    np.savez_compressed(tape / "episode.npz", meta_json=np.asarray(json.dumps(meta)), **arrays)
    (tape / "manifest.json").write_text(json.dumps({**meta, "status": "complete"}))
    (tape / "summary.json").write_text(json.dumps(_ledger_for(arrays)))
    return tape, base


def _ledger_for(arrays: dict) -> dict:
    """The authoritative branch ledger, computed with the env/long_credit convention."""
    times = arrays["time_s"]
    tick = int(np.searchsorted(times, BRANCH_T - 1e-6))
    n = int(round(HORIZON / 0.5))
    kin = np.zeros((len(times), K, 7), np.float32)
    kin[..., 0] = arrays["s_m"]
    kin[..., 4] = arrays["v_mps"]
    mask = arrays["present"].astype(np.uint8)
    ins = {"v0": 3.0, "v1": 13.0, "v2": 2.0}
    dep = {"v0": 2.0, "v1": 11.3, "v2": 1.0, "v3": 9.0}
    ent = {"v0": 0, "v1": 1, "v2": 0, "v3": 0}
    queue = np.zeros((len(times), 2), np.float32)
    for t in range(len(times)):
        now = float(times[t])
        for name in dep:
            waiting = max(0.0, min(ins.get(name, np.inf), now + 0.5) - max(dep[name], now))
            queue[t, ent[name]] += waiting / 0.5
    road = .5 * (np.maximum(0., 1. - kin[tick + 1:tick + 1 + n, ..., 4] / 31.29)
                 * mask[tick + 1:tick + 1 + n]).sum()
    source = .5 * queue[tick:tick + n].sum()
    return {"schema": "c4_loop14_pair_summary/1", "seed": 4107, "warmup_s": BRANCH_T,
            "horizon_s": HORIZON, "requested_population": 4, "start_entry_backlog": [1, 0],
            "ledger": {"speed_deficit_vehicle_seconds": float(road),
                       "source_wait_vehicle_seconds": float(source),
                       "J_vehicle_seconds": float(road + source)}}


def test_exact_entry_backlog_matches_the_request_ledger_convention():
    time_s = np.asarray([0.0, 0.5, 1.0, 1.5])
    vid = np.full((4, 3), -1, np.int32)
    present = np.zeros((4, 3), bool)
    vid[2, 0], present[2, 0] = 0, True        # A inserted at t = 1.0
    arrays, stats = C4.exact_entry_backlog(
        time_s, vid, present, ["A", "B"], ["A", "B"], np.asarray([0.2, 0.0]),
        np.asarray([0, 1]))
    # interval [t, t+0.5): waiting = min(ins, t+0.5) - max(dep, t)
    np.testing.assert_allclose(arrays["entry_backlog"][:, 0], [0.6, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(arrays["entry_backlog"][:, 1], [1.0, 1.0, 1.0, 1.0])
    assert stats["still_waiting_at_end"] == 1
    assert C4.episode_queue_arrays.__name__      # V9ENTRY sibling unchanged below


def test_pair_tape_conversion_exports_reconciled_clocks(tmp_path):
    tape, base = _pair_tape(tmp_path)
    out = tmp_path / "src"
    entry = C4.convert_episode(tape, out)
    episode = out / entry["dir"]
    assert (episode / "entry_backlog.npy").is_file()
    assert (episode / "request_counter.npy").is_file()
    assert (episode / "entry_cav_counter.npy").is_file()
    clocks = entry["queue_arrays"]
    assert clocks["pair_cost_reconciliation"]["difference"] == pytest.approx(0.0, abs=1e-3)
    assert clocks["requests"] if "requests" in clocks else clocks["n_requests"] == 4
    assert clocks["still_waiting_at_end"] == 1
    backlog = np.load(episode / "entry_backlog.npy")
    assert backlog.shape == (13, 2)
    # v1 due 11.3, inserted 13.0: mainline-... ramp entrance waits [11.3, 13.0) = 1.7 s
    # + v3 (never inserted) due 9.0 on the mainline the whole tape.
    tick_11 = int(np.searchsorted(np.arange(13) * 0.5 + 10.0, 11.0))
    assert backlog[tick_11, 0] == 1.0                       # v3 waiting the full interval
    np.testing.assert_allclose(backlog[:, 1].max(), 1.0)
    assert entry["pair_key"] == "w4107m2700p20t10@t12"
    # base episode untouched: only its requests.rou.xml was read
    assert (base / "scenario" / "requests.rou.xml").is_file()


def test_tampered_ledger_refuses_conversion(tmp_path):
    tape, _ = _pair_tape(tmp_path)
    summary = json.loads((tape / "summary.json").read_text())
    summary["ledger"]["source_wait_vehicle_seconds"] += 2.5
    (tape / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(TrainDataRefusal, match="reconcile"):
        C4.convert_episode(tape, tmp_path / "src")


def test_missing_everywhere_leaves_v9entry_path_untouched(tmp_path):
    """No requests anywhere (and no summary): plain V9ENTRY behaviour, no queue arrays."""
    tape, base = _pair_tape(tmp_path)
    (base / "scenario" / "requests.rou.xml").unlink()
    (tape / "summary.json").unlink()
    entry = C4.convert_episode(tape, tmp_path / "src")
    assert "queue_arrays" not in entry
    assert not (tmp_path / "src" / entry["dir"] / "entry_backlog.npy").exists()


def test_foreign_summary_without_branch_keeps_v9entry(tmp_path):
    """Requests + a non-pair summary but no branch meta (e3-style tape): V9ENTRY
    integer-count clocks, never the reconciled pair path, never a refusal."""
    tape, _ = _pair_tape(tmp_path)
    arrays = _tape_arrays()
    _requests_xml(tape)
    meta = dict(schema="c4_sumo_episode/1", seed=4107, demand_vph=2700, penetration=0.2,
                warmup_s=T0, horizon_s=6.0, controller="direct", k_slots=K)
    np.savez_compressed(tape / "episode.npz", meta_json=np.asarray(json.dumps(meta)), **arrays)
    (tape / "manifest.json").write_text(json.dumps({**meta, "status": "complete"}))
    (tape / "summary.json").write_text(json.dumps({"schema": "c4_sumo_episode_summary/1"}))
    entry = C4.convert_episode(tape, tmp_path / "src")
    clocks = entry["queue_arrays"]
    assert "pair_cost_reconciliation" not in clocks
    # v3 (due 9.0, never on the tape) is the only pre-start unknown request
    assert clocks["queue_unknown"] == 1 and clocks["n_requests"] == 4


def test_cqr_rows_may_carry_explicit_roles_and_default_is_unchanged():
    """E3DATA pairs keep their registered split: an explicit row ``role`` overrides the hashed one."""
    from wms.control.long_credit import fit_cqr, world_role
    rng = np.random.default_rng(3)

    def make(worlds, role=None):
        out = []
        for w in worlds:
            for rep in range(2):
                base = float(rng.normal())
                row = {"world": w, "horizon_s": 120, "samples": (base + rng.normal(size=8) * .1).tolist(),
                       "real_benefit": base + float(rng.normal()) * .1 + rep}
                if role:
                    row["role"] = role
                out.append(row)
        return out

    plain = make([f"w{i}" for i in range(30)])
    assert fit_cqr(plain, {})["horizons"]["120"]["fit_worlds"] == sorted(
        {r["world"] for r in plain if world_role(r["world"]) == "fit"})
    explicit = (make([f"a{i}" for i in range(8)], "fit") + make([f"b{i}" for i in range(6)], "calibration")
                + make([f"c{i}" for i in range(4)], "validation"))
    model = fit_cqr(explicit, {})["horizons"]["120"]
    assert model["fit_worlds"] == sorted(f"a{i}" for i in range(8))
    assert model["calibration_worlds"] == sorted(f"b{i}" for i in range(6))
    assert model["validation"]["worlds"] == sorted(f"c{i}" for i in range(4))
