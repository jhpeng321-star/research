"""V9ENTRY evaluation mode: entries ON + the recorded SUMO entrance queue in WM checks.

* ``RolloutFlags.entry_queue_state`` (default off) needs entries v2;
* ``wms.eval.entry_queue``: queue vector of a record (stored, or derived from the regenerated
  request schedule: FIFO insertion up to the newest seen vehicle, demand over the full 10 s bins of
  the last 300 s, CAV share of the inserted requests);
* ``entries_input.entry_process_queue``: backlog seeded, sumo_like_v2 + fresh_no_credit, HDV
  entrants, uncapped pending;
* tool switches (``wm_eval_common.ENTRY_MODES``; default off everywhere).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from wms.eval import entry_queue as EQ
from wms.eval.entries_input import entry_process_queue
from wms.eval.predictor import RolloutFlags
from wms.model import entries as E

ROOT = Path(__file__).resolve().parents[2]


def _fake_requests(monkeypatch, main_dep, ramp_dep, main_cav=None, ramp_cav=None):
    def fake(seed, demand, pen):
        out = {}
        for ent, dep, cav in ((0, main_dep, main_cav), (1, ramp_dep, ramp_cav)):
            dep = np.asarray(dep, float)
            out[ent] = (np.arange(dep.size, dtype=np.int64), dep,
                        np.zeros(dep.size, bool) if cav is None else np.asarray(cav, bool))
        return out
    monkeypatch.setattr(EQ, "_requests", fake)


def _record(seen_main, seen_ramp, hist=(), world="w962005m2400p20t300"):
    veh = [{"id": f"mainline.{i}", "s": 0.0, "lane": 1, "v": 10.0, "present": True,
            "is_cav": False} for i in seen_main]
    veh += [{"id": f"ramp.{i}", "s": 0.0, "lane": 2, "v": 10.0, "present": True,
             "is_cav": False} for i in seen_ramp]
    return {"world": world, "initial_state": {"time_s": 300.0, "vehicles": veh},
            "history": {"ids": list(hist) + ["__history_context_slot9"]},
            "counter": np.ones((2, 30)).tolist()}


def test_flag_requires_entries_v2():
    with pytest.raises(ValueError):
        RolloutFlags(entries=True, entry_queue_state=True)
    assert RolloutFlags(entries=True, entries_v2=True, entry_queue_state=True).entry_queue_state
    assert RolloutFlags().entry_queue_state is False


def test_world_case_from_meta_and_name():
    assert EQ.world_case({"world": "w962000m1900p10t300"}) == (962000, 1900, 0.10)
    assert EQ.world_case({"world_meta": {"world_id": 5, "demand": 2000, "penetration_pct": 40}}) \
        == (5, 2000, 0.40)
    with pytest.raises(ValueError):
        EQ.world_case({"world": "native_1"})


def test_derived_backlog_demand_and_cav_share(monkeypatch):
    # mainline requests every 2 s from t = 1 (151 due by t0 = 300); vehicles 0..99 inserted
    main = 1.0 + 2.0 * np.arange(200)
    cav = np.arange(200) % 4 == 0
    _fake_requests(monkeypatch, main, [10.0, 290.0, 299.0, 301.0], main_cav=cav)
    rec = _record(seen_main=[97, 99], seen_ramp=[0], hist=["mainline.98", "ramp.1"])
    vec, info = EQ.derive_entry_queue(rec, 300.0)
    assert info["last_inserted_0"] == 99 and info["last_inserted_1"] == 1
    assert vec[0] == 150 - 100              # due 0..149 (dep <= 300), inserted 0..99
    assert vec[1] == 1                      # ramp.2 due at 299 s, not inserted; ramp.3 not due
    assert vec[2] == pytest.approx(150 / 300.0)     # requests with age < 300 s
    assert vec[3] == pytest.approx(3 / 300.0)
    assert vec[4] == pytest.approx(np.mean(cav[:100]))   # inserted and recent: 0..99
    assert EQ.state_entry_queue({**rec, "entry_queue": [1, 2, 3, 4]}, 300.0)[1] == "record"
    assert EQ.state_entry_queue(rec, 300.0)[1] == "derived_requests"


def test_derived_queue_guards(monkeypatch):
    _fake_requests(monkeypatch, [1.0, 2.0, 500.0], [3.0])
    with pytest.raises(ValueError):          # a vehicle on the road before its request time
        EQ.derive_entry_queue(_record([2], []), 300.0)
    with pytest.raises(ValueError):          # not a request of this world
        EQ.derive_entry_queue(_record([7], []), 300.0)
    with pytest.raises(ValueError):          # evaluation seed block
        EQ.derive_entry_queue(_record([0], [], world="w964010m2400p20t300"), 300.0)


def test_queue_entry_process_seeds_backlog_with_hdv_entrants(monkeypatch):
    rec = {**_record([0], []), "entry_queue": [7, 2, 0.6, 0.1, 0.5, 0.5]}
    kin = np.zeros((3, 7), np.float32)
    kin[:, 0] = [-200.0, -100.0, 50.0]
    kin[:, 2] = 1.0
    kin[:, 4] = 20.0
    kin[:, 6] = 5.0
    tools = types.SimpleNamespace(entries=E, counter_reader=None)
    proc, n_pad, info = entry_process_queue(rec, 300.0, 120.0, 0, kin, tools)
    assert proc.pending[0].tolist() == [7, 2]
    assert proc.queues[0].cav_share is None                 # entrants stay HDVs
    assert proc.cfg.insert_rule == "sumo_like_v2" and proc.cfg.fresh_no_credit
    assert proc.cfg.max_pending >= 10 ** 9
    assert n_pad == int(np.ceil((0.7 * 120.0 + 9.0) * 1.5)) + 16
    assert info["source"] == "record" and info["vector"][0] == 7


def test_tool_entry_modes_default_off():
    sys.path.insert(0, str(ROOT / "tools"))
    import wm_eval_common as wc
    assert wc.ENTRY_MODES["off"] == {}
    assert wc.ENTRY_MODES["queue"] == {"entries": True, "entries_v2": True,
                                       "entry_queue_state": True}
    for mode in wc.ENTRY_MODES.values():
        RolloutFlags(**mode)                                 # every mode is a valid flag set
    with pytest.raises(SystemExit):
        wc.load_predictor("x.pt", "AS_IS", entries="bogus")
