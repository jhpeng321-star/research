"""S-SUMOCREDIT: confident-deployment gate + real-SUMO cooperation credit (synthetic CPU inputs only)."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wms.control import confident_gate as cg
from wms.control import real_credit14 as rc

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _loop():
    spec = importlib.util.spec_from_file_location("loop14_sumocredit", ROOT / "scripts/c4/loop14.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["loop14_sumocredit"] = mod
    spec.loader.exec_module(mod)
    return mod


def _state(s, lane, cav, present=None):
    s = torch.tensor([s], dtype=torch.float32)
    k = s.shape[1]
    return {"s_m": s, "v_mps": torch.full_like(s, 20.0), "a_mps2": torch.zeros_like(s),
            "lane": torch.tensor([lane]), "length_m": torch.full_like(s, 5.0),
            "present": torch.ones(1, k, dtype=torch.bool) if present is None else torch.tensor([present]),
            "is_cav": torch.tensor([cav])}


# ------------------------------------------------------------------ reliability table
def test_table_is_monotone_posterior_and_small_blocks_stay_closed():
    raw = [-30, -20, -10, -5, 0, 5, 10, 20, 30, 40]
    y = [0, 0, 1, 0, 0, 1, 1, 1, 1, 1]
    t = cg.ReliabilityTable.fit(raw, y, {"split": "TRAIN"})
    probs = [t.probability(x) for x in np.linspace(-40, 50, 50)]
    assert all(b >= a - 1e-12 for a, b in zip(probs, probs[1:]))          # monotone
    for w, n in zip(t.wins, t.counts):
        assert 0 <= w <= n
    assert t.probability(40) == pytest.approx((5 + 1) / (5 + 2))           # pooled top block 5/5
    assert cg.ReliabilityTable.posterior(1, 1) < cg.GATE_PROBABILITY          # 1/1 never opens
    assert cg.ReliabilityTable.posterior(3, 3) == pytest.approx(0.8)
    assert t.meta["n"] == 10 and 0 <= t.meta["brier_in_sample"] <= 1


def test_table_gap_maps_to_lower_block_and_json_roundtrip():
    t = cg.ReliabilityTable.fit([0, 0, 0, 10, 10, 10, 10], [0, 1, 0, 1, 1, 1, 1], {"split": "TRAIN"})
    assert len(t.upper) == 2
    assert t.probability(5) == t.probability(0)                           # between blocks -> lower
    assert t.probability(100) == t.probability(10)
    assert t.probability(float("nan")) == 0.0
    back = cg.ReliabilityTable.from_json(json.loads(json.dumps(t.to_json())))
    assert back.probability(10) == t.probability(10)
    with pytest.raises(ValueError):
        cg.ReliabilityTable.from_json({**t.to_json(), "meta": {"split": "VAL"}})


def test_calibration_rows_refuse_eval_seeds_and_non_train():
    rec = {"schema": cg.RECORD_SCHEMA, "split": "TRAIN", "seed": 968500, "unit": "fleet",
           "reference": "hold", "horizon_s": 20.0, "state_source": "belief",
           "units": [{"raw": 3.0, "sumo_J_plan": 10.0, "sumo_J_ref": 12.0, "valid": True},
                     {"raw": -1.0, "sumo_J_plan": 15.0, "sumo_J_ref": 12.0, "valid": True},
                     {"raw": 9.0, "sumo_J_plan": 1.0, "sumo_J_ref": 2.0, "valid": False}]}
    raw, y, _ = cg.calibration_rows([rec], unit="fleet", reference="hold", horizon_s=20, state_source="belief")
    assert raw == [3.0, -1.0] and y == [1, 0]
    assert cg.calibration_rows([rec], unit="group", reference="hold", horizon_s=20, state_source="belief")[0] == []
    with pytest.raises(ValueError):
        cg.calibration_rows([{**rec, "seed": 964010}], unit="fleet", reference="hold", horizon_s=20,
                            state_source="belief")
    with pytest.raises(ValueError):
        cg.calibration_rows([{**rec, "split": "VAL"}], unit="fleet", reference="hold", horizon_s=20,
                            state_source="belief")


# ------------------------------------------------------------------ gate logic on synthetic beliefs
def test_units_fleet_group_cav():
    st = _state([0, 100, 1000, 1100, 3000, 50], [1, 1, 1, 0, 1, 1], [True, True, True, True, True, False])
    assert cg.gate_units(st, "fleet") == [[4, 3, 2, 1, 0]]
    assert cg.gate_units(st, "group") == [[4], [3, 2], [1, 0]]
    assert cg.gate_units(st, "cav") == [[4], [3], [2], [1], [0]]
    st["lane"][0, 4] = -1                                                   # junction: not acting
    assert cg.gate_units(st, "cav")[0] == [3]


class StubImagination:
    reference, horizon_s = "hold", 20.0

    def __init__(self, J):
        self.J = torch.tensor(J, dtype=torch.float64)
        self.members = None

    def arms_J(self, state, members, **kwargs):
        self.members = members
        return self.J


def _table(unit="group"):
    raw = list(range(-5, 15))
    y = [int(r > 3) for r in raw]
    return cg.ReliabilityTable.fit(raw, y, {"split": "TRAIN", "unit": unit, "reference": "hold",
                                            "horizon_s": 20.0, "state_source": "belief"})


def test_gate_opens_only_calibrated_units_and_logs_rate():
    st = _state([0, 100, 1000, 1100, 3000], [1, 1, 1, 0, 1], [True] * 5)
    table = _table()
    # arms: REF, PLAN_[4], PLAN_[3,2], PLAN_[1,0]  (J costs; raw = J_REF - J_PLAN)
    imag = StubImagination([[[100.0], [90.0], [99.0], [120.0]]])
    gate = cg.ConfidentGate(imag, table, unit="group", state_source="belief")
    d = gate.decide(st, now_s=360.0)
    assert d.raw == [10.0, 1.0, -20.0]
    assert [p >= cg.GATE_PROBABILITY for p in d.probability] == d.open == [True, False, False]
    assert d.open_slots == [4]
    assert imag.members[0].sum() == 0 and imag.members[1].tolist() == [False, False, False, False, True]
    s = gate.summary()
    assert s["units"] == 3 and s["units_open"] == 1 and s["unit_open_rate"] == pytest.approx(1 / 3)
    assert s["threshold"] == 0.8


def test_gate_refuses_mismatched_calibration():
    with pytest.raises(ValueError):
        cg.ConfidentGate(StubImagination([[[0.0]]]), _table("fleet"), unit="group", state_source="belief")
    with pytest.raises(ValueError):
        cg.ConfidentGate(StubImagination([[[0.0]]]), _table("group"), unit="group", state_source="oracle")


def test_gate_with_no_cavs_is_closed():
    st = _state([0, 10], [1, 1], [False, False])
    gate = cg.ConfidentGate(StubImagination([[[0.0]]]), _table(), unit="group", state_source="belief")
    d = gate.decide(st, now_s=1.0)
    assert d.units == [] and gate.summary()["unit_open_rate"] is None


class FakeEnv:
    def __init__(self, ids):
        self.now = 400.0
        self.frame = SimpleNamespace(vid_index=np.arange(len(ids)))
        self.builder = SimpleNamespace(vehicle_ids=list(ids))


class FakeSource:
    name = "oracle"

    def __init__(self, st):
        self.st, self.observed = st, 0

    def inputs(self, env):
        return {"state": self.st}, {"n_present": 4}

    def observe(self, env):
        self.observed += 1


@pytest.mark.parametrize("reference", ["hold", "native"])
def test_gated_executor_applies_plan_only_to_open_cavs(reference):
    st = _state([0, 100, 1000, 1100], [1, 1, 1, 0], [True, True, False, True])
    table = cg.ReliabilityTable.fit([0, 10, 10, 10, 10], [0, 1, 1, 1, 1],
                                    {"split": "TRAIN", "unit": "cav", "reference": reference,
                                     "horizon_s": 20.0, "state_source": "oracle"})
    imag = StubImagination([[[50.0], [40.0], [55.0], [60.0]]])    # units [3], [1], [0]: open, closed, closed
    imag.reference = reference
    gate = cg.ConfidentGate(imag, table, unit="cav", state_source="oracle")
    env = FakeEnv(["c0", "c1", "h2", "c3"])
    plan = np.array([[1.0, 1.0, np.nan, 1.0]])
    lanes = np.array([[2, 0, -1, 1]])

    def plan_fn(env, state):
        return plan, lanes

    def hold_fn(state):
        return np.array([[-0.5, -0.5, np.nan, -0.5]])
    gp = cg.GatedSumoPolicy(plan_fn, gate, FakeSource(st), decision_period_s=5.0,
                            hold_fn=hold_fn if reference == "hold" else None)
    accel, out_lanes = gp(env, st)
    assert gp.open_ids == {"c3"}
    assert accel[0, 3] == 1.0 and out_lanes[0, 3] == 1
    closed = [0, 1]
    if reference == "hold":
        assert accel[0, closed].tolist() == [-0.5, -0.5]
    else:
        assert np.isnan(accel[0, closed]).all()
    assert out_lanes[0, closed].tolist() == [-1, -1] and np.isnan(accel[0, 2])
    assert plan[0, 0] == 1.0                                       # the plan output is not mutated
    env.now = 402.0
    gp(env, st)                                                    # inside the decision period
    assert gate.stats["decisions"] == 1
    assert gp.summary()["cav_tick_open_rate"] == pytest.approx(1 / 3)


def test_arm_imagination_paired_and_reproducible():
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel

    torch.manual_seed(3)
    model = WmSModel(ModelSpec(rollout_mode="hybrid")).eval().requires_grad_(False)
    policy, encoder = CavPolicy(lane_head=False).eval(), RsuGroupEncoder().eval()
    st = _state([-300, -250, -200, -150, -100, -60], [1, 1, 0, 1, 2, 1],
                [True, False, False, True, False, True])
    hist = torch.zeros(1, 6, model.history_len, model.history_features)
    kw = dict(reference="hold", horizon_s=2.0)
    ref = torch.zeros(6, dtype=torch.bool)
    one = ref.clone()
    one[3] = True
    J1 = cg.ArmImagination([model], policy, encoder, **kw).arms_J(st, [ref, ref, one, None], history=hist)
    J2 = cg.ArmImagination([model], policy, encoder, **kw).arms_J(st, [ref, ref, one, None], history=hist)
    assert J1.shape == (1, 4, 1) and J1.dtype == torch.float64
    assert torch.equal(J1, J2)                                     # reproducible
    assert torch.equal(J1[:, 0], J1[:, 1])                         # identical arms: identical J (CRN)
    assert torch.isfinite(J1).all() and (J1 >= 0).all()


# ------------------------------------------------------------------ credit math
def test_credit_terms_signs_and_cooperation():
    J = {"B": 100.0, "ALL": 70.0, "ONLY:a": 95.0, "ONLY:b": 98.0, "GROUP": 85.0, "FLIP:a": 96.0}
    t = rc.credit_terms(J, ["a", "b"], ["a"])
    assert t["D_all"] == 30.0 and t["D_i"] == {"a": 5.0, "b": 2.0} and t["D_G"] == 15.0
    assert t["sum_D_i"] == 7.0 and t["C_G"] == 8.0                 # D positive = better
    assert t["M_i"] == {"a": 15.0 - 4.0}
    assert rc.credit_terms({"B": 1.0, "GROUP": 1.0, "ONLY:a": 1.0}, ["a", "b"])["C_G"] is None
    with pytest.raises(ValueError):
        rc.credit_terms({"ALL": 1.0}, [])


def _points(slope, n=8, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(n):
        imag = {"B": 100.0, "ALL": 100 - 3.0 * i, "ONLY:a": 100 - i, "ONLY:b": 100 - 0.5 * i,
                "GROUP": 100 - 2.5 * i}
        d = {k: 100 - v for k, v in imag.items()}
        real = {k: 200 - slope * d[k] - (noise * rng.normal() if k != "B" else 0.0) for k in imag}
        real["B"] = 200.0
        pts.append({"schema": rc.POINT_SCHEMA, "split": "TRAIN", "seed": 968500 + i, "group": ["a", "b"],
                    "flips": [], "sumo": real, "imagined": imag, "extra_sim_s": 90})
    return pts


def test_slope_multiplier_recovers_scale_and_never_flips_or_amplifies():
    pairs = rc.pairs_by_type(_points(0.5))
    for t in ("all", "individual", "group", "coop"):
        fit = rc.slope_multiplier(pairs[t])
        assert fit["estimated"] and fit["slope"] == pytest.approx(0.5)
        assert fit["multiplier"] == pytest.approx(0.5)             # SE 0: no shrinkage
    assert rc.slope_multiplier(rc.pairs_by_type(_points(-1.0))["group"])["multiplier"] == 0.0
    assert rc.slope_multiplier(rc.pairs_by_type(_points(3.0))["group"])["multiplier"] == 1.0
    noisy = rc.slope_multiplier(rc.pairs_by_type(_points(0.5, noise=40.0))["group"])
    assert 0.0 <= noisy["multiplier"] <= 1.0 and noisy["shrinkage"] < 1.0
    assert rc.slope_multiplier([(1.0, 2.0, 0)]) == {**rc.slope_multiplier([(1.0, 2.0, 0)]), "multiplier": 1.0,
                                                    "estimated": False}


def test_variance_report_and_fit_files(tmp_path, monkeypatch):
    monkeypatch.setenv("C4_ALLOWED_ROOTS", str(tmp_path))
    pts = _points(0.5, noise=5.0)
    rep = rc.variance_report(pts, cycle_budget_sim_s=29_776)
    assert rep["points"] == 8 and rep["extra_sim_s_per_point"] == 90
    row = rep["terms"]["D_G"]
    assert row["n"] == 8 and row["n_for_mean"] == math.ceil(4 * row["sd"] ** 2 / row["mean"] ** 2)
    path = tmp_path / "p.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in pts), encoding="utf-8")
    doc = rc.fit_credit([path])
    assert set(doc["multipliers"]) == set(rc.CORRECTED) and doc["points"] == 8
    out = tmp_path / "REAL_CREDIT.json"
    out.write_text(json.dumps(doc), encoding="utf-8")
    m = rc.load_multipliers(out)
    assert all(0 <= v <= 1 for v in m.values()) and m["team"] == 1.0   # no flips: unchanged
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({**pts[0], "seed": 964001}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        rc.read_points([bad])


# ------------------------------------------------------------------ flag-off identity
def test_trainer_multipliers_off_passes_weights_through(monkeypatch):
    from wms.control import dreamer, policy14

    seen = []

    def fake(self, imagined, scale, weights, *, gradient_components=None):
        seen.append(weights)
        return torch.tensor(0.0), {}
    monkeypatch.setattr(dreamer.DreamerTrainer, "actor_objective", fake)
    trainer = policy14.Policy14Trainer.__new__(policy14.Policy14Trainer)
    weights = {"individual": 0.5, "group": 1.0, "coop": 0.25, "team": 1.0, "all": 1.0}
    trainer.actor_objective({}, 1.0, weights)
    assert seen[-1] is weights                                        # untouched object (bit-identical)
    trainer.credit_multipliers = {"individual": 0.2, "group": 1.0, "coop": 0.0, "team": 0.5}
    trainer.actor_objective({}, 1.0, weights)
    assert seen[-1] == {"individual": 0.1, "group": 1.0, "coop": 0.0, "team": 0.5, "all": 1.0}
    assert weights["individual"] == 0.5


def test_loop_collect_unchanged_without_sumocredit_and_charge_matches_extras():
    from traffic_shepherd.c4.sumocredit import PairExtras

    loop = _loop()
    cfg = json.loads((ROOT / "configs/c4_loop14_gpur.json").read_text(encoding="utf-8"))
    assert loop.collect_sim_s(cfg) == 29_776                        # the registered loop budget
    assert loop.sumocredit_args(cfg, {}) == []
    for x in ({"gate_horizon": 20}, {"gate_horizon": 20, "gate_reference": "native"},
              {"gate_horizon": 20, "gate_unit": "group", "gate_max_units": 3},
              {"gate_horizon": 400, "gate_unit": "fleet"}, {"credit_horizon": 60, "credit_flips": 1},
              {"credit_horizon": 400}, {"gate_horizon": 30, "gated_arm": True, "credit_horizon": 120}):
        c = {**cfg["collect"], "sumocredit": x}
        extras = PairExtras(**{k: v for k, v in x.items() if k != "gated_arm"},
                            gate_table="T.json" if x.get("gated_arm") else None)
        assert loop.sumocredit_sim_s_per_point(c) == extras.sim_s_per_point(int(c["branch_horizon"])), x
    c = {**cfg["collect"], "sumocredit": {"credit_horizon": 60, "credit_imagine": False}}
    args = loop.sumocredit_args({"collect": c}, {})
    assert args[:2] == ["--credit-horizon", "60"] and args[-1] == "--no-credit-imagine"


def test_collect_cli_flags_off_adds_no_options(tmp_path):
    from traffic_shepherd.c4 import collect

    reg = tmp_path / "REG.json"
    reg.write_text(json.dumps({"status": "registered", "split": "TRAIN", "seed_block": [968600, 968610]}))
    base = ["--policy", str(tmp_path / "p.pt"), "--policy-pairs", "--seeds", "968600", "--demands", "2400",
            "--penetrations", "0.2", "--branch-cadence", "60", "--registration", str(reg),
            "--out", str(tmp_path / "out"), "--dry-run"]
    seen = {}
    real_plan = collect.plan_jobs

    def spy(args):
        jobs = real_plan(args)
        seen["opts"] = jobs[0]["opts"]
        return jobs
    collect.plan_jobs = spy
    try:
        assert collect.main(base) == 0
        assert "sumocredit" not in seen["opts"]
        assert collect.main([*base, "--gate-horizon", "20", "--credit-horizon", "60"]) == 0
        assert seen["opts"]["sumocredit"]["gate_horizon"] == 20
        assert seen["opts"]["sumocredit"]["credit_horizon"] == 60
    finally:
        collect.plan_jobs = real_plan
    with pytest.raises(SystemExit):
        collect.main([a for a in base if a != "--policy-pairs"] + ["--gate-horizon", "20"])
