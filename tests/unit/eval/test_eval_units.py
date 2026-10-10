"""Synthetic unit tests of the clean evaluation package (no checkpoint, no real data)."""
import json
import math

import numpy as np
import pytest
import torch

from wms.eval import compare as cmp
from wms.eval import fingerprints as fp
from wms.eval import models, ranking, report, subsets
from wms.eval.backend import RolloutTools, WmsBackend
from wms.eval.errors import ComparisonError
from wms.eval.normalize import iter_windows
from wms.eval.predictor import RolloutFlags, WmsPredictor
from wms.eval.score import ScoreSpec, evaluate_state, score_states


def veh(i, s, v, lane=0, cav=False):
    return {"id": f"v{i}", "s": s, "v": v, "lane": lane, "present": True, "is_cav": cav}


def stream(vehicles_at, n=241):
    return [{"time_s": 0.5 * k, "vehicles": vehicles_at(k)} for k in range(n)]


def truth(k):
    return [veh(1, 10.0 + 5.0 * k, 10.0), veh(2, 50.0 + 2.5 * k, 5.0), veh(3, 0.0, 10.0, cav=True)]


def record(world="w1", key="k", t0=0.0, initial=None):
    return {"world": world, "t0_s": t0, "key": key, "source_arm": "x", "paired": True,
            "initial_state": {"time_s": 0.0, "vehicles": initial or truth(0)},
            "recorded": {"native": stream(truth), "wall": stream(truth)}}


def shifted_predictor(bias):
    def predict(state, controllers, horizon_s, seed):
        n = int(horizon_s / 0.5) + 1
        return [{"time_s": 0.5 * k, "vehicles": [
            veh(1, 10.0 + 5.0 * k + bias, 10.0 + bias / 10), veh(2, 50.0 + 2.5 * k, 5.0),
            veh(3, 0.0, 10.0, cav=True)]} for k in range(n)]
    return predict


SPEC = ScoreSpec(horizons_s=(60.0, 120.0), predict_horizon_s=120.0)


def results(bias, worlds=6):
    return [evaluate_state(record(world=f"w{i}", key=f"k{i}"), shifted_predictor(bias), SPEC)
            for i in range(worlds)]


def test_evaluate_state_reports_bias_and_arms():
    result = evaluate_state(record(), shifted_predictor(4.0), SPEC)
    assert sorted(result["arms"]) == ["native", "wall"]
    # vehicle 1 off by 4 m, vehicle 2 exact -> median 2 m
    assert result["arms"]["wall"]["car"]["60.0"]["hdv"]["ds_med"] == pytest.approx(2.0)
    assert result["arms"]["wall"]["delay"]["predicted"]["60.0"]


def test_failing_arm_does_not_abort_others():
    def predict(state, controllers, horizon_s, seed):
        if controllers["arm"] == "wall":
            raise RuntimeError("boom")
        return shifted_predictor(0.0)(state, controllers, horizon_s, seed)
    result = evaluate_state(record(), predict, SPEC)
    assert result["arms"]["wall"]["error"].startswith("RuntimeError")
    assert result["arms"]["native"]["error"] is None


def test_score_states_preserves_order():
    recs = [record(world=f"w{i}", key=f"k{i}") for i in range(3)]
    out = score_states(recs, lambda: shifted_predictor(0.0), SPEC)
    assert [r["world"] for r in out] == ["w0", "w1", "w2"]


def test_summary_blowup_and_paired_bootstrap():
    good, bad = results(1.0), results(400.0)
    summary = report.horizon_summary(good, "wall", 60.0)
    assert summary["n"] == 6 and summary["median"] == pytest.approx(0.5)
    assert report.blowup_rate(bad, "wall")["rate"] == 1.0
    assert report.blowup_rate(good, "wall")["rate"] == 0.0
    paired = report.paired_bootstrap(good, bad, "wall", 60.0, n_boot=50)
    assert paired["point"] < 0 and paired["hi"] < 0 and paired["n_worlds"] == 6


def test_tested_arm_and_seen_normalisation():
    assert report.tested_arm("wall_same", ["native", "wall"]) == "wall"
    assert report.tested_arm("seen_controls", ["arm1"]) == "arm1"
    norm = report.normalize_seen([{"arms": {"arm1": {"error": None}}}])
    assert list(norm[0]["arms"]) == ["native"] and norm[0]["source_arm"] == "arm1"


def test_rank_pair_and_spearman():
    assert ranking.rank_pair(-1.0, -3.0)["regret_s"] == 0.0
    chosen = ranking.rank_pair(-1.0, 2.0)
    assert chosen["selected_arm"] == "wall" and chosen["regret_s"] == pytest.approx(2.0)
    tie = ranking.rank_pair(0.0, -2.0)
    assert tie["selected_arm"] == "native" and tie["regret_s"] == pytest.approx(2.0)
    assert ranking.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert ranking.spearman([1, 1, 1], [1, 2, 3]) is None
    with pytest.raises(Exception):
        ranking.rank_pair(math.nan, 1.0)


def test_ranking_on_paired_states():
    def arm(pred, rec):
        delay = lambda w: {"60.0": {"merge": w}, "120.0": {"merge": w}}  # noqa: E731
        return {"error": None, "delay": {"predicted": delay(pred), "recorded": delay(rec)}}
    states = [{"world": f"w{i}", "paired": True,
               "arms": {"wall": arm(p, r), "native": arm(0.0, 0.0)}}
              for i, (p, r) in enumerate([(-1.0, -2.0), (1.0, 3.0), (-1.0, 4.0)])]
    out = ranking.ranking(states, n_boot=20)["60.0"]
    assert out["n"] == 3 and out["sign_agreement"] == pytest.approx(2 / 3)
    assert out["always_native_regret_s"] == pytest.approx(2.0 / 3)


def seen_only(states):
    return [{**r, "arms": {"native": r["arms"]["native"]}} for r in states]


def test_compare_two_models_only_and_verdict(tmp_path):
    new = cmp.ModelResult("new", {"wall_same": results(1.0), "wall_pilot": results(1.0),
                                  "seen_controls": seen_only(results(1.0))})
    old = cmp.ModelResult("old", {"wall_same": results(30.0), "wall_pilot": results(30.0),
                                  "seen_controls": seen_only(results(30.0))})
    comparison = cmp.compare(new, old, n_boot=30)
    assert comparison["verdict"] == "improved"
    assert cmp.compare(old, new, n_boot=30)["verdict"] == "regressed"
    md, js = cmp.write_comparison(comparison, tmp_path / "out")
    assert md.read_text(encoding="utf-8").startswith("# Comparison")
    assert json.loads(js.read_text(encoding="utf-8"))["verdict"] == "improved"
    with pytest.raises(ComparisonError):
        cmp.write_comparison(comparison, tmp_path / "out")
    with pytest.raises(ComparisonError):
        cmp.compare(new, cmp.ModelResult("new", new.states_by_set))


def test_fingerprint_components():
    rec = record()
    pred = shifted_predictor(0.0)(rec, {}, 60.0, 0)
    for frame in pred[60:]:                      # vehicle 1 ends in the wrong lane
        frame["vehicles"][0]["lane"] = 1
    split = fp.horizon_split(pred, rec["recorded"]["native"], 0.0, 30.0)
    assert split["n_wrong_lane"] == 1
    assert fp.lane_change_counts(pred)["changes"] == 1
    bias = fp.speed_bias(shifted_predictor(10.0)(rec, {}, 60.0, 0),
                         rec["recorded"]["native"], 0.0, 60.0)
    assert bias["at_horizon"] == pytest.approx(0.5)      # (+1, 0) over two HDVs
    assert fp.state_slice(rec, jam_speed_mps=5.0) == "free_flow"
    assert fp.state_slice(record(initial=[veh(1, 0, 2.0), veh(2, 9, 3.0)])) == "jam"


def test_residual_stats_masks_cavs():
    step = {"present": np.array([True, True]), "is_cav": np.array([False, True]),
            "gate": np.array([0.5]), "physics": np.array([1.0, 9.0]),
            "correction": np.array([-0.5, 9.0]), "accel": np.array([0.5, 9.0])}
    out = fp.residual_stats([step])
    assert out["n_hdv_steps"] == 1 and out["mean_abs_correction"] == 0.5


def test_best_vs_last_and_stage_table():
    series = {"s10": results(10.0), "s30": results(1.0), "s60": results(5.0)}
    out = fp.best_vs_last(series, "wall", 60.0)
    assert out["best"] == "s30" and out["last"] == "s60" and not out["last_is_best"]
    assert out["last_minus_best_m"] > 0
    assert fp.stage_table(series, "wall")["s30"]["blowup"] == 0.0


def test_keep_speed_and_subsets(tmp_path):
    rec = record()
    frames = models.keep_speed(rec, {}, 2.0, 0)
    assert frames[-1]["vehicles"][0]["s"] == pytest.approx(10.0 + 10.0 * 2.0)
    tf = models.teacher_force_cavs(frames, rec, "native")
    assert [v["id"] for v in tf[1]["vehicles"]][-1] == "v3"
    recs = [record(world="wb", key="b", t0=1.0), record(world="wa", key="z", t0=0.0),
            record(world="wa", key="a", t0=2.0)]
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"windows": recs}), encoding="utf-8")
    assert len(list(iter_windows(path))) == 3
    assert [r["key"] for r in subsets.first_states(path, 2)] == ["z", "a"]
    assert [r["key"] for r in subsets.first_per_world(path, 2)] == ["z", "b"]


class StubModel:
    history_len = 10

    def hold_teacher(self, state):
        return state

    def rollout(self, initial, *, n_steps, **kwargs):
        n = initial["s_m"].shape[1]
        speeds = initial["v_mps"][:, None, :].repeat(1, n_steps, 1)
        ticks = torch.arange(1, n_steps + 1)[None, :, None]
        pos = initial["s_m"][:, None, :] + speeds * 0.5 * ticks
        return {"positions": pos, "speeds": speeds, "masks": torch.ones_like(pos, dtype=torch.bool),
                "lanes": initial["lane"][:, None, :].repeat(1, n_steps, 1),
                "state": {"is_cav": initial["is_cav"]}, "n": n}


def stub_tools():
    return RolloutTools(
        junction_carry_presence=lambda present, junction: (np.asarray(present),
                                                           np.zeros_like(present)),
        history_spec=None, history_step_features=None, long_history_features=None,
        action_timeline=None, controllers=("native",))


def test_predictor_with_stub_model_and_guards():
    backend = WmsBackend(model=StubModel(), config={}, tools=stub_tools())
    frames = WmsPredictor(backend).predict_states(record(), {"arm": "native"}, 5.0, 0)
    assert len(frames) == 11 and len(frames[0]["vehicles"]) == 3
    assert frames[-1]["vehicles"][0]["s"] == pytest.approx(10.0 + 10.0 * 5.0)
    with pytest.raises(Exception):
        WmsPredictor(WmsBackend(model=StubModel(), config={"rollout_mode": "driver_params"},
                                tools=stub_tools()))
    with pytest.raises(ValueError):
        RolloutFlags(entries_v2=True)
