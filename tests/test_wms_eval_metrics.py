import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(PKG))
from wms_eval.metrics import car_metrics, paired_difference, traffic_metrics  # noqa: E402


def frame(t, rows):
    return {"time_s": t, "vehicles": rows}


def row(vid, s, lane, v, cav=False):
    return {"id": vid, "s": s, "lane": lane, "v": v, "present": True, "is_cav": cav}


def test_car_metrics_match_by_identity_and_strata():
    recorded = [frame(0, [row("h", 0, 0, 10), row("c", 10, 1, 12, True)]),
                frame(5, [row("h", 50, 0, 10), row("c", 70, 1, 10, True)])]
    predicted = [frame(0, [row("h", 0, 0, 10), row("c", 10, 1, 12, True)]),
                 frame(5, [row("h", 55, 0, 9), row("c", 70, 2, 11, True)])]
    result = car_metrics(predicted, recorded, horizons=(5.0,))
    assert result["5.0"]["all"]["abs_delta_s_m"]["median"] == 2.5
    assert result["5.0"]["cav"]["lane_agreement"] == 0.0
    assert result["5.0"]["hdv"]["matched_fraction"] == 1.0


def test_traffic_metrics_and_pairing_are_deterministic():
    stream = [frame(0, [row("a", 10, 0, 5), row("b", 80, 1, 10)]),
              frame(0.5, [row("a", 12, 0, 5), row("b", 76, 1, 10)])]
    metrics = traffic_metrics(stream, bin_s=30.0, free_speed_mps=10.0)
    assert metrics["0"]["sections"]["upstream"]["stopped_vehicles"] == 0.0
    assert metrics["0"]["sections"]["merge"]["density"] > 0.0
    delta = paired_difference(metrics, metrics)
    assert delta["0"]["sections"]["upstream"]["total_delay"] == 0.0


def test_total_delay_is_normalized_seconds_not_metres():
    # Review blocker 5: (1 - v/v_free)*dt seconds, not (v_free - v)*dt metres.
    stream = [frame(t, [row("a", 10, 0, 8)]) for t in (0.0, 0.5, 1.0, 1.5)]
    metrics = traffic_metrics(stream, t0_s=0.0, bin_s=30.0, free_speed_mps=10.0)
    assert abs(metrics["0"]["sections"]["merge"]["total_delay"] - 0.4) < 1e-9
