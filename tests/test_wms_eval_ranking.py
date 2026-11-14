import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(PKG))
from wms_eval import ranking  # noqa: E402
from wms_eval.metrics import cumulative_delay, throughput, truncate_frames  # noqa: E402


def frame(t, rows):
    return {"time_s": t, "vehicles": rows}


def row(vid, s, lane, v, cav=False, present=True):
    return {"id": vid, "s": s, "lane": lane, "v": v, "present": present, "is_cav": cav}


def test_cumulative_delay_is_windowed_and_network_sums_sections():
    stream = [frame(t, [row("a", 10, 1, 0.0)]) for t in (0.0, 0.5, 30.0, 30.5)]
    at30 = cumulative_delay(stream, t0_s=0.0, horizon_s=30.0, bin_s=30.0, free_speed_mps=10.0)
    at60 = cumulative_delay(stream, t0_s=0.0, horizon_s=60.0, bin_s=30.0, free_speed_mps=10.0)
    assert abs(at30["merge"] - 1.0) < 1e-9
    assert abs(at60["merge"] - 2.0) < 1e-9
    assert abs(at60["network"] - (at60["upstream"] + at60["merge"] + at60["downstream"])) < 1e-9


def test_throughput_counts_distinct_downstream_crossings():
    stream = [frame(0.0, [row("a", 70, 1, 5), row("b", -20, 0, 5)]),
              frame(0.5, [row("a", 80, 1, 5), row("b", -19, 0, 5)]),
              frame(1.0, [row("a", 90, 1, 5), row("b", 60, 0, 5)]),
              frame(1.5, [row("a", 100, 1, 5), row("b", 80, 0, 5)])]
    assert throughput(stream) == 2


def test_truncate_frames_keeps_window_edges():
    stream = [frame(t, [row("a", t, 1, 5)]) for t in (0.0, 0.5, 120.0, 120.5)]
    kept = truncate_frames(stream, t0_s=0.0, horizon_s=120.0)
    assert [f["time_s"] for f in kept] == [0.0, 0.5, 120.0]


def _bundle(delays_native, delays_pi, *, t0=0.0, free=10.0):
    return {
        "native": [frame(t, [row("a", 10, 1, free - d * free / 0.5)]) for t, d in
                   zip((0.0, 0.5, 30.0, 30.5), delays_native)],
        "pi": [frame(t, [row("a", 10, 1, free - d * free / 0.5)]) for t, d in
               zip((0.0, 0.5, 30.0, 30.5), delays_pi)],
    }


def _state_item(pred_native, pred_pi, rec_native, rec_pi):
    def arm(frames):
        return {"error": None,
                "predicted": ranking.traffic_bundle(frames[0], t0_s=0.0, horizon_s=60.0,
                                                    free_speed_mps=10.0),
                "recorded": ranking.traffic_bundle(frames[1], t0_s=0.0, horizon_s=60.0,
                                                   free_speed_mps=10.0)}
    return {"world": "w", "t0_s": 0.0, "seed": 0,
            "arms": {"native": arm((pred_native, rec_native)),
                     "pi": arm((pred_pi, rec_pi))}}


def test_paired_metrics_and_decision_regret_pick_better_arm():
    # recorded: native delay 1.0, pi delay 0.25 (pi much better); predicted also pi better.
    native_frames = _bundle([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    pi_frames = _bundle([0.25, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    item = _state_item(native_frames["native"], pi_frames["native"],
                       native_frames["native"], pi_frames["native"])
    row = ranking.paired_metrics(item, 60.0)
    assert row["network_total_delay"]["recorded"] < 0
    stats = ranking.paired_stats([row for _ in range(3)], "network_total_delay",
                                 noise_floor=0.5)
    assert stats["sign_agreement"] == 1.0
    reg = ranking.decision_regret([row for _ in range(3)], noise_floor=0.5)
    assert reg["model_correct_share"] == 1.0
    assert reg["model_mean_regret"] == 0.0
    assert reg["always_native_mean_regret"] > 0.0


def test_analyze_and_markdown_render():
    native = _bundle([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])["native"]
    pi = _bundle([0.25, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])["native"]
    states = [_state_item(native, pi, native, pi) for _ in range(4)]
    run = {"model": "wms", "checkpoint": "x.pt", "states_path": "u_same.json",
           "horizon_s": 120.0, "cav_mode": "teacher", "states": states}
    report = ranking.analyze([run], noise_floor=1.0)
    assert report["models"]["wms"]["sets"]["u_same"]["horizons"]
    text = ranking.markdown(report)
    assert "Decision regret" in text
    assert "wms" in text
