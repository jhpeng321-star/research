"""Standard horizon report over per-state C1 result files (contracts)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(PKG))

from wms_eval import horizon_report as HR  # noqa: E402


def _arm_block(ds_by_horizon, error=None):
    car = {}
    if not error:
        for horizon, ds in ds_by_horizon.items():
            entry = {"ds_med": ds, "ds_p90": (None if ds is None else ds * 1.5),
                     "dv_med": 0.0, "dv_p90": 0.0, "match": 1.0}
            car[str(float(horizon))] = {"all": dict(entry), "hdv": dict(entry),
                                        "cav": dict(entry, ds_med=0.0, ds_p90=0.0)}
    return {"error": error, "car": car}


def _state(world, t0, arms):
    return {"world": world, "t0_s": t0,
            "arms": {arm: _arm_block(values) for arm, values in arms.items()}}


def _write_states(path: Path, model, set_name, states):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "set": set_name,
                                "states": states}), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# across-state summary
# --------------------------------------------------------------------------- #
def test_summarize_median_iqr_p90_sd():
    s = HR.summarize([4.0, 1.0, 3.0, 2.0])
    assert s["n"] == 4
    assert s["median"] == pytest.approx(2.5)
    assert s["q1"] == pytest.approx(1.75)
    assert s["q3"] == pytest.approx(3.25)
    assert s["iqr"] == pytest.approx(1.5)
    assert s["p90"] == pytest.approx(3.7)
    assert s["sd"] == pytest.approx(1.25 ** 0.5)
    empty = HR.summarize([])
    assert empty["n"] == 0 and empty["median"] is None


def test_car_medians_skips_errors_and_missing_horizons():
    states = [
        _state("wA", 0.0, {"pi": {60.0: 10.0, 120.0: 20.0}}),
        _state("wB", 0.0, {"pi": {60.0: None}}),                 # 60 s missing
        {"world": "wC", "t0_s": 0.0,
         "arms": {"pi": _arm_block({60.0: 99.0}, error="ValueError: boom")}},
    ]
    assert HR.car_medians(states, "pi", 60.0) == [10.0]
    assert HR.car_medians(states, "pi", 120.0) == [20.0]
    block = HR.horizon_block(states, "pi", 60.0)
    assert block["n"] == 1 and block["median"] == 10.0
    assert block["p90_within_median"] == pytest.approx(15.0)


def test_blowup_rate_requires_every_threshold():
    thresholds = ((60.0, 50.0), (120.0, 100.0))
    states = [
        _state("w1", 0.0, {"pi": {60.0: 60.0, 120.0: 150.0}}),   # blow-up
        _state("w2", 0.0, {"pi": {60.0: 60.0, 120.0: 90.0}}),    # fails 120 s only
        _state("w3", 0.0, {"pi": {60.0: 10.0, 120.0: 150.0}}),   # fails 60 s only
        _state("w4", 0.0, {"pi": {60.0: 5.0}}),                  # truncated -> excluded
    ]
    blow = HR.blowup_rate(states, "pi", thresholds)
    assert blow["n_blowup"] == 1 and blow["n_eval"] == 3
    assert blow["n_excluded"] == 1
    assert blow["rate"] == pytest.approx(1.0 / 3.0)
    assert HR.blowup_rate([], "pi", thresholds)["rate"] is None
    # a plain exceedance is not a blow-up: strict > thresholds
    single = [_state("w1", 0.0, {"pi": {60.0: 50.0, 120.0: 100.0}})]
    assert HR.blowup_rate(single, "pi", thresholds)["n_blowup"] == 0


def test_blowup_uses_hdv_stratum_not_all():
    state = {"world": "w1", "t0_s": 0.0, "arms": {"pi": {
        "error": None,
        "car": {"60.0": {"all": {"ds_med": 1.0}, "hdv": {"ds_med": 60.0},
                         "cav": {"ds_med": 0.0}},
                "120.0": {"all": {"ds_med": 1.0}, "hdv": {"ds_med": 150.0},
                          "cav": {"ds_med": 0.0}}}}}}
    assert HR.blowup_rate([state], "pi")["n_blowup"] == 1


# --------------------------------------------------------------------------- #
# arm selection
# --------------------------------------------------------------------------- #
def test_tested_arm_pi_for_unseen_native_for_seen():
    assert HR.tested_arm("u_same", ["native", "pi"]) == "pi"
    assert HR.tested_arm("u_new", ["native", "pi"]) == "pi"
    assert HR.tested_arm("seen", ["native"]) == "native"
    assert HR.tested_arm("seen", ["native", "pi"]) == "native"
    assert HR.tested_arm("u_same", ["native"]) == "native"
    assert HR.tested_arm("seen", []) is None


# --------------------------------------------------------------------------- #
# paired bootstrap over worlds
# --------------------------------------------------------------------------- #
def _mk_states(values):
    """values: {(world, t0): ds} -> per-state result list."""
    return [_state(world, t0, {"pi": {60.0: ds}}) for (world, t0), ds in values.items()]


def test_paired_bootstrap_pairs_by_world_and_t0_and_is_deterministic():
    model_states = _mk_states({("wA", 300.0): 10.0, ("wA", 480.0): 14.0,
                               ("wB", 300.0): 20.0})
    ref_states = _mk_states({("wA", 300.0): 5.0, ("wA", 480.0): 6.0,
                             ("wB", 300.0): 4.0, ("wZ", 0.0): 0.0})
    first = HR.paired_bootstrap(model_states, ref_states, "pi", 60.0,
                                n_boot=200, seed=7)
    second = HR.paired_bootstrap(model_states, ref_states, "pi", 60.0,
                                 n_boot=200, seed=7)
    assert first == second
    # unpaired wZ must not enter; diffs 5, 8, 16 -> point median 8 (model worse)
    assert first["n_pairs"] == 3 and first["n_worlds"] == 2
    assert first["point"] == pytest.approx(8.0)
    assert first["n_boot"] == 200
    assert first["lo"] <= first["point"] <= first["hi"]
    # the reference is better: the interval excludes 0 on the positive side
    assert first["lo"] > 0.0


def test_paired_bootstrap_empty_when_no_shared_worlds():
    model_states = _mk_states({("wA", 0.0): 3.0})
    ref_states = _mk_states({("wB", 0.0): 3.0})
    out = HR.paired_bootstrap(model_states, ref_states, "pi", 60.0, n_boot=10)
    assert out["point"] is None and out["n_boot"] == 0


# --------------------------------------------------------------------------- #
# full report build + markdown
# --------------------------------------------------------------------------- #
def _fixture_sources(tmp_path: Path):
    wms_states = _mk_states({("wA", 300.0): 3.0, ("wA", 480.0): 4.0,
                             ("wB", 300.0): 8.0})
    wms_states[0]["arms"]["native"] = _arm_block({60.0: 2.0, 120.0: 40.0})
    wms_states[1]["arms"]["native"] = _arm_block({60.0: 3.0})
    wms_states[2]["arms"]["native"] = _arm_block({60.0: 6.0, 120.0: 55.0})
    keep_states = _mk_states({("wA", 300.0): 30.0, ("wA", 480.0): 35.0,
                              ("wB", 300.0): 80.0})
    keep_states[0]["arms"]["native"] = _arm_block({60.0: 90.0, 120.0: 150.0})
    keep_states[1]["arms"]["native"] = _arm_block({60.0: 95.0})
    keep_states[2]["arms"]["native"] = _arm_block({60.0: 200.0, 120.0: 400.0})
    src = {
        "wms": {"u_same": _write_states(tmp_path / "wms" / "s.json", "wms",
                                        "u_same", wms_states)},
        "keep_speed": {"u_same": _write_states(tmp_path / "keep" / "s.json",
                                               "keep_speed", "u_same", keep_states)},
    }
    return src, wms_states, keep_states


def test_build_report_and_markdown_end_to_end(tmp_path):
    src, wms_states, keep_states = _fixture_sources(tmp_path)
    report = HR.build_report(src, reference="keep_speed", n_boot=40, seed=1)
    assert report["schema"] == "wms_eval_horizon_report/1"
    assert report["models"]["wms"]["u_same"]["tested_arm"] == "pi"
    # only horizons with data are reported
    assert report["models"]["wms"]["u_same"]["arms"]["pi"]["horizons_present"] == [60.0]
    summary = report["models"]["wms"]["u_same"]["arms"]["pi"]["summary"]["60"]
    assert summary["n"] == 3 and summary["median"] == pytest.approx(4.0)
    paired = report["paired_vs_reference"]["wms"]["u_same"]
    assert paired["arm"] == "pi" and paired["reference"] == "keep_speed"
    ci = paired["horizons"]["60"]
    assert ci["point"] == pytest.approx(-31.0)      # paired diffs -27, -31, -72
    assert ci["n_worlds"] == 2
    # the reference model itself is never paired against itself
    assert "keep_speed" not in report["paired_vs_reference"]
    text = HR.markdown(report)
    assert "# Horizon report" in text and "| model | set | arm |" in text
    assert "Blow-up rule: |ds|>50m @60s and |ds|>100m @120s" in text
    assert "wms / u_same" in text and "-31.00" in text
    # native arm summaries are kept in the JSON, not silently dropped
    assert report["models"]["wms"]["u_same"]["arms"]["native"]["summary"]["60"]["n"] == 3
    # the keep_speed pi arm lacks 120 s data: excluded, never counted as survived
    blow_pi = report["models"]["keep_speed"]["u_same"]["arms"]["pi"]["blowup"]
    assert blow_pi["n_eval"] == 0 and blow_pi["rate"] is None
    blow_native = report["models"]["keep_speed"]["u_same"]["arms"]["native"]["blowup"]
    assert blow_native["rate"] == pytest.approx(1.0) and blow_native["n_eval"] == 2


def test_build_report_uses_reference_arm_per_set(tmp_path):
    # seen-style set: only a native arm on both sides -> paired on native
    a = [_state("wA", 0.0, {"native": {60.0: 5.0}})]
    b = [_state("wA", 0.0, {"native": {60.0: 50.0}})]
    src = {"wms": {"seen": _write_states(tmp_path / "a.json", "wms", "seen", a)},
           "keep_speed": {"seen": _write_states(tmp_path / "b.json", "keep_speed",
                                                "seen", b)}}
    report = HR.build_report(src, reference="keep_speed", n_boot=0)
    assert report["models"]["wms"]["seen"]["tested_arm"] == "native"
    assert report["paired_vs_reference"]["wms"]["seen"]["arm"] == "native"
    assert report["paired_vs_reference"]["wms"]["seen"]["horizons"]["60"]["point"] \
        == pytest.approx(-45.0)


def test_build_report_rejects_mismatched_model_or_set(tmp_path):
    bad = _write_states(tmp_path / "bad.json", "other", "u_same",
                        _mk_states({("wA", 0.0): 1.0}))
    with pytest.raises(HR.HorizonReportError, match="declares model/set"):
        HR.build_report({"wms": {"u_same": bad}})
    empty = _write_states(tmp_path / "empty.json", "wms", "u_same", [])
    with pytest.raises(HR.HorizonReportError, match="no per-state results"):
        HR.build_report({"wms": {"u_same": empty}})


def test_load_results_defaults_names_from_filename(tmp_path):
    states = _mk_states({("wA", 0.0): 1.0})
    path = tmp_path / "states__mymodel__myset.json"
    path.write_text(json.dumps({"states": states}), encoding="utf-8")
    model, set_name, loaded = HR.load_results(path)
    assert (model, set_name, len(loaded)) == ("mymodel", "myset", 1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_writes_markdown_and_json_and_refuses_overwrite(tmp_path, capsys):
    src, _wms, _keep = _fixture_sources(tmp_path)
    out = tmp_path / "HORIZON_REPORT.md"
    argv = ["--states", f"wms=u_same={src['wms']['u_same']}",
            "--states", f"keep_speed=u_same={src['keep_speed']['u_same']}",
            "--reference", "keep_speed", "--bootstrap", "20",
            "--seed", "3", "--out", str(out)]
    assert HR.main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["out_json"]) == tmp_path / "HORIZON_REPORT.json"
    assert "wms" in out.read_text(encoding="utf-8")
    assert json.loads((tmp_path / "HORIZON_REPORT.json").read_text(encoding="utf-8")) \
        ["schema"] == "wms_eval_horizon_report/1"
    assert HR.main(argv) == 2          # no silent overwrite
    assert "error" in capsys.readouterr().out
    bad = ["--states", "wms_u_same", "--out", str(tmp_path / "x.md")]
    assert HR.main(bad) == 2           # malformed entry fails closed
    wrong_suffix = ["--states", f"wms=u_same={src['wms']['u_same']}",
                    "--out", str(tmp_path / "report.txt")]
    assert HR.main(wrong_suffix) == 2


def test_cli_dispatch_via_package_main():
    import inspect

    from wms_eval import cli

    source = inspect.getsource(cli.main)
    assert '"horizon": horizon_report.main' in source
