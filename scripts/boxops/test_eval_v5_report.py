"""Synthetic-only report tests; no project/simulator imports or external temp paths."""
import importlib.util
import json
import math
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "eval_v5_report", Path(__file__).with_name("eval_v5_report.py")
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


@pytest.fixture(autouse=True)
def require_workspace_temp(tmp_path):
    assert tmp_path.is_relative_to(report.WORKSPACE), "use --basetemp runs/active/evalv5_pytest_report"


def summary(controller="native", seed=1, demand=1900, pen=0.05, value=10):
    return {
        "schema": "c4_sumo_episode_summary/1", "controller": controller,
        "seed": seed, "demand_vph": demand, "penetration": pen,
        "warmup_s": 300, "horizon_s": 600, "sim_end_s": 900,
        # Explicit enriched metadata; actual current env summaries omit this model.
        "driver_population": "textbook", "population": 100,
        "episode": {"system_seconds_per_requested_vehicle": value,
                    "source_wait_vehicle_seconds": value * 2, "arrived": 99,
                    "totals": {"speed_deficit_vehicle_seconds": 9999,
                               "braking_vehicle_seconds": 9999}},
        "window": {"source_wait_vehicle_seconds": value,
                   "speed_deficit_vehicle_seconds": value * 3,
                   "arrivals": 42, "braking_vehicle_seconds": 4},
        "minimum_closing_ttc_s": 2.5, "collisions": 0, "teleports": 0,
        "wall_seconds": 12.5,
    }


def write_episode(root, relative, data=None, manifest=None):
    path = root / relative
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(data or summary()), encoding="utf-8")
    if manifest is not None:
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def build(root):
    return report.build_report([root], use_scipy=False)


def test_schema_scopes_and_single_seed(tmp_path):
    write_episode(tmp_path, "episode", manifest={"status": "complete", "policy": "native"})
    result = build(tmp_path)
    metrics = result["cells"][0]["metrics"]
    expected = {"system_seconds_per_requested_vehicle": 10, "source_wait_vehicle_seconds": 20,
                "window_source_wait_vehicle_seconds": 10, "window_speed_deficit_vehicle_seconds": 30,
                "arrivals": 42, "episode_arrivals": 99, "braking_vehicle_seconds": 4,
                "minimum_closing_ttc_s": 2.5, "collisions": 0, "teleports": 0, "wall_seconds": 12.5}
    for name, value in expected.items():
        assert metrics[name]["mean"] == value
        assert metrics[name]["n_seeds"] == 1
        assert metrics[name]["ci95"] is None
    assert result["episodes"][0]["completion"] == "manifest"


def test_student_t_and_empty_data():
    ci = report.ConfidenceIntervals(use_scipy=False)
    result = ci.summarize({1: 1.0, 2: 3.0})
    assert result["mean"] == 2
    assert result["ci95"] == pytest.approx([2 - 12.706205, 2 + 12.706205])
    assert ci.summarize({}) == {"n_seeds": 0, "mean": None, "ci95": None, "seed_means": {}}
    assert ci.summarize({1: 5, 2: 5})["ci95"] == [5, 5]
    many = ci.summarize({i: float(i) for i in range(36)})
    assert many["ci95"][1] - many["mean"] == pytest.approx(
        report.T_CRITICAL[30] * math.sqrt(111) / 6)


def test_optional_scipy_matches_known_t():
    ci = report.ConfidenceIntervals()
    if ci.ppf is None:
        assert "stdlib" in ci.method
    else:
        assert "scipy" in ci.method
        assert ci.summarize({1: 1.0, 2: 3.0})["ci95"] == pytest.approx(
            [2 - 12.706204736, 2 + 12.706204736])


def test_group_means_average_within_seed_before_ci(tmp_path):
    # Seed 1 has two cells; seed 2 has one. Episode mean is 40, seed mean is 55.
    for name, seed, demand, value in (("a", 1, 2400, 0), ("b", 1, 2700, 20),
                                       ("c", 2, 2400, 100)):
        write_episode(tmp_path, name, summary(seed=seed, demand=demand, value=value))
    group = build(tmp_path)["groups"][0]
    assert (group["demand_group"], group["penetration_group"]) == ("heavy", "seen")
    stats = group["metrics"]["system_seconds_per_requested_vehicle"]
    assert stats["mean"] == 55
    assert stats["seed_means"] == {"1": 10, "2": 100}
    assert stats["observations_per_seed"] == {"1": 2, "2": 1}
    assert stats["ci95"] == pytest.approx([55 - 12.706205 * 45, 55 + 12.706205 * 45])


def test_paired_differences_match_cells_then_average_within_seed(tmp_path):
    cases = [
        ("n1", "native", 1, 2400, 100), ("t1", "pi", 1, 2400, 90),
        ("n2", "native", 1, 2700, 200), ("t2", "pi", 1, 2700, 170),
        ("n3", "native", 2, 2400, 1000), ("t3", "pi", 2, 2400, 960),
        ("unmatched_t", "pi", 3, 2400, 99999),
        ("unmatched_n", "native", 2, 2700, 99999),
    ]
    for name, controller, seed, demand, value in cases:
        write_episode(tmp_path, name, summary(controller, seed, demand, value=value))
    result = build(tmp_path)
    stats = result["paired_groups"][0]["metrics"]["system_seconds_per_requested_vehicle"]
    assert stats["mean"] == -30
    assert stats["n_seeds"] == 2
    assert stats["n_observations"] == 3
    assert stats["seed_means"] == {"1": -20, "2": -40}
    assert stats["ci95"] == pytest.approx([-30 - 12.706205 * 10, -30 + 12.706205 * 10])
    cell = next(row for row in result["paired_cells"] if row["demand_vph"] == 2400)
    assert cell["metrics"]["system_seconds_per_requested_vehicle"]["mean"] == -25


def test_missing_nonfinite_and_unmatched_metrics_are_not_zero(tmp_path):
    native = summary()
    native["minimum_closing_ttc_s"] = None
    native["teleports"] = float("inf")
    native["window"].pop("braking_vehicle_seconds")
    treatment = summary("pi")
    treatment["collisions"] = True
    treatment["window"]["speed_deficit_vehicle_seconds"] = float("nan")
    write_episode(tmp_path, "native", native)
    write_episode(tmp_path, "pi", treatment)
    result = build(tmp_path)
    paired = result["paired_cells"][0]["metrics"]
    for name in ("minimum_closing_ttc_s", "teleports", "braking_vehicle_seconds",
                 "collisions", "window_speed_deficit_vehicle_seconds"):
        assert paired[name]["mean"] is None
        assert paired[name]["n_seeds"] == 0
    assert paired["arrivals"]["mean"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("pen,group", [(0.05, "seen"), (0.1, "seen"), (0.2, "seen"),
                                      (0.4, "seen"), (0.15, "unseen"), (0.3, "unseen"),
                                      (0.6, "unseen"), (0.7, "other:0.7")])
def test_penetration_groups(pen, group):
    assert report.penetration_group(pen) == group


@pytest.mark.parametrize("demand,group", [(1900, "light"), (2100, "2100"), (2400, "heavy"),
                                         (2700, "heavy"), (2600, "other:2600")])
def test_demand_groups(demand, group):
    assert report.demand_group(demand) == group


def test_discovery_duplicates_incomplete_and_invalid_summaries(tmp_path):
    write_episode(tmp_path, "ok", manifest={"status": "complete"})
    write_episode(tmp_path, "copy")
    write_episode(tmp_path, "old.incomplete-1", summary(seed=2))
    write_episode(tmp_path, "running", summary(seed=3), manifest={"status": "running"})
    write_episode(tmp_path, "failed", summary(seed=4), manifest={"status": "failed"})
    early = summary(seed=5)
    early["sim_end_s"] = 899
    write_episode(tmp_path, "early", early)
    write_episode(tmp_path, "aggregate", {"episodes_complete": 7})
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "summary.json").write_text("{", encoding="utf-8")
    result = report.build_report([tmp_path, tmp_path / "ok"], use_scipy=False)
    counts = result["diagnostics"]["counts"]
    assert counts["accepted_episodes"] == 1
    assert counts["duplicate_episodes"] == 1
    assert counts["overlapping_paths"] == 1
    assert counts["rejected_summaries"] == 5
    assert counts["pruned_directories"] == 1
    assert result["episodes"][0]["path"].endswith("ok/summary.json")


def test_conflicting_duplicate_excludes_entire_identity(tmp_path):
    write_episode(tmp_path, "a", summary(value=10))
    write_episode(tmp_path, "b", summary(value=20))
    write_episode(tmp_path, "c", summary(seed=2, value=30))
    result = build(tmp_path)
    assert result["diagnostics"]["counts"]["conflicting_episodes"] == 2
    assert result["cells"][0]["metrics"]["system_seconds_per_requested_vehicle"]["mean"] == 30


def test_controller_identity_and_summary_only_relay(tmp_path):
    write_episode(tmp_path, "original", summary("direct", seed=1),
                  manifest={"status": "complete", "policy": "models/policy.pt"})
    for index in (1, 2):
        write_episode(tmp_path, f"controller_00{index}/episodes/same_tag/job", summary("direct"))
    write_episode(tmp_path, "unidentified", summary("direct", seed=3))
    result = build(tmp_path)
    assert {row["controller"] for row in result["cells"]} == {
        "models/policy.pt", "controller_001/same_tag", "controller_002/same_tag"}
    assert result["diagnostics"]["counts"]["rejected_summaries"] == 1
    assert sum(ep["completion"] == "summary_timing" for ep in result["episodes"]) == 2


def test_full_and_relay_policy_copies_are_deduplicated(tmp_path):
    relative = "controller_003/episodes/policy/job"
    write_episode(tmp_path, f"full/{relative}", summary("direct"),
                  manifest={"status": "complete", "policy": "models/policy.pt"})
    write_episode(tmp_path, f"relay/{relative}", summary("direct"))
    result = build(tmp_path)
    assert len(result["episodes"]) == 1
    assert result["episodes"][0]["controller"] == "models/policy.pt"
    assert result["diagnostics"]["counts"]["duplicate_episodes"] == 1


def test_ambiguous_directory_policy_is_excluded(tmp_path):
    relative = "controller_003/episodes/policy/job"
    for name in ("a", "b"):
        write_episode(tmp_path, f"{name}/{relative}", summary("direct"),
                      manifest={"status": "complete", "policy": f"models/{name}/policy.pt"})
    write_episode(tmp_path, f"relay/{relative}", summary("direct", seed=2))
    result = build(tmp_path)
    assert len(result["episodes"]) == 2
    assert result["diagnostics"]["counts"]["ambiguous_policy_episodes"] == 1


def test_manifest_identity_mismatch_and_malformed_manifest_rejected(tmp_path):
    write_episode(tmp_path, "good")
    write_episode(tmp_path, "mismatch", summary(seed=2),
                  manifest={"status": "complete", "seed": 999})
    path = write_episode(tmp_path, "malformed", summary(seed=3))
    (path / "manifest.json").write_text("[]", encoding="utf-8")
    result = build(tmp_path)
    assert result["diagnostics"]["counts"]["rejected_summaries"] == 2


def test_mixed_schedules_rejected(tmp_path):
    write_episode(tmp_path, "a")
    data = summary(seed=2)
    data["horizon_s"] = 500
    write_episode(tmp_path, "b", data)
    with pytest.raises(ValueError, match="mixed warmup/horizon"):
        build(tmp_path)


@pytest.mark.parametrize("second_seed", [1, 2])
def test_actual_manifest_populations_rejected_before_deduplication(tmp_path, second_seed):
    # Same cell, including identical seed/outcomes: must not disappear in deduplication.
    for model, seed in (("textbook", 1), ("legacy3", second_seed)):
        data = summary(seed=seed)
        data.pop("driver_population")
        scenario = {"schema": "native_damping_explicit_requests/1", "population": 100}
        if model == "textbook":
            scenario.update(driver_population=model, driver_population_version="textbook_v1")
        write_episode(tmp_path, model, data,
                      manifest={"status": "complete", "scenario": scenario})
    with pytest.raises(report.PopulationError, match="mixed driver populations"):
        build(tmp_path)


@pytest.mark.parametrize("model", ["textbook", "legacy3"])
def test_actual_manifest_population_retained(tmp_path, model):
    data = summary()
    data.pop("driver_population")
    scenario = {"schema": "native_damping_explicit_requests/1", "population": 100}
    if model == "textbook":
        scenario.update(driver_population=model, driver_population_version="textbook_v1")
    write_episode(tmp_path, "episode", data,
                  manifest={"status": "complete", "scenario": scenario, "policy": "native"})
    result = build(tmp_path)
    episode = result["episodes"][0]
    assert episode["driver_population"] == model
    assert episode["population"] == 100
    assert episode["driver_population_version"] == ("textbook_v1" if model == "textbook" else None)
    assert episode["population_provenance"][0]["source"].startswith("manifest.scenario")
    assert result["methodology"]["driver_population"] == model


def test_metadata_poor_relay_requires_explicit_population(tmp_path):
    data = summary()
    data.pop("driver_population")
    write_episode(tmp_path, "relay", data)
    with pytest.raises(report.PopulationError, match="driver population unavailable"):
        build(tmp_path)
    out = tmp_path / "report"
    assert report.main([str(tmp_path), "--out", str(out), "--no-scipy",
                        "--driver-population", "textbook"]) == 0
    result = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert result["methodology"]["population_declaration"] == "textbook"
    assert result["episodes"][0]["population_provenance"] == [{
        "source": "caller declaration (--driver-population)",
        "driver_population": "textbook", "driver_population_version": None}]


def test_unknown_population_cannot_mix_with_known_population(tmp_path):
    write_episode(tmp_path, "known")
    data = summary(seed=2)
    data.pop("driver_population")
    write_episode(tmp_path, "unknown", data, manifest={"status": "complete"})
    with pytest.raises(report.PopulationError, match="driver population unavailable"):
        build(tmp_path)


def test_population_conflict_with_manifest_or_declaration_fails(tmp_path):
    write_episode(tmp_path, "a")
    with pytest.raises(report.PopulationError, match="conflicting driver population"):
        report.build_report([tmp_path], driver_population="legacy3", use_scipy=False)
    write_episode(tmp_path, "b", summary(seed=2), manifest={
        "status": "complete", "scenario": {"schema": "native_damping_explicit_requests/1"}})
    with pytest.raises(report.PopulationError, match="conflicting driver population"):
        build(tmp_path)


def test_population_count_is_not_model_and_versions_are_retained(tmp_path):
    for seed, count in ((1, 100), (2, 200)):
        data = summary(seed=seed)
        data.update(population=count, driver_population_version="textbook_v1")
        write_episode(tmp_path, str(seed), data)
    result = build(tmp_path)
    assert len(result["cells"]) == 1
    assert [ep["population"] for ep in result["episodes"]] == [100, 200]
    other = summary(seed=3)
    other["driver_population_version"] = "textbook_v2"
    write_episode(tmp_path, "other", other)
    with pytest.raises(report.PopulationError, match="mixed driver population versions"):
        build(tmp_path)


def test_policy_metadata_is_not_inferred_from_checkpoint_filename(tmp_path):
    data = summary("direct")
    write_episode(tmp_path, "episode", data,
                  manifest={"status": "complete", "policy": "models/native.pt"})
    result = build(tmp_path)
    assert result["episodes"][0]["controller"] == "models/native.pt"
    assert result["paired_cells"][0]["metrics"]["arrivals"]["n_seeds"] == 0


def test_paired_sign_guidance_does_not_invert_differences(tmp_path):
    write_episode(tmp_path, "native")
    data = summary("pi", value=8)
    data["window"]["arrivals"] = 50
    data["minimum_closing_ttc_s"] = 3.5
    write_episode(tmp_path, "treatment", data)
    result = build(tmp_path)
    metrics = result["paired_cells"][0]["metrics"]
    assert metrics["system_seconds_per_requested_vehicle"]["mean"] == -2
    assert metrics["arrivals"]["mean"] == 8
    assert metrics["minimum_closing_ttc_s"]["mean"] == 1
    markdown = report.render_markdown(result)
    for section in ("Paired cell differences", "Paired grouped differences"):
        text = markdown.split(f"## {section}", 1)[1].split("\n## ", 1)[0]
        assert "Negative is better for cost metrics" in text
        assert "positive is better for arrivals" in text
        assert "minimum closing TTC" in text
        assert "Differences remain treatment - native" in text


def test_restricted_paths_pruned_before_access(tmp_path, monkeypatch):
    write_episode(tmp_path, "safe")
    # Empty synthetic directories only: never create or open restricted files.
    restricted = [tmp_path / (token.upper() + "_fixture") for token in report.FORBIDDEN]
    for path in restricted:
        path.mkdir()
    original = report.os.scandir

    def guarded_scandir(path):
        assert not report.forbidden(Path(path)), "opened a restricted directory"
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(report.os, "scandir", guarded_scandir)
        result = build(tmp_path)
        for path in restricted:
            with pytest.raises(ValueError, match="restricted"):
                build(path)
        assert result["diagnostics"]["counts"]["pruned_directories"] == len(restricted)
    with pytest.raises(ValueError, match="outside-workspace"):
        report.safe_path(report.WORKSPACE.parent / "not-opened")


@pytest.mark.parametrize("mode,attributes", [(stat.S_IFLNK, 0), (stat.S_IFDIR, 0x400)])
def test_links_and_windows_reparse_points_not_followed(tmp_path, monkeypatch, mode, attributes):
    write_episode(tmp_path, "safe")
    link = tmp_path / "linked"
    link.mkdir()
    original_lstat = Path.lstat
    original_scandir = report.os.scandir

    def lstat(path, *args, **kwargs):
        if path == link:
            return SimpleNamespace(st_mode=mode, st_file_attributes=attributes)
        return original_lstat(path, *args, **kwargs)

    def scandir(path):
        assert Path(path) != link, "traversed a link"
        return original_scandir(path)

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(report.os, "scandir", scandir)
    result = build(tmp_path)
    assert result["diagnostics"]["counts"]["pruned_links"] == 1
    with pytest.raises(ValueError, match="symlink/reparse"):
        report.safe_path(link / "summary.json")


def test_cli_outputs_markdown_json_and_wall_time(tmp_path, capsys):
    inputs = tmp_path / "inputs"
    write_episode(inputs, "native")
    write_episode(inputs, "pi", summary("pi", value=8))
    out = tmp_path / "report"
    assert report.main([str(inputs), "--out", str(out), "--no-scipy"]) == 0
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert payload["schema"] == "c4_eval_v5_report/1"
    assert "Compact smoke table" in markdown
    assert "Per-episode wall time" in markdown
    assert "CI unavailable; n=1" in markdown
    assert "treatment − native" in markdown
    assert "12.5" in markdown
    assert "Reported 2 episodes" in capsys.readouterr().out


def test_empty_input_fails_without_outputs(tmp_path):
    with pytest.raises(SystemExit) as error:
        report.main([str(tmp_path), "--out", str(tmp_path / "out")])
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()
