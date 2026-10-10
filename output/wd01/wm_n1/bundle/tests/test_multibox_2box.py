"""2-box owner plans, co-run model and the box-config loader (CPU only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bundle import multibox as MB
from test_multibox import FakeExecutor, _approved, _crlf_records

BOX_CONFIGS = Path(MB.__file__).resolve().parent / "box_configs"


# --------------------------------------------------------------- loader ------
@pytest.mark.parametrize("plan_dir", [
    "3box", "4box", "2box/a", "2box/b", "2box/c", "2box/c_m1only"])
def test_load_box_configs_dir_all_plans(plan_dir):
    loaded = MB.load_box_configs_dir(BOX_CONFIGS / plan_dir)
    assert loaded["index"]["schema"] == MB.BOX_CONFIGS_INDEX_SCHEMA
    assert loaded["n_boxes"] == len(loaded["boxes"])
    assert loaded["boxes"]
    for box_id, cfg in loaded["boxes"].items():
        assert cfg["schema"] == MB.BOX_CONFIG_SCHEMA
        assert cfg["box_id"] == box_id
        assert cfg["assigned_configs"]
        assert all(c in MB.CONFIG_SPECS for c in cfg["assigned_configs"])


def test_load_box_configs_dir_accepts_index_path():
    a = MB.load_box_configs_dir(BOX_CONFIGS / "2box" / "a" / "index.json")
    b = MB.load_box_configs_dir(BOX_CONFIGS / "2box" / "a")
    assert a["boxes"].keys() == b["boxes"].keys()


def test_assignment_from_box_configs_covers_all_configs():
    for variant in ("a", "b", "c"):
        assign = MB.assignment_from_box_configs(BOX_CONFIGS / "2box" / variant)
        for box_id, configs in assign.items():
            assert MB.configs_for_box(box_id, assign) == configs
            assert all(c in MB.CONFIG_SPECS for c in configs)


def test_2box_assignment_usable_by_run_config(tmp_path):
    assign = MB.assignment_from_box_configs(BOX_CONFIGS / "2box" / "b")
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "member-1", records)
    ex = FakeExecutor()
    receipt = MB.run_config("box-A", "member-1", work_dir=tmp_path,
                            executor=ex, approval_path=appr,
                            timing_gate_path=gate, records=records,
                            assignment=assign)
    assert receipt["config"] == "member-1"
    assert ex.configs() == ["member-1"]
    with pytest.raises(MB.UnassignedConfigRefusal):
        MB.run_config("box-A", "heldout-i-j4b", work_dir=tmp_path,
                      executor=FakeExecutor(), approval_path=appr,
                      timing_gate_path=gate, records=records,
                      assignment=assign)


def test_loader_rejects_member_before_head(tmp_path):
    bad = tmp_path / "boxA.json"
    bad.write_text(json.dumps({
        "schema": MB.BOX_CONFIG_SCHEMA, "box_id": "box-A",
        "assigned_configs": ["member-2", "member-1"]}), encoding="utf-8")
    with pytest.raises(MB.MultiboxError):
        MB.load_box_config(bad)


def test_loader_rejects_unknown_config_and_schema(tmp_path):
    unknown = tmp_path / "u.json"
    unknown.write_text(json.dumps({
        "schema": MB.BOX_CONFIG_SCHEMA, "box_id": "box-A",
        "assigned_configs": ["not-a-config"]}), encoding="utf-8")
    with pytest.raises(MB.MultiboxError):
        MB.load_box_config(unknown)
    wrong = tmp_path / "w.json"
    wrong.write_text(json.dumps({
        "schema": "wrong", "box_id": "box-A",
        "assigned_configs": ["member-1"]}), encoding="utf-8")
    with pytest.raises(MB.MultiboxError):
        MB.load_box_config(wrong)


def test_heldout_specs_have_idea_a_family_and_one_member():
    for config in MB.HELDOUT_CONFIGS:
        spec = MB.CONFIG_SPECS[config]
        assert spec.family == MB.IDEA_A_FAMILY
        assert spec.member == 1
        assert spec.kind == "heldout"


# ------------------------------------------------------------- 2-box plan ----
def test_variant_a_wall_clock_envelope():
    env = MB.plan_two_box_envelope("a")
    assert env["mid"]["wall_clock_hours"] == pytest.approx(102.5)
    assert env["low"]["wall_clock_hours"] == pytest.approx(70.0)
    assert env["high"]["wall_clock_hours"] == pytest.approx(135.0)
    assert tuple(env["mid"]["boxes"]["box-B"]["assigned_configs"]) == (
        "heldout-i-j4b", "heldout-ii-leave-pi",
        "heldout-iii-leave-highest-wall-speed")


def test_variant_b_wall_clock_envelope_and_order():
    env = MB.plan_two_box_envelope("b")
    assert env["mid"]["wall_clock_hours"] == pytest.approx(65.0)
    assert env["low"]["wall_clock_hours"] == pytest.approx(45.0)
    assert env["high"]["wall_clock_hours"] == pytest.approx(85.0)
    box_b = env["mid"]["boxes"]["box-B"]["assigned_configs"]
    assert box_b[0] == "heldout-i-j4b"          # headline FIRST
    assert box_b[-1] == "n-member-1"            # single-member N point


def test_variant_c_co_run_envelope():
    c2 = MB.plan_two_box_envelope("c", concurrency=2)
    assert c2["mid"]["wall_clock_hours"] == pytest.approx(53.0)
    assert c2["mid"]["heldout_i_landing_hours"] == pytest.approx(20.0)
    c3 = MB.plan_two_box_envelope("c", concurrency=3)
    assert c3["mid"]["wall_clock_hours"] == pytest.approx(52.0)
    assert c3["mid"]["heldout_i_landing_hours"] == pytest.approx(26.0)
    assert c3["low"]["wall_clock_hours"] == pytest.approx(36.0)
    assert c3["high"]["wall_clock_hours"] == pytest.approx(68.0)


@pytest.mark.parametrize("variant", ["a", "b", "c"])
def test_member_1_lands_after_one_member(variant):
    conc = 3 if variant == "c" else 1
    plan = MB.plan_two_box(variant, concurrency=conc)
    assert plan["member_1_landing_hours"] == pytest.approx(
        MB.design_gpu_hours("member-1"))


def test_members_2_5_start_only_after_member_1():
    plan = MB.plan_two_box("c", concurrency=3)
    box_a = plan["boxes"]["box-A"]
    head_done = box_a["finish_hours"]["member-1"]
    assert box_a["start_hours"]["member-1"] == pytest.approx(0.0)
    for k in range(2, 6):
        assert box_a["start_hours"][f"member-{k}"] >= head_done - 1e-9


def test_co_run_beats_sequential_box_finish():
    seq = MB.plan_two_box("b")
    c3 = MB.plan_two_box("c", concurrency=3)
    assert c3["box_finish_hours"]["box-A"] < seq["box_finish_hours"]["box-A"]
    assert c3["box_finish_hours"]["box-B"] < seq["box_finish_hours"]["box-B"]
    assert c3["co_run"] is True and seq["co_run"] is False


def test_co_run_speedup_assumption_documented():
    assert MB.co_run_speedup(1) == 1.0
    assert MB.co_run_speedup(2) == 1.3
    assert MB.co_run_speedup(3) == 1.5
    assert MB.CO_RUN_SPEEDUP[3] == 1.5


def test_plan_two_box_rejects_bad_input():
    with pytest.raises(MB.MultiboxError):
        MB.plan_two_box("z")
    with pytest.raises(MB.MultiboxError):
        MB.plan_two_box("a", concurrency=2)     # sequential variant
    with pytest.raises(MB.MultiboxError):
        MB.plan_two_box("c", concurrency=4)


def test_plan_fleet_concurrency_is_monotone_and_default_unchanged():
    assert MB.plan_fleet(3)["wall_clock_hours"] == pytest.approx(54.0)
    assert MB.plan_fleet(4)["wall_clock_hours"] == pytest.approx(41.0)
    conc = MB.plan_fleet(3, concurrency=2)
    assert conc["wall_clock_hours"] <= 54.0
    assert conc["member_1_first_on_box_1"] is True
    assert conc["boxes"]["box-1"][0] == "member-1"


# ----------------------------------------------------- generation/writer -----
@pytest.mark.parametrize("variant,conc", [("a", 1), ("b", 1), ("c", 3)])
def test_write_and_reload_two_box_configs(tmp_path, variant, conc):
    written = MB.write_two_box_configs(tmp_path, variants=(variant,),
                                       c_concurrency=conc)
    assert len(written) == 3
    loaded = MB.load_box_configs_dir(tmp_path / variant)
    plan = MB.plan_two_box(variant, concurrency=conc)
    assert loaded["plan"] == f"2-box-{variant}"
    assert loaded["wall_clock_hours"] == pytest.approx(
        plan["wall_clock_hours"])
    for box_id, cfg in loaded["boxes"].items():
        assert tuple(cfg["assigned_configs"]) == tuple(
            plan["boxes"][box_id]["assigned_configs"])
        assert len(cfg["configs"]) == len(cfg["assigned_configs"])
        for entry in cfg["configs"]:
            assert entry["box_cli"][:4] == [
                "python", "-m", "bundle.multibox", "run-config"]
            assert box_id in entry["box_cli"]


# ----------------------------------------------------- M1-FIRST variant -----
def test_c_m1only_box_configs_and_stop_after():
    loaded = MB.load_box_configs_dir(BOX_CONFIGS / "2box" / "c_m1only")
    assert loaded["index"]["m1_only"] is True
    a = loaded["boxes"]["box-A"]
    b = loaded["boxes"]["box-B"]
    assert tuple(a["assigned_configs"]) == ("member-1",)
    assert a["stop_after"] == "member-1"
    assert tuple(b["assigned_configs"]) == ("heldout-i-j4b",)
    assert b["stop_after"] == "heldout-i-j4b"
    # (ii)/(iii) and the N comparator are NOT queued
    for absent in ("heldout-ii-leave-pi",
                   "heldout-iii-leave-highest-wall-speed", "n-member-1"):
        assert absent not in a["assigned_configs"]
        assert absent not in b["assigned_configs"]


def test_c_m1only_runs_on_pack_v3_with_v3_gate_and_envcheck():
    """Amendment 5: the M1-only configs use pack_v3 / format v3 / K_max 192."""
    from model import constants as C

    for box in ("boxA.json", "boxB.json"):
        data = json.loads(
            (BOX_CONFIGS / "2box" / "c_m1only" / box).read_text(
                encoding="utf-8"))
        assert data["corpus_source"] == MB.M1_ONLY_CORPUS_SOURCE
        assert data["corpus_source"].endswith("pack_v3")
        assert data["format"] == MB.M1_ONLY_FORMAT == C.FORMAT_V3
        assert data["k_max"] == MB.M1_ONLY_K_MAX == C.K_MAX_V3
        assert data["envcheck_argv"][-1] == "v3"
        assert "--format" in data["envcheck_argv"]
        for entry in data["configs"]:
            assert entry["corpus_source"] == MB.M1_ONLY_CORPUS_SOURCE
            assert entry["format"] == C.FORMAT_V3
            assert entry["k_max"] == C.K_MAX_V3


def test_other_plans_stay_on_pack_v2():
    """Only c_m1only moved to pack_v3; the frozen v2 plans are unchanged."""
    for path in sorted((BOX_CONFIGS / "2box").rglob("box*.json")):
        if path.parent.name == "c_m1only":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["corpus_source"].endswith("pack_v2")


def test_m1_only_build_matches_written_files(tmp_path):
    built = MB.build_m1_only_configs()
    written = MB.write_m1_only_configs(tmp_path / "c_m1only")
    assert len(written) == 3
    loaded = MB.load_box_configs_dir(tmp_path / "c_m1only")
    for box_id, cfg in built["boxes"].items():
        assert loaded["boxes"][box_id]["assigned_configs"] == \
            cfg["assigned_configs"]
        assert loaded["boxes"][box_id]["stop_after"] == cfg["stop_after"]


def test_run_box_config_stops_after_member_1_box_a(tmp_path):
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "member-1", records)
    ex = FakeExecutor()
    cfg = BOX_CONFIGS / "2box" / "c_m1only" / "boxA.json"
    out = MB.run_box_config(cfg, work_dir=tmp_path, executor=ex,
                            approval_paths={"member-1": appr},
                            timing_gate_paths={"member-1": gate},
                            records=records)
    assert out["configs"] == ["member-1"]
    assert out["stop_after"] == "member-1"
    assert ex.configs() == ["member-1"]


def test_run_box_config_gpu_b_is_heldout_i_only(tmp_path):
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "heldout-i-j4b", records)
    ex = FakeExecutor()
    cfg = BOX_CONFIGS / "2box" / "c_m1only" / "boxB.json"
    out = MB.run_box_config(cfg, work_dir=tmp_path, executor=ex,
                            approval_paths={"heldout-i-j4b": appr},
                            timing_gate_paths={"heldout-i-j4b": gate},
                            records=records)
    assert out["configs"] == ["heldout-i-j4b"]
    assert ex.configs() == ["heldout-i-j4b"]       # (ii)/(iii)/N not run


def test_run_box_stop_after_rejects_unassigned(tmp_path):
    with pytest.raises(MB.MultiboxError):
        MB.run_box("box-A", work_dir=tmp_path,
                   assignment={"box-A": ("member-1",)},
                   executor=FakeExecutor(), stop_after="member-2")


def test_loader_rejects_stop_after_not_assigned(tmp_path):
    bad = tmp_path / "boxA.json"
    bad.write_text(json.dumps({
        "schema": MB.BOX_CONFIG_SCHEMA, "box_id": "box-A",
        "assigned_configs": ["member-1"], "stop_after": "member-5"}),
        encoding="utf-8")
    with pytest.raises(MB.MultiboxError):
        MB.load_box_config(bad)


def test_quick_tune_plan_shape():
    qt = MB.QUICK_TUNE_PLAN
    assert 0 < qt["subset_frac"] < 1 and 0 < qt["epoch_frac"] < 1
    assert "--quick-tune" in qt["cli"] and "--subset-frac" in qt["cli"]
    assert "--epoch-frac" in qt["cli"]


def test_two_box_markdown_lists_variants_and_assumption():
    md = MB.two_box_plan_markdown()
    for needle in ("Variant (a)", "Variant (b)", "Variant (c)",
                   "1.3x", "1.5x", "timing gate", "held-out (i) lands",
                   "wall-clock"):
        assert needle in md


# --------------------------------------------- A4.2 CAL-half pinning -----
def _iter_str_lists(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_str_lists(v)
    elif isinstance(obj, list):
        if obj and all(isinstance(x, str) for x in obj):
            yield obj
        else:
            for v in obj:
                yield from _iter_str_lists(v)


def test_no_box_config_has_unpinned_cal_split():
    """Every `--split cal` argv in every box config names an A4.2 half."""
    seen_cal = 0
    for path in sorted(BOX_CONFIGS.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for argv in _iter_str_lists(data):
            if "--split" not in argv:
                continue
            i = argv.index("--split")
            if i + 1 < len(argv) and argv[i + 1] == "cal":
                seen_cal += 1
                assert "--cal-half" in argv, f"{path}: unpinned cal argv {argv}"
    assert seen_cal > 0


def test_idea_a_member_selection_eval_pinned_to_cal_tune():
    from evalh import cal_split as CS
    for path in sorted(BOX_CONFIGS.rglob("box*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data.get("configs", []):
            if entry["kind"] != "member" \
                    or entry["family"] != MB.IDEA_A_FAMILY:
                continue
            argv = entry["eval_argv"]
            assert "--cal-half" in argv, f"{path}/{entry['config']}"
            assert argv[argv.index("--cal-half") + 1] == CS.CAL_TUNE
            cal = entry["calibration_argv"]
            assert cal is not None, f"{path}/{entry['config']} no calibration"
            assert cal[cal.index("--cal-half") + 1] == CS.CAL_CAL


@pytest.mark.parametrize("variant,conc", [("a", 1), ("b", 1), ("c", 3)])
def test_two_box_disk_matches_builder(variant, conc):
    built = MB.build_two_box_configs(variant, concurrency=conc)
    for box_id, cfg in built["boxes"].items():
        fname = "boxA.json" if box_id == MB.BOX_A else "boxB.json"
        disk = json.loads((BOX_CONFIGS / "2box" / variant / fname)
                          .read_text(encoding="utf-8"))
        assert disk == cfg
    disk_index = json.loads((BOX_CONFIGS / "2box" / variant / "index.json")
                            .read_text(encoding="utf-8"))
    assert disk_index == built["index"]


def test_m1_only_disk_matches_builder():
    built = MB.build_m1_only_configs()
    for box_id, cfg in built["boxes"].items():
        fname = "boxA.json" if box_id == MB.BOX_A else "boxB.json"
        disk = json.loads((BOX_CONFIGS / "2box" / "c_m1only" / fname)
                          .read_text(encoding="utf-8"))
        assert disk == cfg
    disk_index = json.loads(
        (BOX_CONFIGS / "2box" / "c_m1only" / "index.json")
        .read_text(encoding="utf-8"))
    assert disk_index == built["index"]


@pytest.mark.parametrize("n_boxes,plan_dir", [(3, "3box"), (4, "4box")])
def test_plan_fleet_disk_matches_builder(n_boxes, plan_dir):
    built = MB.build_plan_fleet_configs(n_boxes)
    for i, (box_id, cfg) in enumerate(built["boxes"].items(), start=1):
        disk = json.loads((BOX_CONFIGS / plan_dir / f"box{i}.json")
                          .read_text(encoding="utf-8"))
        assert disk == cfg
    disk_index = json.loads((BOX_CONFIGS / plan_dir / "index.json")
                            .read_text(encoding="utf-8"))
    assert disk_index == built["index"]
