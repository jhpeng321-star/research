"""P10 MULTIBOX tests: config-per-box, refusal, approval binding, order, resume.

All CPU, no SUMO, no real training: the executor is a fake callable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bundle import multibox as MB


# ----------------------------------------------------------------- helpers ---
class FakeExecutor:
    """Records (box_id, config) calls; returns a small receipt."""

    def __init__(self):
        self.calls = []

    def __call__(self, box_id, config, work_dir):
        self.calls.append((box_id, config))
        return {"ok": True, "stages": ["convert", "timing", "train",
                                       "checkpoint", "resume", "eval"]}

    def configs(self):
        return [c for _, c in self.calls]


def _crlf_records(tmp_path):
    """CRLF copies of the four frozen records (LF-sha preserved)."""
    recs = {}
    for key, (path, frozen) in MB.DEFAULT_RECORDS.items():
        raw = Path(path).read_bytes()
        lf = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        p = tmp_path / f"{key}.md"
        p.write_bytes(lf.replace(b"\n", b"\r\n"))
        recs[key] = (p, frozen)
    return recs


def _write_gate(directory, gate_bytes=b'{"gate": 1}\n'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    gate = directory / MB.TIMING_GATE_FILENAME
    gate.write_bytes(gate_bytes)
    return gate


def _write_approval(directory, gate, config, records, **overrides):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "schema": MB.APPROVAL_SCHEMA,
        "approved": True,
        "owner": "owner",
        "config": config,
        "timing_gate_sha256": hashlib.sha256(
            Path(gate).read_bytes()).hexdigest(),
        "registration_lf_sha256": MB.LF_SHA_REGISTRATION,
        "amendment_1_lf_sha256": MB.LF_SHA_AMENDMENT_1,
        "amendment_2_lf_sha256": MB.LF_SHA_AMENDMENT_2,
        "amendment_3_lf_sha256": MB.LF_SHA_AMENDMENT_3,
        "amendment_4_lf_sha256": MB.LF_SHA_AMENDMENT_4,
        "amendment_5_lf_sha256": MB.LF_SHA_AMENDMENT_5,
        "amendment_6_lf_sha256": MB.LF_SHA_AMENDMENT_6,
        "cal_split_lf_sha256": MB.LF_SHA_CAL_SPLIT_A4,
    }
    body.update(overrides)
    path = directory / MB.APPROVAL_FILENAME
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _approved(tmp_path, config, records, **overrides):
    d = tmp_path / config
    gate = _write_gate(d)
    appr = _write_approval(d, gate, config, records, **overrides)
    return appr, gate


# ------------------------------------------------------------ assignment -----
def test_default_assignment_is_one_config_per_box():
    receipt = MB.assignment_receipt()
    assert receipt["one_config_per_box"] is True
    assert receipt["n_configs"] == len(MB.ALL_CONFIGS) == 13
    assert receipt["n_boxes"] == len(MB.ALL_CONFIGS)
    for config in MB.ALL_CONFIGS:
        assert MB.configs_for_box(MB.box_for_config(config)) == (config,)


def test_every_config_assigned_exactly_once():
    assigned = [c for configs in MB.DEFAULT_BOX_ASSIGNMENT.values()
                for c in configs]
    assert sorted(assigned) == sorted(MB.ALL_CONFIGS)


def test_planned_order_member_1_first():
    order = MB.planned_order(("member-3", "heldout-i-j4b", "member-1",
                              "member-2"))
    assert order == ("member-1", "member-2", "member-3", "heldout-i-j4b")


def test_known_heldout_specs():
    assert set(MB.HELDOUT_CONFIGS) == {
        "heldout-i-j4b", "heldout-ii-leave-pi",
        "heldout-iii-leave-highest-wall-speed"}
    assert all(MB.CONFIG_SPECS[c].kind == "heldout" for c in MB.HELDOUT_CONFIGS)
    assert MB.CONFIG_SPECS["heldout-iii-leave-highest-wall-speed"].excluded_arm


# ----------------------------------------------------- variant N comparator ---
def test_variant_n_member_configs_present():
    assert MB.N_MEMBER_CONFIGS == ("n-member-1", "n-member-2", "n-member-3",
                                   "n-member-4", "n-member-5")
    for k in range(1, 6):
        spec = MB.CONFIG_SPECS[f"n-member-{k}"]
        assert spec.kind == "member"
        assert spec.family == MB.N_FAMILY
        assert spec.member == k
        assert f"--member {k}" in spec.train_cli


def test_variant_n_has_its_own_member_1_head():
    assert MB.FAMILY_MEMBER_1[MB.N_FAMILY] == "n-member-1"
    assert MB.FAMILY_MEMBER_1[MB.IDEA_A_FAMILY] == "member-1"
    assert MB.N_MEMBER_CONFIGS[0] not in MB.MEMBER_CONFIGS
    assert set(MB.ALL_MEMBER_CONFIGS) == set(MB.MEMBER_CONFIGS) | set(
        MB.N_MEMBER_CONFIGS)


def test_planned_order_member_1_first_within_each_family():
    order = MB.planned_order(("n-member-3", "member-3", "n-member-1",
                              "member-1", "n-member-2", "member-2",
                              "heldout-i-j4b"))
    assert order == ("member-1", "member-2", "member-3", "n-member-1",
                     "n-member-2", "n-member-3", "heldout-i-j4b")
    for family, head in MB.FAMILY_MEMBER_1.items():
        members = [c for c in order if MB.CONFIG_SPECS[c].kind == "member"
                   and MB.CONFIG_SPECS[c].family == family]
        assert members[0] == head


# ------------------------------------------------------------- refusals ------
def test_run_config_refuses_unassigned_config(tmp_path):
    with pytest.raises(MB.UnassignedConfigRefusal):
        MB.run_config("box-member-1", "member-2", work_dir=tmp_path,
                      executor=FakeExecutor())


def test_run_config_refuses_unknown_box(tmp_path):
    with pytest.raises(MB.MultiboxError):
        MB.run_config("box-nope", "member-1", work_dir=tmp_path,
                      executor=FakeExecutor())


def test_run_config_refuses_unknown_config(tmp_path):
    with pytest.raises(MB.MultiboxError):
        MB.run_config("box-nope", "not-a-config", work_dir=tmp_path,
                      executor=FakeExecutor())


def test_n_member_2_refused_before_n_member_1(tmp_path):
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "n-member-2", records)
    with pytest.raises(MB.BoxOrderError):
        MB.run_config("box-n-member-2", "n-member-2", work_dir=tmp_path,
                      executor=FakeExecutor(), approval_path=appr,
                      timing_gate_path=gate, records=records)


def test_n_family_head_does_not_require_idea_a_head(tmp_path):
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "n-member-1", records)
    ex = FakeExecutor()
    receipt = MB.run_config("box-n-member-1", "n-member-1", work_dir=tmp_path,
                            executor=ex, approval_path=appr,
                            timing_gate_path=gate, records=records)
    assert receipt["config"] == "n-member-1"
    assert MB.family_member1_completed(tmp_path, MB.N_FAMILY)
    assert not MB.family_member1_completed(tmp_path, MB.IDEA_A_FAMILY)


def test_member_2_refused_before_member_1(tmp_path):
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "member-2", records)
    with pytest.raises(MB.BoxOrderError):
        MB.run_config("box-member-2", "member-2", work_dir=tmp_path,
                      executor=FakeExecutor(), approval_path=appr,
                      timing_gate_path=gate, records=records)


# ------------------------------------------------------- approval binding ----
def test_approval_accepts_crlf_copies(tmp_path):
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "member-1", records)
    ex = FakeExecutor()
    receipt = MB.run_config("box-member-1", "member-1", work_dir=tmp_path,
                            executor=ex, approval_path=appr,
                            timing_gate_path=gate, records=records)
    assert receipt["ok"] and receipt["config"] == "member-1"
    assert ex.configs() == ["member-1"]


@pytest.mark.parametrize("field", [
    "timing_gate_sha256", "registration_lf_sha256", "amendment_1_lf_sha256",
    "amendment_2_lf_sha256", "amendment_3_lf_sha256",
    "amendment_4_lf_sha256", "amendment_5_lf_sha256", "cal_split_lf_sha256"])
def test_approval_wrong_sha_refused(tmp_path, field):
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "member-1", records, **{field: "0" * 64})
    with pytest.raises(MB.ConfigApprovalMissing) as ei:
        MB.run_config("box-member-1", "member-1", work_dir=tmp_path,
                      executor=FakeExecutor(), approval_path=appr,
                      timing_gate_path=gate, records=records)
    assert field in str(ei.value)


def test_approval_not_approved_refused(tmp_path):
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "member-1", records, approved=False)
    with pytest.raises(MB.ConfigApprovalMissing):
        MB.run_config("box-member-1", "member-1", work_dir=tmp_path,
                      executor=FakeExecutor(), approval_path=appr,
                      timing_gate_path=gate, records=records)


def test_approval_tampered_record_refused(tmp_path):
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "member-1", records)
    p, sha = records["amendment_3"]
    p.write_bytes(p.read_bytes() + b"tampered\r\n")
    with pytest.raises(MB.ConfigApprovalMissing):
        MB.run_config("box-member-1", "member-1", work_dir=tmp_path,
                      executor=FakeExecutor(), approval_path=appr,
                      timing_gate_path=gate, records=records)


def test_approval_wrong_config_refused(tmp_path):
    records = _crlf_records(tmp_path)
    assignment = {"box-both": ("member-1",)}
    appr, gate = _approved(tmp_path, "member-2", records)   # declares member-2
    with pytest.raises(MB.ConfigApprovalMissing):
        MB.run_config("box-both", "member-1", work_dir=tmp_path,
                      executor=FakeExecutor(), approval_path=appr,
                      timing_gate_path=gate, records=records,
                      assignment=assignment)


def test_approval_tampered_amendment_4_refused(tmp_path):
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "member-1", records)
    p, _ = records["amendment_4"]
    p.write_bytes(p.read_bytes() + b"tampered\r\n")
    with pytest.raises(MB.ConfigApprovalMissing):
        MB.run_config("box-member-1", "member-1", work_dir=tmp_path,
                      executor=FakeExecutor(), approval_path=appr,
                      timing_gate_path=gate, records=records)


def test_approval_tampered_cal_split_refused(tmp_path):
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "member-1", records)
    p, _ = records["cal_split"]
    p.write_bytes(p.read_bytes() + b"\r\n")
    with pytest.raises(MB.ConfigApprovalMissing):
        MB.run_config("box-member-1", "member-1", work_dir=tmp_path,
                      executor=FakeExecutor(), approval_path=appr,
                      timing_gate_path=gate, records=records)


def test_approval_tampered_amendment_5_refused(tmp_path):
    """Amendment 5 (K_max format, v3=192) is a required, CRLF-safe binding."""
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "member-1", records)
    p, _ = records["amendment_5"]
    p.write_bytes(p.read_bytes() + b"tampered\r\n")
    with pytest.raises(MB.ConfigApprovalMissing):
        MB.run_config("box-member-1", "member-1", work_dir=tmp_path,
                      executor=FakeExecutor(), approval_path=appr,
                      timing_gate_path=gate, records=records)


def test_write_config_approval_roundtrips_all_eight_records(tmp_path):
    records = _crlf_records(tmp_path)
    gate = _write_gate(tmp_path / "c_m1only" / "boxA")
    appr = MB.write_config_approval(
        tmp_path / "c_m1only" / "boxA" / MB.APPROVAL_FILENAME,
        config="member-1", timing_gate_path=gate)
    # The generated file binds A4 + A5 + A6 + the CAL split for every config,
    # including the c_m1only boxA/boxB configs.
    assert appr["amendment_4_lf_sha256"] == MB.LF_SHA_AMENDMENT_4
    assert appr["amendment_5_lf_sha256"] == MB.LF_SHA_AMENDMENT_5
    assert appr["amendment_6_lf_sha256"] == MB.LF_SHA_AMENDMENT_6
    assert appr["cal_split_lf_sha256"] == MB.LF_SHA_CAL_SPLIT_A4
    appr_path = tmp_path / "c_m1only" / "boxA" / MB.APPROVAL_FILENAME
    got = MB.require_config_approval(
        appr_path, config="member-1", timing_gate_path=gate, records=records)
    assert got["config"] == "member-1"


def test_approval_tampered_amendment_6_refused(tmp_path):
    """Amendment 6 (state global token, A6) is a required, CRLF-safe binding."""
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "member-1", records)
    p, _ = records["amendment_6"]
    p.write_bytes(p.read_bytes() + b"tampered\r\n")
    with pytest.raises(MB.ConfigApprovalMissing):
        MB.run_config("box-member-1", "member-1", work_dir=tmp_path,
                      executor=FakeExecutor(), approval_path=appr,
                      timing_gate_path=gate, records=records)


def test_approval_wrong_amendment_6_sha_refused(tmp_path):
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "member-1", records,
                           amendment_6_lf_sha256="0" * 64)
    with pytest.raises(MB.ConfigApprovalMissing) as ei:
        MB.run_config("box-member-1", "member-1", work_dir=tmp_path,
                      executor=FakeExecutor(), approval_path=appr,
                      timing_gate_path=gate, records=records)
    assert "amendment_6_lf_sha256" in str(ei.value)


# ------------------------------------------------------------- ordering ------
def test_member_1_first_within_a_box(tmp_path):
    records = _crlf_records(tmp_path)
    assignment = {"box-both": ("member-2", "member-1")}
    appr1, gate1 = _approved(tmp_path, "member-1", records)
    appr2, gate2 = _approved(tmp_path, "member-2", records)
    ex = FakeExecutor()
    out = MB.run_box("box-both", work_dir=tmp_path, executor=ex,
                     approval_paths={"member-1": appr1, "member-2": appr2},
                     timing_gate_paths={"member-1": gate1, "member-2": gate2},
                     records=records, assignment=assignment)
    assert list(out["configs"]) == ["member-1", "member-2"]
    assert ex.configs() == ["member-1", "member-2"]


def test_run_fleet_runs_every_box_member_1_first(tmp_path):
    records = _crlf_records(tmp_path)
    approvals, gates = {}, {}
    for config in MB.ALL_CONFIGS:
        approvals[config], gates[config] = _approved(tmp_path, config, records)
    ex = FakeExecutor()
    out = MB.run_fleet(work_dir=tmp_path, executor=ex,
                       approval_paths=approvals, timing_gate_paths=gates,
                       records=records, parallel=False)
    assert len(out["boxes"]) == len(MB.DEFAULT_BOX_ASSIGNMENT)
    calls = ex.configs()
    assert "member-1" in calls
    assert calls.index("member-1") < calls.index("member-2")
    assert sorted(calls) == sorted(MB.ALL_CONFIGS)


def test_run_fleet_parallel_executes_all(tmp_path):
    records = _crlf_records(tmp_path)
    approvals, gates = {}, {}
    for config in MB.ALL_CONFIGS:
        approvals[config], gates[config] = _approved(tmp_path, config, records)
    ex = FakeExecutor()
    out = MB.run_fleet(work_dir=tmp_path, executor=ex,
                       approval_paths=approvals, timing_gate_paths=gates,
                       records=records, parallel=True, max_workers=4)
    assert out["parallel"] is True
    assert sorted(ex.configs()) == sorted(MB.ALL_CONFIGS)


# --------------------------------------------------------------- resume ------
def test_resume_from_per_box_state(tmp_path):
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "member-1", records)
    ex = FakeExecutor()
    first = MB.run_config("box-member-1", "member-1", work_dir=tmp_path,
                          executor=ex, approval_path=appr,
                          timing_gate_path=gate, records=records)
    assert first.get("skipped") is not True
    assert MB.member1_completed(tmp_path)
    second = MB.run_config("box-member-1", "member-1", work_dir=tmp_path,
                           executor=ex, approval_path=appr,
                           timing_gate_path=gate, records=records)
    assert second["skipped"] is True
    assert ex.configs() == ["member-1"]      # not re-run
    state = json.loads((MB.box_dir(tmp_path, "box-member-1")
                        / MB.STATE_FILENAME).read_text(encoding="utf-8"))
    assert "member-1" in state["completed"]


def test_box_state_is_per_box(tmp_path):
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "member-1", records)
    ex = FakeExecutor()
    a = tmp_path / "run-a"
    b = tmp_path / "run-b"
    MB.run_config("box-member-1", "member-1", work_dir=a, executor=ex,
                  approval_path=appr, timing_gate_path=gate, records=records)
    assert MB.load_box_state(a, "box-member-1")["completed"]
    assert MB.load_box_state(b, "box-member-1")["completed"] == {}


# ------------------------------------------------------- N approval binding --
def test_approval_binds_n_config_and_refuses_mismatch(tmp_path):
    records = _crlf_records(tmp_path)
    appr, gate = _approved(tmp_path, "n-member-1", records)
    ex = FakeExecutor()
    receipt = MB.run_config("box-n-member-1", "n-member-1", work_dir=tmp_path,
                            executor=ex, approval_path=appr,
                            timing_gate_path=gate, records=records)
    assert receipt["kind"] == "member" and receipt["member"] == 1
    assert ex.configs() == ["n-member-1"]

    # The approval is bound to one config; declaring another config refuses.
    other = tmp_path / "other"
    appr2, gate2 = _approved(other, "member-1", records)   # declares member-1
    with pytest.raises(MB.ConfigApprovalMissing):
        MB.run_config("box-n-member-1", "n-member-1", work_dir=other,
                      executor=FakeExecutor(), approval_path=appr2,
                      timing_gate_path=gate2, records=records,
                      assignment={"box-n-member-1": ("n-member-1",)})


# -------------------------------------------------------------- planner ------
def test_design_gpu_hours_within_design_envelope():
    for config in MB.MEMBER_CONFIGS + MB.HELDOUT_CONFIGS:
        lo, hi = MB.hours_range(config)
        assert (lo, hi) == MB.IDEA_A_MEMBER_GPU_HOURS
        assert lo <= MB.design_gpu_hours(config) <= hi
    for config in MB.N_MEMBER_CONFIGS:
        lo, hi = MB.hours_range(config)
        assert (lo, hi) == MB.N_MEMBER_GPU_HOURS
        assert lo <= MB.design_gpu_hours(config) <= hi


@pytest.mark.parametrize("n_boxes", [3, 4])
def test_plan_fleet_assigns_every_config_exactly_once(n_boxes):
    plan = MB.plan_fleet(n_boxes)
    assert plan["n_boxes"] == n_boxes
    assert len(plan["boxes"]) == n_boxes
    scheduled = [c for configs in plan["boxes"].values() for c in configs]
    assert sorted(scheduled) == sorted(MB.ALL_CONFIGS)


@pytest.mark.parametrize("n_boxes", [3, 4])
def test_plan_fleet_member_1_first_on_box_1(n_boxes):
    plan = MB.plan_fleet(n_boxes)
    assert plan["member_1_first_on_box_1"] is True
    assert plan["boxes"]["box-1"][0] == "member-1"
    assert plan["boxes"]["box-2"][0] == "n-member-1"


@pytest.mark.parametrize("n_boxes", [3, 4])
def test_plan_fleet_release_after_each_family_head(n_boxes):
    plan = MB.plan_fleet(n_boxes)
    start, hours = plan["config_start_hours"], plan["config_hours"]
    for family, head in MB.FAMILY_MEMBER_1.items():
        head_done = start[head] + hours[head]
        for config in MB.ALL_MEMBER_CONFIGS:
            spec = MB.CONFIG_SPECS[config]
            if spec.family == family and spec.member > 1:
                assert start[config] >= head_done - 1e-9


def test_plan_fleet_3_and_4_box_wall_clock():
    mid3, mid4 = MB.plan_fleet(3), MB.plan_fleet(4)
    assert mid3["wall_clock_hours"] == pytest.approx(54.0)
    assert mid4["wall_clock_hours"] == pytest.approx(41.0)
    assert mid4["wall_clock_hours"] <= mid3["wall_clock_hours"]
    lo3 = MB.plan_fleet(3, hours_fn=lambda c: MB.hours_range(c)[0])
    hi3 = MB.plan_fleet(3, hours_fn=lambda c: MB.hours_range(c)[1])
    lo4 = MB.plan_fleet(4, hours_fn=lambda c: MB.hours_range(c)[0])
    hi4 = MB.plan_fleet(4, hours_fn=lambda c: MB.hours_range(c)[1])
    assert (lo3["wall_clock_hours"], hi3["wall_clock_hours"]) == (37.0, 71.0)
    assert (lo4["wall_clock_hours"], hi4["wall_clock_hours"]) == (28.0, 54.0)


def test_plan_fleet_rejects_bad_input():
    with pytest.raises(MB.MultiboxError):
        MB.plan_fleet(0)
    with pytest.raises(MB.MultiboxError):
        MB.plan_fleet(3, configs=["member-1", "not-a-config"])
    with pytest.raises(MB.MultiboxError):
        MB.design_gpu_hours("not-a-config")


def test_format_plan_lists_every_config():
    text = MB.format_plan(MB.plan_fleet(3))
    assert "wall_clock_hours" in text
    for config in MB.ALL_CONFIGS:
        assert config in text
