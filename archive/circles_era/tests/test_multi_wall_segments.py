import json

from output.multi_wall_probe_v1.segments import audit_segments


def write_run(tmp_path, *, mode="distinct_upstream_h20", abort=False):
    policy = [
        {"time_s": 600, "activation_index": 0, "activation_mode": mode,
         "phase": "assembly", "member_ids": ["a", "b"], "wall_commands": {"a": -.1},
         "multi_wall_probe": {"mode": mode, "branch": None, "selection": None}},
        {"time_s": 618, "activation_index": 0, "activation_mode": mode,
         "phase": "hold", "member_ids": ["a", "b"], "wall_commands": {"a": -.1},
         "assembly_completed_at_s": 618,
         "multi_wall_probe": {"mode": mode, "branch": None, "selection": None}},
        {"time_s": 638, "activation_index": 0, "activation_mode": mode,
         "phase": "abort" if abort else "release", "member_ids": ["a", "b"],
         "wall_commands": {}, "actual_hold_elapsed_s": 20,
         "multi_wall_probe": {"mode": mode, "branch": None, "selection": None}},
    ]
    if mode == "same_cohort_extend20" and not abort:
        policy[2]["actual_hold_elapsed_s"] = 40
        policy[2]["multi_wall_probe"]["branch"] = {
            "selected": "extend", "proposed_profile": {"hold_s": 40}}
    elif not abort:
        policy += [{"time_s": 639, "activation_index": 1, "activation_mode": mode,
                    "phase": "assembly", "member_ids": ["c", "d"], "wall_commands": {"c": -.1},
                    "multi_wall_probe": {"mode": mode, "selection": {"proposal_binding": "verified"}}},
                   {"time_s": 659, "activation_index": 1, "activation_mode": mode,
                    "phase": "release", "member_ids": ["c", "d"], "wall_commands": {},
                    "actual_hold_elapsed_s": 20,
                    "multi_wall_probe": {"mode": mode, "selection": {"proposal_binding": "verified"}}}]
    for name, rows in (("policy.jsonl", policy),
                       ("formation.jsonl", [{"time_s": r["time_s"], "activation_index": r["activation_index"], "phase": r["phase"]} for r in policy]),
                       ("telemetry.jsonl", [{"kind": "command", "timestamp_s": 639,
                                              "vehicle_id": "a", "owner": "damping",
                                              "released_to_native": True}])):
        (tmp_path / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def test_segments_split_and_verify_relay(tmp_path):
    x = audit_segments(write_run(tmp_path))
    assert len(x["segments"]) == 2
    assert x["checks"]["second_disjoint"]
    assert x["checks"]["second_starts_after_first_terminal"]
    assert x["checks"]["no_wall_commands_after_segment_terminal"]
    assert x["checks"]["first_members_released_to_baseline_ownership"]


def test_abort_is_preserved_without_relay(tmp_path):
    x = audit_segments(write_run(tmp_path, abort=True))
    assert x["preserved_refusal_or_abort"]
    assert len(x["segments"]) == 1


def test_extension_receipt_reports_admitted_hold40(tmp_path):
    p = write_run(tmp_path, mode="same_cohort_extend20")
    x = audit_segments(p)
    assert x["segments"][0]["requested_hold_s"] == 40
    assert x["segments"][0]["admitted_hold_s"] == 40
    assert x["segments"][0]["delivered_hold_s"] == 40
    assert x["segments"][0]["extension_actual_match"]


def test_wall_commands_after_terminal_are_reported(tmp_path):
    p = write_run(tmp_path, abort=True)
    rows = [json.loads(x) for x in (p / "policy.jsonl").read_text().splitlines()]
    rows.append({"time_s": 639, "activation_index": 0, "phase": "hold",
                 "member_ids": ["a", "b"], "wall_commands": {"a": -.1}})
    (p / "policy.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    x = audit_segments(p)
    assert x["segments"][0]["wall_commands_after_terminal"] == 1
    assert not x["checks"]["no_wall_commands_after_segment_terminal"]
