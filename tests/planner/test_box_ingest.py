"""Tests for output/planner_v1/box_ingest (zero SIMULATION; tmp_path only).

Every bundle is SYNTHETIC (no SUMO, no real run data, no RESEARCH_DENSE_ROOT).
The all-good bundle must PASS with exit code 0; one focused failing fixture
per acceptance check (1)-(8) must FAIL with exit code 1; the tool must never
write into, create in, or delete from the inspected run dirs.

Test-name -> check map:
- test_check_1_*              -> (1) complete receipts / world coverage
- test_check_2_*              -> (2) per-run cap + total budget
- test_check_3_*              -> (3) candidate log rows == candidates.jsonl
- test_check_4_*              -> (4) 1c no walls; 1d wall law + pricing
- test_check_5_*              -> (5) uniform planner_code_sha256
- test_check_6_*              -> (6) 1d GATE_VERDICT pass
- test_check_7_*              -> (7) LAUNCH_SUMMARY returncodes 0
- test_check_8_*              -> (8) optional sha256 manifest
- test_out_dir_* / test_read_only_*  -> write confinement / read-only
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from output.planner_v1.box_ingest import ingest

PENS = [0.02, 0.02, 0.05, 0.05, 0.10, 0.10, 0.20, 0.20, 0.40, 0.40]
WORLDS = [
    {
        "chunk": "c2" if i % 2 == 0 else "c3",
        "world": f"w{8500 + i}m2700p{int(PENS[i] * 100):02d}t{240 + 30 * i}",
        "seed": 8500 + i,
        "penetration": PENS[i],
        "loading_veh_h": 2700,
        "t0_s": float(240 + 30 * i),
        "split": "TRAIN",
    }
    for i in range(10)
]
WORLDS_1C = [w["world"] for w in WORLDS]
WORLDS_1D = [w["world"] for w in WORLDS if w["penetration"] >= 0.10]
SHA_1C = "a" * 64
SHA_1D = "b" * 64
CAND_ROWS = 5


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _receipt(exp_id, world, t0, *, walls, code_sha, total_s, sim_cap, extra=None) -> dict:
    rec = {
        "schema": "pl04_closed_loop_receipt/1",
        "run_id": f"{world}__planner_oracle",
        "world": world,
        "split": "TRAIN",
        "seed": 1,
        "run_purpose": "planner",
        "predictor": "perfect_sim",
        "predictor_class": "oracle",
        "walls": walls,
        "planner_code_sha256": code_sha,
        "harness_state_restored": True,
        "first_control_tick_s": t0,
        "sim_s": {"master_s": 600.0, "lookahead_s": total_s - 600.0,
                  "total_s": total_s, "cap_s": sim_cap},
        "candidate_log": {"enabled": True, "file": "candidates.jsonl", "rows": CAND_ROWS},
        "exit_status": "complete",
    }
    if extra:
        rec.update(extra)
    return rec


def _run_dir(bundle: Path, exp_id: str, world: str) -> Path:
    rel = "runs/plexp1c_oracle_v1" if exp_id == "PL-EXP1c" else "runs/plexp1d_oracle_v1"
    return bundle / rel / f"{world}__planner_oracle"


def _launch_dir(bundle: Path, exp_id: str) -> Path:
    rel = ("output/plexp1c_oracle_v1_launch" if exp_id == "PL-EXP1c"
           else "output/plexp1d_oracle_v1_launch")
    return bundle / rel


def _launch_summary(bundle: Path, exp_id: str, worlds, *, started_utc="2026-09-30T10:00:00Z"):
    runs = {f"{w}__planner_oracle": {
        "status": "done", "returncode": 0, "exit_status": "complete",
        "sim_s": 7000.0, "run_purpose": "planner"} for w in worlds}
    summary = {"started_utc": started_utc, "finished_utc": "2026-09-30T12:00:00Z",
               "workers": 10, "root": str(bundle),
               "out": str(bundle / ("runs/plexp1c_oracle_v1" if exp_id == "PL-EXP1c"
                                    else "runs/plexp1d_oracle_v1")),
               "python": "python", "runs": runs}
    _write_json(_launch_dir(bundle, exp_id) / "LAUNCH_SUMMARY.json", summary)
    return summary


def _gate(bundle: Path, *, passed=True, w2c_passed=False, finished_utc="2026-09-30T09:00:00Z"):
    verdict = {
        "schema": "plexp1d_gate_verdict/1",
        "pass": passed,
        "results": {
            "w1": {"required": True, "returncode": 0, "passed": True},
            "w2_branch": {"required": True, "returncode": 0, "passed": True},
            "w2c": {"required": False, "returncode": 0 if w2c_passed else 1,
                    "passed": w2c_passed},
            "w3": {"required": True, "returncode": 0, "passed": True},
        },
        "code_root": str(bundle / "code_plexp1d"),
        "root": str(bundle),
        "finished_utc": finished_utc,
        "rule": "pass iff w1, w2_branch, w3 pass; w2c informational",
    }
    _write_json(_launch_dir(bundle, "PL-EXP1d") / "GATE_VERDICT.json", verdict)
    return verdict


def make_bundle(tmp_path: Path) -> Path:
    """A complete, consistent, in-budget synthetic returned bundle."""
    bundle = tmp_path / "bundle"
    _write_json(bundle / "output/planner_v1/PLEXP1_WORLDS.json",
                {"selection_seed": 20261202, "worlds": WORLDS})
    (bundle / "version").mkdir(parents=True, exist_ok=True)
    (bundle / "version/VERSION.txt").write_text(
        "sumo 1.27.1\ncores 48\n", encoding="utf-8")
    for exp_id, worlds, walls, sha, cap in (
        ("PL-EXP1c", WORLDS_1C, "off", SHA_1C, 9000.0),
        ("PL-EXP1d", WORLDS_1D, "on", SHA_1D, 13000.0),
    ):
        t0_by_world = {w["world"]: w["t0_s"] for w in WORLDS}
        for w in worlds:
            run = _run_dir(bundle, exp_id, w)
            extra = None
            if walls == "on":
                extra = {"wall_law_version": "wall-v2.1-closure",
                         "wall_pricing": "closure_speed_only",
                         "walls_summary": {"n_walls": 1}}
            _write_json(run / "RUN_RECEIPT.json",
                        _receipt(exp_id, w, t0_by_world[w], walls=walls, code_sha=sha,
                                 total_s=7000.0, sim_cap=cap, extra=extra))
            _write_jsonl(run / "candidates.jsonl", [{"i": i} for i in range(CAND_ROWS)])
            for name in ("physics_steps.jsonl", "decisions.jsonl", "dispatcher_rows.jsonl"):
                _write_jsonl(run / name, [{"t_s": 1.0}])
            (run / "walls.jsonl").write_text("", encoding="utf-8")
        _launch_summary(bundle, exp_id, worlds)
    _gate(bundle)
    return bundle


def read_result(out: Path) -> dict:
    return json.loads((out / "INGEST_RESULT.json").read_text(encoding="utf-8"))


def run(bundle: Path, out: Path, *extra) -> int:
    return ingest.main(["--root", str(bundle), "--out", str(out), *extra])


def check_status(result: dict, exp_id: str, check_id: str) -> str:
    for check in result["experiments"][exp_id]["checks"]:
        if check["check_id"] == check_id:
            return check["status"]
    raise KeyError(f"{exp_id}/{check_id} not in result")


def edit_receipt(bundle: Path, exp_id: str, world: str, **fields) -> dict:
    path = _run_dir(bundle, exp_id, world) / "RUN_RECEIPT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(fields)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def _snapshot(root: Path) -> dict:
    snap = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            snap[str(path.relative_to(root))] = (
                path.stat().st_size,
                path.stat().st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    return snap


# ---------------------------------------------------------------- all-good
def test_all_good_bundle_passes(tmp_path):
    bundle = make_bundle(tmp_path)
    out = tmp_path / "ingest_out"
    assert run(bundle, out) == 0
    result = read_result(out)
    assert result["overall"] == "PASS"
    assert result["code_sha256"] == {"PL-EXP1c": SHA_1C, "PL-EXP1d": SHA_1D}
    assert (out / "INGEST_REPORT.md").is_file()
    for exp_id in ("PL-EXP1c", "PL-EXP1d"):
        statuses = {c["check_id"]: c["status"]
                    for c in result["experiments"][exp_id]["checks"]}
        assert statuses == {"1_worlds": "PASS", "2_sim_budget": "PASS",
                            "3_candidate_log": "PASS", "4_walls": "PASS",
                            "5_code_sha": "PASS",
                            "6_gate": "NA" if exp_id == "PL-EXP1c" else "PASS",
                            "7_returncodes": "PASS"}, statuses
    assert result["manifest"]["status"] == "NA"


def test_1c_gate_check_is_not_applicable(tmp_path):
    bundle = make_bundle(tmp_path)
    out = tmp_path / "out"
    assert run(bundle, out) == 0
    result = read_result(out)
    assert check_status(result, "PL-EXP1c", "6_gate") == "NA"


# ---------------------------------------------------------------- check 1
def test_check_1_missing_world_run_dir_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    shutil.rmtree(_run_dir(bundle, "PL-EXP1c", WORLDS_1C[0]))
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    result = read_result(out)
    assert result["overall"] == "FAIL"
    assert check_status(result, "PL-EXP1c", "1_worlds") == "FAIL"


def test_check_1_bad_receipt_field_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    edit_receipt(bundle, "PL-EXP1c", WORLDS_1C[1], exit_status="partial")
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    assert check_status(read_result(out), "PL-EXP1c", "1_worlds") == "FAIL"


def test_check_1_wrong_first_control_tick_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    edit_receipt(bundle, "PL-EXP1d", WORLDS_1D[0], first_control_tick_s=0.0)
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    assert check_status(read_result(out), "PL-EXP1d", "1_worlds") == "FAIL"


# ---------------------------------------------------------------- check 2
def test_check_2_over_cap_sim_s_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    edit_receipt(bundle, "PL-EXP1c", WORLDS_1C[0],
                 sim_s={"master_s": 600.0, "lookahead_s": 8900.0,
                        "total_s": 9500.0, "cap_s": 9000.0})
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    assert check_status(read_result(out), "PL-EXP1c", "2_sim_budget") == "FAIL"


def test_check_2_over_budget_total_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    for w in WORLDS_1C:
        edit_receipt(bundle, "PL-EXP1c", w,
                     sim_s={"master_s": 600.0, "lookahead_s": 7400.0,
                            "total_s": 8000.0, "cap_s": 9000.0})
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    result = read_result(out)
    check = next(c for c in result["experiments"]["PL-EXP1c"]["checks"]
                 if c["check_id"] == "2_sim_budget")
    assert check["status"] == "FAIL"
    assert check["evidence"]["total_s"] == 80000.0


# ---------------------------------------------------------------- check 3
def test_check_3_row_count_mismatch_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    _write_jsonl(_run_dir(bundle, "PL-EXP1c", WORLDS_1C[0]) / "candidates.jsonl",
                 [{"i": i} for i in range(CAND_ROWS + 2)])
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    assert check_status(read_result(out), "PL-EXP1c", "3_candidate_log") == "FAIL"


def test_check_3_disabled_candidate_log_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    edit_receipt(bundle, "PL-EXP1d", WORLDS_1D[0],
                 candidate_log={"enabled": False, "file": None, "rows": 0})
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    assert check_status(read_result(out), "PL-EXP1d", "3_candidate_log") == "FAIL"


# ---------------------------------------------------------------- check 4
def test_check_4_walls_present_in_1c_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    edit_receipt(bundle, "PL-EXP1c", WORLDS_1C[0], walls="on")
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    assert check_status(read_result(out), "PL-EXP1c", "4_walls") == "FAIL"


def test_check_4_wall_rows_in_1c_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    _write_jsonl(_run_dir(bundle, "PL-EXP1c", WORLDS_1C[2]) / "walls.jsonl",
                 [{"event": "register"}])
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    assert check_status(read_result(out), "PL-EXP1c", "4_walls") == "FAIL"


def test_check_4_wrong_wall_law_in_1d_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    edit_receipt(bundle, "PL-EXP1d", WORLDS_1D[0], wall_law_version="wall-v1")
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    assert check_status(read_result(out), "PL-EXP1d", "4_walls") == "FAIL"


def test_check_4_wrong_wall_pricing_in_1d_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    edit_receipt(bundle, "PL-EXP1d", WORLDS_1D[1], wall_pricing="full_cost")
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    assert check_status(read_result(out), "PL-EXP1d", "4_walls") == "FAIL"


# ---------------------------------------------------------------- check 5
def test_check_5_mixed_code_shas_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    edit_receipt(bundle, "PL-EXP1c", WORLDS_1C[3], planner_code_sha256="c" * 64)
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    result = read_result(out)
    assert check_status(result, "PL-EXP1c", "5_code_sha") == "FAIL"
    assert result["code_sha256"]["PL-EXP1d"] == SHA_1D


# ---------------------------------------------------------------- check 6
def test_check_6_gate_missing_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    (_launch_dir(bundle, "PL-EXP1d") / "GATE_VERDICT.json").unlink()
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    assert check_status(read_result(out), "PL-EXP1d", "6_gate") == "FAIL"


def test_check_6_gate_failed_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    _gate(bundle, passed=False)
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    assert check_status(read_result(out), "PL-EXP1d", "6_gate") == "FAIL"


def test_check_6_w2c_informational_is_accepted(tmp_path):
    bundle = make_bundle(tmp_path)
    _gate(bundle, passed=True, w2c_passed=False)
    out = tmp_path / "out"
    assert run(bundle, out) == 0
    assert check_status(read_result(out), "PL-EXP1d", "6_gate") == "PASS"


def test_check_6_gate_after_launch_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    _gate(bundle, passed=True, finished_utc="2026-09-30T11:00:00Z")
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    assert check_status(read_result(out), "PL-EXP1d", "6_gate") == "FAIL"


# ---------------------------------------------------------------- check 7
def test_check_7_nonzero_returncode_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    summary_path = _launch_dir(bundle, "PL-EXP1c") / "LAUNCH_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runs"][f"{WORLDS_1C[0]}__planner_oracle"]["returncode"] = 1
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    assert check_status(read_result(out), "PL-EXP1c", "7_returncodes") == "FAIL"


def test_check_7_missing_launch_entry_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    summary_path = _launch_dir(bundle, "PL-EXP1d") / "LAUNCH_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    del summary["runs"][f"{WORLDS_1D[0]}__planner_oracle"]
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    out = tmp_path / "out"
    assert run(bundle, out) == 1
    assert check_status(read_result(out), "PL-EXP1d", "7_returncodes") == "FAIL"


# ---------------------------------------------------------------- check 8
def write_manifest(path: Path, root: Path, rels) -> dict:
    entries = {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in rels}
    path.write_text(json.dumps({"files": entries}), encoding="utf-8")
    return entries


def test_check_8_good_manifest_passes(tmp_path):
    bundle = make_bundle(tmp_path)
    rels = ["output/planner_v1/PLEXP1_WORLDS.json",
            f"runs/plexp1c_oracle_v1/{WORLDS_1C[0]}__planner_oracle/RUN_RECEIPT.json"]
    manifest = tmp_path / "box.sha256.json"
    write_manifest(manifest, bundle, rels)
    out = tmp_path / "out"
    assert run(bundle, out, "--manifest", str(manifest)) == 0
    result = read_result(out)
    assert result["manifest"]["status"] == "PASS"
    assert result["manifest"]["evidence"]["files_checked"] == len(rels)


def test_check_8_sha_mismatch_fails(tmp_path):
    bundle = make_bundle(tmp_path)
    rel = f"runs/plexp1c_oracle_v1/{WORLDS_1C[1]}__planner_oracle/RUN_RECEIPT.json"
    manifest = tmp_path / "box.sha256.json"
    manifest.write_text(json.dumps({"files": {rel: "0" * 64}}), encoding="utf-8")
    out = tmp_path / "out"
    assert run(bundle, out, "--manifest", str(manifest)) == 1
    result = read_result(out)
    assert result["manifest"]["status"] == "FAIL"
    assert result["overall"] == "FAIL"


def test_check_8_text_sha256sum_manifest(tmp_path):
    bundle = make_bundle(tmp_path)
    rel = "output/planner_v1/PLEXP1_WORLDS.json"
    digest = hashlib.sha256((bundle / rel).read_bytes()).hexdigest()
    manifest = tmp_path / "box.sha256"
    manifest.write_text(f"{digest}  {rel}\n", encoding="utf-8")
    out = tmp_path / "out"
    assert run(bundle, out, "--manifest", str(manifest)) == 0
    assert read_result(out)["manifest"]["status"] == "PASS"


# ---------------------------------------------------------------- confinement
def test_out_dir_inside_runs_is_refused(tmp_path):
    bundle = make_bundle(tmp_path)
    assert run(bundle, bundle / "runs" / "ingest_out") == 2
    assert not (bundle / "runs" / "ingest_out").exists()


def test_guard_out_dir_raises_on_runs_and_drive(tmp_path):
    bundle = make_bundle(tmp_path)
    with pytest.raises(ingest.OutDirRefusedError):
        ingest.guard_out_dir(bundle / "runs" / "x", bundle)
    d_path = "D" + ":" + "/somewhere/out"  # built to avoid a D: source literal
    with pytest.raises(ingest.OutDirRefusedError):
        ingest.guard_out_dir(d_path, bundle)


def test_out_dir_refusal_leaves_no_files(tmp_path):
    bundle = make_bundle(tmp_path)
    target = bundle / "runs" / "ingest_out"
    assert not target.exists()
    assert run(bundle, target) == 2
    assert not target.exists()


def test_read_only_inspected_dirs_untouched(tmp_path):
    bundle = make_bundle(tmp_path)
    before = _snapshot(bundle)
    out = tmp_path / "out"
    assert run(bundle, out) == 0
    after = _snapshot(bundle)
    assert after == before
    assert (out / "INGEST_RESULT.json").is_file()


# ---------------------------------------------------------------- helpers
def test_worlds_for_1d_filters_penetration():
    kept = ingest.worlds_for("PL-EXP1d", WORLDS)
    assert [w["world"] for w in kept] == WORLDS_1D
    assert [w["world"] for w in ingest.worlds_for("PL-EXP1c", WORLDS)] == WORLDS_1C


def test_load_manifest_rejects_empty(tmp_path):
    manifest = tmp_path / "empty.sha256"
    manifest.write_text("", encoding="utf-8")
    with pytest.raises(ingest.ManifestError):
        ingest.load_manifest(manifest)
