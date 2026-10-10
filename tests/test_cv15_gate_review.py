"""Real saved-evidence successor gate, without mocked acceptance or SUMO."""

import json
from pathlib import Path
from output.controller_verification_v15 import runner
from tests.test_cv15_review import produce, expected, oracle


def test_third_gate_checks_real_prior_hashes_and_prefix(tmp_path, monkeypatch):
    reg = {"jobs": {}}
    names = {"retain": "damping", "native": "native"}
    for name, arm in names.items():
        folder = tmp_path / name
        folder.mkdir()
        produce(folder, monkeypatch, arm)
        reg["jobs"][name] = dict(
            output_dir=str(folder), run_dir=str(folder / "actual"), expected=expected(folder, arm)
        )
    reg["jobs"]["pi-external-u"] = {}

    def accept(name):
        spec = reg["jobs"][name]
        run = Path(spec["run_dir"])
        record = dict(
            accepted=True,
            job=name,
            registration_sha256=runner.registration_digest(reg),
            artifact_sha256={
                n: runner.sha256(run / n)
                for n in [
                    "manifest.json",
                    "summary.json",
                    "telemetry.jsonl",
                    "intervals.jsonl",
                    "physics_steps.jsonl",
                    "writer_trace.jsonl",
                ]
            },
        )
        (Path(spec["output_dir"]) / "accepted_review.json").write_text(json.dumps(record))

    for name in names:
        accept(name)
    assert runner._review_passed(reg, "pi-external-u")
    summary = tmp_path / "retain/actual/summary.json"
    original = summary.read_bytes()
    summary.write_bytes(original + b" ")
    assert not runner._review_passed(reg, "pi-external-u")
    summary.write_bytes(original)
    census = tmp_path / "native/actual/physics_steps.jsonl"
    rows = list(oracle.rows(census))
    rows[1199]["vehicles"]["a"]["lane_position_m"] += 1
    census.write_text("".join(json.dumps(r) + "\n" for r in rows))
    accept("native")
    assert oracle.verify_run(tmp_path / "native/actual", reg["jobs"]["native"]["expected"])[
        "integrity_pass"
    ]
    assert not runner._review_passed(reg, "pi-external-u")
