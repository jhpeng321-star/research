"""The launcher: every gate refuses loudly; a green launch dispatches 240."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from campaign_walls_v1 import launch_walls
from campaign_walls_v1 import plan_walls_pilot as plan_mod
from campaign_walls_v1.common import BRANCHES_TOTAL, package_code_digest


class _Bundle:
    """Plan + registration + gates + authorization + verdict, all tmp."""

    def __init__(self, tmp_path, source_bundle):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        self.pkg = pkg
        plan = plan_mod.build_plan(
            source_bundle["plans"]["chunk2"], source_bundle["plans"]["chunk3"],
            bases_roots=source_bundle["bases_roots"])
        self.plan = plan
        plan_dir = pkg / "results_plan"
        self.plan_path = plan_mod.write_plan(plan, plan_dir)
        # complete native sibling receipts (the launcher's plan gate does
        # not read them, but keep the fixture honest)
        for world in plan["worlds"]:
            receipt_dir = (Path(world["native_sibling"]["runs_root"])
                           / world["native_sibling"]["run"])
            receipt_dir.mkdir(parents=True, exist_ok=True)
            (receipt_dir / "RUN_RECEIPT.json").write_text(
                json.dumps({"exit_status": "complete"}), encoding="utf-8")
        self.registration = pkg / "WALLS_PILOT_REGISTRATION.md"
        self.registration.write_text("# registration\n", encoding="utf-8")
        self.registration_sha = hashlib.sha256(
            self.registration.read_bytes()).hexdigest()
        self.authorization = pkg / "AUTHORIZATION.json"
        self.authorization.write_text(json.dumps({
            "schema": "wm59_authorization/1",
            "registration_sha256": self.registration_sha,
            "authorized_by": "owner-test"}), encoding="utf-8")
        self.gates = {
            "wm41_verdict": str(tmp_path / "wm41" / "EXACTNESS_VERDICT.json"),
            "wm41_registration": str(tmp_path / "wm41" / "REGISTRATION.md"),
            "wm43_summary": str(tmp_path / "wm43" / "LAUNCH_SUMMARY.json"),
        }
        Path(self.gates["wm41_verdict"]).parent.mkdir(parents=True)
        Path(self.gates["wm41_registration"]).parent.mkdir(parents=True,
                                                           exist_ok=True)
        Path(self.gates["wm43_summary"]).parent.mkdir(parents=True,
                                                      exist_ok=True)
        Path(self.gates["wm41_verdict"]).write_text(
            json.dumps({"passed": True}), encoding="utf-8")
        Path(self.gates["wm41_registration"]).write_text("# pilot\n",
                                                         encoding="utf-8")
        Path(self.gates["wm43_summary"]).write_text(
            json.dumps({"complete": 15, "failed": 0}), encoding="utf-8")
        self.out_dir = tmp_path / "launch_out"
        self.runs_root = tmp_path / "runs" / "campaign_walls_pilot_v1"
        self.smoke_verdict = tmp_path / "smoke" / "SMOKE_VERDICT.json"
        self.smoke_verdict.parent.mkdir(parents=True)
        self.write_verdict()
        self.disk_free_bytes = 100 * 1024 ** 3

    def write_verdict(self, **overrides):
        payload = {
            "schema": "wm59_walls_pilot_smoke/1", "passed": True,
            "registration_sha256": self.registration_sha,
            "package_code_sha256": package_code_digest(self.pkg),
            "smoke_runs_root": "E:/x", "created_utc": "utc"}
        payload.update(overrides)
        self.smoke_verdict.write_text(json.dumps(payload), encoding="utf-8")

    def launch(self, *, workers=10, launcher=None, resume=False):
        return launch_walls.launch(
            self.plan_path, self.authorization, self.registration,
            self.out_dir, self.runs_root, self.smoke_verdict, workers,
            launcher=launcher or (lambda argv: 0),
            disk_free_fn=lambda p: self.disk_free_bytes,
            resume=resume, package_dir=self.pkg, gate_paths=self.gates)


@pytest.fixture()
def bundle(tmp_path, source_bundle):
    return _Bundle(tmp_path, source_bundle)


def _child_like_recorder(dispatched):
    """A launcher stand-in that also does the child's write-once receipt
    (so resume sees COMPLETE runs), returning exit code 0."""

    def recorder(argv):
        run = argv[argv.index("--run") + 1]
        root = Path(argv[argv.index("--runs-root") + 1])
        target = root / run
        target.mkdir(parents=True)
        (target / "RUN_RECEIPT.json").write_text(
            json.dumps({"schema": "wm59_wall_branch_receipt/1",
                        "exit_status": "complete"}), encoding="utf-8")
        dispatched.append(argv)
        return 0

    return recorder


def test_green_launch_dispatches_240_children_with_bases_roots(bundle):
    dispatched = []
    summary = bundle.launch(launcher=_child_like_recorder(dispatched))
    assert summary["complete"] == BRANCHES_TOTAL
    assert summary["failed"] == 0
    assert len(dispatched) == BRANCHES_TOTAL
    names = {argv[argv.index("--run") + 1] for argv in dispatched}
    assert names == {b["run"] for b in bundle.plan["runs"]}
    bases = {argv[argv.index("--run") + 1]:
             argv[argv.index("--bases-root") + 1] for argv in dispatched}
    for world in bundle.plan["worlds"]:
        for branch in world["branches"]:
            assert bases[branch["run"]] == world["bases_root"]
    written = json.loads((bundle.out_dir / "LAUNCH_SUMMARY.json")
                         .read_text(encoding="utf-8"))
    assert written["schema"] == "wm59_walls_pilot_launch_summary/1"
    assert written["smoke_gate"]["passed"] is True
    assert written["disk_preflight"]["min_free_gb"] == 45.0


def test_workers_must_be_exactly_ten(bundle):
    from campaign_walls_v1.common import WallsPilotError

    with pytest.raises(WallsPilotError, match="exactly the registered 10"):
        bundle.launch(workers=9)
    with pytest.raises(WallsPilotError, match="exactly the registered 10"):
        bundle.launch(workers=14)


def test_smoke_verdict_gates(bundle):
    from campaign_walls_v1.common import WallsPilotError

    bundle.smoke_verdict.unlink()
    with pytest.raises(WallsPilotError, match="no pre-launch smoke"):
        bundle.launch()
    bundle.write_verdict(passed=False)
    with pytest.raises(WallsPilotError, match="did NOT pass"):
        bundle.launch()
    bundle.write_verdict(registration_sha256="0" * 64)
    with pytest.raises(WallsPilotError, match="re-run the smoke"):
        bundle.launch()
    bundle.write_verdict(package_code_sha256="1" * 64)
    with pytest.raises(WallsPilotError, match="re-run the smoke"):
        bundle.launch()


def test_authorization_and_stage_gates_and_disk(bundle, tmp_path):
    from campaign_walls_v1.common import WallsPilotError

    good_authorization = json.dumps({
        "schema": "wm59_authorization/1",
        "registration_sha256": bundle.registration_sha,
        "authorized_by": "owner-test"})
    bundle.authorization.write_text(json.dumps({
        "schema": "other/1", "registration_sha256": bundle.registration_sha}),
        encoding="utf-8")
    with pytest.raises(WallsPilotError, match="schema"):
        bundle.launch()
    bundle.authorization.write_text(json.dumps({
        "schema": "wm59_authorization/1",
        "registration_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(WallsPilotError, match="authorization digest"):
        bundle.launch()
    bundle.authorization.write_text(good_authorization, encoding="utf-8")
    Path(bundle.gates["wm41_verdict"]).write_text(
        json.dumps({"passed": False}), encoding="utf-8")
    with pytest.raises(WallsPilotError, match="WM-41"):
        bundle.launch()
    Path(bundle.gates["wm41_verdict"]).write_text(
        json.dumps({"passed": True}), encoding="utf-8")
    Path(bundle.gates["wm43_summary"]).write_text(
        json.dumps({"complete": 10, "failed": 2}), encoding="utf-8")
    with pytest.raises(WallsPilotError, match="WM-43"):
        bundle.launch()
    Path(bundle.gates["wm43_summary"]).write_text(
        json.dumps({"complete": 15, "failed": 0}), encoding="utf-8")
    bundle.disk_free_bytes = 44 * 1024 ** 3
    with pytest.raises(WallsPilotError, match="44\\.0 GB free"):
        bundle.launch()


def test_existing_runs_root_and_summary_refuse_without_resume(bundle):
    from campaign_walls_v1.common import WallsPilotError

    bundle.runs_root.mkdir(parents=True)
    with pytest.raises(WallsPilotError, match="already exists"):
        bundle.launch()
    # --resume with an existing (empty) runs root proceeds
    summary = bundle.launch(resume=True)
    assert summary["failed"] == 0


def test_resume_keeps_complete_receipts_and_renames_interrupted(bundle):
    dispatched = []
    first = bundle.launch(launcher=_child_like_recorder(dispatched))
    assert first["complete"] == BRANCHES_TOTAL
    summary = bundle.launch(resume=True,
                            launcher=_child_like_recorder(dispatched))
    assert summary["resume"] is True
    assert summary["complete"] == BRANCHES_TOTAL
    statuses = {r["run"]: r["status"] for r in summary["runs"]}
    assert all(s == "resume_kept_complete" for s in statuses.values())
    assert len(dispatched) == BRANCHES_TOTAL     # nothing re-dispatched
    # an INCOMPLETE run dir (no receipt) is archive-renamed, never
    # overwritten, and re-dispatched
    stale = bundle.runs_root / bundle.plan["runs"][1]["run"]
    (stale / "RUN_RECEIPT.json").unlink()
    before = sorted(p.name for p in bundle.runs_root.iterdir())
    summary = bundle.launch(resume=True,
                            launcher=_child_like_recorder(dispatched))
    assert len(dispatched) == BRANCHES_TOTAL + 1
    archived = [n for n in sorted(p.name for p in bundle.runs_root.iterdir())
                if n not in before and "interrupted" in n]
    assert archived, "the interrupted dir was archived"


def test_plan_gate_refuses_wrong_shape(bundle, tmp_path):
    from campaign_walls_v1.common import WallsPilotError

    payload = json.loads(bundle.plan_path.read_text(encoding="utf-8"))
    payload["runs"] = payload["runs"][:-1]
    bad = tmp_path / "BAD_PLAN.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WallsPilotError, match="240"):
        launch_walls.launch(
            bad, bundle.authorization, bundle.registration, bundle.out_dir,
            bundle.runs_root, bundle.smoke_verdict, 10,
            launcher=lambda argv: 0,
            disk_free_fn=lambda p: bundle.disk_free_bytes,
            package_dir=bundle.pkg, gate_paths=bundle.gates)
