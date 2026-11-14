"""Queue tests: stage order/STOP, sealed refusal, ordering, resume, fakes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bundle import approval as approval_mod
from bundle import queue


class Fakes:
    """In-memory corpus/train/timing/approval collaborators (no real work)."""

    def __init__(self, monkeypatch):
        self.calls = []
        monkeypatch.setattr(queue.corpus_pack, "convert", self.convert)
        monkeypatch.setattr(queue.corpus_verify, "verify_pack", self.verify)
        monkeypatch.setattr(queue.train_cli, "record_seed", self.record_seed)
        monkeypatch.setattr(queue.train_cli, "main", self.train_main)
        monkeypatch.setattr(queue.timing_mod, "run_timing_gate", self.run_gate)
        monkeypatch.setattr(queue.approval_mod, "require_approval",
                            self.require)
        monkeypatch.setattr(queue.approval_mod, "require_a_m2_5",
                            self.require_a_m2_5)
        monkeypatch.setattr(queue, "TINY", True)
        self.verify_ok = True

    def convert(self, in_dir, out_dir, *, workers=4):
        self.calls.append(("convert", str(in_dir), str(out_dir)))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return {"packed": 1}

    def verify(self, pack_dir):
        self.calls.append(("verify", str(pack_dir)))
        return {"ok": self.verify_ok, "n_runs": 1, "errors": []}

    def record_seed(self, out_dir, member, seed):
        self.calls.append(("record_seed", int(member), int(seed)))

    def train_main(self, argv):
        member = int(argv[argv.index("--member") + 1])
        out = Path(argv[argv.index("--out") + 1])
        self.calls.append(("train", member))
        out.mkdir(parents=True, exist_ok=True)
        (out / f"member_{member}.pt").write_bytes(b"ckpt")
        return 0

    def run_gate(self, cfg, **kw):
        self.calls.append(("timing", kw.get("device")))
        return {"schema": "wm_n1_timing_gate/1", "passed": True,
                "measured_model": "model.net.WmN1Net",
                "extrapolation": {"total_hours": 1.0}}

    def require(self, path, *, stage):
        self.calls.append(("approval", stage))
        return {"owner": "o", "timing_gate_sha256": "a" * 64}

    def require_a_m2_5(self, path, *, m1_checkpoint_path, winning_config_path):
        self.calls.append(("a_m2_5", str(path)))
        return {"owner": "o", "stage_requested": "members-2-5",
                "m1_checkpoint_resolved": str(m1_checkpoint_path),
                "winning_config_resolved": str(winning_config_path)}

    def trained(self):
        return [c[1] for c in self.calls if c[0] == "train"]


def _manifest(tmp_path):
    up = tmp_path / "upload"
    up.mkdir(parents=True, exist_ok=True)
    payload = b"fake pack bytes"
    (up / "worldA__arm.npy").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    return {"root": str(up), "files": ["worldA__arm.npy"],
            "sha256": {"worldA__arm.npy": sha}}


def _seed_state(work_dir, stages, member1=False):
    work_dir = Path(work_dir)
    state = {"schema": "wm_n1_bundle_state/1", "completed": {}}
    for s in stages:
        state["completed"][s] = {"stage": s, "ok": True,
                                 "completed_utc": "seed"}
    (work_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    if member1:
        mdir = work_dir / "members"
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "member_1.pt").write_bytes(b"ckpt")
    return state


def test_stop_after_timing_gate_is_default(tmp_path, monkeypatch):
    fakes = Fakes(monkeypatch)
    res = queue.bundle_queue(_manifest(tmp_path), tmp_path,
                             tmp_path / "APPROVAL.json", device="cpu")
    assert res["stop_after"] == "timing-gate"
    assert res["stopped_after"] == "timing-gate"
    assert list(res["stages"]) == ["verify-upload", "convert+verify",
                                   "timing-gate"]
    assert fakes.trained() == []
    assert not any(c[0] == "approval" for c in fakes.calls)
    assert (tmp_path / "TIMING_GATE.json").is_file()
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert list(state["completed"]) == ["verify-upload", "convert+verify",
                                        "timing-gate"]


def test_verify_upload_writes_receipt_and_converts(tmp_path, monkeypatch):
    fakes = Fakes(monkeypatch)
    queue.bundle_queue(_manifest(tmp_path), tmp_path,
                       tmp_path / "APPROVAL.json", device="cpu")
    assert ("verify", str(tmp_path / "pack")) in fakes.calls
    assert ("convert", str(tmp_path / "upload"),
            str(tmp_path / "pack")) in fakes.calls


def test_sealed_shard_refusal_chunk5(tmp_path, monkeypatch):
    Fakes(monkeypatch)
    manifest = {"root": str(tmp_path), "files": ["chunk-5/worldX.npy"],
                "sha256": {}}
    with pytest.raises(queue.SealedShardRefusal):
        queue.run_stage("verify-upload", pack_manifest=manifest,
                        work_dir=tmp_path, device="cpu")


def test_sealed_shard_refusal_holdout(tmp_path, monkeypatch):
    Fakes(monkeypatch)
    manifest = {"root": str(tmp_path), "files": ["HOLDOUT__w/base.npy"],
                "sha256": {}}
    with pytest.raises(queue.SealedShardRefusal):
        queue.run_stage("verify-upload", pack_manifest=manifest,
                        work_dir=tmp_path, device="cpu")


def test_sealed_shard_refusal_declared(tmp_path, monkeypatch):
    Fakes(monkeypatch)
    manifest = {"root": str(tmp_path), "files": [], "sha256": {},
                "refused": ["chunk-5/x"]}
    with pytest.raises(queue.SealedShardRefusal):
        queue.run_stage("verify-upload", pack_manifest=manifest,
                        work_dir=tmp_path, device="cpu")


def test_verify_upload_rejects_bad_sha(tmp_path, monkeypatch):
    Fakes(monkeypatch)
    manifest = {"root": str(tmp_path), "files": ["a.npy"],
                "sha256": {"a.npy": "0" * 64}}
    (tmp_path / "a.npy").write_bytes(b"data")
    with pytest.raises(queue.BundleStageError):
        queue.run_stage("verify-upload", pack_manifest=manifest,
                        work_dir=tmp_path, device="cpu")


def test_out_of_order_stage_refused(tmp_path, monkeypatch):
    Fakes(monkeypatch)
    with pytest.raises(queue.BundleOrderError):
        queue.run_stage("member-1", pack_manifest=_manifest(tmp_path),
                        work_dir=tmp_path,
                        approval_path=tmp_path / "APPROVAL.json",
                        device="cpu")


def test_members_2_5_refused_without_member_1_checkpoint(tmp_path, monkeypatch):
    Fakes(monkeypatch)
    _seed_state(tmp_path, ["verify-upload", "convert+verify", "timing-gate",
                           "approval"])
    with pytest.raises(queue.BundleOrderError):
        queue.run_stage("members-2-5", pack_manifest=_manifest(tmp_path),
                        work_dir=tmp_path,
                        approval_path=tmp_path / "APPROVAL.json",
                        device="cpu")


def test_member_1_first_then_members_2_5(tmp_path, monkeypatch):
    fakes = Fakes(monkeypatch)
    manifest = _manifest(tmp_path)
    appr = tmp_path / "APPROVAL.json"
    queue.bundle_queue(manifest, tmp_path, appr, device="cpu",
                       stop_after="member-1")
    assert fakes.trained() == [1]
    queue.bundle_queue(manifest, tmp_path, appr, device="cpu",
                       stop_after="members-2-5")
    trained = fakes.trained()
    assert trained == [1, 2, 3, 4, 5]
    assert trained.index(1) < trained.index(2)
    seeds = [c[2] for c in fakes.calls if c[0] == "record_seed"]
    assert seeds == [101, 202, 303, 404, 505]


def test_stop_after_member_1_is_m1_only(tmp_path, monkeypatch):
    fakes = Fakes(monkeypatch)
    manifest = _manifest(tmp_path)
    res = queue.bundle_queue(manifest, tmp_path, tmp_path / "APPROVAL.json",
                             device="cpu", stop_after="member-1")
    assert res["stopped_after"] == "member-1"
    assert fakes.trained() == [1]                 # members 2-5 NOT run
    assert not any(c[0] == "a_m2_5" for c in fakes.calls)


def test_members_2_5_requires_separate_a_m2_5_gate(tmp_path, monkeypatch):
    fakes = Fakes(monkeypatch)
    _seed_state(tmp_path, ["verify-upload", "convert+verify", "timing-gate",
                           "approval", "member-1"], member1=True)

    def refuse(path, *, m1_checkpoint_path, winning_config_path):
        raise approval_mod.ApprovalMissing("APPROVAL_A_M2_5.json absent")

    monkeypatch.setattr(queue.approval_mod, "require_a_m2_5", refuse)
    with pytest.raises(approval_mod.ApprovalMissing):
        queue.run_stage("members-2-5", pack_manifest=_manifest(tmp_path),
                        work_dir=tmp_path,
                        approval_path=tmp_path / "APPROVAL.json", device="cpu")
    assert fakes.trained() == []                  # no member 2 before the gate


def test_members_2_5_passes_the_a_m2_5_path(tmp_path, monkeypatch):
    fakes = Fakes(monkeypatch)
    _seed_state(tmp_path, ["verify-upload", "convert+verify", "timing-gate",
                           "approval", "member-1"], member1=True)
    a_m2_5 = tmp_path / "owner" / "APPROVAL_A_M2_5.json"
    win = tmp_path / "owner" / "WINNING_CONFIG.json"
    queue.run_stage("members-2-5", pack_manifest=_manifest(tmp_path),
                    work_dir=tmp_path, approval_path=tmp_path / "APPROVAL.json",
                    device="cpu", a_m2_5_approval=a_m2_5, winning_config=win)
    assert fakes.trained() == [2, 3, 4, 5]
    assert ("a_m2_5", str(a_m2_5)) in fakes.calls


def test_resume_skips_completed_stages(tmp_path, monkeypatch):
    fakes = Fakes(monkeypatch)
    manifest = _manifest(tmp_path)
    appr = tmp_path / "APPROVAL.json"
    queue.bundle_queue(manifest, tmp_path, appr, device="cpu",
                       stop_after="member-1")
    n_before = len(fakes.calls)
    queue.bundle_queue(manifest, tmp_path, appr, device="cpu",
                       stop_after="member-1")
    assert len(fakes.calls) == n_before          # nothing re-run
    assert fakes.trained() == [1]


def test_convert_failure_stops_and_records_nothing(tmp_path, monkeypatch):
    fakes = Fakes(monkeypatch)
    fakes.verify_ok = False
    with pytest.raises(queue.BundleStageError):
        queue.bundle_queue(_manifest(tmp_path), tmp_path,
                           tmp_path / "APPROVAL.json", device="cpu",
                           stop_after="convert+verify")
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert "convert+verify" not in state["completed"]


def test_approval_refusal_stops_before_member_1(tmp_path, monkeypatch):
    fakes = Fakes(monkeypatch)

    def refuse(path, *, stage):
        raise approval_mod.ApprovalMissing("no owner approval")

    monkeypatch.setattr(queue.approval_mod, "require_approval", refuse)
    manifest = _manifest(tmp_path)
    with pytest.raises(approval_mod.ApprovalMissing):
        queue.bundle_queue(manifest, tmp_path, tmp_path / "APPROVAL.json",
                           device="cpu", stop_after="member-1")
    assert fakes.trained() == []
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert "approval" not in state["completed"]
