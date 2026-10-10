"""Production member-1 launch contract at the CLI -> run-box-config ->
``run_config`` -> ``bundle.e2e.run_config_job`` seam.

CPU only, ``WD_FORBID_D=1``; the corpus, trainer and bundle are MOCKED (no real
pack, no real training, no CUDA).  The tests pin the production semantics the
old 2-step CPU smoke silently violated: the FULL registered schedule, the
registered member seed, CUDA, the real pack path, fail-closed refusal without
the pack or CUDA, and no member-2..5 / held-out / N / CAL auto-launch.

The frozen-record side is SELF-CONTAINED SYNTHETIC: the eight record fixtures
are written under ``pytest`` ``tmp_path`` from in-file synthetic literals (no
real REGISTRATION/amendment/CAL_SPLIT_A4 file is ever opened, read or copied),
and the expected LF-sha256 bindings in ``bundle.multibox`` plus the A6 freeze
in ``evalh.seals`` are pinned (monkeypatched) to the synthetic values for the
duration of each node.  The production ``require_config_approval`` validator
and the production ``run_config_job`` A6 gate run UNMOCKED against those
fixtures, so a wrong binding or a tampered fixture byte still fails closed.

The state writes never touch ``os.replace`` (denied by the guarded runtime):
an autouse fixture swaps ONLY ``bundle.multibox._write_json`` for an
exclusive fresh-target writer, identical ``json.dumps(indent=1)`` UTF-8
bytes and ``Path`` return, ``O_CREAT|O_EXCL`` creation with no temporary
file, no rename and no delete, so a duplicate write fails rather than
overwrites.  ``run_box_config``, the approval validator, ``_mark_completed``
and the real state reads/persistence stay production.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from bundle import e2e
from bundle import multibox as MB
from evalh import seals as _seals

BOX_CONFIGS = Path(MB.__file__).resolve().parent / "box_configs"
BOX_A = BOX_CONFIGS / "2box" / "c_m1only" / "boxA.json"


# ------------------------------------------------- synthetic record fixtures -
#: In-file synthetic literals for the eight frozen-record slots.  No real
#: registration/amendment/CAL bytes are ever read; each fixture's LF-sha256 is
#: computed from the literal itself and pinned into the expected bindings.
_RECORD_LITERALS = {
    "registration": b"SYNTHETIC TEST FIXTURE registration record\n",
    "amendment_1": b"SYNTHETIC TEST FIXTURE amendment 1 record\n",
    "amendment_2": b"SYNTHETIC TEST FIXTURE amendment 2 record\n",
    "amendment_3": b"SYNTHETIC TEST FIXTURE amendment 3 record\n",
    "amendment_4": b"SYNTHETIC TEST FIXTURE amendment 4 record\n",
    "amendment_5": b"SYNTHETIC TEST FIXTURE amendment 5 record\n",
    "amendment_6": b"SYNTHETIC TEST FIXTURE amendment 6 state-token record\n",
    "cal_split": b'{"synthetic_cal_split_fixture": true}\n',
}

#: frozen-record key -> the ``bundle.multibox`` binding attribute it pins.
_BINDING_ATTRS = {
    "registration": "LF_SHA_REGISTRATION",
    "amendment_1": "LF_SHA_AMENDMENT_1",
    "amendment_2": "LF_SHA_AMENDMENT_2",
    "amendment_3": "LF_SHA_AMENDMENT_3",
    "amendment_4": "LF_SHA_AMENDMENT_4",
    "amendment_5": "LF_SHA_AMENDMENT_5",
    "amendment_6": "LF_SHA_AMENDMENT_6",
    "cal_split": "LF_SHA_CAL_SPLIT_A4",
}


def _lf_sha256(data: bytes) -> str:
    return hashlib.sha256(
        data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _synthetic_records(tmp_path):
    """Synthetic frozen-record fixtures under ``tmp_path`` (literals only)."""
    rec_dir = tmp_path / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for key, literal in _RECORD_LITERALS.items():
        p = rec_dir / f"{key}.md"
        p.write_bytes(literal)
        records[key] = (p, _lf_sha256(literal))
    assert set(records) == set(MB.DEFAULT_RECORDS)     # full key coverage
    return records


def _pin_frozen_bindings(monkeypatch, records):
    """Pin the expected LF-sha bindings to the synthetic fixture values.

    Covers every ``bundle.multibox`` binding constant (approval writer fields
    AND the production validator's ``declared`` table) and the A6 freeze used
    by ``evalh.seals.require_frozen_amendment_6`` inside ``run_config_job``,
    so no node opens a real frozen record.
    """
    for key, attr in _BINDING_ATTRS.items():
        monkeypatch.setattr(MB, attr, records[key][1])
    a6_path, a6_sha = records["amendment_6"]
    monkeypatch.setattr(_seals, "AMENDMENT_6_PATH", a6_path)
    monkeypatch.setattr(_seals, "LF_SHA_AMENDMENT_6", a6_sha)


# ------------------------------------------------ exclusive state writes -----
@pytest.fixture(autouse=True)
def _exclusive_write_json(monkeypatch):
    """Swap ONLY ``MB._write_json`` for an exclusive fresh-target writer.

    Same bytes (``json.dumps(payload, indent=1)`` UTF-8) and same ``Path``
    return as the production writer, but the NEW target is created with
    ``O_CREAT|O_EXCL``: no temporary file, no rename and no delete (all
    denied by the guarded runtime), and a duplicate write to an existing
    target fails instead of overwriting it.  Every node runs fresh
    ``tmp_path`` work with one member per test, so every target write is a
    first write.  Everything else in ``bundle.multibox`` stays production.
    """

    def _write_json_exclusive(path, payload: dict) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(payload, indent=1).encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return path

    monkeypatch.setattr(MB, "_write_json", _write_json_exclusive)


# ----------------------------------------------------------------- fakes -----
class FakeCorpus:
    last = None

    def __init__(self, pack_dir, split="TRAIN", *, pair_sampling=True,
                 horizon_steps=None, k_max=None):
        self.pack_dir = str(pack_dir)
        self.split = split
        self.pair_sampling = pair_sampling
        self.horizon_steps = horizon_steps
        self.k_max = k_max
        FakeCorpus.last = self


class FakeTrainer:
    last = None
    force_steps = None

    def __init__(self, config, corpus, *, member=None, seed=None, device=None,
                 bundle=None):
        self.config = config
        self.corpus = corpus
        self.member = int(member if member is not None else config.member)
        self.seed = int(seed)
        self.device = str(device)
        self.bundle = bundle
        FakeTrainer.last = self

    def fit(self):
        out = Path(self.config.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ckpt = out / f"member_{self.member}.pt"
        ckpt.write_bytes(b"fake-ckpt")
        steps = (int(self.force_steps) if self.force_steps is not None
                 else int(self.config.epochs) * int(self.config.steps_per_epoch))
        return {"member": self.member, "seed": self.seed, "steps": steps,
                "epochs": int(self.config.epochs), "checkpoint": str(ckpt),
                "state_global_token": bool(self.config.state_global_token)}


def _mock_launch(monkeypatch, *, cuda=True):
    import torch
    from train import idea_a as ia

    FakeTrainer.last = None
    FakeTrainer.force_steps = None
    FakeCorpus.last = None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: bool(cuda))
    monkeypatch.setattr(ia, "IdeaACorpus", FakeCorpus)
    monkeypatch.setattr(ia, "IdeaATrainer", FakeTrainer)
    monkeypatch.setattr(ia, "build_idea_a_bundle", lambda cfg: object())


def _write_pack(root, *, k_max=192, fmt="v3"):
    pack = Path(root) / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "INDEX.json").write_text(
        json.dumps({"schema": "wm_n1_corpus_pack/2", "k_max": k_max,
                    "format": fmt}), encoding="utf-8")
    (pack / "manifest.json").write_text(
        json.dumps({"schema": "wm_n1_pack/1", "k_max": k_max,
                    "format": fmt}), encoding="utf-8")
    return pack


def _approved(tmp_path, config, records, **overrides):
    """Synthetic owner approval + timing gate binding the pinned fixtures.

    Emits the same binding fields the production ``write_config_approval``
    writes; the values are read from the (pinned) ``bundle.multibox``
    constants at call time.  The real ``require_config_approval`` validator
    still enforces every field and re-hashes the fixture bytes, so a wrong
    binding or a tampered fixture fails closed exactly as in production.
    """
    d = tmp_path / config
    d.mkdir(parents=True, exist_ok=True)
    gate = d / MB.TIMING_GATE_FILENAME
    gate.write_bytes(b'{"gate": 1}\n')
    body = {
        "schema": MB.APPROVAL_SCHEMA,
        "approved": True,
        "owner": "owner",
        "config": config,
        "timing_gate_sha256": hashlib.sha256(
            gate.read_bytes()).hexdigest(),
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
    appr = d / MB.APPROVAL_FILENAME
    appr.write_text(json.dumps(body), encoding="utf-8")
    return appr, gate


def _run_box_a(tmp_path, records):
    appr, gate = _approved(tmp_path, "member-1", records)
    return MB.run_box_config(
        BOX_A, work_dir=tmp_path, executor=None,
        approval_paths={"member-1": appr},
        timing_gate_paths={"member-1": gate}, records=records)


# --------------------------------------------------- full production run -----
def test_cli_seam_runs_full_registered_member_1(tmp_path, monkeypatch):
    records = _synthetic_records(tmp_path)
    _pin_frozen_bindings(monkeypatch, records)
    _write_pack(tmp_path)
    _mock_launch(monkeypatch, cuda=True)

    out = _run_box_a(tmp_path, records)

    assert out["configs"] == ["member-1"]
    assert out["stop_after"] == "member-1"
    receipt = out["receipts"]["member-1"]
    assert receipt["config"] == "member-1" and receipt["member"] == 1
    assert receipt["device"] == "cuda"
    assert receipt["steps"] == 30000              # 15 x 2000, NOT 2
    assert receipt["expected_steps"] == 30000
    assert receipt["k_max"] == 192 and receipt["format"] == "v3"
    # owner 2.3: pack_source is the EXACT supplied pack path; the kind is kept
    # separately (the old literal "real-pack" lost the actual location).
    assert receipt["pack_source"] == str(tmp_path / "pack")
    assert receipt["source_kind"] == "real-pack"
    assert receipt["seed"] == 101
    assert receipt["state_global_token"] is True

    trainer = FakeTrainer.last
    cfg = trainer.config
    assert cfg.epochs == 15 and cfg.steps_per_epoch == 2000
    assert cfg.batch_size == 8 and cfg.ensemble_k == 5
    assert cfg.member == 1 and cfg.num_workers is None      # host-scaled
    assert cfg.state_global_token is True and cfg.k_max == 192
    assert cfg.out_dir == str(tmp_path / "members" / "member_1")
    assert trainer.device == "cuda" and trainer.seed == 101
    assert FakeCorpus.last.pack_dir == str(tmp_path / "pack")
    assert FakeCorpus.last.k_max == 192 and FakeCorpus.last.split == "TRAIN"

    ckpt = tmp_path / "members" / "member_1" / "member_1.pt"
    assert ckpt.is_file()
    assert receipt["checkpoint"] == str(ckpt)

    # Real state persistence (production save/load path, exclusive writer):
    # the box state file holds the exact member-1 receipt with the production
    # schema, the on-disk bytes are the retained json.dumps(indent=1) UTF-8
    # contract, and a duplicate write to the fresh target is REFUSED instead
    # of overwriting (O_CREAT|O_EXCL), leaving the original bytes intact.
    state_path = tmp_path / "boxes" / "box-A" / "state.json"
    raw = state_path.read_bytes()
    state = json.loads(raw.decode("utf-8"))
    assert state["schema"] == "wm_n1_multibox_box_state/1"
    assert state["box_id"] == "box-A"
    assert set(state["completed"]) == {"member-1"}
    assert (MB.load_box_state(tmp_path, "box-A")["completed"]["member-1"]
            == receipt)
    assert (MB.load_fleet_state(tmp_path)["completed"]["member-1"]["box_id"]
            == "box-A")
    assert raw == json.dumps(state, indent=1).encode("utf-8")
    with pytest.raises(FileExistsError):
        MB.save_box_state(tmp_path, "box-A", dict(state))
    assert state_path.read_bytes() == raw

    # Tamper fail-closed over the synthetic fixtures only: appending bytes to
    # a supplied record must invalidate the approval.  The tamper approval is
    # written under a FRESH parent with the SAME config name ("member-1"), so
    # the validator clears the config-identity check and the ONLY possible
    # refusal is the LF-sha256 re-hash of the tampered bytes; the assertion
    # pins that exact message, proving the pinned fixtures did not bypass the
    # production tamper gate (review 86ebb66).
    cal_path = records["cal_split"][0]
    cal_path.write_bytes(cal_path.read_bytes() + b"tampered\n")
    tamper_parent = tmp_path / "tamper"
    tamper_appr, tamper_gate = _approved(tamper_parent, "member-1", records)
    with pytest.raises(MB.ConfigApprovalMissing) as ei:
        MB.require_config_approval(tamper_appr, config="member-1",
                                   timing_gate_path=tamper_gate,
                                   records=records)
    assert "cal_split.md" in str(ei.value) and "LF-sha256" in str(ei.value)


def test_member_1_ends_after_checkpoint_no_auto_launch(tmp_path, monkeypatch):
    records = _synthetic_records(tmp_path)
    _pin_frozen_bindings(monkeypatch, records)
    _write_pack(tmp_path)
    _mock_launch(monkeypatch, cuda=True)

    _run_box_a(tmp_path, records)

    members = sorted(p.name for p in (tmp_path / "members").glob("member_*"))
    assert members == ["member_1"]                 # no members 2-5
    assert set(MB.load_box_state(tmp_path, "box-A")["completed"]) == {
        "member-1"}                                # persisted, not re-run
    for absent in ("eval", "score", "cal", "heldout-i-j4b"):
        assert not (tmp_path / absent).exists()
    assert not (tmp_path / "boxes" / "box-A" / "eval").exists()


# --------------------------------------------------------- fail closed -------
def test_refuses_without_real_pack(tmp_path, monkeypatch):
    records = _synthetic_records(tmp_path)
    _pin_frozen_bindings(monkeypatch, records)
    _mock_launch(monkeypatch, cuda=True)          # no pack written

    with pytest.raises(e2e.ProductionLaunchError) as ei:
        _run_box_a(tmp_path, records)
    assert "pack" in str(ei.value)
    assert FakeTrainer.last is None and FakeCorpus.last is None


def test_refuses_when_cuda_absent(tmp_path, monkeypatch):
    records = _synthetic_records(tmp_path)
    _pin_frozen_bindings(monkeypatch, records)
    _write_pack(tmp_path)
    _mock_launch(monkeypatch, cuda=False)

    with pytest.raises(e2e.ProductionLaunchError) as ei:
        _run_box_a(tmp_path, records)
    assert "CUDA" in str(ei.value)
    assert FakeTrainer.last is None and FakeCorpus.last is None
    assert not (tmp_path / "members" / "member_1" / "member_1.pt").exists()


def test_refuses_short_schedule(tmp_path, monkeypatch):
    records = _synthetic_records(tmp_path)
    _pin_frozen_bindings(monkeypatch, records)
    _write_pack(tmp_path)
    _mock_launch(monkeypatch, cuda=True)
    FakeTrainer.force_steps = 2                    # the retired smoke schedule

    with pytest.raises(e2e.ProductionLaunchError) as ei:
        _run_box_a(tmp_path, records)
    assert "expected the full registered schedule" in str(ei.value)


def test_refuses_unregistered_pack_k(tmp_path, monkeypatch):
    records = _synthetic_records(tmp_path)
    _pin_frozen_bindings(monkeypatch, records)
    _write_pack(tmp_path, k_max=64, fmt=None)
    _mock_launch(monkeypatch, cuda=True)

    with pytest.raises(e2e.ProductionLaunchError) as ei:
        _run_box_a(tmp_path, records)
    assert "K_max" in str(ei.value)
    assert FakeTrainer.last is None


def test_a6_refused_before_pack_or_cuda(tmp_path, monkeypatch):
    records = _synthetic_records(tmp_path)
    _pin_frozen_bindings(monkeypatch, records)

    _mock_launch(monkeypatch, cuda=False)          # also no pack, no CUDA
    monkeypatch.setattr(_seals, "AMENDMENT_6_PATH",
                        tmp_path / "missing_a6.md")

    with pytest.raises(_seals.Amendment6Error):
        _run_box_a(tmp_path, records)
    assert FakeTrainer.last is None and FakeCorpus.last is None


# ------------------------------------------------- held-out / N refusal ------
def test_refuses_heldout_and_n_launch(tmp_path, monkeypatch):
    _mock_launch(monkeypatch, cuda=True)
    for config in ("heldout-i-j4b", "n-member-1"):
        with pytest.raises(e2e.ProductionConfigRefusal):
            e2e.run_config_job("box-x", config, tmp_path)


def test_run_e2e_smoke_entry_point_preserved():
    assert callable(e2e.run_e2e_smoke)
    assert e2e.run_e2e_smoke.__module__ == "bundle.e2e"


# ------------------------------------------- factory-RNG seed determinism ----
def test_bundle_factory_binds_registered_seed_before_trainer(
        tmp_path, monkeypatch):
    """The production bundle factory consumes the REGISTERED seed, not the
    ambient torch RNG state.

    A synthetic factory stands in for ``build_idea_a_bundle`` at the real
    ``run_config_job`` seam and draws from the global torch RNG exactly where
    the real network initialisation does (the factory runs as a kwarg
    expression BEFORE the trainer body can re-seed).  Two production launches
    whose ambient torch RNG was deliberately scrambled to DIFFERENT unrelated
    states must record identical draws (registered member-1 seed 101); the
    draws must equal a fresh generator seeded with exactly the registered
    seed; the factory's bundle must reach the trainer; and a member-2 launch
    (registered seed 202) from the SAME ambient state as launch 2 must draw
    differently - so the equality is the registered-seed binding, not a
    constant.  The fake-CUDA ``cuda_available`` bypass is exercised on
    purpose: it skips the hardware-check torch import, so only an
    UNCONDITIONAL import+seed before the factory can pass this node.
    """
    import torch
    from train import idea_a as ia

    records = _synthetic_records(tmp_path)
    _pin_frozen_bindings(monkeypatch, records)

    draws, markers = [], []

    def _factory_seeded_draws(cfg):
        # consume the global torch RNG exactly where real nn.Linear-style
        # weight initialisation consumes it
        draws.append([float(x) for x in torch.rand(8)])
        marker = object()
        markers.append(marker)
        return marker

    FakeTrainer.last = None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ia, "IdeaACorpus", FakeCorpus)
    monkeypatch.setattr(ia, "IdeaATrainer", FakeTrainer)
    monkeypatch.setattr(ia, "build_idea_a_bundle", _factory_seeded_draws)

    torch.manual_seed(77011)                       # unrelated ambient state A
    work_a = tmp_path / "runA"
    _write_pack(work_a)
    out_a = e2e.run_config_job("box-a", "member-1", work_a,
                               cuda_available=True)
    torch.manual_seed(313)                         # DIFFERENT ambient state B
    work_b = tmp_path / "runB"
    _write_pack(work_b)
    out_b = e2e.run_config_job("box-b", "member-1", work_b,
                               cuda_available=True)

    assert out_a["seed"] == 101 and out_b["seed"] == 101
    assert len(draws) == 2 and draws[0] == draws[1]  # ambient scrub is inert

    # The draws bind to the FROZEN registered member-1 seed 101: identical to
    # a fresh generator seeded with exactly that value (a fresh CPU generator
    # reproduces the global RNG's post-manual_seed sequence), and the factory
    # product reached the trainer BEFORE the trainer's own re-seed.
    gen = torch.Generator()
    gen.manual_seed(int(out_a["seed"]))
    assert draws[0] == [float(x) for x in torch.rand(8, generator=gen)]
    assert FakeTrainer.last.seed == 101
    assert FakeTrainer.last.bundle is markers[-1]

    # Sensitivity: the registered member-2 seed (202) draws differently from
    # the SAME ambient state as launch 2, proving the draws track the
    # registered seed rather than the ambient scrub or a constant.
    torch.manual_seed(313)
    work_c = tmp_path / "runC"
    _write_pack(work_c)
    out_c = e2e.run_config_job("box-c", "member-2", work_c,
                               cuda_available=True)
    assert out_c["seed"] == 202 and FakeTrainer.last.seed == 202
    assert len(draws) == 3 and draws[2] != draws[1]
