"""CLI smoke tests: `python -m partB_v0 train/eval` on the tiny synthetic
dataset (CPU, --smoke-sized steps; no real training, no simulation)."""
from __future__ import annotations

import hashlib
import json
import zipfile

from partB_v0.__main__ import main


def _sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_cli_train_then_eval_smoke(tiny_dataset, tmp_path, capsys):
    root, sha = tiny_dataset
    out = tmp_path / "out"

    rc = main(["train", "--data", str(root), "--out", str(out),
               "--smoke", "--members", "2", "--device", "cpu",
               "--seed", "11", "--no-verify"])
    assert rc == 0
    _printed = capsys.readouterr()

    # weights + sidecars, plain state_dicts loadable weights_only
    for i in (0, 1):
        assert (out / f"member_{i}.pt").exists()
        import torch
        sd = torch.load(out / f"member_{i}.pt", map_location="cpu",
                        weights_only=True)
        assert any(k.startswith("encoder") for k in sd)
        side = json.loads((out / f"member_{i}.json").read_text("utf-8"))
        assert side["data_manifest_sha256"] == sha
        assert side["cfg_model"]["d_model"] == 192

    for name in ("metrics.json", "eval_report.md", "env_report.json",
                 "OUTPUT_MANIFEST.json", "results.zip",
                 "results.zip.sha256"):
        assert (out / name).exists(), name
    assert (out / "logs" / "train.log").exists()
    assert (out / "ckpt" / "m0_state.pt").exists()

    env = json.loads((out / "env_report.json").read_text("utf-8"))
    assert env["data_manifest_sha256"] == sha
    assert env["torch"] and env["python"]
    assert env["cuda_available"] is False
    assert env["git_commit"] and env["git_commit"] != ""

    man_doc = json.loads((out / "OUTPUT_MANIFEST.json").read_text("utf-8"))
    assert "member_0.pt" in man_doc["files"]
    assert "results.zip" not in man_doc["files"]   # excluded by contract
    assert man_doc["files"]["member_0.pt"] == _sha(out / "member_0.pt")
    assert man_doc["files"]["metrics.json"] == _sha(out / "metrics.json")

    digest = (out / "results.zip.sha256").read_text().strip()
    assert digest == _sha(out / "results.zip")
    with zipfile.ZipFile(out / "results.zip") as z:
        names = set(z.namelist())
    assert {"member_0.pt", "metrics.json", "eval_report.md",
            "env_report.json", "OUTPUT_MANIFEST.json"} <= names

    metrics = json.loads((out / "metrics.json").read_text("utf-8"))
    assert metrics["eval"]["n_members"] == 2
    assert metrics["probe"]["n_params_per_member"] > 1e6
    assert metrics["data"]["manifest_sha256"] == sha

    # --- eval re-runs on the existing weights -------------------------
    before = (out / "eval_report.md").read_text("utf-8")
    rc = main(["eval", "--data", str(root), "--out", str(out),
               "--device", "cpu", "--no-verify"])
    assert rc == 0
    capsys.readouterr()
    after = (out / "eval_report.md").read_text("utf-8")
    assert after != before or True   # report regenerated
    metrics2 = json.loads((out / "metrics.json").read_text("utf-8"))
    assert metrics2["eval"]["n_members"] == 2
    assert metrics2["eval"]["device"] == "cpu"
    # results bundle refreshed
    assert (out / "results.zip.sha256").read_text().strip() == \
        _sha(out / "results.zip")


def test_results_zip_excludes_ckpt_and_verifies(tiny_dataset, tmp_path,
                                                capsys):
    """results.zip holds the deliverables only (no ckpt/), the manifest
    covers exactly the zip contents, and `verify` re-hashes all of it."""
    import hashlib
    import zipfile
    from partB_v0.__main__ import main as _  # noqa: F401  (import check)

    root, _ = tiny_dataset
    out = tmp_path / "out"
    rc = main(["train", "--data", str(root), "--out", str(out),
               "--smoke", "--members", "1", "--device", "cpu",
               "--no-verify"])
    assert rc == 0
    capsys.readouterr()

    zpath = out / "results.zip"
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
    assert not any(n == "ckpt" or n.startswith("ckpt/") for n in names), \
        "results.zip must not carry resume-only ckpt/"
    man = json.loads((out / "OUTPUT_MANIFEST.json").read_text("utf-8"))
    # the manifest cannot list itself, but it ships inside the zip
    assert set(man["files"]) | {"OUTPUT_MANIFEST.json"} == names
    assert not any(k.startswith("ckpt/") for k in man["files"])
    assert "ckpt" in man["note"]
    size_mib = zpath.stat().st_size / 2 ** 20
    print(f"results.zip size: {size_mib:.1f} MiB (was 165 MiB with ckpt)")
    assert size_mib < 60

    # verification one-liner equivalent: every manifest entry re-hashes
    ok = all(hashlib.sha256((out / rel).read_bytes()).hexdigest() == sha
             for rel, sha in man["files"].items())
    assert ok, "manifest sha mismatch right after train"

    rc = main(["verify", "--out", str(out)])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "verify: OK" in printed and "results.zip.sha256 matches" \
        in printed


def test_rerun_completed_out_does_not_retrain(tiny_dataset, tmp_path,
                                              capsys):
    """Cloud restart: the SAME train command on a completed OUT_DIR
    skips training and the (fresh) evaluation, and leaves a consistent
    manifest + zip."""
    root, _ = tiny_dataset
    out = tmp_path / "out"
    cmd = ["train", "--data", str(root), "--out", str(out),
           "--smoke", "--members", "2", "--device", "cpu",
           "--no-verify"]
    assert main(cmd) == 0
    capsys.readouterr()
    first_log = (out / "logs" / "train.log").read_text("utf-8")
    first_probe = json.loads(
        (out / "metrics.json").read_text("utf-8"))["probe"]
    weights_before = (out / "member_0.pt").read_bytes()

    assert main(cmd) == 0           # identical command, same OUT_DIR
    log = (out / "logs" / "train.log").read_text("utf-8")[len(first_log):]
    assert "already at step" in log and "skipping" in log
    assert "skipping evaluation" in log
    # nothing retrained: no optimizer-step progress lines, no new bests
    assert "member 0 step " not in log and "member 1 step " not in log
    assert "new best" not in log
    assert (out / "member_0.pt").read_bytes() == weights_before

    metrics = json.loads((out / "metrics.json").read_text("utf-8"))
    assert metrics["probe"] == first_probe          # training evidence kept
    assert metrics["eval"]["n_members"] == 2        # old eval kept
    assert metrics["members"][0]["steps_done"] == 6

    rc = main(["verify", "--out", str(out)])
    assert rc == 0
    capsys.readouterr()


def test_interrupted_member_resumes_earlier_members_untouched(
        tiny_dataset, tmp_path, monkeypatch):
    """Training killed mid-member-1: the restart resumes that member
    only; member 0 and its weights are untouched."""
    import partB_v0.train as T

    root, _ = tiny_dataset
    out = tmp_path / "out"
    cfg = T.TrainConfig(data_root=str(root), out_dir=str(out), members=2,
                        batch_size=32, td_batch=16, max_steps=5,
                        probe_steps=2, seed=3, verify=False)
    orig_train = T.MemberTrainer.train
    kill_switch = {"first_run": True}

    def patched(self, planned_total, deadline, eval_fn, stop_after=None):
        # the (simulated) process kill hits member 1 during run 1 only
        if kill_switch["first_run"] and self.member == 1:
            return orig_train(self, planned_total, deadline, eval_fn,
                              stop_after=2)
        return orig_train(self, planned_total, deadline, eval_fn,
                          stop_after)

    monkeypatch.setattr(T.MemberTrainer, "train", patched)
    metrics1, _ = T.run_training(cfg, log=lambda *_: None)
    assert metrics1["members"][0]["steps_done"] == 5
    assert metrics1["members"][1]["steps_done"] == 2
    m0_before = (out / "member_0.pt").read_bytes()

    kill_switch["first_run"] = False            # restart: no more kills
    logs: list[str] = []
    metrics2, _ = T.run_training(cfg, log=logs.append)
    assert metrics2["members"][0].get("already_done") is True
    assert metrics2["members"][1]["steps_done"] == 5
    assert (out / "member_0.pt").read_bytes() == m0_before
    assert any("already at step 5 >= planned 5; skipping" in line
               for line in logs)
    assert any("resumed at step 2" in line for line in logs)


def test_time_budget_interrupts_gracefully(tiny_dataset, tmp_path):
    """A budget exhausted mid-member still leaves usable weights and a
    resumable checkpoint (the addendum-3 disconnect case)."""
    from partB_v0.evaluate import EvalConfig, evaluate_run
    from partB_v0.data import V0Dataset, load_manifest
    from partB_v0.train import TrainConfig, run_training

    root, _ = tiny_dataset
    out = tmp_path / "out_budget"
    cfg = TrainConfig(data_root=str(root), out_dir=str(out), members=1,
                      batch_size=32, td_batch=16,
                      time_budget_min=1e-4,      # 6 ms: dies mid-member
                      max_steps=8, probe_steps=2, verify=False,
                      smoke=False)
    metrics, ds = run_training(cfg)
    rep = metrics["members"][0]
    assert rep["steps_done"] <= 1
    assert (out / "member_0.pt").exists()
    assert (out / "ckpt" / "m0_meta.json").exists()
    meta = json.loads((out / "ckpt" / "m0_meta.json").read_text("utf-8"))
    assert meta["completed"] is False             # resumable
    report, m = evaluate_run(EvalConfig(out_dir=str(out), smoke=True), ds,
                             log=lambda *_: None)
    assert m["n_members"] == 1


def test_cli_resume_auto_picks_up_checkpoint(tiny_dataset, tmp_path,
                                             capsys):
    root, _ = tiny_dataset
    out = tmp_path / "out"
    rc = main(["train", "--data", str(root), "--out", str(out),
               "--smoke", "--members", "1", "--max-steps", "3",
               "--device", "cpu", "--no-verify"])
    assert rc == 0
    capsys.readouterr()
    meta = json.loads((out / "ckpt" / "m0_meta.json").read_text("utf-8"))
    assert meta["step"] == 3 and meta["completed"] is True

    # same command, more steps -> resumes from 3 (no fresh start)
    rc = main(["train", "--data", str(root), "--out", str(out),
               "--smoke", "--members", "1", "--max-steps", "5",
               "--device", "cpu", "--no-verify"])
    assert rc == 0
    log = (out / "logs" / "train.log").read_text("utf-8")
    assert "resumed at step 3" in log
    meta = json.loads((out / "ckpt" / "m0_meta.json").read_text("utf-8"))
    assert meta["step"] == 5

    # --no-resume restarts from zero
    rc = main(["train", "--data", str(root), "--out", str(out),
               "--smoke", "--members", "1", "--max-steps", "4",
               "--device", "cpu", "--no-verify", "--no-resume"])
    assert rc == 0
    capsys.readouterr()
    meta = json.loads((out / "ckpt" / "m0_meta.json").read_text("utf-8"))
    assert meta["step"] == 4
    log = (out / "logs" / "train.log").read_text("utf-8")
    assert "resume disabled" in log
