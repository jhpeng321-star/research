"""CLI `net` smoke test: `python -m partB_v0 net --smoke` end to end on
CPU -- concurrent members, OOB early stopping, D14 calibration, Delta
head, bundle, verify (numbers are smoke-only)."""
from __future__ import annotations

import json
import zipfile

from partB_v0.__main__ import main


def test_net_smoke_end_to_end_and_verify(tiny_dataset, tmp_path, capsys):
    root, sha = tiny_dataset
    out = tmp_path / "out"

    rc = main(["net", "--data", str(root), "--out", str(out),
               "--smoke", "--members", "3", "--device", "cpu",
               "--d-model", "64", "--seed", "11", "--no-verify"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "verify: OK" in printed

    for i in (0, 1, 2):
        assert (out / f"member_{i}.pt").exists()
        assert (out / f"member_{i}.json").exists()
    for name in ("stats.json", "calibration.json", "metrics.json",
                 "eval_report.md", "env_report.json",
                 "OUTPUT_MANIFEST.json", "results.zip",
                 "results.zip.sha256"):
        assert (out / name).exists(), name
    assert (out / "ckpt" / "net_state.pt").exists()

    metrics = json.loads((out / "metrics.json").read_text("utf-8"))
    assert metrics["format"] == "wd01_partB_v03/1"
    assert metrics["training"]["steps_done"] == 6
    assert metrics["data"]["manifest_sha256"] == sha
    assert metrics["calibration"]["method"] == "shift"

    ev = metrics["eval_v03"]
    assert ev["n_members"] == 3
    assert ev["item1_median_crps_coverage"], "item-1 rows missing"
    assert ev["item2_calibration"], "calibration table missing"
    assert ev["item5_delta_head"]["n_pairs"] > 0
    assert "sign_agreement" in ev["item5_delta_head"]

    report = (out / "eval_report.md").read_text("utf-8")
    for section in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.",
                    "## 6.", "SMOKE-ONLY"):
        assert section in report, section

    side = json.loads((out / "member_0.json").read_text("utf-8"))
    assert side["format"] == "wd01_partB_v03/1"
    assert side["cfg_model"]["d_model"] == 64

    # the zip whitelists the deliverables (no ckpt/)
    with zipfile.ZipFile(out / "results.zip") as z:
        names = set(z.namelist())
    assert not any(n.startswith("ckpt") for n in names)
    assert {"member_0.pt", "stats.json", "calibration.json",
            "eval_report.md", "metrics.json"} <= names

    # standalone verify subcommand on the same OUT_DIR
    rc = main(["verify", "--out", str(out)])
    assert rc == 0
    assert "verify: OK" in capsys.readouterr().out


def test_net_smoke_resume_picks_up_checkpoint(tiny_dataset, tmp_path,
                                              capsys):
    root, _ = tiny_dataset
    out = tmp_path / "out"
    common = ["net", "--data", str(root), "--out", str(out),
              "--smoke", "--members", "1", "--device", "cpu",
              "--d-model", "64", "--no-verify"]

    assert main(common + ["--max-steps", "3", "--no-eval"]) == 0
    capsys.readouterr()
    meta = json.loads((out / "ckpt" / "net_meta.json").read_text("utf-8"))
    assert meta["step"] == 3

    assert main(common + ["--max-steps", "5"]) == 0
    log = (out / "logs" / "net.log").read_text("utf-8")
    assert "resumed at lockstep step 3" in log
    metrics = json.loads((out / "metrics.json").read_text("utf-8"))
    assert metrics["training"]["steps_done"] == 5
