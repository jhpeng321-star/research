"""CLI dispatch: new-style commands, legacy argv, refusals (missing data fails closed)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from wms.cli import __main__ as cli
from wms.cli import fetch, train

ROOT = Path(__file__).resolve().parents[3]
LEGACY = ["train", "--pack", "P", "--identities", "I"]


def run_main(capsys, *argv):
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


def test_help_lists_the_five_commands(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for command in ("train", "preflight", "eval", "score", "fetch"):
        assert command in out


def test_python_dash_m_wms_runs():
    result = subprocess.run([sys.executable, "-m", "wms", "--help"], cwd=ROOT, capture_output=True,
                            text=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert result.returncode == 0 and "train" in result.stdout


def _without_data(monkeypatch, tmp_path):
    """Point the pack datasets at a missing directory so preflight cannot find any data."""
    for name in ("PACK_V3", "PACK_DSPEED", "PACK_DAMPING"):
        monkeypatch.setenv(f"WMS_DATA_{name}", str(tmp_path / "absent" / name.lower()))


def test_train_preflight_reports_missing_data_as_failures(capsys, tmp_path, monkeypatch):
    _without_data(monkeypatch, tmp_path)
    code, out = run_main(capsys, "train", "--config", "r3", "--out", str(tmp_path / "run"))
    payload = json.loads(out)
    assert code == cli.EXIT_REFUSED and payload["status"] == "failed"
    assert any(f.startswith("missing input") and "absent" in f for f in payload["failures"])
    assert payload["plan"]["stage_steps"] == [2500, 2500, 1875, 1250]


def test_preflight_command_equals_train_preflight(capsys, tmp_path, monkeypatch):
    _without_data(monkeypatch, tmp_path)
    first = run_main(capsys, "preflight", "--config", "r3", "--out", str(tmp_path / "run"))
    second = run_main(capsys, "train", "--config", "r3", "--out", str(tmp_path / "run"))
    assert first == second and first[0] == cli.EXIT_REFUSED


def test_preflight_unresolvable_dataset_is_a_failure_not_a_crash(capsys, tmp_path):
    code, out = run_main(capsys, "preflight", "--config", "r3", "--out", str(tmp_path / "run"),
                         "--set", 'data.windows.train="no_such_dataset/list.json"')
    payload = json.loads(out)
    assert code == cli.EXIT_REFUSED
    assert any("no_such_dataset" in f for f in payload["failures"])


def test_stage_mode_points_to_fetch(capsys):
    code, out = run_main(capsys, "train", "--config", "r3", "--mode", "stage")
    assert code == cli.EXIT_REFUSED and "wms fetch get" in json.loads(out)["error"]


def test_invalid_config_is_refused_with_json(capsys):
    code, out = run_main(capsys, "train", "--config", "r3", "--set", "optim.bogus=1")
    assert code == cli.EXIT_REFUSED and json.loads(out)["status"] == "refused"


def test_training_refusals_are_reported(capsys):
    code, out = run_main(capsys, "train", "--config", "c1fast")
    assert code == cli.EXIT_REFUSED and "num_workers" in json.loads(out)["error"]


def test_set_and_out_overrides_reach_the_config(tmp_path):
    parser = cli.build_parser()
    args = parser.parse_args(["train", "--config", "r3", "--set", "optim.lr=0.5",
                              "--out", str(tmp_path / "w")])
    cfg = train.config_from_args(args)
    assert cfg.optim.lr == 0.5 and Path(cfg.runtime.out_dir) == tmp_path / "w"


def test_nonempty_output_directory_is_refused(tmp_path):
    (tmp_path / "old.txt").write_text("keep", encoding="utf-8")
    cfg = train.load_train_config("r3", [f"runtime.out_dir={json.dumps(str(tmp_path))}"])
    with pytest.raises(Exception, match="not empty"):
        train._fresh_out_dir(cfg)


def test_run_mode_validates_before_touching_data(capsys):
    code, out = run_main(capsys, "train", "--config", "c1fast", "--mode", "run")
    assert code == cli.EXIT_REFUSED and "num_workers" in out


def test_run_mode_without_data_is_refused_before_training(capsys, tmp_path, monkeypatch):
    _without_data(monkeypatch, tmp_path)
    code, out = run_main(capsys, "train", "--config", "r3", "--mode", "run",
                         "--out", str(tmp_path / "fresh"))
    assert code == cli.EXIT_REFUSED and json.loads(out)["status"] == "refused"
    assert not (tmp_path / "fresh" / "TRAIN_RECEIPT.json").exists()


def test_legacy_train_argv_is_parsed_and_validated(capsys):
    code, out = run_main(capsys, *LEGACY, "--num-workers", "3")
    assert code == cli.EXIT_REFUSED and "num_workers" in json.loads(out)["error"]


def test_legacy_argv_errors_become_refusals(capsys):
    code, out = run_main(capsys, "train", "--pack", "A", "--pack", "B", "--identities", "IA")
    assert code == cli.EXIT_REFUSED and "once per --pack" in json.loads(out)["error"]


def test_eval_without_registered_checkpoint_is_refused(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("WMS_DATA_CKPT_C1FAST", str(tmp_path / "absent.pt"))
    code, out = run_main(capsys, "eval", "--config", "eval_unseen", "--checkpoint", "ckpt_c1fast",
                         "--out", str(tmp_path / "o"))
    assert code == cli.EXIT_REFUSED and "not found" in json.loads(out)["error"]


def test_score_without_state_files_is_refused(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("WMS_DATA_EVAL_STATES_WALL_SAME", str(tmp_path / "absent.json"))
    code, out = run_main(capsys, "score", "--config", "eval_unseen", "--out", str(tmp_path / "o"))
    assert code == cli.EXIT_REFUSED and "not found" in json.loads(out)["error"]


def test_fetch_forwards_to_the_tool(capsys, tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"schema_version": 1, "datasets": {}}), encoding="utf-8")
    assert cli.main(["fetch", "--registry", str(registry), "list"]) == 0
    assert cli.main(["fetch", "--registry", str(registry), "get", "nope"]) == 2


def test_fetch_tool_is_loadable_by_path():
    assert callable(fetch.load_fetch_tool().main)
