"""CLI tests: parser, seed recording, member-1-first order, end-to-end main."""
from __future__ import annotations

import json

import pytest

from packsynth import make_runs, write_pack
from train.cli import build_parser, main, record_seed
from train.config import MEMBER_ORDER, TrainConfig


def test_parser_requires_valid_member():
    parser = build_parser()
    args = parser.parse_args(["--member", "1", "--seed", "101",
                              "--corpus", "p", "--out", "o"])
    assert args.member == 1 and args.seed == 101
    with pytest.raises(SystemExit):
        parser.parse_args(["--member", "6", "--seed", "101",
                           "--corpus", "p", "--out", "o"])


def test_record_seed_creates_and_merges(tmp_path):
    record_seed(tmp_path, 1, 101)
    record_seed(tmp_path, 2, 202)
    data = json.loads((tmp_path / "SEEDS.json").read_text(encoding="utf-8"))
    assert data == {"1": 101, "2": 202}


def test_main_trains_member_one(tmp_path, capsys):
    pack = tmp_path / "pack"
    write_pack(pack, make_runs())
    out = tmp_path / "run"
    cfg = TrainConfig(d_model=32, n_heads=4, n_layers=1, ffn=48,
                      bias_hidden=8, counter_channels=8, epochs=1,
                      steps_per_epoch=5, batch_size=4, lr=1e-2, device="cpu")
    cfg_path = cfg.save(tmp_path / "cfg.json")

    rc = main(["--member", "1", "--seed", "101", "--config", str(cfg_path),
               "--corpus", str(pack), "--out", str(out), "--device", "cpu"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["member"] == 1 and summary["seed"] == 101
    assert summary["member_order"] == list(MEMBER_ORDER)
    assert (out / "member_1.pt").is_file()
    assert (out / "SEEDS.json").is_file()
    assert (out / "TRAIN_RECEIPT.json").is_file()
