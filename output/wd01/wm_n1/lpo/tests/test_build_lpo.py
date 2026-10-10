"""Synthetic-fixture tests for the WM-S LPO window builder (no real data)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

WM_N1 = Path(__file__).resolve().parents[2]
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))

from lpo import build_lpo as bl  # noqa: E402


def _write_index(path: Path, runs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "wm_n1_corpus_pack/2", "runs": runs}),
                    encoding="utf-8")
    return path


def _write_schedule(root: Path, run: str, actions: list[tuple[str, str]],
                    arm: str = "switching") -> None:
    run_dir = root / run / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    events = [{"vehicle_id": v, "time_s": 480.0, "action": a} for v, a in actions]
    (run_dir / "switch_schedule.json").write_text(
        json.dumps({"run": run, "arm": arm,
                    "schedule": {"arm": arm, "events": events}}), encoding="utf-8")


def _v3_run(key: str, *, policy: str, arm: str, batch: str = "chunk1",
            world: str = "w1", split: str = "TRAIN") -> dict:
    run = key.split("/", 1)[1]
    return {"key": key, "chunk": batch, "batch": batch, "world": world, "arm": arm,
            "split": split, "policy": policy, "t0_s": 480.0, "n_ticks": 1201,
            "run": run, "root_rel": None}


def test_is_pi_action_and_policy() -> None:
    assert bl.is_pi_action("pi-external-u") and bl.is_pi_action("Fleet-PI")
    assert not bl.is_pi_action("damping") and not bl.is_pi_action("native")
    assert bl.is_pi_policy("pi") and not bl.is_pi_policy("switching")


def test_val_world_is_deterministic_and_grouped() -> None:
    assert bl.val_world("w1") == bl.val_world("w1")
    flags = {bl.val_world(f"w{i}") for i in range(500)}
    assert flags == {True, False}


def test_schedule_resolution_is_batch_scoped(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _write_schedule(root / "runs" / "chunkA", "w__native", [("a", "pi")])
    _write_schedule(root / "runs" / "chunkB_truearms", "w__native", [("a", "native")])
    inv = tmp_path / "inv"
    inv.mkdir()
    (inv / "opened_paths.json").write_text(json.dumps({"paths": [
        str(root / "runs" / "chunkA" / "w__native" / "run" / "switch_schedule.json"),
        str(root / "runs" / "chunkB_truearms" / "w__native" / "run"
            / "switch_schedule.json")]}), encoding="utf-8")
    amap = bl.load_opened_schedule_map(inv)
    roots = (root / "runs",)
    hit_a = bl.resolve_schedule("w__native", "runs/chunkA", amap, roots)
    hit_b = bl.resolve_schedule("w__native", "runs/chunkB_truearms", amap, roots)
    assert hit_a != hit_b and hit_a is not None and hit_b is not None
    row_a = _v3_run("chunkA/w__native", policy="switching", arm="native")
    row_a["root_rel"] = "runs/chunkA"
    row_b = _v3_run("chunkB/w__native", policy="native", arm="native")
    row_b["root_rel"] = "runs/chunkB_truearms"
    assert bl.classify_v3_run(row_a, amap, roots)["pi"] is True
    verdict_b = bl.classify_v3_run(row_b, amap, roots)
    assert verdict_b["pi"] is False


def test_chunk1_override_not_borrowed_by_fix27(tmp_path: Path) -> None:
    ovr = tmp_path / "p40"
    _write_schedule(ovr, "w__switching9", [("a", "damping")])
    assert bl.resolve_schedule("w__switching9", "runs/campaign_stage1_chunk1",
                               {}, (tmp_path / "none",), override_root=ovr)
    assert bl.resolve_schedule("w__switching9", "runs/campaign_stage1_chunk1_fix27",
                               {}, (tmp_path / "none",), override_root=ovr) is None
    assert bl.resolve_schedule("w__switching9", "runs/campaign_stage1_chunk2",
                               {}, (tmp_path / "none",), override_root=ovr) is None


def test_pi_free_pack_arm_assertion(tmp_path: Path) -> None:
    ds = _write_index(tmp_path / "ds" / "INDEX.json", [
        {"key": "chunk2/w__arm9", "run": "w__arm9", "world": "w", "arm": "arm9",
         "split": "TRAIN", "policy": "trajectory", "batch": "chunk2",
         "chunk": "2", "t0_s": 480.0, "n_ticks": 10}])
    with pytest.raises(ValueError, match="unexpected arms"):
        bl.build({"pack_dspeed": ds}, tmp_path / "out", val_modulo=1)


def test_fixed_without_schedule_is_excluded_unverifiable(tmp_path: Path) -> None:
    idx = _write_index(tmp_path / "v3" / "INDEX.json", [
        _v3_run("chunk1/w__fixed1", policy="fixed", arm="fixed1"),
        _v3_run("chunk1/w__base", policy="damping", arm="base")])
    out = tmp_path / "out"
    receipt = bl.build({"pack_v3": idx}, out,
                       data_roots=(tmp_path / "nowhere",), val_modulo=1)
    excl = json.loads((out / "pi_excluded_runs.json").read_text(encoding="utf-8"))
    assert excl["keys"] == ["chunk1/w__fixed1"]
    assert excl["n_unverifiable"] == 1
    assert excl["unverifiable"][0]["key"] == "chunk1/w__fixed1"
    ev = json.loads((out / "pi_eval_states.json").read_text(encoding="utf-8"))
    assert ev["n_windows"] == 0
    assert receipt["n_corpus_train_windows"] == 2


def test_build_splits_pi_and_val(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _write_schedule(root, "w__fixedPI", [("a", "damping"), ("b", "pi-external-u")])
    _write_schedule(root, "w__fixedND", [("a", "native"), ("b", "damping")])
    _write_schedule(root, "w__switchPI", [("a", "pi")], arm="switching")
    _write_schedule(root, "w__pi", [("a", "pi-external-u"), ("b", "native")],
                    arm="switching")
    runs = [
        _v3_run("chunk1/w__fixedPI", policy="fixed", arm="fixed1"),
        _v3_run("chunk1/w__fixedND", policy="fixed", arm="fixed2"),
        _v3_run("chunk1/w__switchPI", policy="switching", arm="switching1"),
        _v3_run("chunk1/w__pi", policy="switching", arm="pi"),
        _v3_run("chunk1/w__base", policy="damping", arm="base"),
        _v3_run("chunk4/w__pifleet", policy="pi", arm="pi", batch="chunk4"),
        _v3_run("walls/w__wmid", policy="wall", arm="w-mid-early", batch="walls"),
        _v3_run("chunk1/w__calpi", policy="pi", arm="pi", split="CALIBRATION"),
    ]
    idx = _write_index(tmp_path / "v3" / "INDEX.json", runs)
    out = tmp_path / "out"
    receipt = bl.build({"pack_v3": idx}, out, data_roots=(root,), val_modulo=1)

    train = json.loads((out / "windows_train_noPI.json").read_text(encoding="utf-8"))
    val = json.loads((out / "windows_val.json").read_text(encoding="utf-8"))
    ev = json.loads((out / "pi_eval_states.json").read_text(encoding="utf-8"))
    excl = json.loads((out / "pi_excluded_runs.json").read_text(encoding="utf-8"))

    train_keys = {w["key"] for w in train["windows"]}
    val_keys = {w["key"] for w in val["windows"]}
    ev_keys = {w["key"] for w in ev["windows"]}
    assert train_keys | val_keys | ev_keys == {
        "chunk1/w__fixedND", "chunk1/w__base", "walls/w__wmid",
        "chunk1/w__fixedPI", "chunk1/w__switchPI", "chunk1/w__pi",
        "chunk4/w__pifleet"}
    assert not (train_keys & val_keys) and not (train_keys & ev_keys)
    assert ev_keys == {"chunk1/w__fixedPI", "chunk1/w__switchPI", "chunk1/w__pi",
                       "chunk4/w__pifleet"}
    assert excl["keys"] == sorted(ev_keys)
    # val_modulo=1 puts every PI-free world in val; train keeps none
    assert train_keys == set() and val_keys == {"chunk1/w__fixedND",
                                                "chunk1/w__base",
                                                "walls/w__wmid"}
    assert receipt["n_pi_eval"] == 4
    src = {w["key"]: w["pi_source"] for w in ev["windows"]}
    assert src["chunk4/w__pifleet"] == "policy"
    assert src["chunk1/w__pi"] == "wm72-pi-arm"
    assert src["chunk1/w__fixedPI"] == "schedule-events"


def test_dspeed_and_damping_pass_through_with_val(tmp_path: Path) -> None:
    ds = _write_index(tmp_path / "ds" / "INDEX.json", [
        {"key": "chunk2/w__arm0", "run": "w__arm0", "world": "wA", "arm": "arm0",
         "split": "TRAIN", "policy": "trajectory", "batch": "chunk2", "chunk": "2",
         "t0_s": 480.0, "n_ticks": 10}])
    dm = _write_index(tmp_path / "dm" / "INDEX.json", [
        {"key": "chunk1/w__base", "run": "w__base", "world": "wB", "arm": "base",
         "split": "TRAIN", "policy": "damping", "batch": "chunk1", "chunk": "1",
         "t0_s": 240.0, "n_ticks": 10}])
    out = tmp_path / "out"
    bl.build({"pack_dspeed": ds, "pack_damping": dm}, out, val_modulo=1)
    val = json.loads((out / "windows_val.json").read_text(encoding="utf-8"))
    pol = sorted(w["policy"] for w in val["windows"])
    assert pol == ["damping", "trajectory"]
    assert json.loads(
        (out / "pi_eval_states.json").read_text(encoding="utf-8"))["n_windows"] == 0
