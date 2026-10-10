"""RUN2 E1/E3 contracts: the stratified pack selector and traffic-level report."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(PKG))

from wms_eval import materialize as M  # noqa: E402
from wms_eval import r2set as R  # noqa: E402
from wms_eval import traffic_eval as TE  # noqa: E402

K = 4
T = 10


def _kin_row(s, lane, v):
    row = np.zeros(7, np.float16)
    row[0] = s
    row[1 + lane] = 1.0
    row[4] = v
    row[6] = 5.0
    return row


def _build_pair_pack(root: Path):
    """A pack with paired native + pi TRAIN runs on one world/t0."""
    pack = root / "pack_v3"
    pack.mkdir(parents=True)
    kin = np.zeros((T, K, 7), np.float16)
    mask = np.zeros((T, K), np.uint8)
    step = np.zeros((T, K, 16), np.float16)
    tmask = np.zeros((T, K), np.uint8)
    for t in range(T):
        kin[t, 0] = _kin_row(10.0 * t, 0, 20.0)
        mask[t, 0] = 1
        if t <= 6:
            kin[t, 1] = _kin_row(500.0 + 15.0 * t, 1, 15.0)
            mask[t, 1] = 1
    for t in range(T - 1):
        tmask[t] = (mask[t] & mask[t + 1]).astype(np.uint8)
    for name, arr in (("veh_kin", kin), ("veh_mask", mask), ("veh_step", step),
                      ("target_mask", tmask)):
        np.save(pack / f"{name}.npy", arr)

    world = "w9001m1000p02t100"
    manifest_items = []
    runs = []
    for arm in ("native", "pi"):
        manifest_items.append({"world": world, "arm": arm, "chunk": "chunk1",
                               "t0_s": 1.0, "tick": 2, "policy": arm,
                               "split": "TRAIN"})
        runs.append({"key": f"chunk1/{world}__{arm}", "chunk": "chunk1",
                     "world": world, "arm": arm, "split": "TRAIN", "t0_s": 1.0,
                     "policy": arm, "n_ticks": T, "tick_start": 0,
                     "tick_end": T, "dec_start": 0, "dec_end": 20})
    manifest = {
        "schema": "wm_n1_pack/1", "k_max": K,
        "arrays": {
            "veh_kin": {"file": "veh_kin.npy", "shape": [T, K, 7]},
            "veh_mask": {"file": "veh_mask.npy", "shape": [T, K]},
            "veh_step": {"file": "veh_step.npy", "shape": [T, K, 16]},
            "target_mask": {"file": "target_mask.npy", "shape": [T, K]},
        },
        "items": manifest_items,
    }
    index = {"schema": "wm_n1_corpus_pack/2", "k_max": K, "runs": runs}
    (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (pack / "INDEX.json").write_text(json.dumps(index), encoding="utf-8")

    side = root / "identities"
    side.mkdir()
    np.save(side / "identities.npy", np.asarray([[0, 1, -1, 2], [0, 1, -1, 2]], np.int64))
    np.save(side / "global_positions.npy", np.asarray([0, 1], np.int64))
    (side / "vehicle_map.json").write_text(json.dumps(
        [["chunk1", world, "vehA"], ["chunk1", world, "vehB"],
         ["chunk1", world, "vehC"]]), encoding="utf-8")
    return pack, side


def test_world_meta_and_strata():
    meta = M.world_meta("w8537m2600p40t240")
    assert meta == {"world": "w8537m2600p40t240", "world_id": 8537,
                    "demand": 2600, "penetration_pct": 40, "t0_s": 240.0}
    assert M.world_meta("not-a-world")["demand"] is None


def test_pair_selection_requires_native_and_pi(tmp_path):
    pack_dir, side_dir = _build_pair_pack(tmp_path)
    packs = {"pack_v3": R.Pack("pack_v3", pack_dir)}
    pairs = R.paired_index(packs["pack_v3"])
    assert pairs == {("w9001m1000p02t100", 1.0)}
    windows = R.select_windows(packs, target=5, max_per_world=5)
    arms = {str(w["arm"]) for w in windows}
    assert arms == {"native", "pi"}


def test_select_windows_drops_unpaired(tmp_path):
    pack_dir, side_dir = _build_pair_pack(tmp_path)
    pack = R.Pack("pack_v3", pack_dir)
    # remove the pi run: the world/t0 now has no pair, so selection is empty
    pack.runs = {k: v for k, v in pack.runs.items() if not k.endswith("__pi")}
    with pytest.raises(R.R2SetError):
        R.select_windows({"pack_v3": pack}, target=5)


def test_build_set_streams_jsonl_and_meta(tmp_path):
    pack_dir, side_dir = _build_pair_pack(tmp_path)
    packs, identities = R.load_packs(
        [f"pack_v3={pack_dir}"], [f"pack_v3={side_dir}"])
    windows = R.select_windows(packs, target=5, max_per_world=5)
    out = tmp_path / "set.jsonl"
    meta = R.build_set(packs, identities, windows, out, target=2,
                       horizon_s=5.0, min_vehicles=1)
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    assert meta["n_states"] == 2
    assert meta["n_paired"] == 2
    record = json.loads(lines[0])
    assert set(record["recorded"]) >= {"native"}
    assert record["world_meta"]["penetration_pct"] == 2


# --------------------------------------------------------------------------- #
# traffic_eval
# --------------------------------------------------------------------------- #
def _section(**metrics):
    base = {"mean_speed": 10.0, "stopped_vehicles": 1.0, "merge_queue_length": 2.0,
            "total_delay": 3.0, "merge_flow": 4.0}
    base.update(metrics)
    return base


def _bins(start, values):
    return {str(i): {"start_s": start + 30.0 * i, "end_s": start + 30.0 * (i + 1),
                     "sections": {"merge": _section(mean_speed=v),
                                  "upstream": _section(mean_speed=v),
                                  "downstream": _section(mean_speed=v)}}
            for i, v in enumerate(values)}


def _payload(t0_s=240.0):
    return {
        "n_states": 1, "n_paired": 1,
        "results": [{
            "index": 0, "world": "wTEST", "t0_s": t0_s, "paired": True,
            "traffic": {
                "predicted": {"native": _bins(t0_s, [1, 2, 3, 4]),
                              "pi": _bins(t0_s, [2, 3, 4, 5])},
                "recorded": {"native": _bins(t0_s, [1, 2, 3, 4]),
                             "pi": _bins(t0_s, [2, 3, 4, 5])},
                "paired_predicted_pi_minus_native": _bins(t0_s, [1, 1, 1, 1]),
                "paired_recorded_pi_minus_native": _bins(t0_s, [1, 1, 1, 1]),
            },
        }],
    }


def test_headline_filters_relative_to_t0():
    # t0=240 s: only bins with offset >= 60 s (indices 2, 3) may enter.
    payload = _payload(t0_s=240.0)
    seen = {key for _res, key, _pb, _rb in TE._bins(payload, "native", min_start_s=60.0)}
    assert seen == {"2", "3"}
    headline = TE.traffic_headline(payload, "native", min_start_s=60.0)
    # merge mean_speed over bins 3 and 4 (values 3, 4) -> 3.5
    assert headline["merge"]["mean_speed"]["recorded"] == pytest.approx(3.5)
    assert headline["merge"]["mean_speed"]["predicted"] == pytest.approx(3.5)


def test_paired_traffic_stats_sign_and_rank():
    payload = _payload(t0_s=0.0)
    stats = TE.paired_traffic_stats(payload, "merge", "mean_speed", min_start_s=0.0)
    assert stats["n"] == 4
    assert stats["sign_agreement"] == pytest.approx(1.0)
    assert stats["mae"] == pytest.approx(0.0)
    assert stats["rank_corr"] is None  # constant deltas -> undefined correlation


def test_report_build_uses_all_runs(tmp_path):
    runs = {"keep_speed": _payload(240.0), "wms_x": _payload(240.0)}
    # build() reads files; write them as EVAL.json dirs
    paths = {}
    for label, payload in runs.items():
        d = tmp_path / label
        d.mkdir()
        (d / "EVAL.json").write_text(json.dumps(payload), encoding="utf-8")
        paths[label] = d
    out = TE.build(paths, tmp_path / "R2.md", min_start_s=60.0, note="note-line")
    text = out.read_text(encoding="utf-8")
    assert "R2_BASELINE" in text
    assert "note-line" in text
    assert "keep_speed" in text and "wms_x" in text
    assert ">= 60 s after t0" in text
