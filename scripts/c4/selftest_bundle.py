#!/usr/bin/env python
"""Self-test of pack_bundle.py / unpack_bundle.py on a tiny synthetic pack (no real data read).

Builds a fake data root (one pack with TRAIN + CALIBRATION runs, tick/decision/item axis arrays, a
window list, one restricted-name file), bundles it, unpacks it and checks that
  * selected rows are bit-identical, unselected rows are zero (CAL bytes never shipped),
  * restricted-name files are dropped and the tar self-check passes,
  * shapes/dtypes/headers are preserved.
Run:  python scripts/c4/selftest_bundle.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> int:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(prefix="c4selftest_") as td:
        td = Path(td)
        data = td / "data"
        pack = data / "packx"
        pack.mkdir(parents=True)
        runs = []
        t = d = 0
        for i, (split, pol) in enumerate([("TRAIN", "native"), ("CALIBRATION", "damping"),
                                          ("TRAIN", "damping"), ("TRAIN", "trajectory")]):
            nt, nd = 10 + i, 2
            runs.append({"key": f"b/w{i}__x", "split": split, "policy": pol, "arm": "x",
                         "tick_start": t, "tick_end": t + nt, "dec_start": d, "dec_end": d + nd,
                         "n_ticks": nt, "n_decision": nd})
            t += nt
            d += nd
        items = [{"tick": r["tick_start"] + 1, "split": r["split"]} for r in runs]
        arrays = {"veh": rng.random((t, 5, 3)).astype("float16"), "plan": rng.random((d, 4)).astype("float32"),
                  "mask": rng.integers(0, 2, (t, 5)).astype("uint8"),
                  "per_item": rng.random((len(items), 6)).astype("float32")}
        for n, a in arrays.items():
            np.save(pack / f"{n}.npy", a)
        (pack / "INDEX.json").write_text(json.dumps({"n_ticks": t, "n_decision": d, "runs": runs}))
        (pack / "manifest.json").write_text(json.dumps({"items": items}))
        (data / "lists").mkdir()
        (data / "lists" / "w.json").write_text(json.dumps({"windows": [
            {"key": r["key"], "pack": "packx", "policy": r["policy"], "split": "TRAIN"}
            for r in runs if r["split"] == "TRAIN"]}))
        (pack / "HOLDOUT_notes.json").write_text("{}")
        manifest = {
            "schema": "c4_bundle_manifest/1", "default_policies": ["native", "trajectory"],
            "code": {"root_files": [], "dirs": [], "files": [], "exclude_globs": [], "text_lf_suffixes": [".sh"]},
            "groups": {"packx": {"kind": "pack", "src": "packx", "dest": "data_cache/packx",
                                 "pack_name": "packx", "lists": ["lists/w.json"], "filter": "policy"}},
            "checkpoints": []}
        mpath = td / "m.json"
        mpath.write_text(json.dumps(manifest))
        bundle, root = td / "bundle", td / "root"
        py = sys.executable
        subprocess.run([py, str(HERE / "pack_bundle.py"), "--out", str(bundle), "--data-root", str(data),
                        "--manifest", str(mpath), "--repo", str(td), "--groups", "packx"], check=True)
        subprocess.run([py, str(HERE / "unpack_bundle.py"), "--bundle", str(bundle), "--root", str(root)],
                       check=True)
        out = root / "data_cache" / "packx"
        assert not (out / "HOLDOUT_notes.json").exists(), "restricted file leaked"
        assert not (out / "__ranges__").exists() or not any((out / "__ranges__").iterdir())
        keep = [0, 3]                     # TRAIN + native/trajectory runs
        drop = [1, 2]                     # CALIBRATION run and the designed (damping) run
        for name, axis in (("veh", "tick"), ("plan", "dec"), ("mask", "tick"), ("per_item", "item")):
            src = np.load(pack / f"{name}.npy")
            got = np.load(out / f"{name}.npy")
            assert got.shape == src.shape and got.dtype == src.dtype, name
            for i, r in enumerate(runs):
                if axis == "tick":
                    sl = slice(r["tick_start"], r["tick_end"])
                elif axis == "dec":
                    sl = slice(r["dec_start"], r["dec_end"])
                else:
                    sl = slice(i, i + 1)
                if i in keep:
                    assert np.array_equal(got[sl], src[sl]), f"{name} run {i} differs"
                else:
                    assert not got[sl].any(), f"{name} run {i} should be zero"
        print("selftest OK: selected rows identical, unselected rows zero, restricted file dropped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
