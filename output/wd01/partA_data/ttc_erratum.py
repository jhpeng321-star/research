"""PART2C ERRATUM (run once): the DATA_CARD TTC columns computed by
build.py::ttc_rate used the wrong direction (front-vs-behind swapped: the
rate counted 'small gap while the FRONT pulls away', not follower TTC).
This script recomputes correct follower->leader TTC<3 / <1.5 s exposures
at the 0.5 s step directly from the per-root npz files and writes
results/ttc_erratum.json.  MANIFEST.json and the data arrays are NOT
touched (part 3 pins the MANIFEST sha256)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pdl as PD
import loader as LD  # noqa: E402


def ttc_root(z):
    lt3 = lt15 = 0
    ticks = int(z["tick"].max()) + 1
    for k in range(ticks):
        ids = np.where(z["tick"] == k)[0]
        if not len(ids):
            continue
        lane = z["lane"][ids]
        s = z["s"][ids].astype(np.float64)
        v = z["v"][ids].astype(np.float64)
        for li in (0, 1, 2):
            sel = np.where(lane == li)[0]
            if len(sel) < 2:
                continue
            o = sel[np.argsort(-s[sel], kind="stable")]   # front first
            for j in range(len(o) - 1):
                l_, f = o[j], o[j + 1]
                gap = s[l_] - s[f]
                closing = v[f] - v[l_]
                if gap > 0 and closing > 0:
                    t = gap / closing
                    lt3 += t < 3.0
                    lt15 += t < 1.5
    return lt3, lt15


def main():
    man = PD.load_json(HERE / "MANIFEST.json")
    rows = []
    for name, r in sorted(man["roots"].items()):
        z = LD.load_root(name)
        lt3, lt15 = ttc_root(z)
        lt3, lt15 = int(lt3), int(lt15)
        n = r["n_vehicle_steps"]
        rows.append(dict(root=name, split=r["split"],
                         ttc_lt3_steps=lt3, ttc_lt15_steps=lt15,
                         ttc_lt3_per_1000=round(1000 * lt3 / max(n, 1), 3),
                         ttc_lt15_per_1000=round(1000 * lt15 / max(n, 1), 3),
                         vehicle_steps=n))
        print(name, rows[-1])
    agg = {}
    for split in ("train", "holdout", "overall"):
        rs = [r for r in rows if split == "overall" or r["split"] == split]
        n = sum(r["vehicle_steps"] for r in rs)
        agg[split] = dict(
            ttc_lt3_steps=sum(r["ttc_lt3_steps"] for r in rs),
            ttc_lt3_per_1000=round(1000 * sum(r["ttc_lt3_steps"] for r in rs)
                                   / max(n, 1), 3),
            ttc_lt15_per_1000=round(1000 * sum(r["ttc_lt15_steps"] for r in
                                               rs) / max(n, 1), 3))
    out = dict(
        erratum=("build.py ttc_rate counted pairs in the wrong direction "
                 "(front vs behind swapped); these are the corrected "
                 "follower->leader TTC exposures at the 0.5 s step"),
        by_split=agg, per_root=rows)
    (HERE / "results" / "ttc_erratum.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print("ERRATUM written;", json.dumps(agg, indent=1))


if __name__ == "__main__":
    sys.exit(main())
