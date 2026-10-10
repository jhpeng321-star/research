"""Spot-check: subset windows decode byte-identically to the full pack."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WM_N1 = REPO / "output" / "wd01" / "wm_n1"
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))

os.environ.pop("WM_N1_TRAIN_IDENTITIES", None)
os.environ["WM_N1_TRAIN_OUTCOMES"] = str(
    REPO / "_out" / "sidecars_packv3_v2" / "outcomes")

import numpy as np  # noqa: E402
from wms.train import TrajectoryDataset  # noqa: E402

WORK = Path(__file__).resolve().parent
SUBSET = Path(r"D:\rd-staging\pack_v3_r3_subset")
FULL = Path(r"D:\rd-staging\wmn1_feat\pack_v3")
REAL_ID = REPO / "_out" / "sidecars_packv3_v2" / "identities"
WL = WORK / "windows_train_pack_v3.json"


def main() -> int:
    keys = sorted(json.loads(WL.read_text(encoding="utf-8"))["windows"],
                  key=lambda w: w["key"])
    sub = TrajectoryDataset.from_real_packs(
        [str(WORK / "alias" / "pack_v3")],
        identity_dirs=[str(SUBSET / "identities")], horizon_steps=240,
        window_list=str(WL))
    full = TrajectoryDataset.from_real_packs(
        [str(FULL)], identity_dirs=[str(REAL_ID)], horizon_steps=240,
        window_list=str(WL))
    assert len(sub) == len(full) == 294, (len(sub), len(full))
    n_ticks = [int(w["n_ticks"]) for w in keys]
    idx = sorted({0, len(sub) // 2, len(sub) - 1,
                  int(np.argmax(n_ticks))})
    fields = ("states", "present", "loss_valid", "hist", "hist_mask",
              "is_cav", "cav_teacher")
    for i in idx:
        a, b = sub[i], full[i]
        for f in fields:
            if not np.array_equal(np.asarray(a[f]), np.asarray(b[f])):
                print(f"MISMATCH index={i} field={f}")
                return 1
        ma, mb = a["meta"], b["meta"]
        for k in ("world", "arm", "t0_s", "policy"):
            if ma.get(k) != mb.get(k):
                print(f"MISMATCH meta[{k}] index={i}: {ma.get(k)} != {mb.get(k)}")
                return 1
        if a.get("controller_id") != b.get("controller_id"):
            print(f"MISMATCH controller_id index={i}")
            return 1
    print(f"DECODE OK: {len(idx)} sampled subset windows identical to full "
          f"pack (indices {idx})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
