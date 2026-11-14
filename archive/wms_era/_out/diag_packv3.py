"""CPU-only, metadata/mmap dataset-build diagnosis for pack_v3 window drop.

Run from output/wd01/wm_n1 so `train`/`wms`/`dspeed` import. Nothing is written
outside the given window lists; no training, no GPU, no SUMO.
"""
from __future__ import annotations

import argparse
import json
import sys

from wms.train import TrajectoryDataset, TrainDataRefusal
from wms import lpo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", action="append", required=True)
    ap.add_argument("--identities", action="append", required=True)
    ap.add_argument("--overlay", action="append", default=None)
    ap.add_argument("--window-list", default=None)
    ap.add_argument("--exclude-window-list", action="append", default=None)
    ap.add_argument("--horizon-steps", type=int, default=240)
    ap.add_argument("--pack-name", action="append", default=None)
    args = ap.parse_args(argv)

    names = args.pack_name
    if args.window_list:
        wl = lpo.window_keys(args.window_list,
                             packs=(names or [__import__("pathlib").Path(p).name
                                              for p in args.pack]))
        print("WINDOW-KEYS-BY-PACK:", json.dumps(
            {k: len(v) for k, v in sorted(wl.items())}))
    try:
        ds = TrajectoryDataset.from_real_packs(
            args.pack, identity_dirs=args.identities, overlays=args.overlay,
            horizon_steps=args.horizon_steps, start_shift_prob=0.0,
            window_list=args.window_list,
            exclude_window_list=args.exclude_window_list,
            entries_v2=True, pack_name_list=names)
    except TrainDataRefusal as exc:
        print("REFUSED:", exc)
        return 2
    print(json.dumps(ds.receipt, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
