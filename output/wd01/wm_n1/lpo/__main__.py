"""CLI: build the WM-S LPO window lists (metadata only).

    python -m lpo --out _out/lpo \
        --pack pack_v3=D:/rd-staging/wmn1_feat/pack_v3/INDEX.json \
        --pack pack_dspeed=E:/rd-wt/dspeed-data/_out/pack_dspeed/INDEX.json \
        --pack pack_damping=E:/rd-wt/damping-snap/_out/pack_damping/INDEX.json \
        --inventory-dir output/wd01/wm_n1/inventory
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .build_lpo import build


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lpo", description=__doc__)
    ap.add_argument("--pack", action="append", required=True,
                    help="NAME=PATH to a pack INDEX.json (repeatable)")
    ap.add_argument("--inventory-dir", default=None,
                    help="inventory dir with opened_paths.json (schedule locator)")
    ap.add_argument("--out", default="_out/lpo")
    ap.add_argument("--val-modulo", type=int, default=10)
    args = ap.parse_args(argv)
    packs = {}
    for item in args.pack:
        name, _, path = item.partition("=")
        if not name or not path:
            ap.error(f"--pack expects NAME=PATH, got {item!r}")
        packs[name] = Path(path)
    receipt = build(packs, Path(args.out), inventory_dir=args.inventory_dir,
                    val_modulo=int(args.val_modulo))
    print(json.dumps({k: receipt[k] for k in (
        "n_train", "n_val", "n_pi_eval", "n_pi_excluded_keys",
        "n_corpus_train_windows", "n_train_windows_by_pack", "pi_eval_by_source")},
        indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
