"""S-COOPW: distribution of the demand-sized entrant pad of ``with_entry_slots`` at longer horizons.

``pad = ceil((demand_main+ramp * h + min(backlog, 0.5 * h)) * 1.5) + 16`` (the existing formula,
before the ``entry_pad_max`` cap) over N plain TRAIN start draws, for each horizon h.  The staged
horizon caps (``[horizon] entry_pad_max``) are set to the observed maximum so no TRAIN start is
truncated.  CPU only, reads start windows only.

    python tools/coopw/entry_pad_scan.py --config configs/c4_policy14_coop.toml --wm-config CFG \
        --wm-ckpt CK --n 400 --horizons 10 30 60 --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/c4_policy14_coop.toml")
    p.add_argument("--wm-ckpt", required=True)
    p.add_argument("--wm-config", required=True)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=965143)
    p.add_argument("--horizons", type=float, nargs="+", default=[10.0, 30.0, 60.0])
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    torch.set_num_threads(2)
    from wms.control import dreamer
    from wms.control.policy14_cli import load_members
    from wms.control.train_cli import _load_starts, _read_config
    from wms.model.entries import EntranceQueue

    raw = _read_config(args.config)
    models, _ = load_members([args.wm_ckpt], [args.wm_config], "cpu")
    train, _ = _load_starts(raw["data"], models, args.seed, entries=True)
    torch.manual_seed(args.seed)
    pads = {h: [] for h in args.horizons}
    for _ in range(args.n):
        q = EntranceQueue.from_vector(train.sample(1).entry_queue[0].numpy())
        for h in args.horizons:
            pads[h].append(int(math.ceil((float(np.sum(q.demand_rate)) * h
                                          + min(float(np.sum(q.backlog)), dreamer._ENTRY_PAD_BACKLOG_RATE * h))
                                         * dreamer._ENTRY_PAD_SAFETY)) + dreamer._ENTRY_PAD_MARGIN)
    out = {str(int(h)): {"max": max(v), "p95": float(np.quantile(v, 0.95)), "median": float(np.median(v)),
                         "share_above_base_cap": float(np.mean(np.asarray(v) > raw["trainer"]["entry_pad_max"]))}
           for h, v in pads.items()}
    Path(args.out).write_text(json.dumps({"n": args.n, "pads": out}, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
