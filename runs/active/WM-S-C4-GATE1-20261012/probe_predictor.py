"""Gate 1 probe 3: end-to-end predictor smoke on one custom state (R5b / C1 / physics)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["TEMP"] = os.environ["TMP"] = "E:/agent-work/tmp"

import numpy as np
import torch

from wms.eval.predictor import RolloutFlags, WmsPredictor
from wms.eval.states import Identity, Pack, build_state

DATA = Path("E:/research-dense/data_cache")
R5B = Path("E:/research-dense/runs/active/WM-S-R5B-20261012/box/gpuA/r5b_main_a1/wm_s_best.pt")
C1 = DATA / "ckpt_c1fast" / "wm_s.pt"


def make_state(packs, ids):
    world, t0 = "w8659m2700p10t420", 420.0
    window = {"pack": "pack_v3", "world": world, "arm": "native", "t0_s": t0,
              "chunk": None, "key": f"{world}__native", "run": None, "batch": None,
              "policy": None, "split": "TRAIN", "_extra_baselines":
              ("pi", "w-lo-early", "w-lo-late", "w-mid-early", "w-mid-late")}
    rec = build_state(window, packs, ids, horizon_s=60.0, history_s=60.0)
    damp = dict(window, pack="pack_damping", arm="base", key=f"{world}__base",
                _extra_baselines=())
    rec2 = build_state(damp, packs, ids, horizon_s=60.0, history_s=60.0)
    rec["recorded"]["base"] = rec2["recorded"]["base"]
    return rec


def physics_only(ckpt: Path) -> WmsPredictor:
    p = WmsPredictor.from_checkpoint(str(ckpt), RolloutFlags(entries=False, entries_v2=False))
    m = p.model
    og, olg = m.gate, m.lane_gate
    m.gate = lambda k: torch.zeros_like(og(k))
    m.lane_gate = lambda k, shared=None: torch.zeros_like(olg(k))
    m.lane_change_head = None
    m.diag_encoder_population = True
    return p


def main():
    torch.set_num_threads(1)
    packs = {"pack_v3": Pack("pack_v3", DATA / "pack_v3"),
             "pack_damping": Pack("pack_damping", DATA / "pack_damping")}
    ids = {"pack_v3": Identity("pack_v3", DATA / "sidecars_packv3_v2" / "identities"),
           "pack_damping": Identity("pack_damping", DATA / "identities_damping")}
    rec = make_state(packs, ids)
    print("arms:", sorted(rec["recorded"]), "t0:", rec["t0_s"], "hist:",
          rec["history"]["history_steps"], rec["history"].get("run_start_limited"))
    for name, pred in (("r5b", WmsPredictor.from_checkpoint(str(R5B), RolloutFlags())),
                       ("c1", WmsPredictor.from_checkpoint(str(C1), RolloutFlags())),
                       ("physics", physics_only(R5B))):
        for arm in ("native", "base", "pi", "w-lo-late"):
            t = time.time()
            try:
                frames = pred.predict_states(rec, {"arm": arm}, 60.0, 0)
                n60 = len(frames[-1]["vehicles"])
                print(f"{name:8s} {arm:10s} frames={len(frames)} n@60s={n60} "
                      f"({time.time()-t:.1f}s)")
            except Exception as exc:
                import traceback
                print(f"{name:8s} {arm:10s} ERROR {type(exc).__name__}: {exc}")
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
