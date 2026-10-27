"""Foreground, CPU-only selection replay on an observable saved cycle-1 state."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from traffic_shepherd.c4.probe import select
from traffic_shepherd.c4.probe_runtime import proposals, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    own = Path.home() / "probe"
    for path in (args.data, args.output, args.calibration):
        if path is not None and not path.resolve().is_relative_to(own):
            raise ValueError("dry run artifacts must stay in ~/probe")
    if os.getloadavg()[0] > (os.cpu_count() or 1):
        raise SystemExit("yielding to live collections: retry when idle")
    torch.set_num_threads(1)
    (own / "DRY_RUN_PID.json").write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
    from wms.control.policy import load_policy_components
    from wms.control.policy14_cli import load_members
    from wms.control.deploy import compact_inputs
    from wms.data.pack_history import prior_history
    root = args.data
    policy, encoder, rules, _ = load_policy_components(str(root / "policy.pt"), device="cpu")
    models, _ = load_members([str(root / "wm/wm_s.pt")], [str(root / "wm/config.resolved.json")], "cpu")
    class Episode:
        def array(self, name):
            return np.load(root / "episode" / (name + ".npy"), mmap_mode="r", allow_pickle=False)

        def run_start(self, tick):
            return 0
    ep = Episode()
    ticks = np.asarray(ep.array("time_s"))
    tick = int(np.searchsorted(ticks, 360.))
    kin = torch.tensor(np.asarray(ep.array("veh_kin")[tick]))[None]
    present = np.asarray(ep.array("veh_mask")[tick], bool)
    lanes = kin[..., 1:4].argmax(-1)
    lanes = torch.where(kin[..., 1:4].sum(-1) > 0, lanes, -1)
    state = {"s_m": kin[..., 0], "lane": lanes, "v_mps": kin[..., 4],
             "a_mps2": kin[..., 5], "length_m": kin[..., 6],
             "present": torch.tensor(present)[None],
             "is_cav": torch.tensor(np.asarray(ep.array("is_cav")[tick], bool))[None]}
    hist, mask = prior_history(ep, tick, present)
    inputs = {"state": state, "history": torch.tensor(hist)[None], "history_mask": torch.tensor(mask)[None]}
    m = models[0]
    if m.long_history:
        hist, mask = prior_history(ep, tick, present, history_s=m.history_s, history_stride_s=m.history_stride_s)
        inputs.update(driver_history=torch.tensor(hist)[None], driver_history_mask=torch.tensor(mask)[None])
    inputs, keep = compact_inputs(inputs)
    state = inputs.pop("state")
    ids = [f"recorded-vid-{int(ep.array('vid')[tick, i])}" for i in keep.tolist()]
    sha = hashlib.sha256((root / "wm/wm_s.pt").read_bytes()).hexdigest()
    factory = SimpleNamespace(components=(policy, encoder, rules), models=models, wm_sha256=sha)
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    records = []
    start = time.monotonic()
    for slot in ("single", "group"):
        candidates = proposals(factory, state, ids, slot, 968157)
        print(f"{slot}: {len(candidates)} candidates, {len(ids)} observed vehicles", flush=True)
        scored = score(factory, state, inputs, ids, candidates, 968157) if candidates else []
        chosen = select(scored, calibration, sha, "w968157m1900p05t300")
        records.append({"slot": slot, "candidates": scored, "selected": chosen,
                        "fallback": "unavailable real-pair CQR" if calibration is None else None})
        print(f"{slot}: scored in {time.monotonic() - start:.1f}s", flush=True)
    args.output.write_text(json.dumps({"schema": "c4_probe_dry_run/1", "seed": 968157, "cycle": 1,
                           "state_t_s": float(ticks[tick]), "wm_sha256": sha, "sumo_sim_s": 0,
                           "policy_sha256": hashlib.sha256((root / "policy.pt").read_bytes()).hexdigest(),
                           "wall_s": time.monotonic() - start, "results": records}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
