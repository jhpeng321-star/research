"""Replay a policy before SUMO. Exit 2 when >95% of acting rows lack residual authority."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import executor, sha, states_from_episode
from wms.control.execution import authority_guard
from wms.control.continuous import native_relative, project_native, residual_cap_mask
from wms.control.execution import inferred_follow_cap
from wms.control.native_cav import head_native_accel
from wms.model.cav import EXEC_CAP_KEY

BAD = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype", re.I)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--episode", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--driver", type=Path, required=True)
    ap.add_argument("--mode", choices=("1", "2"), default="1")
    args = ap.parse_args()
    for path in (args.policy, args.out, args.driver, *args.episode):
        if BAD.search(str(path)) or BAD.search(str(path.resolve())):
            raise ValueError("prohibited path")
    os.environ["C4_EXECUTION_PARITY"] = args.mode
    os.environ["C4_EXECUTION_DRIVER"] = str(args.driver.resolve())
    exe, _ = executor(args.policy)
    reference = getattr(exe.policy, "residual_reference", "base")
    if not exe.residual_on_native:
        raise ValueError("this guard is for the native-residual execution path")
    values, deltas, caps, hashes = [], [], [], {}
    for path in args.episode:
        assert path.name == "episode.npz"
        with np.load(path, allow_pickle=False) as tape:
            z = dict(tape)
        seed = int(json.loads(str(z["meta_json"]))["seed"])
        if 964000 <= seed <= 964199:
            raise ValueError("forbidden seed")
        hashes[str(path.resolve())] = sha(path)
        state = states_from_episode(z)
        for t in z["control_time_s"]:
            if t < 300:
                continue
            i = int(np.searchsorted(z["time_s"], t))
            st = state(i)
            keep = torch.nonzero(st["present"][0]).flatten()
            st = {k: v.index_select(1, keep) for k, v in st.items()}
            with torch.no_grad():
                r = torch.as_tensor(exe(st).residual)
                if args.mode == "2":
                    from wms.control.alwayscmd import executed, inferred_base, inferred_cap
                    b, c = inferred_base(st), inferred_cap(st)
                    base = torch.maximum(torch.minimum(b, c), -st["v_mps"] / .5)
                    delta = executed(st, r, exe.accel_min, exe.accel_max,
                                     residual_reference=reference) - base
                    centre = base if reference == "ctrl0" else b
                    lower = torch.minimum(centre, torch.full_like(centre, exe.accel_min))
                    upper = torch.maximum(centre, torch.full_like(centre, min(exe.accel_max, 1.2)))
                    requested = centre + r
                    cap = ((requested < lower) | (requested > upper) | (requested > c)
                           | (requested < -st["v_mps"] / .5))
                else:
                    st[EXEC_CAP_KEY] = inferred_follow_cap(st,None)
                    base = project_native(st,head_native_accel(st))
                    delta = native_relative(st,r,base)-base
                    cap = residual_cap_mask(st,r,base)
            mask = (st["is_cav"] & (st["lane"] >= 0)).numpy()
            values.extend(r[mask].tolist())
            deltas.extend(delta[mask].tolist())
            caps.extend(cap[mask].tolist())
    receipt = authority_guard(values, checkpoint_sha256=sha(args.policy),
        recordings_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        executed_delta=deltas, capped=caps, residual_reference=reference)
    receipt["recordings"] = hashes
    assert args.out.parent.is_dir()
    args.out.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
