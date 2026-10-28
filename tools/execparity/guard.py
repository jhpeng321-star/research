"""Replay a policy before SUMO. Exit 2 when >95% of acting rows lack residual authority."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import executor, sha, states_from_episode
from wms.control.execution import authority_guard

BAD = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype", re.I)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--episode", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for path in (args.policy, args.out, *args.episode):
        if BAD.search(str(path)) or BAD.search(str(path.resolve())):
            raise ValueError("prohibited path")
    exe, _ = executor(args.policy)
    if not exe.residual_on_native:
        raise ValueError("this guard is for the native-residual execution path")
    values, hashes = [], {}
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
            r = exe(st).residual
            mask = (st["is_cav"] & (st["lane"] >= 0)).numpy()
            values.extend(r[mask].tolist())
    receipt = authority_guard(values, checkpoint_sha256=sha(args.policy),
        recordings_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())
    receipt["recordings"] = hashes
    assert args.out.parent.is_dir()
    args.out.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
