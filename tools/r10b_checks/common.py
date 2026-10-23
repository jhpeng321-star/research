"""R10B reviewer reject checks (M1-M4, fix-B tails): shared definitions.  CPU only, TRAIN states only.

Arms (checkpoints are passed on the command line; see run_all.sh):
  r9        round-9 final checkpoint (re-scored with the current code)
  r10b      round-10b final checkpoint (stage300, fallback best)
  headzero  r10b with follower_courtesy.net.4.weight/bias zeroed (exact ablation, make_headzero.py)
"""
from __future__ import annotations

import os
import sys

ARMS = ("r9", "r10b", "headzero")
LEN_RAMP = 5.0
END_M = 75.44
DT = 0.5
SHM = os.environ.get("R10BCHK_SHM", "/dev/shm/chk_r10b")


def add_paths(repo: str) -> None:
    for p in (repo, repo + "/src", repo + "/tools", repo + "/tools/r10b_checks"):
        if p not in sys.path:
            sys.path.insert(0, p)


def decode_pair_inputs(x):
    """Courtesy head inputs [n,10] -> dA (ramp offset, m), bumper gap (m), d_end (m), v_f, v_r (m/s)."""
    import numpy as np
    gap = np.sinh(x[:, 0]) * 5.0
    return {"gap": gap, "dA": gap + LEN_RAMP, "dend": np.sinh(x[:, 1]) * 20.0,
            "vf": x[:, 2] * 10.0, "vr": x[:, 3] * 10.0}


# negative-control / M4 groups on head partner rows (decoded from the head inputs)
def groups_of(dec):
    import numpy as np
    g = {
        "all": np.ones(len(dec["dA"]), bool),
        "dA<6 (alongside-ish)": dec["dA"] < 6,
        "dA 6-15": (dec["dA"] >= 6) & (dec["dA"] < 15),
        "dA 15-30": (dec["dA"] >= 15) & (dec["dA"] < 30),
        "dA>30 (far)": dec["dA"] > 30,
        "d_end<10": dec["dend"] < 10,
        "d_end 10-40": (dec["dend"] >= 10) & (dec["dend"] < 40),
        "d_end>40": dec["dend"] > 40,
        "alongside (bumper gap<1)": dec["gap"] < 1.0,
    }
    return g
