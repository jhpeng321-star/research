"""S-POLAUDIT-D: imagination-side vs deployment-side C1 residual, bitwise, on recorded states.

Imagination = ``wms.control.bridge.make_cav_controller`` (the WM rollout callback, decision tick);
deployment  = ``wms.control.policy.PolicyExecutor`` (``load_policy_components`` as in deploy.py).
Both from THIS worktree (c4/c4prep lineage).  Also reports the executed-acceleration semantics
each side applies to the same residual: imagination ``safe_accel(compose_native(native, r))``
versus SUMO ``setAcceleration`` capped at the car-following safe speed (speedMode 27 bit 0),
i.e. at most the native follow command for r > 0.

    python tools/polaudit_d/parity_check.py <episode.npz> <policy.pt> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import torch  # noqa: E402

torch.set_num_threads(1)

from wms.control.bridge import make_cav_controller  # noqa: E402
from wms.control.nativeinit import compose_native  # noqa: E402
from wms.control.policy import PolicyExecutor, load_policy_components  # noqa: E402
from wms.control.safety import safe_accel  # noqa: E402
from wms.model.cav import WM_LANE_TARGET_KEY  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from residual_probe import states_from_episode  # noqa: E402


def main(episode, ckpt, out):
    z = dict(np.load(episode, allow_pickle=False))
    state_at = states_from_episode(z)
    policy, enc, rules, _ = load_policy_components(ckpt, device="cpu")
    exe = PolicyExecutor(policy, enc, lane_rules=rules, deterministic=True, device="cpu")
    records = []
    zero_native = lambda st: torch.zeros_like(st["s_m"])  # noqa: E731
    ctrl = make_cav_controller(policy, enc, safe_accel, records, native=zero_native,
                               native_lane_change=True)
    times, ctl_t = z["time_s"], z["control_time_s"]
    max_diff, n_cmp, n_ticks, bitwise = 0.0, 0, 0, True
    gap = []
    for t in ctl_t:
        if t < 300 or n_ticks >= 120:
            continue
        i = int(np.flatnonzero(np.abs(times - t) < 1e-6)[0])
        st = state_at(i)
        dep = exe(st).residual[0]
        cb = dict(st)
        cb["physics_step_index"] = torch.tensor(0)      # every call is a fresh decision tick
        cb[WM_LANE_TARGET_KEY] = st["lane"].clone()
        records.clear()
        with torch.no_grad():
            ctrl(cb)
        imag = records[-1]["actions"][0].double().numpy()
        mask = (st["is_cav"][0] & (st["lane"][0] >= 0)).numpy()
        d = np.abs(imag[mask] - dep[mask])
        bitwise &= bool((imag[mask].astype(np.float32) == dep[mask].astype(np.float32)).all())
        max_diff = max(max_diff, float(d.max()) if d.size else 0.0)
        n_cmp += int(mask.sum())
        n_ticks += 1
        # Same residual, a nominal native command 0: imagination executes compose+safety, SUMO
        # caps at the safe/follow speed (positive residual cannot exceed native).
        native = torch.zeros_like(st["s_m"])
        r = torch.as_tensor(dep, dtype=torch.float32)[None]
        imag_exec = safe_accel(st, compose_native(native, r, -3.0, 2.6), accel_max=2.6)[0].numpy()
        sumo_exec = np.minimum(dep, 0.0)
        gap.extend((imag_exec[mask] - sumo_exec[mask]).tolist())
    res = {"ticks": n_ticks, "cav_rows": n_cmp, "max_abs_residual_diff": max_diff,
           "float32_bitwise_equal": bitwise,
           "executed_gap_imagination_minus_sumo_at_native0": {
               "mean": float(np.mean(gap)), "max": float(np.max(gap)), "min": float(np.min(gap))}}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if bitwise else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
