"""S-POLAUDIT-D zero-SUMO probe: does the deployed C1 native-residual actually move SUMO CAVs?

Replays the RECORDED E3-final 'ours' episode states (observable episode.npz only; no scenario,
route or vType files) through the deployment PolicyExecutor of the frozen scientific source
(E:/agent-work/c4-e3final), for C1 (64f70af4) and the coop-off twin (e1fa4881), and compares
  * the residual each actor emits per control tick,
  * the recorded requested (composed native+residual) command,
  * the realised SUMO acceleration over the two following 0.5 s physics steps.

    python tools/polaudit_d/residual_probe.py <data_root> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path("E:/agent-work/c4-e3final")
sys.path[:0] = [str(SRC / "src"), str(SRC)]

import torch  # noqa: E402

torch.set_num_threads(1)

from wms.control.policy import PolicyExecutor, load_policy_components  # noqa: E402


def states_from_episode(z):
    s, v, a = z["s_m"], z["v_mps"], z["a_mps2"]
    lane, present, is_cav, junction = z["lane"], z["present"], z["is_cav"], z["junction"]
    k = s.shape[1]

    def state(i):
        t = lambda x, dt: torch.as_tensor(np.asarray(x)[None], dtype=dt)  # noqa: E731
        p = present[i]
        return {"s_m": t(np.where(p, s[i], 0.0), torch.float32),
                "v_mps": t(np.where(p, v[i], 0.0), torch.float32),
                "a_mps2": t(np.where(p, a[i], 0.0), torch.float32),
                "lane": t(lane[i].astype(np.int64), torch.long),
                "present": t(p, torch.bool), "is_cav": t(is_cav[i] & p, torch.bool),
                "length_m": t(np.full(k, 5.0, np.float32), torch.float32),
                "junction": t(junction[i], torch.bool)}
    return state


def executor(path):
    policy, enc, rules, payload = load_policy_components(str(path), device="cpu")
    return PolicyExecutor(policy, enc, lane_rules=rules, deterministic=True, device="cpu"), payload


def main(root, out):
    root = Path(root)
    ep = root / "e3final/main/e3final/shard0/jobs/s977000_ours/deploy/episode/episode.npz"
    z = dict(np.load(ep, allow_pickle=False))
    state = states_from_episode(z)
    times = z["time_s"]
    ctl_t = z["control_time_s"]
    req = z["requested_acceleration_mps2"]
    iss = z["issued_acceleration_mps2"]
    ctrl = z["controller"]
    vid = z["vid_index"]
    on, pay_on = executor(root / "e3final/inputs/policy_latest.pt")
    off, pay_off = executor(root / "e3long/inputs/coopoff_cycle1/policy_latest.pt")
    meta = json.loads(str(z["meta_json"]))
    warm = float(meta.get("warmup_s", 300))
    rows = []
    for c, t in enumerate(ctl_t):
        if t < warm - 1e-9:
            continue
        i = int(np.flatnonzero(np.abs(times - t) < 1e-6)[0])
        if i + 2 >= len(times):
            break
        st = state(i)
        r_on = on(st).residual[0]
        r_off = off(st).residual[0]
        acting = st["is_cav"][0].numpy() & (st["lane"][0].numpy() >= 0)
        for k in np.flatnonzero(acting):
            same = (vid[i + 1, k] == vid[i, k]) and (vid[i + 2, k] == vid[i, k]) and z["present"][i + 2, k]
            if not same:
                continue
            v0, v1, v2 = (float(z["v_mps"][j, k]) for j in (i, i + 1, i + 2))
            rows.append((float(t), int(k), float(r_on[k]), float(r_off[k]), float(req[c, k]),
                         float(iss[c, k]), int(ctrl[c, k]), (v1 - v0) / 0.5, (v2 - v1) / 0.5, v0))
    a = np.array(rows, float)
    r_on, r_off, rq, isd, ctl, a1, a2, v0 = (a[:, j] for j in range(2, 10))
    finite = np.isfinite(rq)
    nat_est = rq - r_on          # valid where the composition was not clipped
    res = {
        "episode": str(ep), "n_cav_ticks": int(len(a)),
        "controller_counts": {int(x): int((ctl == x).sum()) for x in np.unique(ctl)},
        "requested_finite_fraction": float(finite.mean()),
        "residual_on": {"mean": float(r_on.mean()), "p05": float(np.quantile(r_on, .05)),
                        "p50": float(np.median(r_on)), "p95": float(np.quantile(r_on, .95)),
                        "frac_positive": float((r_on > 0).mean()), "frac_exact_zero": float((r_on == 0).mean()),
                        "mean_abs": float(np.abs(r_on).mean())},
        "residual_on_minus_off_max_abs": float(np.abs(r_on - r_off).max()),
        "requested_minus_realised_first_substep": {
            "mean": float(np.nanmean(rq[finite] - a1[finite])),
            "mean_abs": float(np.nanmean(np.abs(rq[finite] - a1[finite])))},
        "realised_vs_requested_slope": float(np.polyfit(rq[finite], a1[finite], 1)[0]),
        "realised_vs_native_est_mean_abs": float(np.nanmean(np.abs(nat_est[finite] - a1[finite]))),
        "frac_realised_closer_to_native_than_requested": float(np.mean(
            np.abs(nat_est[finite] - a1[finite]) < np.abs(rq[finite] - a1[finite]))),
        "frac_requested_at_or_above_1p2": float(np.mean(rq[finite] >= 1.2 - 1e-4)),
        "positive_residual_rows": int((r_on[finite] > 0).sum()),
        "negative_residual_rows": int((r_on[finite] < 0).sum()),
    }
    neg = finite & (r_on < -0.05)
    if neg.any():
        res["negative_residual_lt_-0.05"] = {
            "n": int(neg.sum()), "mean_residual": float(r_on[neg].mean()),
            "mean_requested": float(rq[neg].mean()), "mean_native_est": float(nat_est[neg].mean()),
            "mean_realised_a1": float(a1[neg].mean()), "mean_realised_a2": float(a2[neg].mean())}
    pos = finite & (r_on > 0.05)
    if pos.any():
        res["positive_residual_gt_0.05"] = {
            "n": int(pos.sum()), "mean_residual": float(r_on[pos].mean()),
            "mean_requested": float(rq[pos].mean()), "mean_native_est": float(nat_est[pos].mean()),
            "mean_realised_a1": float(a1[pos].mean()), "mean_realised_a2": float(a2[pos].mean())}
    res["payload_meta"] = {k: pay_on.get(k) for k in ("format", "residual_on_native", "lane_control",
                                                       "accel_min_mps2", "accel_max_mps2", "update_count")}
    res["accel_bounds_loaded"] = [on.accel_min, on.accel_max]
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
