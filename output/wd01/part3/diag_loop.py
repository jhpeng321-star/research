"""Closed-loop instrumented rollout: residual saturation, speeds, leaders."""
import sys

HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "partA_data"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import engine as E  # noqa: E402
import p3lib as P  # noqa: E402

dev = P.device()
m = P.PartAModel().to(dev)
sd = torch.load(HERE / "weights" / "run3" / "member0.pt", map_location="cpu",
                weights_only=True)
m.load_state_dict(sd["model_state"])
m.eval()

r = E.Rollout("cv20_7005m2000", 240.0, kind="model",
              weights=HERE / "weights" / "run3" / "member0.pt", dev=dev)
with torch.no_grad():
    for j in range(1, 121):
        for args in r.births.get(j, ()):
            r._insert(*args)
        sl = E.neighbour_slots(r.lane.detach().cpu().numpy(),
                               r.s.detach().cpu().numpy(),
                               r.v.detach().cpu().numpy(),
                               np.ones(len(r.v)))
        sl_t = torch.as_tensor(sl, device=dev)
        ego = P.ego_features(r.lane, r.s.float(), r.v.float(),
                             r.a_prev.float(),
                             torch.ones(len(r.v), device=dev))
        da, lc, aux = m(ego, sl_t, r.z)
        th = m.theta_of(r.z)
        gap = sl_t[:, 0, 1].double()
        rel = sl_t[:, 0, 2].double()
        free = sl_t[:, 0, 0] < 0.5
        gap = torch.where(free, torch.full_like(gap, 250.0), gap)
        dv = torch.where(free, torch.zeros_like(rel), -rel)
        lim = torch.as_tensor(E.lane_speed_limits(
            r.lane.detach().cpu().numpy(),
            r.s.detach().cpu().numpy()), device=dev, dtype=torch.float64)
        v0_ = th["sf"] * lim
        a_idm = P.idm_acc(r.v, gap, dv, th["a_max"], th["b"], th["tau"],
                          th["s0"], v0_)
        res = torch.tanh(da.double()) * P.RES_SCALE
        a = (a_idm + res).clamp(-9.0, 3.0)
        v_new = (r.v + a * 0.5).clamp_min(0.0).clamp_max(1.2 * lim)
        if j % 20 == 0:
            print(f"j={j:3d} |res| mean={float(res.abs().mean()):.2f} "
                  f"a_idm mean={float(a_idm.mean()):6.2f} "
                  f"v mean={float(r.v.mean()):6.2f} "
                  f"v max={float(r.v.max()):6.2f} "
                  f"leader_frac={float((sl[:, 0, 0] > 0).mean()):.2f} "
                  f"n={len(r.vid)} collisions={r.collisions}")
        r.a_prev = a
        r.v = v_new
        r.s = r.s + v_new * 0.5
        r._count_collisions(r.lane, r.s,
                            E.same_lane_leader_idx(r.lane, r.s),
                            r.vid)
        r._drop_exits()
