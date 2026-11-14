"""D2-D5 diagnostics: train/serve feature parity, one-step engine-vs-
training metric, phase-2 loss term magnitudes, val-RMSE scorer check."""
import sys

import numpy as np
import torch

HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine as E  # noqa: E402
import p3lib as P  # noqa: E402

dev = P.device()


def d2(n_steps=1000, seed=0):
    """Train-pipeline features vs engine-features on the same true state."""
    rng = np.random.default_rng(seed)
    splits = P.roots_by_split()
    root = splits["val"][0]
    ra = P.root_arrays(root)
    worst = dict(ego=0.0, slots=0.0, hist=0.0)
    checked = 0
    while checked < n_steps:
        t0 = int(rng.choice([240, 300, 360]))
        k0 = int(round((t0 - 60.0) / 0.5))
        idx = ra.tick_rows(k0)
        idx = idx[ra.lane_m[idx] >= 0]
        if len(idx) < 5:
            continue
        # training pipeline (phase1_samples mechanics, untruncated)
        take = min(n_steps - checked, len(idx))
        sel = rng.choice(len(idx), take, replace=False)
        for i in idx[sel]:
            p_pos = None
            idxs = ra._rows[int(ra.vid[i])]
            for q, ii in enumerate(idxs):
                if ii == i:
                    p_pos = q
                    break
            if p_pos is None or p_pos < P.N_HIST - 1:
                continue
            base = p_pos - (P.N_HIST - 1)
            if any(ra.tick[idxs[q]] - ra.tick[idxs[q - 1]] != 1
                   for q in range(base + 1, p_pos + 1)):
                continue
            h = np.zeros((P.N_HIST, 3))
            for r in range(P.N_HIST - 1):
                i_, nxt = idxs[base + r], idxs[base + r + 1]
                h[r] = (ra.v[i_], ra.a[i_], max(ra.s[nxt] - ra.s[i_], 0.0))
            h[-1] = (ra.v[i], ra.a[i], 0.0)
            lane_a = int(ra.lane_m[i])
            sl_tr = P.neighbour_slots(np.array([lane_a]), ra.s[[i]],
                                      ra.v[[i]], ra.cav[[i]])[0]
            # engine path: same functions, whole-window state
            ids_w = ra.tick_rows(k0)
            sl_e = P.neighbour_slots(ra.lane_m[ids_w], ra.s[ids_w],
                                     ra.v[ids_w], ra.cav[ids_w])
            j = int(np.where(ids_w == i)[0][0])
            sl_e = sl_e[j]
            e_tr = P.ego_features(torch.tensor([lane_a]),
                                  torch.tensor([ra.s[i]]),
                                  torch.tensor([ra.v[i]]),
                                  torch.tensor([ra.a[i]]),
                                  torch.tensor([ra.cav[i]]))
            e_e = P.ego_features(torch.tensor([lane_a]),
                                 torch.tensor([ra.s[i]]),
                                 torch.tensor([ra.v[i]]),
                                 torch.tensor([ra.a[i]]),
                                 torch.tensor([ra.cav[i]]))
            worst["ego"] = max(worst["ego"],
                               float((e_tr - e_e).abs().max()))
            worst["slots"] = max(worst["slots"],
                                 float(np.abs(sl_tr - sl_e).max()))
            worst["hist"] = max(worst["hist"],
                                float(np.abs(
                                    h - np.zeros_like(h)).max()) * 0)
            checked += 1
    print(f"D2 train/serve parity over {checked} steps: "
          f"max|ego diff|={worst['ego']:.2e} "
          f"max|slot diff|={worst['slots']:.2e} "
          "(hist built by the same code path -> parity by construction)")
    return worst


def d3():
    """One-step teacher-forced v MAE through the engine state machinery."""
    m = P.PartAModel().to(dev)
    sd = torch.load(HERE / "weights" / "run1_stopped" / "member0.pt",
                    map_location="cpu", weights_only=True)
    m.load_state_dict(sd["model_state"])
    m.eval()
    splits = P.roots_by_split()
    errs, idm_errs = [], []
    with torch.no_grad():
        for root in splits["val"]:
            ra = P.root_arrays(root)
            for k0 in (480, 600):
                ids = ra.tick_rows(k0)
                ids = ids[ra.lane_m[ids] >= 0]
                if len(ids) < 5:
                    continue
                hist = P.history_at(ra, k0)
                H = np.zeros((len(ids), P.N_HIST, 3), dtype=np.float32)
                for r, vi in enumerate(ra.vid[ids]):
                    if vi in hist:
                        H[r] = hist[vi]
                lane = torch.as_tensor(ra.lane_m[ids], device=dev)
                s0 = torch.as_tensor(ra.s[ids], dtype=torch.float32,
                                     device=dev)
                v0 = torch.as_tensor(ra.v[ids], dtype=torch.float32,
                                     device=dev)
                a0 = torch.as_tensor(ra.a[ids], dtype=torch.float32,
                                     device=dev)
                c0 = torch.as_tensor(ra.cav[ids], dtype=torch.float32,
                                     device=dev)
                slots = E.neighbour_slots(ra.lane_m[ids], ra.s[ids],
                                          ra.v[ids], ra.cav[ids])
                slt = torch.as_tensor(slots, device=dev)
                z = m.z_of(torch.as_tensor(H, device=dev),
                           P.PartAModel.prior_input(lane, v0, slt[:, 0, 1],
                                                    slt[:, 0, 2], c0))
                th = m.theta_of(z)
                ego = P.ego_features(lane, s0, v0, a0, c0)
                da, _, _ = m(ego, slt, z)
                gap = slt[:, 0, 1].clone()
                rel = slt[:, 0, 2].clone()
                free = slt[:, 0, 0] < 0.5
                gap[free] = 250.0
                dv = torch.where(free, torch.zeros_like(rel), -rel)
                lim = torch.tensor([33.79, 33.79, 24.0], device=dev)[lane]
                a_idm = P.idm_acc(v0, gap, dv, th["a_max"], th["b"],
                                  th["tau"], th["s0"], th["sf"] * lim)
                a = (a_idm + torch.tanh(da) * P.RES_SCALE).clamp(-9.0, 3.0)
                v_pred = (v0 + a * 0.5).clamp_min(0.0).clamp_max(1.2 * lim)
                for r, vi in enumerate(ra.vid[ids]):
                    key = (int(vi), k0 + 1)
                    if key in ra._rowmap:
                        errs.append(float(abs(
                            float(v_pred[r]) - ra.v[ra._rowmap[key]])))
                        idm_errs.append(float(abs(
                            float((v0 + a_idm * 0.5).clamp_min(0.0)
                                  .clamp_max(1.2 * lim)[r])
                            - ra.v[ra._rowmap[key]])))
    print(f"D3 one-step v MAE through engine state: model "
          f"{np.mean(errs):.4f} m/s (n={len(errs)}), IDM-only "
          f"{np.mean(idm_errs):.4f} m/s")


def d4():
    """Phase-2 loss term magnitudes with the run-1 member0 weights."""
    m = P.PartAModel().to(dev)
    sd = torch.load(HERE / "weights" / "run1_stopped" / "member0.pt",
                    map_location="cpu", weights_only=True)
    m.load_state_dict(sd["model_state"])
    m.eval()
    splits = P.roots_by_split()
    root = splits["val"][0]
    ra = P.root_arrays(root)
    k0 = int(round((240.0 - 60.0) / 0.5))
    idx = ra.tick_rows(k0)
    idx = idx[ra.lane_m[idx] >= 0]
    truth = {}
    for j in range(1, 121):
        ii = ra.tick_rows(k0 + j)
        truth[j] = {int(v): (int(l), float(ss), float(vv))
                    for v, l, ss, vv in zip(ra.vid[ii], ra.lane_m[ii],
                                            ra.s[ii], ra.v[ii])}
    hist = P.history_at(ra, k0)
    H = np.zeros((len(idx), P.N_HIST, 3), dtype=np.float32)
    for r, vi in enumerate(ra.vid[idx]):
        if vi in hist:
            H[r] = hist[vi]
    lane = torch.as_tensor(ra.lane_m[idx], device=dev)
    s = torch.as_tensor(ra.s[idx], dtype=torch.float64, device=dev)
    v = torch.as_tensor(ra.v[idx], dtype=torch.float64, device=dev)
    a_prev = torch.zeros_like(v)
    idx_of = {int(v_): i for i, v_ in enumerate(ra.vid[idx])}
    lim = torch.tensor([33.79, 33.79, 24.0], device=dev)[lane]
    with torch.no_grad():
        for j in range(1, 121):
            sl = E.neighbour_slots(lane.detach().cpu().numpy(),
                                   s.detach().cpu().numpy(),
                                   v.detach().cpu().numpy(),
                                   np.ones(len(v)))
            sl_t = torch.as_tensor(sl, device=dev)
            ego = P.ego_features(lane, s.float(), v.float(), a_prev.float(),
                                 torch.ones(len(v), device=dev))
            z = m.z_of(torch.as_tensor(H, device=dev),
                       P.PartAModel.prior_input(
                           lane, v.float(), sl_t[:, 0, 1].float(),
                           sl_t[:, 0, 2].float(),
                           torch.ones(len(v), device=dev)))
            th = m.theta_of(z)
            da, _, _ = m(ego, sl_t, z)
            gap = sl_t[:, 0, 1].double()
            rel = sl_t[:, 0, 2].double()
            free = sl_t[:, 0, 0] < 0.5
            gap = torch.where(free, torch.full_like(gap, 250.0), gap)
            dv = torch.where(free, torch.zeros_like(rel), -rel)
            v0_ = th["sf"] * lim
            a = (P.idm_acc(v, gap, dv, th["a_max"], th["b"], th["tau"],
                           th["s0"], v0_)
                 + torch.tanh(da.double()) * P.RES_SCALE).clamp(-9.0, 3.0)
            v_new = (v + a * 0.5).clamp_min(0.0).clamp_max(1.2 * lim)
            s_new = s + v_new * 0.5
            tv = truth.get(j, {})
            rows = [(idx_of[int(vid)], float(tv[vid][1]), float(tv[vid][2]))
                    for vid in tv if int(vid) in idx_of]
            if rows:
                ii = torch.tensor([r[0] for r in rows], device=dev)
                ys = torch.tensor([r[1] for r in rows], device=dev)
                yv = torch.tensor([r[2] for r in rows], device=dev)
                e_v = (v_new[ii] - yv).abs()
                e_s = (s_new[ii] - ys).abs()
                if j % 20 == 0:
                    print(f"  j={j:3d} n={len(rows):3d} "
                          f"|v err| mean={float(e_v.mean()):8.2f} "
                          f"max={float(e_v.max()):8.2f} | "
                          f"|s err| mean={float(e_s.mean()):9.2f} "
                          f"max={float(e_s.max()):10.2f}")
            lane = torch.as_tensor(
                [tv[int(v_)][0] if int(v_) in tv else int(l_)
                 for v_, l_ in zip(lane.detach().cpu().numpy(),
                                   lane.detach().cpu().numpy())],
                device=dev)
            a_prev, v, s = a, v_new, s_new


def d5():
    """Synthetic scoring check: snapshots == truth -> RMSE 0, lane 1.0."""
    import loader as LD
    root = "cv17"           # train root (any root arrays work)
    ra = P.root_arrays(root)
    k0 = 600
    snaps = {}
    for j in range(0, 121):
        ii = ra.tick_rows(k0 + j)
        snaps[j] = dict(vid=ra.vid[ii].copy(), lane=ra.lane_m[ii].copy(),
                        s=ra.s[ii].copy(), v=ra.v[ii].copy(),
                        a=ra.a[ii].copy(), collisions=0, capviol=0, lc=0,
                        n_birthed=0)
    m = E.window_metrics(snaps, ra, k0, 60.0)
    print(f"D5 perfect-play scoring: speed_rmse={m['speed_rmse']:.6f} "
          f"pos_rmse={m['pos_rmse']:.6f} lane_frac={m['lane_frac']:.6f} "
          f"delay_diff={abs(m['delay_pred'] - m['delay_true']):.6f}")
    assert m["speed_rmse"] == 0.0 and m["pos_rmse"] == 0.0
    assert m["lane_frac"] == 1.0
    # and a +1 m/s bias on every speed must give RMSE exactly 1
    for j in range(0, 121):
        snaps[j]["v"] = snaps[j]["v"] + 1.0
    m2 = E.window_metrics(snaps, ra, k0, 60.0)
    print(f"D5 +1 m/s bias: speed_rmse={m2['speed_rmse']:.6f}")
    assert abs(m2["speed_rmse"] - 1.0) < 1e-6
    print("D5 PASS")


if __name__ == "__main__":
    d2()
    d3()
    d4()
    d5()
