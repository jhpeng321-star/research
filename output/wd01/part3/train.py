"""WD-01 part 3 training: phase 1 (teacher-forced 0.5 s step), phase 2
(multi-step rollout fine-tuning, curriculum 10/40/120 steps), ensemble
M=3 (member 0 = full train-minus-val; members 1..2 bootstrap roots),
deterministic validation split, val-based stage selection logged to
results/val_selection.json.  Zero simulation.

Declared CPU-budget reductions (reported in REGISTRATION/REPORT):
<=20000 phase-1 samples per root; 3 phase-1 epochs; curriculum
(10, 200)/(40, 120)/(120, 60) iterations; phase-2 interaction slots
recomputed per step WITHOUT gradient (interaction truncation); lanes
teacher-forced during phase-2 training (the lane head is trained by CE
and evaluated greedily at holdout); z frozen per window from the t0 truth
history (newborns: learned prior); phase-1 loss supervises v' only (s'
follows through the measured integrator), s is supervised in phase 2.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import p3lib as P
import engine as E
import loader as LD
import pdl as PD

T0 = time.time()
import os
CURRICULUM = [(10, 200), (40, 120), (120, 60)]
EPOCHS_P1 = 4
BATCH = 8192
MEMBERS = 3
SMOKE = os.environ.get('P3_SMOKE', '') == '1'
if SMOKE:
    CURRICULUM = [(10, 3), (40, 2), (120, 1)]
    EPOCHS_P1 = 1
    MEMBERS = 1
    BATCH = 4096
LIM = torch.tensor(P.LANE_LIMIT)


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def rollout_loss(model, wins, dev, n_steps):
    """Differentiable multi-step rollout (slots without grad, lanes
    teacher-forced, z frozen at t0).  Returns mean loss over steps."""
    tot = torch.zeros((), device=dev)
    n = 0
    for (ids0, lane0, s0, v0, cav0, H0, truth) in wins:
        sl0 = E.neighbour_slots(lane0.detach().cpu().numpy(),
                                s0.detach().cpu().numpy(),
                                v0.detach().cpu().numpy(),
                                np.ones(len(v0)))
        prior_in = P.PartAModel.prior_input(
            lane0, v0.float(), torch.as_tensor(sl0[:, 0, 1], device=dev),
            torch.as_tensor(sl0[:, 0, 2], device=dev),
            torch.ones(len(v0), device=dev))
        z = model.z_of(H0, prior_in)
        lane, s, v = lane0.clone(), s0.clone(), v0.clone()
        a_prev = torch.zeros_like(v)
        idx_of = {int(v_): i for i, v_ in enumerate(ids0)}
        lim = torch.tensor(P.LANE_LIMIT, device=dev)[lane]
        vcap = 1.2 * lim
        for j in range(1, n_steps + 1):
            sl = E.neighbour_slots(lane.detach().cpu().numpy(),
                                   s.detach().cpu().numpy(),
                                   v.detach().cpu().numpy(),
                                   np.ones(len(v)))
            sl_t = torch.as_tensor(sl, device=dev)
            ego = P.ego_features(lane, s.float(), v.float(), a_prev.float(),
                                 torch.ones(len(v), device=dev))
            da, lc_logits, aux = model(ego, sl_t, z)
            th = model.theta_of(z)
            gap = torch.as_tensor(sl[:, 0, 1], device=dev).double()
            rel = torch.as_tensor(sl[:, 0, 2], device=dev).double()
            free = torch.as_tensor(sl[:, 0, 0] < 0.5, device=dev)
            gap = torch.where(free, torch.full_like(gap, P.GAP_MAX), gap)
            dv = torch.where(free, torch.zeros_like(rel), -rel)
            v0_ = th["sf"] * lim
            a = (P.idm_acc(v, gap, dv, th["a_max"], th["b"], th["tau"],
                           th["s0"], v0_)
                 + torch.tanh(da.double()) * P.RES_SCALE).clamp(-9.0, 3.0)
            v_new = (v + a * P.DT).clamp_min(0.0).clamp_max(vcap)
            s_new = s + v_new * P.DT
            tv = truth.get(j)
            if tv:
                rows = [(idx_of[int(vid)], float(tv[vid][1]),
                         float(tv[vid][2]))
                        for vid in tv if int(vid) in idx_of]
                if rows:
                    ii = torch.tensor([r[0] for r in rows], device=dev)
                    yv = torch.tensor([r[1] for r in rows], device=dev)
                    ys = torch.tensor([r[2] for r in rows], device=dev)
                    # v2: Huber terms only.  The lane CE was removed after
                    # it exploded (~1e9 via masked -1e9 targets once the
                    # predicted lane diverged from truth); the lane head
                    # is trained by CE in phase 1 and evaluated greedily.
                    loss = (F.huber_loss(v_new[ii].float(), yv, delta=3.0)
                            + 0.25 * F.huber_loss(s_new[ii].float(), ys,
                                                  delta=10.0))
                    tot = tot + loss
                    n += 1
            if tv:
                lane = torch.as_tensor(
                    [tv[int(v_)][0] if int(v_) in tv else int(l_)
                     for v_, l_ in zip(lane.detach().cpu().numpy(),
                                       lane.detach().cpu().numpy())],
                    device=dev)
            a_prev, v, s = a, v_new, s_new
    return tot / max(n, 1)


def train_phase1(model, data, dev, epochs=EPOCHS_P1, seed=0):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(data["y_v"])
    lim = torch.tensor(P.LANE_LIMIT, device=dev)
    for ep in range(epochs):
        perm = torch.as_tensor(
            np.random.default_rng(100 + seed * 13 + ep).permutation(n))
        tot, nb = 0.0, 0
        for b0 in range(0, n, BATCH):
            ii = perm[b0:b0 + BATCH]
            hist = torch.as_tensor(data["hist"][ii], device=dev)
            ego = torch.as_tensor(data["ego"][ii], device=dev)
            slots = torch.as_tensor(data["slots"][ii], device=dev)
            prin = torch.as_tensor(data["prin"][ii], device=dev)
            yv = torch.as_tensor(data["y_v"][ii], device=dev)
            yl = torch.as_tensor(data["y_lane"][ii], device=dev)
            ya = torch.as_tensor(data["y_aux"][ii], device=dev)
            z = model.z_of(hist, P.prior_from_raw(prin))
            th = model.theta_of(z)
            ego_raw = data["ego"][ii]
            lane = torch.as_tensor(ego_raw[:, 0].astype(np.int64),
                                   device=dev)
            v_raw = torch.as_tensor(ego_raw[:, 2], dtype=torch.float32,
                                    device=dev)
            cav_raw = torch.as_tensor(ego_raw[:, 4], dtype=torch.float32,
                                      device=dev)
            ego = P.ego_features(lane,
                                 torch.as_tensor(ego_raw[:, 1],
                                                 dtype=torch.float32,
                                                 device=dev),
                                 v_raw,
                                 torch.as_tensor(ego_raw[:, 3],
                                                 dtype=torch.float32,
                                                 device=dev),
                                 cav_raw)
            da, lc_logits, aux = model(ego, slots, z)
            gap = slots[:, 0, 1].double().float()
            rel = slots[:, 0, 2].double().float()
            free = slots[:, 0, 0] < 0.5
            gap = torch.where(free, torch.full_like(gap, P.GAP_MAX), gap)
            dv = -rel
            v_ = v_raw.float()
            v0_ = th["sf"] * lim[lane]
            a = (P.idm_acc(v_, gap, dv, th["a_max"], th["b"],
                           th["tau"], th["s0"], v0_)
                 + torch.tanh(da) * P.RES_SCALE).clamp(-9.0, 3.0)
            v_pred = (v_ + a * P.DT).clamp_min(0.0).clamp_max((1.2 * lim[lane]).detach())
            loss = (F.huber_loss(v_pred.float(), yv, delta=3.0)
                    + 0.3 * F.cross_entropy(lc_logits, yl)
                    + 0.1 * (F.binary_cross_entropy_with_logits(
                        aux[:, 0], ya[:, 0])
                        + F.binary_cross_entropy_with_logits(
                            aux[:, 1], ya[:, 1])))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        log(f"  phase1 ep{ep}: loss {tot / max(nb, 1):.4f}")


def build_phase2_windows(roots, dev, max_windows=64):
    wins = []
    for root in roots:
        ra = P.root_arrays(root)
        for t0 in range(120, 481, 60):
            if len(wins) >= max_windows:
                return wins
            k0 = int(round((t0 - PD.WINDOW_T0) / 0.5))
            idx = ra.tick_rows(k0)
            idx = idx[ra.lane_m[idx] >= 0]
            if len(idx) < 5:
                continue
            truth = {}
            for j in range(1, 241):
                ii = ra.tick_rows(k0 + j)
                if len(ii):
                    truth[j] = {int(v): (int(l), float(ss), float(vv))
                                for v, l, ss, vv in zip(ra.vid[ii],
                                                        ra.lane_m[ii],
                                                        ra.s[ii], ra.v[ii])}
            hist = P.history_at(ra, k0)
            H = np.zeros((len(idx), P.N_HIST, 3), dtype=np.float32)
            for r, vi in enumerate(ra.vid[idx]):
                if vi in hist:
                    H[r] = hist[vi]
            wins.append((ra.vid[idx].copy(),
                         torch.as_tensor(ra.lane_m[idx], dtype=torch.long,
                                         device=dev),
                         torch.as_tensor(ra.s[idx], dtype=torch.float64,
                                         device=dev),
                         torch.as_tensor(ra.v[idx], dtype=torch.float64,
                                         device=dev),
                         torch.ones(len(idx), device=dev),
                         torch.as_tensor(H, device=dev), truth))
    return wins


def val_rollout_rmse(model, splits, dev):
    model.eval()
    se, cnt = 0.0, 0
    tmp = HERE / "results" / "_val_tmp.pt"
    torch.save({"model_state": model.state_dict(), "meta": {}}, tmp)
    for root in splits["val"]:
        for t0 in (240.0, 360.0):
            r = E.Rollout(root, t0, kind="model", weights=tmp, dev=dev)
            r.run()
            m = E.window_metrics(r.snapshots, r.ra, r.k0, 60.0)
            se += m["speed_rmse"] ** 2 * m["matched_vehicle_steps"]
            cnt += m["matched_vehicle_steps"]
    return (se / max(cnt, 1)) ** 0.5


def main():
    man, man_sha = P.guards()
    log(f"guards ok (p2C manifest {man_sha[:12]}...)")
    dev = P.device()
    splits = P.roots_by_split()
    log(f"splits: train={len(splits['train'])} val={splits['val']} "
        f"holdout={len(splits['holdout'])} (holdout untouched)")

    cache = HERE / "results" / "phase1_cache"
    cache.mkdir(parents=True, exist_ok=True)
    per_root = {}
    for root in splits["train"] + splits["val"]:
        f = cache / f"{root}.npz"
        if not f.exists():
            log(f"phase-1 samples: {root}")
            d = E.phase1_samples([root], max_per_root=(2000 if SMOKE else 20000))
            np.savez_compressed(f, **d)
        zz = np.load(f)
        per_root[root] = {k: zz[k] for k in zz.files}
    log("phase-1 caches ready")

    keys = ("hist", "ego", "slots", "prin", "y_v", "y_s", "y_d", "y_lane", "y_aux")

    def member_dataset(roots):
        return {k: np.concatenate([per_root[r][k] for r in roots])
                for k in keys}

    val_log = dict(selection_rule="lowest val rollout speed-RSM at H=60 s "
                   "across {phase1, curric10, curric40, curric120}; "
                   "degradation >2% stops the curriculum and reverts",
                   entries=[])
    best = dict(rmse=None, state=None, stage="none")

    def eval_and_record(model, tag):
        nonlocal best
        rmse = val_rollout_rmse(model, splits, dev)
        val_log["entries"].append(dict(member=0, stage=tag,
                                       val_speed_rmse=round(rmse, 4)))
        if best["rmse"] is None or rmse <= best["rmse"] * 1.02:
            best = dict(rmse=(rmse if best["rmse"] is None
                              else min(best["rmse"], rmse)),
                        state={k: v.clone() for k, v in
                               model.state_dict().items()},
                        stage=tag)
            return True
        return False

    # ---------------- member 0: pilot with val-based selection ----------
    torch.manual_seed(0)
    np.random.seed(0)
    train0 = splits["train"]
    model = P.PartAModel().to(dev)
    data0 = member_dataset(train0)
    log(f"phase-1 samples: {len(data0['y_v'])}")
    train_phase1(model, data0, dev, seed=0)
    if eval_and_record(model, "phase1"):
        log(f"val rmse phase1: {best['rmse']:.4f} (kept)")
    wins0 = build_phase2_windows(train0, dev, 64)
    log(f"phase-2 windows: {len(wins0)}")
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    for steps, iters in CURRICULUM:
        t_stage = time.time()
        log(f"phase2 stage L={steps}: {iters} iterations starting")
        for it in range(iters):
            sel = [wins0[i] for i in torch.randint(0, len(wins0), (4,))]
            loss = rollout_loss(model, sel, dev, steps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 50 == 0:
                log(f"  phase2 L={steps} it={it}: loss {float(loss):.4f}")
        ok = eval_and_record(model, f"curric{steps}")
        log(f"val rmse after L={steps}: "
            f"{val_log['entries'][-1]['val_speed_rmse']:.4f} "
            f"({'kept' if ok else 'degraded'})")
        if not ok:
            model.load_state_dict(best["state"])
            log(f"reverted to {best['stage']}")
            break
    P.dump_json(HERE / "results" / "val_selection.json", val_log)
    (HERE / "weights" / "run3").mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best["state"], "meta": {
        "member": 0, "bootstrap": "full train-minus-val",
        "best_stage": best["stage"],
        "val_speed_rmse_h60": best["rmse"]}},
        HERE / "weights" / "run3" / "member0.pt")
    log(f"member0 saved (stage {best['stage']}, val {best['rmse']:.4f})")

    # ---------------- members 1..M-1: bootstrap roots -------------------
    for k in range(1, MEMBERS):
        rng = np.random.default_rng(1000 + k)
        boot = [train0[i] for i in rng.choice(len(train0), len(train0),
                                              replace=True)]
        log(f"member {k}: bootstrap roots (unique {len(set(boot))})")
        torch.manual_seed(1000 + k)
        model = P.PartAModel().to(dev)
        train_phase1(model, member_dataset(boot), dev, seed=k)
        wins = build_phase2_windows(boot, dev, 48)
        opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        for steps, iters in CURRICULUM:
            t_stage = time.time()
            log(f"member curriculum L={steps}: {iters} iterations starting")
            for it in range(iters):
                sel = [wins[i] for i in torch.randint(
                    0, max(len(wins), 1), (4,))]
                loss = rollout_loss(model, sel, dev, steps)
                opt.zero_grad()
                loss.backward()
                opt.step()
        torch.save({"model_state": model.state_dict(),
                    "meta": {"member": k, "bootstrap": boot,
                             "best_stage": best["stage"]}},
                   HERE / "weights" / "run3" / f"member{k}.pt")
        log(f"member {k} saved")

    log("TRAINING DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
