"""R10B offline check of the learned follower courtesy head (CPU, no rollout, no SUMO, TRAIN only).

Phase ``cache`` (sharded): replay recorded TRAIN windows teacher-forced with the round-10b model
(``one_step_replay``, the same replay the courtesy loss uses; no gradient) and store, for every lane-1
HDV transition row, the courtesy head inputs, the no-courtesy model acceleration ``a_pre`` and the
recorded next-row acceleration (the loss target), plus DIAGNOSTIC-ONLY descriptors computed from the
recorded window (partner offset dA, partner distance to the lane end, partner wait proxy, time to the
partner's merge, C3-type control flag, own speed / time headway / closing speed for matched cells).
Descriptors never enter the head; they are used only to evaluate it like R9_YIELD.md.

Phase ``fit``:
  1. base-bias check: mean recorded-minus-model residual on target-follower rows vs C3-type controls;
  2. lr pilot (production schedule: 30 steps x 4 windows at 5e-5*s, then 60 steps x 5 windows at 3e-5*s,
     Adam) for s in {1, 10, 100, 1000}; rule: the s with the lowest held-out (episode-disjoint) courtesy
     loss;
  3. a long head-only fit (``--fit-steps``, default 400, Adam lr 1e-3) on the fit episodes;
  4. weight rule: geometric-mean gradient norm of the unweighted courtesy loss over the head parameters
     at the zero-initialised head, on production-like batches (4 windows); weight = G_cutin / G_courtesy;
  5. matched-excess report (before = head 0, after) on held-out episodes vs the recorded excess and the
     SUMO TRAIN reference -0.40 m/s^2 (dA 6-15 m), event profile, negative controls.

    python tools/r10b_courtesy_offline.py cache --ckpt <ckpt> --roots <r1> <r2> --out tmp_r10b/cache \
        --shard i --n-shards N [--windows-per-episode 3]
    python tools/r10b_courtesy_offline.py fit --out tmp_r10b/cache --n-shards N --report docs/r10b_offline_fit.json
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from tools.r10_attribution import load_model  # noqa: E402
from wms.data.c4_episodes import build_c4_datasets  # noqa: E402
from wms.data.collate import collate  # noqa: E402
from wms.model.follower_courtesy import FollowerCourtesyHead, follower_courtesy_pairs  # noqa: E402
from wms.train.inputs import build_targets, prepare_inputs  # noqa: E402
from wms.train.losses.cutin_reaction import _recorded, one_step_replay  # noqa: E402
from wms.train.losses.follower_courtesy import REPLAY_KEYS, courtesy_rows  # noqa: E402

CONFIG = "c4_wm_v10round10b.toml"
#: fix-B weighted cut-in gradient norm on its trainable set (docs/r10_weight_rule.json: 2.174 x 0.33918).
G_CUTIN_WEIGHTED = 2.174 * 0.3391763312051088
SUMO_TRAIN_EXCESS = -0.40


def datasets(cfg, roots, horizon_steps, stride_s):
    """Every episode of each (filtered, TRAIN-only) root in one split (role_of = train for all worlds)."""
    out = []
    for root in roots:
        doc = json.loads((Path(root) / "C4_EPISODES_MANIFEST.json").read_text(encoding="utf-8"))
        role_of = {e["world"]: "train" for e in doc["episodes"]}
        built = build_c4_datasets(
            root, horizon_steps=horizon_steps, history_s=cfg.data.history_s,
            history_stride_s=cfg.data.history_stride_s, entries_v2=False, stride_s=stride_s,
            k_slots=None, role_of=role_of, junction_fix=cfg.data.junction_fix,
            command_known=True, merge_pivot_history=cfg.model.merge_decision == "first_passage")
        out.append(built["train"])
    return out


def next_merge_steps(lane: np.ndarray, cont: np.ndarray) -> np.ndarray:
    """[T+1, K] step index at which the slot (now in lane 2) first appears in lane 1; inf if never."""
    t1, k = lane.shape
    out = np.full((t1, k), np.inf)
    for t in range(t1 - 2, -1, -1):
        stay = cont[t] & (lane[t] == 2)
        merge = stay & (lane[t + 1] == 1)
        out[t] = np.where(merge, t + 1, np.where(stay & (lane[t + 1] == 2), out[t + 1], np.inf))
    return out


def window_cache(model, tg, physics):
    _, collected = one_step_replay(model, tg, collect=REPLAY_KEYS)
    rows = courtesy_rows(tg, collected["courtesy_applies"])[0]               # [n, K]
    kin, lane, valid, _, continuous, hdv = _recorded(tg)
    kin, lane, valid, cont, hdv = kin[0], lane[0], valid[0], continuous[0], hdv[0]
    n = tg.n_steps
    l1 = hdv & (lane[:-1] == 1) & (lane[1:] == 1)                           # all lane-1 HDV rows
    lane_np, cont_np = lane.numpy(), np.concatenate([cont.numpy(), np.zeros_like(cont[:1].numpy())])
    nm = next_merge_steps(lane_np, cont_np)
    end = float(physics.merge_zone_s1_m)
    out = []
    for t in range(n):
        sel = l1[t].nonzero().flatten()
        if not len(sel):
            continue
        k_t = kin[t]
        s, v, length = k_t[:, 0], k_t[:, 4], k_t[:, 6]
        pairs = follower_courtesy_pairs(k_t[None], valid[t][None], physics)
        partner = pairs["partner"][0]
        has = rows[t]
        # own lane-1 leader (diagnostic cells)
        main = valid[t] & (lane[t] == 1)
        ds = s[None, :] - s[:, None]                                           # [i, j] = s_j - s_i
        ahead = main[None, :] & (ds > 0)
        dl = torch.where(ahead, ds - length[None, :], torch.full_like(ds, float("inf")))
        gap_l, lead = dl.min(-1)
        hw = torch.where(torch.isfinite(gap_l), gap_l / v.clamp_min(0.5), torch.full_like(v, 99.0))
        closing = torch.where(torch.isfinite(gap_l), v - v[lead], torch.zeros_like(v))
        # waiting ramp cars (diagnostic C3 definition of R9_YIELD.md)
        wait_ramp = valid[t] & (lane[t] == 2) & (s >= 0) & (s <= end)
        any_wait = bool(wait_ramp.any())
        rel = s[None, :] - s[:, None]                                          # s_ramp - s_i
        near = (wait_ramp[None, :] & (rel <= 30) & (rel >= -20)).any(-1)
        c3 = (~has) & torch.tensor(any_wait) & ~near
        p = partner
        da = torch.where(has, s[p] - s, torch.full_like(s, float("nan")))
        dend = torch.where(has, end - s[p], torch.full_like(s, float("nan")))
        tmerge = torch.tensor(nm[t, p.numpy()] - t, dtype=torch.float32) * float(tg.dt_s)
        tmerge = torch.where(has, tmerge, torch.full_like(tmerge, float("nan")))
        rec = kin[t + 1, :, 5]
        row = {
            "x": collected["courtesy_inputs"][0, t, sel].numpy().astype(np.float32),
            "pre": collected["accel_pre_courtesy_mps2"][0, t, sel].numpy().astype(np.float32),
            "rec": rec[sel].numpy().astype(np.float32),
            "has": has[sel].numpy(), "c3": c3[sel].numpy(),
            "da": da[sel].numpy(), "dend": dend[sel].numpy(), "tmerge": tmerge[sel].numpy(),
            "v": v[sel].numpy(), "hw": hw[sel].numpy(), "closing": closing[sel].numpy(),
            "s": s[sel].numpy(), "step": np.full(len(sel), t, np.int32),
            "slot": sel.numpy().astype(np.int32)}
        out.append(row)
    if not out:
        return None
    return {k: np.concatenate([r[k] for r in out]) for k in out[0]}


def cache(args):
    torch.set_num_threads(args.threads)
    cfg, settings, model, report = load_model(CONFIG, args.ckpt)
    assert model.follower_courtesy is not None
    n_steps = int(round(args.horizon_s / 0.5))
    sets = datasets(cfg, args.roots, n_steps, args.stride_s)
    path = Path(f"{args.out}.shard{args.shard}.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    for si, ds in enumerate(sets):
        by_ep: dict[int, list[int]] = {}
        for i, (ep, _tick) in enumerate(ds.positions):
            by_ep.setdefault(ep, []).append(i)
        rng = np.random.default_rng(1000 + si)
        for ep in sorted(by_ep):
            idx = by_ep[ep]
            pick = sorted(rng.choice(idx, size=min(args.windows_per_episode, len(idx)), replace=False))
            name = ds._loader.entries[ep]["episode"]
            jobs.extend((si, int(i), name) for i in pick)
    jobs = jobs[args.shard::args.n_shards]
    done, t0 = [], time.time()
    for si, i, name in jobs:
        ds = sets[si]
        batch = collate([ds[i]])
        inp = prepare_inputs(batch, model, settings, torch.device("cpu"), n_steps)
        tg = build_targets(inp, batch, settings, float(model.physics.dt_s), torch.device("cpu"),
                           physics=model.physics)
        tg.prepared["cutin_reaction_inputs"] = {
            "history": inp.hist, "history_mask": inp.hist_mask,
            "driver_history": inp.observed_hist, "driver_history_mask": inp.observed_hist_mask}
        with torch.no_grad():
            packed = window_cache(model, tg, model.physics)
        if packed is not None:
            packed["episode"] = name
            packed["window"] = f"{name}:{i}"
            done.append(packed)
        print(f"shard {args.shard}: {len(done)}/{len(jobs)} windows {time.time() - t0:.0f}s", flush=True)
        with open(path, "wb") as stream:
            pickle.dump({"windows": done, "init_report": report, "complete": False}, stream)
    with open(path, "wb") as stream:
        pickle.dump({"windows": done, "init_report": report, "complete": True}, stream)


# ------------------------------------------------------------------------------------------- fit
def load_windows(out, n_shards, extra=()):
    """Windows of ``out`` shards plus ``extra`` caches given as ``prefix:n_shards`` (partial shards allowed)."""
    windows = []
    for prefix, count in ((out, n_shards), *((e.rsplit(":", 1)[0], int(e.rsplit(":", 1)[1])) for e in extra)):
        for i in range(count):
            p = Path(f"{prefix}.shard{i}.pkl")
            if p.is_file():
                windows.extend(pickle.load(open(p, "rb"))["windows"])
    return windows


def head_loss(head, w, delta, amin, amax):
    x = torch.from_numpy(w["x"][w["has"]])
    pre = torch.from_numpy(w["pre"][w["has"]])
    rec = torch.from_numpy(w["rec"][w["has"]])
    pred = (pre + head(x)).clamp(amin, amax)
    return F.huber_loss(pred, rec, delta=delta)


def batch_of(windows):
    keys = ("x", "pre", "rec", "has")
    return {k: np.concatenate([w[k] for w in windows]) for k in keys}


def eval_loss(head, windows, delta, amin, amax):
    with torch.no_grad():
        b = batch_of(windows)
        return float(head_loss(head, b, delta, amin, amax))


def train_head(windows, schedule, delta, amin, amax, seed=0, hidden=32, log_every=0):
    torch.manual_seed(seed)
    head = FollowerCourtesyHead(hidden=hidden)
    opt = torch.optim.Adam(head.parameters(), lr=schedule[0][1])
    rng = np.random.default_rng(seed)
    step = 0
    for n_steps, lr, per_batch in schedule:
        for group in opt.param_groups:
            group["lr"] = lr
        for _ in range(n_steps):
            pick = rng.choice(len(windows), size=min(per_batch, len(windows)), replace=False)
            b = batch_of([windows[i] for i in pick])
            if not b["has"].any():
                continue
            opt.zero_grad()
            loss = head_loss(head, b, delta, amin, amax)
            loss.backward()
            opt.step()
            step += 1
            if log_every and step % log_every == 0:
                print(f"  step {step} loss {float(loss):.4f}", flush=True)
    return head


def predict(head, w):
    with torch.no_grad():
        h = head(torch.from_numpy(w["x"])).numpy()
    return np.where(w["has"], h, 0.0)


def cells(v, hw, closing):
    vb = np.digitize(v, [1, 3, 6, 10, 15, 20, 25, 30])
    hb = np.digitize(np.minimum(hw, 99), [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    cb = np.digitize(closing, [-3, -1, -0.3, 0.3, 1, 3])
    return vb * 100 + hb * 10 + cb


def matched_excess(values, all_rows, sel, ctrl, min_ctrl=20):
    """Mean of values[sel] minus the same-cell mean of values[ctrl] (cells: speed x headway x closing)."""
    c = all_rows["cell"]
    means = {}
    cc, vv = c[ctrl], values[ctrl]
    order = np.argsort(cc)
    cc, vv = cc[order], vv[order]
    uniq, start, counts = np.unique(cc, return_index=True, return_counts=True)
    for u, s0, n in zip(uniq, start, counts):
        if n >= min_ctrl:
            means[u] = vv[s0:s0 + n].mean()
    idx = np.nonzero(sel)[0]
    ok = np.array([c[i] in means for i in idx], bool)
    if not ok.any():
        return float("nan"), 0, 0.0
    exc = values[idx[ok]] - np.array([means[c[i]] for i in idx[ok]])
    return float(exc.mean()), int(ok.sum()), float(ok.mean())


def report_excess(head, windows, label):
    rows = {k: np.concatenate([w[k] for w in windows]) for k in
            ("pre", "rec", "has", "c3", "da", "dend", "tmerge", "v", "hw", "closing")}
    rows["cell"] = cells(rows["v"], rows["hw"], rows["closing"])
    h = np.concatenate([predict(head, w) for w in windows]) if head is not None else np.zeros_like(rows["pre"])
    after = rows["pre"] + h
    has, c3 = rows["has"], rows["c3"]
    da, dend, tm = rows["da"], rows["dend"], rows["tmerge"]
    out = {"label": label, "n_rows": int(len(has)), "n_partner_rows": int(has.sum()),
           "n_c3_rows": int(c3.sum())}
    sets = {
        "dA_6_15": has & (da >= 6) & (da < 15),
        "dA_lt20_dend_lt10": has & (da < 20) & (dend < 10),
        "dA_lt20_dend_10_20": has & (da < 20) & (dend >= 10) & (dend < 20),
        "dA_lt20_dend_gt40": has & (da < 20) & (dend > 40),
        "alongside_dA_lt6": has & (da < 6),
        "far_dA_gt30": has & (da > 30),
    }
    for name, sel in sets.items():
        rec_e, n, cov = matched_excess(rows["rec"], rows, sel, c3)
        base_e, _, _ = matched_excess(rows["pre"], rows, sel, c3)
        aft_e, _, _ = matched_excess(after, rows, sel, c3)
        out[name] = {"n": n, "coverage": cov, "recorded_excess": rec_e,
                     "model_excess_before": base_e, "model_excess_after": aft_e,
                     "head_mean": float(h[sel].mean()) if sel.any() else float("nan")}
    prof = {}
    for lo in (-10, -8, -6, -5, -4, -3, -2, -1.5, -1, -0.5):
        sel = has & (tm > -lo - 0.25) & (tm <= -lo + 0.25)
        # tmerge is time UNTIL the merge (>0); event time t - t_m = -tmerge
        rec_e, n, _ = matched_excess(rows["rec"], rows, sel, c3)
        aft_e, _, _ = matched_excess(after, rows, sel, c3)
        base_e, _, _ = matched_excess(rows["pre"], rows, sel, c3)
        prof[str(lo)] = {"n": n, "recorded": rec_e, "before": base_e, "after": aft_e}
    out["event_profile_t_minus_tm_s"] = prof
    # base-bias check: recorded - model residual on target-follower rows vs C3 controls (cell-matched)
    resid = rows["rec"] - rows["pre"]
    out["base_residual_mean"] = {
        "partner_rows": float(resid[has].mean()) if has.any() else float("nan"),
        "partner_dA_6_15": float(resid[sets["dA_6_15"]].mean()) if sets["dA_6_15"].any() else float("nan"),
        "c3_controls": float(resid[c3].mean()) if c3.any() else float("nan"),
        "partner_far_dA_gt30": float(resid[sets["far_dA_gt30"]].mean()) if sets["far_dA_gt30"].any() else float("nan")}
    return out


def fit(args):
    torch.set_num_threads(args.threads)
    windows = load_windows(args.out, args.n_shards, args.extra)
    eps = sorted({w["episode"] for w in windows})
    held = set(eps[::4])
    fit_w = [w for w in windows if w["episode"] not in held]
    held_w = [w for w in windows if w["episode"] in held]
    from wms.config.load import load_train_config
    cfg = load_train_config(ROOT / "configs" / CONFIG, ["runtime.device=cpu"])
    delta = float(cfg.loss.follower_courtesy.huber_delta_mps2)
    from wms.cli.assemble import build_physics
    phys = build_physics(cfg)
    amin, amax = float(phys.accel_min_mps2), float(phys.accel_max_mps2)
    res = {"n_windows": len(windows), "n_fit_windows": len(fit_w), "n_held_windows": len(held_w),
           "n_episodes": len(eps), "held_episodes": sorted(held), "huber_delta": delta}
    print(json.dumps({k: res[k] for k in ("n_windows", "n_fit_windows", "n_held_windows")}), flush=True)
    # 2. lr pilot on the production schedule
    pilot = {}
    for scale in (1, 10, 100, 1000):
        sched = [(30, 5e-5 * scale, 4), (60, 3e-5 * scale, 5)]
        head = train_head(fit_w, sched, delta, amin, amax, seed=0)
        pilot[str(scale)] = {"held_loss": eval_loss(head, held_w, delta, amin, amax),
                             "held_dA_6_15": report_excess(head, held_w, "p")["dA_6_15"]}
        print("pilot", scale, pilot[str(scale)]["held_loss"], flush=True)
    zero = FollowerCourtesyHead()
    pilot["zero_head_held_loss"] = eval_loss(zero, held_w, delta, amin, amax)
    best = min((s for s in ("1", "10", "100", "1000")), key=lambda s: pilot[s]["held_loss"])
    res["lr_pilot"] = pilot
    res["lr_scale_chosen"] = float(best)
    res["lr_rule"] = ("largest-gain rule: the scale in {1,10,100,1000} with the lowest held-out (episode-disjoint TRAIN) "
                      "courtesy loss after the production schedule 30 x 4 windows @5e-5*s + 60 x 5 windows @3e-5*s (Adam)")
    # production-schedule head at the chosen scale
    prod = train_head(fit_w, [(30, 5e-5 * float(best), 4), (60, 3e-5 * float(best), 5)], delta, amin, amax, seed=0)
    # 3. long fit
    long_head = train_head(fit_w, [(args.fit_steps, 1e-3, 4)], delta, amin, amax, seed=0, log_every=50)
    res["long_fit"] = {"steps": args.fit_steps, "lr": 1e-3,
                       "held_loss": eval_loss(long_head, held_w, delta, amin, amax),
                       "fit_loss": eval_loss(long_head, fit_w, delta, amin, amax)}
    # 4. weight rule (zero-init head, production-like batches of 4 windows)
    rng = np.random.default_rng(7)
    norms = []
    for _ in range(8):
        torch.manual_seed(0)
        head = FollowerCourtesyHead()
        pick = rng.choice(len(fit_w), size=4, replace=False)
        loss = head_loss(head, batch_of([fit_w[i] for i in pick]), delta, amin, amax)
        grads = torch.autograd.grad(loss, list(head.parameters()), allow_unused=True)
        norms.append(float(torch.sqrt(sum((g.square().sum() for g in grads if g is not None)))))
    g_court = float(np.exp(np.mean(np.log(norms))))
    res["weight_rule"] = {"g_courtesy_at_weight_1_per_batch": norms, "g_courtesy_geomean": g_court,
                          "g_cutin_weighted": G_CUTIN_WEIGHTED,
                          "weight_geomean_ratio": G_CUTIN_WEIGHTED / g_court,
                          "chosen_weight": float(f"{G_CUTIN_WEIGHTED / g_court:.4g}")}
    # 5. matched excess reports (held-out episodes)
    res["held_before"] = report_excess(None, held_w, "head=0 (round-9 model)")
    res["held_after_production_schedule"] = report_excess(prod, held_w, f"production schedule, scale {best}")
    res["held_after_long_fit"] = report_excess(long_head, held_w, f"{args.fit_steps} steps lr 1e-3")
    res["fit_after_long_fit"] = report_excess(long_head, fit_w, "fit episodes, long fit")
    res["sumo_train_reference_dA_6_15"] = SUMO_TRAIN_EXCESS
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    torch.save({"production": prod.state_dict(), "long": long_head.state_dict()},
               Path(args.out).with_suffix(".heads.pt"))
    for key in ("held_before", "held_after_production_schedule", "held_after_long_fit"):
        r = res[key]["dA_6_15"]
        print(key, json.dumps(r), flush=True)
    print("weight", res["weight_rule"]["chosen_weight"], "lr scale", best, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--roots", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--horizon-s", type=float, default=120.0)
    c.add_argument("--stride-s", type=float, default=30.0)
    c.add_argument("--windows-per-episode", type=int, default=3)
    c.add_argument("--shard", type=int, default=0)
    c.add_argument("--n-shards", type=int, default=1)
    c.add_argument("--threads", type=int, default=2)
    f = sub.add_parser("fit")
    f.add_argument("--out", required=True)
    f.add_argument("--n-shards", type=int, default=1)
    f.add_argument("--report", required=True)
    f.add_argument("--fit-steps", type=int, default=400)
    f.add_argument("--extra", nargs="*", default=[], help="more caches as prefix:n_shards")
    f.add_argument("--threads", type=int, default=8)
    args = p.parse_args(argv)
    cache(args) if args.cmd == "cache" else fit(args)


if __name__ == "__main__":
    main()
