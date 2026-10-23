"""R11 offline check of the structural courtesy fix (CPU, teacher-forced, no rollout, no SUMO, TRAIN only).

Phase ``cache`` (sharded; same windows as tools/r10b_courtesy_offline.py for the same roots/shards):
replay recorded TRAIN windows teacher-forced with a version-2 courtesy model (``one_step_replay``, the
replay the loss uses; no gradient) and store, for every lane-1 HDV transition row, the 13 courtesy
inputs (the first 10 are the round-10b inputs), ``a_pre`` (plant-clamped for the controls in the fit),
the recorded next-row acceleration, the own-state matching features of the contrast term
(``matching_features``) and DIAGNOSTIC-ONLY descriptors (partner offset dA, distance to the lane end,
time to the partner's merge, C3-type control flag, own speed / headway / closing for R9_YIELD cells).

Phase ``fit``: four arms on the same windows, production schedule (30 x 4 windows @5e-5*s + 60 x 5
windows @3e-5*s, Adam), lr scale per the declared rule, plus a long fit:
  A  round-10b loss (absolute recorded fit), round-10b inputs        (reproduces the reviewer's problem)
  B  round-10b loss, R11 inputs
  C  R11 contrast loss, round-10b inputs
  D  R11 contrast loss, R11 inputs                                   (round 11)
and D started from A's production head (zero-expanded; the default round-11 init is the round-10b head).
Reports (held-out episodes and all windows): last-6-s event profile of the absolute prediction vs
recorded, matched excess (R9_YIELD cells, evaluation only), far partners / d_end > 40 / alongside
negative controls, head output quantiles.  Weight rule (gradient-norm parity) and lr rule for D.
"""
from __future__ import annotations

import argparse
import json
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
from tools.r10b_courtesy_offline import (G_CUTIN_WEIGHTED, cells, datasets,  # noqa: E402
                                         matched_excess, next_merge_steps)
from wms.data.collate import collate  # noqa: E402
from wms.model.follower_courtesy import (COURTESY_PAIR_F, FollowerCourtesyHead,  # noqa: E402
                                         follower_courtesy_pairs)
from wms.train.inputs import build_targets, prepare_inputs  # noqa: E402
from wms.train.losses.cutin_reaction import _recorded, one_step_replay  # noqa: E402
from wms.train.losses.follower_courtesy import (REPLAY_KEYS, FollowerCourtesyConfig,  # noqa: E402
                                                courtesy_rows, matched_control_residual,
                                                matching_features)

CONFIG = "c4_wm_v10round10b.toml"
OVERRIDES = ("model.follower_courtesy_version=2",)


def window_cache(model, tg, physics):
    _, collected = one_step_replay(model, tg, collect=REPLAY_KEYS)
    rows = courtesy_rows(tg, collected["courtesy_applies"])[0]
    kin_b, lane, valid, _, continuous, hdv = _recorded(tg)
    phi = matching_features(kin_b, valid, FollowerCourtesyConfig(
        weight=1.0, merge_lane=int(physics.merge_lane), zone_end_m=float(physics.merge_zone_s1_m),
        zone_start_m=float(physics.merge_zone_s0_m)))[0]
    kin, lane, valid, cont, hdv = kin_b[0], lane[0], valid[0], continuous[0], hdv[0]
    n = tg.n_steps
    l1 = hdv & (lane[:-1] == 1) & (lane[1:] == 1)
    lane_np, cont_np = lane.numpy(), np.concatenate([cont.numpy(), np.zeros_like(cont[:1].numpy())])
    nm = next_merge_steps(lane_np, cont_np)
    end = float(physics.merge_zone_s1_m)
    amin, amax = float(physics.accel_min_mps2), float(physics.accel_max_mps2)
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
        main = valid[t] & (lane[t] == 1)
        ds = s[None, :] - s[:, None]
        ahead = main[None, :] & (ds > 0)
        dl = torch.where(ahead, ds - length[None, :], torch.full_like(ds, float("inf")))
        gap_l, lead = dl.min(-1)
        hw = torch.where(torch.isfinite(gap_l), gap_l / v.clamp_min(0.5), torch.full_like(v, 99.0))
        closing = torch.where(torch.isfinite(gap_l), v - v[lead], torch.zeros_like(v))
        wait_ramp = valid[t] & (lane[t] == 2) & (s >= 0) & (s <= end)
        any_wait = bool(wait_ramp.any())
        rel = s[None, :] - s[:, None]
        near = (wait_ramp[None, :] & (rel <= 30) & (rel >= -20)).any(-1)
        c3 = (~has) & torch.tensor(any_wait) & ~near
        p = partner
        da = torch.where(has, s[p] - s, torch.full_like(s, float("nan")))
        dend = torch.where(has, end - s[p], torch.full_like(s, float("nan")))
        tmerge = torch.tensor(nm[t, p.numpy()] - t, dtype=torch.float32) * float(tg.dt_s)
        tmerge = torch.where(has, tmerge, torch.full_like(tmerge, float("nan")))
        pre = collected["accel_pre_courtesy_mps2"][0, t, sel]
        out.append({
            "x": collected["courtesy_inputs"][0, t, sel].numpy().astype(np.float32),
            "pre": pre.numpy().astype(np.float32),
            "pre_clamped": pre.clamp(amin, amax).numpy().astype(np.float32),
            "rec": kin[t + 1, sel, 5].numpy().astype(np.float32),
            "phi": phi[t, sel].numpy().astype(np.float32),
            "has": has[sel].numpy(), "c3": c3[sel].numpy(),
            "da": da[sel].numpy(), "dend": dend[sel].numpy(), "tmerge": tmerge[sel].numpy(),
            "v": v[sel].numpy(), "hw": hw[sel].numpy(), "closing": closing[sel].numpy(),
            "s": s[sel].numpy(), "step": np.full(len(sel), t, np.int32),
            "slot": sel.numpy().astype(np.int32)})
    if not out:
        return None
    return {k: np.concatenate([r[k] for r in out]) for k in out[0]}


def cache(args):
    torch.set_num_threads(args.threads)
    cfg, settings, model, report = load_model(CONFIG, args.ckpt, OVERRIDES)
    assert model.follower_courtesy is not None and model._tracks_courtesy_zone
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
KEYS = ("x", "pre", "pre_clamped", "rec", "has", "phi")


def load_windows(prefix, n_shards):
    windows = []
    for i in range(n_shards):
        p = Path(f"{prefix}.shard{i}.pkl")
        if p.is_file():
            windows.extend(pickle.load(open(p, "rb"))["windows"])
    return windows


def batch_of(windows):
    return {k: np.concatenate([w[k] for w in windows]) for k in KEYS}


def contrast_shift(b):
    """Kernel-matched control residual for the partner rows of the batch (the training term)."""
    has = b["has"]
    ctrl = ~has
    resid = b["rec"][ctrl] - b["pre_clamped"][ctrl]
    shift, neff = matched_control_residual(torch.from_numpy(b["phi"][has]),
                                           torch.from_numpy(b["phi"][ctrl]),
                                           torch.from_numpy(resid))
    return shift.numpy(), neff.numpy()


def head_loss(head, b, arm, delta, amin, amax, shift=None):
    has = b["has"]
    x = torch.from_numpy(b["x"][has][:, :head.n_in])
    pre = torch.from_numpy(b["pre"][has])
    target = torch.from_numpy(b["rec"][has])
    if arm["contrast"]:
        if shift is None:
            shift, _ = contrast_shift(b)
        target = target - torch.from_numpy(shift)
    pred = (pre + head(x)).clamp(amin, amax)
    return F.huber_loss(pred, target, delta=delta)


def n_inputs(arm):
    return 13 if arm["v2"] else COURTESY_PAIR_F


def expanded_copy(head_v1, hidden=32):
    """A version-2 head computing exactly ``head_v1`` (zero columns for the new inputs)."""
    head = FollowerCourtesyHead(hidden=hidden, version=2)
    state = head_v1.state_dict()
    w = torch.zeros_like(head.net[0].weight)
    w[:, :COURTESY_PAIR_F] = state["net.0.weight"]
    state = {**state, "net.0.weight": w}
    head.load_state_dict(state)
    return head


def train_head(windows, schedule, arm, delta, amin, amax, seed=0, init=None):
    torch.manual_seed(seed)
    head = init if init is not None else FollowerCourtesyHead(version=2 if arm["v2"] else 1)
    opt = torch.optim.Adam(head.parameters(), lr=schedule[0][1])
    rng = np.random.default_rng(seed)
    for n_steps, lr, per_batch in schedule:
        for group in opt.param_groups:
            group["lr"] = lr
        for _ in range(n_steps):
            pick = rng.choice(len(windows), size=min(per_batch, len(windows)), replace=False)
            b = batch_of([windows[i] for i in pick])
            if not b["has"].any():
                continue
            opt.zero_grad()
            head_loss(head, b, arm, delta, amin, amax).backward()
            opt.step()
    return head


def groups_of(windows, size=4):
    return [windows[i:i + size] for i in range(0, len(windows), size)]


def eval_loss(head, windows, arm, delta, amin, amax):
    """Mean loss over fixed groups of 4 windows (the contrast matches within each group, as in training)."""
    vals, ns = [], []
    with torch.no_grad():
        for g in groups_of(windows):
            b = batch_of(g)
            if b["has"].any():
                vals.append(float(head_loss(head, b, arm, delta, amin, amax)) * int(b["has"].sum()))
                ns.append(int(b["has"].sum()))
    return sum(vals) / max(sum(ns), 1)


def predict(head, w):
    if head is None:
        return np.zeros(len(w["has"]), np.float32)
    with torch.no_grad():
        h = head(torch.from_numpy(w["x"][:, :head.n_in])).numpy()
    return np.where(w["has"], h, 0.0)


def per_window_shift(windows):
    """Contrast target shift per partner row, matched within fixed groups of 4 windows."""
    out = {}
    for g in groups_of(windows):
        b = batch_of(g)
        shift = np.zeros(len(b["has"]), np.float32)
        if b["has"].any() and (~b["has"]).any():
            s, _ = contrast_shift(b)
            shift[b["has"]] = s
        lo = 0
        for w in g:
            out[w["window"]] = shift[lo:lo + len(w["has"])]
            lo += len(w["has"])
    return out


def report(head, windows, label, amin, amax, shifts):
    rows = {k: np.concatenate([w[k] for w in windows]) for k in
            ("pre", "rec", "has", "c3", "da", "dend", "tmerge", "v", "hw", "closing", "s")}
    rows["cell"] = cells(rows["v"], rows["hw"], rows["closing"], rows["s"])
    h = np.concatenate([predict(head, w) for w in windows])
    shift = np.concatenate([shifts[w["window"]] for w in windows])
    before = np.clip(rows["pre"], amin, amax)
    after = np.clip(rows["pre"] + h, amin, amax)
    has, c3 = rows["has"], rows["c3"]
    da, dend, tm = rows["da"], rows["dend"], rows["tmerge"]
    out = {"label": label, "n_rows": int(len(has)), "n_partner_rows": int(has.sum())}
    groups = {
        "dA_6_15": has & (da >= 6) & (da < 15),
        "dA_lt20_dend_lt10": has & (da < 20) & (dend < 10),
        "dA_lt20_dend_gt40": has & (da < 20) & (dend > 40),
        "alongside_dA_lt6": has & (da < 6),
        "far_dA_gt30": has & (da > 30),
        "all_partner": has,
    }
    for name, sel in groups.items():
        if not sel.any():
            continue
        rec_e, n, cov = matched_excess(rows["rec"], rows, sel, c3)
        aft_e, _, _ = matched_excess(after, rows, sel, c3)
        bef_e, _, _ = matched_excess(before, rows, sel, c3)
        out[name] = {"n": int(sel.sum()), "recorded": float(rows["rec"][sel].mean()),
                     "before": float(before[sel].mean()), "after": float(after[sel].mean()),
                     "abs_bias_after": float((after - rows["rec"])[sel].mean()),
                     "head_mean": float(h[sel].mean()),
                     "contrast_target_head": float((rows["rec"] - rows["pre"] - shift)[sel].mean()),
                     "excess_recorded": rec_e, "excess_before": bef_e, "excess_after": aft_e,
                     "excess_n": n, "excess_coverage": cov}
    prof = {}
    for lo in (-6, -5, -4, -3, -2, -1.5, -1, -0.5):
        sel = has & (tm > -lo - 0.25) & (tm <= -lo + 0.25)
        if not sel.any():
            continue
        rec_e, n, _ = matched_excess(rows["rec"], rows, sel, c3)
        aft_e, _, _ = matched_excess(after, rows, sel, c3)
        prof[str(lo)] = {"n": int(sel.sum()), "recorded": float(rows["rec"][sel].mean()),
                         "before": float(before[sel].mean()), "after": float(after[sel].mean()),
                         "head": float(h[sel].mean()),
                         "excess_recorded": rec_e, "excess_after": aft_e}
    out["event_profile"] = prof
    last3 = has & (tm > 0) & (tm <= 3.0)
    if last3.any():
        out["last_3s"] = {"n": int(last3.sum()), "recorded": float(rows["rec"][last3].mean()),
                          "before": float(before[last3].mean()), "after": float(after[last3].mean()),
                          "head": float(h[last3].mean())}
    hp = h[has]
    if hp.size:
        out["head_quantiles"] = {q: float(np.quantile(hp, float(q))) for q in ("0.001", "0.01", "0.5",
                                                                                "0.99", "0.999")}
        out["head_share_lt_minus3"] = float((hp < -3).mean())
    return out


def fit(args):
    torch.set_num_threads(args.threads)
    windows = load_windows(args.out, args.n_shards)
    eps = sorted({w["episode"] for w in windows})
    held = set(eps[::4])
    fit_w = [w for w in windows if w["episode"] not in held]
    held_w = [w for w in windows if w["episode"] in held]
    from wms.cli.assemble import build_physics
    from wms.config.load import load_train_config
    cfg = load_train_config(ROOT / "configs" / CONFIG, ["runtime.device=cpu"])
    delta = float(cfg.loss.follower_courtesy.huber_delta_mps2)
    phys = build_physics(cfg)
    amin, amax = float(phys.accel_min_mps2), float(phys.accel_max_mps2)
    res = {"n_windows": len(windows), "n_fit_windows": len(fit_w), "n_held_windows": len(held_w),
           "n_episodes": len(eps), "held_episodes": sorted(held), "huber_delta": delta,
           "n_rows": int(sum(len(w["has"]) for w in windows)),
           "n_partner_rows": int(sum(int(w["has"].sum()) for w in windows))}
    print(json.dumps({k: res[k] for k in ("n_windows", "n_fit_windows", "n_held_windows",
                                          "n_partner_rows")}), flush=True)
    shifts_all = per_window_shift(windows)
    arms = {"A_r10b_loss_v1": {"contrast": False, "v2": False},
            "B_r10b_loss_v2": {"contrast": False, "v2": True},
            "C_contrast_v1": {"contrast": True, "v2": False},
            "D_contrast_v2": {"contrast": True, "v2": True}}

    def sched(scale):
        return [(30, 5e-5 * scale, 4), (60, 3e-5 * scale, 5)]

    # lr rule per arm (zero init): lowest held-out loss of the arm's own objective in {1,10,100,1000}
    heads, pilot = {}, {}
    for name, arm in arms.items():
        pilot[name] = {}
        for scale in args.scales:
            head = train_head(fit_w, sched(scale), arm, delta, amin, amax)
            pilot[name][str(scale)] = eval_loss(head, held_w, arm, delta, amin, amax)
            heads[(name, scale)] = head
            print("pilot", name, scale, round(pilot[name][str(scale)], 4), flush=True)
        zero = FollowerCourtesyHead(version=2 if arm["v2"] else 1)
        pilot[name]["zero_head"] = eval_loss(zero, held_w, arm, delta, amin, amax)
    res["lr_pilot_held_loss"] = pilot
    best = {name: min(args.scales, key=lambda s: pilot[name][str(s)]) for name in arms}
    res["lr_scale_by_rule"] = best
    # D from the round-10b-like head (A's production head, zero-expanded): the default round-11 init
    a_prod = heads[("A_r10b_loss_v1", best["A_r10b_loss_v1"])]
    d_from_a = {}
    for scale in args.scales:
        head = train_head(fit_w, sched(scale), arms["D_contrast_v2"], delta, amin, amax,
                          init=expanded_copy(a_prod))
        d_from_a[scale] = head
        pilot.setdefault("D_from_A_init", {})[str(scale)] = eval_loss(
            head, held_w, arms["D_contrast_v2"], delta, amin, amax)
        print("pilot D_from_A", scale, round(pilot["D_from_A_init"][str(scale)], 4), flush=True)
    pilot["D_from_A_init"]["init_only"] = eval_loss(expanded_copy(a_prod), held_w,
                                                    arms["D_contrast_v2"], delta, amin, amax)
    best["D_from_A_init"] = min(args.scales, key=lambda s: pilot["D_from_A_init"][str(s)])
    # long fits (convergence behaviour)
    long_heads = {name: train_head(fit_w, [(args.fit_steps, 1e-3, 4)], arm, delta, amin, amax)
                  for name, arm in arms.items() if name in ("A_r10b_loss_v1", "D_contrast_v2")}
    # weight rule for D (gradient-norm parity with the weighted fix-B term), zero head and A-init head
    rng = np.random.default_rng(7)
    picks = [rng.choice(len(fit_w), size=4, replace=False) for _ in range(8)]

    def gnorm(make):
        norms = []
        for pick in picks:
            head = make()
            loss = head_loss(head, batch_of([fit_w[i] for i in pick]), arms["D_contrast_v2"], delta,
                             amin, amax)
            grads = torch.autograd.grad(loss, list(head.parameters()), allow_unused=True)
            norms.append(float(torch.sqrt(sum(g.square().sum() for g in grads if g is not None))))
        return norms
    n_zero = gnorm(lambda: FollowerCourtesyHead(version=2))
    n_init = gnorm(lambda: expanded_copy(a_prod))
    g_zero, g_init = (float(np.exp(np.mean(np.log(n)))) for n in (n_zero, n_init))
    res["weight_rule"] = {"g_cutin_weighted": G_CUTIN_WEIGHTED,
                          "g_courtesy_zero_head_per_batch": n_zero, "g_courtesy_zero_head": g_zero,
                          "g_courtesy_r10b_init_per_batch": n_init, "g_courtesy_r10b_init": g_init,
                          "weight_zero_head": float(f"{G_CUTIN_WEIGHTED / g_zero:.4g}"),
                          "weight_r10b_init": float(f"{G_CUTIN_WEIGHTED / g_init:.4g}")}
    # reports
    rep = {}
    for subset, ws in (("held", held_w), ("all", windows)):
        rep[subset] = {"head0": report(None, ws, "head = 0", amin, amax, shifts_all)}
        for name in arms:
            rep[subset][name + "_production"] = report(heads[(name, best[name])], ws,
                                                       f"{name} production x{best[name]}",
                                                       amin, amax, shifts_all)
        rep[subset]["D_from_A_init_production"] = report(
            d_from_a[best["D_from_A_init"]], ws, f"D from A-init x{best['D_from_A_init']}",
            amin, amax, shifts_all)
        for name, head in long_heads.items():
            rep[subset][name + "_long"] = report(head, ws, f"{name} long {args.fit_steps} @1e-3",
                                                 amin, amax, shifts_all)
    res["reports"] = rep
    # contrast diagnostics: matched shift on partner rows by group
    shift = np.concatenate([shifts_all[w["window"]] for w in windows])
    has = np.concatenate([w["has"] for w in windows])
    res["contrast_shift_partner_mean"] = float(shift[has].mean())
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    torch.save({name: heads[(name, best[name])].state_dict() for name in arms}
               | {"D_from_A_init": d_from_a[best["D_from_A_init"]].state_dict()},
               Path(args.report).with_suffix(".heads.pt"))
    for subset in ("held", "all"):
        for key, r in rep[subset].items():
            l3 = r.get("last_3s", {})
            far = r.get("far_dA_gt30", {})
            print(subset, key, "last3 rec/after/head", round(l3.get("recorded", np.nan), 3),
                  round(l3.get("after", np.nan), 3), round(l3.get("head", np.nan), 3),
                  "far after-rec", round(far.get("abs_bias_after", np.nan), 3), flush=True)


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
    c.add_argument("--threads", type=int, default=1)
    f = sub.add_parser("fit")
    f.add_argument("--out", required=True)
    f.add_argument("--n-shards", type=int, default=1)
    f.add_argument("--report", required=True)
    f.add_argument("--fit-steps", type=int, default=400)
    f.add_argument("--scales", type=float, nargs="+", default=[1, 10, 100, 1000])
    f.add_argument("--threads", type=int, default=4)
    args = p.parse_args(argv)
    cache(args) if args.cmd == "cache" else fit(args)


if __name__ == "__main__":
    main()
