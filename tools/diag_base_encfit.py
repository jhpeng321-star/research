"""S-BASEDIAG offline test of fix 1 (CPU, TRAIN windows, no GPU, no rollout, no SUMO).

Question: if the IDM core's driver settings were reachable (``driver_param_mode = "physical"``, V6ENC: full
range, learned centre), would the encoder - trained by one-step prediction error only - land on settings that
remove the base over-prediction, without hurting other regimes?

Phase ``capture``: for the decomposition windows (``tools/diag_base_decomp.py`` caches) replay the recorded
window with the model's forward stubbed out (only the replay's own driver encoding runs) and store every
``driver_encoder`` call's inputs (pre-t0 history, mask, interaction, defaults) for the slots it (re)encodes.
Phase ``fit``: rebuild per-row driver parameters from those inputs with a physical-mode encoder initialised
from the round-8 head (centre from the physics defaults), compute the pure IDM core (learned residual OFF) on
the cached lane-1 HDV rows and fit encoder head + centre by Huber loss on the recorded acceleration; optional
textbook driver prior (as the training config's ``driver_prior``).  Episode-held-out evaluation by group.
Diagnostic fit only - nothing is written into a model or a config.
"""
from __future__ import annotations

import argparse
import copy
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

PARAMS = ("v0", "time_headway", "min_gap", "accel", "decel")


# ----------------------------------------------------------------------------------------- capture
def capture(args):
    from tools.r10_attribution import load_model
    from tools.r10b_courtesy_offline import datasets
    from wms.data.collate import collate
    from wms.train.inputs import build_targets, prepare_inputs
    from wms.train.losses.cutin_reaction import one_step_replay
    torch.set_num_threads(args.threads)
    cfg, settings, model, report = load_model(args.config, args.ckpt, tuple(args.set))
    n_steps = int(round(args.horizon_s / 0.5))
    sets = datasets(cfg, args.roots, n_steps, args.stride_s)
    names = {}
    for si, ds in enumerate(sets):
        for ep, entry in enumerate(ds._loader.entries):
            names[entry["episode"]] = si
    wins = []
    for i in range(16):
        p = Path(f"{args.decomp}.shard{i}.pkl")
        if p.is_file():
            wins.extend(w["window"] for w in pickle.load(open(p, "rb"))["windows"])
    wins = wins[args.shard::args.n_shards]
    enc = model.driver_encoder
    out, t0 = [], time.time()
    for wname in wins:
        ep_name, idx = wname.rsplit(":", 1)
        ds = sets[names[ep_name]]
        batch = collate([ds[int(idx)]])
        inp = prepare_inputs(batch, model, settings, torch.device("cpu"), n_steps)
        tg = build_targets(inp, batch, settings, float(model.physics.dt_s), torch.device("cpu"),
                           physics=model.physics)
        tg.prepared["cutin_reaction_inputs"] = {
            "history": inp.hist, "history_mask": inp.hist_mask,
            "driver_history": inp.observed_hist, "driver_history_mask": inp.observed_hist_mask}
        events, step = [], {"t": 0, "prev": None}
        original_enc = enc.forward

        def enc_hook(history, history_mask, interaction, defaults, *a, **kw):
            res = original_enc(history, history_mask, interaction, defaults, *a, **kw)
            slots = torch.arange(history.shape[1]) if step["prev"] is None else step["new"]
            events.append({"t": step["t"], "slots": slots.numpy().astype(np.int32),
                           "history": history[0, slots].detach().clone(),
                           "mask": history_mask[0, slots].detach().clone(),
                           "interaction": interaction[0, slots].detach().clone(),
                           "defaults": defaults[0, slots].detach().clone(),
                           "legacy": res["values"][0, slots, :5].detach().clone()})
            return res

        original_ed = model.encode_drivers

        def ed_hook(state, **kw):
            present = state["present"][0]
            if step["prev"] is not None:
                step["new"] = torch.nonzero(present & ~step["prev"]).flatten()
            return original_ed(state, **kw)

        def fwd_stub(state, k_steps=0, **kw):
            t = int(k_steps)
            step["t"], step["prev"] = t + 1, state["present"][0].clone()
            z = torch.zeros_like(state["veh_kin"][..., 0])
            return {"accel_mps2": z}

        # The replay encodes BEFORE calling forward at step t; track presence so the hook knows the new slots.
        enc.forward = enc_hook
        model.forward = fwd_stub
        model.encode_drivers = ed_hook
        step["new"] = torch.zeros(0, dtype=torch.long)
        try:
            with torch.no_grad():
                one_step_replay(model, tg)
        finally:
            enc.__dict__.pop("forward", None)
            model.__dict__.pop("forward", None)
            model.__dict__.pop("encode_drivers", None)
        out.append({"window": wname, "events": events})
        print(f"shard {args.shard}: {len(out)}/{len(wins)} events {len(events)} {time.time() - t0:.0f}s",
              flush=True)
    with open(f"{args.out}.shard{args.shard}.pkl", "wb") as f:
        pickle.dump(out, f)


# --------------------------------------------------------------------------------------------- fit
def idm_t(v, vl, gap, v0, T, s0, a, b):
    closing = v - vl
    des = s0 + torch.clamp(v * T + v * closing / (2 * torch.sqrt(a * b)), min=0.0)
    acc = a * (1 - (v / v0.clamp_min(1e-3)) ** 4 - (des / gap.clamp_min(0.3)) ** 2)
    acc = acc.clamp(-9.0, 3.0)
    return torch.minimum(torch.maximum(acc, -b), a)


def fit(args):
    from tools.diag_base_decomp_an import load as load_rows
    from tools.r10_attribution import load_model
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    _, _, model, _ = load_model(args.config, args.ckpt, tuple(args.set) + ("model.driver_param_mode=physical",))
    enc_phys = model.driver_encoder
    assert enc_phys.driver_param_mode == "physical"
    caps = {}
    for i in range(16):
        p = Path(f"{args.out}.shard{i}.pkl")
        if p.is_file():
            for w in pickle.load(open(p, "rb")):
                caps[w["window"]] = w["events"]
    # rows from the decomposition cache, in window order
    W = []
    for i in range(16):
        p = Path(f"{args.decomp}.shard{i}.pkl")
        if p.is_file():
            W.extend(pickle.load(open(p, "rb"))["windows"])
    W = [w for w in W if w["window"] in caps]
    episodes = sorted({w["episode"] for w in W})
    rng = np.random.default_rng(0)
    held = set(rng.choice(episodes, size=len(episodes) // 4, replace=False))
    # Build, per window, the event inputs and a row -> (event, position) pointer.
    data = []
    for w in W:
        ev = caps[w["window"]]
        last = {}
        ptr_e = np.full(len(w["rec"]), -1)
        ptr_p = np.full(len(w["rec"]), -1)
        order = np.argsort(w["step"], kind="stable")
        ev_t = [e["t"] for e in ev]
        ei = 0
        for r in order:
            t = int(w["step"][r])
            while ei < len(ev) and ev_t[ei] <= t:
                for pos, sl in enumerate(ev[ei]["slots"]):
                    last[int(sl)] = (ei, pos)
                ei += 1
            got = last.get(int(w["slot"][r]))
            if got is not None:
                ptr_e[r], ptr_p[r] = got
        keep = ptr_e >= 0
        data.append({"w": w, "ev": ev, "pe": ptr_e, "pp": ptr_p, "keep": keep,
                     "held": w["episode"] in held})
    print("windows", len(data), "held-out windows", sum(d["held"] for d in data),
          "rows kept", int(sum(d["keep"].sum() for d in data)), "of", int(sum(len(d["keep"]) for d in data)))

    def params_for(d, encoder):
        outs = []
        for e in d["ev"]:
            r = encoder(e["history"][None], e["mask"][None], e["interaction"][None], e["defaults"][None])
            outs.append(r["values"][0, :, :5])
        flat = torch.cat(outs, 0)
        offs = np.cumsum([0] + [len(e["slots"]) for e in d["ev"]])
        idx = torch.as_tensor(offs[np.maximum(d["pe"], 0)] + np.maximum(d["pp"], 0))
        return flat[idx]

    def rows_t(d):
        w = d["w"]
        return {k: torch.as_tensor(np.asarray(w[k], np.float32)) for k in ("v", "lead_v", "gap", "rec")}

    for d in data:
        d["t"] = rows_t(d)
    prior = {"accel": 1.2, "time_headway": 1.2, "decel": 2.0, "min_gap": 2.0}

    def run_arm(name, encoder, train, prior_w, steps):
        enc = copy.deepcopy(encoder)
        enc.train(False)
        if train:
            params = [p for n, p in enc.named_parameters() if n.startswith("head.") or n == "centre_logits"]
            for p in enc.parameters():
                p.requires_grad_(False)
            for p in params:
                p.requires_grad_(True)
            opt = torch.optim.Adam(params, lr=args.lr)
            fit_d = [d for d in data if not d["held"]]
            for it in range(steps):
                opt.zero_grad()
                batch = [fit_d[j] for j in rng.choice(len(fit_d), size=min(8, len(fit_d)), replace=False)]
                loss = 0.0
                for d in batch:
                    P = params_for(d, enc)
                    tt = d["t"]
                    k = torch.as_tensor(d["keep"])
                    pred = idm_t(tt["v"], tt["lead_v"], tt["gap"], P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4])
                    l = F.huber_loss(pred[k], tt["rec"][k], delta=1.0)
                    if prior_w:
                        names = {"accel": 3, "time_headway": 1, "decel": 4, "min_gap": 2}
                        l = l + prior_w * sum(((P[k, j] - prior[n]) ** 2).mean() for n, j in names.items())
                    loss = loss + l / len(batch)
                loss.backward()
                opt.step()
                if it % 25 == 0 or it == steps - 1:
                    print(f"  {name} step {it} loss {float(loss):.4f}", flush=True)
        res = []
        with torch.no_grad():
            for d in data:
                P = params_for(d, enc)
                tt = d["t"]
                pred = idm_t(tt["v"], tt["lead_v"], tt["gap"], P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4])
                res.append((pred.numpy(), P.numpy()))
        return res

    legacy_model = load_model(args.config, args.ckpt, tuple(args.set))[2]
    arms = {"legacy encoder as is (IDM core only)": run_arm("legacy", legacy_model.driver_encoder, False, 0, 0),
            "physical, init (head from r8, centre = defaults)": run_arm("phys0", enc_phys, False, 0, 0),
            "physical, fit head+centre, no prior": run_arm("physfit", enc_phys, True, 0.0, args.steps),
            "physical, fit head+centre, textbook prior 0.05": run_arm("physprior", enc_phys, True, 0.05, args.steps)}
    # evaluation on held-out windows
    groups = ("partners last 3 s", "partners [-6,-3) s", "unpartnered zone v<2", "slow start v<2 gap 6-30",
              "far upstream s<-400 v<2 gap 6-30", "after merge s>75", "recorded braking rec<-1", "v>15", "all")
    print("\nheld-out episodes: mean (pred - rec) [MAE]; model-as-is column = full model incl. residual")
    print("group | n | rec | model as is | " + " | ".join(arms))
    cat = lambda f: np.concatenate([f(d) for d in data if d["held"]])  # noqa: E731
    rec = cat(lambda d: d["w"]["rec"])
    keep = cat(lambda d: d["keep"])
    final = cat(lambda d: d["w"]["final"])
    v, s, gap = cat(lambda d: d["w"]["v"]), cat(lambda d: d["w"]["s"]), cat(lambda d: d["w"]["gap"])
    has = cat(lambda d: d["w"]["has"]).astype(bool)
    tm, dend = cat(lambda d: d["w"]["tmerge"]), cat(lambda d: d["w"]["dend"])
    zone = has & (dend >= 0) & (dend <= 75.44)
    masks = [zone & (tm > 0) & (tm <= 3), zone & (tm > 3) & (tm <= 6), ~has & (s >= 0) & (s < 75.44) & (v < 2),
             ~has & (v < 2) & (gap > 6) & (gap < 30), (s < -400) & (v < 2) & (gap > 6) & (gap < 30),
             s > 75.44, rec < -1, v > 15, np.ones_like(has)]
    preds = {k: np.concatenate([r[0] for r, d in zip(res, data) if d["held"]]) for k, res in arms.items()}
    pars = {k: np.concatenate([r[1] for r, d in zip(res, data) if d["held"]]) for k, res in arms.items()}
    for g, m in zip(groups, masks):
        m = m & keep
        line = f"{g} | {int(m.sum())} | {rec[m].mean():+.2f} | {final[m].mean() - rec[m].mean():+.2f} [{np.abs(final[m] - rec[m]).mean():.2f}]"
        for k in arms:
            line += f" | {preds[k][m].mean() - rec[m].mean():+.2f} [{np.abs(preds[k][m] - rec[m]).mean():.2f}]"
        print(line)
    print("\nencoded parameters on held-out rows (p10 / p50 / p90):")
    for k in arms:
        P = pars[k][keep]
        print(k, {n: np.percentile(P[:, j], [10, 50, 90]).round(2).tolist() for j, n in enumerate(("v0", "T", "s0", "a", "b"))})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("capture", "fit"))
    p.add_argument("--config", default="c4_wm_v10round10b.toml")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--roots", nargs="+", default=[])
    p.add_argument("--decomp", default="tmp_basediag/r8")
    p.add_argument("--out", default="tmp_basediag/enc")
    p.add_argument("--horizon-s", type=float, default=120.0)
    p.add_argument("--stride-s", type=float, default=30.0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-3)
    args = p.parse_args(argv)
    capture(args) if args.phase == "capture" else fit(args)


if __name__ == "__main__":
    main()
