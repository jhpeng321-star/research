"""R10 offline head pilot (CPU, no rollout, no SUMO): what can the responsible heads reach?

Phase ``cache``: replay recorded TRAIN windows teacher-forced (the cut-in loss replay) and store,
for every supervised row (post-insertion event rows and ordinary lane-1 control rows), the inputs of
the two learned acceleration emitters -- the residual accel head (``residual_net.accel``) and the
temporal MLP (``temporal``) -- together with the gate, the IDM core and the recorded acceleration.

Phase ``fit``: with the trunk, history projection, gate, IDM core and every other module frozen,
train *copies* of the selected emitters with Adam on exactly the cut-in loss (event mean + control
mean, Huber 0.5), 30 steps of 4 windows then 60 steps of 2 windows (round-9/10 stage plan), at the
stage learning rates times a learning-rate scale, and report the held-out (episode-disjoint)
event/control loss and the post-insertion minimum-acceleration deficit.  The pilot never touches
the checkpoint and trains nothing that the real run does not also see; it answers

  * which emitter(s) have the capacity to close the over-reaction (head / temporal / both), and
  * which lr scale moves them far enough in the 90-step budget.

    python tools/r10_head_pilot.py cache --ckpt <ckpt> --roots <r1> <r2> --out tmp_r10/pilot \
        --max-windows 100 --shard i --n-shards N
    python tools/r10_head_pilot.py fit --out tmp_r10/pilot --n-shards N
"""
from __future__ import annotations

import argparse
import copy
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

from tools.r10_attribution import bounded, datasets, load_model  # noqa: E402
from wms.model.gate import saturating_gate  # noqa: E402
from wms.data.collate import collate  # noqa: E402
from wms.train.inputs import build_targets, prepare_inputs  # noqa: E402
from wms.train.losses.cutin_reaction import (CutinReactionConfig, insertion_plan,  # noqa: E402
                                             one_step_predicted_accel)


def replay_cache(model, targets, plan):
    """Replay once; return per-row arrays for supervised rows."""
    store, rows = {}, []
    head, temporal = model.residual_net.accel, model.temporal
    handles = [
        head.register_forward_pre_hook(lambda m, a: store.__setitem__("xa", a[0].detach())),
        temporal.register_forward_pre_hook(lambda m, a: store.__setitem__("xt", a[0].detach()))]
    original = model.forward

    def hooked(state, **kwargs):
        out = original(state, **kwargs)
        rows.append({"xa": store["xa"], "xt": store["xt"], "gate": out["gate"].detach(),
                     "phys": out["physics_accel_mps2"].detach(), "final": out["accel_mps2"].detach()})
        return out

    model.forward = hooked
    try:
        with torch.no_grad():
            one_step_predicted_accel(model, targets)
    finally:
        model.forward = original
        model.__dict__.pop("forward", None)
        for handle in handles:
            handle.remove()
    needed = plan["supervised"] | plan["control"]          # [B, n, K]
    out = {k: [] for k in ("xa", "xt", "gate", "phys", "final", "rec", "mass", "event", "ctrl", "step", "slot",
                                "rec_prev", "rec_next")}
    rec = targets.states[:, 1:targets.n_steps + 1, :, 5]
    prev = targets.states[:, 0:targets.n_steps, :, 5]
    nxt = torch.cat((targets.states[:, 2:targets.n_steps + 1, :, 5],
                     torch.full_like(rec[:, :1], float("nan"))), 1)
    for t, row in enumerate(rows):
        sel = needed[0, t].nonzero().flatten()
        if not len(sel):
            continue
        out["xa"].append(row["xa"][0, sel])
        out["xt"].append(row["xt"][0, sel])
        out["gate"].append(row["gate"][0, sel])
        out["phys"].append(row["phys"][0, sel])
        out["final"].append(row["final"][0, sel])
        out["rec"].append(rec[0, t, sel])
        out["mass"].append(plan["event_weights"][0, t, sel] * max(plan["n_events"], 1))
        out["event"].append(plan["supervised"][0, t, sel])
        out["ctrl"].append(plan["control"][0, t, sel])
        out["step"].append(torch.full((len(sel),), t))
        out["slot"].append(sel)
        out["rec_prev"].append(prev[0, t, sel])
        out["rec_next"].append(nxt[0, t, sel])
    if not out["xa"]:
        return None
    packed = {k: torch.cat(v).numpy() for k, v in out.items()}
    packed["n_events"], packed["n_ctrl"] = plan["n_events"], plan["n_control_rows"]
    # per-event post-insertion minimum accelerations (recorded vs model), for the deficit metric
    packed["events"] = []
    for ev in plan["events"]:
        lo, hi, f = ev["start"], ev["end"], ev["follower"]
        post = list(range(max(lo, ev["step"]), hi + 1))
        if post:
            packed["events"].append({
                "rec": rec[0, post, f].numpy(),
                "final": torch.stack([rows[t]["final"][0, f] for t in post]).numpy(),
                "idx": [int(i) for i in np.nonzero((packed["step"] >= post[0]) & (packed["step"] <= post[-1])
                                                  & (packed["slot"] == f) & packed["event"])[0]],
                "follower": f, "steps": post})
    return packed


def cache(args):
    cfg, settings, model, _ = load_model(args.config, args.ckpt)
    cut = CutinReactionConfig(weight=1.0)
    n_steps = int(round(args.horizon_s / 0.5))
    sets = datasets(cfg, args.roots, n_steps, args.stride_s)
    path = Path(f"{args.out}.shard{args.shard}.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    windows, t0 = [], time.time()
    for root_i, ds in enumerate(sets):
        idx = list(range(len(ds)))
        if args.max_windows:
            rng = np.random.default_rng(0)
            idx = sorted(rng.choice(idx, size=min(args.max_windows, len(idx)), replace=False))
        for i in idx[args.shard::args.n_shards]:
            batch = collate([ds[i]])
            inp = prepare_inputs(batch, model, settings, torch.device("cpu"), n_steps)
            tg = build_targets(inp, batch, settings, float(model.physics.dt_s), torch.device("cpu"),
                               physics=model.physics)
            tg.prepared["cutin_reaction_inputs"] = {
                "history": inp.hist, "history_mask": inp.hist_mask,
                "driver_history": inp.observed_hist, "driver_history_mask": inp.observed_hist_mask}
            plan = insertion_plan(tg, cut)
            if plan["n_event_rows"]:
                packed = replay_cache(model, tg, plan)
                if packed is not None:
                    ep, _tick = ds.positions[i]
                    packed["episode"] = (root_i, int(ep))
                    windows.append(packed)
            print(f"window {len(windows)} {time.time() - t0:.0f}s", flush=True)
            with open(path, "wb") as stream:
                pickle.dump({"windows": windows, "done": False}, stream)
    with open(path, "wb") as stream:
        pickle.dump({"windows": windows, "done": True}, stream)
    print("shard done", args.shard, len(windows), flush=True)


# ---------------------------------------------------------------------------------------- fit
class Emitters:
    """Copies of the learned acceleration emitters plus the gate scalars (all optional-trainable)."""

    def __init__(self, model, trainable: tuple[str, ...]):
        self.model = model
        self.head = copy.deepcopy(model.residual_net.accel)
        self.temporal = copy.deepcopy(model.temporal)
        self.gate = [model.gate_g_max_logit.detach().clone(), model.gate_tau_raw.detach().clone()]
        self.params = []
        for name, module in (("head", self.head), ("temporal", self.temporal)):
            for p in module.parameters():
                p.requires_grad_(name in trainable)
                if name in trainable:
                    self.params.append(p)
        for p in self.gate:
            p.requires_grad_("gate" in trainable)
            if "gate" in trainable:
                self.params.append(p)

    def accel(self, w):
        """Acceleration of cached rows (identical algebra to ``WmSModel.forward``)."""
        model, net = self.model, self.model.residual_net
        act = self.head(w["xa"])[..., 0]
        if net.residual_head_log_scale is not None:
            act = act / torch.exp(net.residual_head_log_scale)
        delta = net.cfg.accel_delta_scale_mps2 * torch.tanh(act)
        temp = self.temporal(w["xt"])[..., 0]
        gate = saturating_gate(self.gate[0], self.gate[1], w["step"].float(),
                               model.physics.max_gap_steps)
        corr = bounded(model, gate * (delta + temp))
        return (w["phys"] + corr).clamp(model.physics.accel_min_mps2, model.physics.accel_max_mps2)


def tensors(window):
    return {k: torch.as_tensor(window[k]) for k in
            ("xa", "xt", "gate", "phys", "final", "rec", "mass", "event", "ctrl", "step",
             "rec_prev", "rec_next")}


def batch_loss(em, windows, huber):
    ev_loss, ct_loss, n_ev, n_ct = 0.0, 0.0, 0, 0
    for w in windows:
        t = w["t"]
        err = F.huber_loss(em.accel(t), t["rec"], delta=huber, reduction="none")
        ev_loss = ev_loss + (err * t["mass"] * t["event"]).sum()
        ct_loss = ct_loss + (err * t["ctrl"]).sum()
        n_ev += w["n_events"]
        n_ct += w["n_ctrl"]
    return ev_loss / max(n_ev, 1), ct_loss / max(n_ct, 1)


def evaluate(em, windows, huber):
    with torch.no_grad():
        ev, ct = batch_loss(em, windows, huber)
        deficits, model_min, rec_min = [], [], []
        ctrl_rows, ctrl_rec = [], []
        for w in windows:
            t = w["t"]
            accel = em.accel(t)
            ctrl_rows.append(accel[t["ctrl"]])
            ctrl_rec.append(t["rec"][t["ctrl"]])
            for e in w["events"]:
                sel = torch.as_tensor(e["idx"], dtype=torch.long)
                if len(sel):
                    deficits.append(float(accel[sel].min() - t["rec"][sel].min()))
                    model_min.append(float(accel[sel].min()))
                    rec_min.append(float(t["rec"][sel].min()))
        rows, rec = torch.cat(ctrl_rows), torch.cat(ctrl_rec)
    out = {"event_loss": float(ev), "control_loss": float(ct), "n_events": len(deficits),
           "control_mean_err": float((rows - rec).mean()),
           "control_p10_model": float(np.percentile(rows.numpy(), 10)),
           "control_p10_rec": float(np.percentile(rec.numpy(), 10))}
    if deficits:
        out.update(min_deficit_mean=float(np.mean(deficits)),
                   min_deficit_median=float(np.median(deficits)),
                   model_min_p50=float(np.percentile(model_min, 50)),
                   model_min_p10=float(np.percentile(model_min, 10)),
                   rec_min_p50=float(np.percentile(rec_min, 50)),
                   rec_min_p10=float(np.percentile(rec_min, 10)))
    return out


def fit_variant(model, train, held, variant, scale, args):
    em = Emitters(model, tuple(variant.split("+")))
    optimizer = torch.optim.Adam(em.params, lr=args.lr[0] * scale)
    gen = np.random.default_rng(args.seed)
    for n, bs, lr in zip(args.stage_steps, args.stage_batch, args.lr):
        for group in optimizer.param_groups:
            group["lr"] = lr * scale
        for _ in range(n):
            pick = gen.choice(len(train), size=min(bs, len(train)), replace=False)
            ev, ct = batch_loss(em, [train[i] for i in pick], args.huber)
            loss = ev + args.control_weight * ct
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(em.params, 1.0)
            optimizer.step()
    out = evaluate(em, held, args.huber)
    out["train"] = evaluate(em, train, args.huber)
    return out


def timing_and_baselines(windows):
    """Where does the recorded acceleration sit relative to the model rows (timing check)?"""
    cat = {k: np.concatenate([w[k][w["ctrl"]] for w in windows])
           for k in ("rec", "rec_prev", "rec_next", "phys", "final")}
    ok = ~np.isnan(cat["rec_next"])
    corr = {}
    for name in ("phys", "final"):
        for ref in ("rec_prev", "rec", "rec_next"):
            m = ok & np.isfinite(cat[ref])
            corr[f"corr({name}, {ref})"] = float(np.corrcoef(cat[name][m], cat[ref][m])[0, 1])
    huber = lambda e: float(F.huber_loss(torch.as_tensor(e), torch.zeros(len(e)),  # noqa: E731
                                         delta=0.5, reduction="mean"))
    return {"control_rows": int(len(cat["rec"])), "correlations": corr,
            "huber_final_vs_rec": huber(cat["final"] - cat["rec"]),
            "huber_phys_vs_rec": huber(cat["phys"] - cat["rec"]),
            "huber_persistence_vs_rec": huber(cat["rec_prev"] - cat["rec"])}


def fit(args):
    cfg, settings, model, _ = load_model(args.config, args.ckpt)
    windows = []
    for i in range(args.n_shards):
        with open(f"{args.out}.shard{i}.pkl", "rb") as stream:
            windows += pickle.load(stream)["windows"]
    for w in windows:
        w["t"] = tensors(w)
    # episode-disjoint split: alternate episodes (sorted) between fit and held-out
    eps = sorted({w["episode"] for w in windows})
    held_eps = {e for e in eps if eps.index(e) % 2 == 1}
    train = [w for w in windows if w["episode"] not in held_eps]
    held = [w for w in windows if w["episode"] in held_eps]
    print(f"windows {len(windows)} train {len(train)} held {len(held)}", flush=True)
    base = Emitters(model, ())
    err = max(float((base.accel(w["t"]) - w["t"]["final"]).abs().max()) for w in windows[:30])
    print("compose vs cached final max abs err", err, flush=True)
    result = {"n_windows": len(windows), "n_train": len(train), "n_held": len(held),
              "compose_check_max_abs_err": err, "timing": timing_and_baselines(windows),
              "baseline_held": evaluate(base, held, args.huber),
              "baseline_train": evaluate(base, train, args.huber), "fits": {}}
    print(json.dumps(result["timing"]), flush=True)
    for variant in args.variants:
        for scale in args.scales:
            res = fit_variant(model, train, held, variant, scale, args)
            result["fits"][f"{variant}@{scale}"] = res
            print(variant, scale, json.dumps({k: v for k, v in res.items() if k != "train"}),
                  flush=True)
    Path(f"{args.out}.fit.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("cache", "fit"))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--roots", nargs="+", default=[])
    p.add_argument("--config", default="c4_wm_v10round9.toml")
    p.add_argument("--out", required=True)
    p.add_argument("--stride-s", type=float, default=10.0)
    p.add_argument("--horizon-s", type=float, default=60.0)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 5.0, 20.0, 100.0])
    p.add_argument("--variants", nargs="+",
                   default=["head", "temporal", "head+temporal", "head+temporal+gate"])
    p.add_argument("--lr", type=float, nargs=2, default=[5e-5, 3e-5])
    p.add_argument("--stage-steps", type=int, nargs=2, default=[30, 60])
    p.add_argument("--stage-batch", type=int, nargs=2, default=[4, 2])
    p.add_argument("--huber", type=float, default=0.5)
    p.add_argument("--control-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    (cache if args.phase == "cache" else fit)(args)


if __name__ == "__main__":
    main()
