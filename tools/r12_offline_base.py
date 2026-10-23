"""R12 offline TRAIN check of the base calibration + the driver-encoder lr rule (CPU, no GPU, no rollout, no SUMO).

Inputs (read-only): S-BASEDIAG's TRAIN captures of the round-8 base (= round 9's car-following base; round 9 trained only
merge_pivot) - the decomposition cache ``<decomp>.shard*.pkl`` (tools/diag_base_decomp.py: per lane-1 HDV row the IDM
core, gate x attention, gate x temporal, final, recorded, state and partner columns) and the encoder-input capture
``<enc>.shard*.pkl`` (tools/diag_base_encfit.py capture: every driver-encoder call's inputs).  96 windows x 120 s, one
per TRAIN episode (v5_native + genrand_plain), held out by episode (1/4).

Arms (one-step teacher-forced, lane-1 HDV rows; mean (pred - rec) and MAE by row class):
  r9            round 9 as is (full model: legacy IDM core + residual)
  r9_core       the legacy IDM core alone
  r12_init      the round-12 init: physical encoder with the head's longitudinal rows zeroed (every driver at the
                learned centre = physics defaults), residual attention read-out zeroed (only gate x temporal kept)
  r12_fit       r12_init after a SHORT CPU fit of the calibrated base: driver-encoder head + centre trained by Huber(IDM
                core + gate x temporal, recorded) + the production driver prior (batch mean, relative, weight 0.05) with
                the PRODUCTION schedule (40 steps x 4 windows @ 5e-5 x scale, 70 x 2 @ 3e-5 x scale; Adam; global clip 1)
  r12_fit_noprior / r12_fit_weak   the same with the prior off / at weight 0.01 (identifiability read)
lr rule: (head scale, centre scale) in {1, 10} x {10, 100, 300, 1000} with the lowest held-out objective (Huber + prior)
after the production schedule (the round-11 rule form).  The production objective is the rollout motion loss, not this
one-step proxy (stated limit); the rule fixes how far Adam can move the centre within the schedule.
Usage: python tools/r12_offline_base.py --workers 6 --out docs/r12_offline_base.json
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

BASEDIAG = Path("E:/agent-work/c4-v10r11/tmp_basediag")
CONFIG = "configs/c4_wm_v10round12.toml"
ZONE_END = 75.44
PRIOR_TARGET = {"accel": 1.2, "time_headway": 1.2, "decel": 2.0, "min_gap": 2.0, "v0": 1.0}
PRIOR_INDEX = {"v0": 0, "time_headway": 1, "min_gap": 2, "accel": 3, "decel": 4}
SCHEDULE = ((40, 4, 5.0e-5), (70, 2, 3.0e-5))
GRID = tuple((h, c) for h in (1.0, 10.0) for c in (10.0, 100.0, 300.0, 1000.0))


def idm(v, vl, gap, v0, T, s0, a, b):
    """IDM core as in tools/diag_base_encfit.idm_t (plant clamp -9..3, then [-b, a])."""
    des = s0 + torch.clamp(v * T + v * (v - vl) / (2 * torch.sqrt(a * b)), min=0.0)
    acc = a * (1 - (v / v0.clamp_min(1e-3)) ** 4 - (des / gap.clamp_min(0.3)) ** 2)
    return torch.minimum(torch.maximum(acc.clamp(-9.0, 3.0), -b), a)


def load_data(decomp: Path, enc: Path):
    caps, wins = {}, []
    for i in range(16):
        p = Path(f"{enc}.shard{i}.pkl")
        if p.is_file():
            for w in pickle.load(open(p, "rb")):
                caps[w["window"]] = w["events"]
        q = Path(f"{decomp}.shard{i}.pkl")
        if q.is_file():
            wins.extend(pickle.load(open(q, "rb"))["windows"])
    wins = [w for w in wins if w["window"] in caps]
    episodes = sorted({w["episode"] for w in wins})
    held = set(np.random.default_rng(0).choice(episodes, size=len(episodes) // 4, replace=False))
    data = []
    for w in wins:
        ev = caps[w["window"]]
        last, ei = {}, 0
        pe = np.full(len(w["rec"]), -1)
        pp = np.full(len(w["rec"]), -1)
        ev_t = [e["t"] for e in ev]
        for r in np.argsort(w["step"], kind="stable"):
            t = int(w["step"][r])
            while ei < len(ev) and ev_t[ei] <= t:
                for pos, sl in enumerate(ev[ei]["slots"]):
                    last[int(sl)] = (ei, pos)
                ei += 1
            got = last.get(int(w["slot"][r]))
            if got is not None:
                pe[r], pp[r] = got
        t = {k: torch.as_tensor(np.asarray(w[k], np.float32)) for k in ("v", "lead_v", "gap", "rec", "temp")}
        # t0 prior slots: first encoder call, slots with observed history (production prior_mask)
        prior_slots = ev[0]["mask"].any(-1) if ev else torch.zeros(0, dtype=torch.bool)
        data.append({"w": w, "ev": ev, "pe": pe, "pp": pp, "keep": torch.as_tensor(pe >= 0), "t": t,
                     "held": w["episode"] in held, "prior_slots": prior_slots})
    return data


def params_for(d, encoder):
    outs = [encoder(e["history"][None], e["mask"][None], e["interaction"][None], e["defaults"][None])["values"][0, :, :5]
            for e in d["ev"]]
    offs = np.cumsum([0] + [len(e["slots"]) for e in d["ev"]])
    idx = torch.as_tensor(offs[np.maximum(d["pe"], 0)] + np.maximum(d["pp"], 0))
    return torch.cat(outs, 0)[idx], outs[0]


def objective(d, encoder, prior_w, speed_limit):
    P, first = params_for(d, encoder)
    tt, k = d["t"], d["keep"]
    pred = idm(tt["v"], tt["lead_v"], tt["gap"], *(P[:, j] for j in range(5))) + tt["temp"]
    huber = F.huber_loss(pred[k], tt["rec"][k], delta=1.0)
    prior = huber.new_zeros(())
    m = d["prior_slots"]
    if prior_w and bool(m.any()):
        for name, j in PRIOR_INDEX.items():
            col = first[m, j] / (speed_limit if name == "v0" else 1.0)
            prior = prior + ((col.mean() - PRIOR_TARGET[name]) / PRIOR_TARGET[name]) ** 2
    return huber, prior_w * prior, pred, P


def build_init_encoder():
    """Round-12 init encoder: physical mode, round-9 head, longitudinal mean rows zeroed (driver_param_init=centre)."""
    from wms.cli.assemble import build_model, build_physics
    from wms.config.load import load_train_config
    cfg = load_train_config(ROOT / CONFIG)
    physics = build_physics(cfg)
    model = build_model(cfg, physics)
    ck = torch.load(ROOT / "tmp_r12/ck/r9_stage300.pt", map_location="cpu", weights_only=True)["model"]
    own = model.state_dict()
    model.load_state_dict({k: v for k, v in ck.items() if k in own and own[k].shape == v.shape}, strict=False)
    enc = model.driver_encoder
    with torch.no_grad():
        enc.head[-1].weight[:5].zero_()
        enc.head[-1].bias[:5].zero_()
    enc.train(False)
    return enc, float(physics.speed_limit_mps)


def fit_arm(args_tuple):
    head_scale, centre_scale, prior_w, decomp, encp, threads, seed, schedule = args_tuple
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    data = load_data(Path(decomp), Path(encp))
    enc0, vlim = build_init_encoder()
    enc = copy.deepcopy(enc0)
    for p in enc.parameters():
        p.requires_grad_(False)
    head = [p for n, p in enc.named_parameters() if n.startswith("head.")]
    centre = [enc.centre_logits]
    for p in head + centre:
        p.requires_grad_(True)
    opt = torch.optim.Adam([{"params": head, "lr": 0.0, "scale": head_scale},
                            {"params": centre, "lr": 0.0, "scale": centre_scale}])
    fit_d = [d for d in data if not d["held"]]
    rng = np.random.default_rng(seed)
    trace = []
    for steps, batch, lr in schedule:
        for g in opt.param_groups:
            g["lr"] = lr * g["scale"]
        for _ in range(steps):
            opt.zero_grad()
            loss = 0.0
            for j in rng.choice(len(fit_d), size=batch, replace=False):
                h, pr, _, _ = objective(fit_d[j], enc, prior_w, vlim)
                loss = loss + (h + pr) / batch
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head + centre, 1.0)
            opt.step()
            lo, hi = enc.physical_lower, enc.physical_upper
            trace.append({"loss": float(loss), "centre": (lo + (hi - lo) * torch.sigmoid(enc.centre_logits.detach())).tolist()})
    held = [d for d in data if d["held"]]
    with torch.no_grad():
        hv = [objective(d, enc, prior_w, vlim) for d in held]
        rows = [d["keep"].sum().item() for d in held]
        held_huber = float(sum(float(h) * n for (h, _, _, _), n in zip(hv, rows)) / max(sum(rows), 1))
        held_prior = float(np.mean([float(p) for _, p, _, _ in hv]))
        preds = [x[2].numpy() for x in hv]
        pars = [x[3].numpy() for x in hv]
    return {"head_scale": head_scale, "centre_scale": centre_scale, "prior_w": prior_w,
            "held_objective": held_huber + held_prior, "held_huber": held_huber, "held_prior": held_prior,
            "trace": trace[::10] + trace[-1:], "preds": preds, "pars": pars}


def init_arm(decomp, encp):
    data = load_data(Path(decomp), Path(encp))
    enc, vlim = build_init_encoder()
    held = [d for d in data if d["held"]]
    with torch.no_grad():
        out = [objective(d, enc, 0.05, vlim) for d in held]
    return [x[2].numpy() for x in out], [x[3].numpy() for x in out]


def classes(held):
    cat = lambda key: np.concatenate([np.asarray(d["w"][key]) for d in held])  # noqa: E731
    v, s, gap, rec = cat("v"), cat("s"), cat("gap"), cat("rec")
    has, tm, dend, da, gap_r = cat("has").astype(bool), cat("tmerge"), cat("dend"), cat("da"), cat("gap_r")
    zone = has & (dend >= 0) & (dend <= ZONE_END)
    keep = np.concatenate([d["keep"].numpy() for d in held])
    out = {
        "slow start (unpartnered, v<2, gap 6-30)": ~has & (v < 2) & (gap > 6) & (gap < 30),
        "far-upstream slow start (s<-400)": (s < -400) & (v < 2) & (gap > 6) & (gap < 30),
        "upstream slow start (unpartnered, s<-100)": ~has & (s < -100) & (v < 2) & (gap > 6) & (gap < 30),
        "approach/zone slow start (unpartnered, -100<=s<=75.44)": ~has & (s >= -100) & (s <= ZONE_END) & (v < 2)
        & (gap > 6) & (gap < 30),
        "free road (5<v<20, gap>60)": (v > 5) & (v < 20) & (gap > 60),
        "target follower last 3 s": zone & (tm > 0) & (tm <= 3),
        "target follower [-6,-3) s": zone & (tm > 3) & (tm <= 6),
        "far partner dA>30 (neg. control)": has & (da > 30),
        "partner d_end>40 (neg. control)": has & (dend > 40),
        "partner alongside gap_r<1 (neg. control)": has & (gap_r < 1),
        "recorded braking rec<-1": rec < -1,
        "after merge zone s>75.44": s > ZONE_END,
        "all lane-1 rows": np.ones_like(has),
    }
    return {k: m & keep for k, m in out.items()}, rec


def summarise(arms: dict, held) -> dict:
    masks, rec = classes(held)
    table = {}
    for name, m in masks.items():
        row = {"n": int(m.sum()), "rec": float(rec[m].mean()) if m.any() else None}
        for arm, pred in arms.items():
            row[arm] = ([round(float((pred[m] - rec[m]).mean()), 3), round(float(np.abs(pred[m] - rec[m]).mean()), 3)]
                        if m.any() else None)
        table[name] = row
    return table


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--decomp", default=str(BASEDIAG / "r8"))
    p.add_argument("--enc", default=str(BASEDIAG / "enc"))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default="docs/r12_offline_base.json")
    p.add_argument("--quick", action="store_true", help="2 grid points, 2+2 steps (smoke of the tool only)")
    a = p.parse_args(argv)
    schedule = ((2, 4, 5.0e-5), (2, 2, 3.0e-5)) if a.quick else SCHEDULE
    t0 = time.time()
    grid = GRID[:2] if a.quick else GRID
    jobs = [(h, c, 0.05, a.decomp, a.enc, a.threads, 0, schedule) for h, c in grid]
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(fit_arm, jobs))
    best = min(results, key=lambda r: r["held_objective"])
    extra = [(best["head_scale"], best["centre_scale"], w, a.decomp, a.enc, a.threads, 0, schedule)
             for w in (0.0, 0.01)]
    with ProcessPoolExecutor(min(a.workers, 2)) as pool:
        noprior, weak = list(pool.map(fit_arm, extra))
    data = load_data(Path(a.decomp), Path(a.enc))
    held = [d for d in data if d["held"]]
    init_pred, init_par = init_arm(a.decomp, a.enc)
    cat = lambda xs: np.concatenate(xs)  # noqa: E731
    arms = {"r9": cat([np.asarray(d["w"]["final"]) for d in held]),
            "r9_core": cat([np.asarray(d["w"]["phys"]) for d in held]),
            "r12_init": cat(init_pred), "r12_fit": cat(best["preds"]),
            "r12_fit_noprior": cat(noprior["preds"]), "r12_fit_weak": cat(weak["preds"])}
    # IDM core alone (the round-8 gate x temporal emission removed): the encoder's own one-step read; the production
    # run retrains the temporal MLP, so the core-only columns are the cleaner read of the settings.
    temp = cat([np.asarray(d["w"]["temp"]) for d in held])
    for name in ("r12_init", "r12_fit", "r12_fit_noprior", "r12_fit_weak"):
        arms[name + "_core"] = arms[name] - temp
    pars = {"r9": cat([np.asarray(d["w"]["params"]) for d in held]), "r12_init": cat(init_par),
            "r12_fit": cat(best["pars"]), "r12_fit_noprior": cat(noprior["pars"]), "r12_fit_weak": cat(weak["pars"])}
    keep = np.concatenate([d["keep"].numpy() for d in held])
    enc0, _ = build_init_encoder()
    lo, hi = enc0.physical_lower.numpy(), enc0.physical_upper.numpy()
    enc_pct = {}
    for arm, P in pars.items():
        P = P[keep]
        enc_pct[arm] = {n: np.percentile(P[:, j], [1, 10, 50, 90, 99]).round(3).tolist()
                        for j, n in enumerate(("v0", "T", "s0", "a", "b"))}
        if arm.startswith("r12"):
            span = hi - lo
            enc_pct[arm]["share_within_1pct_of_bound_a_b"] = [
                float(((P[:, j] - lo[j]) < 0.01 * span[j]).mean() + ((hi[j] - P[:, j]) < 0.01 * span[j]).mean())
                for j in (3, 4)]
    out = {"tool": "tools/r12_offline_base.py", "seconds": round(time.time() - t0, 1),
           "held_windows": len(held), "fit_windows": len(data) - len(held),
           "schedule": schedule, "lr_rule": {"grid": [{k: r[k] for k in ("head_scale", "centre_scale", "held_objective", "held_huber",
                                                   "held_prior")} for r in results],
                       "chosen": {"head_scale": best["head_scale"], "centre_scale": best["centre_scale"]},
                       "centre_trace_chosen": best["trace"],
                       "centre_final": {"chosen": best["trace"][-1]["centre"], "noprior": noprior["trace"][-1]["centre"],
                                        "weak_0.01": weak["trace"][-1]["centre"]}},
           "classes": summarise(arms, held), "encoded_percentiles_p1_p10_p50_p90_p99": enc_pct,
           "sign": "bias = model - recorded (m/s^2); [mean, MAE]"}
    np.savez_compressed(Path(a.out).with_suffix(".preds.npz"), keep=keep, rec=cat([np.asarray(d["w"]["rec"]) for d in held]),
                        **{f"pred_{k}": v.astype(np.float32) for k, v in arms.items()},
                        **{f"col_{k}": cat([np.asarray(d["w"][k]) for d in held]) for k in
                           ("v", "s", "gap", "has", "tmerge", "dend", "da", "gap_r", "inv_between")})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"chosen": out["lr_rule"]["chosen"], "seconds": out["seconds"]}))


if __name__ == "__main__":
    main()
