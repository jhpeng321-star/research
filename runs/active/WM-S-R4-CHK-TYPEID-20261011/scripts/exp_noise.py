"""Noise-model selection for the identifier (label-free) + accuracy by length.

Variants (no truth input anywhere):
  sigma_gauss   accel residual std = KAPPA * sigma_vType (naive first guess)
  gauss_shared  shared std s (label-free pilot: residual std around pilot argmax)
  students      Student-t df=3, shared scale s
  students_lc   students + drop transitions where the same-lane leader changed or
                own lane changed within the previous tick (observable events)

Selection rule: label-free marginal data likelihood under each noise model
(sum over vehicles of logsumexp_type(ll + log prior)); accuracy vs truth is
reported afterwards as evaluation-only diagnostics.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import gammaln

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from typeid import (DT, IDM5, PRIOR, SIGMA, SPEED_FACTOR, TYPE_NAMES,  # noqa: E402
                    idm_accel, leader_arrays)
from wms.eval.normalize import initial_frame, iter_windows  # noqa: E402

TRUTH = json.loads((HERE.parent / "out/truth_full.json").read_text(encoding="utf-8"))
SETS = {
    "wall_same_h60": (ROOT / "data_cache/wallc1_states_h60/wall_same.json", (5,)),
    "wall_same_real": (ROOT / "data_cache/wallc1_states_h60_real/wall_same.json",
                       (5, 10, 20, 30, 60)),
}
LPRIOR = np.array([np.log(PRIOR[t]) for t in TYPE_NAMES])


def prep(rec):
    h = rec["history"]
    kin = np.asarray(h["kin"], dtype=np.float64)
    mask = np.asarray(h["mask"], dtype=bool)
    ids = [str(x) for x in h["ids"]]
    t, n = kin.shape[:2]
    lane = kin[:, :, 1:4].argmax(-1)
    s_, v = kin[:, :, 0], kin[:, :, 4]
    # leader gap/speed/idx per tick
    gap = np.full((t, n), 1.0e3)
    lead_v = v.copy()
    lead_idx = np.full((t, n), -1, dtype=np.int64)
    for ti in range(t):
        ok = mask[ti]
        if not ok.any():
            continue
        ds = s_[ti][None, :] - s_[ti][:, None]
        same = ok[None, :] & (lane[ti][None, :] == lane[ti][:, None])
        bumper = np.where(same & (ds > 0), ds - 5.0, np.inf)
        g = bumper.min(-1)
        idx = bumper.argmin(-1)
        has = np.isfinite(g)
        gap[ti, has] = g[has]
        lead_v[ti, has] = v[ti][idx[has]]
        lead_idx[ti, has] = idx[has]
    ramp = (lane == 2) & (s_ < 0)
    vlim = np.where(ramp, 22.22, 31.29)
    a_obs = (v[1:] - v[:-1]) / DT
    lane_stable = lane[1:] == lane[:-1]
    tok = mask[1:] & mask[:-1] & lane_stable
    leader_changed = np.zeros_like(tok)
    leader_changed[1:] = (lead_idx[1:-1] != lead_idx[:-2]) & (lead_idx[1:-1] >= 0)
    # per-type predicted accel on transitions (input state at tick t)
    ap = np.stack([idm_accel(v[:-1], lead_v[:-1], gap[:-1],
                             np.broadcast_to(vlim[:-1] * SPEED_FACTOR[name], v[:-1].shape),
                             *IDM5[name]) for name in TYPE_NAMES], -1)   # [T-1, N, 3]
    free = tok & (gap[:-1] >= 100.0)
    sf3 = np.array([SPEED_FACTOR[n] for n in TYPE_NAMES])[None, None, :]
    over = np.maximum(v[:-1][..., None] - vlim[:-1][..., None] * sf3, 0.0)
    return dict(ids=ids, tok=tok, a_obs=a_obs, ap=ap, free=free, over=over,
                leader_changed=leader_changed, v=v, lane=lane, mask=mask, gap=gap,
                v_ff=v[:-1], vlim_ff=vlim[:-1])


def ll_gauss(res, std):
    return -0.5 * (res / std) ** 2 - np.log(std)


def ll_students(res, scale, nu=3.0):
    half = (nu + 1.0) / 2.0
    c = gammaln(half) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi * scale * scale)
    return c - half * np.log1p(res * res / (nu * scale * scale))


def evaluate(set_name, set_file, lengths, variant, scale=None, em_rounds=0):
    acc = {L: [0, 0] for L in lengths}
    conf = {L: Counter() for L in lengths}
    marg = 0.0
    preps = []
    metas = []
    for rec in iter_windows(set_file):
        preps.append(prep(rec))
        init = initial_frame(rec)
        cav = {str(r["id"]): bool(r.get("is_cav", False))
               for r in init["vehicles"] if r.get("present", True)}
        metas.append((str(rec.get("world")), cav))
    # optional label-free EM: per-type CF residual mean/std from posteriors (no truth)
    cf_bias = np.zeros(3)
    cf_scale = np.full(3, scale)
    for _ in range(em_rounds):
        sw = np.zeros(3)
        swr = np.zeros(3)
        swr2 = np.zeros(3)
        for d in preps:
            res = d["a_obs"][..., None] - d["ap"]
            tok = d["tok"]
            cf = tok & (d["gap"][:-1] < 100.0)
            v0c = d["vlim_ff"][..., None] * np.array(
                [SPEED_FACTOR[n] for n in TYPE_NAMES])[None, None, :]
            gate = d["v_ff"] >= 0.85 * d["vlim_ff"]
            free = tok & (d["gap"][:-1] >= 100.0)
            over_m = np.maximum(d["over"] - 1.0, 0.0)
            ll = (np.where(cf[..., None], ll_students(res - cf_bias, cf_scale), 0.0)
                  + np.where((free & gate)[..., None], ll_students(res - cf_bias, cf_scale), 0.0)
                  + np.where(free[..., None], -0.5 * over_m ** 2, 0.0))
            x = ll.sum(0) + LPRIOR
            xe = np.exp(x - x.max(-1, keepdims=True))
            post = xe / xe.sum(-1, keepdims=True)                 # [N,3]
            w = post[None] * cf[..., None]                        # [T-1,N,3]
            sw += w.sum((0, 1))
            swr += (w * res).sum((0, 1))
            swr2 += (w * res ** 2).sum((0, 1))
        mu = swr / np.maximum(sw, 1.0)
        var = np.maximum(swr2 / np.maximum(sw, 1.0) - mu ** 2, 0.01)
        cf_bias, cf_scale = mu, np.sqrt(var)
    for (world, cav), d in zip(metas, preps):
        veh_types = TRUTH[world]["vehicles"]
        col = {x: i for i, x in enumerate(d["ids"])}
        veh_types = TRUTH[world]["vehicles"]
        tok = d["tok"] & ~d["leader_changed"] if variant.endswith("_lc") else d["tok"]
        res = d["a_obs"][..., None] - d["ap"]
        cf = tok & (d["gap"][:-1] < 100.0)
        ff = tok & ~cf
        if variant == "sigma_gauss":
            std = np.array([SIGMA[t] for t in TYPE_NAMES])
            ll = ll_gauss(res, std[None, None])
        elif variant in ("gauss_shared", "gauss_shared_lc"):
            ll = ll_gauss(res, scale)
        else:
            ll = ll_students(res, scale)
        if variant == "ff_nearsat_em":
            gate = d["v_ff"] >= 0.85 * d["vlim_ff"]
            free = tok & (d["gap"][:-1] >= 100.0)
            llc = ll_gauss(res - cf_bias, cf_scale)
            over_m = np.maximum(d["over"] - 1.0, 0.0)
            ll = (np.where((cf | (free & gate))[..., None], llc, 0.0)
                  + np.where(free[..., None], -0.5 * over_m ** 2, 0.0))
        elif variant.startswith("cf_"):
            # accel evidence on car-following ticks only
            llc = np.where(cf[..., None], ll, 0.0)
            # speed-vs-v0 evidence on free ticks: only overspeed beyond margin m counts
            m, w = 1.0, 1.0
            over_m = np.maximum(d["over"] - m, 0.0)
            llf = np.where(ff[..., None], -0.5 * (over_m / w) ** 2, 0.0)
            ll = llc + llf
        elif variant == "ff_het":
            # CF: student-t shared scale; FF accel: student-t with scale growing with
            # the candidate-type distance-to-saturation (v0 - v); + overspeed penalty
            v0c = d["vlim_ff"][..., None] * np.array(
                [SPEED_FACTOR[n] for n in TYPE_NAMES])[None, None, :]
            dist = np.maximum(v0c - d["v_ff"][..., None], 0.0)
            sff = scale + 0.35 * dist
            llf = -0.5 * np.log1p(res * res / (3.0 * sff * sff)) - np.log(sff)
            m, w = 1.0, 1.0
            over_m = np.maximum(d["over"] - m, 0.0)
            ll = (np.where(cf[..., None], ll, 0.0)
                  + np.where(ff[..., None], llf, 0.0)
                  + np.where(ff[..., None], -0.5 * (over_m / w) ** 2, 0.0))
        elif variant == "ff_nearsat":
            # FF accel evidence only when v >= 0.85 x lane limit (near saturation)
            gate = d["v_ff"] >= 0.85 * d["vlim_ff"]
            m, w = 1.0, 1.0
            over_m = np.maximum(d["over"] - m, 0.0)
            ll = (np.where((tok & ~ff)[..., None], ll, 0.0)
                  + np.where((ff & gate)[..., None], ll, 0.0)
                  + np.where(ff[..., None], -0.5 * (over_m / w) ** 2, 0.0))
        else:
            ffpen = -0.5 * (d["over"] / 1.5) ** 2
            ll = np.where(tok[..., None], ll, 0.0) + np.where(d["free"][..., None], ffpen, 0.0)
        n_ticks = tok.sum(0)
        nt = len(lengths) and None
        tlen = d["tok"].shape[0]
        for L in lengths:
            use = int(round(L / DT))
            lo = max(0, tlen - use)
            tot = ll[lo:].sum(0)
            for vid, is_cav in cav.items():
                if is_cav or vid not in col or vid not in veh_types:
                    continue
                j = col[vid]
                if n_ticks[j] == 0:
                    post = LPRIOR.copy()
                else:
                    x = tot[j] + LPRIOR
                    x = x - x.max()
                    post = np.exp(x)
                    post /= post.sum()
                am = int(post.argmax())
                acc[L][1] += 1
                acc[L][0] += int(TYPE_NAMES[am] == veh_types[vid])
                conf[L][(veh_types[vid], TYPE_NAMES[am])] += 1
            if L == lengths[-1]:
                x = tot + LPRIOR
                xe = np.exp(x - x.max(-1, keepdims=True))
                logz = np.log(xe.sum(-1)) + x.max(-1)
                marg += float(np.sum(logz[n_ticks > 0]))
    return {L: acc[L][0] / acc[L][1] for L in lengths}, conf, marg


def main() -> int:
    out = {}
    for set_name, (set_file, lengths) in SETS.items():
        recs_pilot = [prep(r) for r in iter_windows(set_file)]
        # label-free pilot: argmax-type residual std (naive sigma likelihood)
        allr = []
        for d in recs_pilot:
            res = d["a_obs"][..., None] - d["ap"]
            lln = np.where(d["tok"][..., None], ll_gauss(
                res, np.array([SIGMA[t] for t in TYPE_NAMES])[None, None]), 0.0).sum(0)
            am = lln.argmax(-1)
            r = res[:, np.arange(res.shape[1]), am]
            allr.extend(r[d["tok"]].tolist())
        scale = float(np.std(allr))
        print(f"{set_name}: label-free shared scale s = {scale:.3f}", flush=True)
        variants = ("sigma_gauss", "gauss_shared", "students", "cf_students",
                    "ff_het", "ff_nearsat", "ff_nearsat_em")
        out[set_name] = {"scale": scale}
        for variant in variants:
            em = 2 if variant == "ff_nearsat_em" else 0
            accs, conf, marg = evaluate(set_name, set_file, lengths, variant, scale,
                                        em_rounds=em)
            out[set_name][variant] = {"acc": accs, "marginal_ll": marg,
                                      "conf_last": {f"{a}|{b}": c for (a, b), c in conf[lengths[-1]].items()}}
            print(f"  {variant:14s} acc={accs} margll={marg:.1f}", flush=True)
    (HERE.parent / "out/noise_study.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
