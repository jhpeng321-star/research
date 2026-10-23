"""S-BASEDIAG: component decomposition of the WM's lane-1 HDV acceleration (CPU, inference only).

Teacher-forced one-step replay (``one_step_replay``, the replay the cut-in / courtesy losses use) on
recorded TRAIN windows.  For every lane-1 HDV transition row it stores

* ``phys``    IDM core with the encoded driver parameters (``physics_accel_mps2``),
* ``attn``    gate * attention/residual-net emission (pre bound),
* ``temp``    gate * temporal-MLP emission (pre bound),
* ``corr``    the bounded learned correction actually added (``bounded(gate*(attn+temp))``),
* ``gate``    gate value,
* ``pre``     acceleration before the plant clamp (courtesy offset excluded), ``final`` after it,
* the IDM inputs the model saw (own lane-1 leader gap / speed) and the encoded IDM parameters,
* the R10B/R11 geometric ramp partner (nearest ramp-lane car the row is the target follower of) and
  DIAGNOSTIC counterfactuals: IDM with the partner treated as a leader at its current position
  (``a_virt``, encoded parameters), and whether an invisible (junction, lane one-hot = 0) vehicle sits
  between the follower and its lane-1 leader.

Nothing is trained or fitted; the virtual-leader numbers are a diagnostic, not a model rule.

    python tools/diag_base_decomp.py --config c4_wm_v10round10b.toml --set model.follower_courtesy_version=2 \
        --ckpt <round-8 stage300> --roots <root>... --out tmp_basediag/r8 --shard 0 --n-shards 8
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from tools.r10_attribution import bounded, load_model  # noqa: E402
from tools.r10b_courtesy_offline import datasets, next_merge_steps  # noqa: E402
from wms.data.collate import collate  # noqa: E402
from wms.model.follower_courtesy import follower_courtesy_pairs  # noqa: E402
from wms.physics.idm import IDMParams, idm_accel  # noqa: E402
from wms.physics.neighbours import lead_neighbour  # noqa: E402
from wms.train.inputs import build_targets, prepare_inputs  # noqa: E402
from wms.train.losses.cutin_reaction import _recorded, one_step_replay  # noqa: E402

PARAMS = ("v0", "time_headway", "min_gap", "accel", "decel")


def replay_components(model, tg):
    captured, store = [], {}
    original = model.forward
    o_res, o_tmp = model._residual_prediction, model._temporal_delta

    def res(*a, **kw):
        out = o_res(*a, **kw)
        store["attn_delta"] = out["accel_delta_mps2"].detach()
        return out

    def tmp(*a, **kw):
        out = o_tmp(*a, **kw)
        store["temp_delta"] = out[..., 0].detach()
        return out

    def hooked(state, **kw):
        out = original(state, **kw)
        row = {k: out[k].detach() for k in ("physics_accel_mps2", "accel_mps2", "gate",
                                            "accel_pre_courtesy_mps2", "courtesy_accel_mps2")
               if k in out}
        row.update(store)
        dp = kw.get("driver_params")
        if dp is not None:
            row["params"] = torch.stack([dp[n].detach() for n in PARAMS], -1)
        captured.append(row)
        return out

    model.forward, model._residual_prediction, model._temporal_delta = hooked, res, tmp
    try:
        with torch.no_grad():
            one_step_replay(model, tg)
    finally:
        for name in ("forward", "_residual_prediction", "_temporal_delta"):
            model.__dict__.pop(name, None)
    return {k: torch.stack([c[k] for c in captured], 1) for k in captured[0]
            if all(k in c for c in captured)}


def window_rows(model, tg):
    phys_cfg = model.physics
    comp = replay_components(model, tg)
    kin_b, lane_b, valid_b, _, cont_b, hdv_b = _recorded(tg)
    kin, lane, valid, cont, hdv = kin_b[0], lane_b[0], valid_b[0], cont_b[0], hdv_b[0]
    n = tg.n_steps
    l1 = hdv & (lane[:-1] == 1) & (lane[1:] == 1)
    nm = next_merge_steps(lane.numpy(), np.concatenate([cont.numpy(), np.zeros_like(cont[:1].numpy())]))
    end = float(phys_cfg.merge_zone_s1_m)
    amin, amax = float(phys_cfg.accel_min_mps2), float(phys_cfg.accel_max_mps2)
    gate = comp["gate"][0]
    attn = gate * comp["attn_delta"][0]
    temp = gate * comp["temp_delta"][0]
    corr = bounded(model, attn + temp)
    phys = comp["physics_accel_mps2"][0]
    final = comp["accel_mps2"][0]
    pre = comp.get("accel_pre_courtesy_mps2", None)
    pre = pre[0] if pre is not None else phys + corr
    params = comp["params"][0] if "params" in comp else None
    # junction / invisible vehicles: occupied in the raw state but no lane one-hot
    raw_kin = tg.states[0, :n + 1]
    invisible = (raw_kin[..., 1:4].sum(-1) == 0) & (raw_kin[..., 6] > 0)
    out = []
    for t in range(n):
        sel = l1[t].nonzero().flatten()
        if not len(sel):
            continue
        k_t = kin[t]
        s, v, length, a_now = k_t[:, 0], k_t[:, 4], k_t[:, 6], k_t[:, 5]
        present = valid[t]
        gap_m, lead_v = lead_neighbour(k_t[None], present[None], phys_cfg)
        gap_m, lead_v = gap_m[0], lead_v[0]
        pairs = follower_courtesy_pairs(k_t[None], present[None], phys_cfg)
        has, partner = pairs["has_partner"][0], pairs["partner"][0]
        s_r, v_r, len_r, a_r = s[partner], v[partner], length[partner], a_now[partner]
        gap_r = s_r - len_r - s
        if params is not None:
            p_t = params[t]
            ip = IDMParams(v0=p_t[:, 0].clamp_min(1e-3), time_headway=p_t[:, 1], s0=p_t[:, 2],
                           a_max=p_t[:, 3], b_comf=p_t[:, 4])
        else:
            ip = phys_cfg.idm_params()
        a_virt = idm_accel(v, v_r, gap_r.clamp_min(0.3), ip, accel_min=amin, accel_max=amax)
        if params is not None:
            a_virt = torch.minimum(torch.maximum(a_virt, -params[t][:, 4]), params[t][:, 3])
        a_free = idm_accel(v, v, torch.full_like(v, 1e4), ip, accel_min=amin, accel_max=amax)
        # invisible vehicle strictly between the follower and the model's leader
        ds = raw_kin[t, :, 0][None, :] - s[:, None]
        inv_between = (invisible[t][None, :] & (ds > 0) & (ds < gap_m[:, None] + 5.0)
                       & (ds < 60.0)).any(-1)
        wait_ramp = present & (lane[t] == 2) & (s >= 0) & (s <= end)
        rel = s[None, :] - s[:, None]
        near = (wait_ramp[None, :] & (rel <= 30) & (rel >= -20)).any(-1)
        c3 = (~has) & bool(wait_ramp.any()) & ~near
        tmerge = torch.tensor(nm[t, partner.numpy()] - t, dtype=torch.float32) * float(tg.dt_s)
        nan = torch.full_like(s, float("nan"))
        row = {
            "phys": phys[t, sel], "attn": attn[t, sel], "temp": temp[t, sel], "corr": corr[t, sel],
            "gate": gate[t, sel], "pre": pre[t, sel], "final": final[t, sel],
            "rec": kin[t + 1, sel, 5], "a_now": a_now[sel],
            "v": v[sel], "s": s[sel], "gap": gap_m[sel], "lead_v": lead_v[sel],
            "has": has[sel], "c3": c3[sel],
            "da": torch.where(has, s_r - s, nan)[sel], "dend": torch.where(has, end - s_r, nan)[sel],
            "gap_r": torch.where(has, gap_r, nan)[sel], "v_r": torch.where(has, v_r, nan)[sel],
            "a_r": torch.where(has, a_r, nan)[sel],
            "tmerge": torch.where(has, tmerge, nan)[sel],
            "a_virt": torch.where(has, a_virt, nan)[sel], "a_free": a_free[sel],
            "inv_between": inv_between[sel],
            "step": torch.full((len(sel),), t, dtype=torch.int32), "slot": sel.to(torch.int32)}
        if params is not None:
            row["params"] = params[t, sel]
        out.append({k: x.numpy() for k, x in row.items()})
    if not out:
        return None
    return {k: np.concatenate([r[k] for r in out]) for k in out[0]}


def run(args):
    torch.set_num_threads(args.threads)
    cfg, settings, model, report = load_model(args.config, args.ckpt, tuple(args.set))
    print("loaded", report["loaded"], "new", report["new"], "unused", report["unused"][:5], flush=True)
    print("physics idm", model.physics.idm(), "accel bounds", model.physics.accel_min_mps2,
          model.physics.accel_max_mps2, "rollout_mode", model.rollout_mode,
          "soft bounds", model.residual_soft_bound_mps2, model.residual_soft_bound_neg_mps2,
          "caps", model.residual_cap_mps2, model.residual_cap_neg_mps2, flush=True)
    n_steps = int(round(args.horizon_s / 0.5))
    sets = datasets(cfg, args.roots, n_steps, args.stride_s)
    jobs = []
    for si, ds in enumerate(sets):
        by_ep: dict[int, list[int]] = {}
        for i, (ep, _tick) in enumerate(ds.positions):
            by_ep.setdefault(ep, []).append(i)
        rng = np.random.default_rng(1000 + si)
        for ep in sorted(by_ep):
            idx = by_ep[ep]
            pick = sorted(rng.choice(idx, size=min(args.windows_per_episode, len(idx)), replace=False))
            jobs.extend((si, int(i), ds._loader.entries[ep]["episode"]) for i in pick)
    rng = np.random.default_rng(7)
    rng.shuffle(jobs)
    jobs = jobs[args.shard::args.n_shards][:args.max_windows]
    path = Path(f"{args.out}.shard{args.shard}.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
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
        packed = window_rows(model, tg)
        if packed is not None:
            packed["episode"], packed["window"] = name, f"{name}:{i}"
            done.append(packed)
        print(f"shard {args.shard}: {len(done)}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
        with open(path, "wb") as f:
            pickle.dump({"windows": done, "report": report, "complete": False}, f)
    with open(path, "wb") as f:
        pickle.dump({"windows": done, "report": report, "complete": True}, f)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--horizon-s", type=float, default=120.0)
    p.add_argument("--stride-s", type=float, default=30.0)
    p.add_argument("--windows-per-episode", type=int, default=1)
    p.add_argument("--max-windows", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=8)
    p.add_argument("--threads", type=int, default=1)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
