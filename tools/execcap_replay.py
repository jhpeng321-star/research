"""EXECCAP verification replay (EXECCAP_NOTES.md): recorded SUMO MAPPO episodes re-run in WM-S.

Derived from the 2026-10-14 cmdreal diagnosis (E:/agent-work/tmp/cmdreal/wm_replay.py, wm_closed.py),
but the capped row now goes through the REAL model switch: ``model.cav_exec_cap = "idm"`` puts the
physics-core ceiling into the callback state and the executor applies ``wms.model.cav.apply_exec_cap``
-- exactly what the bridge / coop-imagine / trainer executors do.  Read-only: no training, no SUMO.

Modes
  --mode closed  rows (one batch) over a recorded window, horizon 120 s:
      0 TEACHER   realized SUMO CAV motion            1 NATIVE  CAVs are WM drivers
      2 CMD       commands -> safe_accel -> direct_command_motion (switch-off executor)
      3 CMD_CAP   CMD + the model's execution cap (cav_exec_cap="idm")  <- the new path
      4 CMD_CAPT  CMD + min(., IDM with the TRUE SUMO vType params)     (reference)
    ``--cmd policy`` (default) = closed loop: the commands come from the MAPPO policy acting on the
    imagined state; ``--cmd issued`` = the recorded issued commands (open loop).
  --mode exec    teacher-forced executor gate: every control tick c of the window starts from the
    RECORDED SUMO state, holds the recorded issued command for 1 s (2 physics steps) through the
    executor with the switch off / on, and compares the realized (v[i+2]-v[i])/1 s of direct CAVs
    with SUMO's (same convention as cmdreal/sumo_stats.py).

    python tools/execcap_replay.py --ckpt <wm .pt> --tag C4 --mode closed --out <jsonl> [--shard i n]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from traffic_shepherd.scenarios import request_schedule  # noqa: E402

DT = 0.5
VLIM = 31.29
HOR = (30, 60, 120)
TYPES = {"aggressive_car": (0.95, 1.08, 3.0, 4.5, 1.5), "normal_car": (1.1, 1.0, 2.6, 4.5, 2.5),
         "cautious_car": (1.5, 0.92, 2.0, 3.0, 3.5)}  # tau, speedFactor, accel, decel, minGap
ROWS = ("TEACHER", "NATIVE", "CMD", "CMD_CAP", "CMD_CAPT")
CONV = "E:/agent-work/tmp/cmdreal/conv"
POLICY = "E:/agent-work/tmp/cmdreal/policy/mappo_ft_it45.pt"
EPROOT = Path("E:/agent-work/tmp/actiondata/episodes/compare2/mappo_ft_it45/episodes/mappo_ft_it45")
K_FULL = 192
DUMP = None   # --dump: per-sample (a_imag, a_sumo, issued, v) arrays of the exec gate
ACC_MIN, ACC_MAX = -3.0, 2.6


def load_model(ckpt):
    from wms.eval.backend import load_backend
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = load_backend(ckpt, device="cpu", cav_exec_cap="idm").model
    bad = [str(w.message) for w in caught if "physics" in str(w.message).lower()]
    if bad:
        raise SystemExit(f"PHYSICS FALLBACK for {ckpt}: {bad}")
    model.eval()
    return model


def leader(s, lane, v, length, present):
    valid = present & (lane >= 0)
    ds = s[:, None, :] - s[:, :, None]
    k = s.shape[-1]
    other = ~torch.eye(k, dtype=torch.bool)[None]
    ahead = valid[:, :, None] & valid[:, None, :] & (lane[:, :, None] == lane[:, None, :]) & (ds >= 0) & other
    dist, idx = torch.where(ahead, ds, torch.full_like(ds, float("inf"))).min(-1)
    has = torch.isfinite(dist)
    gap = torch.where(has, dist - length.gather(-1, idx), torch.full_like(dist, 1e6))
    return gap, torch.where(has, v.gather(-1, idx), v), has


def idm(v, vl, gap, has, v0, tau, a, b, s0):
    sstar = s0 + (v * tau + v * (v - vl) / (2 * torch.sqrt(a * b))).clamp_min(0)
    inter = torch.where(has, (sstar / gap.clamp_min(0.1)) ** 2, torch.zeros_like(v))
    return a * (1 - (v / v0.clamp_min(1e-3)) ** 4 - inter)


class Driver:
    """CAV callback for all rows.  ``cap_rows``: rows using the model's exec cap (state key);
    ``true_rows``: {row: true vType params}; ``cmd_rows``: rows executing commands (others teacher)."""

    def __init__(self, teacher, ctrl, acc, nrow, *, cmd_rows, cap_rows, true_rows, keep,
                 policy=None, step_offset=0):
        self.teacher, self.ctrl, self.acc, self.nrow = teacher, ctrl, acc, nrow
        self.cmd_rows, self.cap_rows, self.true_rows = cmd_rows, cap_rows, true_rows
        self.keep, self.step_offset = keep, step_offset
        self.held = None
        self.fns = None
        if policy is not None:
            from wms.control.policy import load_policy_fn
            base = load_policy_fn(policy)
            self.fns = {r: copy.deepcopy(base) for r in cmd_rows}
        self.log = {}
        self.cap_missing = 0

    def _policy_cmd(self, state, present):
        held = torch.zeros_like(state["s_m"])
        for r in self.cmd_rows:
            full = {}
            for key, dt_ in (("s_m", torch.float32), ("v_mps", torch.float32), ("a_mps2", torch.float32),
                             ("lane", torch.long), ("present", torch.bool), ("length_m", torch.float32)):
                val = state.get(key)
                f_ = torch.full((1, K_FULL), 5.0) if key == "length_m" else torch.zeros(1, K_FULL, dtype=dt_)
                if val is not None:
                    f_[0, self.keep] = val[r].to(dt_)
                full[key] = f_
            cavf = torch.zeros(1, K_FULL, dtype=torch.bool)
            cavf[0, self.keep] = state["is_cav"][r].bool() & present[r]
            full["is_cav"] = cavf
            full["junction"] = torch.zeros(1, K_FULL, dtype=torch.bool)
            out_ = self.fns[r](full)
            row = torch.as_tensor(np.asarray(out_, np.float64)).reshape(-1)[self.keep].float()
            held[r] = torch.nan_to_num(row, nan=0.0)
        return held

    def __call__(self, state):
        from wms.control.safety import safe_accel
        from wms.model.cav import EXEC_CAP_KEY, apply_exec_cap, direct_command_motion
        i = int(torch.as_tensor(state["physics_step_index"]).reshape(-1)[0])
        tch = self.teacher[:, i + 1]
        s, v = state["s_m"], state["v_mps"]
        present = state["present"].bool()
        lane = state["lane"].long()
        length = state.get("length_m", torch.full_like(s, 5.0))
        c = (i + self.step_offset) // 2
        ctrl_c = self.ctrl[c] if self.ctrl.ndim == 2 else self.ctrl[:, c]
        direct = (ctrl_c == 3).expand_as(present) & present & state["is_cav"].bool()
        if i % 2 == 0:
            if self.fns is not None:
                cmd = self._policy_cmd(state, present)
            else:
                acc_c = self.acc[c] if self.acc.ndim == 2 else self.acc[:, c]
                cmd = acc_c.expand_as(s).float()
            self.held = safe_accel({"s_m": s, "v_mps": v, "lane": lane, "present": present,
                                    "is_cav": state["is_cav"].bool() & present, "length_m": length},
                                   torch.where(direct, cmd, torch.zeros_like(cmd)), accel_max=ACC_MAX)
        a = self.held.clamp(ACC_MIN, ACC_MAX)
        if self.cap_rows:
            if EXEC_CAP_KEY not in state:
                self.cap_missing += 1
            rows = torch.zeros(self.nrow, 1, dtype=torch.bool)
            rows[list(self.cap_rows)] = True
            a = apply_exec_cap(state, a, rows.expand_as(a))
        if self.true_rows:
            gap, vl, has = leader(s, lane, v, length, present)
            a = a.clone()
            for r, par in self.true_rows.items():
                a[r] = torch.minimum(a[r], idm(v[r], vl[r], gap[r], has[r], *par))
        vv = torch.where(present, v, torch.zeros_like(v))
        ss = torch.where(present, s, torch.zeros_like(s))
        command = torch.stack((a, vv + ACC_MAX * DT), -1)
        ns, nv, accel, _ = direct_command_motion(ss, vv, command, DT, ACC_MIN, ACC_MAX)
        out = torch.stack((ns, tch[..., 1], nv, accel), -1)
        cmdrows = torch.zeros(self.nrow, dtype=torch.bool)
        cmdrows[list(self.cmd_rows)] = True
        use = direct & cmdrows[:, None]
        out = torch.where(use[..., None], out, tch)
        self.log[i] = torch.where(use, accel, torch.full_like(accel, float("nan"))).numpy()
        return out


def delay(speeds, mask, sel):
    d = DT * torch.relu(1 - speeds / VLIM) * (mask & sel[:, None, :])
    return d.sum(-1).cumsum(-1)


def _window(rec):
    states = torch.as_tensor(rec["states"]).float()
    alive = torch.as_tensor(rec["present"]).bool()
    keep = torch.nonzero(alive[0]).reshape(-1)
    tl = rec["action_timeline"]
    return dict(
        keep=keep, states=states[:, keep], alive=alive[:, keep],
        is_cav=torch.as_tensor(rec["is_cav"]).bool()[:, keep],
        teacher=torch.as_tensor(rec["cav_teacher"]).float()[:, keep],
        ctrl=torch.as_tensor(np.asarray(tl["controller"]))[:, keep],
        acc=torch.as_tensor(np.asarray(tl["acceleration_mps2"])).float()[:, keep],
        hist=torch.as_tensor(rec["hist"])[keep][None],
        hmask=torch.as_tensor(rec["hist_mask"])[keep][None].bool(),
        dh=torch.as_tensor(rec["driver_hist"])[keep][None],
        dhm=torch.as_tensor(rec["driver_hist_mask"])[keep][None].bool())


def _drivers(model, w):
    from wms.train.inputs import state_from_kin
    init = state_from_kin(w["states"][None, 0], w["alive"][None, 0])
    init["is_cav"] = w["is_cav"][None, 0]
    enc_state = dict(init, is_cav=torch.zeros_like(init["is_cav"]))  # every slot as a driver
    kw = {"history": w["hist"], "history_mask": w["hmask"]}
    if model.long_history:
        kw.update(driver_history=w["dh"], driver_history_mask=w["dhm"])
    else:
        kw.update(driver_history=w["hist"], driver_history_mask=w["hmask"])
    _, emb, params = model.encode_drivers(enc_state, **kw)
    return init, kw, emb, params


def _true_params(ep_meta, keep):
    tid = ep_meta["types"]
    vid0 = ep_meta["vid"][keep.numpy()]
    tp = np.array([TYPES[tid[ep_meta["ids"][j]]] if j >= 0 else TYPES["normal_car"] for j in vid0],
                  np.float32)
    tau, sf, a_, b_, s0 = (torch.as_tensor(tp[:, i]) for i in range(5))
    return (VLIM * sf, tau, a_, b_, s0)


@torch.no_grad()
def run_closed(model, rec, ep_meta, policy):
    w = _window(rec)
    states, alive, is_cav, keep = w["states"], w["alive"], w["is_cav"], w["keep"]
    B, T = len(ROWS), states.shape[0] - 1
    init, kw, emb, params = _drivers(model, w)
    drivers = {}
    if params is not None:
        drivers["driver_params"] = {p: params[p].repeat(B, 1) for p in params}
    if emb is not None:
        drivers["driver_embedding"] = emb.repeat(B, 1, 1)
    st = {k: v.repeat(B, *([1] * (v.ndim - 1))) for k, v in init.items()}
    cav_seq = (is_cav & alive)[None].repeat(B, 1, 1)
    cav_seq[1] = False
    st["is_cav"] = cav_seq[:, 0].clone()
    drv = Driver(w["teacher"][None].repeat(B, 1, 1, 1), w["ctrl"], w["acc"], B,
                 cmd_rows=(2, 3, 4), cap_rows=(3,), true_rows={4: _true_params(ep_meta, keep)},
                 keep=keep, policy=policy)
    rep = lambda x: x.repeat(B, *([1] * (x.ndim - 1)))  # noqa: E731
    kw = {k: rep(v) for k, v in kw.items()}
    roll = model.rollout(st, n_steps=T, cav_teacher=None, cav_controller=drv,
                         present_seq=alive[None].repeat(B, 1, 1), is_cav_seq=cav_seq, **kw, **drivers)
    sp, ms, pos = roll["speeds"], roll["masks"].bool(), roll["positions"]
    hdv0, cav0 = alive[0] & ~is_cav[0], alive[0] & is_cav[0]
    true_sp, true_ms, true_pos = states[1:, :, 4][None], alive[1:][None], states[1:, :, 0][None]
    dh_im, dc_im = delay(sp, ms, hdv0[None]), delay(sp, ms, cav0[None])
    dh_tr, dc_tr = delay(true_sp, true_ms, hdv0[None]), delay(true_sp, true_ms, cav0[None])
    dlog = np.full((B, T, states.shape[1]), np.nan, np.float32)
    for i_, a_l in drv.log.items():
        dlog[:, i_] = a_l
    lane_ok = (states[1:, :, 1:4].sum(-1) > 0) & (states[:-1, :, 1:4].sum(-1) > 0)
    real_a = torch.where(lane_ok, states[1:, :, 5], torch.full_like(states[1:, :, 5], float("nan"))).numpy()
    out = {"meta": rec["meta"], "cb_missing_steps": T - len(drv.log), "cap_missing_steps": drv.cap_missing,
           "n_slots": int(keep.numel()), "n_cav": int(cav0.sum()), "h": {}}
    for h in HOR:
        n = int(h / DT)
        m_both = ms[:, n - 1] & true_ms[:, n - 1]
        err_h = (pos[:, n - 1] - true_pos[:, n - 1]).abs()
        res = {"hdv_delay_true": float(dh_tr[0, n - 1]), "cav_delay_true": float(dc_tr[0, n - 1])}
        for r, name in enumerate(ROWS):
            mh, mc = m_both[r] & hdv0, m_both[r] & cav0
            res[name] = {
                "hdv_delay": float(dh_im[r, n - 1]), "cav_delay": float(dc_im[r, n - 1]),
                "hdv_abs_ds_med": float(err_h[r][mh].median()) if mh.any() else None,
                "cav_abs_ds_mean": float(err_h[r][mc].mean()) if mc.any() else None,
            }
        dmask = np.isfinite(dlog[2, :n])
        res["sumo_direct_mean_accel"] = float(np.nanmean(real_a[:n][dmask])) if dmask.any() else None
        res["cmd_direct_mean_accel"] = {nm: float(np.nanmean(dlog[r, :n][np.isfinite(dlog[r, :n])]))
                                        if np.isfinite(dlog[r, :n]).any() else None
                                        for r, nm in enumerate(ROWS) if r >= 2}
        out["h"][h] = res
    return out


@torch.no_grad()
def run_exec(model, rec, ep_meta, max_s=120.0):
    """Teacher-forced 1 s executor replay from every recorded control tick (switch off vs on)."""
    from wms.train.inputs import state_from_kin
    w = _window(rec)
    states, alive, is_cav, keep = w["states"], w["alive"], w["is_cav"], w["keep"]
    T = states.shape[0] - 1
    init, kw, emb, params = _drivers(model, w)
    vid = ep_meta["vid_all"]
    tick0 = ep_meta["tick"]
    ticks = [i for i in range(0, min(T - 1, int(max_s / DT)), 2)]
    nb = len(ticks)
    idx = torch.as_tensor(ticks)
    s0 = state_from_kin(states[idx], alive[idx])
    s0["is_cav"] = is_cav[idx] & alive[idx]
    out = {"meta": rec["meta"], "n_ticks": nb}
    errs = {}
    tp = _true_params(ep_meta, keep)
    for mode in ("off", "idm", "true", "idm_truep"):
        # "true" (reference): switch off + min(., IDM with the TRUE SUMO vType params of the slot)
        # "idm_truep" (diagnostic): the model cap, CAV-slot driver settings replaced by the TRUE
        # vType values (isolates physics-core vs inferred-settings error)
        model.cav_exec_cap = "off" if mode == "true" else mode[:3]
        b = nb
        drivers = {}
        if params is not None:
            drivers["driver_params"] = {p: params[p].repeat(b, 1) for p in params}
        if emb is not None:
            drivers["driver_embedding"] = emb.repeat(b, 1, 1)
        if mode == "idm_truep" and params is not None:
            cav_slots = is_cav[0] & alive[0]
            v0, tau, a_, b_, s0_ = tp
            for name, value in (("v0", v0), ("time_headway", tau), ("min_gap", s0_), ("accel", a_),
                                ("decel", b_)):
                drivers["driver_params"][name] = torch.where(
                    cav_slots[None], value[None].to(params[name].dtype), drivers["driver_params"][name])
        teacher = torch.stack([w["teacher"][i:i + 3] for i in ticks])          # [b, 3, K, 4]
        ctrl = w["ctrl"][[i // 2 for i in ticks]]                               # [b, K]
        acc = w["acc"][[i // 2 for i in ticks]]
        # ctrl/acc [b, 1, K]: one held command per row (control index 0 of the 1 s replay);
        # every row is a command row and uses the model cap when the switch is on.
        drv = Driver(teacher, ctrl[:, None, :], acc[:, None, :], b, cmd_rows=tuple(range(b)),
                     cap_rows=tuple(range(b)) if mode.startswith("idm") else (),
                     true_rows=({r: tp for r in range(b)} if mode == "true" else {}), keep=keep)
        present_seq = torch.stack([alive[i:i + 3] for i in ticks])
        roll = model.rollout({k: v.clone() for k, v in s0.items()}, n_steps=2, cav_teacher=None,
                             cav_controller=drv, present_seq=present_seq,
                             is_cav_seq=torch.stack([(is_cav & alive)[i:i + 3] for i in ticks]),
                             **drivers)
        v_im = roll["speeds"][:, 1]                                              # after 2 steps
        v_now = states[idx, :, 4]
        v_real = states[idx + 2, :, 4]
        lane_now = states[idx, :, 1:4].sum(-1) > 0
        # same vehicle over the 1 s AND the vehicle the driver settings were encoded for (t0)
        same = torch.as_tensor(np.stack([(vid[tick0 + i + 2][keep.numpy()] == vid[tick0][keep.numpy()])
                                         & (vid[tick0 + i][keep.numpy()] == vid[tick0][keep.numpy()])
                                         for i in ticks]))
        sel = (ctrl == 3) & is_cav[idx] & alive[idx] & alive[idx + 2] & lane_now & same \
            & roll["masks"][:, 1].bool()
        a_im = (v_im - v_now) / 1.0
        a_real = (v_real - v_now) / 1.0
        e = (a_im - a_real)[sel]
        errs[mode] = e
        out[mode] = {"n": int(sel.sum()), "mae": float(e.abs().mean()) if e.numel() else None,
                     "bias": float(e.mean()) if e.numel() else None,
                     "mean_a_imag": float(a_im[sel].mean()) if e.numel() else None,
                     "mean_a_sumo": float(a_real[sel].mean()) if e.numel() else None,
                     "mean_issued": float(acc[sel].mean()) if e.numel() else None,
                     "cap_missing_steps": drv.cap_missing}
        out[mode]["sum_abs"] = float(e.abs().sum())
        out[mode]["median_abs"] = float(e.abs().median()) if e.numel() else None
        out[mode]["frac_abs_lt_0.15"] = float((e.abs() < 0.15).float().mean()) if e.numel() else None
        if DUMP is not None:
            DUMP.setdefault(mode, []).append(np.stack([a_im[sel].numpy(), a_real[sel].numpy(),
                                                       acc[sel].numpy(), v_now[sel].numpy()], -1))
    model.cav_exec_cap = "idm"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("closed", "exec"), default="closed")
    ap.add_argument("--cmd", choices=("policy", "issued"), default="policy")
    ap.add_argument("--t0", type=float, nargs="+", default=[480])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pens", type=int, nargs="+", default=[20, 40])
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--dump", default=None, help="exec mode: npz of per-sample gate rows")
    args = ap.parse_args()
    global DUMP
    if args.dump:
        DUMP = {}
    torch.set_num_threads(args.threads)
    model = load_model(args.ckpt)
    from wms.data.c4_episodes import build_c4_datasets
    ds = build_c4_datasets(CONV, horizon_steps=int(max(HOR) / DT), history_s=60.0, history_stride_s=1.0,
                           entries_v2=False, stride_s=120.0, start_s=360.0)["train"]
    loader = ds.loader if hasattr(ds, "loader") else ds._loader
    meta_cache = {}
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(json.loads(x)["meta"]["world"], json.loads(x)["meta"]["t0_s"])
                for x in out.read_text().splitlines() if x.strip()}
    n = elig = 0
    for idx in range(len(ds)):
        e, tick = loader.index[idx]
        t0 = tick * DT
        if all(abs(t0 - x) > 1e-6 for x in args.t0):
            continue
        world = loader.entries[e]["world"]
        if int(world[11:13]) not in args.pens:
            continue
        elig += 1
        if (elig - 1) % args.shard[1] != args.shard[0] or (world, t0) in done:
            continue
        if args.limit is not None and n >= args.limit:
            break
        if e not in meta_cache:
            seed, dem, pen = int(world[1:5]), int(world[6:10]), int(world[11:13]) / 100
            z = np.load(EPROOT / f"s{seed}_m{dem}_p{int(round(pen * 1000)):04d}" / "episode.npz")
            req = request_schedule(seed, dem, penetration=pen)
            meta_cache[e] = {"types": {r.vehicle_id: r.driver_type for r in req},
                             "ids": z["vehicle_ids"], "vid_all": z["vid_index"]}
        rec = loader(idx)
        m = dict(meta_cache[e])
        m["vid"], m["tick"] = m["vid_all"][tick], tick
        started = time.monotonic()
        if args.mode == "closed":
            res = run_closed(model, rec, m, POLICY if args.cmd == "policy" else None)
        else:
            res = run_exec(model, rec, m)
        res.update(tag=args.tag, mode=args.mode, cmd=args.cmd, ckpt=args.ckpt,
                   wall_s=time.monotonic() - started)
        with out.open("a") as f:
            f.write(json.dumps(res) + "\n")
        n += 1
        print(world, t0, f"{res['wall_s']:.0f}s", flush=True)
        if DUMP is not None:
            np.savez(args.dump, **{k: np.concatenate(v) for k, v in DUMP.items()})


if __name__ == "__main__":
    main()
