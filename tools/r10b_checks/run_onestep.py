"""R10B checks, one-step (teacher-forced recorded-state replay) phase: M3 negative controls and the fix-B tail reads.

For TRAIN windows (corpus roots with train-role episodes only; the same replay the cut-in / courtesy losses use), every arm's
model acceleration is stored for ALL HDV transition rows in lanes 0 and 1 together with recorded-state descriptors
(partner geometry, ramp-car-behind flag, leader gap, insertion-event membership).  No gradient, no SUMO.

  python run_onestep.py --repo REPO --config c4_wm_v10round10b.toml --var r9=CK --var r10b=CK --var headzero=CK \
      --roots ROOT [ROOT...] --stride-eps 4 --workers 24 --out DIR [--limit N]
Windows: one 120 s window (drawn with a fixed rng) of every ``stride_eps``-th episode of each root.

R12 options (all default off; without them the outputs are unchanged):
  --replay-present filled   the replay STATE of every arm uses the filled window presence (``inp.present``, the round-12
                            ``loss.recorded_replay_present = "filled"`` switch); supervision rows stay loss_valid rows.
  --base-cols               append BASE_COLS: own-lane leader bumper gap / leader speed from the replay presence and the
                            per-row encoded IDM accel / decel the model received (driver_params; NaN if the model has none).
  --arm-set ARM=KEY=VALUE   a config override for one arm only (e.g. r9=model.driver_param_mode=legacy when --config is the
                            physical-encoder round-12 config).  An arm whose checkpoint was trained with a different
                            driver_param_mode than its loaded config is refused (the encoder head would be misread)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import END_M, add_paths  # noqa: E402

DT_S = 0.5
COLS = ["t", "slot", "lane", "fin", "pre", "rec", "has", "dA", "dend", "bgap", "rampbehind", "gapL", "ev", "onset",
        "inwin", "ctrl", "v", "off", "win",
        # R11 (appended; analyze.py reads columns by name): time until the partner's merge (s, partner rows), time since
        # the insertion step (s, cut-in event rows), position, closing speed to the same-lane leader
        "tm", "te", "s", "closing"]
# R12 (appended only with --base-cols): own-lane leader bumper gap and leader speed in the REPLAY state (replay presence:
# filled with --replay-present filled), and the encoded IDM accel / decel the model received on that row.
BASE_COLS = ["gapF", "vLF", "dacc", "ddec"]
G: dict = {}


def stored_driver_param_mode(payload) -> str:
    """driver_param_mode a checkpoint was trained with (flat run config; absent = legacy)."""
    cfg = payload.get("config") or {}
    mode = cfg.get("driver_param_mode")
    if mode is None and isinstance(cfg.get("model"), dict):
        mode = cfg["model"].get("driver_param_mode")
    return str(mode or "legacy")


def parse_arm_sets(items) -> dict:
    """["r9=model.x=1", ...] -> {"r9": ("model.x=1", ...)}"""
    out: dict = {}
    for item in items:
        arm, sep, kv = item.partition("=")
        if not sep or "=" not in kv:
            raise SystemExit(f"--arm-set needs ARM=KEY=VALUE, got {item!r}")
        out.setdefault(arm, []).append(kv)
    return {k: tuple(v) for k, v in out.items()}


def _init(repo, config, specs, roots, horizon_s, stride_s, replay_present="loss_valid", base_cols=False,
          arm_sets=None):
    import torch
    torch.set_num_threads(1)
    add_paths(repo)
    from tools.r10_attribution import load_model
    from tools.r10b_courtesy_offline import datasets
    G["arms"] = {}
    G["replay_present"] = replay_present
    G["base_cols"] = bool(base_cols)
    arm_sets = arm_sets or {}
    for name, ck in specs:
        # R11: a version-2 head (13 pair inputs) needs model.follower_courtesy_version=2, otherwise load_model's
        # same-shape matching would silently drop its first layer.
        payload = torch.load(ck, map_location="cpu", weights_only=True)
        stored = payload["model"].get("follower_courtesy.net.0.weight")
        v2 = stored is not None and stored.shape[-1] == 13
        cfg, settings, model, report = load_model(
            config, ck, (("model.follower_courtesy_version=2",) if v2 else ()) + tuple(arm_sets.get(name, ())))
        del payload
        assert model.follower_courtesy is not None
        if stored is not None:   # the stored head must be loaded as is (same-shape matching drops mismatches silently)
            assert torch.equal(model.follower_courtesy.net[0].weight.detach(), stored), name
        G["arms"][name] = (cfg, settings, model, report)
    cfg0 = next(iter(G["arms"].values()))[0]
    G["sets"] = datasets(cfg0, roots, int(round(horizon_s / 0.5)), stride_s)
    G["n_steps"] = int(round(horizon_s / 0.5))


def one_window(job):
    import torch
    from tools.r10_attribution import build_targets, prepare_inputs, collate
    from wms.model.follower_courtesy import follower_courtesy_pairs
    from wms.train.losses.cutin_reaction import (CutinReactionConfig, _recorded, insertion_plan, one_step_replay)
    from wms.train.losses.follower_courtesy import REPLAY_KEYS
    si, i, name_win = job
    out = {}
    # Both arms read identical observable records. Decode/collate once; model-
    # dependent encoding and target construction remain separate and unchanged.
    shared_batch = None
    if os.environ.get('WMS_LOOPLONG_SPEED') == '1':
        shared_batch = collate([G['sets'][si][i]])
    for arm, (cfg, settings, model, _rep) in G["arms"].items():
        ds = G["sets"][si]
        batch = shared_batch if shared_batch is not None else collate([ds[i]])
        inp = prepare_inputs(batch, model, settings, torch.device("cpu"), G["n_steps"])
        tg = build_targets(inp, batch, settings, float(model.physics.dt_s), torch.device("cpu"),
                           physics=model.physics)
        tg.prepared["cutin_reaction_inputs"] = {
            "history": inp.hist, "history_mask": inp.hist_mask,
            "driver_history": inp.observed_hist, "driver_history_mask": inp.observed_hist_mask}
        if G.get("replay_present", "loss_valid") == "filled":
            # R12 fix D, exactly as the round-12 runner: the replay state is the filled window presence.
            tg.prepared["cutin_reaction_inputs"]["present"] = inp.present
        base_cols = bool(G.get("base_cols", False))
        captured: dict = {}
        if base_cols:
            def _fwd(state, *args, _orig=model.forward, **kw):
                params = kw.get("driver_params")
                if params is not None and "accel" in params and "decel" in params:
                    captured[int(kw.get("k_steps", len(captured)))] = (params["accel"].detach()[0].clone(),
                                                                       params["decel"].detach()[0].clone())
                return _orig(state, *args, **kw)
            model.forward = _fwd
        try:
            with torch.no_grad():
                accel, col = one_step_replay(model, tg, collect=tuple(REPLAY_KEYS) + ("courtesy_accel_mps2",))
                plan = insertion_plan(tg, CutinReactionConfig(weight=1.0))
        finally:
            if base_cols:
                del model.forward
        if base_cols:
            from wms.train.losses.cutin_reaction import replay_present
            rpres = replay_present(tg)[0]
        kin, lane, valid, _, cont, hdv = _recorded(tg)
        kin, lane, valid, hdv = kin[0], lane[0], valid[0], hdv[0]
        n = tg.n_steps
        phys = model.physics
        ev_id = np.full((n, kin.shape[1]), -1, np.int32)
        ev_step = np.full((n, kin.shape[1]), np.nan, np.float32)
        from tools.r10b_courtesy_offline import next_merge_steps
        cont_np = cont[0].numpy()
        nm = next_merge_steps(lane.numpy(), np.concatenate([cont_np, np.zeros_like(cont_np[:1])]))
        onset = np.zeros((n, kin.shape[1]), bool)
        inwin = np.zeros((n, kin.shape[1]), bool)
        for e_i, e in enumerate(plan["events"]):
            f = e["follower"]
            for t in range(e["start"], e["end"] + 1):
                ev_id[t, f] = e_i
                ev_step[t, f] = e["step"]
                inwin[t, f] = True
            onset[e["step"], f] = True
        ctrl = plan["control"][0].numpy()
        rows = []
        for t in range(n):
            sel = (hdv[t] & ((lane[t] == 1) | (lane[t] == 0)) & (lane[t + 1] == lane[t])).nonzero().flatten()
            if not len(sel):
                continue
            k_t = kin[t]
            s, v, length = k_t[:, 0], k_t[:, 4], k_t[:, 6]
            pairs = follower_courtesy_pairs(k_t[None], valid[t][None], phys)
            has = pairs["has_partner"][0]
            partner = pairs["partner"][0]
            ramp_wait = valid[t] & (lane[t] == 2) & (s >= 0) & (s <= END_M)
            rel = s[:, None] - s[None, :]                       # s_i - s_j
            behind = (ramp_wait[None, :] & (rel >= 0) & (rel < 20)).any(-1)
            rampbehind = (~has) & behind & (lane[t] == 1)
            ds_ = s[None, :] - s[:, None]                       # s_j - s_i
            same = valid[t][None, :] & (lane[t][None, :] == lane[t][:, None]) & (ds_ > 0)
            gl, lead_i = torch.where(same, ds_ - length[None, :], torch.full_like(ds_, float("inf"))).min(-1)
            closing = torch.where(torch.isfinite(gl), v - v[lead_i], torch.zeros_like(v))
            tm = torch.where(has, torch.tensor((nm[t, partner.numpy()] - t) * DT_S, dtype=torch.float32),
                             torch.full_like(s, float("nan")))
            te = torch.tensor((t - ev_step[t]) * DT_S, dtype=torch.float32)
            dA = torch.where(has, s[partner] - s, torch.full_like(s, float("nan")))
            dend = torch.where(has, END_M - s[partner], torch.full_like(s, float("nan")))
            bgap = torch.where(has, s[partner] - length[partner] - s, torch.full_like(s, float("nan")))
            fin = accel[0, t]
            pre = col["accel_pre_courtesy_mps2"][0, t]
            rec = kin[t + 1, :, 5]
            cols_t = [torch.full((len(sel),), float(t)), sel.float(), lane[t][sel].float(), fin[sel], pre[sel],
                      rec[sel], has[sel].float(), dA[sel], dend[sel], bgap[sel], rampbehind[sel].float(),
                      gl[sel], torch.tensor(ev_id[t])[sel].float(), torch.tensor(onset[t])[sel].float(),
                      torch.tensor(inwin[t])[sel].float(), torch.tensor(ctrl[t])[sel].float(), v[sel], col["courtesy_accel_mps2"][0, t][sel],
                      torch.full((len(sel),), float(i)), tm[sel], te[sel], s[sel], closing[sel]]
            if base_cols:
                same_r = rpres[t][None, :] & (lane[t][None, :] == lane[t][:, None]) & (ds_ > 0)
                gf, lead_f = torch.where(same_r, ds_ - length[None, :], torch.full_like(ds_, float("inf"))).min(-1)
                vlf = torch.where(torch.isfinite(gf), v[lead_f], torch.full_like(v, float("nan")))
                gf = torch.where(torch.isfinite(gf), gf, torch.full_like(gf, float("nan")))
                acc_dec = captured.get(t)
                nan = torch.full_like(s, float("nan"))
                da, dd = (acc_dec[0].float(), acc_dec[1].float()) if acc_dec is not None else (nan, nan)
                cols_t += [gf[sel], vlf[sel], da[sel], dd[sel]]
            m = torch.stack(cols_t, -1)
            rows.append(m.numpy())
        # recorded outcome of every event window: min recorded accel (next-row) over the window
        evb = {}
        for e_i, e in enumerate(plan["events"]):
            f = e["follower"]
            evb[e_i] = float(min(float(kin[t + 1, f, 5]) for t in range(e["start"], e["end"] + 1)))
        ncol = len(COLS) + (len(BASE_COLS) if base_cols else 0)
        arr = np.concatenate(rows).astype(np.float32) if rows else np.zeros((0, ncol), np.float32)
        out[arm] = (arr, evb)
    return name_win, out


def work(job):
    return one_window(job)


def check_driver_modes(repo, config, specs, arm_sets) -> dict:
    """R12 guard, in the main process (a failing Pool initializer would respawn forever): every arm's checkpoint
    driver_param_mode must equal the one its (overridden) config builds.  Returns {arm: mode}."""
    import torch
    from wms.config.load import load_train_config
    out = {}
    for name, ck in specs:
        stored = stored_driver_param_mode(torch.load(ck, map_location="cpu", weights_only=True))
        cfg = load_train_config(os.path.join(repo, "configs", config), ["runtime.device=cpu", *arm_sets.get(name, ())])
        loaded = str(getattr(cfg.model, "driver_param_mode", "legacy"))
        if stored != loaded:
            raise SystemExit(f"arm {name}: checkpoint driver_param_mode {stored!r} but the loaded config gives "
                             f"{loaded!r}; pass --arm-set {name}=model.driver_param_mode={stored}")
        out[name] = loaded
    return out


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", default="c4_wm_v10round10b.toml")
    ap.add_argument("--var", action="append", default=[])
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--stride-eps", type=int, default=4)
    ap.add_argument("--horizon-s", type=float, default=120.0)
    ap.add_argument("--stride-s", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replay-present", choices=("loss_valid", "filled"), default="loss_valid",
                    help="replay-state presence for every arm (R12; default = the pre-R12 loss_valid replay)")
    ap.add_argument("--base-cols", action="store_true", help="append BASE_COLS (R12 base-calibration check)")
    ap.add_argument("--arm-set", action="append", default=[], help="ARM=KEY=VALUE config override for one arm")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    specs = [tuple(v.split("=", 1)) for v in a.var]
    arm_sets = parse_arm_sets(a.arm_set)
    cols = COLS + (BASE_COLS if a.base_cols else [])
    add_paths(a.repo)
    check_driver_modes(a.repo, a.config, specs, arm_sets)
    from tools.r10b_courtesy_offline import datasets
    from wms.config.load import load_train_config
    cfg0 = load_train_config(os.path.join(a.repo, "configs", a.config), ["runtime.device=cpu"])
    sets = datasets(cfg0, a.roots, int(round(a.horizon_s / 0.5)), a.stride_s)
    jobs = []
    for si, ds in enumerate(sets):
        by_ep: dict = {}
        for i, (ep, _tick) in enumerate(ds.positions):
            by_ep.setdefault(ep, []).append(i)
        rng = np.random.default_rng(2000 + si)
        for ep in sorted(by_ep)[::a.stride_eps]:
            idx = by_ep[ep]
            pick = int(rng.choice(idx))
            jobs.append((si, pick, f"{si}:{ds._loader.entries[ep]['episode']}:{pick}"))
    if a.limit:
        jobs = jobs[: a.limit]
    print("windows", len(jobs), flush=True)
    del sets
    t0 = time.time()
    results = {arm: [] for arm, _ in specs}
    events = {arm: {} for arm, _ in specs}
    from multiprocessing import Pool
    with Pool(a.workers, initializer=_init,
              initargs=(a.repo, a.config, specs, a.roots, a.horizon_s, a.stride_s, a.replay_present, a.base_cols,
                        arm_sets)) as pool:
        for j, (nm, out) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            for arm, (arr, evb) in out.items():
                arr[:, COLS.index("win")] = float(j)
                results[arm].append(arr)
                events[arm][j] = evb
            if (j + 1) % 8 == 0:
                print(f"{j+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    for arm, _ in specs:
        np.savez_compressed(os.path.join(a.out, f"onestep_{arm}.npz"),
                            rows=np.concatenate(results[arm]) if results[arm] else np.zeros((0, len(cols)), np.float32),
                            cols=np.array(cols))
        json.dump({str(k): v for k, v in events[arm].items()}, open(os.path.join(a.out, f"onestep_{arm}_events.json"), "w"))
    done = {"windows": len(jobs), "seconds": time.time() - t0, "cols": cols}
    if a.replay_present != "loss_valid" or a.base_cols or arm_sets:   # R12 keys only when an R12 option is used
        done.update(replay_present=a.replay_present, base_cols=bool(a.base_cols),
                    arm_sets={k: list(v) for k, v in arm_sets.items()})
    json.dump(done, open(os.path.join(a.out, "ONESTEP_DONE.json"), "w"))
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
