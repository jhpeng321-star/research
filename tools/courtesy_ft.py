"""V10COURTESY: CPU fine-tune of v5m with ``physics.lane_learnable`` and a paired
``model.courtesy_inputs`` on/off comparison.

The fine-tune optimises the checkpoint's own loss recipe (position/speed/lane,
aggregate, lane-event, ...) plus ``loss.courtesy_weight``.  Evaluation reports the
teacher-forced follower-acceleration error on yield events and free-rollout stuck
counts / merge distance on 16 TRAIN-val starts.

TRAIN-val converted episodes only (``merge_calib/data/{v5_native,genrand_plain}``); never the
evaluation seeds 964000-964199 and never sealed/holdout/cal_split/chunk5/j4b/inc1 paths.

Subcommands (run from the worktree root with ``PYTHONPATH="src;.;tools"``)::

    python tools/courtesy_ft.py finetune --ckpt CKPT --native-dir DIR --genrand-dir DIR
        --n-native 8 --n-genrand 8 --steps 200 --lr 1e-4 --horizon-s 30
        --courtesy-inputs --courtesy-weight 4 --out OUTDIR --tag on

    python tools/courtesy_ft.py yielddiag --arm "off=CKPT" --arm "on=CKPT"
        --native-dir DIR --genrand-dir DIR --n-native 8 --n-genrand 8 --out yields.json

    python tools/courtesy_ft.py eval --arm "off=CKPT" --arm "on=CKPT"
        --native-dir DIR --genrand-dir DIR --n-native 8 --n-genrand 8 --out rollout.json

The fine-tune keeps the checkpoint's own loss recipe (position/speed/lane, aggregate, residual
penalty, flip-rate, driver prior) and adds the all-risk lane-event loss of the ``hazard_v5``
lanes; every parameter is trainable, the learning rate is small, and no rule is removed beyond
``lane_learnable`` itself.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
EVAL_SEEDS = range(964000, 964200)
DT = 0.5
ZONE_END_M = 75.44


def guard(path: str) -> str:
    s = str(path).replace("\\", "/").lower()
    if any(t in s for t in FORBIDDEN):
        raise SystemExit(f"refusing restricted path {path}")
    return str(path)


def _physics_override(path: str, lane_learnable: bool) -> dict:
    raw = json.loads((Path(path).parent / "physics.json").read_text(encoding="utf-8"))
    raw["lane_learnable"] = bool(lane_learnable)
    return raw


# --------------------------------------------------------------------------- episodes
def select_episodes(root: Path, n: int, offset: int = 0) -> list[dict]:
    """The ``n`` TRAIN-val manifest entries at ``offset`` in the source's ``val_list.txt``."""
    lines = [ln.strip() for ln in (root / "val_list.txt").read_text(encoding="utf-8").split()]
    lines = [ln for ln in lines if ln]
    chosen = lines[offset:offset + n]
    doc = json.loads((root / "C4_EPISODES_MANIFEST.json").read_text(encoding="utf-8"))
    by_dir = {e["dir"]: e for e in doc["episodes"]}
    out = []
    for ln in chosen:
        if ln not in by_dir:
            raise SystemExit(f"{root}: {ln} not in manifest")
        e = by_dir[ln]
        if e.get("split_role") != "val":
            raise SystemExit(f"{ln}: not a val episode")
        if int(e["seed"]) in EVAL_SEEDS:
            raise SystemExit(f"evaluation seed {e['seed']} refused")
        out.append(e)
    return out


def build_window_dataset(entries: list[dict], root: Path, horizon_steps: int,
                         stride_s: float, history_s: float, event_windows: bool = False):
    from wms.data.c4_episodes import C4WindowLoader, _open_episode, episode_window_ticks
    from wms.data.dataset import TrajectoryDataset

    index: list[tuple[int, int]] = []
    n_all = 0
    for i, entry in enumerate(entries):
        ep = _open_episode(str(root / entry["dir"]))
        ticks = episode_window_ticks(ep, horizon_steps=horizon_steps, stride_s=stride_s,
                                     start_s=None, warmup_s=float(entry["warmup_s"]))
        n_all += len(ticks)
        if event_windows:
            import torch
            from wms.physics.config import PhysicsConfig
            from wms.train.courtesy import courtesy_follower_mask
            kin = np.asarray(ep.array("veh_kin"), np.float32)
            present = np.asarray(ep.array("veh_mask"), bool)
            ev = courtesy_follower_mask(
                torch.as_tensor(kin[None]), torch.as_tensor(present[None]),
                PhysicsConfig(merge_zone_s1_m=75.44, merge_lane=2), behind_m=40.0,
                dend_m=40.0)[0].numpy().any(axis=1)
            ticks = [t for t in ticks if bool(ev[t:t + horizon_steps].any())]
        index.extend((i, t) for t in ticks)
    if not index:
        raise SystemExit(f"{root}: no window with a full {horizon_steps * DT:.0f} s horizon")
    loader = C4WindowLoader(root, entries, index, horizon_steps=horizon_steps,
                            history_s=history_s, history_stride_s=1.0, entries_v2=False)
    receipt = {"mode": "c4-episodes", "source": str(root), "n_windows": len(index),
               "n_episodes": len(entries), "horizon_steps": int(horizon_steps),
               "stride_s": float(stride_s), "event_windows": bool(event_windows),
               "n_windows_all": int(n_all)}
    ds = TrajectoryDataset([{} for _ in index], loader=loader, receipt=receipt)
    ds.positions = list(index)
    return ds


# --------------------------------------------------------------------------- fine-tune
def finetune(args) -> int:
    import torch
    from torch.utils.data import ConcatDataset, DataLoader

    from wms.config.train import TrainConfig
    from wms.data.collate import collate
    from wms.model.checkpoint import load_checkpoint, save_checkpoint
    from wms.train.losses.compose import build_loss
    from wms.train.settings import TrainSettings
    from wms.train.unroll import UnrollRunner

    torch.set_num_threads(int(args.threads))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    ckpt = guard(args.ckpt)
    phys = _physics_override(ckpt, lane_learnable=True)
    loaded = load_checkpoint(ckpt, device="cpu", physics=phys,
                             config_factory=TrainConfig.from_legacy_flat)
    model, train_cfg = loaded.model, loaded.train_cfg
    settings = TrainSettings.from_config(train_cfg)
    settings = dataclasses.replace(
        settings, device="cpu", threads=1, num_workers=0, batch_size=int(args.batch),
        lr=float(args.lr), select_by_long_horizon=False, val_all_horizons=False,
        selection_mode="long_horizon", loss_share_logging=False,
        courtesy_weight=float(args.courtesy_weight))
    os.makedirs(args.out, exist_ok=True)

    model.apply_direct_commands = False
    model.set_courtesy_inputs(bool(args.courtesy_inputs))
    if float(args.desaturate) != 1.0:
        # v5m's residual accel head is saturated (raw ~ -26 -> |tanh| ~ 1), so the
        # courtesy adapter would receive a zero gradient.  Both on/off arms get the
        # same rescale; forward values move only by the small tanh change.
        with torch.no_grad():
            last = model.residual_net.accel[-1]
            last.weight.mul_(float(args.desaturate))
            last.bias.mul_(float(args.desaturate))
    if getattr(settings, "lane_hazard_decouple", False):
        model.lane_hazard_detach = True
    if settings.lane_logit_cap is not None:
        model.lane_logit_cap = float(settings.lane_logit_cap)
    if settings.residual_soft_bound_mps2 is not None:
        model.residual_soft_bound_mps2 = float(settings.residual_soft_bound_mps2)
        model.residual_soft_bound_neg_mps2 = settings.residual_soft_bound_neg_mps2

    native = Path(guard(args.native_dir))
    genrand = Path(guard(args.genrand_dir))
    nat_eps = select_episodes(native, int(args.n_native), 0)
    gr_eps = select_episodes(genrand, int(args.n_genrand), 0)
    horizon_steps = max(2, int(round(float(args.horizon_s) / DT)))
    ds_nat = build_window_dataset(nat_eps, native, horizon_steps, args.stride,
                                  settings.history_s, bool(args.event_windows))
    ds_gr = build_window_dataset(gr_eps, genrand, horizon_steps, args.stride,
                                 settings.history_s, bool(args.event_windows))
    dataset = ConcatDataset([ds_nat, ds_gr])
    loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0, drop_last=False)

    composer = build_loss(settings, physics=model.physics)
    runner = UnrollRunner(model, settings, composer, projection=None,
                          device=torch.device("cpu"))
    if args.train_lane_only:
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith("lane_v5"))
    params = [p for p in model.parameters() if p.requires_grad]
    if args.courtesy_inputs and float(args.courtesy_lr_mult) != 1.0:
        base = [p for n, p in model.named_parameters() if p.requires_grad
                and "courtesy" not in n]
        new = [p for n, p in model.named_parameters() if p.requires_grad
               and "courtesy" in n]
        optimizer = torch.optim.Adam(
            [{"params": base, "lr": settings.lr},
             {"params": new, "lr": settings.lr * float(args.courtesy_lr_mult)}],
            weight_decay=settings.weight_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=settings.lr, weight_decay=settings.weight_decay)
    unroll = int(args.unroll_steps) if args.unroll_steps else horizon_steps
    trunc = int(args.trunc) if args.trunc else unroll
    model.train()
    print(json.dumps({"event": "start", "windows": len(dataset), "steps": args.steps,
                      "lr": settings.lr, "batch": settings.batch_size, "unroll": unroll,
                      "trunc": trunc, "horizon_s": args.horizon_s, "lane_event_w":
                      settings.lane_event_weight, "lane_only": bool(args.train_lane_only)}),
          flush=True)
    log = []
    t0 = time.time()
    step = 0
    if args.train_lane_only:
        # Isolate the lane heads: the all-risk lane-event rows are the loss (recorded rows,
        # no free-rollout term); the backbone is frozen, so this measures only whether the
        # lane network can absorb the signal, not the WM's long-horizon stability.
        import torch.nn.functional as F
        from wms.train.losses.lane_event import lane_v5_batch_rows
        device = torch.device("cpu")
        for batch in loader:
            if step >= int(args.steps):
                break
            rows = lane_v5_batch_rows(model, batch, settings, device)
            if int(rows["features"].shape[0]) == 0:
                continue
            logits = model.lane_v5(rows["features"]).gather(
                -1, rows["head"].unsqueeze(-1)).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, rows["label"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(params, settings.grad_clip))
            optimizer.step()
            step += 1
            log.append({"step": step, "loss": float(loss.detach()),
                        "lane_event": float(loss.detach()),
                        "rows": int(logits.numel()),
                        "grad_norm": gn, "seconds": time.time() - t0})
            if step % 25 == 0 or step == 1:
                print(json.dumps({"step": step, "rows": int(logits.numel()),
                                  "lane_event": round(float(loss.detach()), 4),
                                  "grad_norm": round(gn, 3),
                                  "elapsed": round(time.time() - t0, 1)}), flush=True)
    else:
        while step < int(args.steps):
            for batch in loader:
                if step >= int(args.steps):
                    break
                optimizer.zero_grad(set_to_none=True)
                rec = runner.run(batch, unroll_steps=unroll, truncation_steps=trunc)
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                       for p in model.parameters()):
                    raise RuntimeError("non-finite gradient")
                gn = float(torch.nn.utils.clip_grad_norm_(params, settings.grad_clip))
                optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise RuntimeError("non-finite parameter")
                step += 1
                log.append({"step": step, "loss": float(rec["loss"]),
                            "lane_event": float(rec.get("lane_event", 0.0)),
                            "grad_norm": gn, "seconds": time.time() - t0})
                if step % 25 == 0 or step == 1:
                    print(json.dumps({"step": step, "loss": round(float(rec["loss"]), 4),
                                      "lane_event": round(float(rec.get("lane_event", 0.0)), 4),
                                      "grad_norm": round(gn, 3),
                                      "elapsed": round(time.time() - t0, 1)}), flush=True)

    save_cfg = dataclasses.replace(
        train_cfg,
        model=dataclasses.replace(train_cfg.model,
                                  courtesy_inputs=bool(args.courtesy_inputs)),
        loss=dataclasses.replace(train_cfg.loss,
                                 courtesy_weight=float(args.courtesy_weight)))
    tag = str(getattr(args, "tag", "on" if args.courtesy_inputs else "off"))
    out_ckpt = Path(args.out) / f"wm_s_courtesy_{tag}.pt"
    save_checkpoint(out_ckpt, model, save_cfg,
                    extra={"lane_learnable_ft": {"steps": step, "lr": float(settings.lr),
                                                 "unroll_steps": unroll, "horizon_s":
                                                 float(args.horizon_s),
                                                 "n_native": len(nat_eps),
                                                 "n_genrand": len(gr_eps)}})
    (Path(args.out) / "physics.json").write_text(
        json.dumps(dataclasses.asdict(model.physics), indent=1), encoding="utf-8")
    (Path(args.out) / "finetune_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(json.dumps({"event": "done", "checkpoint": str(out_ckpt), "steps": step,
                      "first_loss": log[0]["loss"] if log else None,
                      "last_loss": log[-1]["loss"] if log else None,
                      "first_lane_event": log[0]["lane_event"] if log else None,
                      "last_lane_event": log[-1]["lane_event"] if log else None}),
          flush=True)
    return 0


# --------------------------------------------------------------------------- rollout
def _set_physics(model, **changes):
    from wm_eval_common import _physics_holders

    base = getattr(model, "_ft_base_physics", None)
    if base is None:
        base = model.physics
        model._ft_base_physics = base
    new = dataclasses.replace(base, **changes)
    for owner, name in _physics_holders(model):
        setattr(owner, name, new)
    model.physics = new
    return new


def _first_off(lane, msk, k):
    idx = np.flatnonzero(msk[:, k] & (lane[:, k] != 2))
    return int(idx[0]) if idx.size else -1


def _events(lane, msk, sel):
    """Count overtake (1->0), return (0->1) and merge (2->1) events of ``sel`` vehicles."""
    counts = {"overtake": 0, "return": 0, "merge": 0}
    for k in np.flatnonzero(sel):
        for t in range(lane.shape[0] - 1):
            if not (msk[t, k] and msk[t + 1, k]):
                continue
            a, b = int(lane[t, k]), int(lane[t + 1, k])
            if a == 1 and b == 0:
                counts["overtake"] += 1
            elif a == 0 and b == 1:
                counts["return"] += 1
            elif a == 2 and b == 1:
                counts["merge"] += 1
    return counts


def _side_metrics(pos, lane, msk, is_cav, end=ZONE_END_M):
    """Departure / stuck / event metrics of one side (WM or SUMO)."""
    ramp = np.flatnonzero(~np.asarray(is_cav, bool) & (lane[0] == 2))
    dep_d, stuck = [], 0
    for k in ramp:
        j = _first_off(lane, msk, k)
        if j < 1:
            stuck += 1
            continue
        dep_d.append(float(end - pos[j - 1, k]))
    hdv = ~np.asarray(is_cav, bool)
    ev = _events(lane, msk, hdv)
    dd = np.asarray(dep_d, float)
    return {"ramp_cars": int(ramp.size), "merges": int(dd.size), "stuck": int(stuck),
            "median_dep_d_m": float(np.median(dd)) if dd.size else float("nan"),
            "share_last5m": float(np.mean(dd < 5.0)) if dd.size else float("nan"),
            **ev}


def _work(payload):
    idx, rec, arm_name, p, cap, horizon_s = payload
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles

    out = {"idx": idx, "arm": arm_name, "error": None}
    try:
        n_steps = int(round(horizon_s / DT))
        t0, ids, kin, is_cav = t0_vehicles(rec)
        teacher, present_seq, _ = recorded_teacher(rec, t0, ids, kin, n_steps, rec["arm"],
                                                   p.backend.tools)
        if teacher is None:
            out["error"] = "no arm"
            return out
        p.predict_states(rec, {**rec.get("cav_controllers", {}), "arm": rec["arm"]}, horizon_s, 0)
        r = cap["last"]
        n = len(ids)
        tr = teacher[0].numpy()
        rp = present_seq[0].numpy().astype(bool)
        pos = np.concatenate([tr[:1, :, 0], r["positions"][0, :, :n].numpy()], 0)
        lan = np.concatenate([tr[:1, :, 1], r["lanes"][0, :, :n].numpy()], 0).astype(int)
        msk = np.concatenate([rp[:1], r["masks"][0, :, :n].numpy().astype(bool)], 0)
        hdv = ~np.asarray(is_cav, bool)
        fin = bool(np.isfinite(pos).all() and np.isfinite(lan).all())
        out["finite"] = fin
        out["wm"] = _side_metrics(pos, lan, msk, is_cav)
        tr_lane = tr[:, :, 1].astype(int)
        out["sumo"] = _side_metrics(tr[:, :, 0], tr_lane, rp, is_cav)
        for ts in (10, 30, 60, 120):
            i = int(ts / DT)
            both = msk[i] & rp[i] & hdv
            d = np.abs(pos[i] - tr[i, :, 0])
            out[f"ds{ts}"] = float(np.median(d[both])) if both.any() else float("nan")
        out["blowup"] = bool((not fin) or (np.isfinite(out["ds60"]) and out["ds60"] > 50.0
                                           and np.isfinite(out["ds120"]) and out["ds120"] > 100.0))
        return out
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def _pooled(results):
    ok = [r for r in results if not r.get("error")]
    starts = len(ok)
    merges = sum(r["wm"]["merges"] for r in ok)
    med = [r["wm"]["median_dep_d_m"] for r in ok if np.isfinite(r["wm"]["median_dep_d_m"])]
    share = [r["wm"]["share_last5m"] for r in ok if np.isfinite(r["wm"]["share_last5m"])]
    agg = {"starts": starts, "errors": len(results) - starts}
    for side in ("wm", "sumo"):
        agg[f"{side}_merges"] = sum(r[side]["merges"] for r in ok)
        agg[f"{side}_ramp_cars"] = sum(r[side]["ramp_cars"] for r in ok)
        agg[f"{side}_stuck"] = sum(r[side]["stuck"] for r in ok)
        dd = [r[side]["median_dep_d_m"] for r in ok if np.isfinite(r[side]["median_dep_d_m"])]
        agg[f"{side}_median_dep_d_m"] = float(np.median(dd)) if dd else float("nan")
        ss = [r[side]["share_last5m"] for r in ok if np.isfinite(r[side]["share_last5m"])]
        agg[f"{side}_share_last5m"] = float(np.mean(ss)) if ss else float("nan")
        for e in ("overtake", "return", "merge"):
            agg[f"{side}_{e}"] = sum(r[side][e] for r in ok)
    agg["wm_merges_per_run"] = merges / starts if starts else float("nan")
    agg["sumo_merges_per_run"] = agg["sumo_merges"] / starts if starts else float("nan")
    agg["wm_median_dep_d_m"] = float(np.median(med)) if med else float("nan")
    agg["wm_share_last5m"] = float(np.mean(share)) if share else float("nan")
    for ts in (10, 30, 60, 120):
        v = [r[f"ds{ts}"] for r in ok if np.isfinite(r[f"ds{ts}"])]
        agg[f"ds{ts}"] = float(np.median(v)) if v else float("nan")
    agg["blowups"] = sum(1 for r in ok if r["blowup"])
    agg["nonfinite"] = sum(1 for r in ok if not r.get("finite", True))
    return agg


E: dict = {}


def _init(arm_specs):
    import torch
    from wm_eval_common import load_predictor, guard as wguard

    torch.set_num_threads(1)
    for name, path, ll in arm_specs:
        p, cap = load_predictor(wguard(path), "AS_IS", hdv_exits="sumo", entries="off")
        if ll:
            _set_physics(p.model, lane_learnable=True)
        E[name] = (p, cap)



def _yield_event_mask(states, present, physics):
    from wms.train.courtesy import courtesy_follower_mask
    return courtesy_follower_mask(states, present, physics, behind_m=40.0, dend_m=40.0)


def _yield_work(payload):
    """Teacher-forced one-step follower acceleration on yield events of one start."""
    import torch
    from wm_eval_common import _physics_holders  # noqa: F401  (imported for parity)
    from wms.eval.rollout_inputs import (action_timeline_inputs, recorded_teacher,
                                         t0_vehicles)

    idx, rec, arm_name, p, cap, horizon_s = payload
    out = {"idx": idx, "arm": arm_name, "error": None}
    try:
        n_steps = int(round(horizon_s / DT))
        t0_s, ids, kin, is_cav = t0_vehicles(rec)
        teacher, present_seq, junction_seq = recorded_teacher(
            rec, t0_s, ids, kin, n_steps, rec["arm"], p.backend.tools)
        if teacher is None:
            out["error"] = "no arm"
            return out
        model = p.model
        driver_history, driver_history_mask = p._history.build(rec, ids)
        action_timeline = action_timeline_inputs(rec, {**rec.get("cav_controllers", {}),
                                                       "arm": rec["arm"]}, ids, 0,
                                                 p.backend.tools)
        initial = {
            "s_m": torch.as_tensor(kin[None, :, 0]), "v_mps": torch.as_tensor(kin[None, :, 4]),
            "a_mps2": torch.zeros((1, len(ids))),
            "lane": torch.as_tensor(kin[None, :, 1:4].argmax(-1)),
            "length_m": torch.as_tensor(kin[None, :, 6]),
            "present": torch.ones((1, len(ids)), dtype=torch.bool),
            "is_cav": torch.as_tensor(is_cav[None]),
        }
        tr = teacher[0].numpy()
        obs = torch.zeros(1, n_steps + 1, len(ids), 7)
        obs[0, :, :, 0] = teacher[0, :, :, 0]
        lane = teacher[0, :, :, 1].long().clamp(0, 2)
        obs[0, :, :, 1:4] = torch.nn.functional.one_hot(lane, 3).to(obs.dtype)
        obs[0, :, :, 4] = teacher[0, :, :, 2]
        obs[0, :, :, 5] = teacher[0, :, :, 3]
        obs[0, :, :, 6] = torch.as_tensor(kin[None, :, 6])
        obs_mask = (present_seq if present_seq is not None
                    else torch.ones(1, n_steps + 1, len(ids), dtype=torch.bool))
        with torch.no_grad():
            roll = model.rollout(
                initial, n_steps=n_steps, k0=0, cav_teacher=teacher,
                present_seq=present_seq, junction_seq=junction_seq,
                driver_history=driver_history, driver_history_mask=driver_history_mask,
                action_timeline=action_timeline, lane_seed=0,
                observe_states=obs, observe_mask=obs_mask)
        # roll["accel"] is the model's action at t; the state at t is recorded.
        pred = roll["accel"][0].numpy()
        states_t = np.zeros((n_steps, len(ids), 7), np.float64)
        states_t[..., 0] = tr[:-1, :, 0]
        states_t[..., 1:4] = np.eye(3)[tr[:-1, :, 1].astype(int).clip(0, 2)]
        states_t[..., 4] = tr[:-1, :, 2]
        states_t[..., 5] = tr[:-1, :, 3]
        states_t[..., 6] = kin[None, :, 6]
        rec_a = states_t[..., 5]
        mask_t = np.asarray(obs_mask[0, :-1].numpy(), bool)
        ev = _yield_event_mask(torch.as_tensor(states_t[None]), torch.as_tensor(mask_t[None]),
                               model.physics)[0].numpy()
        m = ev & mask_t & np.isfinite(pred) & np.isfinite(rec_a)
        if not m.any():
            out.update({"events": 0, "mae": float("nan"), "bias": float("nan"),
                        "rmse": float("nan")})
            return out
        err = pred[m] - rec_a[m]
        out.update({"events": int(m.sum()),
                    "mae": float(np.mean(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err * err))),
                    "bias": float(np.mean(err)),
                    "rec_brake_share": float(np.mean(rec_a[m] < -0.5)),
                    "pred_brake_share": float(np.mean(pred[m] < -0.5)),
                    "mean_rec_accel": float(np.mean(rec_a[m])),
                    "mean_pred_accel": float(np.mean(pred[m]))})
        return out
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def yielddiag(args) -> int:
    """Teacher-forced follower-accel error on yield events over the 16 TRAIN-val starts."""
    from multiprocessing import Pool
    native = Path(guard(args.native_dir))
    genrand = Path(guard(args.genrand_dir))
    nat_eps = select_episodes(native, int(args.n_native), int(args.offset))
    gr_eps = select_episodes(genrand, int(args.n_genrand), int(args.offset))
    arms = []
    for spec in args.arm:
        name, rest = spec.split("=", 1)
        path, _, opt = rest.partition("|")
        arms.append((name, guard(path), "ll" in [o for o in opt.split(",") if o]))
    records = _build_records(nat_eps, native) + _build_records(gr_eps, genrand)
    _init(arms)
    tasks = [(idx, rec, name, E[name][0], E[name][1], float(args.horizon_s))
             for idx, rec in enumerate(records) for name, _, _ in arms]
    res = []
    if args.workers <= 1:
        for t in tasks:
            res.append(_yield_work(t))
    else:
        payloads = [(t[0], t[1], t[2], t[5]) for t in tasks]
        with Pool(args.workers, initializer=_init, initargs=(arms,)) as pool:
            res.extend(pool.imap_unordered(_yield_mp, payloads, chunksize=1))
    by_arm = {}
    for name, _, _ in arms:
        rows = [r for r in res if r["arm"] == name and not r.get("error")]
        ev = sum(r.get("events", 0) for r in rows)
        by_arm[name] = {
            "starts": len(rows), "events": int(ev),
            "errors": sum(1 for r in res if r["arm"] == name and r.get("error")),
            "mae": float(np.average([r["mae"] for r in rows],
                                    weights=[max(1, r["events"]) for r in rows])) if rows else float("nan"),
            "rmse": float(np.sqrt(np.average([r["rmse"] ** 2 for r in rows],
                                             weights=[max(1, r["events"]) for r in rows]))) if rows else float("nan"),
            "bias": float(np.average([r["bias"] for r in rows],
                                     weights=[max(1, r["events"]) for r in rows])) if rows else float("nan"),
            "rec_brake_share": float(np.mean([r["rec_brake_share"] for r in rows if r.get("events")])) if any(r.get("events") for r in rows) else float("nan"),
            "pred_brake_share": float(np.mean([r["pred_brake_share"] for r in rows if r.get("events")])) if any(r.get("events") for r in rows) else float("nan"),
        }
    doc = {"n_starts": len(records), "horizon_s": float(args.horizon_s),
           "subsets": {"native": [e["episode"] for e in nat_eps],
                       "genrand": [e["episode"] for e in gr_eps]},
           "per_arm": by_arm, "per_start": res}
    Path(args.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps(by_arm, indent=1), flush=True)
    return 0


def _yield_mp(payload):
    idx, rec, arm_name, horizon_s = payload
    p, cap = E[arm_name]
    return _yield_work((idx, rec, arm_name, p, cap, horizon_s))


def evaluate(args) -> int:
    from multiprocessing import Pool

    native = Path(guard(args.native_dir))
    genrand = Path(guard(args.genrand_dir))
    nat_eps = select_episodes(native, int(args.n_native), int(args.offset))
    gr_eps = select_episodes(genrand, int(args.n_genrand), int(args.offset))
    arms = []
    for spec in args.arm:
        name, rest = spec.split("=", 1)
        path, _, opt = rest.partition("|")
        arms.append((name, guard(path), "ll" in [o for o in opt.split(",") if o]))
    records = _build_records(nat_eps, native) + _build_records(gr_eps, genrand)
    print(json.dumps({"event": "eval", "arms": [a[0] for a in arms],
                      "starts": len(records)}), flush=True)
    _init(arms)
    tasks = []
    for idx, rec in enumerate(records):
        for name, path, ll in arms:
            p, cap = E[name]
            tasks.append((idx, rec, name, p, cap, float(args.horizon_s)))
    res = []
    t0 = time.time()
    if args.workers <= 1:
        for t in tasks:
            res.append(_work(t))
    else:
        # Windows spawns fresh processes: reload the arms in each worker (predictors are not
        # picklable), then stream the light (idx, rec, arm, horizon) payloads.
        payloads = [(t[0], t[1], t[2], t[5]) for t in tasks]
        with Pool(args.workers, initializer=_init, initargs=(arms,)) as pool:
            for j, r in enumerate(pool.imap_unordered(_work_mp, payloads, chunksize=1)):
                res.append(r)
                if (j + 1) % 16 == 0:
                    print(f"  {j + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    by_arm = {name: _pooled([r for r in res if r["arm"] == name]) for name, _, _ in arms}
    out = {"arms": [a[0] for a in arms], "n_starts": len(records),
           "offset": int(args.offset), "horizon_s": float(args.horizon_s),
           "subsets": {"native": [e["episode"] for e in nat_eps],
                       "genrand": [e["episode"] for e in gr_eps]},
           "per_arm": by_arm}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["per_arm"], indent=1), flush=True)
    return 0


def _build_records(entries, root):
    from c4_episode_states import episode_record
    from merge_calib import episode_dict

    out = []
    for e in entries:
        ep = episode_dict(root / e["dir"])
        rec = episode_record(ep, t0_s=420.0, arm=str(e["arm"]), world=str(e["world"]))
        rec["source"] = str(root.name)
        rec["cav_controllers"] = {}
        out.append(rec)
    return out


# Multiprocessing worker: predictors cannot be pickled; rebuild from the global of a fork.
def _work_mp(payload):
    idx, rec, arm_name, horizon_s = payload
    p, cap = E[arm_name]
    return _work((idx, rec, arm_name, p, cap, horizon_s))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("finetune")
    f.add_argument("--ckpt", required=True)
    f.add_argument("--native-dir", required=True)
    f.add_argument("--genrand-dir", required=True)
    f.add_argument("--n-native", type=int, default=8)
    f.add_argument("--n-genrand", type=int, default=8)
    f.add_argument("--steps", type=int, default=300)
    f.add_argument("--lr", type=float, default=1e-4)
    f.add_argument("--batch", type=int, default=4)
    f.add_argument("--horizon-s", type=float, default=30.0)
    f.add_argument("--unroll-steps", type=int, default=None)
    f.add_argument("--trunc", type=int, default=None)
    f.add_argument("--stride", type=float, default=10.0)
    f.add_argument("--threads", type=int, default=8)
    f.add_argument("--train-lane-only", action="store_true",
                   help="freeze every module except lane_v5 (isolate lane-head learnability)")
    f.add_argument("--courtesy-inputs", action="store_true")
    f.add_argument("--courtesy-weight", type=float, default=1.0)
    f.add_argument("--courtesy-lr-mult", type=float, default=10.0)
    f.add_argument("--desaturate", type=float, default=1.0,
                   help="rescale the residual accel output layer (v5m is saturated; "
                        "use 0.02 for the courtesy fine-tune, same for both arms)")
    f.add_argument("--event-windows", action="store_true",
                   help="train on windows that contain a courtesy yield event (same for both arms)")
    f.add_argument("--tag", default=None)
    f.add_argument("--seed", type=int, default=101)
    f.add_argument("--out", required=True)

    e = sub.add_parser("eval")
    e.add_argument("--arm", action="append", required=True,
                   help="NAME=PATH[|ll] (ll = lane_learnable at inference)")
    e.add_argument("--native-dir", required=True)
    e.add_argument("--genrand-dir", required=True)
    e.add_argument("--n-native", type=int, default=8)
    e.add_argument("--n-genrand", type=int, default=8)
    e.add_argument("--offset", type=int, default=0)
    e.add_argument("--horizon-s", type=float, default=120.0)
    e.add_argument("--workers", type=int, default=1)
    e.add_argument("--out", required=True)

    y = sub.add_parser("yielddiag")
    y.add_argument("--arm", action="append", required=True)
    y.add_argument("--native-dir", required=True)
    y.add_argument("--genrand-dir", required=True)
    y.add_argument("--n-native", type=int, default=8)
    y.add_argument("--n-genrand", type=int, default=8)
    y.add_argument("--offset", type=int, default=0)
    y.add_argument("--horizon-s", type=float, default=120.0)
    y.add_argument("--workers", type=int, default=1)
    y.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "finetune":
        return finetune(a)
    if a.cmd == "yielddiag":
        return yielddiag(a)
    return evaluate(a)


if __name__ == "__main__":
    raise SystemExit(main())
