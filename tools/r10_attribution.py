"""R10 component attribution of the post-cut-in follower response (CPU, no SUMO, no training).

For recorded TRAIN windows (local r9_train copy; the 16 local episodes carry the manifest
role ``val`` of the original TRAIN split, i.e. they were not fit by round 8), the follower of
every recorded ramp->lane-1 insertion is replayed **teacher-forced** (recorded state at each
step, causal histories, freshly encoded driver parameters) with the same replay the cut-in
loss uses (``one_step_predicted_accel``).  At matched recorded state the predicted
acceleration is decomposed into

* ``phys``      IDM core with the encoded driver parameters (``physics_accel_mps2``),
* ``attn``      gated, bounded attention-residual head (temporal emission zeroed),
* ``temporal``  gated temporal emission (full residual minus attention-only residual),
* ``yield``     SECUREGAP follower-yield head (round-8 ``secure_gap`` model only),
* ``final``     the model acceleration after clamps,

and compared with the recorded acceleration of the next row.  Also reports the IDM-only and
residual-removed counterfactuals.  No parameter is trained; nothing here selects a threshold.

    python tools/r10_attribution.py --ckpt <wm_s_stage300.pt> --roots <root> [<root> ...] \
        --out tmp_r10/attribution.json [--stride-s 10] [--horizon-s 60] [--max-windows N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wms.cli.assemble import build_model, build_physics            # noqa: E402
from wms.config.load import load_train_config                      # noqa: E402
from wms.data.c4_episodes import build_c4_datasets                  # noqa: E402
from wms.data.collate import collate                               # noqa: E402
from wms.train.inputs import build_targets, prepare_inputs         # noqa: E402
from wms.train.losses.cutin_reaction import (CutinReactionConfig,   # noqa: E402
                                             insertion_plan, one_step_predicted_accel)
from wms.train.settings import TrainSettings                       # noqa: E402

KEYS = ("physics_accel_mps2", "accel_mps2", "raw_residual_mps2", "accel_delta_mps2",
        "gate", "temporal_residual_mps2")


def load_model(config: str, ckpt: str, overrides=()):
    path = ROOT / "configs" / config
    changes = ["runtime.device=cpu", "data.entries_v2=false", f"model.init_from={ckpt}", *overrides]
    checkpoint_physics = False
    if path.suffix == ".json":
        # LOOP14 records the accepted run's resolved config, not its original TOML path.
        # Decode the same nested schema and apply the identical scoring-only overrides.
        from wms.config.load import apply_overrides
        from wms.config.schema import from_mapping
        from wms.config.train import TrainConfig
        raw = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_physics = raw.get("checkpoint_physics", False)
        cfg = from_mapping(TrainConfig, apply_overrides(raw.get("config", raw), changes))
    else:
        cfg = load_train_config(path, changes)
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    physics = build_physics(cfg)
    if checkpoint_physics:
        from wms.physics.config import PhysicsConfig
        saved = payload.get("config", {}).get("physics")
        if not isinstance(saved, dict):
            raise ValueError("checkpoint_physics requires an explicit checkpoint physics config")
        physics = PhysicsConfig(**saved)
    model = build_model(cfg, physics)
    own = model.state_dict()
    matched = {k: v for k, v in payload["model"].items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)
    settings = TrainSettings.from_config(cfg)
    model.residual_soft_bound_mps2 = settings.residual_soft_bound_mps2
    model.residual_soft_bound_neg_mps2 = settings.residual_soft_bound_neg_mps2
    model.apply_direct_commands = False
    model.eval().requires_grad_(False)
    report = {"loaded": len(matched), "new": sorted(set(own) - set(payload["model"])),
              "unused": sorted(set(payload["model"]) - set(own))}
    return cfg, settings, model, report


def datasets(cfg, roots, horizon_steps, stride_s):
    out = []
    pivot = cfg.model.merge_decision == "first_passage"
    for root in roots:
        built = build_c4_datasets(
            root, horizon_steps=horizon_steps, history_s=cfg.data.history_s,
            history_stride_s=cfg.data.history_stride_s, entries_v2=False, stride_s=stride_s,
            k_slots=None, role_of=None, junction_fix=cfg.data.junction_fix,
            command_known=True, merge_pivot_history=pivot)
        out.append(built["train"])
    return out


def capture_replay(model, targets):
    """One replay with hooks that record the per-step components of the acceleration.

    ``attn_delta`` / ``temp_delta`` are the attention-head and temporal emissions before the
    gate; the model adds ``gate * (attn_delta + temp_delta)`` (soft-bounded) to the IDM core.
    Direct temporal mode only: the temporal emission is then stateless, so no second replay is
    needed to split it from the attention head.
    """
    captured, store = [], {}
    original = model.forward
    original_res, original_tmp = model._residual_prediction, model._temporal_delta

    def res(*args, **kwargs):
        out = original_res(*args, **kwargs)
        store["attn_delta"] = out["accel_delta_mps2"].detach()
        return out

    def tmp(*args, **kwargs):
        out = original_tmp(*args, **kwargs)
        store["temp_delta"] = out[..., 0].detach()
        return out

    def hooked(state, **kwargs):
        out = original(state, **kwargs)
        row = {k: out[k].detach() for k in KEYS if k in out}
        row.update(store)
        captured.append(row)
        return out

    if getattr(model, "temporal_residual_mode", "direct") != "direct":
        raise ValueError("r10 attribution supports the direct temporal path only")
    model.forward, model._residual_prediction, model._temporal_delta = hooked, res, tmp
    try:
        with torch.no_grad():
            accel = one_step_predicted_accel(model, targets)
    finally:
        model.forward = original
        model._residual_prediction, model._temporal_delta = original_res, original_tmp
        for name in ("forward", "_residual_prediction", "_temporal_delta"):
            model.__dict__.pop(name, None)
    stacked = {k: torch.stack([c[k] for c in captured], 1) for k in captured[0]
               if all(k in c for c in captured)}
    return accel, stacked


def window_rows(model, settings, batch, n_steps, cut_cfg):
    inp = prepare_inputs(batch, model, settings, torch.device("cpu"), n_steps)
    targets = build_targets(inp, batch, settings, float(model.physics.dt_s), torch.device("cpu"),
                            physics=model.physics)
    targets.prepared["cutin_reaction_inputs"] = {
        "history": inp.hist, "history_mask": inp.hist_mask,
        "driver_history": inp.observed_hist, "driver_history_mask": inp.observed_hist_mask}
    plan = insertion_plan(targets, cut_cfg)
    if not plan["n_event_rows"]:
        return None
    accel, full = capture_replay(model, targets)
    phys = full["physics_accel_mps2"]
    final = full["accel_mps2"]
    gate = full["gate"]
    raw_attn = gate * full["attn_delta"]
    raw_temp = gate * full["temp_delta"]
    rec = targets.states[:, 1:n_steps + 1, :, 5]
    return dict(plan=plan, targets=targets, phys=phys, final=final, raw_attn=raw_attn,
                raw_temp=raw_temp, rec=rec)


def bounded(model, raw):
    """The model's own residual bound applied to a (gated) raw correction."""
    from wms.model.residual_bound import bounded_accel_correction
    return bounded_accel_correction(
        raw, soft_bound_pos=model.residual_soft_bound_mps2,
        soft_bound_neg=model.residual_soft_bound_neg_mps2,
        cap_pos=model.residual_cap_mps2, cap_neg=model.residual_cap_neg_mps2)


def stats(values, prefix):
    a = np.asarray(values, float)
    return {prefix + "_p50": float(np.percentile(a, 50)),
            prefix + "_p10": float(np.percentile(a, 10)), prefix + "_mean": float(a.mean())}


def bootstrap(values, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(values, float)
    means = [rng.choice(a, size=len(a), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(events):
    """Decomposition table from per-event arrays (post-insertion rows only)."""
    out = {"n_events": len(events)}
    if not events:
        return out
    cols = {k: [] for k in ("rec", "final", "phys", "idm_attn", "idm_temp")}
    for e in events:
        cols["rec"].append(e["rec"].min())
        cols["final"].append(e["final"].min())
        cols["phys"].append(e["phys"].min())
        cols["idm_attn"].append((e["phys"] + e["corr_attn"]).min())
        cols["idm_temp"].append((e["phys"] + e["corr_temp"]).min())
    for k, v in cols.items():
        out.update(stats(v, k + "_min"))
    rec = np.asarray(cols["rec"])
    for k in ("final", "phys", "idm_attn", "idm_temp"):
        diff = np.asarray(cols[k]) - rec
        out[f"{k}_min_minus_rec_mean"] = float(diff.mean())
        out[f"{k}_min_minus_rec_median"] = float(np.median(diff))
        out[f"{k}_min_minus_rec_ci95"] = list(bootstrap(diff))
        out[f"share_{k}_min_below_rec_by_0p5"] = float(np.mean(diff < -0.5))
    for k in ("final", "phys"):
        out[f"{k}_minus_rec_rowmean"] = float(np.concatenate([e[k] - e["rec"] for e in events]).mean())
    at = [int(np.argmin(e["final"])) for e in events]
    for k in ("corr_attn", "corr_temp", "corr_all", "phys", "rec"):
        out[k + "_rowmean"] = float(np.concatenate([e[k] for e in events]).mean())
        out[k + "_at_final_min_mean"] = float(np.mean([e[k][i] for e, i in zip(events, at)]))
    return out


def event_record(model, res, ev):
    b, f, lo, hi, e = (ev[x] for x in ("batch", "follower", "start", "end", "step"))
    rows = [t for t in range(lo, hi + 1) if t >= e]       # post-insertion prediction rows
    if not rows:
        return None
    sl = slice(rows[0], rows[-1] + 1)
    kin = res["targets"].states[b, e]
    ins = ev["inserter"]
    attn, temp = res["raw_attn"][b, sl, f], res["raw_temp"][b, sl, f]
    return {"rec": res["rec"][b, sl, f].numpy(), "phys": res["phys"][b, sl, f].numpy(),
            "final": res["final"][b, sl, f].numpy(),
            "corr_attn": bounded(model, attn).numpy(), "corr_temp": bounded(model, temp).numpy(),
            "corr_all": bounded(model, attn + temp).numpy(),
            "gap_m": float(kin[ins, 0] - kin[f, 0] - kin[ins, 6]),
            "closing_mps": float(kin[f, 4] - kin[ins, 4]), "follower_v": float(kin[f, 4]),
            "censored": bool(ev["censored"])}


def run(args):
    import pickle
    cut_cfg = CutinReactionConfig(weight=1.0)
    cfg, settings, model, report = load_model(args.config, args.ckpt)
    n_steps = int(round(args.horizon_s / 0.5))
    print("model", args.config, report["loaded"], "tensors; new:", len(report["new"]),
          "mode", cfg.model.merge_decision, flush=True)
    sets = datasets(cfg, args.roots, n_steps, args.stride_s)
    events, control = [], {k: [] for k in ("rec", "final", "phys", "attn", "temp")}
    t_start, windows = time.time(), 0
    shard_path = Path(f"{args.out}.shard{args.shard}.pkl")
    shard_path.parent.mkdir(parents=True, exist_ok=True)

    def save(done):
        # incremental save: a crash (e.g. out of memory) never loses finished windows
        with open(shard_path, "wb") as stream:
            pickle.dump({"events": events, "windows": windows, "done": done,
                         "controls": {k: np.concatenate(v) for k, v in control.items() if v}},
                        stream)

    for ds in sets:
        idx = list(range(len(ds)))
        if args.max_windows:
            rng = np.random.default_rng(0)
            idx = sorted(rng.choice(idx, size=min(args.max_windows, len(idx)), replace=False))
        for i in idx[args.shard::args.n_shards]:
            res = window_rows(model, settings, collate([ds[i]]), n_steps, cut_cfg)
            windows += 1
            if res is not None:
                for ev in res["plan"]["events"]:
                    record = event_record(model, res, ev)
                    if record is not None:
                        events.append(record)
                mask = res["plan"]["control"]
                if mask.any():
                    for k, src in (("rec", res["rec"]), ("final", res["final"]),
                                   ("phys", res["phys"]),
                                   ("attn", bounded(model, res["raw_attn"])),
                                   ("temp", bounded(model, res["raw_temp"]))):
                        control[k].append(src[mask].numpy())
            print(f"window {windows} events {len(events)} {time.time() - t_start:.0f}s", flush=True)
            save(False)
    save(True)
    print("shard done", args.shard, windows, len(events), flush=True)


def finish(args):
    import pickle
    events, controls, windows, not_done = [], {}, 0, []
    for i in range(args.n_shards):
        with open(f"{args.out}.shard{i}.pkl", "rb") as stream:
            part = pickle.load(stream)
        events += part["events"]
        windows += part["windows"]
        if not part.get("done"):
            not_done.append(i)
        for k, v in part["controls"].items():
            controls.setdefault(k, []).append(v)
    ctrl = {k: np.concatenate(v) for k, v in controls.items()}
    ctrl_stats = {}
    if ctrl:
        rec = ctrl["rec"]
        ctrl_stats = {"n_rows": int(len(rec)), "rec_mean": float(rec.mean()),
                      "final_mean": float(ctrl["final"].mean()),
                      "phys_mean": float(ctrl["phys"].mean()),
                      "final_minus_rec_mean": float((ctrl["final"] - rec).mean()),
                      "phys_minus_rec_mean": float((ctrl["phys"] - rec).mean()),
                      "corr_attn_mean": float(ctrl["attn"].mean()),
                      "corr_temp_mean": float(ctrl["temp"].mean()),
                      "rec_p10": float(np.percentile(rec, 10)),
                      "final_p10": float(np.percentile(ctrl["final"], 10)),
                      "phys_p10": float(np.percentile(ctrl["phys"], 10))}
    result = {"config": args.config, "ckpt": str(args.ckpt), "windows": windows,
              "horizon_s": args.horizon_s, "shards_not_done": not_done,
              "events": summarize(events), "controls": ctrl_stats,
              "censored_events": int(sum(e["censored"] for e in events))}
    # matched (gap, closing) tertile cells, edges from the pooled recorded events
    if len(events) >= 30:
        gaps = np.array([e["gap_m"] for e in events])
        clos = np.array([e["closing_mps"] for e in events])
        ge, ce = np.percentile(gaps, [33.3, 66.7]), np.percentile(clos, [33.3, 66.7])
        cells = {}
        for gi in range(3):
            for ci in range(3):
                sel = [e for e, g, c in zip(events, gaps, clos)
                       if np.searchsorted(ge, g) == gi and np.searchsorted(ce, c) == ci]
                if len(sel) >= 5:
                    cells[f"g{gi}c{ci}"] = {
                        "n": len(sel),
                        "rec_min_p50": float(np.median([e["rec"].min() for e in sel])),
                        "final_min_p50": float(np.median([e["final"].min() for e in sel])),
                        "phys_min_p50": float(np.median([e["phys"].min() for e in sel]))}
        result["matched_cells"] = {"gap_edges": ge.tolist(), "closing_edges": ce.tolist(),
                                   "cells": cells}
    Path(args.result or args.out + ".json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--config", default="c4_wm_v10round9.toml")
    p.add_argument("--out", required=True)
    p.add_argument("--stride-s", type=float, default=10.0)
    p.add_argument("--horizon-s", type=float, default=60.0)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--merge", action="store_true", help="merge shard pickles into --out")
    p.add_argument("--result", default=None)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.merge:
        finish(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
