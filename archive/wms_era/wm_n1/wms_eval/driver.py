"""Offline per-driver recovery scoring for the WM-S2 VAL/TEST packs.

The trainer is TRAIN-only, so the held-out VAL/TEST worlds are scored offline.
This module has two CPU-only steps:

1. ``materialise`` reads a hetpack corpus pack (``manifest.json`` + ``INDEX.json``
   + the memmapped ``veh_kin``/``veh_mask``/``target_mask`` arrays) and writes one
   state record per manifest item -- the item's snapshot tick plus its pre-t0
   observed history.  Vehicle ids are not stored in the pack, so slots are named
   ``slot<k>``; the dense truth / bundle supplies the slot -> driver truth map.
2. ``driver`` runs the trained driver encoder on every record and scores the
   predicted parameters against the driver bundle (``[n_items,k_max,11]``) or the
   dense per-run truth (``wms.dense_truth``): per-parameter MAE/RMSE/bias, rank
   correlation, and per anchor-class breakdown.

Memory rule: packs are opened ``mmap_mode='r'`` and one record is processed at a
time; the pack is never fully loaded.  This module touches no CAL/HOLDOUT/sealed
data and never modifies the pack.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_HISTORY = 10          # 5 s of pre-t0 history at 0.5 s
DT_S = 0.5


class DriverScoreError(RuntimeError):
    """The pack, sidecar or checkpoint cannot be scored."""


# --------------------------------------------------------------- pack read ----
def _read_pack_arrays(pack_dir: Path):
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    index = json.loads((pack_dir / "INDEX.json").read_text(encoding="utf-8"))
    arrays = manifest.get("arrays") or {}
    for name in ("veh_kin", "veh_mask", "target_mask"):
        if name not in arrays:
            raise DriverScoreError(f"pack {pack_dir} lacks the '{name}' array")
    kin = np.load(pack_dir / arrays["veh_kin"]["file"], mmap_mode="r",
                  allow_pickle=False)
    mask = np.load(pack_dir / arrays["veh_mask"]["file"], mmap_mode="r",
                   allow_pickle=False)
    target = np.load(pack_dir / arrays["target_mask"]["file"], mmap_mode="r",
                     allow_pickle=False)
    return manifest, index, kin, mask, target


def _state_records(manifest, index, kin, mask, target, *, max_items,
                   history_steps, history_s, history_stride_s):
    """Yield bounded, native-tick history records one at a time."""
    runs = index.get("runs") or []
    tick_start = {}
    for rec in runs:
        tick_start[(rec.get("world"), rec.get("arm"))] = int(rec["tick_start"])
    items = manifest.get("items") or []
    if max_items is not None:
        items = items[: int(max_items)]
    for position, item in enumerate(items):
        tick = int(item["tick"])
        row = np.asarray(kin[tick], dtype=np.float32)
        present = np.asarray(mask[tick]).astype(bool)
        k = row.shape[0]
        lo = max(0, tick - history_steps + 1)
        start = int(tick_start.get((item.get("world"), item.get("arm")), 0))
        if history_s is not None:
            lo = max(start, lo)
        hist_kin = np.asarray(kin[lo:tick + 1], dtype=np.float32)
        hist_mask = np.asarray(mask[lo:tick + 1]).astype(bool)
        hist_target = np.asarray(target[lo:tick], dtype=bool) if tick > lo else \
            np.zeros((0, k), dtype=bool)
        vehicles = []
        for slot in range(k):
            lane = int(np.argmax(row[slot, 1:4])) if row[slot, 1:4].sum() > 0 else -1
            vehicles.append({"id": f"slot{slot}", "s": float(row[slot, 0]),
                             "lane": lane, "v": float(row[slot, 4]),
                             "present": bool(present[slot]), "is_cav": False})
        record = {
            "world": item.get("world"), "arm": item.get("arm"),
            "t0_s": float(item.get("t0_s", 0.0)), "tick": tick,
            "tick_start": int(tick_start.get((item.get("world"),
                                              item.get("arm")), 0)),
            "item_position": int(position),
            "split": item.get("split"), "policy": item.get("policy"),
            "state": {"time_s": float(item.get("t0_s", 0.0)), "vehicles": vehicles},
            "veh_kin": row.tolist(), "veh_mask": present.tolist(),
            "history": {"kin": hist_kin.tolist(), "mask": hist_mask.tolist(),
                        "target_mask": hist_target.tolist()},
        }
        if history_s is not None:
            record["history"].update(
                history_s=float(history_s), history_stride_s=float(history_stride_s),
                dt_s=DT_S, available_span_s=(len(hist_kin) - 1) * DT_S,
                run_start_limited=(tick - history_steps + 1 < start))
        yield record


def materialise_pack_states(pack_dir, out_path, *, max_items=None,
                            history_steps=DEFAULT_HISTORY,
                            history_s=None, history_stride_s=1.0) -> dict:
    """Write state records; longer histories are streamed without corpus buffering."""
    history_steps = int(history_steps)
    if history_s is not None:
        from wms.history import history_spec
        history_steps, _, _ = history_spec(history_s, history_stride_s, DT_S)
    pack_dir = Path(pack_dir)
    manifest, index, kin, mask, target = _read_pack_arrays(pack_dir)
    records = _state_records(manifest, index, kin, mask, target, max_items=max_items,
                              history_steps=history_steps, history_s=history_s,
                              history_stride_s=history_stride_s)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if history_s is not None:
        n = len(manifest.get("items") or [])
        if max_items is not None:
            n = len((manifest.get("items") or [])[:int(max_items)])
        header = {"schema": "wms_driver_states/1", "pack": str(pack_dir),
                  "history_steps": history_steps, "n_states": n}
        with out_path.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(header)[:-1] + ', "states": [')
            for i, record in enumerate(records):
                if i:
                    stream.write(", ")
                json.dump(record, stream)
            stream.write("]}")
        return {"n_states": n, "out": str(out_path)}
    records = list(records)
    payload = {"schema": "wms_driver_states/1", "pack": str(pack_dir),
               "history_steps": history_steps, "n_states": len(records),
               "states": records}
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return {"n_states": len(records), "out": str(out_path)}


# ------------------------------------------------------------- encoder pass ---
def history_features_from_kin(kin_hist, mask_hist, *, history_len=None):
    """``[T,K,7]`` + ``[T,K]`` -> per-slot ``[K,H,F]`` history + ``[K,H]`` mask.

    The gap feature follows the trainer's ``_prior_history`` (same-lane, positive
    leader gap, clamped).  When ``history_len`` is given the per-slot history is
    front-padded with zeros (mask False) or right-aligned/truncated to match the
    model's fixed window, exactly as the trainer does.
    """
    import torch
    from wms.model import history_step_features

    kin = torch.as_tensor(np.asarray(kin_hist, dtype=np.float32))
    present = torch.as_tensor(np.asarray(mask_hist).astype(bool))
    if kin.ndim != 3 or kin.shape[-1] != 7:
        raise DriverScoreError("history kin must be [T,K,7]")
    t, k = kin.shape[:2]
    lane = kin[..., 1:4].argmax(-1)
    gap = torch.full((t, k), 250.0)
    for ti in range(t):
        s = kin[ti, :, 0]
        ds = s.unsqueeze(0) - s.unsqueeze(1)
        same = (lane[ti].unsqueeze(0) == lane[ti].unsqueeze(1))
        ok = present[ti].unsqueeze(0) & same & (ds > 0)
        g = torch.where(ok, ds - kin[ti, :, 6].unsqueeze(0),
                        torch.full_like(ds, 250.0))
        gap[ti] = g.min(-1).values.clamp(0.3, 250.0)
    feat = history_step_features(kin, gap)              # [T,K,F]
    out = feat.transpose(0, 1).numpy()                  # [K,T,F]
    out_mask = present.transpose(0, 1).numpy()          # [K,T]
    if history_len is not None:
        want = int(history_len)
        if want < 1:
            raise DriverScoreError("history_len must be positive")
        if out.shape[1] > want:                          # keep the most recent
            out = out[:, -want:]
            out_mask = out_mask[:, -want:]
        elif out.shape[1] < want:                        # front-pad zeros
            pad = want - out.shape[1]
            out = np.concatenate((np.zeros((out.shape[0], pad, out.shape[2]),
                                           dtype=out.dtype), out), axis=1)
            out_mask = np.concatenate(
                (np.zeros((out_mask.shape[0], pad), dtype=bool), out_mask), axis=1)
    return out, out_mask


def predict_records(model, records):
    """Run the driver encoder on each record; returns ``[(values, uncertainty)]``."""
    import torch

    want = int(getattr(model, "history_features", 6))
    hist_len = int(getattr(model, "history_len", DEFAULT_HISTORY))
    out = []
    for rec in records:
        long_history = getattr(model, "history_s", None) is not None
        if long_history:
            from .wms_plugin import WmsCheckpointModel
            holder = WmsCheckpointModel.__new__(WmsCheckpointModel)
            holder.model, holder.config, holder._torch = model, {}, torch
            feat, hmask = holder._driver_history(rec, [
                f"slot{k}" for k in range(len(rec["veh_kin"]))])
            hist, hmsk = feat, hmask
        else:
            feat, hmask = history_features_from_kin(rec["history"]["kin"],
                                                    rec["history"]["mask"],
                                                    history_len=hist_len)
            hist, hmsk = torch.as_tensor(feat[None]), torch.as_tensor(hmask[None])
        if not long_history and feat.shape[-1] != want:
            raise DriverScoreError(
                f"history width {feat.shape[-1]} != model {want}")
        kin = torch.as_tensor(np.asarray(rec["veh_kin"], dtype=np.float32))[None]
        present = torch.as_tensor(np.asarray(rec["veh_mask"]).astype(bool))[None]
        initial = {"veh_kin": kin, "present": present}
        with torch.no_grad():
            res = model(initial, k_steps=0, history=None if long_history else hist,
                        history_mask=None if long_history else hmsk,
                        driver_history=hist, driver_history_mask=hmsk)
        if "driver_parameters" not in res:
            raise DriverScoreError(
                "checkpoint is not a driver_params model (no driver_parameters)")
        out.append((res["driver_parameters"][0].numpy(),
                    res["driver_uncertainty"][0].numpy()))
    return out


# ------------------------------------------------------------- truth source ---
def dense_truth_lookup(dense):
    def lookup(record, slot):
        # The dense truth is run-local (tick relative to the run's t_origin);
        # the pack materialiser stores the pack-global row index.
        local = int(record["tick"]) - int(record.get("tick_start", 0))
        got = dense.lookup(record["world"], record["arm"], local, int(slot))
        if got is None:
            return None
        _vid, params, cls = got
        if not np.all(np.isfinite(params)):
            return None
        return np.asarray(params, dtype=np.float64), int(cls), True
    return lookup


def bundle_truth_lookup(bundle):
    cache = {}

    def lookup(record, slot):
        position = int(record["item_position"])
        item = cache.get(position)
        if item is None:
            item = bundle.item_index_of(position)
            cache[position] = item
        params = np.asarray(bundle.params[item, slot], dtype=np.float64)
        if not np.all(np.isfinite(params)):
            return None
        return params, int(bundle.classes[item, slot]), True
    return lookup


# ------------------------------------------------------------------ scoring ---
def score_predictions(records, predictions, truth_lookup, *, class_names):
    """Aggregate the per-parameter / rank / per-class recovery report."""
    from wms.driver_sidecar import driver_recovery

    values, uncertainty, truth, classes, present = [], [], [], [], []
    for rec, (vals, unc) in zip(records, predictions):
        k = np.asarray(vals).shape[0]
        tv = np.full((k, 11), np.nan)
        tc = np.full(k, -1, dtype=np.int64)
        tp = np.zeros(k, dtype=bool)
        for slot in range(k):
            got = truth_lookup(rec, slot)
            if got is None:
                continue
            p, c, ok = got
            if not ok:
                continue
            tv[slot], tc[slot], tp[slot] = p, c, True
        values.append(np.asarray(vals))
        uncertainty.append(np.asarray(unc))
        truth.append(tv)
        classes.append(tc)
        present.append(tp)
    report = driver_recovery(np.asarray(values), np.asarray(uncertainty),
                             np.asarray(truth), np.asarray(classes),
                             present=np.asarray(present))
    report["n_records"] = len(records)
    return report


# ------------------------------------------------------------------- CLI ------
def _cmd_materialise(args) -> int:
    summary = materialise_pack_states(args.pack, args.out,
                                      max_items=args.max_items,
                                      history_steps=args.history_steps,
                                      history_s=args.history_s,
                                      history_stride_s=args.history_stride_s)
    print(json.dumps(summary, indent=2))
    return 0


def _load_records(path: Path) -> list:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("states")
    if not isinstance(records, list):
        raise DriverScoreError(f"{path}: no 'states' list")
    return records


def _predictions_payload(ckpt, records, predictions) -> dict:
    """Per-vehicle recovered parameters + uncertainty (no truth needed).

    The old-data chain has no per-driver truth sidecar, so the encoder
    parameters are learned only through the trajectory loss.  This dump keeps
    the recovered ``[K, 9]`` values/uncertainty per vehicle (named by the
    materialiser's ``slot<k>`` id) so a later offline check can compare them
    against the three known vTypes.  Nothing here is supervision.
    """
    from wms.driver_encoder import PARAMETER_NAMES

    out_records = []
    for rec, (values, uncertainty) in zip(records, predictions):
        values = np.asarray(values, dtype=np.float64)
        uncertainty = np.asarray(uncertainty, dtype=np.float64)
        vehicles = (rec.get("state") or {}).get("vehicles") or []
        slots = []
        for k in range(values.shape[0]):
            slot_id = (str(vehicles[k]["id"]) if k < len(vehicles)
                       and vehicles[k].get("id") is not None else f"slot{k}")
            slots.append({"id": slot_id,
                          "values": [float(x) for x in values[k]],
                          "uncertainty": [float(x) for x in uncertainty[k]]})
        out_records.append({
            "world": rec.get("world"), "arm": rec.get("arm"),
            "t0_s": rec.get("t0_s"), "tick": rec.get("tick"),
            "item_position": rec.get("item_position"),
            "policy": rec.get("policy"), "slots": slots})
    return {"schema": "wms_driver_predictions/1", "checkpoint": str(ckpt),
            "parameter_names": list(PARAMETER_NAMES),
            "n_records": len(out_records), "records": out_records}


def _cmd_driver(args) -> int:
    from wms.driver_sidecar import DRIVER_CLASSES, load_driver_sidecar
    from wms_eval.wms_plugin import WmsCheckpointModel

    records = _load_records(Path(args.states))
    sidecar = load_driver_sidecar(args.sidecar) if args.sidecar else None
    physics = None
    if args.physics:
        physics = json.loads(Path(args.physics).read_text(encoding="utf-8"))
        if isinstance(physics, dict) and isinstance(physics.get("physics"), dict):
            physics = physics["physics"]
    holder = WmsCheckpointModel(args.ckpt, physics=physics)
    model = holder.model or holder._load()
    predictions = predict_records(model, records)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "DRIVER_PRED.json"
    pred_path.write_text(json.dumps(_predictions_payload(args.ckpt, records,
                                                         predictions),
                                    indent=2, sort_keys=True), encoding="utf-8")
    if sidecar is None:
        print(json.dumps({"mode": "predictions-only", "n_records": len(records),
                          "out": str(pred_path)}))
        return 0
    if hasattr(sidecar, "lookup") and not hasattr(sidecar, "item_sidecar"):
        truth = dense_truth_lookup(sidecar)
    elif hasattr(sidecar, "item_sidecar"):
        truth = bundle_truth_lookup(sidecar)
    else:
        raise DriverScoreError(
            "--sidecar must be a hetpack bundle or a wms.dense_truth directory")
    report = score_predictions(records, predictions, truth,
                               class_names=DRIVER_CLASSES)
    (out_dir / "DRIVER_RECOVERY.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"n_scored": report["n_scored"],
                      "out": str(out_dir / "DRIVER_RECOVERY.json"),
                      "predictions": str(pred_path)}))
    return 0


def add_parsers(sub) -> None:
    m = sub.add_parser("materialise",
                       help="write state records from a hetpack corpus pack")
    m.add_argument("--pack", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--max-items", type=int, default=None)
    m.add_argument("--history-steps", type=int, default=DEFAULT_HISTORY)
    m.add_argument("--history-s", type=float, default=None,
                   help="opt-in native pre-t0 duration; overrides --history-steps")
    m.add_argument("--history-stride-s", type=float, default=1.0)
    m.set_defaults(func=_cmd_materialise)
    d = sub.add_parser("driver",
                       help="score per-driver recovery on materialised states")
    d.add_argument("--states", required=True)
    d.add_argument("--ckpt", required=True)
    d.add_argument("--sidecar", default=None,
                   help="optional driver bundle dir or wms.dense_truth dir; when "
                        "absent, only the per-vehicle DRIVER_PRED.json dump is "
                        "written (old data has no per-driver truth sidecar)")
    d.add_argument("--out", required=True)
    d.add_argument("--physics", default=None)
    d.set_defaults(func=_cmd_driver)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="wms_eval driver-tools")
    sub = parser.add_subparsers(dest="command", required=True)
    add_parsers(sub)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (DriverScoreError, OSError, ValueError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 2
