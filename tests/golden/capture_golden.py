"""Capture golden reference outputs from the CURRENT WM-S code (wms/r3-int lineage).

CPU only, no GPU, no SUMO.  Everything is deterministic: fixed seeds, fixed thread
count (2), ``torch.use_deterministic_algorithms(True)``.  See tests/golden/README.md.

    python tests/golden/capture_golden.py --out tests/golden/data            # all parts
    python tests/golden/capture_golden.py --out /tmp/x --parts selection,batches

The same functions are imported by test_golden_current.py, which re-captures into
a temporary directory and compares against the stored artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WM_N1 = ROOT / "output" / "wd01" / "wm_n1"
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))
DATA = Path(__file__).resolve().parent / "data"

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("WMS_EVAL_THREADS", "2")

import numpy as np  # noqa: E402
import torch  # noqa: E402

THREADS = 2
SEED_MODEL = 101          # fresh R3 model init + trainer seed (== cfg.seed)
SEED_BATCH_PERM = 2026    # window sampling permutation
SEED_ITEM = 7000          # + item number: start-shift draws inside the dataset
SEED_LOSS = 900           # + batch number: RNG before each loss/grad pass
SEED_TRAIN = 1100         # + step number: RNG before each training step
N_BATCHES = 3
BATCH_SIZE = 2
HISTORY_S = 60.0
ALL_PARTS = ("selection", "batches", "forward", "losses", "train", "eval")

# ---- external, read-only inputs -------------------------------------------------
LISTS = Path("E:/rd-wt/ws-r3-int/_out/lpwo_v3")
SIDE = Path("E:/rd-wt/ws-packv3fix/_out/sidecars_packv3_v2")   # == ws-r3-int sidecars
PACKS = ("D:/rd-staging/wmn1_feat/pack_v3",
         "E:/rd-wt/dspeed-data/_out/pack_dspeed",
         "E:/rd-wt/damping-snap/_out/pack_damping")
IDENTITIES = (str(SIDE / "identities"), "E:/rd-wt/a9/_out/identities_dspeed",
              "E:/rd-wt/damping-snap/_out/identities_damping")
OVERLAYS = ("E:/rd-wt/dspeed-data/_out/labels_overlay_dspeed",
            "E:/rd-wt/damping-snap/_out/labels_overlay_damping")
OUTCOMES = str(SIDE / "outcomes")
MERGE_PARAMS = "E:/rd-wt/ws-r3-int/_out/merge_params.json"
C1_RUN = Path("E:/research-dense/runs/active/WM-S-C1FAST-20261006/box/full_struct")
STATES = Path("E:/rd-wt/ws-wallc1/_out/wallc1_states_v2/wall_same.json")
SCRATCH_OUT = ROOT / "tests" / "golden" / "_scratch_out"   # never created by the CLI


def setup_determinism() -> None:
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


# ---------------------------------------------------------------------------------
# R3 configuration (mirrors train_r3_chain.sh preset R3; CPU, batch 2, no switching)
# ---------------------------------------------------------------------------------
def r3_argv(unroll_s: float, history_s: float | None) -> list[str]:
    argv = ["train"]
    argv += ["--pack", PACKS[0], "--identities", IDENTITIES[0], "--outcomes", OUTCOMES]
    argv += ["--pack", PACKS[1], "--identities", IDENTITIES[1], "--overlay", OVERLAYS[0]]
    argv += ["--pack", PACKS[2], "--identities", IDENTITIES[2], "--overlay", OVERLAYS[1]]
    argv += ["--window-list", str(LISTS / "windows_train_noPI_noWall.json"),
             "--val-window-list", str(LISTS / "windows_val_noPI_noWall.json"),
             "--exclude-window-list", str(LISTS / "excluded_runs_PI_wall_v3.json"),
             "--cav-drive", "teacher", "--device", "cpu", "--threads", str(THREADS),
             "--num-workers", "0", "--batch-size", str(BATCH_SIZE), "--sumo-fidelity",
             "--start-shift-prob", "0.75", "--truncation-steps", "20",
             "--merge-cal-params", MERGE_PARAMS, "--merge-cal", "--loss-share-logging",
             "--residual-soft-bound-mps2", "4", "--residual-mag-weight", "0.01",
             "--lane-rate-weight", "1.0", "--lane-rate-target", "0.0123",
             "--select-by-long-horizon",
             "--rollout-mode", "residual", "--lane-head", "off",
             "--asym-speed-loss", "2", "--decel-weight", "5", "--decel-threshold", "3",
             "--propagation-loss", "1", "--prop-window", "3",
             "--seed", str(SEED_MODEL), "--unroll-s", repr(float(unroll_s)),
             "--out", str(SCRATCH_OUT)]
    if history_s is not None:
        argv += ["--history-s", repr(float(history_s)), "--history-stride-s", "1"]
    return argv


_BUILD_CACHE: dict = {}


def build_r3(unroll_s: float, history_s: float | None) -> dict:
    """Build (dataset, val dataset, trainer, args) exactly as ``wms.cli train`` does."""
    key = (float(unroll_s), history_s)
    if key in _BUILD_CACHE:
        return _BUILD_CACHE[key]
    os.environ["WM_N1_TRAIN_OUTCOMES"] = OUTCOMES
    os.environ["WM_N1_TRAIN_IDENTITIES"] = IDENTITIES[0]
    from wms import cli
    from wms.train import WmSTrainer

    captured: dict = {}

    class _Capture(WmSTrainer):
        def fit(self, dataset, val_dataset=None):   # capture instead of training
            captured.update(trainer=self, dataset=dataset, val=val_dataset)
            return {}

    argv = r3_argv(unroll_s, history_s)
    saved = cli.WmSTrainer
    cli.WmSTrainer = _Capture
    try:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            rc = cli.main(argv)
    finally:
        cli.WmSTrainer = saved
    if rc != 0 or "trainer" not in captured:
        raise RuntimeError(f"cli.main refused the golden configuration rc={rc}")
    captured["argv"] = argv
    captured["args"] = cli._parser().parse_args(argv)
    _BUILD_CACHE[key] = captured
    return captured


def fresh_r3_trainer(base: dict):
    """Fresh R3-config model (seed SEED_MODEL) + trainer; identical on every call."""
    from wms import cli
    from wms.train import WmSTrainer

    cfg = base["trainer"].config
    physics = base["trainer"].model.physics
    torch.manual_seed(SEED_MODEL)
    model = cli._build_model(base["args"], physics, cfg)
    return WmSTrainer(model, cfg)


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------
def _np(t):
    return t.detach().cpu().numpy()


def sha256_bytes(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def state_checksum(model) -> dict:
    sd = {k: _np(v).astype(np.float32) for k, v in sorted(model.state_dict().items())}
    flat = [v.ravel() for v in sd.values()]
    return {"sha256_float32": sha256_bytes(*flat),
            "n_tensors": len(sd), "n_params": int(sum(v.size for v in flat)),
            "per_tensor_sum": {k: float(v.astype(np.float64).sum()) for k, v in sd.items()},
            "per_tensor_abs_sum": {k: float(np.abs(v.astype(np.float64)).sum())
                                   for k, v in sd.items()}}


def state_npz(model) -> dict:
    return {k.replace(".", "/"): _np(v) for k, v in sorted(model.state_dict().items())}


def write_npz(path: Path, arrays: dict) -> None:
    np.savez_compressed(path, **arrays)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def _scalar_record(record: dict) -> dict:
    out = {}
    for k, v in record.items():
        if k == "seconds":
            continue
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
            out[k] = [float(x) for x in v]
        elif isinstance(v, dict):
            out[k] = v
    return out


# ---------------------------------------------------------------------------------
# (1) data path: window selection
# ---------------------------------------------------------------------------------
def window_selection() -> dict:
    """Keys of the kept training/validation windows, in dataset order, by pack."""
    from train.idea_a import IdeaACorpus
    from wms import lpo
    from wms.train import _select_lpo_positions

    base = build_r3(120.0, None)    # the true R3 horizon (curriculum max 120 s)
    result = {"receipt": {}, "kept_keys": {}}
    for split, ds, name in (("train", base["dataset"], "windows_train_noPI_noWall.json"),
                            ("val", base["val"], "windows_val_noPI_noWall.json")):
        rc = ds.receipt
        result["receipt"][split] = {k: rc[k] for k in (
            "kept_by_pack", "total_by_pack", "n_records", "n_corpus_windows",
            "window_filter", "mode", "horizon_steps") if k in rc}
        result["receipt"][split]["n_dataset_items"] = len(ds)
    # item keys in dataset order: replicate the metadata-only selection
    corpus = IdeaACorpus(PACKS[0], split="TRAIN", horizon_steps=240,
                         extra_pack_dirs=list(PACKS[1:]),
                         extra_overlay_dir=list(OVERLAYS),
                         extra_identity_dirs=list(IDENTITIES[1:]))
    names = lpo.pack_names(list(PACKS))
    for split, fname in (("train", "windows_train_noPI_noWall.json"),
                         ("val", "windows_val_noPI_noWall.json")):
        selected, _ = _select_lpo_positions(
            corpus, names, window_list=str(LISTS / fname),
            exclude_window_list=[str(LISTS / "excluded_runs_PI_wall_v3.json")])
        rows = []
        for i in selected:
            item = corpus._items[corpus._order[i]]
            rows.append([names[int(corpus._global_packs[i])], *lpo.item_key(item)])
        result["kept_keys"][split] = rows
        by_pack: dict = {}
        for r in rows:
            by_pack[r[0]] = by_pack.get(r[0], 0) + 1
        result["receipt"][split]["recomputed_kept_by_pack"] = dict(sorted(by_pack.items()))
        if len(rows) != result["receipt"][split]["n_dataset_items"]:
            raise RuntimeError(f"{split}: selection/dataset size mismatch")
    result["expected_pack_v3"] = {"train": 294, "val": 47}
    return result


def capture_selection(out: Path) -> None:
    write_json(out / "selection.json", window_selection())


# ---------------------------------------------------------------------------------
# (1b) decoded batches
# ---------------------------------------------------------------------------------
BATCH_TENSORS = ("states", "present", "slot_index", "loss_valid", "vehicle_ids",
                 "cav_teacher", "is_cav", "hist", "hist_mask", "driver_hist",
                 "driver_hist_mask", "controller_id")


def batch_variant_name(unroll_s: int, hist: bool) -> str:
    return f"batches_u{unroll_s}_hist{'on' if hist else 'off'}"


def decode_batches(unroll_s: int, hist: bool):
    """3 fixed batches of 2 windows (start-shift draws seeded per item)."""
    from wms.train import collate

    base = build_r3(float(unroll_s), HISTORY_S if hist else None)
    ds = base["dataset"]
    g = torch.Generator().manual_seed(SEED_BATCH_PERM)
    perm = [int(i) for i in torch.randperm(len(ds), generator=g)[:N_BATCHES * BATCH_SIZE]]
    batches, meta = [], {"indices": perm, "items": []}
    for j in range(N_BATCHES):
        items = []
        for n in range(BATCH_SIZE):
            idx = perm[j * BATCH_SIZE + n]
            torch.manual_seed(SEED_ITEM + j * BATCH_SIZE + n)
            items.append(ds[idx])
            meta["items"].append({"dataset_index": idx,
                                  "start_shift_ticks": int(items[-1].get("start_shift_ticks") or 0)})
        batch = collate(items)
        meta.setdefault("identities", []).append(
            [None if x is None else json.loads(json.dumps(x, default=str))
             for x in batch["identities"]])
        batches.append(batch)
    return base, batches, meta


def capture_batches(out: Path) -> None:
    for unroll_s in (10, 60):
        for hist in (True, False):
            base, batches, meta = decode_batches(unroll_s, hist)
            arrays, shapes = {}, {}
            for j, batch in enumerate(batches):
                for name in BATCH_TENSORS:
                    if name in batch:
                        arrays[f"b{j}_{name}"] = _np(batch[name])
                        shapes[f"b{j}_{name}"] = [list(batch[name].shape), str(batch[name].dtype)]
            tag = batch_variant_name(unroll_s, hist)
            write_npz(out / f"{tag}.npz", arrays)
            meta.update(unroll_s=unroll_s, history_s=HISTORY_S if hist else None,
                        n_batches=N_BATCHES, batch_size=BATCH_SIZE, shapes=shapes,
                        seeds={"perm": SEED_BATCH_PERM, "item": SEED_ITEM},
                        start_shift_prob=0.75)
            write_json(out / f"{tag}.json", meta)


# ---------------------------------------------------------------------------------
# (2) model forward
# ---------------------------------------------------------------------------------
def rollout_outputs(model, batch: dict, n_steps: int, *, controller_id=True) -> dict:
    """Teacher-forced rollout with the trainer's argument convention (no grad)."""
    from wms.train import _state_from_kin

    states = batch["states"].float()
    present = batch["present"].bool()
    is_cav = batch["is_cav"].bool()
    teacher = batch["cav_teacher"].float()
    initial = _state_from_kin(states[:, 0], present[:, 0])
    initial["is_cav"] = is_cav[:, 0]
    hist, hist_mask = batch.get("hist"), batch.get("hist_mask")
    kwargs = dict(history=hist, history_mask=hist_mask,
                  driver_history=batch.get("driver_hist", hist),
                  driver_history_mask=batch.get("driver_hist_mask", hist_mask))
    with torch.no_grad():
        roll = model.rollout(
            initial, n_steps=n_steps, k0=0, cav_teacher=teacher[:, :n_steps + 1],
            present_seq=present[:, :n_steps + 1], is_cav_seq=is_cav[:, :n_steps + 1],
            controller_id=batch.get("controller_id") if controller_id else None, **kwargs)
        gate = model.gate(torch.arange(n_steps))
    out = {"accel_mps2": roll["accel"], "raw_residual_mps2": roll["raw_residual_mps2"],
           "gate": gate, "lane_logits": roll["lane_logits"], "positions": roll["positions"],
           "speeds": roll["speeds"], "lanes": roll["lanes"], "masks": roll["masks"],
           "lane_change_probabilities": roll["lane_change_probabilities"]}
    return {k: _np(v) for k, v in out.items()}


def load_c1_model():
    """The C1 checkpoint exactly as wms_eval loads it (physics from TRAIN_RECEIPT)."""
    from wms_eval.models import flags_for
    from wms_eval.wms_plugin import WmsCheckpointModel

    receipt = json.loads((DATA / "c1" / "TRAIN_RECEIPT.json").read_text(encoding="utf-8"))
    wrapper = WmsCheckpointModel(DATA / "c1" / "wm_s.pt", flags_for("wms"),
                                 physics=receipt["physics"], model_kind="structured")
    model = wrapper._load()
    return wrapper, model


def capture_forward(out: Path) -> None:
    # C1: legacy path, history disabled batches.
    _, batches_off, _ = decode_batches(10, False)
    _, batches_off60, _ = decode_batches(60, False)
    wrapper, c1 = load_c1_model()
    c1.eval()
    arrays = {}
    for j in range(N_BATCHES):
        for k, v in rollout_outputs(c1, batches_off[j], 20).items():
            arrays[f"u10_b{j}_{k}"] = v
    for k, v in rollout_outputs(c1, batches_off60[0], 120).items():
        arrays[f"u60_b0_{k}"] = v
    write_npz(out / "forward_c1.npz", arrays)
    write_json(out / "forward_c1.json", {
        "checkpoint": "tests/golden/data/c1/wm_s.pt",
        "checkpoint_sha256": hashlib.sha256((DATA / "c1" / "wm_s.pt").read_bytes()).hexdigest(),
        "flags": "TRAIN_RECEIPT.json config (no history, no soft bound, merge_cal False)",
        "batches": "batches_u10_histoff / batches_u60_histoff", "mode": "eval, no_grad",
        "state_checksum": state_checksum(c1)})
    # R3: fresh initialised model, history-on batches.
    base, batches_on, _ = decode_batches(10, True)
    _, batches_on60, _ = decode_batches(60, True)
    trainer = fresh_r3_trainer(base)
    model = trainer.model
    write_npz(out / "r3_init_state.npz", state_npz(model))
    model.eval()
    arrays = {}
    for j in range(N_BATCHES):
        for k, v in rollout_outputs(model, batches_on[j], 20).items():
            arrays[f"u10_b{j}_{k}"] = v
    for k, v in rollout_outputs(model, batches_on60[0], 120).items():
        arrays[f"u60_b0_{k}"] = v
    write_npz(out / "forward_r3.npz", arrays)
    write_json(out / "forward_r3.json", {
        "init_seed": SEED_MODEL, "mode": "eval, no_grad",
        "state_checksum": state_checksum(model),
        "model_attrs": {"residual_soft_bound_mps2": model.residual_soft_bound_mps2,
                        "residual_soft_bound_neg_mps2": model.residual_soft_bound_neg_mps2,
                        "history_s": model.history_s, "rollout_mode": model.rollout_mode,
                        "lane_head": "off" if getattr(model, "lane_change_head", None) is None else "history"}})


# ---------------------------------------------------------------------------------
# (3) losses + gradient norms; (4) three optimiser steps
# ---------------------------------------------------------------------------------
def grad_groups(model) -> dict:
    per_param, groups, total = {}, {}, 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            per_param[name] = {"norm": 0.0, "sum": 0.0}
            continue
        g = p.grad.detach().double()
        n2 = float(g.pow(2).sum())
        per_param[name] = {"norm": n2 ** 0.5, "sum": float(g.sum())}
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0.0) + n2
        total += n2
    return {"per_param": per_param,
            "per_group_norm": {k: v ** 0.5 for k, v in sorted(groups.items())},
            "global_norm": total ** 0.5}


def capture_losses(out: Path) -> None:
    base, batches, _ = decode_batches(10, True)
    trainer = fresh_r3_trainer(base)
    rows = []
    for j in range(N_BATCHES):
        torch.manual_seed(SEED_LOSS + j)
        trainer.optimizer.zero_grad(set_to_none=True)
        rec = trainer._forward_backward(batches[j], unroll_steps=20)
        rows.append({"batch": j, "unroll_steps": 20, "record": _scalar_record(rec),
                     "grad": grad_groups(trainer.model)})
    write_json(out / "losses_r3.json", {
        "config": "R3 preset, fresh model seed %d, history 60 s, unroll 10 s" % SEED_MODEL,
        "loss_terms": ["position", "speed", "lane", "residual_mag", "lane_rate",
                       "asym_speed", "decel_weight", "propagation"],
        "rows": rows})


def capture_train(out: Path) -> None:
    base, batches, _ = decode_batches(10, True)
    trainer = fresh_r3_trainer(base)
    before = state_checksum(trainer.model)["sha256_float32"]
    steps = []
    for j in range(3):
        torch.manual_seed(SEED_TRAIN + j)
        rec = trainer.train_batch(batches[j], unroll_steps=20)
        steps.append({"step": j + 1, "batch": j, "record": _scalar_record(rec),
                      "checksum": state_checksum(trainer.model)["sha256_float32"]})
    write_json(out / "train3_r3.json", {
        "seed": SEED_MODEL, "optimizer": "Adam lr 1e-3, grad_clip 1.0 (cfg)",
        "init_checksum": before, "steps": steps,
        "final": state_checksum(trainer.model)})
    write_npz(out / "train3_final_params.npz", state_npz(trainer.model))


# ---------------------------------------------------------------------------------
# (5) eval rollouts (verbatim logic of ws-main2score/_out/main2_score/main2_score.py)
# ---------------------------------------------------------------------------------
HORIZONS = (10.0, 30.0, 60.0, 120.0)
DELAY_HORIZONS = (60.0, 120.0)
EVAL_HORIZON_S = 60.0


def eval_state_records(n: int = 3) -> list[dict]:
    from wms_eval.c1v2 import iter_windows

    index = []
    for i, rec in enumerate(iter_windows(STATES)):
        index.append((str(rec.get("world")), float(rec["t0_s"]), str(rec.get("key")), i))
    chosen = sorted(index)[:n]
    wanted = {i for *_, i in chosen}
    found = {}
    for i, rec in enumerate(iter_windows(STATES)):
        if i in wanted:
            found[i] = rec
        if len(found) == len(wanted):
            break
    return [found[i] for *_, i in chosen]


def _section_delays(frames, t0_s, v0, horizons):
    import math
    from wms_eval.metrics import SECTIONS, traffic_metrics

    out = {}
    for h in horizons:
        selected = [f for f in frames if t0_s <= float(f["time_s"]) < t0_s + h]
        if not selected:
            continue
        bins = traffic_metrics(selected, t0_s=t0_s, bin_s=h, free_speed_mps=v0)
        block = {}
        for section in SECTIONS:
            vals = [bins[b]["sections"][section].get("total_delay") for b in bins]
            vals = [v for v in vals if v is not None and math.isfinite(float(v))]
            if vals:
                block[section] = float(sum(vals))
        out[str(h)] = block
    return out


def frames_to_arrays(frames: list[dict], prefix: str) -> dict:
    """Compact, exact array form of predicted frames (id-sorted vehicle tables)."""
    arrays = {f"{prefix}_time_s": np.asarray([float(f["time_s"]) for f in frames])}
    keys = None
    for idx, f in enumerate(frames):
        rows = sorted(f.get("vehicles", []), key=lambda r: str(r["id"]))
        if keys is None:
            keys = sorted(k for k in rows[0] if k != "id" and isinstance(rows[0][k], (int, float, bool))) if rows else []
        arrays[f"{prefix}_f{idx:04d}_ids"] = np.asarray([str(r["id"]) for r in rows])
        for k in keys:
            arrays[f"{prefix}_f{idx:04d}_{k}"] = np.asarray([float(r.get(k, np.nan)) for r in rows])
    return arrays


def capture_eval(out: Path) -> None:
    from wms_eval.c1v2 import _car_compact
    from wms_eval.data import recorded_for
    from wms_eval.metrics import car_metrics
    from wms_eval.models import flags_for
    from wms_eval.wms_plugin import WmsCheckpointModel

    receipt = json.loads((DATA / "c1" / "TRAIN_RECEIPT.json").read_text(encoding="utf-8"))
    predictor = WmsCheckpointModel(DATA / "c1" / "wm_s.pt", flags_for("wms"),
                                   physics=receipt["physics"],
                                   model_kind="structured").predict_states
    results, arrays = [], {}
    for n, record in enumerate(eval_state_records(3)):
        t0_s = float(record["t0_s"])
        v0 = float(record.get("v0_mps", 13.89))
        controllers = record.get("cav_controllers", {})
        arms = {}
        for arm in sorted((record.get("recorded") or {}).keys()):
            predicted = predictor(record, {**controllers, "arm": arm}, EVAL_HORIZON_S, 0)
            recorded = recorded_for(record, arm)
            car = _car_compact(car_metrics(predicted, recorded, t0_s=t0_s,
                                           horizons=tuple(HORIZONS)))
            delays = {"predicted": _section_delays(predicted, t0_s, v0, DELAY_HORIZONS),
                      "recorded": _section_delays(recorded, t0_s, v0, DELAY_HORIZONS)}
            arms[arm] = {"car": car, "delay": delays, "n_frames": len(predicted)}
            arrays.update(frames_to_arrays(predicted, f"s{n}_{arm}"))
        results.append({"world": str(record.get("world")), "t0_s": t0_s,
                        "key": record.get("key"), "arms": arms})
    write_json(out / "eval_rollouts.json", {
        "states_file": "wallc1_states_v2/wall_same.json (sorted by (world,t0_s,key), first 3)",
        "horizon_s": EVAL_HORIZON_S, "cav_mode": "teacher-forced recorded arm",
        "entries": "off (checkpoint config)", "model": "C1 full_struct wm_s.pt",
        "states": results})
    write_npz(out / "eval_pred_frames.npz", arrays)


# ---------------------------------------------------------------------------------
def prepare_inputs(out: Path) -> None:
    """Copy the (small, read-only) C1 checkpoint into the workspace: the eval loader
    refuses checkpoints outside the workspace root."""
    dst = out / "c1"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("wm_s.pt", "TRAIN_RECEIPT.json"):
        if not (dst / name).exists():
            (dst / name).write_bytes((C1_RUN / name).read_bytes())


def capture(out: Path, parts=ALL_PARTS) -> dict:
    setup_determinism()
    out.mkdir(parents=True, exist_ok=True)
    if out.resolve() != DATA.resolve():
        # eval/forward read the C1 checkpoint from DATA (inside the workspace)
        prepare_inputs(DATA)
    else:
        prepare_inputs(out)
    timings = {}
    funcs = {"selection": capture_selection, "batches": capture_batches,
             "forward": capture_forward, "losses": capture_losses,
             "train": capture_train, "eval": capture_eval}
    for part in ALL_PARTS:
        if part in parts:
            t = time.time()
            funcs[part](out)
            timings[part] = round(time.time() - t, 1)
            print(f"[golden] {part}: {timings[part]} s", flush=True)
    return timings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--parts", default=",".join(ALL_PARTS))
    args = ap.parse_args(argv)
    parts = tuple(p for p in args.parts.split(",") if p)
    bad = set(parts) - set(ALL_PARTS)
    if bad:
        raise SystemExit(f"unknown parts {sorted(bad)}")
    out = Path(args.out)
    timings = capture(out, parts)
    meta = {"schema": "wms_golden/1", "git_head": git_head(),
            "source_branch": "wms/r3-int (bf7d4cb)", "parts": list(parts),
            "timings_s": timings, "threads": THREADS,
            "seeds": {"model": SEED_MODEL, "batch_perm": SEED_BATCH_PERM,
                      "item": SEED_ITEM, "loss": SEED_LOSS, "train": SEED_TRAIN},
            "torch": torch.__version__, "numpy": np.__version__,
            "deterministic_algorithms": True}
    if set(parts) == set(ALL_PARTS):
        write_json(out / "meta.json", meta)
    else:
        write_json(out / "meta_partial.json", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
