"""DIAG-HIST shared library (analysis only; no training/simulation/GPU).

Reproduces the round-4 clean scorer contract exactly (bit-identical check done
against ``WM-S-R4-20261011/score/cache``): teacher-forced CAVs, entries OFF,
120 s rollout, seed 0, horizons (10, 30, 60, 120), per-state HDV median |ds|.
Everything is written under this run directory only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

DIAG = Path("E:/research-dense/runs/active/WM-S-R4-DIAG-HIST-20261011")
OUT = DIAG / "out"
SCRIPTS = DIAG / "scripts"
DATA = Path("E:/research-dense/data_cache")
SRC = Path("E:/research-dense/src")
TMP = Path("E:/agent-work/tmp")
PY = "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe"

MODELS = {
    "C1": DATA / "ckpt_c1fast" / "wm_s.pt",
    "R4": Path("E:/research-dense/runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt"),
}
R3_DIAG = Path("E:/research-dense/runs/active/WM-S-R3-DIAG-20261010")
R4_SCORE = Path("E:/research-dense/runs/active/WM-S-R4-20261011/score")
SETS = {
    "std_q20": DATA / "wallc1_states_h60" / "wall_same.json",
    "std_full": DATA / "wallc1_states_h60" / "wall_same.json",
    "real_q20": DATA / "wallc1_states_h60_real" / "wall_same.json",
    "real_full": DATA / "wallc1_states_h60_real" / "wall_same.json",
}
HORIZONS = (10.0, 30.0, 60.0, 120.0)
DT_S = 0.5
N_STEPS = 240

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def ensure_env(workers_threads: int = 1) -> None:
    os.environ["TEMP"] = str(TMP)
    os.environ["TMP"] = str(TMP)
    os.environ["OMP_NUM_THREADS"] = str(workers_threads)
    os.environ["MKL_NUM_THREADS"] = str(workers_threads)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- #
# state readers / subset rules (identical to score_round4.quick20)
# --------------------------------------------------------------------------- #
def iter_records(path: Path):
    from wms.eval.normalize import iter_windows
    return iter_windows(path)


def quick_keys(path: Path, max_worlds: int = 20):
    """Harness rule: sort by (world, t0_s, key); first state per world; first N worlds."""
    by_world = {}
    for record in iter_records(path):
        world = str(record.get("world"))
        ordered = (world, float(record.get("t0_s", 0.0)), str(record.get("key")))
        if world not in by_world or ordered < by_world[world]:
            by_world[world] = ordered
    worlds = sorted(by_world)
    return sorted(by_world[w] for w in worlds[:max_worlds])


def quick20(path: Path, max_worlds: int = 20) -> list[dict]:
    keys = {(w, t, k) for w, t, k in
            [(x[0], x[1], x[2]) for x in _quick_triples(path, max_worlds)]}
    out = []
    for record in iter_records(path):
        if (str(record.get("world")), float(record.get("t0_s", 0.0)),
                str(record.get("key"))) in keys:
            out.append(record)
    out.sort(key=lambda r: (str(r.get("world")), float(r.get("t0_s", 0.0)),
                            str(r.get("key"))))
    return out


def _quick_triples(path: Path, max_worlds: int):
    by_world = {}
    for record in iter_records(path):
        world = str(record.get("world"))
        ordered = (world, float(record.get("t0_s", 0.0)), str(record.get("key")))
        if world not in by_world or ordered < by_world[world]:
            by_world[world] = ordered
    worlds = sorted(by_world)
    return [by_world[w] for w in worlds[:max_worlds]]


def ident(record: dict) -> tuple:
    return (str(record.get("world")), float(record.get("t0_s", 0.0)),
            str(record.get("key")))


# --------------------------------------------------------------------------- #
# predictors
# --------------------------------------------------------------------------- #
def make_predictor(name: str):
    import torch
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    torch.set_num_threads(1)
    return WmsPredictor.from_checkpoint(MODELS[name], RolloutFlags())


def score_rows(predictor, records: list[dict]) -> list[dict]:
    from wms.eval.score import ScoreSpec, evaluate_state
    spec = ScoreSpec(horizons_s=HORIZONS, predict_horizon_s=120.0, seed=0)
    rows = []
    for record in records:
        rows.append(evaluate_state(record, predictor, spec))
    return rows


def summary_for(rows: list[dict]) -> dict:
    """Horizon medians + blow-ups in the round-4 scorer format."""
    from wms.eval.report import blowup_rate, horizon_summary
    arm = "wall"
    out = {"n_states": len(rows),
           "ds": {f"{h:g}": horizon_summary(rows, arm, h, "ds_med") for h in HORIZONS},
           "dv": {f"{h:g}": horizon_summary(rows, arm, h, "dv_med") for h in HORIZONS},
           "blowup": blowup_rate(rows, arm)}
    return out


def paired(rows_a: list[dict], rows_b: list[dict], n_boot: int = 2000) -> dict:
    from wms.eval.report import paired_bootstrap
    return {f"{h:g}": paired_bootstrap(rows_a, rows_b, "wall", h, n_boot=n_boot, seed=0)
            for h in HORIZONS}


# --------------------------------------------------------------------------- #
# step-0 encoder probing (replicates WmSModel._encode_drivers at rollout step 0)
# --------------------------------------------------------------------------- #
def encode_state(predictor, record: dict) -> dict:
    """Run the R4 driver encoder exactly as rollout step 0 does; return per-slot outputs."""
    import numpy as np
    import torch
    from wms.eval.history_inputs import HistoryInputs
    from wms.eval.rollout_inputs import t0_vehicles

    model = predictor.model
    hist_inputs = HistoryInputs(predictor.backend)
    t0_s, ids, kin, is_cav = t0_vehicles(record)
    driver_history, driver_history_mask = hist_inputs.build(record, ids)
    kin_t = torch.as_tensor(kin[None, :, :]).float()
    present = torch.ones(kin_t.shape[:2], dtype=torch.bool)
    interaction = model._driver_interaction(kin_t, present)
    defaults = model._default_driver_parameters(kin_t, 1, kin_t.shape[1])
    encoded = model.driver_encoder(driver_history, driver_history_mask, interaction,
                                   defaults)
    cav = np.asarray(is_cav, bool)
    return {
        "ids": [str(v) for v in ids],
        "is_cav": cav,
        "values": encoded["values"][0].detach().numpy().astype(float),      # [K,9]
        "embedding": encoded["embedding"][0].detach().numpy().astype(float),  # [K,102]
        "uncertainty": encoded["uncertainty"][0].detach().numpy().astype(float),
        "shrinkage": encoded["shrinkage"][0].detach().numpy().astype(float),  # [K]
        "hist_valid_ticks": (driver_history_mask[0].numpy().astype(int)).sum(-1),  # [K]
        "defaults": defaults[0, 0].detach().numpy().astype(float),
    }


def train_window_iter(n_windows: int, seed: int = 0):
    """Yield the trainer's step-0 encoder inputs for sampled TRAIN windows.

    Mirrors ``_PackWindowLoader``: original tick, or a shifted later tick with
    probability 0.75 (uniform 1..room, room = run_end - tick - 1 - 240).
    """
    import numpy as np
    import torch
    from wms.data.corpus import PackCorpus
    from wms.data.pack_history import prior_history
    from wms.data.windows import select_lpo_positions

    corpus = PackCorpus.open(DATA / "pack_damping",
                             extra_pack_dirs=[DATA / "pack_dspeed", DATA / "pack_v3"])
    names = [p.name for p in corpus.pack_dirs]
    positions, _ = select_lpo_positions(
        corpus, names,
        window_list=DATA / "lpwo_v3" / "windows_train_noPI_noWall.json",
        exclude_window_list=[DATA / "lpwo_v3" / "excluded_runs_PI_wall_v3.json"])
    veh_kin = corpus.array("veh_kin")
    veh_mask = corpus.array("veh_mask")
    veh_step = corpus.array("veh_step")
    torch.manual_seed(seed)
    order = torch.randperm(len(positions)).tolist()
    picked = 0
    for i in order:
        if picked >= n_windows:
            break
        meta = corpus.meta(positions[i])
        tick0 = int(meta["tick"])
        shift = 0
        if float(torch.rand(())) < 0.75:
            bounds = corpus.run_bounds(tick0)
            if bounds is not None:
                room = bounds[1] - tick0 - 1 - 240
                shift = int(torch.randint(1, room + 1, ())) if room > 0 else 0
        tick = tick0 + shift
        pres = np.asarray(veh_mask[tick], bool)
        if pres.sum() < 5:
            continue
        cav = np.asarray(veh_step[tick][:, 7], float) > 0.5
        kin = np.asarray(veh_kin[tick], np.float32)
        feat, fmask = prior_history(corpus, tick, pres, history_s=60.0,
                                    history_stride_s=1.0)
        yield {"tick": tick, "shift": shift, "world": str(meta.get("world")),
               "pack": str(meta.get("batch") or meta.get("chunk")),
               "kin": kin, "present": pres, "is_cav": cav,
               "feat": feat.astype(np.float32), "feat_mask": fmask}
        picked += 1


def encode_train_window(predictor, win: dict) -> dict:
    """Encoder outputs for one training window (same call as encode_state)."""
    import torch
    model = predictor.model
    kin_t = torch.as_tensor(win["kin"][None, :, :]).float()
    present = torch.as_tensor(win["present"][None, :]).bool()
    interaction = model._driver_interaction(kin_t, present)
    defaults = model._default_driver_parameters(kin_t, 1, kin_t.shape[1])
    encoded = model.driver_encoder(torch.as_tensor(win["feat"][None]),
                                   torch.as_tensor(win["feat_mask"][None]).bool(),
                                   interaction, defaults)
    return {"values": encoded["values"][0].detach().numpy().astype(float),
            "embedding": encoded["embedding"][0].detach().numpy().astype(float),
            "shrinkage": encoded["shrinkage"][0].detach().numpy().astype(float),
            "defaults": defaults[0, 0].detach().numpy().astype(float)}


# --------------------------------------------------------------------------- #
# rollout arrays (R3-DIAG convention)
# --------------------------------------------------------------------------- #
def frames_to_arrays(frames, t0: float, ids: list[str], n_steps: int = N_STEPS):
    import numpy as np
    n = len(ids)
    col = {v: i for i, v in enumerate(ids)}
    s = np.full((n_steps + 1, n), np.nan, dtype=np.float32)
    v = np.full((n_steps + 1, n), np.nan, dtype=np.float32)
    lane = np.full((n_steps + 1, n), -9, dtype=np.int8)
    pres = np.zeros((n_steps + 1, n), dtype=bool)
    cav = np.zeros((n_steps + 1, n), dtype=bool)
    for fr in frames:
        k = int(round((float(fr["time_s"]) - t0) / DT_S))
        if k < 0 or k > n_steps:
            continue
        for r in fr.get("vehicles", []):
            if not r.get("present", True):
                continue
            j = col.get(str(r["id"]))
            if j is None:
                continue
            s[k, j] = float(r["s"])
            v[k, j] = float(r["v"])
            lane[k, j] = int(r.get("lane", -1))
            pres[k, j] = True
            cav[k, j] = bool(r.get("is_cav", False))
    return s, v, lane, pres, cav


def rollout_arrays(predictor, record: dict, horizon_s: float = 120.0):
    """Full rollout -> (metrics rows shape, ids, pred arrays, recorded arrays)."""
    import numpy as np
    from wms.eval.metrics import car_metrics
    from wms.eval.normalize import recorded_for
    from wms.eval.score import car_compact
    t0 = float(record["t0_s"])
    controllers = record.get("cav_controllers", {})
    frames = predictor.predict_states(record, {**controllers, "arm": "wall"}, horizon_s, 0)
    recorded = recorded_for(record, "wall")
    metrics = car_compact(car_metrics(frames, recorded, t0_s=t0, horizons=HORIZONS))
    ids = sorted({str(r["id"]) for fr in frames for r in fr.get("vehicles", [])}
                 | {str(r["id"]) for fr in recorded for r in fr.get("vehicles", [])})
    ps, pv, pl, pp, pc = frames_to_arrays(frames, t0, ids)
    rs, rv, rl, rp, rc = frames_to_arrays(recorded, t0, ids)
    arrays = {"ids": ids, "p_s": ps, "p_v": pv, "p_lane": pl, "p_pres": pp, "p_cav": pc,
              "r_s": rs, "r_v": rv, "r_lane": rl, "r_pres": rp, "r_cav": rc}
    return metrics, arrays


def true_lane_changes(lane, pres, cav, ids, lo: int = 1, hi: int = 240) -> dict:
    """True-definition lane changes (junction forward-filled), R3-DIAG convention."""
    import numpy as np
    filled = lane.copy()
    for t in range(1, filled.shape[0]):
        bad = filled[t] < 0
        filled[t, bad] = filled[t - 1, bad]
    changes = steps = merges = 0
    for t in range(lo, hi + 1):
        ok = pres[t] & pres[t - 1] & ~cav[t] & ~cav[t - 1]
        ok &= filled[t] >= 0
        changes += int((ok & (filled[t] != filled[t - 1])).sum())
        merges += int((ok & (filled[t - 1] == 2) & (filled[t] == 1)).sum())
        steps += int(ok.sum())
    return {"changes": changes, "merges_2to1": merges, "hdv_steps": steps}


def q(v, ps=(5, 25, 50, 75, 95)):
    import numpy as np
    a = np.asarray(v, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return {str(p): float(np.percentile(a, p)) for p in ps}
