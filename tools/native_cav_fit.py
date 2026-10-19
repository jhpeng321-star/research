"""V10NATIVECAV: fit and score a learned native-CAV twin on TRAIN-val native episodes.

The WM's counterfactual ``base`` rows and the entrance-zone CAVs follow a *native* twin.
The legacy twin (fixed IDM) hard-brakes far more often than the recorded SUMO CAVs, which
flatters the policy in ``policy - native`` comparisons.  This tool fits a small supervised
head (``wms.control.native_cav``) to recorded native-CAV accelerations and reports the
before/after metrics on held-out TRAIN-val worlds:

* acceleration distribution, hard-brake share (a < -2.5 m/s^2), jerk;
* time headway at braking events;
* one-step acceleration error and a 10 s teacher-forced rollout error.

Only ``split_role == "val"`` episodes are read (never the evaluation seeds 964000-964199,
sealed/holdout/cal_split/chunk5/j4b/inc1 paths).  Worlds are split deterministically
70/30 into fit and held-out-eval; inside fit, 80/20 into train/early-stop.

    python tools/native_cav_fit.py [--workers 3] [--epochs 40]
        [--data-root E:/research-dense/.../merge_calib/data/v5_native]
        [--out-root E:/research-dense/.../v10nativecav]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "3")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from wms.control.contract import ACCEL_MIN_MPS2  # noqa: E402
from wms.control.counterfactual import idm_native_accel  # noqa: E402
from wms.control.native_cav import (FEATURE_NAMES, HEAD_ACCEL_MAX_MPS2, HEAD_SCHEMA,  # noqa: E402
                                    feature_matrix, head_native_accel, leader_gap_dv)

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
EVAL_SEEDS = range(964000, 964200)
DT_S = 0.5
HARD_BRAKE_MPS2 = -2.5
DEFAULT_DATA = Path("E:/research-dense/runs/active/C4-V7FT-20261017/merge_calib/data/v5_native")
DEFAULT_OUT = Path("E:/research-dense/runs/active/C4-V7FT-20261017/v10nativecav")
HIDDEN = 32
BATCH = 8192
HARD_WEIGHT = 5.0
ROLLOUT_STEPS = 20
ROLLOUT_EVERY = 20
ROLLOUT_MAX_PER_EPISODE = 80


def guard(path: Path) -> Path:
    text = str(path).replace("\\", "/").lower()
    if any(token in text for token in FORBIDDEN):
        raise SystemExit(f"refusing restricted path {path}")
    return path


def world_bucket(world: str, salt: str = "") -> int:
    return int(hashlib.sha1((world + salt).encode("utf-8")).hexdigest()[:8], 16) % 10


def read_episodes(data_root: Path) -> tuple[list[dict], list[dict]]:
    manifest = json.loads((data_root / "C4_EPISODES_MANIFEST.json").read_text(encoding="utf-8"))
    val = [e for e in manifest["episodes"] if e.get("split_role") == "val"]
    for episode in val:
        if int(episode["seed"]) in EVAL_SEEDS:
            raise SystemExit(f"evaluation seed {episode['seed']} refused")
        guard(data_root / episode["dir"])
    if len(val) < 2:
        raise SystemExit("no TRAIN-val episodes")
    fit = [e for e in val if world_bucket(e["world"]) < 7]
    ev = [e for e in val if world_bucket(e["world"]) >= 7]
    return fit, ev


def split_fit(fit: list[dict]) -> tuple[list[dict], list[dict]]:
    train = [e for e in fit if world_bucket(e["world"], "#train") < 8]
    stop = [e for e in fit if world_bucket(e["world"], "#train") >= 8]
    if not train or not stop:
        cut = max(1, int(0.8 * len(fit)))
        train, stop = fit[:cut], fit[cut:]
    return train, stop


def _lane(kin: np.ndarray) -> np.ndarray:
    onehot = kin[..., 1:4] > 0.5
    return np.where(onehot.any(-1), onehot.argmax(-1), -1)


def episode_arrays(data_root: Path, episode: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directory = data_root / episode["dir"]
    kin = np.load(directory / "veh_kin.npy", allow_pickle=False)
    mask = np.load(directory / "veh_mask.npy", allow_pickle=False).astype(bool)
    cav = np.load(directory / "is_cav.npy", allow_pickle=False).astype(bool)
    return kin, mask, cav


def training_rows(data_root: Path, episodes: list[dict], max_states: int = 400):
    xs, ys = [], []
    for episode in episodes:
        kin, mask, cav = episode_arrays(data_root, episode)
        steps = np.arange(0, kin.shape[0], max(1, kin.shape[0] // max_states))[:max_states]
        k = np.asarray(kin[steps], np.float32)
        m = mask[steps]
        c = cav[steps]
        lane = _lane(k)
        feats = feature_matrix(k[..., 0], k[..., 4], k[..., 5], lane, m, k[..., 6])
        sel = m & c & (lane >= 0)
        xs.append(feats[sel])
        ys.append(k[..., 5][sel])
    return np.concatenate(xs), np.concatenate(ys)


def build_mlp(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(len(FEATURE_NAMES), HIDDEN), torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN, 1))


def train_head(x_train, y_train, x_stop, y_stop, epochs: int, seed: int = 0):
    net = build_mlp(seed)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    x_train, y_train = torch.as_tensor(x_train), torch.as_tensor(y_train)
    x_stop, y_stop = torch.as_tensor(x_stop), torch.as_tensor(y_stop)
    best = (math.inf, None, 0)
    history = []
    for epoch in range(epochs):
        net.train()
        order = torch.randperm(len(x_train))
        total = 0.0
        for start in range(0, len(x_train), BATCH):
            index = order[start:start + BATCH]
            x, y = x_train[index], y_train[index]
            pred = net(x).squeeze(-1)
            weight = torch.where(y < HARD_BRAKE_MPS2, torch.tensor(HARD_WEIGHT),
                                 torch.tensor(1.0))
            loss = torch.mean(weight * (pred - y).square())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(index)
        net.eval()
        with torch.no_grad():
            stop_pred = net(x_stop).squeeze(-1)
        rmse = float(torch.sqrt(torch.mean((stop_pred - y_stop).square())))
        train_loss = total / len(x_train)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "stop_rmse": rmse})
        if rmse < best[0]:
            best = (rmse, {key: value.detach().clone() for key, value in net.state_dict().items()},
                    epoch + 1)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch + 1:3d} train {train_loss:.5f} stop_rmse {rmse:.5f}", flush=True)
    net.load_state_dict(best[1])
    return net, {"best_epoch": best[2], "best_stop_rmse": best[0], "history": history}


def export_head(net: torch.nn.Module, info: dict, path: Path, train_worlds: int,
                stop_worlds: int, rows: int) -> dict:
    layers = []
    for module in net:
        if isinstance(module, torch.nn.Linear):
            layers.append({"w": module.weight.detach().numpy().round(7).tolist(),
                           "b": module.bias.detach().numpy().round(7).tolist()})
    payload = {
        "schema": HEAD_SCHEMA,
        "features": list(FEATURE_NAMES),
        "input_dim": len(FEATURE_NAMES),
        "hidden": HIDDEN,
        "activation": "tanh",
        "accel_min": ACCEL_MIN_MPS2,
        "accel_max": HEAD_ACCEL_MAX_MPS2,
        "layers": layers,
        "fit": {
            "train_worlds": train_worlds,
            "early_stop_worlds": stop_worlds,
            "rows": rows,
            "best_epoch": info["best_epoch"],
            "best_stop_rmse": info["best_stop_rmse"],
            "hard_brake_weight": HARD_WEIGHT,
            "loss": "weighted MSE",
        },
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def _state(kin: np.ndarray, mask: np.ndarray, cav: np.ndarray) -> dict:
    lane = _lane(kin)
    return {
        "s_m": torch.as_tensor(kin[..., 0]),
        "v_mps": torch.as_tensor(kin[..., 4]),
        "a_mps2": torch.as_tensor(kin[..., 5]),
        "lane": torch.as_tensor(lane),
        "present": torch.as_tensor(mask),
        "is_cav": torch.as_tensor(cav),
        "length_m": torch.as_tensor(kin[..., 6]),
    }


def predict_episode(kin: np.ndarray, mask: np.ndarray, cav: np.ndarray, kind: str,
                    head=None, a_max: float | None = None, chunk: int = 64) -> np.ndarray:
    out = np.zeros(kin.shape[:2], np.float32)
    with torch.no_grad():
        for start in range(0, kin.shape[0], chunk):
            stop = min(kin.shape[0], start + chunk)
            state = _state(np.asarray(kin[start:stop], np.float32), mask[start:stop],
                           cav[start:stop])
            if kind == "head":
                pred = head_native_accel(state, head)
            else:
                pred = idm_native_accel(state, a_max=float(a_max))
            out[start:stop] = pred.numpy()
    return out


def _dist(values: np.ndarray) -> dict:
    values = np.asarray(values, np.float32)
    q = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {"n": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()),
            "p1": float(q[0]), "p5": float(q[1]), "p25": float(q[2]), "p50": float(q[3]),
            "p75": float(q[4]), "p95": float(q[5]), "p99": float(q[6]),
            "hard_brake_share": float(np.mean(values < HARD_BRAKE_MPS2))}


def jerk_values(accel: np.ndarray, mask: np.ndarray, cav: np.ndarray, lane: np.ndarray,
                vid: np.ndarray) -> np.ndarray:
    ok = (mask[:-1] & mask[1:] & cav[:-1] & cav[1:] & (lane[:-1] >= 0) & (lane[1:] >= 0)
          & (vid[:-1] == vid[1:]) & (vid[:-1] >= 0))
    diff = np.abs(accel[1:] - accel[:-1])[ok] / DT_S
    return diff


def episode_metrics(data_root: Path, episodes: list[dict], head, kind: str,
                    a_max: float | None = None) -> dict:
    recorded, predicted = [], []
    headway_all, headway_brake, headway_hard = [], [], []
    rec_headway_all, rec_headway_brake, rec_headway_hard = [], [], []
    jerk_rec, jerk_pred = [], []
    one_errors = []
    for episode in episodes:
        kin, mask, cav = episode_arrays(data_root, episode)
        lane = _lane(kin)
        valid = mask & cav & (lane >= 0)
        pred = predict_episode(kin, mask, cav, kind, head=head, a_max=a_max)
        a = kin[..., 5]
        recorded.append(a[valid])
        predicted.append(pred[valid])
        one_errors.append((pred - a)[valid])
        vid = np.load(data_root / episode["dir"] / "vid.npy", allow_pickle=False)
        jerk_rec.append(jerk_values(a, mask, cav, lane, vid))
        jerk_pred.append(jerk_values(pred, mask, cav, lane, vid))
        state = _state(kin, mask, cav)
        gap, _dv, has = leader_gap_dv(state)
        gap = gap.numpy()
        hv = np.where(has.numpy() & valid, gap / np.maximum(kin[..., 4], 0.1), np.nan)
        headway_all.append(hv[np.isfinite(hv)])
        headway_brake.append(hv[valid & (pred < -1.0) & np.isfinite(hv)])
        headway_hard.append(hv[valid & (pred < HARD_BRAKE_MPS2) & np.isfinite(hv)])
        rec_headway_all.append(hv[np.isfinite(hv)])
        rec_headway_brake.append(hv[valid & (a < -1.0) & np.isfinite(hv)])
        rec_headway_hard.append(hv[valid & (a < HARD_BRAKE_MPS2) & np.isfinite(hv)])
    rec = np.concatenate(recorded)
    prd = np.concatenate(predicted)
    err = np.concatenate(one_errors)
    return {
        "recorded": _dist(rec),
        "predicted": _dist(prd),
        "one_step": {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
                     "bias": float(np.mean(err))},
        "jerk": {"recorded_mean": float(np.concatenate(jerk_rec).mean()),
                 "recorded_p95": float(np.percentile(np.concatenate(jerk_rec), 95)),
                 "predicted_mean": float(np.concatenate(jerk_pred).mean()),
                 "predicted_p95": float(np.percentile(np.concatenate(jerk_pred), 95))},
        "headway": {
            "recorded_median": float(np.nanmedian(np.concatenate(rec_headway_all))),
            "recorded_braking_median": (float(np.nanmedian(np.concatenate(rec_headway_brake)))
                                        if rec_headway_brake else None),
            "recorded_hard_brake_median": (float(np.nanmedian(np.concatenate(rec_headway_hard)))
                                           if rec_headway_hard else None),
            "recorded_hard_brake_n": int(sum(len(x) for x in rec_headway_hard)),
            "predicted_median": float(np.nanmedian(np.concatenate(headway_all))),
            "predicted_braking_median": (float(np.nanmedian(np.concatenate(headway_brake)))
                                         if headway_brake else None),
            "predicted_hard_brake_median": (float(np.nanmedian(np.concatenate(headway_hard)))
                                            if headway_hard else None),
            "predicted_hard_brake_n": int(sum(len(x) for x in headway_hard)),
        },
    }


def rollout_metric(data_root: Path, episodes: list[dict], head, kind: str,
                   a_max: float | None = None, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    speed_err, position_err, accel_err = [], [], []
    n_rollouts = 0
    with torch.no_grad():
        for episode in episodes:
            kin, mask, cav = episode_arrays(data_root, episode)
            lane = _lane(kin)
            v_all = kin[..., 4]
            s_all = kin[..., 0]
            a_all = kin[..., 5]
            T, K = mask.shape
            starts = []
            for t0 in range(0, max(1, T - ROLLOUT_STEPS - 1), ROLLOUT_EVERY):
                for slot in np.nonzero(mask[t0] & cav[t0] & (lane[t0] >= 0))[0]:
                    window = slice(t0, t0 + ROLLOUT_STEPS + 1)
                    if (mask[window, slot].all() and cav[window, slot].all()
                            and (lane[window, slot] == lane[t0, slot]).all()):
                        starts.append((t0, int(slot)))
            if len(starts) > ROLLOUT_MAX_PER_EPISODE:
                pick = rng.choice(len(starts), ROLLOUT_MAX_PER_EPISODE, replace=False)
                starts = [starts[i] for i in pick]
            if not starts:
                continue
            n = len(starts)
            t0 = np.array([x[0] for x in starts])
            slot = np.array([x[1] for x in starts])
            s_f = kin[t0, slot, 0].astype(np.float32).copy()
            v_f = kin[t0, slot, 4].astype(np.float32).copy()
            lane_f = lane[t0, slot].copy()
            rows = np.arange(n)
            for h in range(ROLLOUT_STEPS):
                t = t0 + h
                k = np.asarray(kin[t], np.float32).copy()
                m = mask[t].copy()
                c = cav[t].copy()
                ln = lane[t].copy()
                k[rows, slot, 0] = s_f
                k[rows, slot, 4] = v_f
                m[rows, slot] = True
                c[rows, slot] = True
                ln[rows, slot] = lane_f
                k[rows, slot, 1:4] = 0.0
                k[rows, slot, 1 + lane_f] = 1.0
                k[rows, slot, 6] = 5.0
                state = _state(k, m, c)
                acc = (head_native_accel(state, head) if kind == "head"
                       else idm_native_accel(state, a_max=float(a_max)))
                accel_f = acc[rows, slot].numpy()
                accel_err.append(accel_f - a_all[t, slot])
                speed = np.clip(v_f + DT_S * accel_f, 0.0, None)
                moving = np.where(accel_f < 0, np.clip(v_f / np.maximum(-accel_f, 1e-8), 0, DT_S),
                                  DT_S)
                position = s_f + v_f * moving + 0.5 * accel_f * moving ** 2
                s_f, v_f = position.astype(np.float32), speed.astype(np.float32)
            t_end = t0 + ROLLOUT_STEPS
            speed_err.append(v_f - v_all[t_end, slot])
            position_err.append(s_f - s_all[t_end, slot])
            n_rollouts += n
    speed_err = np.concatenate(speed_err) if speed_err else np.zeros(1, np.float32)
    position_err = np.concatenate(position_err) if position_err else np.zeros(1, np.float32)
    accel_err = np.concatenate(accel_err) if accel_err else np.zeros(1, np.float32)
    return {"n_rollouts": int(n_rollouts),
            "speed_rmse_10s": float(np.sqrt(np.mean(speed_err ** 2))),
            "speed_mae_10s": float(np.mean(np.abs(speed_err))),
            "position_rmse_10s": float(np.sqrt(np.mean(position_err ** 2))),
            "position_mae_10s": float(np.mean(np.abs(position_err))),
            "position_p90_abs_10s": float(np.percentile(np.abs(position_err), 90)),
            "accel_rmse": float(np.sqrt(np.mean(accel_err ** 2)))}


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path: Path, result: dict) -> None:
    m = result["metrics"]
    rows = [
        ("accel mean (m/s^2)", lambda x: fmt(x["predicted"]["mean"])),
        ("accel std", lambda x: fmt(x["predicted"]["std"])),
        ("p1", lambda x: fmt(x["predicted"]["p1"])),
        ("p5", lambda x: fmt(x["predicted"]["p5"])),
        ("p50", lambda x: fmt(x["predicted"]["p50"])),
        ("p95", lambda x: fmt(x["predicted"]["p95"])),
        ("p99", lambda x: fmt(x["predicted"]["p99"])),
        ("hard-brake share (<-2.5)", lambda x: f"{100 * x['predicted']['hard_brake_share']:.3f} %"),
        ("jerk mean |da|/dt", lambda x: fmt(x["jerk"]["predicted_mean"])),
        ("jerk p95", lambda x: fmt(x["jerk"]["predicted_p95"])),
        ("one-step accel MAE", lambda x: fmt(x["one_step"]["mae"], 4)),
        ("one-step accel RMSE", lambda x: fmt(x["one_step"]["rmse"], 4)),
        ("one-step bias", lambda x: fmt(x["one_step"]["bias"], 4)),
        ("headway at braking, median (s)", lambda x: fmt(x["headway"]["predicted_braking_median"])),
        ("headway at hard brake, median (s)", lambda x: fmt(x["headway"]["predicted_hard_brake_median"])),
        ("hard-brake events", lambda x: str(x["headway"]["predicted_hard_brake_n"])),
        ("10 s rollout speed RMSE", lambda x: fmt(x["rollout"]["speed_rmse_10s"], 4)),
        ("10 s rollout speed MAE", lambda x: fmt(x["rollout"]["speed_mae_10s"], 4)),
        ("10 s rollout position RMSE", lambda x: fmt(x["rollout"]["position_rmse_10s"], 4)),
        ("10 s rollout position MAE", lambda x: fmt(x["rollout"]["position_mae_10s"], 4)),
        ("10 s rollout position p90 |err|", lambda x: fmt(x["rollout"]["position_p90_abs_10s"], 4)),
        ("rollout accel RMSE", lambda x: fmt(x["rollout"]["accel_rmse"], 4)),
    ]
    lines = [
        "# NATIVECAV: learned native-CAV twin vs the recorded SUMO native CAV",
        "",
        "V10NATIVECAV, TRAIN-val only.  Recorded data = the SUMO native CAV (the reference); "
        "legacy = the WM's fixed IDM twin; head = the fitted native-CAV head.",
        "",
        f"* data: `{result['data_root']}`",
        f"* fit worlds: {result['split']['fit_worlds']} "
        f"(train {result['split']['train_worlds']}, early-stop {result['split']['stop_worlds']}); "
        f"held-out eval worlds: {result['split']['eval_worlds']}",
        f"* head: {len(FEATURE_NAMES)}-{HIDDEN}-{HIDDEN}-1 tanh MLP, "
        f"{result['head']['rows']} rows, best epoch {result['head']['best_epoch']}, "
        f"early-stop RMSE {result['head']['best_stop_rmse']:.4f}",
        f"* weights: committed at `configs/native_cav_head_v5.json` "
        f"(fit copy `{result['head']['path']}`)",
        f"* switch: `counterfactual.native_cav = \"head\"` in "
        f"`configs/c4_policy_v10nativecav_v5m.toml`; default is `\"idm\"`",
        f"* evaluation events: {m['recorded']['recorded']['n']:,} CAV steps, "
        f"{m['recorded']['headway']['recorded_hard_brake_n']:,} recorded hard-brake events, "
        f"{m['head']['rollout']['n_rollouts']:,} teacher-forced 10 s rollouts",
        "",
        "**Result.**  The head removes the native twin's over-braking: hard-brake share "
        f"{100 * m['legacy_1.2']['predicted']['hard_brake_share']:.2f} % (a_max=1.2) and "
        f"{100 * m['legacy_2.6']['predicted']['hard_brake_share']:.2f} % (a_max=2.6) -> "
        f"{100 * m['head']['predicted']['hard_brake_share']:.2f} % (recorded "
        f"{100 * m['recorded']['recorded']['hard_brake_share']:.2f} %); one-step RMSE "
        f"{m['legacy_1.2']['one_step']['rmse']:.3f} -> {m['head']['one_step']['rmse']:.3f}; "
        f"10 s speed RMSE {m['legacy_1.2']['rollout']['speed_rmse_10s']:.3f} -> "
        f"{m['head']['rollout']['speed_rmse_10s']:.3f}.",
        "",
        "| metric | recorded (SUMO) | legacy IDM a_max=2.6 | legacy IDM a_max=1.2 | head |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, get in rows:
        leg26 = m["legacy_2.6"]
        leg12 = m["legacy_1.2"]
        if label == "accel mean (m/s^2)":
            rec = fmt(m["recorded"]["recorded"]["mean"])
        elif label == "accel std":
            rec = fmt(m["recorded"]["recorded"]["std"])
        elif label in ("p1", "p5", "p50", "p95", "p99"):
            rec = fmt(m["recorded"]["recorded"][label])
        elif label.startswith("hard-brake share"):
            rec = f"{100 * m['recorded']['recorded']['hard_brake_share']:.3f} %"
        elif label.startswith("jerk"):
            rec = fmt(m["recorded"]["jerk"]["recorded_mean" if "mean" in label else "recorded_p95"])
        elif label.startswith("headway"):
            rec = fmt(m["recorded"]["headway"]["recorded_braking_median" if "braking" in label
                                          else "recorded_hard_brake_median"])
        elif label == "hard-brake events":
            rec = str(m["recorded"]["headway"]["recorded_hard_brake_n"])
        else:
            rec = "-"
        lines.append(f"| {label} | {rec} | {get(leg26)} | {get(leg12)} | {get(m['head'])} |")
    lines += [
        "",
        "## What changed",
        "",
        "The legacy native twin is a fixed IDM (`v0=31.29, b=2.0, s0=2.0, T=1.2, delta=4`, "
        "`a_min=-3`) whose same-lane leader term over-brakes.  The learned head predicts the "
        "recorded native-CAV acceleration from the same inputs (own speed, leader bumper gap, "
        "speed difference, lane, merge-frame position, leader presence).  The switch is "
        "`counterfactual.native_cav = \"head\"`; the default stays `\"idm\"`, so existing "
        "checkpoints/runs are bit-identical.",
        "",
        "## Fit",
        "",
        f"Worlds are split by a SHA-1 hash of the world id (70 % fit / 30 % held-out eval); "
        f"inside fit, 80 % train / 20 % early-stop.  Loss = MSE with a {HARD_WEIGHT:g}x weight "
        "on hard-brake rows (0.2 % of steps); Adam, batch 8192.  No hand-set behaviour rules: "
        "all head weights are fitted to recorded accelerations.  The output is clamped to "
        f"[{ACCEL_MIN_MPS2}, {HEAD_ACCEL_MAX_MPS2}] (SUMO cav_typical accel).",
        "",
        "## Metric definitions",
        "",
        "* One-step: predicted acceleration at the recorded state vs the recorded acceleration "
        "of that state (the recorded `a[t]` is applied over `[t, t+0.5 s)`; checked against "
        "`v[t+1]-v[t]`).",
        "* Jerk: `|a[t+1]-a[t]| / 0.5 s` for the same vehicle on consecutive states.",
        "* Headway: same-lane bumper gap / own speed at CAV steps; medians at model braking "
        "(`a < -1`) and hard braking (`a < -2.5`).",
        "* 10 s rollout: teacher-forced leaders.  A CAV is integrated for 20 x 0.5 s with the "
        "twin while all other vehicles keep their recorded positions/speeds; error is measured "
        "against the recorded CAV state at 10 s (and the acceleration command over the window).",
        "",
        "## Reproduce",
        "",
        "```",
        "set PYTHONPATH=src;.;tools",
        "python tools/native_cav_fit.py --workers 3 --epochs 40   # fit + score",
        "python tools/native_cav_fit.py --no-fit --workers 3      # score the committed weights",
        "python -m pytest tests/unit/test_c4_nativecav.py -q",
        "```",
        "",
        "## Caveats",
        "",
        "* The recorded `a` is the realized SUMO acceleration, not a commanded one; the head is "
        "a behaviour model, not a controller.",
        "* The 10 s rollout is single-CAV teacher-forced, so it measures twin fidelity rather "
        "than closed-loop traffic stability.",
        "* Only TRAIN-val worlds are used; no evaluation seed 964000-964199 and no sealed/"
        "holdout path was read.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-fit", action="store_true",
                        help="score the existing weights in the out root")
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(3, args.workers)))
    data_root = guard(args.data_root)
    out_root = guard(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    fit, ev = read_episodes(data_root)
    train, stop = split_fit(fit)
    print(f"worlds: fit {len(fit)} (train {len(train)}, stop {len(stop)}), eval {len(ev)}",
          flush=True)

    weights = out_root / "native_cav_head_v5.json"
    if args.no_fit and weights.is_file():
        payload = json.loads(weights.read_text(encoding="utf-8"))
        info = {"best_epoch": payload["fit"]["best_epoch"],
                "best_stop_rmse": payload["fit"]["best_stop_rmse"]}
    else:
        x_train, y_train = training_rows(data_root, train)
        x_stop, y_stop = training_rows(data_root, stop)
        print(f"rows: train {len(x_train)}, stop {len(x_stop)}", flush=True)
        net, info = train_head(x_train, y_train, x_stop, y_stop, args.epochs, args.seed)
        payload = export_head(net, info, weights, len(train), len(stop), len(x_train))

    from wms.control.native_cav import NativeCavHead

    head = NativeCavHead(payload, weights)
    metrics = {}
    for label, kind, cap in (("legacy_2.6", "idm", 2.6), ("legacy_1.2", "idm", 1.2),
                             ("head", "head", None)):
        print(f"scoring {label}", flush=True)
        metrics[label] = episode_metrics(data_root, ev, head, kind, cap)
        metrics[label]["rollout"] = rollout_metric(data_root, ev, head, kind, cap,
                                                   seed=args.seed)
    metrics["recorded"] = {
        "recorded": metrics["head"]["recorded"],
        "jerk": metrics["head"]["jerk"],
        "headway": metrics["head"]["headway"],
    }

    result = {
        "status": "completed",
        "data_root": str(data_root),
        "split": {"fit_worlds": len(fit), "train_worlds": len(train),
                  "stop_worlds": len(stop), "eval_worlds": len(ev)},
        "head": {"path": str(weights), "rows": payload["fit"]["rows"],
                 "best_epoch": info["best_epoch"], "best_stop_rmse": info["best_stop_rmse"]},
        "metrics": metrics,
        "elapsed_s": time.time() - started,
    }
    (out_root / "nativecav_metrics.json").write_text(json.dumps(result, indent=1),
                                                    encoding="utf-8")
    write_report(out_root / "NATIVECAV.md", result)
    print(json.dumps({"metrics": str(out_root / "nativecav_metrics.json"),
                      "report": str(out_root / "NATIVECAV.md"),
                      "weights": str(weights)}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
