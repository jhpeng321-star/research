"""Observable-only empirical-Bayes SUMO-IDM calibration; no WM weights trained.

Run on an existing NumPy/SciPy environment. The only dataset reads are the six
allowlisted observable files in explicitly resolved native FIT episodes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

NAMES = ("v0", "time_headway", "accel", "decel", "min_gap")
FILES = ("episode.json", "time_s.npy", "veh_kin.npy", "veh_mask.npy", "is_cav.npy", "vid.npy")
FIT_WORLDS = tuple(range(969400, 969432))
# Optimizer initial point only, not adopted driver settings or a prior mean.
INITIAL = np.array([31.29, 1.0, 2.6, 4.5, 2.5])
EPS = 1e-7  # SUMO NUMERICAL_EPS; no behavioral speed/gap threshold.


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def native_episode(root: Path, world: int) -> Path:
    if world not in FIT_WORLDS:
        raise ValueError("Only registered FIT worlds 969400-969431 are permitted")
    # Never open split files, assignment files, or arbitrary episode globs.
    manifest = root / f"world_{world}.json"
    entry = json.loads(manifest.read_text())["native"]["episode"]
    if not entry.startswith(f"w{world}m") or not entry.endswith("__native"):
        raise ValueError("Unexpected native episode identity")
    if "/" in entry or "\\" in entry or ".." in entry:
        raise ValueError("Invalid episode name")
    return root / "episodes" / entry


def read_episode(path: Path, world: int):
    meta = json.loads((path / "episode.json").read_text())
    if meta["seed"] != world or meta["arm"] != "native":
        raise ValueError("Native FIT episode identity mismatch")
    arrays = {n: np.load(path / f"{n}.npy", mmap_mode="r", allow_pickle=False)
              for n in ("time_s", "veh_kin", "veh_mask", "is_cav", "vid")}
    hashes = {n: hashlib.sha256((path / n).read_bytes()).hexdigest() for n in FILES}
    return arrays, {"world": world, "episode": path.name, "sha256": hashes}


def extract_rows(a, start=240.0, stop=360.0):
    """Rows: time, VID, x, v, next_v, bumper_gap, leader_v, leader_ID, lane, dt.

    Mainline only: ramp/junction braking is not identifiable from same-lane IDM
    alone. Actual stopped vehicles are included. Stable leaders are required at
    BOTH ends, avoiding a cut-in's newly recorded leader being used retroactively.
    """
    kin, present, vid = a["veh_kin"], a["veh_mask"].astype(bool), a["vid"]
    times, cav = a["time_s"], a["is_cav"].astype(bool)
    rows = []
    counts = {"intervals": 0, "stable_mainline_hdv": 0, "accepted": 0}
    history = {}
    for ti in np.flatnonzero((times[:-1] >= start) & (times[1:] <= stop)):
        k, kn = np.asarray(kin[ti]), np.asarray(kin[ti + 1])
        lane = np.where(k[:, 1:4].sum(-1) == 1, k[:, 1:4].argmax(-1), -1)
        ln = np.where(kn[:, 1:4].sum(-1) == 1, kn[:, 1:4].argmax(-1), -1)
        leaders = []
        for kj, lj, mj in ((k, lane, present[ti]), (kn, ln, present[ti + 1])):
            ds = kj[None, :, 0] - kj[:, None, 0]
            mask = mj[None, :] & (lj[None, :] == lj[:, None]) & (ds > 0)
            distances = np.where(mask, ds, np.inf)
            index = distances.argmin(-1)
            has = np.isfinite(distances.min(-1))
            leaders.append((index, has))
        il, has = leaders[0]
        iln, hasn = leaders[1]
        lid = np.where(has, vid[ti, il], -1)
        lidn = np.where(hasn, vid[ti + 1, iln], -1)
        dt = float(times[ti + 1] - times[ti])
        ego = (present[ti] & present[ti + 1] & ~cav[ti] & ~cav[ti + 1]
               & (vid[ti] >= 0) & (vid[ti] == vid[ti + 1]))
        if times[ti + 1] <= 300:
            for idx in np.flatnonzero(ego):
                history[int(vid[ti, idx])] = history.get(int(vid[ti, idx]), 0) + dt
        ego &= (lane >= 0) & (lane <= 1) & (ln == lane)
        counts["intervals"] += int(present[ti].sum())
        counts["stable_mainline_hdv"] += int(ego.sum())
        gap = np.where(has, k[il, 0] - k[:, 0] - k[il, 6], np.inf)
        valid = (ego & (lid == lidn) & (has == hasn) & (gap >= 0)
                 & np.isfinite(k).all(-1) & np.isfinite(kn).all(-1)
                 & (k[:, 4] >= 0) & (kn[:, 4] >= 0) & (k[:, 6] > 0))
        for idx in np.flatnonzero(valid):
            rows.append((times[ti], vid[ti, idx], k[idx, 0], k[idx, 4], kn[idx, 4],
                         gap[idx], k[il[idx], 4] if has[idx] else 0, lid[idx], lane[idx], dt))
        counts["accepted"] += int(valid.sum())
    return np.asarray(rows, dtype=np.float64).reshape(-1, 10), history, counts


def next_speed(v, gap, vl, params, dt=0.5):
    """SUMO 1.27.1 IDM::_v, source lines 225-268; physical bumper gap input.

    The SUMO API subtracts minGap and _v adds it back; our recorded physical gap
    requires neither operation. Includes the independent freeSpeed bound and
    base CF acceleration/deceleration envelope, but not LC/junction messages.
    """
    p = np.asarray(params)
    vmax, tau, accel, decel, s0 = np.moveaxis(p, -1, 0)
    v = np.asarray(v, dtype=float)
    speed = v.copy()
    distance = np.array(gap, dtype=float, copy=True)
    n = max(1, int(float(dt) / 0.25 + 0.5))
    h = dt / n
    # IDM freeSpeed calls _v with gap=1e6, respectMinGap=false. The
    # interaction is negligible except at extreme optimizer trial settings.
    free = v.copy()
    fg = np.full_like(v, 1e6)
    for _ in range(n):
        desired = s0 + np.maximum(0, speed * tau + speed * (speed - vl)
                                 / (2 * np.sqrt(accel * decel)))
        aa = accel * (1 - (speed / np.maximum(EPS, vmax)) ** 4
                      - (desired / np.maximum(EPS, distance)) ** 2)
        speed = np.maximum(0, speed + h * aa)
        distance -= np.maximum(0, speed - vl) * h
        fs = np.maximum(0, free * tau + free * (free - vmax)
                        / (2 * np.sqrt(accel * decel)))
        fa = accel * (1 - (free / np.maximum(EPS, vmax)) ** 4 - (fs / fg) ** 2)
        free = np.maximum(0, free + h * fa)
        fg = np.maximum(EPS, fg - np.maximum(0, free - vmax) * h)
    # MSCFModel::finalizeSpeed permits emergency braking when the follow speed
    # demands it. SUMO default emergencyDecel=9 is physical, not fitted here.
    lower = np.minimum(np.maximum(0, v - np.maximum(decel, 1.5) * dt),
                       np.maximum(np.minimum(speed, free), np.maximum(0, v - 9 * dt)))
    upper = np.minimum(v + accel * dt, np.minimum(speed, free))
    return np.maximum(lower, upper)


def prediction(rows, logp):
    return (next_speed(rows[:, 3], rows[:, 5], rows[:, 6], np.exp(logp), rows[0, 9])
            - rows[:, 3]) / rows[:, 9]


def observed(rows):
    return (rows[:, 4] - rows[:, 3]) / rows[:, 9]


def posterior_job(job):
    key, rows, mu, var, sigma = job
    std = np.sqrt(var)
    y = observed(rows)

    def residual(z):
        return np.concatenate(((prediction(rows, z) - y) / sigma, (z - mu) / std))

    result = least_squares(residual, mu, bounds=(-10, 10), max_nfev=100,
                           ftol=1e-7, xtol=1e-7, gtol=1e-7)
    covariance = np.linalg.inv(result.jac.T @ result.jac)
    jd = result.jac[:-len(mu)]
    # Expected SSE under local Gaussian posterior, for residual-noise M-step.
    expected_sse = np.square(prediction(rows, result.x) - y).sum()
    expected_sse += sigma ** 2 * np.einsum("ni,ij,nj->", jd, covariance, jd)
    return key, result.x, covariance, float(expected_sse), len(rows), result.success


def fit_population(groups, workers=4, iterations=20):
    pooled = np.concatenate(list(groups.values()))
    # Deterministic evenly spaced numerical subsample for the INITIAL estimate.
    init_rows = pooled[np.linspace(0, len(pooled) - 1, min(len(pooled), 30000)).astype(int)]
    initial = least_squares(lambda z: prediction(init_rows, z) - observed(init_rows),
                            np.log(INITIAL), bounds=(-10, 10), max_nfev=200)
    mu = initial.x
    var = np.full(len(mu), 0.25)  # numerical EM initialization only; all updated.
    sigma = max(1e-6, float(np.sqrt(np.mean(np.square(
        prediction(init_rows, mu) - observed(init_rows))))))
    trace = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for iteration in range(iterations):
            jobs = ((key, rows, mu, var, sigma) for key, rows in groups.items())
            post = list(executor.map(posterior_job, jobs, chunksize=16))
            means = np.stack([r[1] for r in post])
            covs = np.stack([r[2] for r in post])
            new_mu = means.mean(0)
            new_var = np.mean(np.square(means - new_mu) + np.diagonal(covs, axis1=1, axis2=2), 0)
            new_sigma = np.sqrt(sum(r[3] for r in post) / sum(r[4] for r in post))
            change = max(float(np.max(abs(new_mu - mu))),
                         float(np.max(abs(np.log(new_var / var)))),
                         float(abs(np.log(new_sigma / sigma))))
            trace.append({"iteration": iteration, "median": dict(zip(NAMES, np.exp(new_mu).tolist())),
                          "log_std": np.sqrt(new_var).tolist(), "sigma_accel": float(new_sigma),
                          "max_log_change": change, "optimizer_success": sum(r[5] for r in post)})
            print(json.dumps(trace[-1]), flush=True)
            mu, var, sigma = new_mu, np.maximum(new_var, 1e-10), max(1e-6, new_sigma)
            if change < 1e-3:
                break
        # Final E-step must use the EXPORTED hyperparameters.
        post = list(executor.map(posterior_job,
                    ((k, r, mu, var, sigma) for k, r in groups.items()), chunksize=16))
    return mu, var, sigma, post, trace, initial.x


def summarize(errors):
    e = np.asarray(errors)
    if not e.size:
        return {"n": 0}
    return {"n": int(e.size), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(abs(e))), "bias": float(np.mean(e))}


def validate(world_rows, posterior, mu, pooled_mu):
    acc = {name: [] for name in ("initial_defaults", "pooled_fit", "population", "posterior")}
    rollout = {str(h): {name: {"position_m": [], "speed_mps": []} for name in
                         ("constant_speed", "pooled_fit", "population", "posterior")}
               for h in (5, 10)}
    world_metrics = {}
    for world, rows in world_rows.items():
        valid = rows[(rows[:, 0] >= 300) & (rows[:, 0] < 360)]
        errs = {name: [] for name in acc}
        for identity in np.unique(valid[:, 1]).astype(int):
            rr = valid[valid[:, 1] == identity]
            pp = posterior.get(f"{world}:{identity}", mu)
            for name, z in (("initial_defaults", np.log(INITIAL)), ("pooled_fit", pooled_mu),
                            ("population", mu), ("posterior", pp)):
                errs[name].extend((prediction(rr, z) - observed(rr)).tolist())
            # Ten-second scheduled starts, full observed stable-leader windows.
            for start in range(300, 351, 10):
                for horizon in (5, 10):
                    segment = rr[(rr[:, 0] >= start) & (rr[:, 0] < start + horizon)]
                    if (len(segment) != int(horizon / 0.5)
                            or not np.allclose(segment[:, 0], np.arange(start, start + horizon, 0.5))
                            or len(np.unique(segment[:, 7])) != 1
                            or len(np.unique(segment[:, 8])) != 1):
                        continue
                    # Last endpoint x = x_last + v_next*dt under SUMO Euler.
                    target_x = segment[-1, 2] + segment[-1, 4] * segment[-1, 9]
                    target_v = segment[-1, 4]
                    for name, z in (("constant_speed", None), ("pooled_fit", pooled_mu),
                                    ("population", mu), ("posterior", pp)):
                        x, v = segment[0, 2:4]
                        for row in segment:
                            if z is not None:
                                gap = row[5] + row[2] - x
                                v = float(next_speed(v, gap, row[6], np.exp(z), row[9]))
                            x += v * row[9]
                        result = rollout[str(horizon)][name]
                        result["position_m"].append(x - target_x)
                        result["speed_mps"].append(v - target_v)
        world_metrics[str(world)] = {name: summarize(e) for name, e in errs.items()}
        for name in acc:
            acc[name].extend(errs[name])
    return {"contract": "Later native FIT [300,360); recorded stable leaders/lanes; no future parameter inference",
            "acceleration_mps2": {name: summarize(e) for name, e in acc.items()},
            "teacher_forced_leader_rollouts": {h: {name: {metric: summarize(e) for metric, e in d.items()}
                for name, d in models.items()} for h, models in rollout.items()},
            "per_world_acceleration_mps2": world_metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--worlds", type=int, nargs="+", default=list(FIT_WORLDS))
    args = parser.parse_args()
    if not set(args.worlds) <= set(FIT_WORLDS):
        raise ValueError("Non-FIT world rejected before any data read")
    if not args.out.parent.is_dir():
        raise ValueError("Output parent must already exist")
    args.out.mkdir(exist_ok=False)
    started = time.time()
    groups, world_rows, history, provenance, extraction = {}, {}, {}, [], {}
    for world in args.worlds:
        a, source = read_episode(native_episode(args.root, world), world)
        rows, hist, counts = extract_rows(a)
        world_rows[world] = rows
        provenance.append(source)
        extraction[str(world)] = counts
        for identity, duration in hist.items():
            key = f"{world}:{identity}"
            history[key] = duration
            rr = rows[(rows[:, 1] == identity) & (rows[:, 0] < 300)]
            if len(rr):
                groups[key] = rr
        print(f"extracted {world}: {len(rows)} rows", flush=True)
    write_json(args.out / "provenance.json", {"sources": provenance, "extraction": extraction,
               "allowed_files": list(FILES), "fit_interval_s": [240, 300],
               "worlds": args.worlds, "arm": "native"})
    mu, var, sigma, post, trace, pooled_mu = fit_population(groups, args.workers, args.iterations)
    population = {"schema": "wm_phys.idm_population.v1", "parameter_order": list(NAMES),
                  "distribution": "diagonal_lognormal", "log_mean": mu.tolist(),
                  "log_cov": np.diag(var).tolist(), "median": dict(zip(NAMES, np.exp(mu).tolist())),
                  "mean": dict(zip(NAMES, np.exp(mu + var / 2).tolist())),
                  "delta": 4, "sigma_accel_mps2": float(sigma),
                  "n_drivers": len(groups), "n_observations": sum(len(r) for r in groups.values()),
                  "fit_worlds": args.worlds, "fit_interval_s": [240, 300],
                  "speedFactor_mapping": "v0 / current lane speed limit in m/s; do not multiply v0 twice",
                  "likelihood": "Gaussian one-step effective acceleration; Laplace approximate marginal ML EM",
                  "em_trace": trace, "pooled_fit_median": dict(zip(NAMES, np.exp(pooled_mu).tolist()))}
    drivers = {}
    for key, z, covariance, _, n, success in post:
        drivers[key] = {"log_mean": z.tolist(), "log_cov": covariance.tolist(),
                        "median": dict(zip(NAMES, np.exp(z).tolist())), "delta": 4,
                        "n_observations": n, "observed_history_s": history[key],
                        "optimizer_success": bool(success)}
    for key, duration in history.items():
        if key not in drivers:
            drivers[key] = {"log_mean": mu.tolist(), "log_cov": np.diag(var).tolist(),
                            "median": population["median"], "delta": 4, "n_observations": 0,
                            "observed_history_s": duration, "optimizer_success": True}
    write_json(args.out / "idm_population.json", population)
    write_json(args.out / "idm_posteriors.json", {"schema": "wm_phys.idm_posteriors.v1",
               "parameter_order": list(NAMES), "conditioning_end_s": 300,
               "identity_key": "world:observable_vid", "fallback": "idm_population.json",
               "drivers": drivers})
    result = validate(world_rows, {k: v["log_mean"] for k, v in drivers.items()}, mu, pooled_mu)
    result["elapsed_s"] = time.time() - started
    write_json(args.out / "validation.json", result)
    print(json.dumps(result["acceleration_mps2"]), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
