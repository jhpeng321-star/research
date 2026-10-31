"""Per-driver maximum-likelihood fit of physics parameters on recorded motion.

Consumes a ``setdata_hdv_histories/1`` npz (tools/rbsetdata/extract_hdv_histories.py)
and, given a physics model function (default: SUMO 1.27.1 IDM step,
tools/rbsetdata/ref_models.py::idm_sumo_step), computes the per-driver (or
population) Gaussian likelihood of the RECORDED motion and maximises it over
the model parameters with a deterministic coordinate-descent / golden-section
search on CPU.  This is classic MLE on SUMO's own model form - no black-box
training.

Car-following loss (velocity-space Gaussian):
    for consecutive ticks t, t+1 of one segment with a recorded leader at t
    and no junction transit in between, predict v_pred = fn(v[t], gap[t],
    lead_v[t], dt) and score
        -log N(v[t+1] | v_pred, sigma^2)
    sigma is maximised out in closed form (sigma = RMSE of residuals).
    gap = lead_gap_m (net bumper gap), lead_v = v[t] + lead_dv_mps.

Lane-change loss (scaffold, ref_models.lc2013_scaffold_logit): at every
eligible pre-move tick on the mainline (lane 0/1, no junction at t or t+1,
v > 3 m/s) each adjacent mainline lane is a choice with logit from the LC2013
incentive terms; the loss is multinomial cross-entropy of the recorded
decision (no change = stay option with logit 0).  Fitted by the same
coordinate descent over the scaffold weights.

Usage::

    python tools/rbsetdata/fit_likelihood.py --hist hist.npz --out params.csv \
        [--mode both|population|per_driver] [--include-cav] \
        [--min-cf-ticks 30] [--sweeps 4] [--cf-model module:fn] \
        [--summary params_summary.json]

The injected model function (``--cf-model yourmodule:your_fn``) must have the
signature ``fn(v, gap, lead_v, dt=0.5) -> v_next_pred`` (module importable
from sys.path).  CPU only, numpy only, deterministic.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from ref_models import (IDM_BOUNDS, IDM_DEFAULTS, LC_BOUNDS, LC_DEFAULTS,
                        idm_sumo_step, lc2013_scaffold_logit)

PARAM_ORDER = ("v0", "T", "a", "b", "s0")
LC_PARAM_ORDER = ("w_keep_right", "w_speed_gain", "w_gap", "bias")
# only a numerical floor (log argument); with closed-form sigma the NLL is a
# monotone function of the RMSE, so minimising it = minimising RMSE.
MIN_SIGMA_MPS = 1e-6


def load_hist(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        if str(z["schema"]) != "setdata_hdv_histories/1":
            raise ValueError(f"{path}: schema {str(z['schema'])!r} unexpected")
        return {k: z[k] for k in z.files}


def segment_slices(seg_id: np.ndarray) -> dict[int, slice]:
    """Contiguous per-segment row ranges (the extractor writes them contiguously)."""
    edges = np.flatnonzero(np.diff(seg_id)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [seg_id.size]))
    return {int(seg_id[a]): slice(a, b) for a, b in zip(starts, ends)}


def cf_samples(hist: dict, seg_sl: dict[int, slice], seg_mask: np.ndarray) -> dict:
    """Eligible car-following update rows: (v, gap, lead_v, v_next) pairs."""
    v = hist["v_mps"]
    out = {"seg": [], "i": []}
    for seg, sl in seg_sl.items():
        if not seg_mask[seg]:
            continue
        if sl.stop - sl.start < 2:
            continue
        i_all = np.arange(sl.start, sl.stop - 1)
        ok = (np.isfinite(hist["lead_gap_m"][i_all])
              & (hist["lead_gap_m"][i_all] > 0.001)
              & (hist["junction"][i_all] == 0)
              & (hist["junction"][i_all + 1] == 0))
        idx = i_all[ok]
        out["seg"].append(np.full(idx.size, seg, np.int64))
        out["i"].append(idx)
    out["seg"] = np.concatenate(out["seg"]) if out["seg"] else np.zeros(0, np.int64)
    out["i"] = np.concatenate(out["i"]) if out["i"] else np.zeros(0, np.int64)
    i = out["i"]
    out["v"] = v[i].astype(np.float64)
    out["gap"] = hist["lead_gap_m"][i].astype(np.float64)
    out["lead_v"] = (v[i] + hist["lead_dv_mps"][i]).astype(np.float64)
    out["v_next"] = v[i + 1].astype(np.float64)
    dt = (float(np.median(hist["t"][i + 1] - hist["t"][i]))
          if i.size else 0.5)
    out["dt"] = dt if dt > 0 else 0.5
    return out


def gaussian_nll(residuals: np.ndarray) -> tuple[float, float]:
    """MLE sigma NLL; returns (total_nll, sigma)."""
    n = residuals.size
    if n == 0:
        return float("nan"), float("nan")
    sigma = max(float(np.sqrt(np.mean(residuals ** 2))), MIN_SIGMA_MPS)
    return float(0.5 * n * (np.log(2.0 * np.pi * sigma * sigma) + 1.0)), sigma


def _golden_min(fn, lo: float, hi: float, iters: int = 30) -> float:
    """Golden-section minimisation of fn over [lo, hi] (fn unimodal here)."""
    invphi = 0.5 * (5.0 ** 0.5 - 1.0)
    a, b = float(lo), float(hi)
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = fn(d)
    return 0.5 * (a + b)


def _coordinate_descent(state: dict, keys: tuple, bounds: dict, objective,
                        sweeps: int, grid: int = 12) -> dict:
    """Deterministic robust CD: per coordinate, a coarse grid scan (robust to
    the multi-valley IDM likelihood) then golden refinement in the best
    bracket; an update is kept only if the objective improved."""
    state = dict(state)
    best = objective(state)
    for _ in range(int(sweeps)):
        improved = False
        for p in keys:
            lo, hi = (float(x) for x in bounds[p])
            xs = list(np.linspace(lo, hi, grid))
            vals = [objective({**state, p: float(x)}) for x in xs]
            j = int(np.argmin(vals))
            span = (xs[j + 1] - xs[j - 1]) / 2.0 if 0 < j < grid - 1 else (hi - lo) / grid
            blo = max(lo, xs[j] - span)
            bhi = min(hi, xs[j] + span)
            if bhi > blo:
                x = float(np.clip(_golden_min(
                    lambda x_, p=p, state=state: objective({**state, p: float(x_)}),
                    blo, bhi), lo, hi))
            else:
                x = xs[j]
            trial = {**state, p: x}
            val = objective(trial)
            if val < best - 1e-12:
                state, best, improved = trial, val, True
        if not improved:
            break
    return state


def fit_cf(samples: dict, cf_fn, sweeps: int = 4,
           fixed: dict | None = None) -> tuple[dict, float, float]:
    """Coordinate-descent MLE of {v0,T,a,b,s0}; returns (params, nll, sigma).

    ``cf_fn`` must accept numpy arrays elementwise:
    ``v_next_pred = cf_fn(v, gap, lead_v, dt, **params)``.
    """
    if samples["v"].size == 0:
        return {p: float("nan") for p in PARAM_ORDER}, float("nan"), float("nan")
    fixed = fixed or {}
    start = {p: float(fixed.get(p, IDM_DEFAULTS[p])) for p in PARAM_ORDER}

    def nll_of(params: dict) -> float:
        v_pred = cf_fn(samples["v"], samples["gap"], samples["lead_v"],
                       samples["dt"], **params)
        nll, _ = gaussian_nll(samples["v_next"] - np.asarray(v_pred, np.float64))
        return nll

    free = tuple(p for p in PARAM_ORDER if p not in fixed)
    bounds = {p: IDM_BOUNDS[p] for p in free}
    state = _coordinate_descent(start, free, bounds, nll_of, sweeps)
    v_pred = cf_fn(samples["v"], samples["gap"], samples["lead_v"],
                   samples["dt"], **state)
    nll, sigma = gaussian_nll(samples["v_next"] - np.asarray(v_pred, np.float64))
    return state, nll, sigma


def lc_samples(hist: dict, seg_sl: dict[int, slice], seg_mask: np.ndarray) -> dict:
    """Multinomial LC decisions per eligible pre-move tick (mainline only)."""
    v, lane, junction = hist["v_mps"], hist["eff_lane"], hist["junction"]
    lc_by_seg: dict[int, list[tuple[int, int]]] = {}
    for seg, to, t in zip(hist["lc_seg_id"], hist["lc_to_lane"], hist["lc_t"]):
        lc_by_seg.setdefault(int(seg), []).append((float(t), int(to)))
    rows = []
    for seg, sl in seg_sl.items():
        if not seg_mask[seg]:
            continue
        events = dict(lc_by_seg.get(seg, []))
        for i in range(sl.start, sl.stop - 1):
            if junction[i] or junction[i + 1] or v[i] <= 3.0:
                continue
            from_lane = int(lane[i])
            if from_lane not in (0, 1) or int(lane[i + 1]) < 0:
                continue
            targets = [l for l in (0, 1) if l != from_lane]
            if not targets:
                continue
            feat = {}
            for tgt in targets:
                key = "right" if tgt > from_lane else "left"
                lead_gap = hist[f"{key}_lead_gap_m"][i]
                fol_gap = hist[f"{key}_follow_gap_m"][i]
                lead_dv = hist[f"{key}_lead_dv_mps"][i]
                if not (np.isfinite(lead_gap) and np.isfinite(fol_gap)):
                    feat = None
                    break
                feat[tgt] = (float(lead_dv), float(lead_gap), 0.0,
                             float(fol_gap))
            if not feat:
                continue
            nxt = events.get(float(hist["t"][i + 1]))
            y = nxt if nxt in feat else -1  # -1 = stayed
            rows.append((seg, i, y, feat))
    return {"rows": rows}


def lc_cross_entropy(params: dict, rows: list) -> float:
    """Mean multinomial CE over decisions (stay = logit 0)."""
    total = 0.0
    for _seg, _i, y, feat in rows:
        logits = [0.0] + [lc2013_scaffold_logit(f[0], f[1], f[2], f[3],
                                                int(tgt == 1), params)
                          for tgt, f in feat.items()]
        m = max(logits)
        exps = [np.exp(l - m) for l in logits]
        z = sum(exps)
        idx = 0 if y == -1 else 1 + sorted(feat).index(y)
        total -= np.log(exps[idx] / z)
    return total / max(len(rows), 1)


def fit_lc(rows: list, sweeps: int = 3) -> tuple[dict, float]:
    state = {p: float(LC_DEFAULTS[p]) for p in LC_PARAM_ORDER}
    if not rows:
        return {p: float("nan") for p in LC_PARAM_ORDER}, float("nan")

    def objective(params: dict) -> float:
        return lc_cross_entropy(params, rows)

    state = _coordinate_descent(state, LC_PARAM_ORDER, LC_BOUNDS, objective, sweeps)
    return state, lc_cross_entropy(state, rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hist", required=True)
    ap.add_argument("--out", required=True, help="per-driver params CSV")
    ap.add_argument("--summary", default=None, help="JSON summary path")
    ap.add_argument("--mode", choices=("both", "population", "per_driver"),
                    default="both")
    ap.add_argument("--include-cav", action="store_true")
    ap.add_argument("--min-cf-ticks", type=int, default=30)
    ap.add_argument("--sweeps", type=int, default=4)
    ap.add_argument("--cf-model", default="ref_models:idm_default_fn",
                    help="module:fn with signature fn(v, gap, lead_v, dt, **params)")
    args = ap.parse_args()

    mod_name, fn_name = args.cf_model.split(":")
    cf_fn = getattr(importlib.import_module(mod_name), fn_name)

    hist = load_hist(Path(args.hist))
    # the flat tick table is float32; segment/event ids are integers by construction
    hist["seg_id"] = hist["seg_id"].astype(np.int64)
    hist["lc_seg_id"] = hist["lc_seg_id"].astype(np.int64)
    hist["mg_seg_id"] = hist["mg_seg_id"].astype(np.int64)
    seg_sl = segment_slices(hist["seg_id"])
    seg_mask = (~hist["is_cav"].astype(bool) if not args.include_cav
                else np.ones(hist["is_cav"].size, bool))

    population = cf_samples(hist, seg_sl, seg_mask)
    pop_params, pop_nll, pop_sigma = fit_cf(population, cf_fn, args.sweeps)
    print(f"population: {population['v'].size} CF ticks, "
          f"NLL={pop_nll:.1f}, sigma={pop_sigma:.3f} m/s, "
          f"params={ {k: round(v, 3) for k, v in pop_params.items()} }")

    lc_rows_all = lc_samples(hist, seg_sl, seg_mask)["rows"]
    lc_params, lc_ce = fit_lc(lc_rows_all, sweeps=3)
    print(f"LC scaffold: {len(lc_rows_all)} decisions, CE={lc_ce:.4f}, "
          f"params={ {k: round(v, 3) for k, v in lc_params.items()} }")

    header = ("mode,seg_id,veh_id,world,is_cav,n_cf_ticks,n_lc,"
              + ",".join(PARAM_ORDER) + ",sigma_mps,mean_nll,"
              + ",".join(LC_PARAM_ORDER) + ",lc_ce,n_lc_decisions")
    lines = [header]
    lines.append(",".join(["population", "", "", "", "",
                           str(population["v"].size), ""]
                          + [f"{pop_params[p]:.4f}" for p in PARAM_ORDER]
                          + [f"{pop_sigma:.4f}",
                             f"{pop_nll / max(population['v'].size, 1):.4f}"]
                          + [f"{lc_params[p]:.4f}" for p in LC_PARAM_ORDER]
                          + [f"{lc_ce:.4f}", str(len(lc_rows_all))]))

    if args.mode in ("both", "per_driver"):
        for seg in sorted(seg_sl):
            if not seg_mask[seg]:
                continue
            sl = seg_sl[seg]
            sub = cf_samples(hist, {seg: sl}, np.ones(hist["seg_id"].max() + 1, bool))
            if sub["v"].size == 0:
                continue
            if sub["v"].size < args.min_cf_ticks:
                params = {p: float("nan") for p in PARAM_ORDER}
                nll, sigma = float("nan"), float("nan")
                mean_nll = float("nan")
            else:
                params, nll, sigma = fit_cf(sub, cf_fn, args.sweeps)
                mean_nll = nll / sub["v"].size
            rows = [r for r in lc_rows_all if r[0] == seg]
            if len(rows) >= 8:
                lp, lce = fit_lc(rows, sweeps=2)
            else:
                lp = {p: float("nan") for p in LC_PARAM_ORDER}
                lce = float("nan")
            lines.append(",".join([
                "per_driver", str(seg), str(hist["veh_id"][seg]),
                str(hist["world_id"][hist["world"][seg]]),
                str(hist["is_cav"][seg]), str(sub["v"].size),
                str(hist["n_lc"][seg])]
                + [f"{params[p]:.4f}" for p in PARAM_ORDER]
                + [f"{sigma:.4f}", f"{mean_nll:.4f}"]
                + [f"{lp[p]:.4f}" for p in LC_PARAM_ORDER]
                + [f"{lce:.4f}" if lce == lce else "", str(len(rows))]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_path = (Path(args.summary) if args.summary
                    else out.with_name(out.stem + "_summary.json"))
    summary_path.write_text(json.dumps({
        "hist": str(args.hist), "cf_model": args.cf_model,
        "mode": args.mode, "include_cav": bool(args.include_cav),
        "population": {**pop_params, "sigma_mps": pop_sigma,
                       "nll": pop_nll, "n_cf_ticks": int(population["v"].size)},
        "lc_scaffold": {**lc_params, "cross_entropy": lc_ce,
                        "n_decisions": len(lc_rows_all)},
    }, indent=1), encoding="utf-8")
    print(f"wrote {out} and {summary_path}")


if __name__ == "__main__":
    main()
