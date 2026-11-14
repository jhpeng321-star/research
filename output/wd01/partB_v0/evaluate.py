"""V0 evaluation on CALIBRATION (PLAN_PARTB_V0_TRAINING.md items 1-5).

1. Calibration and sharpness per tau, per chunk, switching-mix and
   damping-fleet rows separately: CRPS of the ensemble mixture vs the
   TRAIN climatology (per tau x policy x demand empirical; falls back to
   tau x policy then tau when a group is thin), randomized-PIT, central
   10-90% coverage.
2. TD consistency: residual V(s_t, tau) - [D[t,t+30] + V(s_{t+30},
   tau-30)] on complete CAL TD tuples, per tau.
3. Policy contrast at each CAL world's t0 (the planner's quantity, CRN
   pairs): predicted dV = V(s_t0, 570, damping_fleet) - V(s_t0, 570,
   switching_mix) vs the realised base-minus-switching-arm delay
   (v0.3.2 fix #3: the campaign '__native' arms are random per-CAV
   switching schedules, NOT a fleet-native policy).  The switching arm
   has no label row at exactly t0 (its decision times start at
   t0 + 10 s), so the realised switching side is the tau=570 label at
   t0 + 10 s -- a 10 s window offset that joins the plan's
   single-realisation noise caveat.
4. Safety heads, diagnostic only (G-1/G-2 gated): P_coll Brier score and
   TTC<3 exposure CRPS vs climatology.
5. Planner-call speed: predictions per second for 50 plans x M5 x K16.
6. PERMANENT baselines (B3a, item D15): climatology / ridge / point-GBM
   on the diagnosis summary features, fitted on TRAIN, MAE(median) and
   CRPS per tau x policy x chunk -- for BOTH value models.
7. HARD acceptance per tau (B3a, item D16): the evaluated model must beat
   RIDGE on BOTH MAE(median) and CRPS, or the report says exactly where
   it does not.  Report-only in B3a; selection is TRAIN OOB only
   (v0.3.2 fix #12 removed the CAL-based selector).

Every eval_report states the IDEAL SENSING PREMISE up front (v0.3.2 fix
#6): state features come from the full true state; the reconstruction
gap is measured separately (RG-0).

The evaluated value model is either the net member MIXTURE (twohot pmfs,
mean probability per bin) or the calibrated quantile-GBM (gbm.py,
quantile-grid predictive); both answer the same predictor interface.
Numbers from a --smoke run are smoke-only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import POLICY_OF_SOURCE
from .distributions import DistMetrics, crps_empirical, spearman
from .model import (V0ValueConfig, V0ValueNet, check_token_widths,
                    policy_id)

EVAL_BS = 512
SMOKE_GROUP_CAP = 40          # CAL label rows per (chunk, source, tau)
SMOKE_TD_CAP = 60             # CAL TD rows per (tau, source)
SMOKE_SPEED_PLANS = 10        # smoke speed item: 10 plans x M x K16


# ----------------------------------------------------------------- utils --

def load_members(out_dir, device: torch.device, log=print,
                 members: int | None = None):
    """Load member_{i}.pt (+ JSON sidecars) -> [(idx, net, sidecar)].

    Strict (addendum 4c): ALL expected members must be present.  With
    `members` given the expected set is 0..members-1; without it the
    expected set is 0..max(present).  EVERY missing index is named in the
    error -- the loader never stops silently at the first gap."""
    out = Path(out_dir)
    present = sorted(int(p.stem.split("_")[1])
                     for p in out.glob("member_*.pt"))
    expected = (list(range(members)) if members is not None
                else list(range(present[-1] + 1)) if present else [])
    missing = [i for i in expected if i not in present]
    if missing or (members is not None and not present):
        raise FileNotFoundError(
            f"missing member weights in {out}: {missing} "
            f"(expected 0..{members - 1 if members else '?'}, "
            f"present {present})")
    if not expected:
        raise FileNotFoundError(f"no member_*.pt weights in {out}")
    nets = []
    for i in expected:
        w = out / f"member_{i}.pt"
        side_path = out / f"member_{i}.json"
        side = json_loads(side_path) if side_path.exists() else {}
        net = V0ValueNet(V0ValueConfig.from_json(
            side.get("cfg_model", {}))).to(device)
        state = torch.load(w, map_location=device, weights_only=True)
        check_token_widths(state, net.cfg, where=f"member_{i}.pt")
        net.load_state_dict(state)
        net.eval()
        nets.append((i, net, side))
    log(f"eval: loaded {len(nets)} members from {out}")
    return nets


def json_loads(path: Path) -> dict:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def _softmax_np(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits.detach().float().cpu(), dim=-1).numpy()


@torch.no_grad()
def forward_mixture(nets, ds, row_idx, head: str = "v_delay",
                    bs: int = EVAL_BS, collect_members: bool = False):
    """Mixture pmf over label rows (optionally the per-member pmfs)."""
    row_idx = np.asarray(row_idx, dtype=np.int64)
    mix, members = None, []
    for _, net, _ in nets:
        probs = []
        for s in range(0, len(row_idx), bs):
            out = net(ds.gather(row_idx[s:s + bs]))
            probs.append(_softmax_np(out[head]))
        p = np.concatenate(probs, axis=0)
        members.append(p)
        mix = p if mix is None else mix + p
    mix = mix / len(nets)
    return (mix, np.stack(members)) if collect_members else mix


@torch.no_grad()
def mixture_point(nets, ds, g_idx, tau, policy, pre_fix,
                  head: str = "v_delay", bs: int = EVAL_BS) -> np.ndarray:
    """Mixture posterior mean at explicit (global positions, tau, policy)."""
    g_idx = np.asarray(g_idx, dtype=np.int64)
    tau = np.broadcast_to(np.asarray(tau, np.float32), g_idx.shape)
    policy = np.broadcast_to(np.asarray(policy, np.int64), g_idx.shape)
    pre_fix = np.broadcast_to(np.asarray(pre_fix, np.float32), g_idx.shape)
    acc = np.zeros(len(g_idx))
    for _, net, _ in nets:
        vals = []
        for s in range(0, len(g_idx), bs):
            batch = ds.gather_global(g_idx[s:s + bs], tau[s:s + bs],
                                     policy[s:s + bs], pre_fix[s:s + bs])
            out = net(batch)
            pmf = _softmax_np(out[head])
            vals.append(pmf @ met_centers())
        acc += np.concatenate(vals)
    return acc / len(nets)


_MET = None


def met_centers() -> np.ndarray:
    global _MET
    if _MET is None:
        _MET = DistMetrics()
    return _MET.centers


def met() -> DistMetrics:
    global _MET
    if _MET is None:
        _MET = DistMetrics()
    return _MET


@torch.no_grad()
def member_crps_subset(net, ds, tau: float, policy: str,
                       col: str = "total_delay_ref_s") -> float:
    """Mean CRPS of ONE member on CAL rows of (tau, policy).  NOT a
    selection signal any more: v0.3.2 fix #12 removed the CAL-based
    selector (train.oob_crps300 selects on TRAIN world-grouped OOB)."""
    r = ds.rows
    sel = r[(r["split"] == "CALIBRATION") & (np.isclose(r["tau"], tau))
            & (r["policy"] == policy)]
    if len(sel) == 0:
        return float("nan")
    idx = sel.index.to_numpy()
    probs = []
    for s in range(0, len(idx), EVAL_BS):
        out = net(ds.gather(idx[s:s + EVAL_BS]))
        probs.append(_softmax_np(out["v_delay"]))
    probs = np.concatenate(probs, axis=0)
    return float(np.mean(met().crps(probs, r.loc[idx, col].to_numpy())))


def subset_rows(ds, cap: int) -> np.ndarray:
    """Smoke subset: first `cap` CAL rows per (chunk, source, tau)."""
    r = ds.rows
    cal = r[r["split"] == "CALIBRATION"]
    keep = [g.index.to_numpy()[:cap]
            for _, g in cal.groupby(["chunk_id", "source", "tau"],
                                    sort=True)]
    return np.sort(np.concatenate(keep)) if keep else np.array([], int)


# ----------------------------------------------------------- climatology --

class Climatology:
    """TRAIN empirical distributions per tau x policy x demand (with
    fallback to tau x policy, then tau, when a group has < min_n rows)."""

    LEVELS = ("tpd", "tp", "t", "thin")

    def __init__(self, train_rows, col: str, min_n: int = 5):
        self.col = col
        self.min_n = min_n
        self.g: dict = {}
        for tau, tg in train_rows.groupby("tau"):
            self._add((float(tau),), tg)
            for pol, pg in tg.groupby("policy"):
                self._add((float(tau), pol), pg)
                for dem, dg in pg.groupby("demand_veh_h"):
                    self._add((float(tau), pol, int(dem)), dg)

    def _add(self, key, frame):
        self.g[key] = frame[self.col].to_numpy()

    def sample(self, tau, policy, demand):
        """(samples, level) with the documented fallback order."""
        keys = ((float(tau), policy, int(demand)),
                (float(tau), policy), (float(tau),))
        for key, lv in zip(keys, self.LEVELS):
            s = self.g.get(key)
            if s is not None and len(s) >= self.min_n:
                return s, lv
        s = self.g.get((float(tau),))
        return (s, "thin") if s is not None else (None, "none")

    def crps(self, y, tau, policy, demand):
        """Per-row CRPS + level used; y/tau/demand aligned arrays."""
        y = np.asarray(y, float)
        out = np.full(len(y), np.nan)
        level = np.full(len(y), "none", dtype=object)
        keys: dict = {}
        for i, (t, p, d) in enumerate(zip(tau, policy, demand)):
            keys.setdefault((float(t), p, int(d)), []).append(i)
        for key, rows in keys.items():
            s, lv = self.sample(key[0], key[1], key[2])
            if s is None:
                continue
            out[rows] = crps_empirical(s, y[rows])
            level[rows] = lv
        return out, level


# ------------------------------------------------------------------- run --

@dataclass
class EvalConfig:
    out_dir: str
    device: str = "auto"
    smoke: bool = False
    speed_plans: int = 50
    speed_taus: int = 16
    value_model: str = "net"        # "net" (member mixture) | "gbm" (B3a)
    members: int | None = None      # expected members (strict load_members)
    feature_set: str = "oracle"     # "oracle" | "deployable" (v0.3.1)
    compare_metrics: str | None = None   # oracle metrics.json for the
                                         # with-vs-without-schedule table


# -------------------------------------------------------------- predictor --

class _NetDist:
    """Twohot-pmf views (delay or exposure head) for the net mixture."""

    def __init__(self, probs: np.ndarray):
        self.probs = probs

    def median(self) -> np.ndarray:
        return met().quantile(self.probs, 0.5)

    def crps(self, y: np.ndarray) -> np.ndarray:
        return met().crps(self.probs, y)

    def pit(self, y: np.ndarray) -> np.ndarray:
        return met().pit(self.probs, y)

    def coverage(self, y: np.ndarray) -> np.ndarray:
        return met().central_coverage(self.probs, y)


class _GbmDist:
    """Quantile-grid views (gbm.py) for the GBM value model."""

    def __init__(self, model, quantiles: dict):
        self.model = model
        self.q = quantiles

    def median(self) -> np.ndarray:
        return self.q[0.5]

    def crps(self, y: np.ndarray) -> np.ndarray:
        from .gbm import crps_from_quantiles
        return crps_from_quantiles(sorted(self.q), self.q, y)

    def pit(self, y: np.ndarray) -> np.ndarray:
        from .gbm import pit_from_quantiles
        return pit_from_quantiles(sorted(self.q), self.q, y)

    def coverage(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        return (y >= self.q[0.1]) & (y <= self.q[0.9])


class _NetPredictor:
    """The B2 evaluated model: mixture of members' twohot pmfs."""

    value_model = "net"

    def __init__(self, nets):
        self.nets = nets
        self.n_members = len(nets)

    def delay_dist(self, ds, idx) -> _NetDist:
        return _NetDist(forward_mixture(self.nets, ds, idx, "v_delay"))

    def exp_dist(self, ds, idx) -> _NetDist:
        return _NetDist(forward_mixture(self.nets, ds, idx, "v_exp"))

    def point_global(self, ds, g_idx, tau, policy, pre_fix) -> np.ndarray:
        return mixture_point(self.nets, ds, g_idx, tau, policy, pre_fix)

    def point_td_pairs(self, ds, td_idx):
        b_t, b_n, _d30 = ds.gather_td(td_idx)
        return (mixture_point_from_batches(self.nets, b_t),
                mixture_point_from_batches(self.nets, b_n))

    def coll_prob(self, ds, idx) -> np.ndarray:
        vals = []
        for _, net, _ in self.nets:
            out = []
            for s in range(0, len(idx), EVAL_BS):
                o = net(ds.gather(idx[s:s + EVAL_BS]))
                out.append(torch.sigmoid(
                    o["p_coll"].detach().float().cpu()).numpy())
            vals.append(np.concatenate(out))
        return np.mean(vals, axis=0)

    def speed(self, ds, cal_idx, n_plans, k_taus, device) -> dict:
        return _speed_benchmark(self.nets, ds, cal_idx, n_plans, k_taus,
                                device)


class _GbmPredictor:
    """The B3a evaluated model: calibrated quantile-GBM (gbm_value.pkl)."""

    value_model = "gbm"

    def __init__(self, model):
        self.model = model
        self.n_members = 0

    def delay_dist(self, ds, idx) -> _GbmDist:
        return _GbmDist(self.model,
                        self.model.predict_quantiles_rows(ds, idx, "delay"))

    def exp_dist(self, ds, idx) -> _GbmDist:
        return _GbmDist(self.model,
                        self.model.predict_quantiles_rows(ds, idx, "exp"))

    def point_global(self, ds, g_idx, tau, policy, pre_fix) -> np.ndarray:
        return self.model.point_global(ds, g_idx, tau, policy, pre_fix)

    def point_td_pairs(self, ds, td_idx):
        td = ds.td.iloc[np.asarray(td_idx, dtype=np.int64)]
        pre = np.array(
            [1.0 if ds.man.entries[ds.src_names[s]]["pre_fix"] else 0.0
             for s in td["src_idx"]], dtype=np.float64)
        pol = np.array(
            [policy_id(POLICY_OF_SOURCE[
                ds.man.entries[ds.src_names[s]]["source"]])
             for s in td["src_idx"]], dtype=np.int64)
        tau = td["tau"].to_numpy().astype(np.float64)
        v_t = self.model.point_global(ds, td["t_idx"].to_numpy(), tau,
                                      pol, pre)
        v_n = self.model.point_global(ds, td["tnext_idx"].to_numpy(),
                                      tau - 30.0, pol, pre)
        return v_t, v_n

    def coll_prob(self, ds, idx) -> np.ndarray:
        return self.model.coll_prob_rows(ds, idx)

    def speed(self, ds, cal_idx, n_plans, k_taus, device) -> dict:
        idx = cal_idx[:max(1, min(n_plans, len(cal_idx)))]
        taus = np.linspace(30.0, 570.0, k_taus).astype(np.float64)
        g_idx = np.repeat(ds.rows["t_idx"].to_numpy()[idx], k_taus)
        tau = np.tile(taus, len(idx))
        pre = np.repeat(ds.rows["pre_fix"].to_numpy()[idx]
                        .astype(np.float64), k_taus)
        n_total = int(len(g_idx))
        t0 = time.perf_counter()
        self.model.point_global(ds, g_idx, tau, 0, pre)
        dt = time.perf_counter() - t0
        return dict(plans=int(len(idx)), taus=int(k_taus), members=1,
                    predictions=n_total, seconds=_r(dt, 3),
                    predictions_per_s=_r(n_total / dt))


def _expected_members(out_dir, log=print) -> int | None:
    """Member count from metrics.json's train config (None if unknown)."""
    try:
        doc = json_loads(Path(out_dir) / "metrics.json")
        n = int((doc.get("config") or {}).get("members"))
        return n if n > 0 else None
    except Exception:                                   # noqa: BLE001
        return None


def make_predictor(cfg: EvalConfig, ds, log=print):
    """Load the evaluated value model: the member mixture (net) or the
    calibrated quantile-GBM (gbm)."""
    if cfg.value_model == "gbm":
        from .gbm import GbmValueModel
        return _GbmPredictor(GbmValueModel.load(cfg.out_dir, log=log))
    from .train import resolve_device
    device = resolve_device(cfg.device)
    members = cfg.members
    if members is None:
        members = _expected_members(cfg.out_dir, log=log)
    return _NetPredictor(load_members(cfg.out_dir, device, log=log,
                                      members=members))


def evaluate_run(cfg: EvalConfig, ds, log=print) -> tuple[str, dict]:
    """Evaluate the value model in cfg.out_dir (net mixture or GBM) on
    the CALIBRATION split.

    Returns (eval_report.md text, metrics dict).  `ds` is a loaded
    V0Dataset covering TRAIN (climatology) + CALIBRATION rows.
    """
    from .train import resolve_device
    device = resolve_device(cfg.device)
    pred = make_predictor(cfg, ds, log=log)
    met_ = met()
    r = ds.rows
    train_rows = r[r["split"] == "TRAIN"]
    cal = r[r["split"] == "CALIBRATION"]
    cal_idx = subset_rows(ds, SMOKE_GROUP_CAP) if cfg.smoke \
        else cal.index.to_numpy()
    sub = r.loc[cal_idx]
    m: dict = {"n_members": pred.n_members,
               "value_model": pred.value_model,
               "feature_set": getattr(ds, "feature_set", "oracle"),
               "n_cal_rows": int(len(cal_idx)),
               "device": str(device), "smoke": bool(cfg.smoke)}

    # ---------------- item 1: CRPS vs climatology, PIT, coverage -------
    y = sub["total_delay_ref_s"].to_numpy()
    dist = pred.delay_dist(ds, cal_idx)
    crps_model = dist.crps(y)
    med_model = dist.median()
    pit = dist.pit(y)
    cov90 = dist.coverage(y)
    clim = Climatology(train_rows, "total_delay_ref_s")
    crps_clim, level = clim.crps(y, sub["tau"].to_numpy(),
                                 sub["policy"].tolist(),
                                 sub["demand_veh_h"].to_numpy())
    t1 = []
    ch = sub["chunk_id"].to_numpy()
    pol_arr = sub["policy"].to_numpy()
    tau_arr = sub["tau"].to_numpy()
    ok = np.isfinite(crps_model) & np.isfinite(crps_clim)
    for (chunk, pol, tau), g in sub.groupby(["chunk_id", "policy", "tau"]):
        gi = (ok & (ch == chunk) & (pol_arr == pol)
              & np.isclose(tau_arr, tau))
        if not gi.any():
            continue
        t1.append(dict(
            chunk=chunk, policy=pol, tau=float(tau), n=int(gi.sum()),
            mae_model=_r(np.abs(med_model[gi] - y[gi]).mean()),
            crps_model=_r(crps_model[gi].mean()),
            crps_clim=_r(crps_clim[gi].mean()),
            pit_mean=_r(pit[gi].mean(), 3),
            coverage90=_r(cov90[gi].mean(), 3),
            clim_level=str(level[gi][0])))
    m["item1_crps_pit_coverage"] = t1
    hist, _ = np.histogram(pit, bins=np.linspace(0, 1, 11))
    m["item1_pit_hist_overall"] = hist.astype(int).tolist()
    m["item1_pit_hist_per_tau"] = {
        f"{tau:.0f}": np.histogram(pit[np.isclose(tau_arr, tau)],
                                   bins=np.linspace(0, 1, 11))[0]
        .astype(int).tolist()
        for tau in sorted({float(t) for t in tau_arr})}

    # ---------------- item 2: TD residual ------------------------------
    td = ds.td
    td_ok = ((td["split"] == "CALIBRATION")
             & (td["label_mc_complete"] == 1)
             & (td["tau"] >= 60.0)
             & (td["t_idx"] >= 0) & (td["tnext_idx"] >= 0)).to_numpy()
    td_idx = np.flatnonzero(td_ok)
    if cfg.smoke:
        keep = []
        tdc = td.iloc[td_idx]
        for _, g in tdc.groupby(["tau", "source"], sort=True):
            keep.append(g.index.to_numpy()[:SMOKE_TD_CAP])
        td_idx = np.sort(np.concatenate(keep)) if keep else td_idx
    t2 = []
    if len(td_idx):
        d30 = td["d30_s"].to_numpy()[td_idx].astype(np.float64)
        v_t, v_n = pred.point_td_pairs(ds, td_idx)
        resid = v_t - (d30 + v_n)
        tds = td.iloc[td_idx]
        for (tau, pol), g in tds.assign(resid=resid).groupby(
                ["tau", "policy"]):
            x = g["resid"].to_numpy()
            t2.append(dict(
                tau=float(tau), policy=pol, n=int(len(x)),
                mean=_r(np.mean(x)), std=_r(np.std(x)),
                q05=_r(np.quantile(x, 0.05)),
                q50=_r(np.quantile(x, 0.50)),
                q95=_r(np.quantile(x, 0.95))))
    m["item2_td_residual"] = t2

    # ---------------- item 3: policy contrast at t0 --------------------
    contrast_rows, t3 = _policy_contrast(pred, ds, met_)
    m["item3_policy_contrast"] = t3
    m["item3_rows"] = contrast_rows

    # ---------------- item 4: safety heads (diagnostic) ----------------
    dist_exp = pred.exp_dist(ds, cal_idx)
    y_exp = sub["ttc_lt3_veh_s"].to_numpy()
    crps_exp = dist_exp.crps(y_exp)
    clim_exp = Climatology(train_rows, "ttc_lt3_veh_s")
    crps_exp_clim, _lv = clim_exp.crps(y_exp, sub["tau"].to_numpy(),
                                       sub["policy"].tolist(),
                                       sub["demand_veh_h"].to_numpy())
    p_coll = pred.coll_prob(ds, cal_idx)
    coll_y = sub["collision_any"].to_numpy().astype(float)
    t4 = []
    for tau, g in sub.assign(p=p_coll, yy=coll_y,
                             ce=crps_exp, cec=crps_exp_clim).groupby("tau"):
        t4.append(dict(
            tau=float(tau), n=int(len(g)),
            p_coll_brier=_r(((g["p"] - g["yy"]) ** 2).mean()),
            collisions=int(g["yy"].sum()),
            exp_crps_model=_r(g["ce"].mean()),
            exp_crps_clim=_r(g["cec"].mean())))
    m["item4_safety_diagnostic"] = t4

    # ---------------- item 5: planner-call speed -----------------------
    n_plans = SMOKE_SPEED_PLANS if cfg.smoke else cfg.speed_plans
    k_taus = cfg.speed_taus
    t5 = pred.speed(ds, cal_idx, n_plans, k_taus, device)
    m["item5_speed"] = t5

    # ------------- items 6-7: baselines + HARD acceptance (B3a) --------
    if len(cal_idx):
        from .gbm import baselines_section, oob_acceptance_section
        m["item6_baselines"] = baselines_section(
            ds, cal_idx, med_model, crps_model, log=log,
            gbm_max_iter=40 if cfg.smoke else 300)
        # item 7b: the same acceptance on TRAIN-internal world-grouped
        # OOB predictions (gbm models store them at fit time)
        mo = getattr(pred, "model", None)
        if getattr(mo, "oof_delay", None) and \
                getattr(mo, "oof_grouping", "") == "world-grouped":
            log("eval: computing TRAIN-internal world-grouped OOB "
                "HARD acceptance")
            m["item7b_oob_acceptance"] = oob_acceptance_section(
                ds, mo, log=log)
        else:
            m["item7b_oob_acceptance"] = dict(
                acceptance=[],
                note=("not available for this value model (needs the gbm "
                      "fit's world-grouped OOF delay predictions)"),
                overall=dict(hard_pass=False, n_taus=0, n_pass=0))
    else:
        m["item6_baselines"] = dict(
            rows=[], acceptance=[], approx="",
            overall=dict(hard_pass=False, n_taus=0, n_pass=0))
        m["item7b_oob_acceptance"] = dict(
            acceptance=[], note="no CAL rows",
            overall=dict(hard_pass=False, n_taus=0, n_pass=0))
    if pred.value_model == "gbm":
        m["gbm_variant"] = getattr(pred.model, "variant", "pure")
        if hasattr(pred.model, "feature_set"):
            m["gbm_feature_set"] = pred.model.feature_set
        m["gbm_data_manifest_sha256"] = \
            getattr(pred.model, "data_manifest_sha256", "")
        if hasattr(pred.model, "tau_variants"):
            m["gbm_tau_variants"] = {f"{t:.0f}": v for t, v
                                     in sorted(getattr(pred.model,
                                                       "tau_variants")
                                               .items())}
        if getattr(pred.model, "selection", None):
            m["gbm_selection"] = list(pred.model.selection)

    m["item1_summary"] = dict(
        crps_model=_r(np.nanmean(crps_model)),
        crps_climatology=_r(np.nanmean(crps_clim)),
        coverage90=_r(np.mean(cov90), 3),
        pit_mean=_r(np.mean(pit), 3))
    if cfg.compare_metrics and m["feature_set"] == "deployable":
        m["item8_with_without_schedule"] = _compare_with_oracle(
            cfg.compare_metrics, m, log=log)
    report = render_report(cfg, m, ds)
    return report, m


def _compare_with_oracle(metrics_path, m: dict, log=print) -> dict:
    """With-vs-without-schedule table (deployable runs only): per tau, CAL
    MAE/CRPS of the oracle-arrivals hybrid (from the v0.3 metrics.json
    given), the deployable hybrid, the deployable ridge and climatology
    (all from this run's baselines section)."""
    import json
    doc = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    ev = doc.get("eval") if isinstance(doc.get("eval"), dict) else doc

    def _all_rows(item6):
        return {float(r["tau"]): r for r in (item6.get("rows") or [])
                if r.get("chunk") == "all" and r.get("policy") == "all"}

    orc = _all_rows(ev.get("item6_baselines") or {})
    dep = _all_rows(m.get("item6_baselines") or {})
    rows = []
    for tau in sorted(set(orc) & set(dep)):
        o, d = orc[tau], dep[tau]
        rows.append(dict(
            tau=tau, n=int(d["n"]),
            mae_hyb_oracle=o.get("mae_model"),
            crps_hyb_oracle=o.get("crps_model"),
            mae_hyb_dep=d["mae_model"], crps_hyb_dep=d["crps_model"],
            mae_ridge_dep=d.get("mae_ridge"),
            crps_ridge_dep=d.get("crps_ridge"),
            mae_clim=d["mae_climatology"], crps_clim=d["crps_climatology"]))
    out = dict(oracle_metrics_path=str(metrics_path),
               oracle_manifest_sha256=(doc.get("data") or {})
               .get("manifest_sha256", ""),
               oracle_variant=(doc.get("gbm") or {}).get("variant"),
               rows=rows)
    log(f"eval: with-vs-without-schedule table built from {metrics_path} "
        f"({len(rows)} shared taus)")
    return out


def _r(x, nd=1) -> float:
    """Round a scalar to nd decimals; NaN stays NaN."""
    return round(float(x), nd) if np.isfinite(x) else float("nan")


def mixture_point_from_batches(nets, batch: dict) -> np.ndarray:
    """Mixture posterior mean for an already-gathered batch dict."""
    acc = None
    with torch.no_grad():
        for _, net, _ in nets:
            out = net(batch)
            v = torch.softmax(out["v_delay"].detach().float().cpu(), -1) \
                .numpy() @ met_centers()
            acc = v if acc is None else acc + v
    return acc / len(nets)


def _policy_contrast(pred, ds, met_):
    """Predicted dV at t0 vs realised base-minus-switching-arm delay
    (v0.3.2 fix #3: the switching arm is a random per-CAV mix, not a
    fleet-native policy)."""
    r = ds.rows
    cal = r[r["split"] == "CALIBRATION"]
    rows_out, preds, reals = [], [], []
    for (world, chunk), g in cal.groupby(["world", "chunk_id"]):
        base = g[g["source"] == "base"]
        sw = g[g["source"] == "native"]
        if base.empty or sw.empty:
            continue
        t0 = float(base["t0_s"].iloc[0])
        b0 = base[np.isclose(base["t"], t0) & np.isclose(base["tau"], 570.0)]
        n0 = sw[np.isclose(sw["t"], t0 + 10.0)
                & np.isclose(sw["tau"], 570.0)]
        if b0.empty or n0.empty:
            continue                      # cannot form the pair
        g_idx = int(b0["t_idx"].iloc[0])
        pre = 1.0 if bool(b0["pre_fix"].iloc[0]) else 0.0
        vd = pred.point_global(ds, [g_idx], 570.0,
                               _pid("damping_fleet"), pre)[0]
        vn = pred.point_global(ds, [g_idx], 570.0, _pid("switching_mix"),
                               pre)[0]
        d_pred = float(vd - vn)
        d_real = float(b0["total_delay_ref_s"].iloc[0]
                       - n0["total_delay_ref_s"].iloc[0])
        preds.append(d_pred)
        reals.append(d_real)
        rows_out.append(dict(world=str(world), chunk=str(chunk),
                             t0_s=t0, dV_pred=_r(d_pred),
                             dV_realised=_r(d_real),
                             sign_ok=bool(np.sign(d_pred)
                                          == np.sign(d_real))))
    t3 = dict(n_worlds=len(preds),
              caveat=("single-realisation truth (A2 step-2 noise caveat); "
                      "realised switching side uses the tau=570 label at "
                      "t0+10 s (switching-arm decision times start after "
                      "t0; 10 s window offset); the '__native' arm is a "
                      "random per-CAV switching mix, NOT a fleet-native "
                      "policy (v0.3.2 fix #3); replicate campaign would "
                      "supply expected-value truth"))
    if len(preds) >= 2:
        preds_a, reals_a = np.asarray(preds), np.asarray(reals)
        t3.update(
            sign_agreement=_r(np.mean(np.sign(preds_a) == np.sign(reals_a)),
                              3),
            spearman=_r(spearman(preds_a, reals_a), 3),
            mae_model=_r(np.mean(np.abs(preds_a - reals_a))),
            mae_zero_baseline=_r(np.mean(np.abs(reals_a))))
    elif len(preds) == 1:
        t3.update(mae_model=_r(abs(preds[0] - reals[0])),
                  mae_zero_baseline=_r(abs(reals[0])))
    return rows_out, t3


def _pid(name: str) -> int:
    from .model import policy_id
    return policy_id(name)


def _speed_benchmark(nets, ds, cal_idx, n_plans: int, k_taus: int,
                     device) -> dict:
    """Predictions/s for n_plans plans x M members x k_taus taus."""
    idx = cal_idx[:max(1, min(n_plans, len(cal_idx)))]
    taus = np.linspace(30.0, 570.0, k_taus, dtype=np.float32)
    g_idx = np.repeat(ds.rows["t_idx"].to_numpy()[idx], k_taus)
    tau = np.tile(taus, len(idx))
    pol = np.zeros(len(g_idx), dtype=np.int64)      # switching_mix (idx 0)
    pre = ds.rows["pre_fix"].to_numpy()[idx].astype(np.float32)
    pre = np.repeat(pre, k_taus)
    n_total = len(g_idx) * len(nets)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for _, net, _ in nets:
            for s in range(0, len(g_idx), EVAL_BS):
                net(ds.gather_global(g_idx[s:s + EVAL_BS],
                                     tau[s:s + EVAL_BS],
                                     pol[s:s + EVAL_BS],
                                     pre[s:s + EVAL_BS]))
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dict(plans=int(len(idx)), taus=int(k_taus),
                members=len(nets), predictions=int(n_total),
                seconds=_r(dt, 3), predictions_per_s=_r(n_total / dt))


def render_report(cfg: EvalConfig, m: dict, ds) -> str:
    """eval_report.md text (all numbers smoke-only when cfg.smoke)."""
    man = ds.man
    from .outputs import git_commit
    if m["value_model"] == "gbm":
        var = m.get("gbm_variant", "pure")
        if var == "hybrid":
            vm = ("gbm — HYBRID: per tau/target a ridge point model on "
                  "the summary features + calibrated quantile-GBM on its "
                  "residuals (gbm_value.pkl; no member weights)")
        elif var == "auto":
            tv = m.get("gbm_tau_variants", {})
            by = {}
            for t, v in tv.items():
                by.setdefault(v, []).append(t)
            per_tau = "; ".join(f"{v}@{','.join(ts)}"
                                for v, ts in sorted(
                                    by.items(), key=lambda kv: -len(kv[1])))
            vm = ("gbm — AUTO (per-tau TRAIN-OOB selection): "
                  + (per_tau if per_tau else "selection not recorded")
                  + " (gbm_value.pkl; ridge-only taus ship the ridge point "
                  "+ conformal residual quantiles, no GBM quantile models)")
        else:
            vm = ("gbm — PURE: calibrated quantile-GBM on the target "
                  "(gbm_value.pkl; no member weights)")
        wm = f"- value model: {vm}"
    else:
        vm = f"net mixture of {m['n_members']} members"
        wm = f"- weights: `{cfg.out_dir}` ({m['n_members']} members)"
    fs = m.get("feature_set", "oracle")
    if fs == "deployable_strict":
        fl = ("feature set: **deployable_strict (HEADLINE, CAV-STRICT)** — "
              "RSU inputs are CAV self state + CAV self-reports only: no "
              "all-vehicle entrance counts, entrance queues zeroed, no "
              "penetration-revealing ratio; no future-arrival schedule, "
              "no future CAV share, no time-to-branch, no t0 (v0.3.2 fix "
              "#1); pre_fix kept as a training feature but FIXED to the "
              "post-fix value at predict (v0.3.2 fix #11)")
    elif fs in ("deployable_detector", "deployable"):
        fl = ("feature set: **deployable_detector (labelled SENSITIVITY, "
              "not a deployable claim)** — all-vehicle entrance-detector "
              "counts + CAV counts + entrance queues; no future-arrival "
              "schedule, no future CAV share, no time-to-branch, no t0 "
              "(v0.3.2 fix #1); pre_fix FIXED to the post-fix value at "
              "predict (v0.3.2 fix #11)")
    elif fs == "deployable":
        fl = ("feature set: **deployable** — RSU-observable inputs only: "
              "no future-arrival schedule, no future CAV share, no "
              "time-to-branch, no true penetration; observed entrance/"
              "CAV-entry counts and the configured demand rate instead")
    else:
        fl = ("feature set: **oracle** — includes the exact future-arrival "
              "schedule (offline/CRN-paired planner-evaluation upper bound; "
              "see PARTB_LEAK_AUDIT.md)")
    # v0.3.2 fix #6: every eval_report states the sensing premise
    premise = ("ideal sensing premise: state features from the full true "
               "state; reconstruction gap measured separately (RG-0)")
    lines = [
        "# WD-01 Part B V0 — CALIBRATION evaluation report", "",
        wm,
        fl,
        f"- {premise}",
        f"- device: {m['device']} | smoke run: **{m['smoke']}**"
        + ("  — ALL NUMBERS BELOW ARE SMOKE-ONLY" if m["smoke"] else ""),
        f"- CAL rows scored: {m['n_cal_rows']}",
        f"- data MANIFEST sha256: `{man.sha256}`",
        f"- git: `{git_commit()}`",
        "",
    ]
    # ---- section 0: per-tau variant selection (auto runs) ------------
    sel = m.get("gbm_selection") or []
    if m["value_model"] == "gbm" and m.get("gbm_variant") == "auto":
        lines += [
            "## 0. Per-tau value-model variant selected on TRAIN OOB "
            "(world-grouped)", "",
            "Shipping decision, made on TRAIN rows ONLY (CAL never touched "
            "by selection/calibration; guard raises on any CALIBRATION "
            "row): per tau the hybrid candidate vs ridge-only (ridge point "
            "+ conformal residual quantiles from the SAME world-grouped "
            "TRAIN OOF folds), compared by mean OOB CRPS of the delay "
            "head. Exact tie ships ridge (the simpler model).", "",
            "| tau | chosen | OOB CRPS hybrid | OOB CRPS ridge | n TRAIN "
            "| select target | folds |",
            "|---|---|---|---|---|---|---|"]
        for e in sel:
            lines.append(
                f"| {e['tau']:.0f} | {e['chosen']} "
                f"| {e['oob_crps']['hybrid']:.1f} "
                f"| {e['oob_crps']['ridge']:.1f} | {e['n']} "
                f"| {e['select_target']} | {e.get('folds', '')} |")
        tv = m.get("gbm_tau_variants", {})
        ridge_taus = sorted(float(t) for t, v in tv.items()
                            if v == "ridge")
        hyb_taus = sorted(float(t) for t, v in tv.items()
                          if v == "hybrid")
        if ridge_taus:
            lines += [
                "",
                f"Headline: **deployable long-horizon skill ~ ridge** — "
                f"the per-tau TRAIN-OOB selection shipped ridge-only at "
                f"tau {', '.join(f'{t:.0f}' for t in ridge_taus)}"
                + (f" (hybrid kept at tau "
                   f"{', '.join(f'{t:.0f}' for t in hyb_taus)})"
                   if hyb_taus else "") + "."]
    lines += [
        "",
        "## 1. Calibration & sharpness per tau (switching mix and damping "
        "fleet separately, per chunk)", "",
        "| chunk | policy | tau | n | CRPS model | CRPS climatology | "
        "PIT mean | 10-90 coverage | clim level |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for x in m["item1_crps_pit_coverage"]:
        lines.append(
            f"| {x['chunk']} | {x['policy']} | {x['tau']:.0f} | {x['n']} "
            f"| {x['crps_model']:.1f} | {x['crps_clim']:.1f} "
            f"| {x['pit_mean']:.3f} | {x['coverage90']:.3f} "
            f"| {x['clim_level']} |")
    s = m["item1_summary"]
    lines += [
        f"Overall: CRPS model {s['crps_model']:.1f} vs climatology "
        f"{s['crps_climatology']:.1f}; 10-90% coverage "
        f"{s['coverage90']:.3f}; PIT mean {s['pit_mean']:.3f}.",
        "",
        "### PIT histogram per tau (randomized, 10 bins)", "",
        "| tau | " + " | ".join(f"{i / 10:.1f}-{(i + 1) / 10:.1f}"
                                for i in range(10)) + " |",
        "|---|" + "---|" * 10,
    ]
    for tau_s, counts in m["item1_pit_hist_per_tau"].items():
        lines.append(f"| {tau_s} | "
                     + " | ".join(str(int(c)) for c in counts) + " |")
    lines += [
        "",
        "## 2. TD consistency: V(s,tau) - [D30 + V(s+30, tau-30)]", "",
        "| tau | policy | n | mean | std | q05 | q50 | q95 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for x in m["item2_td_residual"]:
        lines.append(
            f"| {x['tau']:.0f} | {x['policy']} | {x['n']} "
            f"| {x['mean']:.1f} | {x['std']:.1f} | {x['q05']:.1f} "
            f"| {x['q50']:.1f} | {x['q95']:.1f} |")
    lines += ["", "## 3. Damping fleet vs switching mix at t0 (CRN pairs "
              "-- the campaign arm is a random per-CAV switching schedule, "
              "NOT a fleet-native policy)", ""]
    if m["item3_rows"]:
        lines += ["| world | chunk | dV pred | dV realised | sign ok |",
                  "|---|---|---|---|---|"]
        for x in m["item3_rows"]:
            lines.append(
                f"| {x['world']} | {x['chunk']} | {x['dV_pred']:.1f} "
                f"| {x['dV_realised']:.1f} | {x['sign_ok']} |")
    t3 = m["item3_policy_contrast"]
    lines += ["", f"n_worlds = {t3['n_worlds']}"]
    if "sign_agreement" in t3:
        lines += [
            f"sign agreement {t3['sign_agreement']:.2f}; Spearman "
            f"{t3['spearman']:.3f}; MAE(model) {t3['mae_model']:.1f} vs "
            f"MAE(dV=0 baseline) {t3['mae_zero_baseline']:.1f}"]
    lines += ["", f"Caveat: {t3['caveat']}", "",
              "## 4. Safety heads — DIAGNOSTIC ONLY (G-1/G-2 gated)", "",
              "| tau | n | P_coll Brier | collisions | exposure CRPS "
              "model | exposure CRPS clim |", "|---|---|---|---|---|---|"]
    for x in m["item4_safety_diagnostic"]:
        lines.append(
            f"| {x['tau']:.0f} | {x['n']} | {x['p_coll_brier']:.4f} "
            f"| {x['collisions']} | {x['exp_crps_model']:.1f} "
            f"| {x['exp_crps_clim']:.1f} |")
    t5 = m["item5_speed"]
    lines += [
        "", "## 5. Planner-call speed", "",
        f"{t5['plans']} plans x {t5['members']} members x {t5['taus']} "
        f"taus = {t5['predictions']} predictions in {t5['seconds']:.3f} s "
        f"-> **{t5['predictions_per_s']:.0f} predictions/s**"]

    # ------------- items 6-7: baselines + HARD acceptance (B3a) --------
    b = m.get("item6_baselines") or {}

    def _f(x, spec=".1f") -> str:
        return format(x, spec) if x is not None and np.isfinite(x) \
            else "nan"

    lines += [
        "", "## 6. Baselines on CAL — PERMANENT (B3 item D15)", "",
        "MAE of the median point forecast and CRPS per tau x policy x "
        "chunk for climatology, ridge and a point-GBM fitted on TRAIN on "
        "the diagnosis summary features, against the evaluated model "
        f"(**{m['value_model']}**).", "",
        "| chunk | policy | tau | n | MAE clim | CRPS clim | MAE ridge "
        "| CRPS ridge | MAE gbm-base | CRPS gbm-base | MAE model "
        "| CRPS model |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in b.get("rows", []):
        lines.append(
            f"| {x['chunk']} | {x['policy']} | {x['tau']:.0f} | {x['n']} "
            f"| {_f(x['mae_climatology'])} | {_f(x['crps_climatology'])} "
            f"| {_f(x['mae_ridge'])} | {_f(x['crps_ridge'])} "
            f"| {_f(x['mae_gbm_baseline'])} | {_f(x['crps_gbm_baseline'])} "
            f"| {_f(x['mae_model'])} | {_f(x['crps_model'])} |")
    lines += [
        "", "## 7. HARD acceptance vs RIDGE, per tau (B3 item D16)", "",
        "PASS requires the evaluated model to beat RIDGE on BOTH "
        "MAE(median) and CRPS at that tau (CAL, all chunks and policies)."
        "  A tie is a TIE, not a pass: with no strict win on both metrics "
        "and no strict loss on either (e.g. a ridge-only tau IS the ridge "
        "baseline), the verdict reads TIE.",
        "",
        "| tau | n | MAE model | MAE ridge | CRPS model | CRPS ridge "
        "| MAE beat | CRPS beat | verdict |",
        "|---|---|---|---|---|---|---|---|---|"]
    for a in b.get("acceptance", []):
        lines.append(
            f"| {a['tau']:.0f} | {a['n']} | {_f(a['mae_model'])} "
            f"| {_f(a['mae_ridge'])} | {_f(a['crps_model'])} "
            f"| {_f(a['crps_ridge'])} | {a['beats_ridge_mae']} "
            f"| {a['beats_ridge_crps']} "
            f"| {a.get('result', 'PASS' if a['hard_pass'] else 'FAIL')} |")
    ov = b.get("overall", {})
    lines += [
        "",
        f"Overall HARD acceptance vs ridge: "
        f"**{ov.get('result', 'PASS' if ov.get('hard_pass') else 'FAIL')}**"
        f" ({ov.get('n_pass', 0)}/{ov.get('n_taus', 0)} taus pass"
        + (f", {ov.get('n_tie', 0)} tie" if ov.get("n_tie") else "")
        + "; a tie is a TIE, not a pass)."]
    if b.get("approx"):
        lines += ["", f"CRPS notes: {b['approx']}"]

    # ------------- item 7b: TRAIN-internal world-grouped OOB -----------
    ob = m.get("item7b_oob_acceptance")
    if ob:
        lines += [
            "", "## 7b. HARD acceptance vs RIDGE on TRAIN OOB "
            "(world-grouped k-fold)", "",
            "Same acceptance on TRAIN-internal out-of-fold predictions, "
            "so the verdict does not rest on CAL alone.  Tie semantics as "
            "section 7: a tie is a TIE, not a pass.", ""]
        if ob.get("acceptance"):
            lines += [
                "| tau | n | MAE model | MAE ridge | CRPS model | "
                "CRPS ridge | MAE beat | CRPS beat | verdict |",
                "|---|---|---|---|---|---|---|---|---|"]
            for a in ob["acceptance"]:
                lines.append(
                    f"| {a['tau']:.0f} | {a['n']} | {_f(a['mae_model'])} "
                    f"| {_f(a['mae_ridge'])} | {_f(a['crps_model'])} "
                    f"| {_f(a['crps_ridge'])} | {a['beats_ridge_mae']} "
                    f"| {a['beats_ridge_crps']} "
                    f"| {a.get('result', 'PASS' if a['hard_pass'] else 'FAIL')} |")
            obv = ob.get("overall", {})
            lines += [
                "",
                f"Overall OOB HARD acceptance vs ridge: "
                f"**{obv.get('result', 'PASS' if obv.get('hard_pass') else 'FAIL')}** "
                f"({obv.get('n_pass', 0)}/{obv.get('n_taus', 0)} taus "
                f"pass"
                + (f", {obv.get('n_tie', 0)} tie"
                   if obv.get("n_tie") else "") + ")."]
        if ob.get("note"):
            lines += ["", f"OOB notes: {ob['note']}"]

    # ------------- item 8: with-vs-without-schedule comparison ---------
    c8 = m.get("item8_with_without_schedule")
    if c8:
        lines += [
            "", "## 8. With vs without the future-arrival schedule (CAL, "
            "all chunks and policies)", "",
            "Per tau: the oracle-arrivals hybrid (v0.3, upper bound; "
            "exact future schedule as input) vs the deployable hybrid, "
            "the deployable ridge and climatology — the last three on "
            "RSU-observable inputs only.", "",
            f"- oracle metrics: `{c8['oracle_metrics_path']}` "
            f"(MANIFEST sha256 `{c8['oracle_manifest_sha256'][:12]}...`, "
            f"variant={c8.get('oracle_variant')})",
            "",
            "| tau | n | MAE hyb-oracle | CRPS hyb-oracle | MAE hyb-dep "
            "| CRPS hyb-dep | MAE ridge-dep | CRPS ridge-dep | MAE clim "
            "| CRPS clim |",
            "|---|---|---|---|---|---|---|---|---|---|"]
        for x in c8.get("rows", []):
            lines.append(
                f"| {x['tau']:.0f} | {x['n']} "
                f"| {_f(x['mae_hyb_oracle'])} | {_f(x['crps_hyb_oracle'])} "
                f"| {_f(x['mae_hyb_dep'])} | {_f(x['crps_hyb_dep'])} "
                f"| {_f(x['mae_ridge_dep'])} | {_f(x['crps_ridge_dep'])} "
                f"| {_f(x['mae_clim'])} | {_f(x['crps_clim'])} |")
    lines += [""]
    return "\n".join(lines)


# ===========================================================================
# v0.3 net evaluation (B3b: BRIEF_B3_V03.md A5, B6 reporting, C13, D14).
# NEW functions only -- the v0.2/GBM evaluation above is untouched (B3a
# owns its report-table edits).  All v0.3 pmfs live on the NORMALISED grid
# (A1/A2); reporting multiplies by the TRAIN per-tau medians for real
# units.  Numbers from smoke runs are smoke-only.
# ===========================================================================

import wdpaths  # noqa: E402  (E:-native roots; no D:)

# The net's acceptance BAR (addendum 7): beat the per-tau-selected
# deployable hybrid of the v0.3.1 run, per tau, on BOTH MAE(median) and
# CRPS.  Read-only; absent -> the report says "bar not available".
V031_BAR_DIR = (Path(wdpaths.RESEARCH_DENSE_ROOT) / "output"
                / "partB_results" / "v0.3.1_selected_deployable")


def load_net_v03(out_dir, device: torch.device, log=print) -> dict:
    """Load the v0.3 net bundle: member_{k}.pt/json (format-gated),
    stats.json (required: the A4 statistics ship with the weights) and
    calibration.json (optional; D14)."""
    from .model import (V03ValueConfig, V03ValueNet, check_token_widths)
    from .train_net import FORMAT_V03
    out = Path(out_dir)
    stats_path = out / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"stats.json missing in {out} -- the v0.3 weights require the "
            "saved TRAIN standardisation/target statistics (A4)")
    stats = json_loads(stats_path)
    calib = None
    cal_path = out / "calibration.json"
    if cal_path.exists():
        from .calibration import CalMap
        calib = CalMap.load(cal_path)
    nets, cfgs, sidecars = [], [], []
    i = 0
    while (w := out / f"member_{i}.pt").exists():
        side = json_loads(out / f"member_{i}.json") \
            if (out / f"member_{i}.json").exists() else {}
        fmt = side.get("format", FORMAT_V03)
        if fmt != FORMAT_V03:
            raise ValueError(
                f"member_{i}.json has format {fmt!r}, expected "
                f"{FORMAT_V03!r} (a v0.2 member_* file? use load_members)")
        cfg = V03ValueConfig.from_json(side.get("cfg_model", {}))
        net = V03ValueNet(cfg).to(device)
        state = torch.load(w, map_location=device, weights_only=True)
        check_token_widths(state, net.cfg, where=f"member_{i}.pt")
        net.load_state_dict(state)
        net.eval()
        nets.append(net)
        cfgs.append(cfg)
        sidecars.append(side)
        i += 1
    if not nets:
        raise FileNotFoundError(f"no member_*.pt v0.3 weights in {out}")
    log(f"eval v0.3: loaded {len(nets)} members from {out}"
        + (f" + calibration ({calib.method})" if calib is not None
           else " (uncalibrated)"))
    return dict(nets=nets, cfgs=cfgs, sidecars=sidecars, stats=stats,
                calib=calib)


@torch.no_grad()
def net_mixture_probs(bundle, data, row_idx, head: str = "v_delay",
                      bs: int = EVAL_BS) -> np.ndarray:
    """Mixture pmf of the v0.3 members over label rows (normalised)."""
    idx = np.asarray(row_idx, dtype=np.int64)
    mix = None
    for net in bundle["nets"]:
        probs = []
        for s in range(0, len(idx), bs):
            out = net(data.gather_rows(idx[s:s + bs]))
            probs.append(_softmax_np(out[head]))
        p = np.concatenate(probs, axis=0)
        mix = p if mix is None else mix + p
    return mix / len(bundle["nets"])


@torch.no_grad()
def net_delta_pred(bundle, data, split: str = "CALIBRATION",
                   bs: int = EVAL_BS) -> tuple[np.ndarray, np.ndarray]:
    """C13: ensemble-mean delta prediction (real units) + realised delta
    on the CRN pairs of `split`."""
    g, tau, _dn, dreal, pre = data.pair_tensors(split)
    if g.shape[0] == 0:
        return np.array([]), np.array([])
    n = int(g.shape[0])
    pred = np.zeros(n)
    for net in bundle["nets"]:
        acc = []
        for s in range(0, n, bs):
            batch = data.gather_global(
                g[s:s + bs], tau[s:s + bs],
                torch.zeros(min(bs, n - s), dtype=torch.long,
                            device=g.device),
                pre[s:s + bs])
            acc.append(net(batch)["delta"].detach().float().cpu().numpy())
        pred += np.concatenate(acc)
    pred /= len(bundle["nets"])
    tau_np = tau.to(torch.float64).cpu().numpy()
    return pred * data.delta_scale.at(tau_np), \
        dreal.to(torch.float64).cpu().numpy()


def _cal_rows(data, cap: int | None) -> np.ndarray:
    idx = np.flatnonzero(data.split_cal)
    if cap is None:
        return idx
    keep = []
    r_tau = data.row_tau[idx]
    r_pol = data.row_policy[idx]
    for tau in np.unique(r_tau):
        for pol in (0, 1):
            sel = idx[(np.isclose(r_tau, tau)) & (r_pol == pol)]
            keep.append(sel[:cap])
    return np.sort(np.concatenate(keep)) if keep else idx


def fit_run_calibration(bundle, data, out_dir, *,
                        method: str = "shift", smoke: bool = False,
                        seed: int | None = None, members: int | None = None,
                        log=print) -> dict:
    """D14: fit the per-tau map on the pooled OOB TRAIN predictions of
    the ensemble and save calibration.json (ships with the weights).
    OOB TRAIN only -- CAL is never touched here (B6 guard).

    Fallback (documented in the returned info and calibration.json): when
    the pooled OOB set is EMPTY -- possible on tiny data where a member's
    bootstrap happens to cover every TRAIN world -- the map is fitted on
    TRAIN-in-bag rows instead of refusing; the fit source is recorded."""
    from .calibration import FIT_LEVELS, fit_per_tau, quantile_matrix
    from .model import PmfGrid
    if not data.oob_rows:
        if seed is None:
            seed = bundle["sidecars"][0].get("member_seed", 0)
        members = members or len(bundle["nets"])
        data.set_members(seed, members)
    pooled = np.unique(np.concatenate(
        [np.asarray(x) for x in data.oob_rows if len(x)] or
        [np.array([], np.int64)]))
    fit_source = "oob"
    if len(pooled) == 0:
        fit_source = "train_inbag_fallback"
        pooled = np.flatnonzero(data.split_train)
        log("calibration: pooled OOB is EMPTY (bootstrap covered every "
            "TRAIN world); fitting on TRAIN-in-bag rows instead (CAL is "
            "still untouched)")
    cap = 40 if smoke else None
    if cap is not None and len(pooled) > cap * max(1, len(data.oob_rows)):
        rng = np.random.default_rng(20260929)
        pooled = np.sort(rng.choice(pooled, size=cap * max(
            1, len(data.oob_rows)), replace=False))
    if len(pooled) == 0:
        raise RuntimeError("no rows available for calibration fitting")
    probs = net_mixture_probs(bundle, data, pooled)
    y = data.y_norm["delay"][torch.as_tensor(pooled,
                                             device=data.device)] \
        .to(torch.float64).cpu().numpy()
    tau = data.row_tau[pooled].astype(np.float64)
    cfg0 = bundle["cfgs"][0]
    grid = PmfGrid(cfg0.sym_min, cfg0.sym_max, cfg0.n_bins)
    qmat = quantile_matrix(grid, probs, FIT_LEVELS)
    calib = fit_per_tau(qmat, y, tau, method=method)
    calib.save(Path(out_dir) / "calibration.json")
    info = dict(method=method, n_oob=int(len(pooled)), fit_source=fit_source,
                per_tau={k: v.get("n") for k, v in
                         calib.per_tau.items()})
    log(f"calibration: fitted {method} maps on {len(pooled)} pooled "
        f"{fit_source} rows for taus {sorted(info['per_tau'])} -> "
        f"calibration.json")
    return info


def evaluate_net_v03(out_dir, data, device, *, smoke: bool = False,
                     log=print) -> tuple[str, dict]:
    """v0.3 net evaluation on CALIBRATION: median point forecast (A5),
    calibration before/after (D14), member spread, TD-vs-bootstrap
    consistency (B8), the paired Delta head (C13) and the v0.3.1
    acceptance BAR.  CAL is REPORTED here only -- never selected on."""
    from .calibration import CAL_LEVELS, coverage, quantile_matrix
    from .distributions import spearman
    from .model import PmfGrid

    bundle = load_net_v03(out_dir, device, log=log)
    cfg0 = bundle["cfgs"][0]
    grid = PmfGrid(cfg0.sym_min, cfg0.sym_max, cfg0.n_bins)
    cal_idx = _cal_rows(data, 40 if smoke else None)
    sub_tau = data.row_tau[cal_idx]
    sub_pol = np.where(data.row_policy[cal_idx] == 0, "switching_mix",
                       "damping_fleet")
    sub_chunk = data.row_chunk[cal_idx]
    y_norm = data.y_norm["delay"][torch.as_tensor(
        cal_idx, device=data.device)].to(torch.float64).cpu().numpy()
    y_real = y_norm * data.med_row["delay"][cal_idx]
    scale = data.med_row["delay"][cal_idx]

    probs = net_mixture_probs(bundle, data, cal_idx)
    med_norm = grid.median(probs)
    pred_real = med_norm * scale
    crps = grid.crps_real(probs, y_norm, scale)
    pit = grid.pit(probs, y_norm)
    qmat_cal = quantile_matrix(grid, probs, CAL_LEVELS)
    cov90 = ((y_norm[:, None] <= qmat_cal[:, [8]]) &
             (y_norm[:, None] >= qmat_cal[:, [0]])).squeeze(1)

    m: dict = dict(n_members=len(bundle["nets"]),
                   n_cal_rows=int(len(cal_idx)),
                   calibrated=bundle["calib"] is not None,
                   smoke=bool(smoke))
    t1 = []
    for chunk in dict.fromkeys(sub_chunk):
        for pol in ("switching_mix", "damping_fleet"):
            for tau in np.unique(sub_tau):
                gi = ((sub_chunk == chunk) & (sub_pol == pol)
                      & np.isclose(sub_tau, tau))
                if not gi.any():
                    continue
                t1.append(dict(
                    chunk=str(chunk), policy=pol, tau=float(tau),
                    n=int(gi.sum()),
                    mae_median=_r(np.abs(pred_real[gi] - y_real[gi]).mean()),
                    bias=_r((pred_real[gi] - y_real[gi]).mean()),
                    crps=_r(crps[gi].mean()),
                    coverage90=_r(cov90[gi].mean(), 3),
                    pit_mean=_r(pit[gi].mean(), 3)))
    m["item1_median_crps_coverage"] = t1
    m["item1_summary"] = dict(
        mae_median=_r(np.abs(pred_real - y_real).mean()),
        crps=_r(np.nanmean(crps)), coverage90=_r(cov90.mean(), 3),
        pit_mean=_r(pit.mean(), 3))
    m["item1_per_tau"] = _net_per_tau_table(
        sub_tau, pred_real, y_real, crps)

    # -------- D14: calibration before/after (CAL, reported only) -------
    # The fitted map's levels are FIT_LEVELS; apply it at its own levels
    # and read the CAL_LEVELS columns back out (a shift vector fitted at
    # 19 levels cannot broadcast onto a 9-column matrix).
    from .calibration import FIT_LEVELS as _FIT_LEVELS
    calib = bundle["calib"]
    t2 = []
    cal_before_rows, cal_after_rows = [], []
    cal_cols = [list(_FIT_LEVELS).index(lv) for lv in CAL_LEVELS]
    qmat_fit_lvls = quantile_matrix(grid, probs, _FIT_LEVELS)
    before_qmat_cal = qmat_fit_lvls[:, cal_cols]
    after_qmat_cal = (calib.apply(qmat_fit_lvls, sub_tau)[:, cal_cols]
                      if calib is not None else before_qmat_cal)
    for tau in np.unique(sub_tau):
        gi = np.isclose(sub_tau, tau)
        cov_b = coverage(y_norm[gi], before_qmat_cal[gi])
        cov_a = coverage(y_norm[gi], after_qmat_cal[gi])
        cal_b = float(np.mean(np.abs(cov_b - np.asarray(CAL_LEVELS))))
        cal_a = float(np.mean(np.abs(cov_a - np.asarray(CAL_LEVELS))))
        cal_before_rows.append(cal_b)
        cal_after_rows.append(cal_a)
        hist, _ = np.histogram(pit[gi], bins=np.linspace(0, 1, 11))
        t2.append(dict(
            tau=float(tau), n=int(gi.sum()),
            cal_before=_r(cal_b, 3), cal_after=_r(cal_a, 3),
            cov10_before=_r(cov_b[0], 3), cov10_after=_r(cov_a[0], 3),
            cov50_before=_r(cov_b[4], 3), cov50_after=_r(cov_a[4], 3),
            cov90_before=_r(cov_b[8], 3), cov90_after=_r(cov_a[8], 3),
            pit_hist=hist.astype(int).tolist()))
    m["item2_calibration"] = t2
    m["item2_summary"] = dict(
        method=(calib.method if calib is not None else "none"),
        cal_before=_r(float(np.mean(cal_before_rows)), 3),
        cal_after=_r(float(np.mean(cal_after_rows)), 3))

    # -------- member spread (per tau) -----------------------------------
    member_med = []
    for net in bundle["nets"]:
        pb = _one_net_probs(net, data, cal_idx)
        member_med.append(grid.median(pb) * scale)
    mm = np.stack(member_med)                     # [M, N]
    t3 = []
    for tau in np.unique(sub_tau):
        gi = np.isclose(sub_tau, tau)
        sd = mm[:, gi].std(axis=0)
        t3.append(dict(tau=float(tau), n=int(gi.sum()),
                       spread_median=_r(float(np.median(sd))),
                       spread_mean=_r(float(sd.mean()))))
    m["item3_member_spread"] = t3

    # -------- TD-vs-bootstrap consistency (B8) --------------------------
    td = data.td
    td_ok = ((td["split"] == "CALIBRATION") & (td["tau"] >= 60.0)
             & (td["t_idx"] >= 0) & (td["tnext_idx"] >= 0)).nonzero()[0]
    if smoke and len(td_ok) > 240:
        td_ok = td_ok[:: max(1, len(td_ok) // 240)]
    t4 = []
    if len(td_ok):
        itt = torch.as_tensor(td_ok, dtype=torch.long, device=data.device)
        n = len(td_ok)
        v_t, v_n = np.zeros(n), np.zeros(n)
        for net in bundle["nets"]:
            acc_t, acc_n = [], []
            for s in range(0, n, EVAL_BS):
                sl = slice(s, min(n, s + EVAL_BS))
                pre_t = data.t_td["pre"][itt][sl]
                bt = data.gather_global(data.t_td["t_idx"][itt][sl],
                                        data.t_td["tau"][itt][sl],
                                        data.t_td["policy_id"][itt][sl],
                                        pre_t)
                bn = data.gather_global(
                    data.t_td["tnext_idx"][itt][sl],
                    data.t_td["tau"][itt][sl] - 30.0,
                    data.t_td["policy_id"][itt][sl], pre_t)
                ot = net(bt)["v_delay"].detach().float()
                on = net(bn)["v_delay"].detach().float()
                acc_t.append(torch.softmax(ot, -1).cpu().numpy())
                acc_n.append(torch.softmax(on, -1).cpu().numpy())
            v_t += grid.median(np.concatenate(acc_t))
            v_n += grid.median(np.concatenate(acc_n))
        v_t /= len(bundle["nets"])
        v_n /= len(bundle["nets"])
        d30 = np.asarray(data.td["d30"])[itt].astype(np.float64)
        tau_t = np.asarray(data.td["tau"])[itt].astype(np.float64)
        pol_t = np.where(np.asarray(data.td["policy_id"])[itt] == 0,
                         "switching_mix", "damping_fleet")
        med_t = data.med["delay"].at(tau_t, pol_t)
        med_n = data.med["delay"].at(tau_t - 30.0, pol_t)
        resid = (v_t * med_t) - (d30 + v_n * med_n)
        chain = np.asarray(data.td_chain["ok"])[itt].astype(np.float64)
        for tau in np.unique(tau_t):
            gi = np.isclose(tau_t, tau)
            t4.append(dict(
                tau=float(tau), n=int(gi.sum()),
                n_step_frac=_r(float(chain[gi].mean()), 3),
                mean=_r(resid[gi].mean()), q50=_r(np.quantile(resid[gi],
                                                              0.5))))
    m["item4_td_consistency"] = t4

    # -------- Delta head (C13) ------------------------------------------
    pred_d, real_d = net_delta_pred(bundle, data, "CALIBRATION")
    t5 = dict(n_pairs=int(len(pred_d)))
    if len(pred_d):
        t5.update(
            sign_agreement=_r(float(np.mean(
                np.sign(pred_d) == np.sign(real_d))), 3),
            spearman=_r(spearman(pred_d, real_d), 3),
            mae_model=_r(float(np.abs(pred_d - real_d).mean())),
            mae_zero_baseline=_r(float(np.abs(real_d).mean())),
            pred_median=_r(float(np.median(pred_d))),
            real_median=_r(float(np.median(real_d))))
    m["item5_delta_head"] = t5

    # ---- PERMANENT baselines + HARD acceptance vs ridge (D15/D16) -----
    # Same estimator and table as the v0.2/GBM report (gbm.py): ridge and
    # point-GBM on the diagnosis summary features, fitted on TRAIN, MAE of
    # the median and CRPS on CAL per tau -- the net's median (A5) and its
    # exact mixture CRPS are directly comparable.
    if len(cal_idx):
        from .gbm import baselines_section
        log("eval: fitting the PERMANENT ridge/GBM baselines "
            "(diag summary features, TRAIN) for the net report")
        m["item7_baselines"] = baselines_section(
            data.ds, cal_idx, pred_real, crps, log=log,
            gbm_max_iter=40 if smoke else 300)
    else:
        m["item7_baselines"] = dict(
            rows=[], acceptance=[], approx="no CAL rows",
            overall=dict(hard_pass=False, n_taus=0, n_pass=0))

    # -------- acceptance BAR vs the v0.3.1 selected deployable ----------
    m["item6_bar"] = bar_section(m.get("item1_per_tau") or [], log=log)

    report = render_net_report(m)
    return report, m


def _net_per_tau_table(sub_tau, pred_real, y_real, crps) -> list[dict]:
    """Per-tau (all chunks/policies) MAE/CRPS of the net -- the rows the
    acceptance BAR compares against the v0.3.1 deployable hybrid."""
    out = []
    for tau in np.unique(sub_tau):
        gi = np.isclose(sub_tau, tau)
        out.append(dict(
            tau=float(tau), n=int(gi.sum()),
            mae_median=_r(float(np.abs(pred_real[gi] - y_real[gi]).mean())),
            crps=_r(float(np.nanmean(crps[gi])))))
    return out


def bar_section(per_tau_rows: list[dict], log=print) -> dict:
    """The net's acceptance BAR (addendum 7): per tau, beat the
    per-tau-selected DEPLOYABLE model of the v0.3.1 run
    (output/partB_results/v0.3.1_selected_deployable) on BOTH MAE(median)
    and CRPS.  Read-only; absent -> 'bar not available'."""
    metrics_path = V031_BAR_DIR / "metrics.json"
    if not metrics_path.exists():
        log("bar: v0.3.1 selected deployable run not present -> "
            "'bar not available'")
        return dict(available=False, note="bar not available",
                    bar_dir=str(V031_BAR_DIR))
    doc = json_loads(metrics_path)
    ev = doc.get("eval") if isinstance(doc.get("eval"), dict) else {}
    bar = {float(a["tau"]): a for a in
           (ev.get("item6_baselines") or {}).get("acceptance", [])}
    rows, n_pass = [], 0
    for r in per_tau_rows:
        b = bar.get(float(r["tau"]))
        if b is None:
            continue
        mae_ok = bool(np.isfinite(r["mae_median"])
                      and r["mae_median"] < b.get("mae_model", np.inf))
        crps_ok = bool(np.isfinite(r["crps"])
                       and r["crps"] < b.get("crps_model", np.inf))
        passed = mae_ok and crps_ok
        n_pass += int(passed)
        rows.append(dict(
            tau=float(r["tau"]), n=int(r["n"]),
            mae_net=r["mae_median"], mae_bar=b.get("mae_model"),
            crps_net=r["crps"], crps_bar=b.get("crps_model"),
            beats_bar_mae=mae_ok, beats_bar_crps=crps_ok,
            bar_pass=passed))
    log(f"bar: compared {len(rows)} taus against "
        f"{V031_BAR_DIR} ({n_pass} pass)")
    return dict(available=True, bar_dir=str(V031_BAR_DIR),
                bar_manifest_sha256=(doc.get("data") or {})
                .get("manifest_sha256", ""),
                rows=rows,
                overall=dict(n_taus=len(rows), n_pass=n_pass,
                             hard_pass=bool(rows)
                             and n_pass == len(rows)))


def _one_net_probs(net, data, row_idx, bs: int = EVAL_BS) -> np.ndarray:
    idx = np.asarray(row_idx, dtype=np.int64)
    probs = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            out = net(data.gather_rows(idx[s:s + bs]))
            probs.append(_softmax_np(out["v_delay"]))
    return np.concatenate(probs, axis=0)


def render_net_report(m: dict) -> str:
    lines = [
        "# WD-01 Part B v0.3 net — CALIBRATION evaluation report", "",
        f"- members: {m['n_members']} | calibrated: "
        f"{m['item2_summary']['method']} | smoke run: **{m['smoke']}**"
        + ("  — ALL NUMBERS BELOW ARE SMOKE-ONLY" if m["smoke"] else ""),
        f"- CAL rows scored: {m['n_cal_rows']}",
        # v0.3.2 fix #6: every eval_report states the sensing premise
        "- ideal sensing premise: state features from the full true "
        "state; reconstruction gap measured separately (RG-0)", "",
        "## 1. Median point forecast, CRPS, coverage per tau "
        "(switching mix and damping fleet separately, per chunk)", "",
        "| chunk | policy | tau | n | MAE median | bias | CRPS | "
        "cov 10-90 | PIT mean |", "|---|---|---|---|---|---|---|---|---|"]
    for x in m["item1_median_crps_coverage"]:
        lines.append(
            f"| {x['chunk']} | {x['policy']} | {x['tau']:.0f} | {x['n']} "
            f"| {x['mae_median']:.1f} | {x['bias']:.1f} "
            f"| {x['crps']:.1f} | {x['coverage90']:.3f} "
            f"| {x['pit_mean']:.3f} |")
    s = m["item1_summary"]
    lines += [f"Overall: MAE(median) {s['mae_median']:.1f}; CRPS "
              f"{s['crps']:.1f}; 10-90% coverage {s['coverage90']:.3f}; "
              f"PIT mean {s['pit_mean']:.3f}.", "",
              "## 2. Post-hoc per-tau calibration (D14): CAL before/after"
              " — reported only, never used for selection", "",
              "| tau | n | CAL before | CAL after | cov10 b/a | "
              "cov50 b/a | cov90 b/a | PIT hist (before) |",
              "|---|---|---|---|---|---|---|---|"]
    for x in m["item2_calibration"]:
        lines.append(
            f"| {x['tau']:.0f} | {x['n']} | {x['cal_before']:.3f} "
            f"| {x['cal_after']:.3f} | {x['cov10_before']:.3f}/"
            f"{x['cov10_after']:.3f} | {x['cov50_before']:.3f}/"
            f"{x['cov50_after']:.3f} | {x['cov90_before']:.3f}/"
            f"{x['cov90_after']:.3f} | {x['pit_hist']} |")
    s2 = m["item2_summary"]
    lines += [f"Overall CAL: {s2['cal_before']:.3f} before -> "
              f"{s2['cal_after']:.3f} after ({s2['method']}).", "",
              "## 3. Member spread (per tau)", "",
              "| tau | n | median spread | mean spread |",
              "|---|---|---|---|"]
    for x in m["item3_member_spread"]:
        lines.append(f"| {x['tau']:.0f} | {x['n']} "
                     f"| {x['spread_median']:.1f} "
                     f"| {x['spread_mean']:.1f} |")
    lines += ["", "## 4. TD vs bootstrap consistency: "
              "V(s,tau) - [D30 + V(s+30, tau-30)] (real units)", "",
              "| tau | n | n-step frac | mean | q50 |", "|---|---|---|---|"
              "---|"]
    for x in m["item4_td_consistency"]:
        lines.append(f"| {x['tau']:.0f} | {x['n']} | {x['n_step_frac']:.3f}"
                     f" | {x['mean']:.1f} | {x['q50']:.1f} |")
    t5 = m["item5_delta_head"]
    lines += ["", "## 5. Paired Delta head (C13): damping_fleet - "
              "switching_mix at t0 (v0.3.2 fix #13: the campaign "
              "'__native' arm is a random per-CAV switching schedule, "
              "NOT a fleet-native policy)", ""]
    if t5["n_pairs"]:
        lines += [
            f"n_pairs = {t5['n_pairs']}; sign agreement "
            f"{t5['sign_agreement']:.3f}; Spearman {t5['spearman']:.3f}; "
            f"MAE(model) {t5['mae_model']:.1f} vs MAE(dV=0 baseline) "
            f"{t5['mae_zero_baseline']:.1f}; median pred "
            f"{t5['pred_median']:.1f} vs median real "
            f"{t5['real_median']:.1f}"]
    else:
        lines += ["no CAL pairs available"]
    b = m.get("item6_bar") or {}
    lines += ["", "## 6. Acceptance BAR vs the v0.3.1 selected deployable "
              "model (per tau, CAL)", ""]
    if not b.get("available"):
        lines += [f"**{b.get('note', 'bar not available')}** "
                  f"(looked for {b.get('bar_dir', '')})"]
    else:
        lines += [
            f"BAR run: `{b['bar_dir']}` (MANIFEST sha256 "
            f"`{b['bar_manifest_sha256'][:12]}...`).  PASS per tau "
            f"requires the net to beat the deployable hybrid on BOTH "
            f"MAE(median) and CRPS.  Smoke-run numbers cannot show "
            f"this; the final GPU run produces the verdict.", "",
            "| tau | n | MAE net | MAE bar | CRPS net | CRPS bar | "
            "MAE beat | CRPS beat | BAR |",
            "|---|---|---|---|---|---|---|---|---|"]

        def _f(x):
            return format(x, ".1f") if x is not None \
                and np.isfinite(x) else "nan"
        for x in b.get("rows", []):
            lines.append(
                f"| {x['tau']:.0f} | {x['n']} | {_f(x['mae_net'])} "
                f"| {_f(x['mae_bar'])} | {_f(x['crps_net'])} "
                f"| {_f(x['crps_bar'])} | {x['beats_bar_mae']} "
                f"| {x['beats_bar_crps']} "
                f"| {'PASS' if x['bar_pass'] else 'FAIL'} |")
        ov = b.get("overall", {})
        verdict = "PASS" if ov.get("hard_pass") else "FAIL"
        lines += ["", f"Overall BAR: **{verdict}** "
                      f"({ov.get('n_pass', 0)}/{ov.get('n_taus', 0)} "
                      f"taus pass)."]

    # ------- section 7: PERMANENT baselines + HARD acceptance -----------
    bl = m.get("item7_baselines") or {}

    def _f2(x):
        return format(x, ".1f") if x is not None and np.isfinite(x) \
            else "nan"

    lines += [
        "", "## 7. Baselines on CAL — PERMANENT (B3 items D15/D16)", "",
        "MAE of the median point forecast and CRPS per tau x policy x "
        "chunk for climatology, ridge and a point-GBM fitted on TRAIN on "
        "the diagnosis summary features, against the v0.3 net "
        "(**net**).", "",
        "| chunk | policy | tau | n | MAE clim | CRPS clim | MAE ridge "
        "| CRPS ridge | MAE gbm-base | CRPS gbm-base | MAE net "
        "| CRPS net |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in bl.get("rows", []):
        lines.append(
            f"| {x['chunk']} | {x['policy']} | {x['tau']:.0f} | {x['n']} "
            f"| {_f2(x['mae_climatology'])} | {_f2(x['crps_climatology'])} "
            f"| {_f2(x['mae_ridge'])} | {_f2(x['crps_ridge'])} "
            f"| {_f2(x['mae_gbm_baseline'])} "
            f"| {_f2(x['crps_gbm_baseline'])} "
            f"| {_f2(x['mae_model'])} | {_f2(x['crps_model'])} |")
    lines += [
        "", "### 7a. HARD acceptance vs RIDGE, per tau", "",
        "PASS requires the net to beat RIDGE on BOTH MAE(median) and "
        "CRPS at that tau (CAL, all chunks and policies).", "",
        "| tau | n | MAE net | MAE ridge | CRPS net | CRPS ridge "
        "| MAE beat | CRPS beat | HARD |",
        "|---|---|---|---|---|---|---|---|---|"]
    for a in bl.get("acceptance", []):
        lines.append(
            f"| {a['tau']:.0f} | {a['n']} | {_f2(a['mae_model'])} "
            f"| {_f2(a['mae_ridge'])} | {_f2(a['crps_model'])} "
            f"| {_f2(a['crps_ridge'])} | {a['beats_ridge_mae']} "
            f"| {a['beats_ridge_crps']} "
            f"| {'PASS' if a['hard_pass'] else 'FAIL'} |")
    ovb = bl.get("overall", {})
    lines += [
        "",
        f"Overall HARD acceptance vs ridge: "
        f"**{'PASS' if ovb.get('hard_pass') else 'FAIL'}** "
        f"({ovb.get('n_pass', 0)}/{ovb.get('n_taus', 0)} taus pass)."]
    if bl.get("approx"):
        lines += ["", f"CRPS notes: {bl['approx']}"]
    lines.append("")
    return "\n".join(lines)
