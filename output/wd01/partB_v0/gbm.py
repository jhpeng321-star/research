"""Quantile-GBM value model + permanent ridge/GBM baselines (B3a).

The B3 diagnosis (partB_diag/diag_tau30.py + addendum 5a) showed ridge and
GBM on the summary features beat the v0.2 net by ~9x MAE at tau=30, so a
calibrated quantile-GBM ships as the fallback value model (BRIEF_B3_V03
item D17) and the ridge/point-GBM baselines stay in the eval report
permanently (item D15) behind a per-tau HARD acceptance vs ridge (D16).

Variants (`--gbm-variant`):

  * hybrid (default, B3a follow-up): per tau and target a RIDGE point
    model on the summary features PLUS quantile-GBM models fitted on the
    ridge residuals (y - ridge), same conformal calibration; every
    quantile prediction is ridge(X) + residual_q(X).  Trees cannot
    extrapolate the near-linear delay growth with queue/demand at long
    tau (pure GBM lost to ridge at tau >= 120 in gbm_v02_b3a), so the
    linear model carries the level/trend and the GBM only corrects it.
  * pure: the original quantile-GBM on y directly.
  * auto (v0.3.1 selection): per tau, 'hybrid' vs 'ridge-only' (ridge
    point + conformal residual quantiles from the SAME world-grouped
    TRAIN OOF folds) is compared by TRAIN world-grouped OOB CRPS of the
    delay head ONLY, and the better one ships for that tau (exact tie
    ships ridge, the simpler model).  The choice + both OOB scores are
    recorded in the JSON sidecar, metrics.json and eval_report section 0.
    Selection sees TRAIN rows only: select_variant_by_oob() raises
    CalibrationLeakError if any CALIBRATION row reaches it (unit-tested
    guard); CAL is read once, for the final tables, after the fit.

Predict-side speed (planner benchmark >= 1000 pred/s on a loaded CPU):
all rows are BATCHED per (grid tau, quantile) model into ONE predict
call -- off-grid taus reuse the shared endpoints' batches and
interpolate vectorised -- and every predict path runs inside
threadpool_limits(1).  Measured on the saturated box: the FIRST
unbounded-threads HGB predict paid a ~3 min OpenMP pool spin-up (this
exact overhead made gbm_v02_b3a report 2 predictions/s); at 1 thread a
warm call is ~7-60 ms, so ~5 grid taus x a handful of quantile models
is well above the bar.  load() warms the pool so no timed call pays it.

Summary features are EXACTLY the diagnosis features: per-lane counts and
mean speeds, corridor-third counts, context, demand, plus policy_id and
pre_fix per row.  Per tau and per target (total_delay_ref_s /
ttc_lt3_veh_s / hard_brake_s) a HistGradientBoostingRegressor with
loss="quantile" is fitted per quantile level on TRAIN rows only;
collision_any uses a HistGradientBoostingClassifier.  Post-hoc per-tau
quantile calibration is a conformal shift per (tau, target, quantile):
out-of-fold residuals from a TRAIN k-fold hold-out, shifted by their
empirical q-quantile (marginal hold-out coverage >= nominal), with
monotone (cum-max) enforcement across quantile levels at predict time.

CRPS of the quantile predictive approximates CRPS = 2*Int_0^1 QL_q dq by
trapezoid over the fitted grid plus flat tail extrapolation of QL outside
it; PIT is the piecewise-linear quantile CDF inverted at y, with the tail
mass randomized U(0, q_min) / U(q_max, 1).

Fitting is CPU-only, minutes (this is NOT net training; the owner rule
"no CPU training" refers to the net).  All sklearn fits run inside
threadpool_limits (default 2 OpenMP threads): unbounded HGB on a box
whose CPU is saturated by an unrelated process thrashes (>30 min per
fit measured; limited: ~5-9 s).  Early stopping (n_iter_no_change=15)
bounds the iteration count with a WORLD-GROUPED validation holdout
passed via fit(X_val=...) (v0.3.2 fix #10: sklearn's internal random-row
validation split lets a world straddle fit and validation); without a
usable grouped holdout (or on sklearn without X_val) early stopping is
OFF at fixed max_iter.  A deployable feature set additionally fixes
pre_fix to the post-fix value on every predict path (v0.3.2 fix #11).
ZERO simulation.
"""
from __future__ import annotations

import contextlib
import json
import time
import weakref
from pathlib import Path

import numpy as np

from .data import TAUS
from .model import POLICIES_V04A

SCHEMA = "wd01_partB_v0_gbm/4"       # /4: sensing variants + grouped ES +
LEGACY_SCHEMAS = ("wd01_partB_v0_gbm/1",      # pre_fix fixed at predict;
                  "wd01_partB_v0_gbm/2",      # /1 = pure; /2 = hybrid;
                  "wd01_partB_v0_gbm/3")      # /3 = auto per-tau selection
PKL_NAME = "gbm_value.pkl"
JSON_NAME = "gbm_value.json"

# quantile grid: 0.1/0.9 give the central 10-90 band; the outer pair
# extends the CRPS/PIT support toward the tails
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
# the diagnostic exposure/brake heads only need the inner grid (the
# planner value is v_delay); keeps the fit count ~40% lower
INNER_QUANTILES = (0.1, 0.5, 0.9)

TARGETS = ("delay", "exp", "brake")
TARGET_COL = {"delay": "total_delay_ref_s", "exp": "ttc_lt3_veh_s",
              "brake": "hard_brake_s", "coll": "collision_any"}


@contextlib.contextmanager
def _limit_threads(n: int):
    """Cap OpenMP/BLAS threads for the enclosed fits (threadpoolctl when
    available; the fits still work unlimited when it is missing)."""
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=n):
            yield
    except ImportError:
        yield

# summary-matrix cache: summary_features() rebuilds lane-count reductions
# over all decision times, so cache one matrix per dataset instance
_SUMMARY_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


# --------------------------------------------------------------- features --

# deployable demand block (build_v031.py): per-minute observed entrance
# rates /50 -- realised insertions per entrance over (t-60, t], (t-180, t],
# (t-300, t], then the CAV subset of the same counts
DEMAND_DEP_NAMES = (
    "obs_ent_main_60", "obs_ent_main_180", "obs_ent_main_300",
    "obs_ent_ramp_60", "obs_ent_ramp_180", "obs_ent_ramp_300",
    "obs_cav_main_60", "obs_cav_main_180", "obs_cav_main_300",
    "obs_cav_ramp_60", "obs_cav_ramp_180", "obs_cav_ramp_300")

# v0.3.2 #5 strict (HEADLINE, CAV-STRICT) demand block: the designated-CAV
# counts only -- no all-vehicle entrance counts, no penetration-revealing
# ratio (build_v032.py _feat_dep_str.npz sidecars)
STRICT_DEMAND_DEP_NAMES = DEMAND_DEP_NAMES[6:]

# feature-set switch: oracle = historical full-state blocks;
# deployable_strict = v0.3.2 HEADLINE (CAV self-reports only);
# deployable_detector = v0.3.2 labelled SENSITIVITY (all-vehicle entrance
# counts + queues); "deployable" = deprecated v0.3.1 alias of the
# detector-shaped blocks without the #1 zeroed context col.
FEATURE_SETS = ("oracle", "deployable", "deployable_detector",
                "deployable_strict")

# ---- v0.4a design (WD-01-PB-V04A) -----------------------------------------
# A v0.4a dataset (ds.v04a) uses a different design matrix:
#   [ lane / context / demand summary  +  per-action CAV counts at t
#     (native / damping / pi / unknown: the per-CAV action tokens with the
#     unknown mask, CAV_RAW 20),
#     one-hot canonical policy class (POLICIES_V04A),
#     planner-known schedule summary pi_native / pi_damping / pi_pi ]
# and NO pre_fix column (registration section 3: penetration, pre_fix,
# speed_mode and world strings are never model inputs).
V04A_ACTION_COUNT_NAMES = ("cav_n_native", "cav_n_damping", "cav_n_pi",
                           "cav_n_unknown")
V04A_POLICY_NAMES = tuple(f"pol_{p}" for p in POLICIES_V04A)
V04A_PI_NAMES = ("pi_native", "pi_damping", "pi_pi")
V04A_FLEET_ACTION = {"native_fleet": 0, "damping_fleet": 1, "pi_fleet": 2}
DESIGN_LEGACY, DESIGN_V04A = "legacy", "v04a"

# v0.3.2 fix #11: pre_fix stays a training feature, but a deployable model
# ALWAYS predicts at the post-fix value (deployment runs post-fix), so the
# pre_fix column is overridden to 0.0 on every predict path and the report
# says so.
PRE_FIX_PREDICT_VALUE = 0.0
PRE_FIX_PREDICT_NOTE = (
    "deployable feature set: pre_fix is a label/stratifier only -- kept "
    "as a training feature, FIXED to the post-fix value "
    f"{PRE_FIX_PREDICT_VALUE:.1f} on every predict path (owner decision, "
    "output/wd01/OWNER_DECISIONS_WD01.md); the report says so")


def summary_features(ds) -> tuple[np.ndarray, list[str]]:
    """[G, F] summary matrix per global decision time (the exact
    diag_tau30.py features; G = decision times).

    Columns follow ds.feature_set: oracle keeps the historical
    lane + context + future-schedule demand block; the deployable sets
    have the same lane/context columns (context[2] time-to-branch and --
    v0.3.2 fix #1 -- context[1] since-t0 are identically 0 in the
    deployable blocks) with demand_ replaced by the observed entrance/CAV
    rates: 12 columns for the detector blocks, the 6 CAV-only columns of
    STRICT_DEMAND_DEP_NAMES for the strict (HEADLINE) blocks."""
    cached = _SUMMARY_CACHE.get(ds)
    if cached is not None:
        return cached
    F = ds.feats
    lb = F["lane_bins"]                      # [G, 3, L, 3] count, mean v, std
    cnt = lb[..., 0]
    spd = lb[..., 1]
    lane_n = cnt.sum(axis=2)                                 # [G, 3]
    lane_v = (cnt * spd).sum(axis=2) / np.maximum(lane_n, 1e-6)
    L = cnt.shape[2]
    thirds = [cnt[:, :, i * L // 3:(i + 1) * L // 3].sum(axis=(1, 2))
              for i in range(3)]
    X = np.column_stack([lane_n, lane_v, *thirds, F["context"],
                         F["demand"]]).astype(np.float64)
    if getattr(ds, "feature_set", "oracle") == "oracle":
        dem_names = [f"demand_{i}" for i in range(F["demand"].shape[1])]
    elif F["demand"].shape[1] == len(STRICT_DEMAND_DEP_NAMES):
        dem_names = list(STRICT_DEMAND_DEP_NAMES)
    else:
        dem_names = DEMAND_DEP_NAMES[:F["demand"].shape[1]]
    names = ([f"lane{i}_count" for i in range(3)]
             + [f"lane{i}_vmean" for i in range(3)]
             + [f"third{i}_count" for i in range(3)]
             + [f"ctx_{i}" for i in range(F["context"].shape[1])]
             + list(dem_names))
    if getattr(ds, "v04a", False):
        # per-action CAV counts from the v0.4 CAV tokens (action one-hot
        # at token cols 5:10 = native/damping/pi/wall/unknown; wall is
        # empty in v0.4a) over the PRESENT tokens (cav_mask)
        tok, msk = F["cav_tokens"], F["cav_mask"] > 0
        act = tok[..., 5:10]
        cnt_a = np.stack([(act[..., k] > 0.5) & msk for k in (0, 1, 2, 4)],
                         axis=-1).sum(axis=1).astype(np.float64)
        X = np.column_stack([X, cnt_a])
        names = names + list(V04A_ACTION_COUNT_NAMES)
    try:
        _SUMMARY_CACHE[ds] = (X, names)
    except TypeError:                               # pragma: no cover
        pass
    return X, names


def row_feature_matrix(ds, row_idx=None,
                       pre_fix_override: float | None = None) -> tuple[
        np.ndarray, list[str]]:
    """[N, F+2] per LABEL row (or a given row subset): summary features at
    the row's decision time + policy_id + pre_fix (the exact diag_tau30
    design matrix).

    pre_fix_override (v0.3.2 fix #11): when given, the pre_fix column is
    replaced by this constant on every row -- the deployable predict path
    fixes it to the post-fix value PRE_FIX_PREDICT_VALUE."""
    S, names = summary_features(ds)
    r = ds.rows
    if getattr(ds, "v04a", False):
        ri = (np.arange(len(r)) if row_idx is None
              else np.asarray(row_idx, dtype=np.int64))
        return v04a_design(
            S[r["t_idx"].to_numpy()[ri]],
            r["policy"].to_numpy()[ri],
            np.column_stack([r[c].to_numpy()[ri] for c in V04A_PI_NAMES]),
        ), names + list(V04A_POLICY_NAMES) + list(V04A_PI_NAMES)
    if row_idx is None:
        t_idx, pol, pre = (r["t_idx"].to_numpy(),
                           r["policy_id"].to_numpy(),
                           r["pre_fix"].astype(float).to_numpy())
    else:
        ri = np.asarray(row_idx, dtype=np.int64)
        t_idx = r["t_idx"].to_numpy()[ri]
        pol = r["policy_id"].to_numpy()[ri]
        pre = r["pre_fix"].astype(float).to_numpy()[ri]
    if pre_fix_override is not None:
        pre = np.full(len(pre), float(pre_fix_override))
    return np.column_stack([S[t_idx], pol, pre]), names + ["policy_id",
                                                           "pre_fix"]


def v04a_design(S_rows: np.ndarray, policy_names, pi_shares) -> np.ndarray:
    """[N, F + 5 + 3] v0.4a design: state summary rows (already carrying
    the per-action CAV counts), one-hot policy class, schedule summary."""
    pol = np.asarray(policy_names)
    onehot = np.stack([(pol == p) for p in POLICIES_V04A],
                      axis=1).astype(np.float64)
    if not onehot.sum(axis=1).all():
        bad = sorted(set(pol[onehot.sum(axis=1) == 0].tolist()))
        raise ValueError(f"unknown v0.4a policy classes {bad}")
    return np.column_stack([np.asarray(S_rows, dtype=np.float64), onehot,
                            np.asarray(pi_shares, dtype=np.float64)])


def counterfactual_design(S_rows: np.ndarray, policy: str) -> np.ndarray:
    """Design rows for the SAME state under a FLEET policy (native_fleet /
    damping_fleet / pi_fleet): every present CAV (the sum of the four
    per-action count columns) carries the fleet action, one-hot policy,
    pi = one-hot fleet action.  Used to score V(s, tau, a) on a state
    logged under another arm (contrast queries)."""
    if policy not in V04A_FLEET_ACTION:
        raise ValueError(f"counterfactual policy must be a fleet class, "
                         f"got {policy!r}")
    S2 = np.array(S_rows, dtype=np.float64, copy=True)
    n = S2[:, -4:].sum(axis=1)
    S2[:, -4:] = 0.0
    S2[:, -4 + V04A_FLEET_ACTION[policy]] = n
    pi = np.zeros((len(S2), 3))
    pi[:, V04A_FLEET_ACTION[policy]] = 1.0
    return v04a_design(S2, np.full(len(S2), policy), pi)


def _aux_mask(rows) -> np.ndarray:
    """Per-row auxiliary flag (v0.4a: training only); all False for the
    legacy roots."""
    if "aux" in rows.columns:
        return rows["aux"].to_numpy(dtype=bool)
    return np.zeros(len(rows), dtype=bool)


def _tau_mask(rows, tau, split=None):
    m = np.isclose(rows["tau"].to_numpy(), tau)
    if split is not None:
        m &= (rows["split"] == split).to_numpy()
    return m


# ------------------------------------------------- selection guard (v0.3.1) --

class CalibrationLeakError(RuntimeError):
    """Variant selection / conformal calibration received a CALIBRATION
    row.  Selection and calibration run on TRAIN rows only; CAL is read
    once, for the final report tables."""


def guard_no_calibration_rows(splits, context: str = "") -> None:
    """Raise CalibrationLeakError unless every split value is TRAIN.

    `splits` is the per-row split array of exactly the rows the caller
    is about to use for variant selection or conformal calibration."""
    vals = np.asarray(splits, dtype=object)
    bad = sorted({str(s) for s in vals.tolist() if s != "TRAIN"})
    if bad:
        raise CalibrationLeakError(
            f"{context}: received non-TRAIN row(s) with split(s) "
            f"{bad} -- variant selection and conformal calibration must "
            f"see TRAIN rows only (CAL is read once, for the final "
            f"tables)")


# ------------------------------------------------------------ the model --

class GbmValueModel:
    """Per-(tau, target, quantile) HGB quantile regressors + per-tau
    collision classifier, with conformal quantile shifts.

    variant="hybrid" (default) fits the HGBs on the RIDGE RESIDUALS of a
    per-(tau, target) ridge point model (self.point_models) and every
    prediction is ridge + residual quantile; variant="pure" keeps the
    original quantile-GBM on y.  The per-tau world-grouped OOF delay
    quantile predictions (the same folds the conformal shifts use) are
    kept in self.oof_delay for the TRAIN-internal HARD-acceptance table.

    Planner-facing predict API: median + quantiles per tau/policy
    (predict_quantiles_rows / median_rows / point_global).
    """

    def __init__(self, quantiles: tuple = QUANTILES, max_iter: int = 300,
                 learning_rate: float = 0.05, cal_folds: int = 2,
                 seed: int = 0, threads: int = 2,
                 variant: str = "hybrid", feature_set: str = "oracle"):
        if variant not in ("hybrid", "pure", "auto"):
            raise ValueError(f"unknown gbm variant {variant!r} "
                             f"(hybrid | pure | auto)")
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {feature_set!r} "
                             f"({' | '.join(FEATURE_SETS)})")
        self.variant = variant
        self.feature_set = feature_set
        self.design = DESIGN_LEGACY       # set to v04a by fit(ds.v04a)
        self.aux_excluded_from_calibration = False
        # v0.3.2 fix #11: pre_fix stays a training feature but a
        # deployable model always PREDICTS at the post-fix value
        # (PRE_FIX_PREDICT_VALUE); oracle keeps the row's true flag.
        self.pre_fix_at_predict = (None if feature_set == "oracle"
                                   else PRE_FIX_PREDICT_VALUE)
        self.quantiles = tuple(sorted(float(q) for q in quantiles))
        if 0.5 not in self.quantiles:
            raise ValueError("the quantile grid must contain 0.5 (median)")
        self.target_quantiles = {
            "delay": self.quantiles,
            "exp": tuple(q for q in self.quantiles
                         if q in INNER_QUANTILES),
            "brake": tuple(q for q in self.quantiles
                           if q in INNER_QUANTILES)}
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.cal_folds = cal_folds
        self.seed = int(seed)
        self.threads = int(threads)          # fit threads
        self.predict_threads = 1             # 1 beats 2 warm on a loaded box
        self.taus: list[float] = []
        self.feature_names: list[str] = []
        self.models: dict = {}            # (tau, target, q) -> regressor
        self.point_models: dict = {}      # (tau, target) -> ridge (hybrid)
        self.coll: dict = {}              # tau -> classifier | rate float
        self.shifts: dict = {}            # (tau, target, q) -> float
        self.oof_delay: dict = {}         # tau -> {q: OOF pred [n_tr] f32}
        self.oof_grouping: str = ""       # world-grouped | row-kfold | in-fit
        # v0.3.2 fix #10: HGB early stopping must never split random ROWS
        # (a world would straddle fit/validation); the validation split is
        # a world-grouped holdout passed via fit(X_val=...) when sklearn
        # supports it, else early stopping is OFF at fixed max_iter.
        self.es_grouping: str = ""        # world-grouped | fixed-iter
        self.es_val_support = None        # sklearn X_val support (probed)
        # auto variant (v0.3.1): per-tau shipped variant + the TRAIN-OOB
        # selection record (sidecar/metrics/report evidence)
        self.tau_variants: dict = {}      # tau -> "hybrid" | "ridge" | "pure"
        self.selection: list = []         # select_variant_by_oob() records
        self.fit_seconds: dict = {}
        self.sklearn_version = ""
        self.data_manifest_sha256 = ""
        self._qcache: dict = {}           # (tau, target) -> [(q, model)]

    # ------------------------------------------------------------- fit --
    def _hgb(self, target: str, q: float, early_stopping: bool = True):
        from sklearn.ensemble import HistGradientBoostingRegressor
        if early_stopping:
            return HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=self.max_iter,
                learning_rate=self.learning_rate, random_state=self.seed,
                early_stopping=True, n_iter_no_change=15,
                validation_fraction=0.1)
        # fixed max_iter (early stopping OFF): used when no world-grouped
        # validation holdout is possible (v0.3.2 fix #10 -- never split
        # random rows)
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=self.max_iter,
            learning_rate=self.learning_rate, random_state=self.seed,
            early_stopping=False)

    def _ridge(self):
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)

    def _es_val_split(self, X, y, grp) -> tuple | None:
        """World-grouped early-stopping holdout (v0.3.2 fix #10): one
        GroupShuffleSplit 10% holdout over WORLD ids, so no world straddles
        fit and validation rows.  Returns (fit_idx, val_idx) or None when
        the slice has fewer than two worlds (then early stopping stays OFF
        at fixed max_iter -- a random-row validation split would leak)."""
        if grp is None:
            return None
        n = len(grp)
        n_groups = len(np.unique(grp))
        if n < 20 or n_groups < 2:
            return None
        from sklearn.model_selection import GroupShuffleSplit
        (fit_i, val_i), = GroupShuffleSplit(
            n_splits=1, test_size=0.1,
            random_state=self.seed).split(np.zeros((n, 1)), groups=grp)
        if len(val_i) == 0 or len(fit_i) == 0:
            return None
        return fit_i, val_i

    def _fit_hgb(self, target: str, q: float, X, y, grp):
        """Fit one quantile HGB with the world-grouped ES holdout passed
        as the explicit validation set (sklearn >= 1.7 X_val), falling
        back to fixed max_iter (early stopping OFF) when unsupported or
        the slice is too small.  Records self.es_grouping."""
        es = self._es_val_split(X, y, grp)
        if es is None:
            self.es_grouping = "fixed-iter"
            return self._hgb(target, q, early_stopping=False) \
                .fit(X, y)
        fit_i, val_i = es
        if self.es_val_support is None:
            # probe once: does this sklearn accept explicit validation
            # data on HGB.fit? (added in 1.7)
            try:
                import inspect
                self.es_val_support = "X_val" in inspect.signature(
                    type(self._hgb(target, q)).fit).parameters
            except (ValueError, TypeError):         # pragma: no cover
                self.es_val_support = False
        if not self.es_val_support:
            self.es_grouping = "fixed-iter"
            return self._hgb(target, q, early_stopping=False) \
                .fit(X, y)
        self.es_grouping = "world-grouped"
        return self._hgb(target, q, early_stopping=True).fit(
            X[fit_i], y[fit_i], X_val=X[val_i], y_val=y[val_i])

    def _folds(self, Xtr, grp, n_tr: int):
        """OOF fold indices + grouping tag; world-grouped GroupKFold when
        the groups allow it (a world must not appear on both sides of a
        calibration fold), row KFold when too few groups, None (in-fit
        residuals) when the slice is tiny."""
        if n_tr < 20 or self.cal_folds < 2:
            return None, "in-fit"
        folds = max(2, min(self.cal_folds, n_tr // 10))
        n_groups = len(np.unique(grp))
        if n_groups >= folds:
            from sklearn.model_selection import GroupKFold
            return (list(GroupKFold(n_splits=folds)
                         .split(Xtr, groups=grp)), "world-grouped")
        from sklearn.model_selection import KFold
        return (list(KFold(n_splits=folds, shuffle=True,
                           random_state=self.seed).split(Xtr)),
                "row-kfold")

    def _residual_target(self, target, tau, Xtr, y):
        """Training target for the quantile models: y (pure) or the ridge
        residual y - ridge(X) (hybrid, and auto's hybrid-chosen taus);
        fits+stores the ridge itself."""
        if self.variant == "pure":
            return y
        rid = self._ridge().fit(Xtr, y)
        self.point_models[(float(tau), target)] = rid
        return y - rid.predict(Xtr)

    def _oof_pred(self, target, tau, q, Xtr, y, fold_idx,
                  grp=None) -> np.ndarray:
        """Out-of-fold predictions of the FULL pipeline (hybrid: ridge +
        residual GBM refitted per fold; pure: the quantile GBM; auto's
        hybrid candidate uses the hybrid pipeline).  Fold-internal fits
        use the world-grouped ES holdout (fix #10; grp = fold rows'
        world ids)."""
        if not fold_idx:
            oof = self.models[(float(tau), target, q)].predict(Xtr)
            if self.variant != "pure":
                oof = oof + self.point_models[(float(tau),
                                               target)].predict(Xtr)
            return oof
        oof_q = np.full(len(y), np.nan)
        for fit_i, cal_i in fold_idx:
            g_f = None if grp is None else grp[fit_i]
            if self.variant != "pure":
                rf = self._ridge().fit(Xtr[fit_i], y[fit_i])
                zf = y[fit_i] - rf.predict(Xtr[fit_i])
                gf = self._fit_hgb(target, q, Xtr[fit_i], zf, g_f)
                oof_q[cal_i] = rf.predict(Xtr[cal_i]) \
                    + gf.predict(Xtr[cal_i])
            else:
                gf = self._fit_hgb(target, q, Xtr[fit_i], y[fit_i], g_f)
                oof_q[cal_i] = gf.predict(Xtr[cal_i])
        return oof_q

    def _hybrid_candidate_oof(self, target, tau, Xtr, y, fold_idx,
                              grp=None):
        """OOF delay-side candidate predictions of the HYBRID pipeline
        (used by auto selection before any final model exists).  Without
        folds (smoke) the candidates are in-fit temporary fits."""
        qs = self.target_quantiles[target]
        if fold_idx:
            return {q: self._oof_pred(target, tau, q, Xtr, y, fold_idx,
                                      grp=grp)
                    for q in qs}
        rf = self._ridge().fit(Xtr, y)
        base = rf.predict(Xtr)
        return {q: base + self._fit_hgb(target, q, Xtr, y - base,
                                        None).predict(Xtr)
                for q in qs}

    def _variant_at(self, tau) -> str:
        """Shipped variant for one grid tau ('ridge' where auto chose the
        ridge-only candidate; the global variant elsewhere)."""
        v = self.tau_variants.get(float(tau))
        if v:
            return v
        return "hybrid" if self.variant == "auto" else self.variant

    def _ridge_oof(self, Xtr, y, fold_idx):
        """World-grouped OOF predictions of the ridge point model on the
        SAME folds the hybrid candidate's conformal shifts use (the
        ridge-only candidate of --gbm-variant auto; in-fit predictions
        when there are no folds, mirroring _oof_pred)."""
        if not fold_idx:
            return self._ridge().fit(Xtr, y).predict(Xtr)
        oof = np.full(len(y), np.nan)
        for fit_i, cal_i in fold_idx:
            rf = self._ridge().fit(Xtr[fit_i], y[fit_i])
            oof[cal_i] = rf.predict(Xtr[cal_i])
        return oof

    def _oof_candidate(self, cand, target, tau, q, Xtr, yf, fold_idx,
                       grp=None):
        """Candidate OOF predictions, computed lazily once per
        (target, q) and reused for both selection and the shifts."""
        if cand is None:
            return self._oof_pred(target, tau, q, Xtr, yf, fold_idx,
                                  grp=grp)
        got = cand.get(target, {}).get(q)
        if got is None:
            got = self._oof_pred(target, tau, q, Xtr, yf, fold_idx,
                                 grp=grp)
            cand.setdefault(target, {})[q] = got
        return got

    def fit(self, ds, log=print) -> "GbmValueModel":
        import sklearn
        self.sklearn_version = sklearn.__version__
        self.data_manifest_sha256 = ds.man.sha256
        r = ds.rows
        X, self.feature_names = row_feature_matrix(ds)
        if getattr(ds, "v04a", False):
            self.design = DESIGN_V04A
            self.pre_fix_at_predict = None      # no pre_fix column at all
            self.aux_excluded_from_calibration = True
        aux_all = _aux_mask(r)
        grp_all = (r["chunk_id"].astype(str) + "|"
                   + r["world"].astype(str)).to_numpy()
        t0_all = time.perf_counter()
        with _limit_threads(self.threads):
            for tau in TAUS:
                tr = _tau_mask(r, tau, "TRAIN")
                n_tr = int(tr.sum())
                if n_tr == 0:
                    continue
                self.taus.append(float(tau))
                t0 = time.perf_counter()
                y_all = {t: r[TARGET_COL[t]].to_numpy()[tr]
                         for t in TARGETS}
                Xtr = X[tr]
                # v0.4a: auxiliary rows TRAIN the models but never enter
                # selection or conformal calibration (registration s.2)
                sc = ~aux_all[tr]
                fold_idx, self.oof_grouping = self._folds(Xtr, grp_all[tr],
                                                          n_tr)
                splits_tau = r["split"].to_numpy()[tr]   # TRAIN by mask;
                # the guard below turns any regression into a hard error
                guard_no_calibration_rows(
                    splits_tau,
                    f"conformal calibration/selection tau={tau:.0f}")
                cand: dict = {}          # (target -> q -> OOF preds) cache
                rid_oof: dict = {}       # target -> ridge-only OOF preds
                chosen = self.variant
                if self.variant == "auto":
                    # --- per-tau selection on TRAIN world-grouped OOB
                    #     CRPS of the delay head ONLY (hybrid candidate
                    #     vs ridge-only on the SAME folds) ---
                    y_d = y_all["delay"]
                    rid_oof = {t: self._ridge_oof(Xtr, y_all[t], fold_idx)
                               for t in TARGETS}
                    cand["delay"] = self._hybrid_candidate_oof(
                        "delay", tau, Xtr, y_d, fold_idx, grp=grp_all[tr])

                    def _conformal(preds):
                        """per-q conformal shift + monotone, exactly the
                        shipped calibration for those OOF predictions."""
                        out = {}
                        for q, p in preds.items():
                            d = (y_d - p)[sc]
                            d = d[np.isfinite(d)]
                            out[q] = p + (float(np.quantile(d, q))
                                          if len(d) else 0.0)
                        return self._monotone(out)

                    ph = _conformal(cand["delay"])
                    _rd = (y_d - rid_oof["delay"])[sc]
                    pr = self._monotone({
                        q: rid_oof["delay"]
                        + float(np.quantile(_rd[np.isfinite(_rd)], q))
                        for q in cand["delay"]})
                    entry = select_variant_by_oob(
                        tau, "delay", y_d[sc], splits_tau[sc],
                        {q: v[sc] for q, v in ph.items()},
                        {q: v[sc] for q, v in pr.items()})
                    entry["folds"] = self.oof_grouping
                    self.selection.append(entry)
                    chosen = entry["chosen"]
                    self.tau_variants[float(tau)] = chosen
                    log(f"gbm[auto]: tau={tau:.0f} OOB delay CRPS "
                        f"hybrid={entry['oob_crps']['hybrid']:.1f} vs "
                        f"ridge={entry['oob_crps']['ridge']:.1f} -> ships "
                        f"{chosen}")
                else:
                    self.tau_variants[float(tau)] = self.variant
                for target in TARGETS:
                    yf = y_all[target]
                    if chosen == "ridge":
                        # --- ridge-only tau: ridge point + conformal
                        #     residual quantiles from the SAME
                        #     world-grouped OOF folds as the candidate ---
                        rid = self._ridge().fit(Xtr, yf)
                        self.point_models[(float(tau), target)] = rid
                        resid = (yf - rid_oof[target])[sc]
                        fin = resid[np.isfinite(resid)]
                        oof_pt = rid_oof[target]
                        for q in self.target_quantiles[target]:
                            self.shifts[(float(tau), target, q)] = \
                                float(np.quantile(fin, q)) \
                                if len(fin) else 0.0
                            if target == "delay":
                                self.oof_delay.setdefault(
                                    float(tau), {})[q] = \
                                    np.full(len(yf), oof_pt,
                                            dtype=np.float32)
                        continue
                    ztr = self._residual_target(target, tau, Xtr, yf)
                    # --- final models on ALL TRAIN rows of this tau ---
                    for q in self.target_quantiles[target]:
                        reg = self._fit_hgb(target, q, Xtr, ztr,
                                            grp_all[tr])
                        self.models[(float(tau), target, q)] = reg
                    # --- conformal shifts: q-quantile of OOF residuals
                    #     of the FULL pipeline; delay OOF kept for the
                    #     TRAIN-internal acceptance table ---
                    for q in self.target_quantiles[target]:
                        oof_q = self._oof_candidate(cand, target, tau, q,
                                                    Xtr, yf, fold_idx,
                                                    grp=grp_all[tr])
                        d = (yf - oof_q)[sc]
                        d = d[np.isfinite(d)]
                        self.shifts[(float(tau), target, q)] = \
                            float(np.quantile(d, q)) if len(d) else 0.0
                        if target == "delay":
                            self.oof_delay.setdefault(float(tau), {})[q] = \
                                np.asarray(oof_q, dtype=np.float32)
                y_coll = r["collision_any"].to_numpy()[tr].astype(int)
                if len(np.unique(y_coll)) >= 2:
                    from sklearn.ensemble import HistGradientBoostingClassifier
                    es = self._es_val_split(Xtr, y_coll, grp_all[tr])
                    use_es = False
                    if es is not None and self.es_val_support is not False:
                        # world-grouped validation holdout (fix #10); it
                        # needs BOTH classes on each side (an explicit
                        # validation set with an unseen label errors)
                        fit_i, val_i = es
                        use_es = (np.bincount(y_coll[fit_i],
                                              minlength=2).min() > 0
                                  and np.bincount(y_coll[val_i],
                                                  minlength=2).min() > 0)
                    if use_es:
                        clf = HistGradientBoostingClassifier(
                            max_iter=self.max_iter,
                            learning_rate=self.learning_rate,
                            random_state=self.seed, early_stopping=True,
                            n_iter_no_change=15).fit(
                                Xtr[fit_i], y_coll[fit_i],
                                X_val=Xtr[val_i], y_val=y_coll[val_i])
                        self.es_grouping = "world-grouped"
                    else:
                        clf = HistGradientBoostingClassifier(
                            max_iter=self.max_iter,
                            learning_rate=self.learning_rate,
                            random_state=self.seed,
                            early_stopping=False).fit(Xtr, y_coll)
                        self.es_grouping = "fixed-iter"
                    self.coll[float(tau)] = clf
                else:                   # single-class fallback: base rate
                    self.coll[float(tau)] = float(y_coll.mean())
                dt = time.perf_counter() - t0
                self.fit_seconds[float(tau)] = round(dt, 1)
                log(f"gbm[{self.variant}]: tau={tau:.0f} fitted "
                    f"({n_tr} TRAIN rows, "
                    f"{sum(len(v) for v in self.target_quantiles.values())}"
                    f" quantile models + coll, folds={self.oof_grouping}"
                    f") in {dt:.1f} s")
        log(f"gbm[{self.variant}]: total fit {time.perf_counter() - t0_all:.1f} s "
            f"(taus={self.taus}, threads={self.threads}, "
            f"OOF={self.oof_grouping})")
        return self

    # --------------------------------------------------------- predict --
    def matrix_for_design(self, S_rows, policy: str) -> np.ndarray:
        """v0.4a counterfactual design rows (fleet policy) for state
        summary rows S_rows; see counterfactual_design."""
        return counterfactual_design(S_rows, policy)

    def _pre_fix_override(self, pre_fix) -> float | None:
        """v0.3.2 fix #11: a deployable model always predicts at the
        post-fix pre_fix value, whatever the caller passes (the report
        says so; see PRE_FIX_PREDICT_NOTE)."""
        if self.pre_fix_at_predict is None:
            return None
        return float(self.pre_fix_at_predict)

    def _matrix_global(self, ds, g_idx, policy, pre_fix) -> np.ndarray:
        S, _ = summary_features(ds)
        g_idx = np.asarray(g_idx, dtype=np.int64)
        X = S[g_idx]
        pre = self._pre_fix_override(pre_fix)
        if pre is not None:
            pre_fix = np.full(len(g_idx), pre)
        return np.column_stack([
            X, np.broadcast_to(np.asarray(policy, dtype=np.float64),
                               (len(g_idx),)),
            np.broadcast_to(np.asarray(pre_fix, dtype=np.float64),
                            (len(g_idx),))])

    def _row_matrix(self, ds, row_idx) -> np.ndarray:
        return row_feature_matrix(
            ds, row_idx,
            pre_fix_override=self._pre_fix_override(None))[0]

    def design_rows(self, ds, row_idx) -> np.ndarray:
        """Design matrix of LABEL rows in the model's own design."""
        return self._row_matrix(ds, row_idx)

    def _qmodels(self, tau: float, target: str) -> list:
        """[(q, model)] sorted by q, cached per (tau, target) -- the
        planner path must not re-walk the models dict per call.  Empty
        for a ridge-only tau (no GBM quantile models were fitted)."""
        if self._variant_at(tau) == "ridge":
            return []
        cached = self._qcache.get((tau, target))
        if cached is None:
            cached = [(q, self.models[(tau, target, q)])
                      for q in sorted(self.target_quantiles[target])]
            self._qcache[(tau, target)] = cached
        return cached

    def _base_part(self, X, tau: float, target: str):
        """The linear point part every quantile prediction adds (hybrid,
        and auto's hybrid-chosen taus); 0.0 for the pure variant."""
        if self._variant_at(tau) == "pure":
            return 0.0
        return self.point_models[(float(tau), target)].predict(X)

    def _predict_raw_batch(self, X, tau: float, target: str) -> dict:
        """Raw (uncalibrated) predictions for rows ALL at one grid tau:
        ONE predict call per (q, model), the shared ridge predicted once.
        A ridge-only tau predicts the ridge point at every level (the
        level structure is entirely in the conformal shifts)."""
        if self._variant_at(tau) == "ridge":
            base = self.point_models[(float(tau), target)].predict(X)
            return {q: base for q in self.target_quantiles[target]}
        base = self._base_part(X, tau, target)
        return {q: mdl.predict(X) + base
                for q, mdl in self._qmodels(tau, target)}

    def predict_quantiles_matrix(self, X, taus, target: str = "delay",
                                 calibrated: bool = True) -> dict:
        """Calibrated quantile predictions {q: [n]}; taus must be on the
        trained grid (off-grid taus get NaN -- use point_matrix).

        Calibrated = per-level conformal shift THEN monotone (cum-max)
        rearrangement, so the shipped quantiles are always monotone; the
        median of this path is EXACTLY point_matrix's on-grid median."""
        qs = self.target_quantiles[target]
        taus = np.asarray(taus, dtype=np.float64)
        out = {q: np.full(len(X), np.nan) for q in qs}
        if not len(taus):
            return out
        with _limit_threads(self.predict_threads):
            for tau in self.taus:
                m = np.isclose(taus, tau)
                if not m.any():
                    continue
                raw = self._predict_raw_batch(X[m], tau, target)
                preds = raw
                if calibrated:
                    preds = {q: raw[q] + self.shifts.get((tau, target, q),
                                                         0.0)
                             for q in raw}
                preds = self._monotone(preds)
                for q in qs:
                    out[q][m] = preds[q]
        return out

    def predict_quantiles_rows(self, ds, row_idx, target: str = "delay",
                               calibrated: bool = True) -> dict:
        X = self._row_matrix(ds, row_idx)
        taus = np.asarray(ds.rows["tau"].to_numpy()[row_idx])
        return self.predict_quantiles_matrix(X, taus, target,
                                             calibrated=calibrated)

    @staticmethod
    def _monotone(preds: dict) -> dict:
        out, run = {}, None
        for q in sorted(preds):
            run = preds[q] if run is None else np.maximum(run, preds[q])
            out[q] = run
        return out

    def median_rows(self, ds, row_idx, target: str = "delay") -> np.ndarray:
        return self.predict_quantiles_rows(ds, row_idx, target)[0.5]

    def crps_rows(self, ds, row_idx, target: str = "delay") -> np.ndarray:
        y = ds.rows[TARGET_COL[target]].to_numpy()[np.asarray(row_idx)]
        preds = self.predict_quantiles_rows(ds, row_idx, target)
        return crps_from_quantiles(sorted(preds), preds, y)

    def pit_rows(self, ds, row_idx, target: str = "delay",
                 seed: int = 20260928) -> np.ndarray:
        y = ds.rows[TARGET_COL[target]].to_numpy()[np.asarray(row_idx)]
        preds = self.predict_quantiles_rows(ds, row_idx, target)
        return pit_from_quantiles(sorted(preds), preds, y, seed=seed)

    def coverage_rows(self, ds, row_idx, target: str = "delay") -> np.ndarray:
        pr = self.predict_quantiles_rows(ds, row_idx, target)
        y = ds.rows[TARGET_COL[target]].to_numpy()[np.asarray(row_idx)]
        return (y >= pr[0.1]) & (y <= pr[0.9])

    def _median_at_grid(self, X, tau: float, target: str) -> np.ndarray:
        """Calibrated median for rows ALL at one grid tau: max over the
        q <= 0.5 models of (raw + shift) -- identical to taking [0.5] of
        predict_quantiles_matrix, but only the 4 lower-tail models are
        queried and each in ONE batched call.  A ridge-only tau is the
        ridge point + its median residual offset."""
        tau = float(tau)
        if self._variant_at(tau) == "ridge":
            base = self.point_models[(tau, target)].predict(X)
            off = max(self.shifts.get((tau, target, q), 0.0)
                      for q in self.target_quantiles[target]
                      if q <= 0.5 + 1e-12)
            return base + off
        med = None
        base = self._base_part(X, tau, target)
        for q, mdl in self._qmodels(tau, target):
            if q > 0.5 + 1e-12:
                break
            v = mdl.predict(X) + base \
                + self.shifts.get((float(tau), target, q), 0.0)
            med = v if med is None else np.maximum(med, v)
        return med

    def point_matrix(self, X, taus, target: str = "delay") -> np.ndarray:
        """Median point forecast; off-grid tau interpolates linearly
        between the surrounding grid taus (TD queries tau - 30).

        Batched: every row is routed to its grid tau (exact) or the two
        interpolation endpoints, so the whole call costs ONE predict per
        NEEDED (grid tau, lower-tail quantile) model -- never per row and
        never per plan (measured: this is what took gbm_v02_b3a from
        ~2 pred/s to the planner bar)."""
        taus = np.asarray(taus, dtype=np.float64)
        med = np.full(len(X), np.nan)
        if not len(taus) or not self.taus:
            return med
        tq = np.round(taus, 6)
        grid = np.round(np.asarray(self.taus, dtype=np.float64), 6)
        with _limit_threads(self.predict_threads):
            rows_of: dict = {float(g): [] for g in grid}
            off: dict = {}            # off-grid tau -> (lo, hi, weight)
            for t in np.unique(tq):
                hit = np.flatnonzero(tq == t)
                k = int(np.searchsorted(grid, t))
                if k < len(grid) and grid[k] == t:
                    rows_of[float(t)].append((float(t), hit))
                elif len(grid) >= 2 and grid[0] < t < grid[-1]:
                    lo, hi = grid[k - 1], grid[k]
                    off[float(t)] = (float(lo), float(hi),
                                     float((t - lo) / (hi - lo)))
                    rows_of[float(lo)].append((float(t), hit))
                    rows_of[float(hi)].append((float(t), hit))
            # ONE calibrated-median pass per needed grid tau over the
            # concatenated rows of every query tau that needs it
            cal: dict = {}            # (grid tau, query tau) -> values
            for g, parts in rows_of.items():
                if not parts:
                    continue
                ri = np.concatenate([h for _t, h in parts])
                vals = self._median_at_grid(X[ri], g, target)
                o = 0
                for t, h in parts:
                    cal[(g, t)] = vals[o:o + len(h)]
                    med[h] = cal[(g, t)]          # on-grid rows keep this
                    o += len(h)
            for t, (lo, hi, w) in off.items():
                m = tq == t
                med[m] = cal[(lo, t)] + w * (cal[(hi, t)] - cal[(lo, t)])
        return med

    def _cal_median_vec(self, X, tau: float, target: str) -> np.ndarray:
        return self._median_at_grid(X, float(tau), target)

    def point_global(self, ds, g_idx, tau, policy, pre_fix,
                     target: str = "delay") -> np.ndarray:
        g_idx = np.asarray(g_idx, dtype=np.int64)
        X = self._matrix_global(ds, g_idx, policy, pre_fix)
        tau_a = np.broadcast_to(np.asarray(tau, np.float64), (len(g_idx),))
        return self.point_matrix(X, tau_a, target)

    def coll_prob_rows(self, ds, row_idx) -> np.ndarray:
        X = self._row_matrix(ds, row_idx)
        taus = np.asarray(ds.rows["tau"].to_numpy()[row_idx])
        out = np.full(len(X), np.nan)
        with _limit_threads(self.predict_threads):
            for tau in self.taus:
                m = np.isclose(taus, tau)
                if not m.any():
                    continue
                c = self.coll[tau]
                if isinstance(c, float):
                    out[m] = np.full(int(m.sum()), c)
                else:
                    out[m] = c.predict_proba(X[m])[:, 1]
        return out

    # ------------------------------------------------------ serialize --
    def save(self, out_dir) -> dict:
        import pickle
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        body = dict(schema=SCHEMA, variant=self.variant,
                    feature_set=self.feature_set,
                    pre_fix_at_predict=self.pre_fix_at_predict,
                    design=self.design,
                    aux_excluded_from_calibration=(
                        self.aux_excluded_from_calibration),
                    es_grouping=self.es_grouping,
                    quantiles=list(self.quantiles),
                    target_quantiles={k: list(v) for k, v
                                      in self.target_quantiles.items()},
                    taus=list(self.taus),
                    feature_names=list(self.feature_names),
                    models=self.models, point_models=self.point_models,
                    coll=self.coll, shifts=self.shifts,
                    oof_delay=self.oof_delay,
                    oof_grouping=self.oof_grouping,
                    tau_variants=dict(self.tau_variants),
                    selection=list(self.selection))
        pkl = out / PKL_NAME
        tmp = Path(str(pkl) + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(body, fh, protocol=5)
        tmp.replace(pkl)
        from .outputs import git_commit, sha256_of
        side = dict(
            schema=SCHEMA, variant=self.variant,
            feature_set=self.feature_set,
            pre_fix_at_predict=self.pre_fix_at_predict,
            design=self.design,
            aux_excluded_from_calibration=(
                self.aux_excluded_from_calibration),
            pre_fix_predict_note=(PRE_FIX_PREDICT_NOTE
                                  if self.pre_fix_at_predict is not None
                                  else None),
            early_stopping=dict(
                validation=("world-grouped holdout via HGB fit(X_val=..) "
                            "(v0.3.2 #10; no random-row split)"
                            if self.es_grouping == "world-grouped" else
                            "fixed max_iter, early stopping off (v0.3.2 "
                            "#10: no random-row validation split)"),
                grouping=self.es_grouping),
            oof_grouping=self.oof_grouping,
            sklearn_version=self.sklearn_version,
            quantiles=list(self.quantiles), taus=list(self.taus),
            n_features=len(self.feature_names),
            feature_names=list(self.feature_names),
            tau_variants={f"{t:.0f}": v for t, v
                          in sorted(self.tau_variants.items())},
            selection=list(self.selection),
            selection_note=("variant 'auto': per tau, hybrid vs ridge-only "
                            "compared by TRAIN world-grouped OOB CRPS of "
                            "the delay head only; exact tie ships ridge"),
            shifts={f"{t}|{tg}|{q}": s for (t, tg, q), s
                    in self.shifts.items()},
            fit_seconds=self.fit_seconds,
            max_iter=self.max_iter, learning_rate=self.learning_rate,
            cal_folds=self.cal_folds, seed=self.seed,
            data_manifest_sha256=self.data_manifest_sha256,
            pkl_sha256=sha256_of(pkl), git_commit=git_commit(),
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        (out / JSON_NAME).write_text(json.dumps(side, indent=1),
                                     encoding="utf-8")
        return side

    @classmethod
    def load(cls, out_dir, log=print) -> "GbmValueModel":
        import pickle
        out = Path(out_dir)
        pkl = out / PKL_NAME
        if not pkl.exists():
            raise FileNotFoundError(f"no {PKL_NAME} in {out} (train with "
                                    f"--value-model gbm first)")
        with open(pkl, "rb") as fh:
            body = pickle.load(fh)
        schema = body.get("schema")
        if schema in (SCHEMA, LEGACY_SCHEMAS[1], LEGACY_SCHEMAS[2]):
            # /4, /3 and /2 all record the variant in the body (/2 = the
            # pre-auto hybrid schema, /3 = the pre-v0.3.2 auto schema).
            # Regression: /2 was once mapped to "pure" here, silently
            # loading every hybrid pkl as residual-only predictions.
            variant = body.get("variant", "hybrid")
        elif schema == LEGACY_SCHEMAS[0]:
            variant = "pure"        # /1 pkls predate the hybrid variant
        else:
            raise ValueError(f"unexpected gbm schema {schema!r}")
        m = cls(quantiles=tuple(body["quantiles"]), variant=variant,
                feature_set=body.get("feature_set", "oracle"))
        m.taus = list(body["taus"])
        m.feature_names = list(body["feature_names"])
        m.models = body["models"]
        m.point_models = body.get("point_models", {})
        m.coll = body["coll"]
        m.shifts = body["shifts"]
        m.oof_delay = body.get("oof_delay", {})
        m.oof_grouping = body.get("oof_grouping", "")
        m.es_grouping = body.get("es_grouping", "")
        m.design = body.get("design", DESIGN_LEGACY)
        m.aux_excluded_from_calibration = bool(
            body.get("aux_excluded_from_calibration", False))
        m.pre_fix_at_predict = body.get(
            "pre_fix_at_predict",
            None if m.feature_set == "oracle" else PRE_FIX_PREDICT_VALUE)
        m.tau_variants = {float(t): v for t, v
                          in body.get("tau_variants", {}).items()}
        m.selection = list(body.get("selection", []))
        if body.get("target_quantiles"):
            m.target_quantiles = {t: tuple(qs) for t, qs
                                  in body["target_quantiles"].items()}
        import sklearn
        side_path = out / JSON_NAME
        if side_path.exists():
            side = json.loads(side_path.read_text(encoding="utf-8"))
            m.sklearn_version = side.get("sklearn_version", "")
            m.fit_seconds = side.get("fit_seconds", {})
            m.data_manifest_sha256 = side.get("data_manifest_sha256", "")
            if side.get("pkl_sha256"):
                from .outputs import sha256_of
                got = sha256_of(pkl)
                if got != side["pkl_sha256"]:
                    raise ValueError(f"{PKL_NAME} sha mismatch: sidecar "
                                     f"{side['pkl_sha256']} != file {got}")
        if m.sklearn_version and m.sklearn_version != sklearn.__version__:
            log(f"gbm: WARNING model fitted with sklearn "
                f"{m.sklearn_version}, running {sklearn.__version__}")
        # warm the OpenMP predict pool NOW (one tiny batched call) so no
        # timed planner call pays the spin-up -- the exact overhead that
        # made the gbm_v02_b3a planner benchmark report 2 predictions/s
        try:
            with _limit_threads(m.predict_threads):
                m._median_at_grid(
                    np.zeros((2, max(1, len(m.feature_names)))),
                    m.taus[0], "delay")
        except Exception:                                   # noqa: BLE001
            pass
        log(f"gbm: loaded value model from {pkl} "
            f"(variant={m.variant}, taus={m.taus}, "
            f"{len(m.models)} quantile models, "
            f"{len(m.point_models)} ridge point models"
            + (", per-tau: "
               + ", ".join(f"{t:.0f}->{v}" for t, v
                           in sorted(m.tau_variants.items()))
               if m.variant == "auto" and m.tau_variants else "") + ")")
        return m


# ------------------------------------------------ quantile distributions --

def crps_from_quantiles(qs, preds: dict, y: np.ndarray) -> np.ndarray:
    """CRPS ~= 2 * [q0*QL(q0) + trapz(QL, qs) + (1-q1)*QL(q1)]: trapezoid
    over the fitted quantile grid with flat extrapolation of the quantile
    loss outside it (QL_q = pinball loss; CRPS = 2*Int QL dq)."""
    qs = sorted(float(q) for q in qs)
    y = np.asarray(y, dtype=np.float64)
    ql = [np.maximum(q * (y - preds[q]), (q - 1.0) * (y - preds[q]))
          for q in qs]
    total = qs[0] * ql[0] + (1.0 - qs[-1]) * ql[-1]
    for a, b, fa, fb in zip(qs[:-1], qs[1:], ql[:-1], ql[1:]):
        total = total + 0.5 * (b - a) * (fa + fb)
    return 2.0 * total


def select_variant_by_oob(tau, target, y, splits, preds_hybrid: dict,
                          preds_ridge: dict) -> dict:
    """Per-tau shipping decision: 'hybrid' vs 'ridge-only' by mean OOB
    CRPS on TRAIN rows ONLY.

    `preds_hybrid`/`preds_ridge` are the calibrated (shifted, monotone)
    per-quantile OOF predictive dicts for the two candidates, `splits`
    the split value of exactly those rows.  Raises
    CalibrationLeakError if any row is not TRAIN (the guard the brief
    requires).  Exact CRPS tie ships 'ridge' (the simpler model)."""
    y = np.asarray(y, dtype=np.float64)
    guard_no_calibration_rows(
        splits, f"select_variant_by_oob(tau={tau}, target={target})")
    n = len(y)
    if any(len(v) != n for v in preds_hybrid.values()) \
            or any(len(v) != n for v in preds_ridge.values()):
        raise ValueError("predictive arrays must align with y")
    crps_h = float(np.mean(crps_from_quantiles(sorted(preds_hybrid),
                                               preds_hybrid, y)))
    crps_r = float(np.mean(crps_from_quantiles(sorted(preds_ridge),
                                               preds_ridge, y)))
    chosen = "hybrid" if crps_h < crps_r else "ridge"
    return dict(tau=float(tau), select_target=str(target),
                chosen=chosen, n=int(n),
                oob_crps={"hybrid": crps_h, "ridge": crps_r})


# --------------------------------------------------- acceptance verdict --

def acceptance_verdict(mae_m, mae_r, crps_m, crps_r,
                       eps: float = 1e-9) -> dict:
    """PASS / TIE / FAIL vs ridge with explicit tie semantics (v0.3.1):
    a TIE on any metric is a TIE of the verdict, never a pass.

    PASS = strictly beats ridge on BOTH MAE and CRPS.  TIE = no strict
    loss on either metric but not a strict win on both (e.g. the shipped
    ridge-only IS the ridge baseline at that tau).  FAIL = strictly
    worse than ridge on at least one metric.  `hard_pass` keeps the
    historical strict semantics (ties are not passes)."""
    vals = (mae_m, mae_r, crps_m, crps_r)
    finite = all(np.isfinite(v) for v in vals)
    if not finite:
        return dict(beats_ridge_mae=False, beats_ridge_crps=False,
                    tie_mae=False, tie_crps=False, result="FAIL",
                    hard_pass=False)
    beats_m = mae_m < mae_r - eps
    beats_c = crps_m < crps_r - eps
    tie_m = abs(mae_m - mae_r) <= eps
    tie_c = abs(crps_m - crps_r) <= eps
    loses_m = mae_m > mae_r + eps
    loses_c = crps_m > crps_r + eps
    if beats_m and beats_c:
        result = "PASS"
    elif loses_m or loses_c:
        result = "FAIL"
    else:
        result = "TIE"
    return dict(beats_ridge_mae=bool(beats_m),
                beats_ridge_crps=bool(beats_c), tie_mae=bool(tie_m),
                tie_crps=bool(tie_c), result=result,
                hard_pass=bool(beats_m and beats_c))


def pit_from_quantiles(qs, preds: dict, y: np.ndarray,
                       seed: int = 20260928) -> np.ndarray:
    """PIT of the quantile-step predictive: piecewise-linear interpolation
    of (pred_q, q) at y; tail mass randomized U(0, q_min) / U(q_max, 1)."""
    qs = sorted(float(q) for q in qs)
    y = np.asarray(y, dtype=np.float64)
    P = np.column_stack([preds[q] for q in qs])
    lo = P.min(axis=1)
    hi = P.max(axis=1)
    u = np.empty(len(y))
    rng = np.random.default_rng(seed)
    below = y < lo
    above = y > hi
    mid = ~(below | above)
    u[below] = rng.uniform(0.0, qs[0], size=int(below.sum()))
    u[above] = rng.uniform(qs[-1], 1.0, size=int(above.sum()))
    for i in np.flatnonzero(mid):
        u[i] = np.interp(y[i], np.maximum.accumulate(P[i]), qs)
    return np.clip(u, 1e-6, 1.0 - 1e-6)


# --------------------------------------------------------- gbm train run --

def run_gbm_training(cfg, log=print):
    """`train --value-model gbm`: fit + save the GBM value model on CPU.

    Same data verification / MANIFEST contract as the net path; the SAME
    CAL evaluation follows in the CLI (__main__), so eval_report.md,
    metrics.json and results.zip are produced identically.  Returns
    (metrics, ds, refit_skipped); a re-run against an unchanged dataset
    skips the fit (cloud-restart parity with the net path's skip)."""
    from .data import V0Dataset, load_manifest
    # v0.4a: --train-only fits from the TRAIN split alone, so the
    # CALIBRATION files are not even opened before the single final read
    splits = (("TRAIN",) if cfg.extras.get("train_only")
              else ("TRAIN", "CALIBRATION"))
    man = load_manifest(cfg.data_root, splits=splits,
                        chunks=cfg.chunks,
                        exclude_superseded=cfg.exclude_superseded,
                        exclude_aux=bool(cfg.extras.get("exclude_aux",
                                                        False)),
                        verify_files=cfg.verify, log=log)
    feature_set = str(getattr(cfg, "features", "oracle"))
    ds = V0Dataset(man, feature_set=feature_set, log=log)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    side_path = out / JSON_NAME
    refit_skipped = False
    if cfg.resume and side_path.exists():
        try:
            side = json.loads(side_path.read_text(encoding="utf-8"))
            if side.get("data_manifest_sha256") == man.sha256:
                GbmValueModel.load(out, log=log)      # integrity check
                refit_skipped = True
                log(f"gbm: existing model matches data MANIFEST "
                    f"{man.sha256[:12]}...; skipping the fit (delete "
                    f"{PKL_NAME} to refit)")
        except Exception as e:                          # noqa: BLE001
            log(f"gbm: existing model unusable "
                f"({type(e).__name__}: {e}); refitting")
    if refit_skipped:
        side = json.loads(side_path.read_text(encoding="utf-8"))
    else:
        model = GbmValueModel(
            variant=str(cfg.extras.get("gbm_variant", "hybrid")),
            max_iter=int(cfg.extras.get("gbm_max_iter", 300)),
            cal_folds=int(cfg.extras.get("gbm_cal_folds", 3)),
            seed=cfg.seed % 2 ** 31,
            feature_set=feature_set)
        model.fit(ds, log=log)
        side = model.save(out)
        log(f"gbm: saved {PKL_NAME} + {JSON_NAME} "
            f"(variant={side['variant']}, sklearn "
            f"{side['sklearn_version']})")
    metrics = dict(
        config={**cfg.to_json(), "value_model": "gbm"},
        device="cpu", value_model="gbm",
        data=dict(manifest_sha256=man.sha256, n_sources=len(man),
                  n_superseded_skipped=man.n_superseded_skipped,
                  files_verified=man.files_verified,
                  feature_set=feature_set),
        gbm=dict(sklearn_version=side.get("sklearn_version"),
                 variant=side.get("variant"),
                 feature_set=side.get("feature_set", feature_set),
                 oof_grouping=side.get("oof_grouping"),
                 taus=side.get("taus"), quantiles=side.get("quantiles"),
                 tau_variants=side.get("tau_variants", {}),
                 selection=side.get("selection", []),
                 fit_seconds=side.get("fit_seconds"),
                 refit_skipped=refit_skipped),
        members=[])
    return metrics, ds, refit_skipped


# ------------------------------------------------- permanent baselines --

def fit_baselines(ds, log=print, max_iter: int = 300,
                  min_rows: int = 8) -> dict:
    """Per-tau ridge + point-GBM on the summary features, fitted on TRAIN
    (diag_tau30.py estimators, kept permanently in the eval report).  OOF
    residuals (5-fold ridge / 3-fold GBM) give each a per-tau residual
    sample so their CRPS is the NRG-empirical CRPS of point + residual
    sample.  Tiny TRAIN slices (tests) fall back to in-fit residuals;
    slices under `min_rows` are SKIPPED entirely with a note (their
    ridge/GBM fits are meaningless and HGB's early-stopping split raises
    on n_samples=1 -- BUNDLE_V0.3_RECORD.md integration note), so tiny
    synthetic fixtures degrade gracefully instead of crashing."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict

    def gbm_est():
        # v0.3.2 fix #10: early stopping OFF at fixed max_iter -- the
        # internal validation split would be random ROWS (a world would
        # straddle fit/validation), which is exactly the leak removed.
        return HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=max_iter, learning_rate=0.05,
            random_state=0, early_stopping=False)

    r = ds.rows
    X, _names = row_feature_matrix(ds)
    y_all = r["total_delay_ref_s"].to_numpy()
    aux = _aux_mask(r)
    out: dict = {}
    skipped: dict = {}
    with _limit_threads(2):
        for tau in TAUS:
            tr = _tau_mask(r, tau, "TRAIN")
            n = int(tr.sum())
            if n == 0:
                continue
            if n < min_rows:
                skipped[float(tau)] = n
                log(f"baselines: tau={tau:.0f} SKIPPED ({n} TRAIN rows < "
                    f"min_rows={min_rows}; fit is meaningless on tiny "
                    f"slices)")
                continue
            Xtr, ytr = X[tr], y_all[tr]
            sc = ~aux[tr]                 # v0.4a: score rows (non-aux)
            t0 = time.perf_counter()
            use_cv = n >= 40
            rid = Ridge(alpha=1.0).fit(Xtr, ytr)
            if use_cv:
                rid_oof = cross_val_predict(
                    Ridge(alpha=1.0), Xtr, ytr,
                    cv=KFold(n_splits=5, shuffle=True, random_state=0))
            else:
                rid_oof = rid.predict(Xtr)
            gbm = gbm_est().fit(Xtr, ytr)
            if use_cv:
                gbm_oof = cross_val_predict(
                    gbm_est(), Xtr, ytr,
                    cv=KFold(n_splits=3, shuffle=True, random_state=0))
            else:
                gbm_oof = gbm.predict(Xtr)
            out[float(tau)] = dict(
                ridge=rid, gbm=gbm,
                resid_ridge=_resid_sample(np.asarray(ytr - rid_oof), sc),
                resid_gbm=_resid_sample(np.asarray(ytr - gbm_oof), sc),
                seconds=round(time.perf_counter() - t0, 1))
            log(f"baselines: tau={tau:.0f} ridge + point-GBM fitted on "
                f"{n} TRAIN rows in {out[float(tau)]['seconds']} s "
                f"({'OOF' if use_cv else 'in-fit'} residuals)")
    return out, skipped


def _resid_sample(resid: np.ndarray, score_mask: np.ndarray,
                  cap: int = 4000) -> np.ndarray:
    """Residual sample of a baseline for its empirical CRPS.  Legacy
    (no auxiliary rows): the first `cap` residuals, unchanged.  With
    auxiliary rows (v0.4a): the non-auxiliary residuals, a seeded random
    subsample when longer than `cap` (never the dataset-order prefix)."""
    if score_mask.all():
        return resid[:cap]
    res = resid[score_mask]
    if len(res) > cap:
        rng = np.random.default_rng(0)
        res = res[np.sort(rng.choice(len(res), size=cap, replace=False))]
    return res


def baselines_section(ds, cal_idx, model_median, model_crps,
                      log=print, gbm_max_iter: int = 300) -> dict:
    """Item 6 table rows + item 7 HARD acceptance (B3 items D15/D16).

    MAE(median) and CRPS on CAL per tau x policy x chunk for climatology,
    ridge, GBM-baseline and the evaluated value model; then, per tau
    (all chunks/policies), whether the evaluated model BEATS RIDGE on
    both MAE and CRPS.  `model_median`/`model_crps` are the evaluated
    model's per-row point (median) and CRPS on `cal_idx` (net mixture or
    GBM value model)."""
    from .distributions import crps_empirical
    from .evaluate import Climatology
    r = ds.rows
    train_rows = r[r["split"] == "TRAIN"]
    base, skipped = fit_baselines(ds, log=log, max_iter=gbm_max_iter)
    cal_idx = np.asarray(cal_idx, dtype=np.int64)
    sub = r.loc[cal_idx]
    y = sub["total_delay_ref_s"].to_numpy()
    taus = sub["tau"].to_numpy()
    ch = sub["chunk_id"].to_numpy()
    pol = sub["policy"].to_numpy()

    clim = Climatology(train_rows, "total_delay_ref_s")
    clim_med = np.empty(len(sub))
    for i, (tau, p, d) in enumerate(zip(sub["tau"], sub["policy"],
                                        sub["demand_veh_h"])):
        s, _lv = clim.sample(float(tau), p, int(d))
        clim_med[i] = float(np.median(s)) if s is not None and len(s) \
            else np.nan
    crps_clim_rows, _clim_lv = clim.crps(y, taus, pol.tolist(),
                                         sub["demand_veh_h"].to_numpy())

    pred = {"climatology": clim_med, "model": np.asarray(model_median,
                                                         dtype=float)}
    crps_rows = {"climatology": crps_clim_rows,
                 "model": np.asarray(model_crps, dtype=float)}
    if base:
        # v0.3.2 fix #11: on a deployable dataset the baselines are
        # queried exactly like the deployed model -- pre_fix fixed to the
        # post-fix value -- so the acceptance comparison is apples to
        # apples (the report states this).
        pre_ov = None if (getattr(ds, "feature_set", "oracle") == "oracle"
                          or getattr(ds, "v04a", False)) \
            else PRE_FIX_PREDICT_VALUE
        Xc = row_feature_matrix(ds, cal_idx, pre_fix_override=pre_ov)[0]
        pred["ridge"] = np.full(len(y), np.nan)
        pred["gbm_baseline"] = np.full(len(y), np.nan)
        crps_rows["ridge"] = np.full(len(y), np.nan)
        crps_rows["gbm_baseline"] = np.full(len(y), np.nan)
        for tau, b in base.items():
            m = np.isclose(taus, tau)
            if not m.any():
                continue
            pr = b["ridge"].predict(Xc[m])
            pg = b["gbm"].predict(Xc[m])
            pred["ridge"][m] = pr
            pred["gbm_baseline"][m] = pg
            # NRG CRPS of the predictive (point_i + residual sample):
            # shared sample = OOF residuals, per-row target = y - point
            crps_rows["ridge"][m] = crps_empirical(b["resid_ridge"],
                                                   y[m] - pr)
            crps_rows["gbm_baseline"][m] = crps_empirical(b["resid_gbm"],
                                                          y[m] - pg)
    mae = {k: np.abs(v - y) for k, v in pred.items()}

    names = ("climatology", "ridge", "gbm_baseline", "model")
    chunks = sorted(set(ch.tolist())) + ["all"]
    pols = sorted(set(pol.tolist())) + ["all"]
    taus_sorted = sorted({float(t) for t in taus})
    rows = []
    for chunk in chunks:
        for p in pols:
            for tau in taus_sorted:
                m = np.isclose(taus, tau)
                if chunk != "all":
                    m &= ch == chunk
                if p != "all":
                    m &= pol == p
                if not m.any():
                    continue
                row = dict(chunk=chunk, policy=p, tau=tau, n=int(m.sum()))
                for k in names:
                    row[f"mae_{k}"] = _r3(np.nanmean(mae[k][m])) \
                        if k in mae else None
                    row[f"crps_{k}"] = \
                        _r3(np.nanmean(crps_rows[k][m])) \
                        if crps_rows.get(k) is not None else None
                rows.append(row)

    acc = []
    for tau in taus_sorted:
        m = np.isclose(taus, tau)
        mae_m = float(np.nanmean(mae["model"][m]))
        crp_m = float(np.nanmean(crps_rows["model"][m]))
        if "ridge" in mae:
            with np.errstate(invalid="ignore"):
                mae_r = float(np.nanmean(mae["ridge"][m]))
                crp_r = (float(np.nanmean(crps_rows["ridge"][m]))
                         if crps_rows.get("ridge") is not None
                         else float("nan"))
        else:                       # every baseline skipped (tiny fixture)
            mae_r = crp_r = float("nan")
        v = acceptance_verdict(mae_m, mae_r, crp_m, crp_r)
        acc.append(dict(
            tau=tau, n=int(m.sum()), mae_model=_r3(mae_m),
            mae_ridge=_r3(mae_r), crps_model=_r3(crp_m),
            crps_ridge=_r3(crp_r), **v))
    overall = dict(hard_pass=bool(acc) and all(a["hard_pass"] for a in acc),
                  n_taus=len(acc), n_pass=sum(a["hard_pass"] for a in acc),
                  n_tie=sum(a["result"] == "TIE" for a in acc),
                  result=("PASS" if acc and all(a["result"] == "PASS"
                                                for a in acc)
                          else "FAIL" if any(a["result"] == "FAIL"
                                             for a in acc) else "TIE")
                  if acc else "FAIL")
    notes = []
    if skipped:
        notes.append("baselines SKIPPED (TRAIN rows < min_rows=8): "
                     + ", ".join(f"tau={t:.0f} (n={n})"
                                 for t, n in sorted(skipped.items()))
                     + "; their ridge/GBM columns are NaN")
    return dict(
        rows=rows, acceptance=acc, overall=overall,
        skipped_baselines={f"{t:.0f}": n for t, n in skipped.items()},
        approx=("GBM value-model CRPS: quantile-grid approximation; "
                "ridge/GBM-baseline CRPS: point forecast plus TRAIN OOF "
                "residual sample (NRG empirical); climatology CRPS: "
                "TRAIN empirical per tau x policy x demand")
        + ("; " + " ".join(notes) if notes else ""))


def oob_acceptance_section(ds, model, log=print,
                           n_splits: int = 5) -> dict:
    """HARD acceptance vs RIDGE on TRAIN-internal OOB, world-grouped.

    Model side: the per-tau WORLD-GROUPED OOF delay quantile predictions
    stored at fit time (the same folds the conformal shifts use --
    a world never appears on both sides of a fold), calibrated with the
    shipped shifts.  Ridge side: a world-grouped GroupKFold OOF ridge per
    tau (the CAL table's estimator) with NRG CRPS from its OOF residual
    sample.  The shifts were fitted on this same OOF sample, so this
    table is TRAIN-internal evidence that the acceptance is not only a
    CAL property; CAL stays the out-of-sample check."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, cross_val_predict

    from .distributions import crps_empirical

    r = ds.rows
    oof = getattr(model, "oof_delay", None) or {}
    grouping = getattr(model, "oof_grouping", "")
    if not oof or grouping != "world-grouped":
        return dict(
            rows=[], acceptance=[],
            note=(f"OOB table needs world-grouped OOF delay predictions "
                  f"from the fit (got grouping={grouping!r}); skipped"),
            overall=dict(hard_pass=False, n_taus=0, n_pass=0))
    X, _names = row_feature_matrix(ds)
    y_all = r["total_delay_ref_s"].to_numpy()
    grp_all = (r["chunk_id"].astype(str) + "|"
               + r["world"].astype(str)).to_numpy()
    qs = sorted(model.target_quantiles["delay"])
    i50 = qs.index(0.5)
    acc = []
    for tau in model.taus:
        tr = _tau_mask(r, tau, "TRAIN")
        if oof.get(float(tau)) is None or int(tr.sum()) == 0:
            continue
        y = y_all[tr]
        Xtr, grp = X[tr], grp_all[tr]
        k = max(2, min(int(n_splits), len(np.unique(grp))))
        oof_rid = cross_val_predict(
            Ridge(alpha=1.0), Xtr, y,
            cv=GroupKFold(n_splits=k), groups=grp)
        sc = ~_aux_mask(r)[tr]            # v0.4a: score non-aux rows only
        resid = (y - oof_rid)[sc]
        P = np.column_stack([np.asarray(oof[float(tau)][q], dtype=float)
                             for q in qs])
        P = P + np.array([model.shifts.get((float(tau), "delay", q), 0.0)
                          for q in qs])
        P = np.maximum.accumulate(P, axis=1)      # monotone rearrangement
        preds = {q: P[sc, i] for i, q in enumerate(qs)}
        y_s, oof_s = y[sc], oof_rid[sc]
        mae_m = float(np.mean(np.abs(preds[0.5] - y_s)))
        mae_r = float(np.mean(np.abs(oof_s - y_s)))
        crp_m = float(np.mean(crps_from_quantiles(qs, preds, y_s)))
        crp_r = float(np.mean(crps_empirical(resid, y_s - oof_s)))
        v = acceptance_verdict(mae_m, mae_r, crp_m, crp_r)
        acc.append(dict(
            tau=float(tau), n=int(sc.sum()), mae_model=_r3(mae_m),
            mae_ridge=_r3(mae_r), crps_model=_r3(crp_m),
            crps_ridge=_r3(crp_r), **v))
        log(f"oob: tau={tau:.0f} world-grouped {k}-fold OOF acceptance "
            f"computed ({int(tr.sum())} TRAIN rows)")
    overall = dict(hard_pass=bool(acc) and all(a["hard_pass"] for a in acc),
                   n_taus=len(acc), n_pass=sum(a["hard_pass"] for a in acc),
                   n_tie=sum(a["result"] == "TIE" for a in acc),
                   result=("PASS" if acc and all(a["result"] == "PASS"
                                                 for a in acc)
                           else "FAIL" if any(a["result"] == "FAIL"
                                              for a in acc) else "TIE")
                   if acc else "FAIL")
    return dict(
        rows=[], acceptance=acc, overall=overall,
        note=(f"model: world-grouped OOF from the fit's conformal folds "
              f"({grouping}), calibrated with the shipped shifts (fitted "
              f"on this sample -- TRAIN-internal evidence); ridge: "
              f"world-grouped GroupKFold OOF point + its OOF residual "
              f"sample (NRG CRPS)"))


def _r3(x) -> float:
    return round(float(x), 3) if np.isfinite(x) else float("nan")
