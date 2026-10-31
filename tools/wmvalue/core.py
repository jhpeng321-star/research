"""Pure CPU value features and registered paired statistics; no simulator imports.

Inputs must already be observation-adapted. A global tape with all-present masks
is not an observation contract. See README.md for the required audit boundary.
"""
from itertools import combinations

import numpy as np

from .audit import ARMS, allowed_seed

TIMES = (300, 360, 420, 480, 540)
METHODS = ("A", "B", "C", "D", "E")
EDGES = np.array([-500., -250., 0., 100., 300., 600.])
MODEL_PARAMS = dict(loss="squared_error", max_iter=150, max_leaf_nodes=15,
                    learning_rate=.05, l2_regularization=10, random_state=20261031)


def finite(value, shape=None):
    a = np.asarray(value)
    if shape is not None and a.shape != shape:
        raise ValueError(f"shape {a.shape} != {shape}")
    if not np.isfinite(a).all():
        raise ValueError("nonfinite data")
    return a


def boolean(value, shape=None):
    a = finite(value, shape)
    if not np.isin(a, (0, 1)).all():
        raise ValueError("not a boolean mask")
    return a.astype(bool)


def features(state):
    """Seven aggregates per lane/bin; only sensor-visible kinematics are accessed.

    kin columns = longitudinal position, lane, speed, acceleration. There is no
    speed/gap behavioral threshold and no population or controller identifier.
    coverage is an explicit 3x5 mask, distinct from an empty observed segment.
    """
    time = float(state["time_s"])
    if not np.isfinite(time) or not 300 <= time <= 900:
        raise ValueError("time outside registered continuation window")
    k = np.asarray(state["kin"])
    if k.ndim != 2 or k.shape[1] != 4:
        raise ValueError("kin must be [slots,4]")
    n = len(k)
    present = boolean(state["present"], (n,))
    sensed = boolean(state["sensor_mask"], (n,))
    cav = boolean(state["known_cav"], (n,))
    coverage = boolean(state["coverage"], (3, 5))
    # Unobserved values may be NaN: do not inspect or aggregate them.
    visible = present & sensed
    obs = finite(k[visible])
    if len(obs) and (not np.isin(obs[:, 1], (-1, 0, 1, 2)).all() or (obs[:, 2] < 0).any()):
        raise ValueError("invalid visible lane/speed")
    c = cav[visible]
    values = []
    for lane in range(3):
        for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
            mask = (obs[:, 1] == lane) & (obs[:, 0] >= lo) & (obs[:, 0] < hi)
            if mask.any() and not coverage[lane, i]:
                raise ValueError("visible vehicle outside declared coverage")
            v, a = obs[mask, 2], obs[mask, 3]
            values.extend([float(coverage[lane, i]), len(v), len(v)/(hi-lo),
                           float(v.mean()) if len(v) else 0.,
                           float(v.std()) if len(v) else 0.,
                           float(a.mean()) if len(v) else 0., int(c[mask].sum())])
    if "queue" in state or "causal_counts" in state:
        raise ValueError("MAIN18:12 prohibits source queue/request features")
    flow = finite(state["flow_counts"], (2, 2))  # entrance x (cumulative, trailing60)
    if (flow < 0).any() or (flow[:, 1] > flow[:, 0]).any():
        raise ValueError("invalid boundary flow count")
    return np.array([*values, *flow.ravel(), int(c.sum()), 900-time], dtype=float)


def action_summary(commands, cav_mask):
    """Present first-window commands only, never observed future vehicle motion.

    Commands [CAV slots,2] = issued acceleration and target speed. An explicit
    mask distinguishes a missing target-speed command from numeric speed zero.
    """
    commands = np.asarray(commands)
    if commands.ndim != 2 or commands.shape[1] != 2:
        raise ValueError("commands must be [slots,2]")
    use = boolean(cav_mask, commands.shape)
    out = []
    for col in range(2):
        v = finite(commands[use[:, col], col])
        out.extend([len(v), float(v.mean()) if len(v) else 0.,
                    float(v.std()) if len(v) else 0.])
    return np.asarray(out, dtype=float)


def remaining_labels(time_s, cumulative, objective600):
    """Exact real cost-to-go; cumulative clock is measured since control t300."""
    times, clock = finite(time_s), finite(cumulative)
    if times.ndim != 1 or clock.shape != times.shape or len(times) < 1201:
        raise ValueError("incomplete cost clock")
    if times[0] != 300 or times[-1] != 900 or not np.allclose(np.diff(times), .5):
        raise ValueError("expected full absolute t300..900 half-second clock")
    if clock[0] != 0 or (np.diff(clock) < -1e-9).any():
        raise ValueError("nonmonotone/nonzero-origin cost clock")
    if not np.isfinite(objective600) or not np.isclose(clock[-1], objective600, rtol=1e-8, atol=1e-6):
        raise ValueError("real endpoint objective disagreement")
    return clock[-1]-clock


class Value:
    def __init__(self, *, actions=False):
        self.actions = actions
        self.model = None

    def fit(self, rows):
        """No reserved seed may even enter preprocessing or sample weighting."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        from threadpoolctl import threadpool_limits

        if not rows:
            raise ValueError("no FIT rows")
        x, y, groups = [], [], []
        for row in rows:
            seed = allowed_seed(row["seed"], "fit")
            if row.get("role") != "fit":
                raise ValueError("non-FIT role")
            time = row["state"]["time_s"]
            if time not in range(300, 901, 30):
                raise ValueError("FIT state not on registered grid")
            target = float(row["remaining_J"])
            if not np.isfinite(target) or target < 0 or (time == 900 and target != 0):
                raise ValueError("invalid real target")
            f = features(row["state"])
            if self.actions:
                f = np.r_[f, finite(row["action_summary"], (6,))]
            x.append(f)
            y.append(target)
            groups.append((seed, row["arm"]))
        # Equal world mass, equal arm mass within each world, equal rows per arm.
        from collections import Counter
        count = Counter(groups)
        arms = {s: len({a for ss, a in groups if ss == s}) for s, _ in groups}
        w = np.array([1/(arms[s]*count[(s, a)]) for s, a in groups])
        w *= len(w)/w.sum()
        with threadpool_limits(limits=1):
            self.model = HistGradientBoostingRegressor(**MODEL_PARAMS).fit(np.stack(x), y, sample_weight=w)
        self.fit_seeds = sorted({s for s, _ in groups})
        return self

    def predict(self, state, summary=None):
        if self.model is None:
            raise ValueError("value is not fitted")
        f = features(state)
        if self.actions:
            f = np.r_[f, finite(summary, (6,))]
        if state["time_s"] == 900:
            return 0.
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            return float(self.model.predict(f[None])[0])


def arm_means(fit_rows):
    cells = {}
    for row in fit_rows:
        allowed_seed(row["seed"], "fit")
        if row.get("role") != "fit":
            raise ValueError("arm means require FIT rows")
        t = row["state"]["time_s"]
        if t in TIMES and row["arm"] in ARMS:
            cells.setdefault((row["arm"], t), []).append(float(row["remaining_J"]))
    expected = {(a, t) for a in ARMS for t in TIMES}
    if set(cells) != expected:
        raise ValueError("incomplete arm/time FIT baseline")
    return {key: float(finite(value).mean()) for key, value in cells.items()}


def assemble_predictions(seed, t0, real, rollouts, value, direct, summaries, baseline):
    """A/B/C from matched K8 WM trajectories supplied by a hash-bound adapter.

    Each arm/future: endpoint time, full remaining integral (A), and windows
    60/90 with state and partial integral. V is applied per future BEFORE mean.
    """
    allowed_seed(seed, "test")
    if t0 not in TIMES or set(real) != set(ARMS) or set(rollouts) != set(ARMS):
        raise ValueError("unregistered decision state/arms")
    out = {}
    for arm in ARMS:
        futures = rollouts[arm]
        if len(futures) != 8 or [f["future"] for f in futures] != list(range(8)):
            raise ValueError("require paired ordered K8 futures")
        a, b, c = [], [], []
        for f in futures:
            if f["start_time"] != t0 or f["end_time"] != 900:
                raise ValueError("wrong WM endpoint")
            a.append(float(f["J"]))
            for horizon, dest in ((60, b), (90, c)):
                window = f["windows"][str(horizon)]
                if window["state"]["time_s"] != t0+horizon:
                    raise ValueError("wrong bootstrap clock")
                dest.append(float(window["J"])+value.predict(window["state"]))
        out[arm] = dict(A=float(finite(a).mean()), B=float(finite(b).mean()),
                        C=float(finite(c).mean()),
                        D=direct.predict(real[arm], summaries[arm]), E=baseline[(arm, t0)])
    return out


def paired_rows(arm_rows):
    cells = {}
    for row in arm_rows:
        seed = allowed_seed(row["seed"], "test")
        t, arm = row["t0"], row["arm"]
        if t not in TIMES or arm not in ARMS or (seed, t, arm) in cells:
            raise ValueError("invalid/duplicate prediction cell")
        finite([row["real"], *[row[m] for m in METHODS]])
        cells[seed, t, arm] = row
    expected = {(s, t, a) for s in range(969432, 969440) for t in TIMES for a in ARMS}
    if set(cells) != expected:
        raise ValueError("require all 200 arm cells; no complete-case dropping")
    out = []
    for s in range(969432, 969440):
        for t in TIMES:
            for first, second in combinations(ARMS, 2):
                a, b = cells[s, t, first], cells[s, t, second]
                out.append(dict(seed=s, t0=t, pair=[first, second],
                                **{m: float(b[m]-a[m]) for m in ("real", *METHODS)}))
    return out


def ranks(x):
    x = finite(x)
    _, inv, count = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(count)
    return ((ends-count+ends-1)/2.)[inv]


def metrics(real, pred):
    real, pred = finite(real), finite(pred)
    if real.shape != pred.shape or not real.size:
        raise ValueError("empty/mismatched metrics")
    r, p = ranks(real), ranks(pred)
    r, p = r-r.mean(), p-p.mean()
    den = float(np.sqrt(np.dot(r, r)*np.dot(p, p)))
    return np.array([np.mean(np.sign(real) == np.sign(pred)),
                     float(np.dot(r, p)/den) if den else 0.,
                     np.mean(np.abs(real-pred))])


def evaluate(arm_rows, draws=10000):
    if draws != 10000:
        raise ValueError("registered bootstrap count is 10000")
    rows = paired_rows(arm_rows)
    values = {key: np.array([r[key] for r in rows]).reshape(8, 50)
              for key in ("real", *METHODS)}
    real = values["real"]
    point = {m: metrics(real.ravel(), values[m].ravel()) for m in METHODS}
    rng = np.random.default_rng(20261031)
    boot = {m: np.empty((draws, 3)) for m in METHODS}
    for i in range(draws):
        worlds = rng.integers(0, 8, 8)
        y = real[worlds].ravel()
        for m in METHODS:
            boot[m][i] = metrics(y, values[m][worlds].ravel())
    summary = {}
    for m in METHODS:
        ci = np.percentile(boot[m], [2.5, 97.5], axis=0)
        summary[m] = {name: dict(value=float(point[m][j]), ci95=ci[:, j].tolist())
                      for j, name in enumerate(("sign_accuracy", "spearman", "mae"))}
        summary[m]["predicted_zero_pairs"] = int((values[m] == 0).sum())
    contrasts, winners = {}, []
    for m in ("B", "C"):
        passes = []
        for ref in ("A", "D", "E"):
            sign = boot[m][:, 0]-boot[ref][:, 0]
            mae = boot[ref][:, 2]-boot[m][:, 2]
            sci, mci = np.percentile(sign, [2.5, 97.5]), np.percentile(mae, [2.5, 97.5])
            passes.extend([bool(sci[0] > 0), bool(mci[0] > 0)])
            contrasts[f"{m}-{ref}"] = dict(
                sign_gain=float(point[m][0]-point[ref][0]), sign_ci95=sci.tolist(),
                mae_reduction=float(point[ref][2]-point[m][2]), mae_ci95=mci.tolist())
        if all(passes):
            winners.append(m)
    return dict(schema="wmvalue_results/1", n_worlds=8, n_pairs=len(rows),
                real_zero_pairs=int((real == 0).sum()), metrics=summary,
                contrasts=contrasts, winners=winners, verdict="PASS" if winners else "FAIL",
                bootstrap=dict(unit="whole world", draws=draws, seed=20261031))
