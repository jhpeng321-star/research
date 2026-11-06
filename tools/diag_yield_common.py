import numpy as np
C = "ep t vid k cav s v a gapL dvL vL hasL dA vR aR waitR dendR tf nRamp dB nzone pen dem vidR lane".split()
OUT = "E:/agent-work/diag_yield_out"


def load_samples():
    x = np.load(OUT + "/samples.npy").astype(np.float64)
    return {c: x[:, i] for i, c in enumerate(C)}


def cells(d, edges=None):
    """Covariate cell id per row: speed x time-headway x closing x position x demand."""
    v = d["v"]
    ve = np.array([0.5, 2, 4, 7, 10, 14, 18, 22, 26])
    vb = np.digitize(v, ve)
    thw = np.where(d["hasL"] > 0, d["gapL"] / np.maximum(v, 0.5), np.inf)
    te = np.array([0.5, 1, 1.5, 2.5, 4, 8, 20])
    tb = np.where(d["hasL"] > 0, np.digitize(thw, te), len(te) + 1)
    cb = np.where(d["hasL"] > 0, np.digitize(d["dvL"], [-1.0, -0.2, 0.2, 1.0]), 5)
    sb = np.digitize(d["s"], [-100, -50, -10, 20, 50])
    db = (d["dem"] >= 2400).astype(int)
    cid = (((vb * 10 + tb) * 7 + cb) * 7 + sb) * 2 + db
    return cid.astype(np.int64)


def control_table(d, mask_ctrl, cid, min_n=20):
    """Mean/sd of a in each cell among controls."""
    import collections
    keys = cid[mask_ctrl]; a = d["a"][mask_ctrl]
    u, inv = np.unique(keys, return_inverse=True)
    n = np.bincount(inv); mu = np.bincount(inv, a) / n
    sd = np.sqrt(np.maximum(np.bincount(inv, a * a) / n - mu ** 2, 0))
    # also p10
    return {int(k): (int(nn), float(m), float(s)) for k, nn, m, s in zip(u, n, mu, sd)}


def expected_a(cid, table, min_n=20):
    out = np.full(len(cid), np.nan)
    for i, c in enumerate(cid):
        pass
    return out


def ctrl_model(d, hdv_only=True, min_n=20):
    """Cell means of a among controls (no ramp-lane car within [s-30, s+150])."""
    cid = cells(d)
    ctrl = (d["lane"] == 0) & ((d["cav"] == 0) if hdv_only else True)
    u, inv = np.unique(cid, return_inverse=True)
    n = np.bincount(inv[ctrl], minlength=len(u)); mu = np.bincount(inv[ctrl], d["a"][ctrl], minlength=len(u)) / np.maximum(n, 1)
    return u, np.where(n >= min_n, mu, np.nan)


def lookup(u, mu, cid):
    i = np.searchsorted(u, cid); i = np.clip(i, 0, len(u) - 1)
    return np.where(u[i] == cid, mu[i], np.nan)


def event_cells(ev, dem):
    """cells() on event arrays [N, T, F]: demand per event."""
    N, T, _ = ev.shape
    d = {"v": ev[..., 1], "s": ev[..., 0], "gapL": ev[..., 4], "dvL": ev[..., 5] - ev[..., 1],
         "hasL": np.where(np.isfinite(ev[..., 4]) & (ev[..., 6] > 0), 1.0, 0.0),
         "dem": np.repeat(dem[:, None], T, 1)}
    flat = {k: x.reshape(-1) for k, x in d.items()}
    return cells(flat).reshape(N, T)
