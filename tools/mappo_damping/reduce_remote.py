"""Remote, read-only reducer of recorded observable views (no simulator/model imports).

Header injected by collect.py defines ROOT, LO, HI, CORPUS, ARMS, BAD.
Reads only time_s/veh_kin/veh_mask/vid/is_cav/episode.json of each view episode.
Window = branch [t0, t0+300 s] on the 0.5 s state grid. Realised acceleration is
the identity-continuous forward difference of recorded speed; jerk its forward
difference. No behavioural thresholds; groups are defined geometrically.
"""
import os
os.nice(19)
from pathlib import Path
import hashlib, json
import numpy as np


def safe(p):
    p = Path(p)
    assert not any(x in str(p).lower() for x in BAD), str(p)
    return p


def digest(p):
    return hashlib.sha256(safe(p).read_bytes()).hexdigest()


def acc(store, key, x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    s = store.setdefault(key, [0, 0.0, 0.0])
    s[0] += int(x.size); s[1] += float(x.sum()); s[2] += float((x * x).sum())


def within_var(store, key, V, M):
    """Pooled within-vehicle speed variance over frames M (vehicle x frame)."""
    s = store.setdefault(key, [0, 0.0])
    for i in range(V.shape[0]):
        x = V[i, M[i]]
        if x.size >= 2:
            s[0] += int(x.size); s[1] += float(((x - x.mean()) ** 2).sum())


rows = []
for seed in range(LO, HI):
    w = json.loads(safe(Path(ROOT) / f'world_{seed}.json').read_bytes())
    for arm in ARMS:
        ep = safe(Path(ROOT) / 'episodes' / w[arm]['episode'])
        meta = json.loads((ep / 'episode.json').read_bytes())
        assert meta['seed'] == seed and meta['arm'] == arm
        t = np.load(ep / 'time_s.npy', allow_pickle=False)
        kin = np.load(ep / 'veh_kin.npy', allow_pickle=False)
        mask = np.load(ep / 'veh_mask.npy', allow_pickle=False).astype(bool)
        vid = np.load(ep / 'vid.npy', allow_pickle=False)
        cav = np.load(ep / 'is_cav.npy', allow_pickle=False).astype(bool)
        dt = float(np.median(np.diff(t)))
        i0 = int(meta['t0_index']); i1 = i0 + int(round(300 / dt))
        assert abs(t[i0] - 300.0) < 1e-9 and i1 < len(t) and abs(t[i1] - 600.0) < 1e-9
        sl = slice(i0, i1 + 1); T = i1 - i0 + 1
        K, Mk, I, C = kin[sl], mask[sl], vid[sl], cav[sl]
        ids = np.unique(I[Mk]); n = len(ids); pos = {v: j for j, v in enumerate(ids)}
        V = np.full((n, T), np.nan); S = np.full((n, T), np.nan); L = np.full((n, T), -1)
        A5 = np.full((n, T), np.nan); flag = np.zeros(n, int); flag_set = [set() for _ in range(n)]
        for f in range(T):
            m = Mk[f]; sid = I[f][m]
            assert len(np.unique(sid)) == len(sid), 'duplicate identity in frame'
            j = np.array([pos[v] for v in sid])
            V[j, f] = K[f][m][:, 4]; S[j, f] = K[f][m][:, 0]; A5[j, f] = K[f][m][:, 5]
            oh = K[f][m][:, 1:4]; known = oh.sum(-1) > 0
            L[j[known], f] = oh[known].argmax(-1)
            for jj, c in zip(j, C[f][m]): flag_set[jj].add(bool(c))
        assert all(len(x) == 1 for x in flag_set), 'CAV flag not constant per identity'
        is_cav = np.array([next(iter(x)) for x in flag_set])
        P = np.isfinite(V)
        a = (V[:, 1:] - V[:, :-1]) / dt          # forward difference at frame f
        aP = P[:, 1:] & P[:, :-1]
        a = np.where(aP, a, np.nan)
        j_ = (a[:, 1:] - a[:, :-1]) / dt
        # leader in same known lane: nearest larger position
        lead_cav = np.zeros((n, T), bool); lead_hdv = np.zeros((n, T), bool)
        for f in range(T):
            for lane in (0, 1, 2):
                k = np.where(P[:, f] & (L[:, f] == lane))[0]
                if len(k) < 2: continue
                o = k[np.argsort(S[k, f], kind='stable')]
                fol, lea = o[:-1], o[1:]
                lead_cav[fol, f] = is_cav[lea]; lead_hdv[fol, f] = ~is_cav[lea]
        up = P & (L >= 0) & (L < 2) & (S < 0)
        hdv = ~is_cav[:, None] & P
        groups = {
            'cav': is_cav[:, None] & P,
            'hdv_follow_cav': hdv & lead_cav,
            'hdv_follow_hdv': hdv & lead_hdv,
            'hdv': hdv,
            'all': P,
            'upstream_all': up,
            'upstream_hdv': up & ~is_cav[:, None],
        }
        out = {}; var = {}
        for g, G in groups.items():
            acc(out, g + '|v', V[G])
            acc(out, g + '|a', a[G[:, :-1] & aP])
            acc(out, g + '|j', j_[G[:, :-2] & aP[:, :-1] & aP[:, 1:]])
            within_var(var, g, V, G)
        # recorded column 5 is SUMO's acceleration over the step ending at f+1
        ok = aP & np.isfinite(A5[:, 1:])
        dif = a[ok] - A5[:, 1:][ok]
        rows.append(dict(corpus=CORPUS, seed=seed, arm=arm, demand=meta['demand_vph'],
                         penetration=meta['penetration'], n_ids=int(n), n_cav=int(is_cav.sum()),
                         moments=out, within_var=var,
                         check_a_vs_col5=dict(n=int(dif.size), max_abs=float(np.abs(dif).max()) if dif.size else None,
                                              rms=float(np.sqrt((dif ** 2).mean())) if dif.size else None),
                         sha256={k: digest(ep / f'{k}.npy') for k in ('time_s', 'veh_kin', 'veh_mask', 'vid', 'is_cav')}
                         | {'episode.json': digest(ep / 'episode.json')},
                         episode=w[arm]['episode']))
print(json.dumps(dict(corpus=CORPUS, root=ROOT, rows=rows), allow_nan=False))
