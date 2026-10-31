"""Shared schema, exact registered detector, and observable trajectory reduction."""
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ('native', 'ctrl0', 'mappo', 'pi', 'circles')
REGIONS = ('upstream', 'merge', 'ramp')
HS = (120, 300, 600)
SEEDS = tuple(range(969420, 969440))
BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')


def safe(path):
    p = Path(path)
    if any(x in str(p).lower() for x in BAD):
        raise ValueError('Prohibited path: '+str(p))
    return p


def digest(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(safe(path).read_text(encoding='utf-8'))


def save_new(path, obj):
    """Exclusive, fsynced immutable receipt. Caller holds the run lock."""
    import os
    with safe(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, allow_nan=False, indent=1)
        f.flush()
        os.fsync(f.fileno())


def reference(key, namespace):
    ref = load(HERE/'references.json')[key]
    assert hashlib.sha256(ref['code'].encode()).hexdigest() == ref['code_sha256']
    exec(compile(ref['code'], ref['source']+' [registered extraction]', 'exec'), namespace)
    return ref['code_sha256']


_detector = {'np': np}


def breakdown(values):
    if 'breakdown' not in _detector:
        reference('breakdown', _detector)
    return _detector['breakdown'](values)


def region_masks(pos, lane):
    valid = (lane >= 0) & (lane <= 2)
    return {'upstream': valid & (lane < 2) & (pos < 0),
            'merge': valid & (lane < 2) & (pos >= 0) & (pos <= 75.44),
            'ramp': valid & (lane == 2)}


def reduce_trajectory(times, pos, speed, lane, present, *, t0=300., horizon=600.):
    """Flattened irregular samples or regular [time,vehicle] arrays; equal-time samples.

    The WM adapter supplies *pre-step* states including inserted vehicles, matching
    recorded raw+restored-t0 means. No endpoint duplication or absent/invalid lanes.
    """
    pos, speed, lane, present = [np.asarray(x) for x in (pos, speed, lane, present)]
    times = np.asarray(times)
    if times.ndim == 1 and pos.ndim == 2:
        times = np.broadcast_to(times[:, None], pos.shape)
    times = np.broadcast_to(times, pos.shape)
    assert pos.shape == speed.shape == lane.shape == present.shape
    use = present.astype(bool) & (times >= t0) & (times < t0+horizon)
    assert np.isfinite(speed[use]).all() and np.isfinite(pos[use]).all()
    n = int(horizon/10)
    idx = np.floor((times-t0)/10).astype(int)
    out = {}
    for reg, mask in region_masks(pos, lane).items():
        keep = use & mask
        count = np.bincount(idx[keep], minlength=n)[:n]
        total = np.bincount(idx[keep], weights=speed[keep].astype(float), minlength=n)[:n]
        out[reg] = dict(speed=[float(v/c) if c else None for v, c in zip(total, count)],
                        post_counts=count.tolist())
    return out


def attach_events(bins, real):
    for reg in REGIONS:
        prefix = real['bins'][reg]['speed'][:6]
        assert len(bins[reg]['speed']) == 60
        bins[reg]['speed'] = prefix+bins[reg]['speed']
        bins[reg]['events'] = {str(h): breakdown(bins[reg]['speed'][:6+h//10]) for h in HS}
    return bins


def wilson(k, n, alpha=.05):
    if not n:
        return [None, None]
    z = NormalDist().inv_cdf(1-alpha/2)
    p = k/n
    d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(c-half), float(c+half)]


def crps(x, y, fair=False):
    """S-WMBREAKDOWN empirical/fair CRPS, sorted O(K log K) algebra."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if not n or not np.isfinite(x).all() or not np.isfinite(y):
        return None
    if fair and n < 2:
        return None
    pair_half = float(np.dot(2*np.arange(n)-n+1, x))
    return float(np.abs(x-y).mean()-pair_half/(n*(n-1) if fair else n*n))


def onset_vector(event, h):
    """Pre-existing mass, discrete CDF, censored mass; unknown stays unscorable."""
    if event['event'] is None:
        return None
    pre = bool(event.get('preexisting', False))
    cdf = [float(bool(event['event']) and event['onset_s'] <= t) for t in range(0, h, 10)]
    return np.array([float(pre), *cdf, float(not pre and not event['event'])])
