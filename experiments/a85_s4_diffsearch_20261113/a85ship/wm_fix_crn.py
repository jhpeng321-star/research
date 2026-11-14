"""E1-FIX (Amendment 74, POST HOC REPAIR), Fix 2: common random numbers for the frozen WM planning API.

Drop-in replacement of ``wms.planapi:evaluate`` (bound through the WMA backend ``api`` override, the WM tree is
NOT edited). It calls the unmodified ``wms.planapi.evaluate`` once, with two random sources pinned for the call so
that every row (action a, draw k) depends on (seed, state digest, k) only - never on the action list, the row
position, the chunk size (max_rows), the worker process or what that process computed before:

  lane uniforms   the registered stream ``sha256(seed, k, state digest)`` - already common, untouched.
  arrivals        registered defect: every chunk built ``TorchEntryProcess(seed, batch=b)``, whose per-step draws
                  are shared by all rows of the chunk and restart from the same seed in every chunk (so all K draws
                  saw the same arrivals), and took the entrant CAV share from the action in row 0 OF THE CHUNK (0.0
                  when the base policy led the chunk, the member share of some plan otherwise - with
                  WZPROP_SPLIT_ACTIONS a task without the base policy always used a plan's share).
                  Here: one arrival stream per draw index, seeded ``arrival_seed(seed, digest, k)``, advanced once
                  per step and used by every row of that draw; lane split from the t0 vehicles; entrant CAV share
                  fixed at ENTRANT_CAV_SHARE = 0.0 (the value the base policy received).
  merge noise     registered defect (found while building the fix): ``merge_pivot.update_state`` drew its per-slot
                  stream keys with ``torch.randint`` from the process-global, unseeded RNG, [rows, slots], once per
                  rollout. The keyed logistic merge noise therefore differed between the plan row and the base row
                  and between runs (it depended on the worker's earlier calls). Here: one key row per draw index
                  from ``pivot_seed(seed, digest, k)``, shared by every action row of that draw.

planapi orders rows draw-major (``[(a, k) for k in range(K) for a in actions]``) and chunks them by max_rows; the
draw index of a chunk row is recovered as ``row_draws(offset, b, n_actions)`` and checked against the chunk's
lane-uniform calls (which carry the true k).

Guards: the global torch RNG is re-seeded per call; every remaining draw from a global RNG during a rollout is
counted by call site and logged (expected: none); cudnn is set deterministic. One JSON line per call is appended
to $WM_FIX_LOG_DIR/crn_<pid>.jsonl when that variable is set.

The pure functions (arrival_seed, pivot_seed, lane_split, row_draws) are tested on the laptop without torch.
"""
import hashlib
import json
import os

ENTRANT_CAV_SHARE = 0.0
SALT = 'e1fix-a74-v1'
_GLOBAL_DRAWS = ('rand', 'randn', 'randn_like', 'rand_like', 'randint', 'randint_like', 'bernoulli', 'multinomial',
                 'normal', 'randperm', 'poisson')
_STATE = dict(installed=False, counts={})


def _seed(tag, seed, digest, k):
    return int(hashlib.sha256(f'{SALT}:{tag}:{int(seed)}:{int(k)}:{digest}'.encode()).hexdigest()[:15], 16)


def arrival_seed(seed, digest, k):
    """Seed of the arrival stream of (search/CRN seed, start-state digest, draw index). Pure."""
    return _seed('arrivals', seed, digest, k)


def pivot_seed(seed, digest, k):
    """Seed of the merge-noise stream keys of (seed, start-state digest, draw index). Pure."""
    return _seed('pivotkeys', seed, digest, k)


def torch_seed(seed, digest, k):
    return _seed('torch', seed, digest, k) % (2**63-1)


def lane_split(vehicles):
    """Share of t0 through-lane vehicles on WM lane 1 (the entry process's lane split), from the state rows. Pure."""
    n0 = sum(1 for r in vehicles if int(r.get('lane', 0)) == 0)
    n1 = sum(1 for r in vehicles if int(r.get('lane', 0)) == 1)
    return 0.5 if n0+n1 == 0 else n1/float(n0+n1)


def row_draws(offset, n_rows, n_actions):
    """Draw index of each row of a chunk that starts at global row ``offset`` (planapi's draw-major order). Pure."""
    return [(int(offset)+i)//int(n_actions) for i in range(int(n_rows))]


def _install():
    """Once per process: determinism switches and counters on every global-RNG torch draw."""
    if _STATE['installed']:
        return
    import traceback
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def wrap(name):
        real = getattr(torch, name)

        def counted(*a, **kw):
            if kw.get('generator') is None:
                fr = traceback.extract_stack(limit=2)[0]
                site = f'{name}@{os.path.basename(fr.filename)}:{fr.lineno}'
                _STATE['counts'][site] = _STATE['counts'].get(site, 0)+1
            return real(*a, **kw)
        counted.__wrapped__ = real
        setattr(torch, name, counted)
    for name in _GLOBAL_DRAWS:
        if hasattr(torch, name):
            wrap(name)
    _STATE['installed'] = True


class _TorchProxy:
    """``torch`` as seen by wms.model.merge_pivot during one call: randint returns the key row of each row's draw."""

    def __init__(self, real, key_seed_of, ctx):
        self.__dict__.update(_real=real, _key_seed_of=key_seed_of, _ctx=ctx, calls=0)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def randint(self, low, high, size, **kw):
        size = tuple(int(x) for x in size)
        ks = self._ctx['ks']
        if len(size) != 2 or kw.get('generator') is not None or ks is None or len(ks) != size[0]:
            raise RuntimeError(f'E1-FIX: unexpected merge_pivot randint{size}')
        t = self._real
        row = {}
        for k in sorted(set(ks)):
            gen = t.Generator(device='cpu').manual_seed(self._key_seed_of(k))
            row[k] = t.randint(int(low), int(high), (size[1],), generator=gen, dtype=kw.get('dtype', t.long))
        self.__dict__['calls'] += 1
        return t.stack([row[k] for k in ks]).to(kw.get('device') or 'cpu')


def evaluate(handle, state, actions, *, K=8, H=300.0, seed=0, trace=None):
    """``{action_id: [J_1..J_K]}`` as wms.planapi.evaluate, draws common across actions and chunking (see module)."""
    import numpy as np
    import torch
    import wms.planapi as P
    import wms.model.merge_pivot as MP
    from wms.model import entries as EN
    _install()
    digest = P.state_digest(state)
    p_lane1 = lane_split(state['vehicles'])
    n_act = len(actions)
    ctx = dict(offset=0, ks=None, seen=[])

    class CommonEntries(EN.TorchEntryProcess):
        """One arrival stream per draw index; rows of the same draw share it (as the registered process shared its
        single stream), whatever chunk they are in."""

        def __init__(self, counter, ks):
            super().__init__(counter, seed=0, batch=len(ks))
            self.ks = list(ks)
            self.rngs = {k: np.random.default_rng(arrival_seed(seed, digest, k)) for k in sorted(set(ks))}
            self.rng = None

        def _observe(self, st):
            first = self.p_lane1 is None
            super()._observe(st)
            if first:
                self.p_lane1, self.cav_share = p_lane1, ENTRANT_CAV_SHARE

        def insert(self, st):                       # TorchEntryProcess.insert with one queue per draw index
            self._observe(st)
            new_mask = torch.zeros(st['present'].shape, dtype=torch.bool, device=st['present'].device)
            work = EN._Working(st)
            by_k = {}
            for k, rng in self.rngs.items():
                self.rng = rng
                q = [[] for _ in range(EN.N_ENTRANCES)]
                for entrance, u_lane, u_cav in self._draw():
                    q[entrance].append((u_lane, u_cav))
                by_k[k] = q
            self.rng = None
            if work.present.shape[0] != len(self.ks):
                raise RuntimeError('E1-FIX: entry batch does not match the chunk rows')
            for b in range(work.present.shape[0]):
                for e in range(EN.N_ENTRANCES):
                    self._insert_entrance(work, b, e, by_k[self.ks[b]][e])
            return self._commit(st, work, new_mask)

    def entries(counter, n_rows, _seed_, _dev):
        ks = row_draws(ctx['offset'], n_rows, n_act)
        if ks != ctx['seen'][-int(n_rows):]:
            raise RuntimeError('E1-FIX: chunk draw indices do not match the lane-uniform calls')
        ctx['offset'] += int(n_rows)
        ctx['ks'] = ks
        if counter is None:
            return None
        proc = CommonEntries(np.asarray(counter, np.float32), ks)
        proc.cav_share = ENTRANT_CAV_SHARE
        return proc

    real_uniform, real_entries, real_torch = P._draw_uniform, P._entries, MP.torch

    def uniform(s, k, d, n_steps, n):
        ctx['seen'].append(int(k))
        return real_uniform(s, k, d, n_steps, n)
    proxy = _TorchProxy(real_torch, lambda k: pivot_seed(seed, digest, k), ctx)
    _STATE['counts'] = {}
    P._draw_uniform, P._entries, MP.torch = uniform, entries, proxy
    try:
        torch.manual_seed(torch_seed(seed, digest, 0))
        out = P.evaluate(handle, state, actions, K=K, H=H, seed=seed, trace=trace)
    finally:
        P._draw_uniform, P._entries, MP.torch = real_uniform, real_entries, real_torch
    if ctx['offset'] != n_act*int(K):
        raise RuntimeError('E1-FIX: not every row went through the common-draw path')
    d = os.environ.get('WM_FIX_LOG_DIR')
    if d:
        log = dict(pid=os.getpid(), digest=digest[:16], seed=int(seed), K=int(K), actions=n_act,
                   max_rows=int(handle.max_rows), entries=state.get('counter') is not None,
                   pivot_key_calls=proxy.calls, residual_global_rng_draws=dict(_STATE['counts']))
        with open(os.path.join(d, f'crn_{os.getpid()}.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(log, sort_keys=True)+'\n')
    return out
