"""World-model contract of the WM-C proposer, the CPU StubWM, and the WM-A (c4/wz-wm-port) binding.

Contract (what the proposer needs from any WM):

    observe(moment) -> state
        moment = dict(moment_id (opaque hash; the sid encodes world labels and is withheld), t0, map,
                      cavs=[CAV self-states id/edge/lane/x/v at t0], state_file|None (a locator of the saved
                      t0 SUMO state, not a feature), inserted_t0 (source-loop bookkeeping), crn_seed).
        World labels (cell/demand/penetration/world seed) and global aggregates (approach_speed,
        closing_lane_slow, lane_counts) are NOT passed. The returned state must be built from
        RSU-observable content only (CAV self-states + CAV-sensed neighbours).
    evaluate(state, actions, K, H, seed) -> {action_id: [J_1 .. J_K]}
        J_k = predicted team delay (veh-s; road speed deficit + source waiting) over [t0, t0+H) on WM
        draw k. Draw k must be a deterministic function of (state, seed, k) shared by every action in
        the call (common random numbers); the proposer always puts the base and the needed solos in
        the same call as the groups they are compared with.

Actions (one dict each): id, kind ('base'|'group'|'solo'), family, key, members (N3 member specs:
id, role 'hold', params v_w/hold_s/dmax/amax/release_x[/start_s/anchor/anchor_release_x], lane, x, v),
law ('hold_speed': a = clip((v_w - v)/1 s, -dmax, amax) until release; native before start_s and after
release), profile ({vid: [a_s or None]} open-loop approximation for profile-only WMs), duration_s.

WM-A binding: WMA_READY.json (written by WM-A) is expected to carry
    "api": {"load": "pkg.mod:fn",          optional; fn(**load_kwargs) -> handle
            "observe": "pkg.mod:fn",       optional; fn(handle, moment) -> state (default: CAV rows)
            "evaluate": "pkg.mod:fn"},     required; fn(handle, state, actions, K=, H=, seed=)
    "load_kwargs": {...}                   optional
evaluate may return {aid: [float | {'team_delay': x}]}, {str(H): {...}} or {'scores': {str(H): {...}}}.
CLI/env overrides (--wm-api evaluate=pkg.mod:fn) cover a READY file that names the API differently.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path

from . import families as F
from . import fakes
from . import geometry as G

READY_NAME = 'WMA_READY.json'
READY_ENV = 'WZWM_WMA_READY'
DEFAULT_READY_LOCATIONS = ('WMA_READY.json', 'E:/agent-work/c4-wz-wm-port/WMA_READY.json')


class WMContractError(ValueError):
    """The WM returned something that violates the contract (missing ids, wrong K, non-finite)."""


def moment_input(m, state_file=None):
    """Observable moment dict passed to observe(): no world labels, no global aggregates."""
    G.check_seed(m['seed'], where=f"(moment {m.get('sid')})")
    return dict(moment_id=moment_id(m['sid']), t0=int(m['t0']), map=G.MAP, cavs=[dict(c) for c in m['all_cavs_t0']],
                state_file=None if state_file is None else str(state_file), inserted_t0=m.get('inserted_t0'),
                crn_seed=F.crn_seed(m['sid']))


def moment_id(sid):
    return hashlib.sha256(f'wmc-moment-v1:{sid}'.encode()).hexdigest()[:16]


def default_observe(moment):
    rows = [r for r in (G.wm_row(c) for c in moment['cavs']) if r is not None]
    return dict(moment_id=moment['moment_id'], t0=moment['t0'], cavs=moment['cavs'], wm_rows=rows, source='cav_self_states',
                state_file=moment.get('state_file'))


def action(aid, kind, family, key, members, H, cavs=None):
    prof = F.open_loop_profile(members, H, cavs) if members else {}
    return dict(id=aid, kind=kind, family=family, key=key, members=[dict(m) for m in members], law='hold_speed',
                profile=prof, duration_s=max([len(v) for v in prof.values()] or [0]))


def normalize(raw, actions, K, H):
    """Contract check + {aid: [float]*K}."""
    if isinstance(raw, dict) and 'scores' in raw and isinstance(raw['scores'], dict):
        raw = raw['scores']
    if isinstance(raw, dict) and str(H) in raw and isinstance(raw[str(H)], dict):
        raw = raw[str(H)]
    out = {}
    for a in actions:
        draws = raw.get(a['id']) if isinstance(raw, dict) else None
        if draws is None:
            raise WMContractError(f"WM returned no draws for action {a['id']!r}")
        vals = []
        for d in draws:
            v = d.get('team_delay', d.get('J')) if isinstance(d, dict) else d
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
                raise WMContractError(f"non-finite WM cost for {a['id']!r}: {v!r}")
            vals.append(float(v))
        if len(vals) != int(K):
            raise WMContractError(f"WM returned {len(vals)} draws for {a['id']!r}, expected K={K}")
        out[a['id']] = vals
    return out


class StubWM:
    """CPU stand-in until WMA_READY.json exists: truth = fakes.PlantedWorld, plus a CRN-common draw
    term, an action-specific draw term (prediction uncertainty) and an optional systematic bias.
    Never used for a real comparison: proposals carry backend='stub' and the plug-in refuses them
    unless --allow-stub."""

    name = 'stub'

    def __init__(self, world=None, noise_sd=60.0, common_sd=200.0, bias=0.0, base_level=7500.0, sys_sd=0.0):
        self.world = world or fakes.PlantedWorld()
        self.noise_sd, self.common_sd, self.bias, self.base_level = float(noise_sd), float(common_sd), float(bias), base_level
        self.sys_sd = float(sys_sd)                # per-action systematic error (a mis-ranking WM)
        self.meta = dict(backend='stub', noise_sd=self.noise_sd, common_sd=self.common_sd, bias=self.bias,
                         sys_sd=self.sys_sd, world=type(self.world).__name__)

    def observe(self, moment):
        return default_observe(moment)

    def evaluate(self, state, actions, K, H, seed):
        scale = float(H) / G.H_MAIN
        out = {}
        for a in actions:
            eff = self.world.effect(a['members']) * (1.0 + self.bias) * scale
            akey = a['key'] if a['kind'] != 'base' else 'base'
            if a['members'] and self.sys_sd:
                eff += fakes._rng('stub_sys', state['moment_id'], akey).gauss(0.0, self.sys_sd)
            draws = []
            for k in range(int(K)):
                common = fakes._rng('stub_common', state['moment_id'], seed, k).gauss(0.0, self.common_sd)
                own = fakes._rng('stub_own', state['moment_id'], seed, k, akey).gauss(0.0, self.noise_sd) if a['members'] else 0.0
                draws.append(self.base_level * scale + common - eff + own)
            out[a['id']] = draws
        return out


def _resolve(spec):
    mod, _, fn = str(spec).partition(':')
    if not mod or not fn:
        raise ValueError(f'API spec must be module:function, got {spec!r}')
    return getattr(importlib.import_module(mod), fn)


def find_ready(path=None):
    cands = [path] if path else [os.environ.get(READY_ENV)] + list(DEFAULT_READY_LOCATIONS)
    for c in cands:
        if c and Path(c).is_file():
            return G.check_path(Path(c))
    return None


class WMABackend:
    """The WM-A ported model bound through WMA_READY.json (+ optional overrides)."""

    name = 'wma'

    def __init__(self, ready=None, api=None, load_kwargs=None):
        path = find_ready(ready)
        if path is None:
            raise FileNotFoundError('WMA_READY.json not found (WM-A port not delivered); use --wm stub for a dry run')
        text = path.read_bytes()
        doc = json.loads(text)
        spec = dict(doc.get('api') or {})
        spec.update({k: v for k, v in (api or {}).items() if v})
        if 'evaluate' not in spec:
            raise ValueError(f'{path} has no api.evaluate; pass --wm-api evaluate=pkg.mod:fn')
        self._evaluate = _resolve(spec['evaluate'])
        self._observe = _resolve(spec['observe']) if spec.get('observe') else None
        load = _resolve(spec['load']) if spec.get('load') else None
        kw = dict(doc.get('load_kwargs') or {})
        kw.update(load_kwargs or {})
        self.handle = load(**kw) if load else None
        self.meta = dict(backend='wma', ready=str(path), ready_sha256=hashlib.sha256(text).hexdigest(), api=spec,
                         wma_commits=doc.get('commits'), load_kwargs=kw)

    def observe(self, moment):
        return self._observe(self.handle, moment) if self._observe else default_observe(moment)

    def evaluate(self, state, actions, K, H, seed):
        return self._evaluate(self.handle, state, actions, K=int(K), H=int(H), seed=int(seed))


class _Factory:
    def __init__(self, obj):
        self.obj = obj
        self.name = getattr(obj, 'name', type(obj).__name__)
        self.meta = getattr(obj, 'meta', dict(backend=self.name))

    def observe(self, moment):
        return self.obj.observe(moment)

    def evaluate(self, state, actions, K, H, seed):
        return self.obj.evaluate(state, actions, K, H, seed)


def load_backend(spec='auto', api=None, ready=None, **kw):
    """'stub' | 'wma' | 'wma:PATH' | 'auto' (WM-A if WMA_READY.json exists, else refuse) |
    'pkg.mod:factory' (factory(**kw) -> object with observe/evaluate)."""
    if spec == 'stub':
        return StubWM(**kw)
    if spec in ('wma', 'auto') or str(spec).startswith('wma:'):
        path = str(spec)[4:] if str(spec).startswith('wma:') else ready
        if spec == 'auto' and find_ready(path) is None:
            raise FileNotFoundError('auto: WMA_READY.json not found; the real comparison needs the WM-A port '
                                    '(use --wm stub only for dry runs)')
        return WMABackend(path, api=api, load_kwargs=kw or None)
    return _Factory(_resolve(spec)(**kw))
