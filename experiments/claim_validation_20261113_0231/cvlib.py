"""Shared, SUMO-free helpers for the claim-validation experiments E1 (WM ordering vs random ordering of a fixed
plan pool) and E2 (resource-matched uniform hold / onset-synchronised copy / same-state timing replay).

Everything here is pure Python (stdlib only). The descriptor language, the 16-hex strategy key, the static legality
rules and the bank document are MIRRORS of the registered code so that the local stages and tests need no code tree:

    tools/wzprop/strategy_search.py   strategy_key, validate, random_strategy   (code_v12r7b, sha 02e119c1)
    tools/coopmwm/oracle_plan.py      strategy_key, select_members, StrategyCore (sha 73cb1ca6)
    tools/coopmwm/a37_cellbank.py     executor_refusal, bank_doc, boot_ci, unc_verdict, admitted_plans (sha 5d9811a0)

`crosscheck()` asserts mirror == registered code whenever the code tree is importable (always run it on the box).
Nothing in this file reads a WM score, a SUMO outcome or a world seed list.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys

RUN_ID = '20261113_0231'
HERE = Path(__file__).resolve().parent
CELL = 'D2800_p60'
SEARCH_SEEDS = (11, 23, 42, 67, 89, 103)
SIZES = (3, 4, 5, 6)
PER_SIZE = 30
TOP_K = 6
PREFIXES = (2, 4, 6)
N_VERIFY_WORLDS = 24
HORIZON_S = 1200           # nm_finaleval.HORIZON_S
WARMUP_S = 300             # oracle_plan / a37_cellbank warmup_s (no plan starts before 300 s)

#: the registered descriptor language (strategy_search.ZONES keys, random_strategy spread choices / wall probability)
ZONES = ('full', 'near', 'far')
SPREADS = (200.0, 300.0)
WALL_P_SMALL = 0.4         # random_strategy: trigger 'wall' iff n <= 3 and rng.random() < 0.4
SLOTS = [(ln, rk) for ln in (0, 1, 2) for rk in (0, 1)]
WALL_SLOTS = [(ln, 0) for ln in (0, 1, 2)]

#: registered files (sha256 checked wherever they are read)
ORIGINAL_BANK_SHA256 = '7e7dc76b8a0cd5c158205a0e3f6ef96d73a36a483c46d03ac422428dc8c573ff'
U0_MAPPO_SHA256 = 'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'   # nm_finaleval.FROZEN_MAPPO_SHA
WM_CKPT_SHA256 = '6da9d3337aad78799c2e6ac97c87e79b927346011f55861e98887bb89728c063'    # ckpt/ckpt/wm_s.pt
WM_CKPT_TAR_SHA256 = 'f9b3a854dbba7adc646a13b0308f40628fef799f6cdbf5900057e084707baa5e'
WM_CODE_TAR_SHA256 = '320e19f3de8bddf4ea789b9c911e68b8dadafdda8a09600cd42e27728bd25de6'  # code_v12r7b.tar

#: world seeds that must never be proposed (task rule); the registered blocks are parameters decided by MAIN
BANNED_SEED_RANGES = ((962000, 962999), (964000, 964199), (977000, 977999), (989600, 989699))


class Refused(RuntimeError):
    """A stage refuses to continue (never overwrite, never guess)."""


# ------------------------------------------------------------------------------------------------ io
def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def sha_json(obj):
    return sha_bytes(canon(obj).encode())


def seal(doc, field='doc_sha256'):
    body = {k: v for k, v in doc.items() if k != field}
    return dict(body, **{field: sha_json(body)})


def verify_seal(doc, field='doc_sha256', what='document'):
    if seal(doc, field)[field] != doc.get(field):
        raise Refused(f'{what}: {field} differs (file edited after it was written)')
    return doc


def text_of(doc):
    return json.dumps(doc, indent=1, sort_keys=True)+'\n'


def write_once(path, text):
    """Checkpoint/resume primitive: an existing file must be byte-identical (then it is kept: resume), otherwise the
    stage refuses. New files are written atomically. -> 'kept' | 'written'."""
    path = Path(path)
    data = text.encode('utf-8')
    if path.exists():
        if path.read_bytes() != data:
            raise Refused(f'{path}: exists with different content (never overwrite)')
        return 'kept'
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return 'written'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stable_rng(tag, seed):
    """An RNG stream named by ``tag`` (independent streams for the pool and the Random arm)."""
    return random.Random(int(hashlib.sha256(f'{tag}|{int(seed)}'.encode()).hexdigest()[:16], 16))


# ------------------------------------------------------------------------------------------------ descriptors
def make_role(lane, rank, v_w, dmax, hold_s, onset_s):
    """Role in the registered value types (float speed / decel, int seconds): the types decide the strategy key."""
    return dict(lane=int(lane), rank=int(rank), v_w=float(v_w), dmax=float(dmax), hold_s=int(hold_s),
                onset_s=int(onset_s), lc_target=None)


def make_strategy(name, trigger, zone, spread_m, roles, cooldown_s=None):
    s = dict(name=name, trigger=trigger, zone=zone, spread_m=float(spread_m), slots=1, roles=list(roles),
             gate=None, start_gate=None, phases=None)
    s['cooldown_s'] = int(max(r['onset_s']+r['hold_s'] for r in roles) if cooldown_s is None else cooldown_s)
    return s


def strategy_key(s):
    """strategy_search.strategy_key / oracle_plan.strategy_key (mirror; crosscheck() proves equality)."""
    body = {k: s[k] for k in ('trigger', 'zone', 'spread_m', 'cooldown_s', 'slots', 'roles')}
    for k in ('gate', 'start_gate', 'phases'):
        if s.get(k) is not None:
            body[k] = s[k]
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def normalise(s):
    """The full descriptor in a canonical form for de-duplication: roles sorted by (lane, rank) (member selection is
    per role, so role order only permutes the member list) and, for the wall trigger, no spread (the executor ignores
    spread_m there and uses the geometry constant). The name is not part of a descriptor."""
    roles = sorted((dict(lane=int(r['lane']), rank=int(r['rank']), v_w=float(r['v_w']), dmax=float(r['dmax']),
                         hold_s=int(r['hold_s']), onset_s=int(r['onset_s']), lc_target=r.get('lc_target'))
                    for r in s['roles']), key=lambda r: (r['lane'], r['rank']))
    return dict(trigger=s['trigger'], zone=s['zone'],
                spread_m=None if s['trigger'] == 'wall' else float(s['spread_m']),
                cooldown_s=int(s['cooldown_s']), slots=int(s['slots']), gate=s.get('gate'),
                start_gate=s.get('start_gate'), phases=s.get('phases'), roles=roles)


def desc_hash(s):
    return sha_json(normalise(s))


def load_grid(cfg):
    """The search grid actually used (cfg['search_grid'] with cfg['onset_max'] applied, as strategy_search.grids)."""
    g = cfg.get('search_grid')
    if not g:
        raise Refused('cfg has no search_grid (the registered D2800_p60 searches all set one)')
    grid = dict(v_w=[float(x) for x in g['v_w']], dmax=[float(x) for x in g['dmax']],
                hold_s=[int(x) for x in g['hold_s']], onset_s=[int(x) for x in g['onset_s']])
    if cfg.get('onset_max') is not None:
        grid['onset_s'] = [x for x in grid['onset_s'] if x <= int(cfg['onset_max'])]
    return grid


def static_legality(s, grid, min_roles=3, max_roles=6):
    """None when ``s`` is a legal, executable hold-only plan; else the rejection reason. One rule for both arms:
    strategy_search.validate (on the search grid) + a37_cellbank.executor_refusal + hold-only / unconditional."""
    if any(r.get('lc_target') is not None for r in s['roles']):
        return 'non_hold_plan'
    if s.get('gate') is not None or s.get('start_gate') is not None or s.get('phases') is not None:
        return 'gate_or_phases'
    if s.get('trigger') not in ('wall', 'lanes') or s.get('zone') not in ZONES:
        return 'trigger_or_zone'
    if not min_roles <= len(s['roles']) <= max_roles:
        return 'role_count'
    if int(s.get('slots', 0)) != 1:
        return 'slots'
    seen = set()
    for r in s['roles']:
        if r['lane'] not in (0, 1, 2) or r['rank'] not in (0, 1):
            return 'lane_or_rank'
        for k in ('v_w', 'dmax', 'hold_s', 'onset_s'):
            if r[k] not in grid[k]:
                return f'{k}_off_grid'
        if (r['lane'], r['rank']) in seen:
            return 'duplicate_role'
        seen.add((r['lane'], r['rank']))
    if s['trigger'] == 'wall' and any(r['rank'] for r in s['roles']):
        return 'wall_rank'
    if not 0 < float(s['spread_m']) <= 300.0:
        return 'spread'
    if int(s['cooldown_s']) < max(r['onset_s']+r['hold_s'] for r in s['roles'])//2:
        return 'cooldown'
    return None


def category(s):
    """Structural category (no outcome, no score): size, trigger, zone, staggered onsets?, lanes covered,
    uniform target speed?"""
    on = [r['onset_s'] for r in s['roles']]
    return '|'.join([f'n{len(s["roles"])}', s['trigger'], s['zone'], 'staggered' if max(on) > min(on) else 'sync',
                     f'lanes{len({r["lane"] for r in s["roles"]})}',
                     'vuniform' if len({r['v_w'] for r in s['roles']}) == 1 else 'vmixed'])


# ------------------------------------------------------------------------------------------------ banks
def executor_refusal(s):
    """a37_cellbank.executor_refusal on the unconditional language (mirror)."""
    if any(r.get('lc_target') is not None for r in s['roles']):
        return 'non_hold_plan'
    if s.get('gate') is not None or s.get('start_gate') is not None or s.get('phases') is not None:
        return 'gate_or_phases'
    if s.get('trigger') not in ('wall', 'lanes') or s.get('zone') not in ZONES:
        return 'trigger_or_zone'
    return None


def bank_doc(plans, cell_key=CELL):
    """a37_cellbank.bank_doc (mirror): unconditional rows + templates for plans = [dict(family, strategy,
    lb95_veh_s)]. Row order is irrelevant to the executor: it sorts unconditional rows by lb95 desc, then key."""
    rows, tpls = [], []
    for p in plans:
        s = p['strategy']
        if s['name'] != p['family'] or executor_refusal(s) is not None:
            raise Refused(f'{p["family"]}: not an executable hold plan')
        cond = dict(trigger=s['trigger'], zone=s['zone'], spread_m=float(s['spread_m']),
                    cooldown_s=int(s['cooldown_s']), slots=int(s['slots']), gate=s['gate'],
                    start_gate=s['start_gate'], phases=s['phases'], situation=dict(unconditional=True),
                    roles=[dict(sumo_lane=int(r['lane']), rank=int(r['rank']), v_w=float(r['v_w']),
                                dmax=float(r['dmax']), hold_s=int(r['hold_s']), onset_s=int(r['onset_s']),
                                lc_target_sumo=r['lc_target']) for r in s['roles']])
        rows.append(dict(regime=cell_key, family=s['name'], lb95_veh_s=float(p.get('lb95_veh_s', 0.0)),
                         plan_source='wm_search', row_class='unconditional', situational=dict(unconditional=True),
                         template_sha256=None, strategy_key=strategy_key(s)))
        tpls.append(dict(cell=cell_key, family=s['name'], offer_condition=cond,
                         template=dict(source=dict(strategy_key=strategy_key(s)))))
    return dict(schema='coopmwm.a37_cellbank.v1.bank', rows=rows, clean5c=dict(templates=tpls))


def template_to_strategy(t, name=None):
    """Bank template -> search-language strategy (wm_desc_score.to_search; search lane == SUMO lane index)."""
    oc = t['offer_condition']
    roles = [dict(lane=int(r['sumo_lane']), rank=r['rank'], v_w=r['v_w'], dmax=r['dmax'], hold_s=r['hold_s'],
                  onset_s=r['onset_s'], lc_target=r.get('lc_target_sumo')) for r in oc['roles']]
    return dict(name=name or t['family'], trigger=oc['trigger'], zone=oc['zone'], spread_m=oc['spread_m'],
                cooldown_s=oc['cooldown_s'], slots=oc['slots'], roles=roles, gate=oc.get('gate'),
                start_gate=oc.get('start_gate'), phases=oc.get('phases'))


def bank_strategies(bank, cell_key=CELL):
    """[dict(family, strategy, lb95_veh_s)] of a bank document in the EXECUTOR's priority order (lb95 desc, key,
    family); refuses a template whose recomputed key differs from the registered one."""
    lb = {r['family']: float(r['lb95_veh_s']) for r in bank['rows'] if r['regime'] == cell_key}
    out = []
    for t in bank['clean5c']['templates']:
        if t['cell'] != cell_key:
            continue
        s = template_to_strategy(t)
        if strategy_key(s) != t['template']['source']['strategy_key']:
            raise Refused(f'{t["family"]}: recomputed strategy key differs from the bank')
        out.append(dict(family=t['family'], strategy=s, lb95_veh_s=lb[t['family']]))
    out.sort(key=lambda p: (-p['lb95_veh_s'], strategy_key(p['strategy']), p['family']))
    return out


def schema_name(bank_schema_from_tree=None):
    return bank_schema_from_tree or 'coopmwm.a37_cellbank.v1.bank'


# ------------------------------------------------------------------------------------------------ seeds
def check_world_seeds(seeds):
    """Refuse a banned world seed. Freshness (every previously used panel) is NOT decided here: MAIN audits and
    registers the blocks; this only blocks the ranges named in the task."""
    bad = [int(s) for s in seeds for lo, hi in BANNED_SEED_RANGES if lo <= int(s) <= hi]
    if bad:
        raise Refused(f'banned world seeds {sorted(set(bad))[:8]}')
    return [int(s) for s in seeds]


def load_seed_blocks(path, keys, n_worlds):
    """{key: [world seeds]} from a MAIN-registered JSON {key: {"start": int, "n": int}} (or explicit lists).
    No default exists on purpose: this code never chooses a world seed."""
    doc = read_json(path)
    out = {}
    for k in keys:
        b = doc.get(str(k))
        if b is None:
            raise Refused(f'seed blocks: no block for {k}')
        seeds = list(b) if isinstance(b, list) else list(range(int(b['start']), int(b['start'])+int(b['n'])))
        if len(seeds) != n_worlds or len(set(seeds)) != n_worlds:
            raise Refused(f'seed blocks: {k} needs {n_worlds} distinct worlds')
        out[str(k)] = check_world_seeds(seeds)
    return out


# ------------------------------------------------------------------------------------------------ code tree
def code_modules(code_root):
    """(oracle_plan, a37_cellbank) from the registered tree, or None when it is not importable here (sparse local
    checkout). Read-only: nothing is written into the tree (-B / dont_write_bytecode)."""
    if code_root and (Path(code_root)/'pytools'/'tools'/'coopmwm'/'oracle_plan.py').is_file():
        code_root = Path(code_root)/'pytools'          # box source-copy layout: <src>/ops + <src>/pytools/tools
    if not code_root or not (Path(code_root)/'tools'/'coopmwm'/'oracle_plan.py').is_file():
        return None
    sys.dont_write_bytecode = True
    root = str(Path(code_root))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        return (importlib.import_module('tools.coopmwm.oracle_plan'),
                importlib.import_module('tools.coopmwm.a37_cellbank'))
    except Exception as e:  # noqa: BLE001  (missing sibling module / numpy / torch on this host)
        return ('unimportable', f'{type(e).__name__}: {e}')


def crosscheck(code_root, strategies, cell_key=CELL):
    """mirror == registered code for every strategy: key, executor refusal, bank document, admission bootstrap.
    -> dict(status='PASS'|'UNAVAILABLE', ...); raises Refused on any difference."""
    mods = code_modules(code_root)
    if mods is None or mods[0] == 'unimportable':
        return dict(status='UNAVAILABLE', reason='code tree not importable here' if mods is None else mods[1])
    O, C = mods
    for s in strategies:
        if O.strategy_key(s) != strategy_key(s):
            raise Refused(f'{s["name"]}: strategy_key mirror differs')
        if C.executor_refusal(s) != executor_refusal(s):
            raise Refused(f'{s["name"]}: executor_refusal mirror differs')
    plans = [dict(family=s['name'], strategy=s, lb95_veh_s=float(i)) for i, s in enumerate(strategies)]
    if canon(C.bank_doc(plans, cell_key)) != canon(bank_doc(plans, cell_key)):
        raise Refused('bank_doc mirror differs')
    from stats import unc_boot_ci
    d = [((i*37) % 11)-5.5 for i in range(24)]
    if tuple(C.boot_ci(d)) != tuple(unc_boot_ci(d)):
        raise Refused('boot_ci mirror differs')
    core = C.CellBankCore(C.bank_doc(plans, cell_key), cell_key) if strategies else None
    return dict(status='PASS', n=len(strategies), a37_digest=C.a37_digest(),
                executable=None if core is None else [p['family'] for p in core.plans],
                refused=None if core is None else core.refused)
