"""S-WMVERIFY (phase wm_verify): SUMO paired-seed verification of WORLD-MODEL-PROPOSED strategies on the MAPPO u0
base, solo2 incumbent in every seed group, output in EXACTLY the S-REGIMEU0 ANALYSIS schema (input of FULLPOL's
clean5 EV builder ``fullpol_learn.build_guidance(mappo=ANALYSIS.json)``).

Harness = S-REGIMEU0 (tools.coopmwm.regimeu0 + imitreplay, c4/imitreplay c08ff784): same clean4 plant and wz3to2
map, MAPPO u0 (the S-REGIMEU0 inputs, sha-checked), precharged ledger (never re-run), regimeu0.worker, native warmup
0-300 s, episodes 1,200 s, J = team delay over [300, 1200) s (regimeu0.control_j), paired CRN per seed group (same
scenario bytes, SUMO seed, noise seed), bootstrap 10,000 resamples RNG 20261106 (regimeu0.boot_ci).
Convention: diff = J(arm) - J(reference), veh-s; NEGATIVE = the arm is better.

Arms of a seed group (cell, seed):
  u0            MAPPO u0 alone
  wall_h_solo2  the incumbent, regimeu0.RuleHook('wall_h_solo2') VERBATIM (the S-REGIMEU0 arm, same name)
  solo2_core    SOLO2 through strategy_hook.StrategyCore (executor-parity control, first `parity_seeds` seeds/cell;
                reported in VERIFY.json only, never an ANALYSIS family)
  <spec name>   candidates (WM-C ARMS.json groups wm/wm_fair/structured/random, or spec files): trigger,
                selection and timing = strategy_hook.StrategyCore; members executed by wmverify_hook: hold-only
                strategies through the plant writer (as solo2), strategies with lane changes through the LCEXEC
                POLICY executor (lc_families.Protocol + own-sensing lane gate + env.lc_tick, c4/lcexec 0fe9574e):
                'verified' = executable by the policy executor
Seeds: 989950-989999 only (MAIN allocation; LSSEARCH has 989900-989949). Map registry: map_scenarios.EXTRA_TRAIN_RANGES.

    python -m tools.wzprop.wmverify validate     --arms ARMS.json
    python -m tools.wzprop.wmverify registration --arms ARMS.json --cells D3200_p60:989950-989965 ... --cap N
    python -m tools.wzprop.wmverify dryrun       --arms ARMS.json --cells ... --out NEW_DIR      (no SUMO)
    python -m tools.wzprop.wmverify plan         --root R --regimeu0-root U --arms ARMS.json --cells ... --cap N
                                                 --registration .../REGISTRATION.md [--prior-root R0] [--no-identity]
    python -m tools.wzprop.wmverify run          --root R [--workers 16]   (cpuV: nice 19, CUDA_VISIBLE_DEVICES=,
                                                 COOPMWM_MAP_SPEC=wz3to2 spec, COOPMWM_G5_RUNTIME=1)
    python -m tools.wzprop.wmverify worker       --root R --job ID
    python -m tools.wzprop.wmverify check        --root R [R ...]          (integrity; exit 1 on failure)
    python -m tools.wzprop.wmverify analyze      --root R [R ...] --out NEW_DIR   (ANALYSIS.json, VERIFY.json,
                                                 RESULTS.jsonl, CHECK.json)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from . import geometry as G
from . import strategy_search as SS
from . import wmverify_hook as VH

SCHEMA = 'wmverify.v1'
STABLE_ID = 'S-WMVERIFY'
PHASE = 'wm_verify'
SEED_RANGE = (989950, 989999)            # MAIN 2026-11-06 night: wm_verify 989950-989999; LSSEARCH 989900-989949
FORBIDDEN_SEEDS = ((989600, 989699), (962000, 962999), (964000, 964199), (977000, 977999))
BASE_ARM = 'u0'
INCUMBENT = 'wall_h_solo2'
PARITY_ARM = 'solo2_core'
RESERVED = (BASE_ARM, 'base', INCUMBENT, PARITY_ARM)
GROUPS = ('wm', 'wm_fair', 'structured', 'random')
SINGLE_GROUPS = ('best_beating_solo2', 'best_fair')    # single specs in ARMS.json (dedup by key)
EPISODE_S = 1200
DEFAULT_LANE_EXECUTOR = 'lcexec'       # MAIN ~00:0x: lane programs through the LCEXEC policy executor
DEFAULT_HOLD_EXECUTOR = 'plant'        # hold strategies run as S-REGIMEU0 solo2 (plant single writer)
NAME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_.-]{0,63}$')
SPEC_KEYS = frozenset({'name', 'trigger', 'zone', 'spread_m', 'cooldown_s', 'slots', 'gate', 'roles'})
REQUIRED_KEYS = SPEC_KEYS - {'gate'}
ROLE_KEYS = frozenset({'lane', 'rank', 'v_w', 'dmax', 'hold_s', 'onset_s', 'lc_target'})
# Registered values beyond strategy_search.validate (which checks v_w/dmax/hold_s/onset_s against the NEWMAP2 R grid
# + WMDATA offsets via lcm_plan, lanes, ranks, adjacency, gates, zones):
SPREADS = (G.WALL_SPREAD, G.R_SPREAD)    # pg_plan WALL_SPREAD 200 m / R spread 300 m
SLOTS = (1,)                             # S-REGIMEU0: one activation at a time (cooldown >= hold)
CLOSING_SUMO_LANE = 0                    # up lane 0 ends at X_B (WM lane 2)
IDENTITY_JOB = 'identity_s989599_u0'


# ------------------------------------------------------------------------------------------ helpers
def _harness():
    from tools.coopmwm import imitreplay as ir
    from tools.coopmwm import regimeu0 as R
    return ir, R


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, indent=1, sort_keys=True, default=str) + '\n')


def _read(path):
    import gzip
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    return json.loads(path.read_text(encoding='utf-8'))


def guard_path(path):
    """Refuse restricted paths before opening (geometry + imitreplay restricted tokens)."""
    G.check_path(path)
    low = str(path).replace('\\', '/').lower()
    for tok in ('hetdata_truth', 'vehicle_params', 'driver_params', 'vtype', 'sealed'):
        if tok in low:
            raise ValueError(f'restricted path refused (never opened): {path}')
    return Path(path)


def check_seed(seed):
    s = int(seed)
    if any(lo <= s <= hi for lo, hi in FORBIDDEN_SEEDS):
        raise ValueError(f'held-out / evaluation seed {s} refused')
    if not SEED_RANGE[0] <= s <= SEED_RANGE[1]:
        raise ValueError(f'seed {s} outside the wm_verify range {SEED_RANGE[0]}-{SEED_RANGE[1]}')
    return s


def parse_cells(items):
    """['D3200_p60:989950-989965', ...] -> [(cell_key, demand, share, seeds)] (distinct seeds across cells)."""
    out, seen = [], set()
    for item in items:
        cell, span = item.split(':')
        m = re.fullmatch(r'D(\d+)_p(\d+)', cell)
        if m is None:
            raise ValueError(f'cell key {cell!r}')
        lo, hi = (int(x) for x in span.split('-'))
        seeds = tuple(check_seed(s) for s in range(lo, hi + 1))
        if seen & set(seeds):
            raise ValueError(f'seeds reused across cells: {sorted(seen & set(seeds))}')
        seen.update(seeds)
        out.append((cell, int(m.group(1)), int(m.group(2)) / 100.0, seeds))
    return out


# ------------------------------------------------------------------------------------------ spec validation
def check_spec(spec, lane_executor=DEFAULT_LANE_EXECUTOR):
    """Strict validation: known keys only, registered ranges only (strategy_search.validate + SPREADS/SLOTS/cooldown),
    CAV programs only (no vehicle ids anywhere; execution selects is_cav rows), executable lane targets."""
    if not isinstance(spec, dict):
        raise ValueError('spec must be a dict')
    unknown = set(spec) - SPEC_KEYS
    if unknown:
        raise ValueError(f'unknown spec keys {sorted(unknown)}')
    missing = REQUIRED_KEYS - set(spec)
    if missing:
        raise ValueError(f'missing spec keys {sorted(missing)}')
    name = spec['name']
    if not isinstance(name, str) or not NAME_RE.match(name) or name in RESERVED or 'minus' in name:
        raise ValueError(f'strategy name {name!r} not allowed')
    if not isinstance(spec['roles'], list) or not spec['roles']:
        raise ValueError('roles must be a non-empty list')
    for r in spec['roles']:
        if not isinstance(r, dict) or set(r) != ROLE_KEYS:
            raise ValueError(f'role keys must be exactly {sorted(ROLE_KEYS)}')
        for k in ('lane', 'rank', 'hold_s', 'onset_s'):
            if type(r[k]) is not int:
                raise ValueError(f'role {k} must be an int')
        if r['lc_target'] is not None and type(r['lc_target']) is not int:
            raise ValueError('lc_target must be an int or None')
        for k in ('v_w', 'dmax'):
            if isinstance(r[k], bool) or not isinstance(r[k], (int, float)):
                raise ValueError(f'role {k} must be a number')
    for k in ('cooldown_s', 'slots'):
        if type(spec[k]) is not int:
            raise ValueError(f'{k} must be an int')
    s = dict(spec, gate=spec.get('gate'), roles=[dict(r) for r in spec['roles']])
    SS.validate(s)
    if float(s['spread_m']) not in SPREADS:
        raise ValueError(f"spread_m {s['spread_m']} not registered {SPREADS}")
    if s['slots'] not in SLOTS:
        raise ValueError(f"slots {s['slots']} not registered {SLOTS}")
    g = SS.grids()
    if int(s['cooldown_s']) > max(g['hold_s']) + max(g['onset_s']):
        raise ValueError('cooldown above the registered hold + onset range')
    if VH.has_lane_roles(s):
        if lane_executor not in VH.EXECUTORS:
            raise ValueError(f'unknown lane executor {lane_executor!r}')
        if lane_executor == 'plant' and any(r['lc_target'] == CLOSING_SUMO_LANE for r in s['roles']):
            raise ValueError('closing-lane entry is not executable on the frozen G5 lane path (use lcexec)')
    return s


def candidates(arms_doc, cell, groups=GROUPS, k=4, extra=(), lane_executor=DEFAULT_LANE_EXECUTOR):
    """[(source, spec)] for one cell: ARMS.json groups (first k each), then extra specs; SOLO2 duplicates dropped,
    identical strategies deduplicated by strategy_key, names unique."""
    out, keys, names = [], {SS.strategy_key(dict(SS.SOLO2)): INCUMBENT}, {}
    doc = (arms_doc or {}).get(cell, {}) if arms_doc else {}
    items = []
    for grp in groups:
        got = doc.get(grp)
        items += [(grp, s) for s in ((got or [])[:k] if grp not in SINGLE_GROUPS else ([got] if got else []))]
    items += [('extra', s) for s in extra]
    for source, raw in items:
        s = check_spec(raw, lane_executor)
        key = SS.strategy_key(s)
        if key in keys:
            continue
        if s['name'] in names:
            raise ValueError(f"two different strategies named {s['name']!r}")
        keys[key] = names[s['name']] = s['name']
        out.append((source, s))
    return out


# ------------------------------------------------------------------------------------------ plan
def build_groups(cells, specs_by_cell, world_of, *, parity_seeds=2, identity=True,
                 lane_executor=DEFAULT_LANE_EXECUTOR, hold_executor=DEFAULT_HOLD_EXECUTOR):
    """Identity group first, then paired seed groups round-robin over cells (seed index major) so a partial run
    covers every cell."""
    groups = []
    if identity:
        groups.append(dict(stage='identity', jobs=[dict(id=IDENTITY_JOB, arm=BASE_ARM, strategy=None, strategy_spec=None,
                                                        executor='u0', world=world_of(989599),
                                                        sim_s=EPISODE_S)]))
    depth = max(len(seeds) for *_x, seeds in cells)
    for i in range(depth):
        for cell, demand, share, seeds in cells:
            if i >= len(seeds):
                continue
            s = seeds[i]
            w = world_of(int(s), int(demand), float(share))
            if w['cell_key'] != cell:
                raise ValueError(f'world cell {w["cell_key"]} != {cell}')
            jobs = [dict(id=f'{cell}_s{s}_{BASE_ARM}', arm=BASE_ARM, strategy=None, strategy_spec=None, executor='u0',
                         world=w, sim_s=EPISODE_S),
                    dict(id=f'{cell}_s{s}_{INCUMBENT}', arm=INCUMBENT, strategy=INCUMBENT, strategy_spec=None,
                         executor='regimeu0', world=w, sim_s=EPISODE_S)]
            if i < parity_seeds:
                jobs.append(dict(id=f'{cell}_s{s}_{PARITY_ARM}', arm=PARITY_ARM, strategy=PARITY_ARM,
                                 strategy_spec=dict(SS.SOLO2, name=PARITY_ARM, gate=None), executor='plant',
                                 world=w, sim_s=EPISODE_S))
            for _source, spec in specs_by_cell[cell]:
                jobs.append(dict(id=f'{cell}_s{s}_{spec["name"]}', arm=spec['name'], strategy=spec['name'],
                                 strategy_spec=spec, world=w, sim_s=EPISODE_S,
                                 executor=VH.resolve_executor(spec, lane_executor=lane_executor,
                                                              hold_executor=hold_executor)))
            groups.append(dict(stage='paired', cell=cell, jobs=jobs))
    return groups


def module_shas():
    here = Path(__file__).resolve().parent
    coop = here.parent / 'coopmwm'
    files = dict(wmverify=here / 'wmverify.py', wmverify_hook=here / 'wmverify_hook.py',
                 strategy_hook=here / 'strategy_hook.py', strategy_search=here / 'strategy_search.py',
                 regimeu0=coop / 'regimeu0.py', imitreplay=coop / 'imitreplay.py', lc_runtime=coop / 'lc_runtime.py',
                 lc_program=coop / 'lc_program.py', lc_families=coop / 'lc_families.py', lc_roles=coop / 'lc_roles.py',
                 map_scenarios=coop / 'map_scenarios.py')
    return {k: _sha_file(p) for k, p in files.items() if p.exists()}


def registration_check(text, *, cap, cells):
    """The run's REGISTRATION.md must carry the S-WMVERIFY block (stable id, phase, seed range, cap) before physics."""
    need = [STABLE_ID, PHASE, f'{SEED_RANGE[0]}-{SEED_RANGE[1]}']
    missing = [x for x in need if x not in text]
    if f'{int(cap):,}' not in text and str(int(cap)) not in text:
        missing.append(f'cap {int(cap):,}')
    for cell, *_ in cells:
        if cell not in text:
            missing.append(cell)
    if missing:
        raise ValueError(f'REGISTRATION.md lacks the S-WMVERIFY registration items {missing}')


def family_records(specs_by_cell, *, lane_executor=DEFAULT_LANE_EXECUTOR, hold_executor=DEFAULT_HOLD_EXECUTOR):
    """FULLPOL per-family metadata (frozen at plan time): spec, key, executor, the lcexec-free-program-v1 slot
    template (lc_families.template_of-valid), the offer condition (trigger geometry used) and the LCEXEC parity
    verdict id of the executor that verified it."""
    out = {}
    for cell, lst in specs_by_cell.items():
        for source, s in lst:
            if s['name'] not in out:
                ex = VH.resolve_executor(s, lane_executor=lane_executor, hold_executor=hold_executor)
                tpl, tsha = VH.family_template(s)
                out[s['name']] = dict(
                    spec=s, strategy_key=SS.strategy_key(s), executor=ex, cells={}, template=tpl,
                    template_sha256=tsha, offer_condition=VH.offer_condition(s),
                    lcexec_parity=dict(VH.LCEXEC_PARITY) if ex == 'lcexec' else None,
                    verification_note=('members executed by the LCEXEC policy executor (Protocol + own-sensing lane '
                                       'gate; MAPPO pre-safety filter bypassed as in S-LCEXEC-SMOKE)'
                                       if ex == 'lcexec' else
                                       'hold members executed through the plant single writer with MAPPO safety '
                                       'downstream (S-REGIMEU0 solo2 path), NOT through the LCEXEC template'))
            out[s['name']]['cells'][cell] = source
    return out


def plan(root, *, regimeu0_root, arms_path, cells, cap, registration_md=None, groups=GROUPS, k=4, parity_seeds=2,
         identity=True, lane_executor=DEFAULT_LANE_EXECUTOR, hold_executor=DEFAULT_HOLD_EXECUTOR, spec_paths=(),
         prior_roots=(), dry_run=False):
    ir, R = _harness()
    root = guard_path(root)
    if (root / 'PLAN.json').exists():
        raise ValueError('PLAN.json already frozen')
    cells = list(cells)
    arms_doc = json.loads(guard_path(arms_path).read_text(encoding='utf-8')) if arms_path else {}
    extra = []
    for p in spec_paths:
        doc = json.loads(guard_path(p).read_text(encoding='utf-8'))
        extra += doc if isinstance(doc, list) else [doc]
    specs = {cell: candidates(arms_doc, cell, groups, k, extra, lane_executor) for cell, *_ in cells}
    prior = set()
    for pr in prior_roots:
        body = _read(Path(pr) / 'PLAN.json')
        prior |= {g['jobs'][0]['world']['seed'] for g in body['groups'] if g['stage'] == 'paired'}
        if identity and any(g['stage'] == 'identity' for g in body['groups']):
            raise ValueError('identity episode already planned in a prior root (use --no-identity)')
    clash = prior & {s for *_x, seeds in cells for s in seeds}
    if clash:
        raise ValueError(f'seeds already planned in a prior root: {sorted(clash)}')
    gr = build_groups(cells, specs, R.world_of, parity_seeds=parity_seeds, identity=identity,
                      lane_executor=lane_executor, hold_executor=hold_executor)
    planned = sum(j['sim_s'] for g in gr for j in g['jobs'])
    if planned > int(cap):
        raise ValueError(f'planned {planned} sim-s > cap {cap}')
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    reg_sha = None
    if not dry_run:
        text = guard_path(registration_md).read_text(encoding='utf-8')
        registration_check(text, cap=cap, cells=cells)
        reg_sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
        src = Path(regimeu0_root) / 'inputs'
        (root / 'inputs').mkdir(exist_ok=True)
        for name in ('CONFIG.json', 'u0.pt'):
            shutil.copyfile(guard_path(src / name), root / 'inputs' / name)
            files[name] = dict(source=str(src / name), sha256=ir.sha_file(root / 'inputs' / name))
        ref = _read(Path(regimeu0_root) / 'PLAN.json')['inputs']['u0.pt']['sha256']
        if files['u0.pt']['sha256'] != ref:
            raise ValueError('MAPPO u0 differs from the S-REGIMEU0 base')
        files['u0.pt']['regimeu0_sha256'] = ref
        if arms_path:
            shutil.copyfile(guard_path(arms_path), root / 'inputs' / 'ARMS.json')
            files['ARMS.json'] = dict(source=str(arms_path), sha256=ir.sha_file(root / 'inputs' / 'ARMS.json'))
    families = family_records(specs, lane_executor=lane_executor, hold_executor=hold_executor)
    shas = module_shas()
    reg = dict(schema=SCHEMA, stable_id=STABLE_ID, phase=PHASE, cap_sim_s=int(cap), dry_run=bool(dry_run),
               seeds=sorted(s for *_x, seeds in cells for s in seeds), cells=[c[0] for c in cells],
               seed_range=list(SEED_RANGE), registration_md_sha256=reg_sha, module_sha256=shas, unix=time.time())
    _write_new(root / 'REGISTRATION.json', reg)
    body = dict(schema=SCHEMA + '.plan', stable_id=STABLE_ID, phase=PHASE, cap_sim_s=int(cap), dry_run=bool(dry_run),
                inputs=files, groups=gr, families=families, incumbent=INCUMBENT, parity_arm=PARITY_ARM,
                parity_seeds=int(parity_seeds), lane_executor=lane_executor, hold_executor=hold_executor,
                planned_sim_s=planned,
                cells=[dict(cell=c, demand=d, share=p, seeds=list(s)) for c, d, p, s in cells],
                module_sha256=shas, registration_md_sha256=reg_sha, unix=time.time())
    body['plan_sha256'] = ir.digest({k2: v for k2, v in body.items() if k2 != 'unix'})
    _write_new(root / 'PLAN.json', body)
    return body


# ------------------------------------------------------------------------------------------ worker / run
def _find_job(body, ident):
    for g in body['groups']:
        for j in g['jobs']:
            if j['id'] == ident:
                return j
    raise KeyError(ident)


def make_job_hook(job, R):
    """Hook of one job: u0 / incumbent through regimeu0.RuleHook verbatim, candidates through wmverify_hook."""
    if job['executor'] == 'u0':
        return R.RuleHook(None)
    if job['executor'] == 'regimeu0':
        if job['strategy'] != INCUMBENT:
            raise ValueError('regimeu0 executor runs the incumbent only')
        return R.RuleHook(INCUMBENT)
    if job['executor'] in VH.EXECUTORS:
        return VH.make_hook(job['strategy_spec'], warmup_s=R.WARMUP_S, executor=job['executor'])
    raise ValueError(job['executor'])


def needs_lc_runtime(job):
    cls = VH.EXECUTORS.get(job.get('executor'))
    return bool(cls is not None and cls.needs_lc_runtime)


def worker(root, ident):
    """regimeu0.worker (identical plant, ledger checks and SUMMARY) with this job's hook; sidecar WMVERIFY.json."""
    ir, R = _harness()
    from tools.coopmwm import accel_plant
    root = Path(root)
    body = _read(root / 'PLAN.json')
    if body.get('dry_run'):
        raise ValueError('dry-run plan: no physics')
    job = _find_job(body, ident)
    hook = make_job_hook(job, R)
    lc = needs_lc_runtime(job)
    original = (R.RuleHook, R._job, accel_plant.bind_runtime, os.environ.get('COOPMWM_LC_RUNTIME'))
    lc_record = None
    if lc:
        os.environ['COOPMWM_LC_RUNTIME'] = '1'
        bind = accel_plant.bind_runtime

        def bind_runtime(j):
            exact = bind(j)
            from tools.coopmwm import lc_runtime
            lc_runtime.install(j)                # after map_runtime.install (LCEXEC contract); idempotent
            return exact
        accel_plant.bind_runtime = bind_runtime
    else:
        os.environ.pop('COOPMWM_LC_RUNTIME', None)   # u0 / incumbent / hold arms: the exact S-REGIMEU0 code path
    R.RuleHook = lambda _name: hook
    R._job = lambda _root, _ident: dict(job)
    try:
        summary = R.worker(root, ident)
        if lc:
            from tools.coopmwm import lc_runtime
            lc_record = lc_runtime.installed_record()
    finally:
        R.RuleHook, R._job, accel_plant.bind_runtime = original[:3]
        if original[3] is None:
            os.environ.pop('COOPMWM_LC_RUNTIME', None)
        else:
            os.environ['COOPMWM_LC_RUNTIME'] = original[3]
    side = dict(schema=SCHEMA + '.job', id=ident, arm=job['arm'], executor=job['executor'],
                strategy_key=None if job['strategy_spec'] is None else SS.strategy_key(job['strategy_spec']),
                activations=len(getattr(hook, 'acts', []) or []),
                receipts=hook.receipts() if hasattr(hook, 'receipts') else None, lc_runtime=lc_record,
                error=bool(summary.get('error')))
    _write_new(root / 'jobs' / ident / 'WMVERIFY.json', side)
    return summary


def run(root, **kw):
    """regimeu0.run (precharged ledger, cap, STOP file, load-aware dispatch); workers = this module; ledger phase
    label wm_verify."""
    ir, R = _harness()
    import subprocess
    if _read(Path(root) / 'PLAN.json').get('dry_run'):
        raise ValueError('dry-run plan: no physics')
    real, phase = subprocess.Popen, ir.PHASE

    def popen(cmd, *a, **k2):
        if isinstance(cmd, list) and 'tools.coopmwm.regimeu0' in cmd:
            cmd = [c if c != 'tools.coopmwm.regimeu0' else 'tools.wzprop.wmverify' for c in cmd]
        return real(cmd, *a, **k2)
    R.subprocess.Popen = popen
    ir.PHASE = PHASE
    try:
        return R.run(root, **kw)
    finally:
        R.subprocess.Popen = real
        ir.PHASE = phase


# ------------------------------------------------------------------------------------------ analysis (zero physics)
def _rec(R, ir, s, base):
    """regimeu0.analyze per-arm record (verbatim)."""
    rec = dict(status='ok', J=R.control_j(s), totals=s.get('totals'), safety=ir.safety(s),
               activations=len(s.get('activations') or ()), lane_held_steps=s.get('lane_held_steps'))
    return rec


WMSEARCH_SCHEMA = 'wmc.wmsearch_u0.v1.plan'      # WM-C S-WMSEARCH roots (arm ids wm_k/str_k/rnd_k, solo2 incumbent)


def arm_alias(body):
    """Arm id -> ANALYSIS family name. wm_verify roots: identity. S-WMSEARCH roots (wmsearch_u0 PLAN with 'specs'):
    the spec name, the SOLO2-keyed arm -> 'wall_h_solo2' (StrategyCore SOLO2 on the same plant; the RuleHook /
    StrategyCore parity is what the wm_verify solo2_core arm checks in SUMO)."""
    specs = body.get('specs') if body.get('schema') == WMSEARCH_SCHEMA else None
    if not specs:
        return lambda arm: arm
    solo = SS.strategy_key(dict(SS.SOLO2))
    names = {}
    for arm, spec in specs.items():
        if spec is None:
            names[arm] = BASE_ARM
        elif SS.strategy_key(spec) == solo:
            names[arm] = INCUMBENT
        else:
            names[arm] = spec['name']
    if len(set(names.values())) != len(names):
        raise ValueError('S-WMSEARCH arms do not map to distinct family names')
    return lambda arm: names.get(arm, arm)


def wmsearch_families(body):
    """VERIFY families of an S-WMSEARCH root: executed by strategy_hook.make_hook (G5 lane path), NOT the LCEXEC
    policy executor (closing-lane lane targets were refused by the frozen env there)."""
    out = {}
    alias = arm_alias(body)
    for arm, spec in (body.get('specs') or {}).items():
        if spec is None or alias(arm) in (BASE_ARM, INCUMBENT):
            continue
        tpl, tsha = VH.family_template(spec)
        out[alias(arm)] = dict(spec=spec, strategy_key=SS.strategy_key(spec), executor='plant', cells={},
                               template=tpl, template_sha256=tsha, offer_condition=VH.offer_condition(spec),
                               lcexec_parity=None, source_arm=arm,
                               verification_note='S-WMSEARCH: WM-C strategy_hook.make_hook on the u0 plant (hold law '
                                                 'through the plant writer, lane targets through the frozen G5 path: '
                                                 'closing-lane entries refused); not the LCEXEC policy executor')
        for g in body['groups']:
            if g['stage'] == 'paired':
                out[alias(arm)]['cells'][g['cell']] = 'wmsearch'
    return out


def collect(roots):
    """regimeu0.analyze's per-seed rows (verbatim records) + identity + parity/side records."""
    ir, R = _harness()
    roots = [Path(r) for r in ([roots] if isinstance(roots, (str, Path)) else roots)]
    rows, side, identity, parity, failed, arms = [], {}, None, [], [], []
    for root in roots:
        body = _read(root / 'PLAN.json')
        alias = arm_alias(body)

        def load(ident):
            p = root / 'jobs' / ident / 'SUMMARY.json.gz'
            return ir._read(p) if p.exists() else None
        for g in body['groups']:
            if g['stage'] == 'identity':
                s = load(g['jobs'][0]['id'])
                if s is not None:
                    tot = (s.get('totals') or {}).get('team_delay_vs')
                    identity = dict(team_delay_vs=tot, expected=R.IDENTITY['expected_team_delay_vs'],
                                    equal=tot is not None and abs(tot - R.IDENTITY['expected_team_delay_vs']) < 1e-6,
                                    error=(s.get('error') or {}).get('error'))
                continue
            sums = {alias(j['arm']): load(j['id']) for j in g['jobs']}
            for j in g['jobs']:
                if alias(j['arm']) not in arms and j['arm'] != PARITY_ARM:
                    arms.append(alias(j['arm']))
                p = root / 'jobs' / j['id'] / 'WMVERIFY.json'
                if p.exists():
                    side[(g['jobs'][0]['world']['seed'], alias(j['arm']))] = _read(p)
            w = g['jobs'][0]['world']
            base = sums.get(BASE_ARM)
            row = dict(seed=w['seed'], cell=w['cell_key'], root=str(root), arms={})
            for arm, s in sums.items():
                if arm == PARITY_ARM:
                    if s is not None:
                        inc = sums.get(INCUMBENT)
                        parity.append(dict(seed=w['seed'], cell=w['cell_key'], error=(s.get('error') or {}).get('error'),
                                           J=None if s.get('error') else R.control_j(s),
                                           J_incumbent=None if inc is None or inc.get('error') else R.control_j(inc),
                                           acts=len(s.get('activations') or ()),
                                           acts_incumbent=None if inc is None else len(inc.get('activations') or ()),
                                           first_divergence_s=None if inc is None or inc.get('error') or s.get('error')
                                           else ir.first_divergence(inc, s)))
                    continue
                if s is None:
                    row['arms'][arm] = dict(status='not_run')
                    continue
                if s.get('error'):
                    row['arms'][arm] = dict(status='failed', error=s['error']['error'][:200])
                    failed.append(dict(seed=w['seed'], cell=w['cell_key'], arm=arm, error=s['error']['error'][:400]))
                    continue
                rec = _rec(R, ir, s, base)
                if arm != BASE_ARM and base is not None and not base.get('error'):
                    rec['diff'] = rec['J'] - R.control_j(base)
                    rec['first_divergence_s'] = ir.first_divergence(base, s)
                    first_act = min((a['t'] for a in s.get('activations') or ()), default=None)
                    rec['first_activation_s'] = first_act
                    rec['crn_ok'] = rec['first_divergence_s'] is None or (
                        first_act is not None and rec['first_divergence_s'] >= first_act)
                row['arms'][arm] = rec
            rows.append(row)
    return dict(roots=roots, rows=rows, side=side, identity=identity, parity=parity, failed=failed, arms=arms)


def cells_from_rows(rows, arms):
    """regimeu0.analyze per-cell statistics, verbatim, with ALL_ARMS replaced by ``arms`` (order kept)."""
    import random
    ir, R = _harness()
    BOOT, BOOT_SEED = R.BOOT, R.BOOT_SEED
    cells = {}
    for cell in sorted({r['cell'] for r in rows}):
        rs = [r for r in rows if r['cell'] == cell]
        out = {}
        for arm in [x for x in arms if any(x in r['arms'] for r in rs)]:
            recs = [r['arms'][arm] for r in rs if r['arms'].get(arm, {}).get('status') == 'ok']
            a = dict(n=len(recs), mean_J=(sum(x['J'] for x in recs) / len(recs)) if recs else None,
                     failed=sum(r['arms'].get(arm, {}).get('status') == 'failed' for r in rs))
            diffs = [x['diff'] for x in recs if x.get('diff') is not None]
            if arm != 'u0' and diffs:
                lo, hi = R.boot_ci(diffs)
                a.update(n_paired=len(diffs), mean_diff=sum(diffs) / len(diffs), ci95=[lo, hi],
                         benefit_rate=sum(d < 0 for d in diffs) / len(diffs), harm_rate=sum(d > 0 for d in diffs) / len(diffs),
                         crn_ok=sum(bool(x.get('crn_ok')) for x in recs),
                         mean_activations=sum(x['activations'] for x in recs) / len(recs))
            for k in ('collisions', 'teleports', 'ttc_below_3_vs', 'hard_brake'):
                vals = [x['safety'].get(k) for x in recs if x['safety'].get(k) is not None]
                a[k] = sum(vals) if vals else None
                a[k + '_mean'] = (sum(vals) / len(vals)) if vals else None
            out[arm] = a
        base_mean = out.get('u0', {}).get('mean_J')
        for arm, a in out.items():
            if arm != 'u0' and a.get('mean_diff') is not None and base_mean:
                a['rel_diff'] = a['mean_diff'] / base_mean
                a['rel_ci95'] = [a['ci95'][0] / base_mean, a['ci95'][1] / base_mean]
        present = [x for x in arms if any(x in r['arms'] for r in rs)]
        ok = [r for r in rs if all(r['arms'].get(x, {}).get('status') == 'ok' for x in present)]
        solos = [x for x in present if 'solo' in x]
        for group in [g for g in R.GROUP_ARMS if g in present]:
            if not ok or not solos:
                continue
            rng = random.Random(BOOT_SEED)

            def stat(rs_, group=group):
                mean = {x: sum(r['arms'][x]['J'] for r in rs_) / len(rs_) for x in (group, *solos)}
                return mean[group] - min(mean[x] for x in solos)
            draws = sorted(stat([ok[rng.randrange(len(ok))] for _ in ok]) for _ in range(BOOT))
            beats = [r['arms'][group]['J'] < min(r['arms'][x]['J'] for x in solos) for r in ok]
            key = 'group_minus_best_solo' if group == 'wall_h' else f'{group}_minus_best_solo'
            out[key] = dict(point=stat(ok), ci95=[draws[int(.025 * BOOT)], draws[int(.975 * BOOT) - 1]], n=len(ok),
                            solos=solos, group=group, beats_all_solos_fraction=sum(beats) / len(beats))
        cells[cell] = out
    return cells


def analysis(roots, col=None):
    """ANALYSIS.json body in EXACTLY the S-REGIMEU0 schema (keys of regimeu0.analyze)."""
    ir, R = _harness()
    col = col or collect(roots)
    order = [BASE_ARM] + [a for a in col['arms'] if a != BASE_ARM]
    rep = dict(schema=R.SCHEMA + '.analysis', roots=[str(r) for r in col['roots']],
               spent=sum(ir.spent(r) for r in col['roots']),
               plan_sha256=[_read(Path(r) / 'PLAN.json')['plan_sha256'] for r in col['roots']], cells={},
               seeds=col['rows'])
    if col['identity'] is not None:
        rep['identity'] = col['identity']
    rep['cells'] = cells_from_rows(col['rows'], order)
    return rep


def _mean(x):
    return sum(x) / len(x) if x else None


def _paired(rows, arm, ref, field):
    out = []
    for r in rows:
        a, b = r['arms'].get(arm, {}), r['arms'].get(ref, {})
        if a.get('status') == 'ok' and b.get('status') == 'ok':
            va = a['J'] if field == 'J' else (a.get('safety') or {}).get(field)
            vb = b['J'] if field == 'J' else (b.get('safety') or {}).get(field)
            if va is not None and vb is not None:
                out.append((r['seed'], va - vb, a, b))
    return out


def contrasts(rows, arms, ref=INCUMBENT):
    """Per cell per strategy: paired arm - ref (solo2 by default) with the regimeu0 bootstrap; per-seed fractions;
    safety deltas vs u0 and vs ref. diff < 0 = the strategy is better."""
    ir, R = _harness()
    out = {}
    for cell in sorted({r['cell'] for r in rows}):
        rs = [r for r in rows if r['cell'] == cell]
        cell_out = {}
        for arm in [a for a in arms if a not in (BASE_ARM, ref) and any(a in r['arms'] for r in rs)]:
            p = _paired(rs, arm, ref, 'J')
            d = [x[1] for x in p]
            rec = dict(reference=ref, n=len(d))
            if d:
                lo, hi = R.boot_ci(d)
                ref_mean = _mean([x[3]['J'] for x in p])
                rec.update(mean_diff=_mean(d), ci95=[lo, hi], rel_diff=_mean(d) / ref_mean if ref_mean else None,
                           rel_ci95=[lo / ref_mean, hi / ref_mean] if ref_mean else None,
                           better_share=sum(x < 0 for x in d) / len(d), worse_share=sum(x > 0 for x in d) / len(d),
                           beats_ref_ci=hi < 0, worse_than_ref_ci=lo > 0,
                           per_seed={str(s): v for s, v, _a, _b in p})
            pu = _paired(rs, arm, BASE_ARM, 'J')
            both = [r for r in rs if all(r['arms'].get(x, {}).get('status') == 'ok' for x in (arm, ref, BASE_ARM))]
            rec['beats_u0_and_ref_share'] = (sum(r['arms'][arm]['J'] < min(r['arms'][ref]['J'], r['arms'][BASE_ARM]['J'])
                                                 for r in both) / len(both)) if both else None
            rec['vs_u0'] = dict(n=len(pu), benefit_share=(sum(x[1] < 0 for x in pu) / len(pu)) if pu else None,
                                harm_share=(sum(x[1] > 0 for x in pu) / len(pu)) if pu else None)
            saf = {}
            for k in ('hard_brake', 'ttc_below_3_vs', 'collisions', 'teleports'):
                for r_name in (BASE_ARM, ref):
                    q = [x[1] for x in _paired(rs, arm, r_name, k)]
                    if q:
                        lo2, hi2 = R.boot_ci(q)
                        saf[f'{k}_minus_{r_name}'] = dict(n=len(q), mean=_mean(q), ci95=[lo2, hi2])
            totals = {k: sum(((r['arms'][arm].get('safety') or {}).get(k) or 0.) for r in rs
                             if r['arms'].get(arm, {}).get('status') == 'ok') for k in ('collisions', 'teleports')}
            rec['safety'] = dict(saf, totals=totals,
                                 failed=sum(r['arms'].get(arm, {}).get('status') == 'failed' for r in rs))
            cell_out[arm] = rec
        out[cell] = cell_out
    return out


def checks(roots, col=None):
    """Integrity checks (must pass before ANALYSIS.json is used): identity episode J, CRN (no divergence before the
    first activation), executor parity (solo2_core == wall_h_solo2), ledger/cap, inputs; warnings: failed jobs,
    lane programs without executed lane requests, zero-activation strategies; safety totals reported separately."""
    ir, R = _harness()
    col = col or collect(roots)
    roots = col['roots']
    res, warn = [], []

    def add(name, ok, **detail):
        res.append(dict(name=name, ok=bool(ok), **detail))
    planned_identity = any(g['stage'] == 'identity' for r in roots for g in _read(Path(r) / 'PLAN.json')['groups'])
    if planned_identity:
        idn = col['identity']
        add('identity_u0_989599', idn is not None and idn['equal'], value=idn,
            expected=R.IDENTITY['expected_team_delay_vs'])
    bad = [dict(seed=r['seed'], cell=r['cell'], arm=a, first_divergence_s=x.get('first_divergence_s'),
                first_activation_s=x.get('first_activation_s'))
           for r in col['rows'] for a, x in r['arms'].items() if x.get('status') == 'ok' and a != BASE_ARM
           and x.get('crn_ok') is False]
    add('crn_no_divergence_before_first_activation', not bad, violations=bad)
    zero = [dict(seed=r['seed'], cell=r['cell'], arm=a, diff=x.get('diff'))
            for r in col['rows'] for a, x in r['arms'].items() if x.get('status') == 'ok' and a != BASE_ARM
            and x.get('activations') == 0 and x.get('diff') not in (None, 0.0)]
    add('crn_zero_activation_equals_u0', not zero, violations=zero)
    par_bad = [p for p in col['parity'] if p['error'] or p['J'] is None or p['J_incumbent'] is None
               or abs(p['J'] - p['J_incumbent']) > 1e-6 or p['acts'] != p['acts_incumbent']
               or p['first_divergence_s'] is not None]
    planned_parity = any(j['arm'] == PARITY_ARM for r in roots for g in _read(Path(r) / 'PLAN.json')['groups']
                         for j in g['jobs'])
    add('executor_parity_solo2_core_equals_incumbent', (bool(col['parity']) or not planned_parity) and not par_bad,
        n=len(col['parity']), planned=planned_parity, violations=par_bad)
    led = []
    for r in roots:
        reg = _read(Path(r) / 'REGISTRATION.json')
        charged = {c['id'] for c in ir.charges(r)}
        done = {p.parent.name for p in Path(r).glob('jobs/*/DONE.json')}
        fail = {p.parent.name for p in Path(r).glob('jobs/*/FAILED.json')}
        dirs = {p.name for p in Path(r).glob('jobs/*') if p.is_dir() and p.name != '_logs'}
        led.append(dict(root=str(r), cap=reg['cap_sim_s'], spent=ir.spent(r), charged=len(charged), done=len(done),
                        failed=len(fail), unfinished=sorted(charged - done - fail),
                        uncharged_dirs=sorted(dirs - charged)))
    add('ledger_within_cap_no_uncharged_jobs', all(x['spent'] <= x['cap'] and not x['uncharged_dirs'] for x in led),
        roots=led)
    shas, refs = set(), set()
    for r in roots:
        body = _read(Path(r) / 'PLAN.json')
        u0 = body['inputs'].get('u0.pt') or {}
        shas.add(u0.get('sha256'))
        # wmsearch_u0.plan refuses a u0 differing from the S-REGIMEU0 base at plan time (no stored reference)
        refs.add(u0.get('regimeu0_sha256') if body.get('schema') != WMSEARCH_SCHEMA else u0.get('sha256'))
    add('inputs_u0_identical_and_equal_regimeu0', len(shas) == 1 and shas == refs and None not in shas,
        u0_sha256=sorted(x for x in shas if x))
    if col['failed']:
        warn.append(dict(name='failed_jobs', jobs=col['failed']))
    lane_arms = {}
    for (seed, arm), sd in col['side'].items():
        if sd.get('executor') != 'lcexec' or not sd.get('receipts'):
            continue
        rc = sd['receipts']
        pr = rc.get('protocol') or {}
        a = lane_arms.setdefault(arm, dict(seeds=0, activations=0, sessions=0, refused=0, lc_ticks=0, status={},
                                           gate_withheld={}))
        a['seeds'] += 1
        a['activations'] += rc.get('activations', 0)
        a['sessions'] += pr.get('sessions', 0)
        a['refused'] += pr.get('n_refusals', 0)
        a['lc_ticks'] += rc.get('lc_ticks', 0)
        for k, v in (rc.get('lc_vehicle_status') or {}).items():
            a['status'][k] = a['status'].get(k, 0) + v
        for k, v in (pr.get('gate_withheld') or {}).items():
            a['gate_withheld'][k] = a['gate_withheld'].get(k, 0) + v
    for arm, a in lane_arms.items():
        if a['refused']:
            warn.append(dict(name='lcexec_refused_activations', arm=arm, **a))
        if a['activations'] and not (a['status'].get('issued', 0) + a['status'].get('held', 0)):
            warn.append(dict(name='lane_program_without_executed_lane_requests', arm=arm, **a))
    acts = {}
    for r in col['rows']:
        for a, x in r['arms'].items():
            if x.get('status') == 'ok' and a != BASE_ARM:
                acts[a] = acts.get(a, 0) + x.get('activations', 0)
    for a, n in acts.items():
        if n == 0:
            warn.append(dict(name='strategy_never_activated', arm=a))
    safety = {}
    for r in col['rows']:
        for a, x in r['arms'].items():
            if x.get('status') == 'ok':
                s = safety.setdefault(a, dict(collisions=0., teleports=0.))
                for k in ('collisions', 'teleports'):
                    s[k] += float((x.get('safety') or {}).get(k) or 0.)
    for f in col['failed']:
        safety.setdefault(f['arm'], dict(collisions=0., teleports=0.)).setdefault('failed', 0)
        safety[f['arm']]['failed'] = safety[f['arm']].get('failed', 0) + 1
    return dict(ok=all(x['ok'] for x in res), checks=res, warnings=warn, lane_receipts=lane_arms, safety=safety,
                safety_clean=all(not s.get('collisions') and not s.get('teleports') and not s.get('failed')
                                 for s in safety.values()))


def results_rows(rows):
    """REGIMEU0_RESULTS.jsonl episode rows (FULLPOL's alternative input; J_300_1200_vs = control-window J)."""
    out = []
    for r in rows:
        for arm, x in sorted(r['arms'].items()):
            if x.get('status') not in ('ok', 'failed'):
                continue
            if x['status'] == 'failed':
                out.append(dict(arm=arm, cell=r['cell'], seed=r['seed'], root=Path(r['root']).name, status='failed',
                                error=x.get('error')))
                continue
            tot, saf = x.get('totals') or {}, x.get('safety') or {}
            out.append(dict(J_300_1200_vs=x['J'], activations=x['activations'], arm=arm, cell=r['cell'],
                            collisions=saf.get('collisions'), crn_ok=x.get('crn_ok'), diff_vs_u0=x.get('diff'),
                            first_activation_s=x.get('first_activation_s'), first_divergence_s=x.get('first_divergence_s'),
                            hard_brake_vs=saf.get('hard_brake'), road_delay_vs=tot.get('road_delay_vs'),
                            root=Path(r['root']).name, seed=r['seed'], source_wait_vs=tot.get('source_wait_vs'),
                            status='ok', team_delay_vs=tot.get('team_delay_vs'), teleports=saf.get('teleports'),
                            ttc_below_3_vs=saf.get('ttc_below_3_vs')))
    return out


def analyze(roots, out_dir=None):
    """ANALYSIS.json (S-REGIMEU0 schema), VERIFY.json (contrasts vs solo2, integrity, safety, families),
    RESULTS.jsonl, CHECK.json into a NEW directory (never overwritten)."""
    roots = [Path(r) for r in ([roots] if isinstance(roots, (str, Path)) else roots)]
    col = collect(roots)
    rep = analysis(roots, col)
    chk = checks(roots, col)
    families = {}
    for r in roots:
        body = _read(r / 'PLAN.json')
        for name, f in (body.get('families') or wmsearch_families(body)).items():
            families.setdefault(name, f)
    order = [BASE_ARM] + [a for a in col['arms'] if a != BASE_ARM]
    ver = dict(schema=SCHEMA + '.verify', stable_id=STABLE_ID, phase=PHASE, roots=[str(r) for r in roots],
               plan_sha256=rep['plan_sha256'], incumbent=INCUMBENT,
               convention='diff = J(arm) - J(reference) over [300, 1200) s, veh-s; negative = the arm is better',
               families=families, vs_incumbent=contrasts(col['rows'], order), integrity=chk,
               parity=col['parity'])
    summ = {}
    for cell, arms in ver['vs_incumbent'].items():
        beat = sorted(a for a, v in arms.items() if v.get('beats_ref_ci'))
        best = min((a for a in arms if arms[a].get('mean_diff') is not None), key=lambda a: arms[a]['mean_diff'],
                   default=None)
        summ[cell] = dict(n_strategies=len(arms), beat_incumbent_ci=beat, best=best,
                          best_vs_incumbent=None if best is None else {k: arms[best].get(k) for k in
                                                                       ('n', 'mean_diff', 'ci95', 'rel_diff',
                                                                        'better_share')})
    ver['summary'] = summ
    if out_dir is not None:
        out = guard_path(out_dir)
        out.mkdir(parents=True, exist_ok=False)
        text = json.dumps(rep, indent=1, sort_keys=True, default=str) + '\n'
        (out / 'ANALYSIS.json').open('x', encoding='utf-8').write(text)
        ver['analysis_sha256'] = hashlib.sha256(text.encode('utf-8')).hexdigest()
        _write_new(out / 'VERIFY.json', ver)
        _write_new(out / 'CHECK.json', chk)
        with (out / 'RESULTS.jsonl').open('x', encoding='utf-8') as f:
            for row in results_rows(col['rows']):
                f.write(json.dumps(row, sort_keys=True, default=str) + '\n')
    return rep, ver


# ------------------------------------------------------------------------------------------ registration text
def budget(cells, n_candidates_by_cell, parity_seeds=2, identity=True):
    total = EPISODE_S if identity else 0
    for cell, _d, _p, seeds in cells:
        arms = 2 + n_candidates_by_cell[cell]
        total += len(seeds) * arms * EPISODE_S + min(parity_seeds, len(seeds)) * EPISODE_S
    return total


def registration_text(*, cells, specs_by_cell, cap, parity_seeds=2, identity=True,
                      lane_executor=DEFAULT_LANE_EXECUTOR, hold_executor=DEFAULT_HOLD_EXECUTOR, arms_sha256=None,
                      regimeu0_u0_sha256=None, smoke_seeds=1):
    planned = budget(cells, {c: len(v) for c, v in specs_by_cell.items()}, parity_seeds, identity)
    shas = module_shas()
    lines = [
        f'### {STABLE_ID} (phase {PHASE}) - registration (append BEFORE any physics)',
        '- Question: do the WORLD-MODEL-PROPOSED strategies (WM-C S-WMSEARCH ARMS.json) beat the solo2 incumbent '
        '(and MAPPO u0) per demand cell on fresh TRAIN-type seeds, when executed by the POLICY executor? Output = '
        'verified EV rows for the FULLPOL clean5 EV-BC builder (S-REGIMEU0 ANALYSIS schema) + per-family LCEXEC '
        'templates and offer conditions (VERIFY.json).',
        '- Closest prior: S-REGIMEU0 (ANALYSIS c0b578f8): solo2 (wall_h_solo2) -24% at D3200, -14% at D2800_p60, '
        '+31-39% at D2400; walls dominated by solo2. S-WMSEARCH amendment 1 (WM/structured/random vs solo2 on '
        'REGIME seeds 989770-989785). Exact difference: fresh seeds 989950-989999, WM-proposed specs incl. lane '
        'programs executed by the LCEXEC policy executor, incumbent in every seed group, executor-parity control.',
        f'- Base: MAPPO u0 d9042cc5 = the S-REGIMEU0 inputs u0.pt (sha256 {regimeu0_u0_sha256 or "<from regimeu0 PLAN>"}'
        '), coopmwm clean4 plant, wz3to2, native warmup 0-300 s, episodes 1,200 s, J = team delay over [300, 1200) s.',
        f'- Seeds: {SEED_RANGE[0]}-{SEED_RANGE[1]} only (TRAIN-type, never evaluation; MAIN allocation 2026-11-06; '
        'LSSEARCH has 989900-989949; 989900-989999 found unused in all 921 local branches, worktrees and run '
        'registries on 2026-11-06). Registered in map_scenarios.EXTRA_TRAIN_RANGES on c4/wmverify only.',
        '- Cells / seeds: ' + '; '.join(f'{c} {s[0]}-{s[-1]} ({len(s)})' for c, _d, _p, s in cells) + '.',
        f'- Arms per seed group (paired CRN): u0; {INCUMBENT} (incumbent, regimeu0.RuleHook verbatim); {PARITY_ARM} '
        f'(SOLO2 via strategy_hook.StrategyCore on the plant path, executor parity, first {parity_seeds} seeds per '
        'cell, VERIFY only); candidates [source, strategy key, executor]:']
    for cell, lst in specs_by_cell.items():
        lines.append(f'  - {cell}: ' + ', '.join(
            f'{s["name"]} [{src}, {SS.strategy_key(s)}, '
            f'{VH.resolve_executor(s, lane_executor=lane_executor, hold_executor=hold_executor)}]' for src, s in lst))
    lines += [
        f'- ARMS.json sha256: {arms_sha256 or "<ARMS.json>"}.',
        '- Executors: trigger, member selection and timing = WM-C strategy_hook.StrategyCore (same select_members, '
        'roles, gate, cooldown as the WM search). Hold-only strategies (executor plant): hold law clip(v_w - v, -dmax, '
        '1.0) as the absolute pre-safety request through the plant single writer (MAPPO safety downstream), own-lane '
        'G5 target = the S-REGIMEU0 solo2 path. Strategies with lane changes (executor lcexec): the LCEXEC POLICY '
        'executor (c4/lcexec 0fe9574e, COOPMWM_LC_RUNTIME=1): member_specs -> lc_roles.member_program -> '
        'template_from_program -> Protocol.start / receive / commands (own 100 m sensing, own-sensing lane gate) -> '
        'env.lc_tick, exactly the S-LCEXEC-SMOKE production path; program members bypass the MAPPO pre-safety filter '
        'as in S-LCEXEC-SMOKE (speedMode 27, direct_target bounds, env edge-end guard stay). Verified = executable '
        'by the policy executor. CAV rows only; HDVs never commanded. Parameter ranges: strategy_search.validate '
        '(NEWMAP2 R grid, WMDATA offsets, zones, gates) + spread {200, 300} m + one slot + cooldown <= 240 s; '
        'anything else is rejected.',
        f'- Code: c4/wmverify; module sha256 {json.dumps(shas, sort_keys=True)}.',
        f'- Budget (phase {PHASE}): cap {int(cap):,} sim-s; planned {planned:,} sim-s '
        f'({"identity 1,200 + " if identity else ""}seeds x arms x 1,200 + parity).',
        f'- Stage A (smoke): identity + the first {smoke_seeds} seed(s) of every cell, all arms -> '
        '`python -m tools.wzprop.wmverify check` must pass (identity 53,269.32889342696 veh-s, CRN, executor '
        'parity, ledger, inputs; warnings reviewed: LCEXEC refusals, lane programs without executed lane requests, '
        'never-activated strategies) before stage B (remaining seeds, --prior-root A --no-identity).',
        '- Analysis (fixed before results): regimeu0.analyze per cell (paired diff vs u0, bootstrap 10,000 RNG '
        '20261106, 2.5/97.5%), strategy - solo2 paired contrasts (same bootstrap), per-seed better/worse shares, '
        'beats-u0-and-solo2 share, safety (collisions, teleports, hard braking, TTC<3 s; deltas vs u0 and vs solo2). '
        'diff = arm - reference; negative = better. No thresholds tuned on these seeds.',
        '- Stop rules: STOP file; cap; failed integrity check -> stop and report; charged jobs are never re-run.',
        '- Outputs: ANALYSIS.json (S-REGIMEU0 schema; FULLPOL build_guidance(mappo=...)), VERIFY.json (families: '
        'spec, executor, LCEXEC template, offer condition, parity verdict id; contrasts; integrity), RESULTS.jsonl, '
        'CHECK.json in a new analysis directory.',
    ]
    return '\n'.join(lines) + '\n', planned


# ------------------------------------------------------------------------------------------ dry run (no SUMO)
def bind_map_for_dryrun():
    """Bind the wz3to2 descriptor in THIS process when nothing is bound (the lcexec executor reads md.current())."""
    from tools.coopmwm import map_descriptor as md
    from tools.coopmwm.lc_writer_harness import MAP_DIR
    if md._BOUND is None and not os.environ.get(md.ENV):
        md.bind(md.load(MAP_DIR))
    desc = md.current()
    if desc.kind != 'lane_drop':
        raise ValueError(f'dry run needs the wz3to2 process map (bound: {desc.name})')
    return desc


def dryrun(out_dir, *, arms_path, cells, groups=GROUPS, k=4, parity_seeds=2, lane_executor=DEFAULT_LANE_EXECUTOR,
           hold_executor=DEFAULT_HOLD_EXECUTOR, fake_seconds=900, spec_paths=()):
    """Dry plan (no inputs, no physics) + every arm's hook on the zero-physics fake plant for the first seed of each
    cell; the incumbent (regimeu0.RuleHook) and solo2_core must issue identical commands every tick."""
    from . import wmverify_fake as F
    ir, R = _harness()
    bind_map_for_dryrun()
    body = plan(out_dir, regimeu0_root=None, arms_path=arms_path, cells=cells, cap=10 ** 9, groups=groups, k=k,
                parity_seeds=parity_seeds, lane_executor=lane_executor, hold_executor=hold_executor,
                spec_paths=spec_paths, dry_run=True)
    report = dict(schema=SCHEMA + '.dryrun', plan_sha256=body['plan_sha256'], planned_sim_s=body['planned_sim_s'],
                  jobs=sum(len(g['jobs']) for g in body['groups']), cells={},
                  note='fake plant: plumbing only, numbers are not evidence')
    for cell, _d, share, seeds in cells:
        g = next(x for x in body['groups'] if x['stage'] == 'paired' and x['cell'] == cell)
        arms = {}
        traces = {}
        for j in g['jobs']:
            hook = make_job_hook(j, R)
            res = F.run_episode(hook, seed=seeds[0], share=share, seconds=fake_seconds,
                                lc_runtime=needs_lc_runtime(j), record=True)
            traces[j['arm']] = res['trace']
            arms[j['arm']] = dict(executor=j['executor'], activations=len(getattr(hook, 'acts', []) or []),
                                  command_ticks=sum(1 for t in res['trace'] if t['desired']),
                                  fake_team_delay_vs=res['totals'].get('team_delay_vs'),
                                  receipts=hook.receipts() if hasattr(hook, 'receipts') else None)
        par = None
        if PARITY_ARM in traces:
            a, b = traces[INCUMBENT], traces[PARITY_ARM]
            diff = [x['t'] for x, y in zip(a, b) if x['desired'] != y['desired'] or x['g5'] != y['g5']]
            par = dict(identical=not diff and len(a) == len(b), differing_ticks=diff[:10],
                       incumbent_command_ticks=arms[INCUMBENT]['command_ticks'])
        report['cells'][cell] = dict(seed=seeds[0], arms=arms, solo2_parity=par)
    _write_new(Path(out_dir) / 'DRYRUN.json', report)
    return report


# ------------------------------------------------------------------------------------------ CLI
def _cells_arg(ap):
    ap.add_argument('--cells', nargs='+', required=True, help='CELL:first-last, e.g. D3200_p60:989950-989965')


def _executor_args(ap):
    ap.add_argument('--lane-executor', default=DEFAULT_LANE_EXECUTOR, help='strategies with lane changes')
    ap.add_argument('--hold-executor', default=DEFAULT_HOLD_EXECUTOR, help='hold-only strategies')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop.wmverify', description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('validate')
    a.add_argument('--arms')
    a.add_argument('--spec', action='append', default=[])
    _executor_args(a)
    for name in ('registration', 'dryrun', 'plan'):
        a = sub.add_parser(name)
        a.add_argument('--arms')
        a.add_argument('--spec', action='append', default=[])
        _cells_arg(a)
        a.add_argument('--groups', default=','.join(GROUPS))
        a.add_argument('--k', type=int, default=4)
        a.add_argument('--parity-seeds', type=int, default=2)
        _executor_args(a)
        if name == 'registration':
            a.add_argument('--cap', type=int, required=True)
            a.add_argument('--regimeu0-root')
            a.add_argument('--no-identity', action='store_true')
        if name == 'dryrun':
            a.add_argument('--out', required=True)
            a.add_argument('--fake-seconds', type=int, default=900)
        if name == 'plan':
            a.add_argument('--root', required=True)
            a.add_argument('--regimeu0-root', required=True)
            a.add_argument('--cap', type=int, required=True)
            a.add_argument('--registration', required=True, help='the run REGISTRATION.md with the S-WMVERIFY block')
            a.add_argument('--prior-root', action='append', default=[])
            a.add_argument('--no-identity', action='store_true')
    a = sub.add_parser('run')
    a.add_argument('--root', required=True)
    a.add_argument('--workers', type=int, default=16)
    a.add_argument('--low-workers', type=int, default=8)
    a.add_argument('--load-high', type=float, default=44.0)
    a.add_argument('--normal-priority', action='store_true', help='only if MAIN allows (regimeu0.run rule)')
    a = sub.add_parser('worker')
    a.add_argument('--root', required=True)
    a.add_argument('--job', required=True)
    a = sub.add_parser('check')
    a.add_argument('--root', required=True, nargs='+')
    a = sub.add_parser('analyze')
    a.add_argument('--root', required=True, nargs='+')
    a.add_argument('--out', required=True)
    x = ap.parse_args(argv)
    if x.cmd == 'validate':
        doc = json.loads(guard_path(x.arms).read_text()) if x.arms else {}
        specs = [json.loads(guard_path(p).read_text()) for p in x.spec]
        out = {}
        for cell, d in doc.items():
            for grp in GROUPS + SINGLE_GROUPS + ('incumbent',):
                items = d.get(grp)
                for s in (items if isinstance(items, list) else [items] if items else []):
                    try:
                        check_spec(s, x.lane_executor)
                        out.setdefault(cell, []).append(dict(group=grp, name=s.get('name'), ok=True))
                    except (ValueError, KeyError, TypeError) as exc:
                        out.setdefault(cell, []).append(dict(group=grp, name=s.get('name'), ok=False, error=str(exc)))
        for s in specs:
            for t in (s if isinstance(s, list) else [s]):
                try:
                    check_spec(t, x.lane_executor)
                    out.setdefault('spec', []).append(dict(name=t.get('name'), ok=True))
                except (ValueError, KeyError, TypeError) as exc:
                    out.setdefault('spec', []).append(dict(name=t.get('name'), ok=False, error=str(exc)))
        print(json.dumps(out, indent=1))
        return
    if x.cmd in ('registration', 'dryrun', 'plan'):
        cells = parse_cells(x.cells)
        groups = tuple(g for g in x.groups.split(',') if g)
        extra = []
        for p in x.spec:
            d = json.loads(guard_path(p).read_text())
            extra += d if isinstance(d, list) else [d]
    if x.cmd == 'registration':
        doc = json.loads(guard_path(x.arms).read_text()) if x.arms else {}
        specs = {c: candidates(doc, c, groups, x.k, extra, x.lane_executor) for c, *_ in cells}
        u0 = None
        if x.regimeu0_root:
            u0 = _read(Path(x.regimeu0_root) / 'PLAN.json')['inputs']['u0.pt']['sha256']
        text, planned = registration_text(cells=cells, specs_by_cell=specs, cap=x.cap, parity_seeds=x.parity_seeds,
                                          identity=not x.no_identity, lane_executor=x.lane_executor,
                                          hold_executor=x.hold_executor,
                                          arms_sha256=_sha_file(x.arms) if x.arms else None, regimeu0_u0_sha256=u0)
        if planned > x.cap:
            raise SystemExit(f'planned {planned} > cap {x.cap}')
        sys.stdout.write(text)
    elif x.cmd == 'dryrun':
        rep = dryrun(x.out, arms_path=x.arms, cells=cells, groups=groups, k=x.k, parity_seeds=x.parity_seeds,
                     lane_executor=x.lane_executor, hold_executor=x.hold_executor, fake_seconds=x.fake_seconds,
                     spec_paths=x.spec)
        print(json.dumps({c: dict(parity=v['solo2_parity'], arms={a: dict(acts=r['activations'], executor=r['executor'])
                                                                    for a, r in v['arms'].items()})
                          for c, v in rep['cells'].items()}, indent=1))
    elif x.cmd == 'plan':
        body = plan(x.root, regimeu0_root=x.regimeu0_root, arms_path=x.arms, cells=cells, cap=x.cap,
                    registration_md=x.registration, groups=groups, k=x.k, parity_seeds=x.parity_seeds,
                    identity=not x.no_identity, lane_executor=x.lane_executor, hold_executor=x.hold_executor,
                    spec_paths=x.spec, prior_roots=x.prior_root)
        print(json.dumps(dict(plan_sha256=body['plan_sha256'], planned_sim_s=body['planned_sim_s'],
                              arms=sorted(body['families']), groups=len(body['groups']))))
    elif x.cmd == 'run':
        print(run(x.root, workers=x.workers, low_workers=x.low_workers, load_high=x.load_high,
                  normal_priority=x.normal_priority))
    elif x.cmd == 'worker':
        s = worker(x.root, x.job)
        print(json.dumps(dict(id=s['id'], error=bool(s['error']), totals=s['totals'], acts=len(s['activations']))))
    elif x.cmd == 'check':
        res = checks(x.root)
        print(json.dumps(res, indent=1, default=str))
        if not res['ok']:
            raise SystemExit(1)
    else:
        rep, ver = analyze(x.root, x.out)
        print(json.dumps(dict(integrity_ok=ver['integrity']['ok'], safety_clean=ver['integrity']['safety_clean'],
                              summary=ver['summary']), indent=1, default=str))


if __name__ == '__main__':
    main()
