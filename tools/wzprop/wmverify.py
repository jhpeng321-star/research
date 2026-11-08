"""S-WMVERIFY (phase wm_verify): SUMO paired-seed verification of WORLD-MODEL-PROPOSED strategies on the MAPPO u0
base, the gentle incumbent wall_g_solo2 in every seed group, output in EXACTLY the S-REGIMEU0 ANALYSIS schema (input
of FULLPOL's clean5 EV builder ``fullpol_learn.build_guidance(mappo=ANALYSIS.json)``) + per-group fairness (owner
2026-11-07, REGISTRATION 04:03/04:09: ALLOWED = every origin-lane / HDV / CAV group's mean per-vehicle delay change vs
u0 <= 10 s; ANALYSIS_ALLOWED.json keeps only the allowed families; the firm 15 m/s solo2 is BANNED).

Harness = S-REGIMEU0 (tools.coopmwm.regimeu0 + imitreplay, c4/imitreplay c08ff784): same clean4 plant and wz3to2
map, MAPPO u0 (the S-REGIMEU0 inputs, sha-checked), precharged ledger (never re-run), regimeu0.worker, native warmup
0-300 s, episodes 1,200 s, J = team delay over [300, 1200) s (regimeu0.control_j), paired CRN per seed group (same
scenario bytes, SUMO seed, noise seed), bootstrap 10,000 resamples RNG 20261106 (regimeu0.boot_ci).
Convention: diff = J(arm) - J(reference), veh-s; NEGATIVE = the arm is better.

Arms of a seed group (cell, seed):
  u0            MAPPO u0 alone
  wall_g_solo2  the incumbent (22 m/s, 120 s, dmax 1.0, cooldown 120), regimeu0.RuleHook('wall_g_solo2') VERBATIM
  inc_core      SS.INCUMBENT through strategy_hook.StrategyCore (executor-parity control, first `parity_seeds`
                seeds/cell; VERIFY.json only, never an ANALYSIS family)
  wall_h_solo2  optional (--banned-seeds) BANNED firm-hold comparator via RuleHook; labelled, never a family
  <spec name>   candidates (WM-C ARMS.json v3 groups reference / wm / wm_bounded / wm_fairer / structured / random
                [/ wm_fair / best / best_beating_incumbent], or spec files; spec v3: start_gate, phases): trigger,
                selection, phases and timing = strategy_hook.StrategyCore; members executed by wmverify_hook:
                hold-only strategies through the plant writer (as the incumbent), lane-change strategies through the LCEXEC
                POLICY executor (lc_families.Protocol + own-sensing lane gate + env.lc_tick, c4/lcexec 0fe9574e):
                'verified' = executable by the policy executor
Seeds: 989950-989999 only (MAIN allocation; LSSEARCH has 989900-989949). Map registry: map_scenarios.EXTRA_TRAIN_RANGES.

    python -m tools.wzprop.wmverify validate     --arms ARMS.json
    python -m tools.wzprop.wmverify registration --arms ARMS.json --cells D3200_p60:989950-989965 ... --cap N
    python -m tools.wzprop.wmverify dryrun       --arms ARMS.json --cells ... --out NEW_DIR      (no SUMO)
    python -m tools.wzprop.wmverify plan         --root R --regimeu0-root U --arms ARMS.json --cells ... --cap N
                                                 --registration .../REGISTRATION.md [--prior-root R0]
                                                 (every root carries its own identity episode: no bypass)
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
from . import wmverify_fair as WF
from . import wmverify_hook as VH
from . import wmverify_n2 as N2

SCHEMA = 'wmverify.v1'
STABLE_ID = 'S-WMVERIFY'
PHASE = 'wm_verify'
SEED_RANGES = ((989950, 989999),          # MAIN 2026-11-06 night: wm_verify 989950-989999; LSSEARCH 989900-989949
               (991140, 991199),          # MAIN 2026-11-08: 12-seed p20 verification (MULTICRIT used 991100-991132)
               (991200, 991299),          # MAIN 2026-11-08 r4: audited unused (REGISTRATION, every PLAN, job dirs)
               (991300, 991347),          # MAIN 2026-11-08 23:57 second look (mut_0d50e4e1): audited unused
               (991348, 991399),          # MAIN 991200+ allocation (r4-r6 UNC rounds): audited unused 2026-11-09
               (991400, 991419),          # MAIN 2026-11-09 01:55 mut_cfdbafe5 confirmation: audited unused
               (991420, 991499))          # MAIN 2026-11-09 03:1x r6 48-seed confirmation (991420-991467) + r7: audited
FORBIDDEN_SEEDS = ((989600, 989699), (962000, 962999), (964000, 964199), (977000, 977999))
BASE_ARM = 'u0'
INCUMBENT = 'wall_g_solo2'               # owner 2026-11-07 (REGISTRATION 04:03): gentle incumbent, RuleHook verbatim
BANNED_ARM = 'wall_h_solo2'              # the firm 15 m/s solo2: BANNED, optional labelled comparator only
PARITY_ARM = 'inc_core'                  # SS.INCUMBENT through StrategyCore (executor parity vs the RuleHook)
RESERVED = (BASE_ARM, 'base', INCUMBENT, BANNED_ARM, PARITY_ARM, 'solo2_core')
NOT_FAMILIES = (PARITY_ARM, BANNED_ARM)  # never ANALYSIS families (VERIFY only)
GROUPS = ('reference', 'wm', 'wm_bounded', 'wm_fairer', 'structured', 'random')    # ARMS.json v3 lists
OPTIONAL_GROUPS = ('wm_fair',)
SINGLE_GROUPS = ('best_beating_incumbent', 'best')    # single spec / {archive: spec} in ARMS.json (dedup by key)
REG_HEADER = f'### {STABLE_ID} (phase {PHASE}) - registration'     # exact header line (review REV-WMV3 b4)
VEHICLE_LOG_DIR = 'vehicle_logs'         # <root>/vehicle_logs/<job id>/VEHICLES.jsonl.gz (N6 VehicleLog port)
EPISODE_S = 1200
DEFAULT_LANE_EXECUTOR = 'lcexec'       # MAIN ~00:0x: lane programs through the LCEXEC policy executor
DEFAULT_HOLD_EXECUTOR = 'plant'        # hold strategies run as the S-REGIMEU0 RuleHook path (plant single writer)
NAME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_.-]{0,63}$')
SPEC_KEYS = frozenset({'name', 'trigger', 'zone', 'spread_m', 'cooldown_s', 'slots', 'gate', 'start_gate', 'phases',
                       'roles'})
REQUIRED_KEYS = SPEC_KEYS - {'gate', 'start_gate', 'phases'}
ROLE_KEYS = frozenset({'lane', 'rank', 'v_w', 'dmax', 'hold_s', 'onset_s', 'lc_target'})
# Registered values beyond strategy_search.validate (which checks v_w/dmax/hold_s/onset_s against the NEWMAP2 R grid
# + WMDATA offsets via lcm_plan, lanes, ranks, adjacency, gates, zones):
SPREADS = (G.WALL_SPREAD, G.R_SPREAD)    # pg_plan WALL_SPREAD 200 m / R spread 300 m
SLOTS = (1,)                             # S-REGIMEU0: one activation at a time (cooldown >= hold)
CLOSING_SUMO_LANE = 0                    # up lane 0 ends at X_B (WM lane 2)
IDENTITY_JOB = 'identity_s989599_u0'
N2_FIXTURE_S = 700                       # n2_parity fixture: fake plant 300 s warmup + 400 s control
# owner re-plan 2026-11-07 (PAUSE_WMVERIFY_20261107): hold-arm continuation (only_executors=('plant',) on seeds a
# STOPPED prior root planned but never charged) + the opt-in ungated pilot (executor lcexec_ungated, pilot roots only)
PILOT = 'ungated'
PILOT_SUFFIX = '_ungated'
PILOT_EXECUTOR = 'lcexec_ungated'
PILOT_LABEL = ('UNGATED PILOT (WMVERIFY-only, owner 2026-11-07): lane requests failing ONLY the own-sensing gap test '
               'passed to SUMO (laneChangeMode 513 decides); old gate verdict logged as would_withhold; descriptive, '
               'n = 2 seeds per arm; NEVER pooled with the gated arms, never an ANALYSIS / ANALYSIS_ALLOWED / bank family')


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
    if not any(lo <= s <= hi for lo, hi in SEED_RANGES):
        raise ValueError(f'seed {s} outside the wm_verify ranges {SEED_RANGES}')
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
    ph = spec.get('phases')
    if ph is not None and (not isinstance(ph, list) or any(not isinstance(p, list) or any(type(i) is not int for i in p)
                                                           for p in ph)):
        raise ValueError('phases must be None or a list of lists of role indices')
    if spec.get('start_gate') is not None and not isinstance(spec['start_gate'], str):
        raise ValueError('start_gate must be None or a registered name')
    s = dict(spec, gate=spec.get('gate'), start_gate=spec.get('start_gate'),
             phases=None if ph is None else [list(p) for p in ph], roles=[dict(r) for r in spec['roles']])
    SS.validate(s)                       # registered grids/gates/phases; the banned 15 m/s hold is rejected
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
    out, keys, names = [], {SS.strategy_key(dict(SS.INCUMBENT)): INCUMBENT, SS.strategy_key(dict(SS.SOLO2)): BANNED_ARM}, {}
    doc = (arms_doc or {}).get(cell, {}) if arms_doc else {}
    items = []
    for grp in groups:
        got = doc.get(grp)
        if grp == 'best':
            items += [(f'best:{a}', s) for a, s in sorted((got or {}).items()) if s]
        elif grp in SINGLE_GROUPS:
            items += [(grp, got)] if got else []
        else:
            kk = int(k.get(grp, 0)) if isinstance(k, dict) else int(k)     # per-group k (matched budgets)
            items += [(grp, s) for s in (got or [])[:kk]]
    items += [('extra', s) for s in extra]
    for source, raw in items:
        try:
            pre = SS.strategy_key(raw)
        except (KeyError, TypeError):
            pre = None
        if pre in keys and keys[pre] in (INCUMBENT, BANNED_ARM):
            continue                         # the incumbent / banned comparator run as their own arms
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
def pairs_of(arms_doc, cells):
    """PAIR arms (owner 2026-11-09 01:38): {cell: [(name, mini-bank)]} from an ARMS-format doc whose cell entry
    carries 'pairs': [{name, specs: [...]}] (executor: the reviewed ORACLE hook, tools.wzprop.wmverify_oracle)."""
    from . import wmverify_oracle as WO
    out = {}
    for cell, *_x in cells:
        items = ((arms_doc or {}).get(cell) or {}).get('pairs')
        if items:
            out.update(WO.pair_arms({cell: items}))
    return out


def build_groups(cells, specs_by_cell, world_of, *, parity_seeds=2, identity=True,
                 lane_executor=DEFAULT_LANE_EXECUTOR, hold_executor=DEFAULT_HOLD_EXECUTOR, banned_seeds=0,
                 pairs_by_cell=None):
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
            if i < banned_seeds:
                jobs.append(dict(id=f'{cell}_s{s}_{BANNED_ARM}', arm=BANNED_ARM, strategy=BANNED_ARM,
                                 strategy_spec=None, executor='regimeu0', world=w, sim_s=EPISODE_S,
                                 label='banned comparator (owner 2026-11-07)'))
            if i < parity_seeds:
                jobs.append(dict(id=f'{cell}_s{s}_{PARITY_ARM}', arm=PARITY_ARM, strategy=PARITY_ARM,
                                 strategy_spec=dict(SS.INCUMBENT, name=PARITY_ARM), executor='plant',
                                 world=w, sim_s=EPISODE_S))
            for _source, spec in specs_by_cell[cell]:
                jobs.append(dict(id=f'{cell}_s{s}_{spec["name"]}', arm=spec['name'], strategy=spec['name'],
                                 strategy_spec=spec, world=w, sim_s=EPISODE_S,
                                 executor=VH.resolve_executor(spec, lane_executor=lane_executor,
                                                              hold_executor=hold_executor)))
            names = {j['arm'] for j in jobs}
            for name, bank in (pairs_by_cell or {}).get(cell, ()):
                if name in names or name in RESERVED:
                    raise ValueError(f'pair arm name {name!r} clashes with another arm')
                jobs.append(dict(id=f'{cell}_s{s}_{name}', arm=name, strategy=name, strategy_spec=None,
                                 executor='oracle_bank', oracle_bank=bank, world=w, sim_s=EPISODE_S))
            groups.append(dict(stage='paired', cell=cell, jobs=jobs))
    return groups


def select_executors(specs_by_cell, executors, *, lane_executor=DEFAULT_LANE_EXECUTOR,
                     hold_executor=DEFAULT_HOLD_EXECUTOR):
    """Keep only the candidates whose resolved executor is in ``executors`` (re-plan B2: the hold arms, 'plant')."""
    if executors is None:
        return specs_by_cell
    keep = set(executors)
    return {cell: [(src, s) for src, s in lst
                   if VH.resolve_executor(s, lane_executor=lane_executor, hold_executor=hold_executor) in keep]
            for cell, lst in specs_by_cell.items()}


def pilot_specs(specs_by_cell, arms):
    """The pilot arms (names) of every cell: each must be a planned candidate executed by the gated LCEXEC executor."""
    out = {}
    for cell, lst in specs_by_cell.items():
        by = {s['name']: (src, s) for src, s in lst}
        missing = [a for a in arms if a not in by]
        if missing:
            raise ValueError(f'pilot arms not among the {cell} candidates: {missing}')
        bad = [a for a in arms if VH.resolve_executor(by[a][1]) != 'lcexec']
        if bad:
            raise ValueError(f'pilot arms must be lcexec (lane-program) candidates: {bad}')
        if len(set(arms)) != len(arms):
            raise ValueError('duplicate pilot arms')
        out[cell] = [by[a] for a in arms]
    return out


def build_pilot_groups(cells, pilot_by_cell, world_of):
    """Ungated pilot: the mandatory identity episode, then per seed ONLY the pilot jobs (arm '<name>_ungated',
    executor lcexec_ungated, the unchanged spec). Their u0 / incumbent / gated references are the prior root's
    charged jobs of the same seed (paired CRN: same world); no reference job is ever re-run."""
    groups = [dict(stage='identity', jobs=[dict(id=IDENTITY_JOB, arm=BASE_ARM, strategy=None, strategy_spec=None,
                                                executor='u0', world=world_of(989599), sim_s=EPISODE_S)])]
    depth = max(len(seeds) for *_x, seeds in cells)
    for i in range(depth):
        for cell, demand, share, seeds in cells:
            if i >= len(seeds):
                continue
            s = seeds[i]
            w = world_of(int(s), int(demand), float(share))
            if w['cell_key'] != cell:
                raise ValueError(f'world cell {w["cell_key"]} != {cell}')
            jobs = []
            for _source, spec in pilot_by_cell[cell]:
                gated = f'{cell}_s{s}_{spec["name"]}'
                jobs.append(dict(id=gated + PILOT_SUFFIX, arm=spec['name'] + PILOT_SUFFIX, strategy=spec['name'],
                                 strategy_spec=spec, executor=PILOT_EXECUTOR, world=w, sim_s=EPISODE_S,
                                 label=PILOT_LABEL,
                                 pilot=dict(gated_arm=spec['name'], gated_job=gated, u0_job=f'{cell}_s{s}_{BASE_ARM}',
                                            incumbent_job=f'{cell}_s{s}_{INCUMBENT}')))
            groups.append(dict(stage='pilot', cell=cell, jobs=jobs))
    return groups


def _prior_jobs(prior_roots):
    """{job id: (root, job)} over the prior roots' paired groups (pilot pairing; identity episodes and superseded
    never-charged groups skipped)."""
    out = {}
    sup = superseded_groups(prior_roots)
    for pr in prior_roots:
        body = _read(Path(pr) / 'PLAN.json')
        for g in body['groups']:
            if g['stage'] != 'paired' or (body.get('plan_sha256'), g['jobs'][0]['world']['seed']) in sup:
                continue
            for j in g['jobs']:
                if j['id'] in out:
                    raise ValueError(f'job id {j["id"]} planned in two prior roots')
                out[j['id']] = (Path(pr), j)
    return out


def _committed(root):
    """Phase sim-s a prior root holds: what it charged when STOPPED (never resumed) or complete, else its plan."""
    ir, _R = _harness()
    body = _read(Path(root) / 'PLAN.json')
    spent = ir.spent(root)
    return spent if (Path(root) / 'STOP').exists() else max(spent, int(body.get('planned_sim_s') or 0))


def module_shas():
    here = Path(__file__).resolve().parent
    coop = here.parent / 'coopmwm'
    files = dict(wmverify=here / 'wmverify.py', wmverify_hook=here / 'wmverify_hook.py',
                 strategy_hook=here / 'strategy_hook.py', strategy_search=here / 'strategy_search.py',
                 regimeu0=coop / 'regimeu0.py', imitreplay=coop / 'imitreplay.py', lc_runtime=coop / 'lc_runtime.py',
                 lc_program=coop / 'lc_program.py', lc_families=coop / 'lc_families.py', lc_roles=coop / 'lc_roles.py',
                 map_scenarios=coop / 'map_scenarios.py', oracle_plan=coop / 'oracle_plan.py',
                 wmverify_oracle=here / 'wmverify_oracle.py')
    return {k: _sha_file(p) for k, p in files.items() if p.exists()}


def registration_blocks(text):
    """[(header line index, block text)] of every exact S-WMVERIFY registration header in REGISTRATION.md; a block
    runs to the next markdown heading."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if line.rstrip() == REG_HEADER:
            body = []
            for x in lines[i + 1:]:
                if x.startswith('#'):
                    break
                body.append(x)
            out.append((i, '\n'.join(body)))
    return out


def _field(block, pattern):
    m = re.search(pattern, block, re.M)
    return None if m is None else m.group(1)


def registration_check(text, *, cap, planned, jobs_sha256, arms_sha256, module_sha256):
    """REGISTRATION.md must contain the exact header line and, in that block, this run's jobs digest, cap and
    planned budget, ARMS.json sha256 and module sha256 (stray tokens elsewhere never open the gate)."""
    blocks = registration_blocks(text)
    if not blocks:
        raise ValueError(f'REGISTRATION.md lacks the exact header line {REG_HEADER!r}')
    problems = []
    for _i, block in blocks:
        got = dict(jobs=_field(block, r'^- Plan jobs sha256: ([0-9a-f]{64})\s*$'),
                   cap=_field(block, r'cap ([0-9,]+) sim-s; planned [0-9,]+ sim-s'),
                   planned=_field(block, r'cap [0-9,]+ sim-s; planned ([0-9,]+) sim-s'),
                   arms=_field(block, r'^- ARMS\.json sha256: (\S+?)\.\s*$'),
                   modules=_field(block, r'module sha256 (\{.*\})\.\s*$'))
        if got['jobs'] != jobs_sha256:
            problems.append(dict(block_jobs=got['jobs']))
            continue
        mods = json.loads(got['modules']) if got['modules'] else None
        bad = [k for k, ok in (('cap', got['cap'] == f'{int(cap):,}'), ('planned', got['planned'] == f'{int(planned):,}'),
                               ('arms', got['arms'] == (arms_sha256 or '<none>')),
                               ('modules', mods == module_sha256)) if not ok]
        if not bad:
            return True
        problems.append(dict(mismatch=bad))
    raise ValueError(f'no S-WMVERIFY registration block matches this plan: {problems}')


def jobs_digest(groups):
    """Digest of the planned jobs (worlds, arms, specs, executors, sim_s) - the registered identity of the run."""
    from tools.coopmwm import imitreplay as ir
    return ir.digest(groups)


def family_records(specs_by_cell, *, lane_executor=DEFAULT_LANE_EXECUTOR, hold_executor=DEFAULT_HOLD_EXECUTOR):
    """FULLPOL per-family metadata (frozen at plan time): spec, key, executor, the lcexec-free-program-v1 slot
    template (lc_families.template_of-valid), the offer condition (trigger geometry used) and the LCEXEC parity
    verdict id of the executor that verified it."""
    out = {}
    for cell, lst in specs_by_cell.items():
        for source, s in lst:
            if s['name'] not in out:
                ex = VH.resolve_executor(s, lane_executor=lane_executor, hold_executor=hold_executor)
                tpls = [VH.family_template(s, phase=p) for p in range(VH.n_phases(s))]
                tpl, tsha = tpls[0]
                out[s['name']] = dict(
                    spec=s, strategy_key=SS.strategy_key(s), executor=ex, cells={}, template=tpl,
                    template_sha256=tsha, offer_condition=VH.offer_condition(s),
                    phase_templates=None if len(tpls) == 1 else [dict(phase=p, template=d, template_sha256=h)
                                                                 for p, (d, h) in enumerate(tpls)],
                    lcexec_parity=dict(VH.LCEXEC_PARITY) if ex == 'lcexec' else None,
                    n2_parity=None if ex == 'lcexec' else N2.parity(s, executor=ex, seconds=N2_FIXTURE_S,
                                                                       n2_pin=ex == 'plant'),
                    verification_note=('members executed by the LCEXEC policy executor (Protocol + own-sensing lane '
                                       'gate; MAPPO pre-safety filter bypassed as in S-LCEXEC-SMOKE)'
                                       if ex == 'lcexec' else
                                       'hold members executed through the plant single writer with MAPPO safety '
                                       'downstream (S-REGIMEU0 solo2 path), NOT through the LCEXEC template; '
                                       'policy-side parity = n2_parity (FULLPOL N2 hold executor)'))
            out[s['name']]['cells'][cell] = source
    return out


def plan(root, *, regimeu0_root, arms_path, cells, cap, registration_md=None, groups=GROUPS, k=4, parity_seeds=2,
         lane_executor=DEFAULT_LANE_EXECUTOR, hold_executor=DEFAULT_HOLD_EXECUTOR, spec_paths=(),
         prior_roots=(), dry_run=False, banned_seeds=0, only_executors=None, supersede_uncharged=False,
         pilot_arms=None):
    """Every root carries its own identity episode (mandatory; review REV-WMV3 blocker 3).

    Re-plan modes (owner 2026-11-07; both off by default = the 0eee867c plan):
    * ``only_executors=('plant',)`` + ``supersede_uncharged``: hold-arm continuation (B2). Seeds of a prior root are
      refused unless that root is STOPPED and NONE of the seed group's jobs was charged there (the stopped root's
      never-run groups are superseded, recorded in PLAN 'supersedes' and skipped by collect); the phase total
      (prior roots' committed sim-s + this plan) must stay within the cap.
    * ``pilot_arms=[names]``: the ungated pilot (executor lcexec_ungated): the seeds MUST be planned and charged in a
      prior root with u0, the incumbent and the gated arm DONE (pairing); pilot job ids differ from every prior id."""
    ir, R = _harness()
    root = guard_path(root)
    if (root / 'PLAN.json').exists():
        raise ValueError('PLAN.json already frozen')
    cells = list(cells)
    for _c, _d, _p, seeds in cells:
        for x in seeds:
            check_seed(x)
    if not dry_run and lane_executor != 'lcexec':
        raise ValueError('lane programs are verified only by the LCEXEC policy executor (non-dry plans)')
    if pilot_arms and (only_executors is not None or supersede_uncharged or banned_seeds):
        raise ValueError('the pilot plan has no hold-continuation / banned-comparator options')
    arms_doc = json.loads(guard_path(arms_path).read_text(encoding='utf-8')) if arms_path else {}
    extra = []
    for p in spec_paths:
        doc = json.loads(guard_path(p).read_text(encoding='utf-8'))
        extra += doc if isinstance(doc, list) else [doc]
    specs = {cell: candidates(arms_doc, cell, groups, k, extra, lane_executor) for cell, *_ in cells}
    specs = select_executors(specs, only_executors, lane_executor=lane_executor, hold_executor=hold_executor)
    wanted = {s for *_x, seeds in cells for s in seeds}
    prior, supersedes = set(), []
    for pr in ([] if pilot_arms else prior_roots):
        body = _read(Path(pr) / 'PLAN.json')
        charged = {c['id'] for c in ir.charges(pr)} if supersede_uncharged else set()
        sup = []
        for g in body['groups']:
            if g['stage'] != 'paired':
                continue
            seed = g['jobs'][0]['world']['seed']
            if (supersede_uncharged and seed in wanted and (Path(pr) / 'STOP').exists()
                    and not any(j['id'] in charged for j in g['jobs'])):
                sup.append(dict(seed=seed, cell=g['cell'], job_ids=[j['id'] for j in g['jobs']]))
            else:
                prior.add(seed)
        if sup:
            supersedes.append(dict(root=str(pr), plan_sha256=body['plan_sha256'], groups=sup,
                                   rule='prior root STOPPED; none of these groups charged; never resumed'))
    clash = prior & wanted
    if clash:
        raise ValueError(f'seeds already planned in a prior root: {sorted(clash)}')
    pilot = None
    if pilot_arms:
        pspecs = pilot_specs(specs, list(pilot_arms))
        gr = build_pilot_groups(cells, pspecs, R.world_of)
        specs = {}
        pj = _prior_jobs(prior_roots)
        pairs = []
        for g in gr:
            for j in g['jobs']:
                if j['id'] in pj:
                    raise ValueError(f'pilot job id {j["id"]} equals a prior job id (charged jobs are never re-run)')
                if g['stage'] != 'pilot':
                    continue
                refs = {}
                for key in ('u0_job', 'incumbent_job', 'gated_job'):
                    ref = j['pilot'][key]
                    if ref not in pj:
                        raise ValueError(f'pilot reference {ref} is not planned in a prior root')
                    proot, pjob = pj[ref]
                    if pjob['world'] != j['world']:
                        raise ValueError(f'pilot reference {ref}: world differs (no paired CRN)')
                    if not dry_run and not (proot / 'jobs' / ref / 'DONE.json').exists():
                        raise ValueError(f'pilot reference {ref} is not DONE in {proot}')
                    if key == 'gated_job' and (pjob['strategy_spec'] != j['strategy_spec']
                                               or pjob['executor'] != 'lcexec'):
                        raise ValueError(f'pilot reference {ref}: not the gated lcexec run of the same spec')
                    refs[key] = dict(root=str(proot), id=ref)
                pairs.append(dict(job=j['id'], arm=j['arm'], seed=j['world']['seed'], **refs))
        pilot = dict(kind=PILOT, executor=PILOT_EXECUTOR, label=PILOT_LABEL, arms=list(pilot_arms), pairs=pairs)
    else:
        gr = build_groups(cells, specs, R.world_of, parity_seeds=parity_seeds, identity=True,
                          lane_executor=lane_executor, hold_executor=hold_executor, banned_seeds=banned_seeds,
                          pairs_by_cell=pairs_of(arms_doc, cells))
    planned = sum(j['sim_s'] for g in gr for j in g['jobs'])
    jobs_sha = jobs_digest(gr)
    shas = module_shas()
    if planned > int(cap):
        raise ValueError(f'planned {planned} sim-s > cap {cap}')
    phase = None
    if pilot_arms or supersede_uncharged or only_executors is not None:
        prior_committed = {str(pr): _committed(pr) for pr in prior_roots}
        phase = dict(prior_committed=prior_committed, this_plan=planned,
                     total=sum(prior_committed.values()) + planned, cap=int(cap))
        if phase['total'] > int(cap):
            raise ValueError(f'phase total {phase["total"]} sim-s (prior roots + this plan) > cap {cap}')
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    reg_sha = None
    if not dry_run:
        text = guard_path(registration_md).read_text(encoding='utf-8')
        registration_check(text, cap=cap, planned=planned, jobs_sha256=jobs_sha,
                           arms_sha256=ir.sha_file(guard_path(arms_path)) if arms_path else None, module_sha256=shas)
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
    reg = dict(schema=SCHEMA, stable_id=STABLE_ID, phase=PHASE, cap_sim_s=int(cap), dry_run=bool(dry_run),
               seeds=sorted(s for *_x, seeds in cells for s in seeds), cells=[c[0] for c in cells],
               seed_range=[list(r) for r in SEED_RANGES], registration_md_sha256=reg_sha, module_sha256=shas, unix=time.time())
    _write_new(root / 'REGISTRATION.json', reg)
    body = dict(schema=SCHEMA + '.plan', stable_id=STABLE_ID, phase=PHASE, cap_sim_s=int(cap), dry_run=bool(dry_run),
                inputs=files, groups=gr, families=families, incumbent=INCUMBENT, parity_arm=PARITY_ARM,
                banned_comparator=BANNED_ARM if banned_seeds else None, banned_seeds=int(banned_seeds),
                parity_seeds=int(parity_seeds), lane_executor=lane_executor, hold_executor=hold_executor,
                planned_sim_s=planned, jobs_sha256=jobs_sha,
                cells=[dict(cell=c, demand=d, share=p, seeds=list(s)) for c, d, p, s in cells],
                module_sha256=shas, registration_md_sha256=reg_sha, unix=time.time())
    if only_executors is not None:
        body['only_executors'] = sorted(only_executors)
    if supersedes:
        body['supersedes'] = supersedes
    if pilot is not None:
        body['pilot'] = pilot
    if phase is not None:
        body['phase_budget'] = phase
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
        if job['strategy'] not in (INCUMBENT, BANNED_ARM):
            raise ValueError('regimeu0 executor runs the incumbent / the banned comparator only')
        return R.RuleHook(job['strategy'])
    if job['executor'] in VH.EXECUTORS:
        return VH.make_hook(job['strategy_spec'], warmup_s=R.WARMUP_S, executor=job['executor'])
    if job['executor'] == 'oracle_bank':                 # PAIR arms: the reviewed ORACLE hook on a mini-bank
        from . import wmverify_oracle as WO
        return WO.make_recorder(job['oracle_bank'], job['world']['cell_key'])
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
    if job['executor'] in VH.PILOT_EXECUTORS and not body.get('pilot'):
        raise ValueError(f'executor {job["executor"]!r} runs only in a registered pilot root')
    hook = make_job_hook(job, R)
    vlog = WF.VehicleLog(root / VEHICLE_LOG_DIR / ident)   # N6 eval-only ground-truth log on EVERY arm (u0 included)
    lc = needs_lc_runtime(job)
    original = (R.RuleHook, R._job, accel_plant.bind_runtime, os.environ.get('COOPMWM_LC_RUNTIME'))
    lc_record = None
    bind = accel_plant.bind_runtime
    if lc:
        os.environ['COOPMWM_LC_RUNTIME'] = '1'
    else:
        os.environ.pop('COOPMWM_LC_RUNTIME', None)   # u0 / incumbent / hold arms: the exact S-REGIMEU0 code path

    def bind_runtime(j):
        exact = bind(j)
        if lc:
            from tools.coopmwm import lc_runtime
            lc_runtime.install(j)                # after map_runtime.install (LCEXEC contract); idempotent
        return exact
    accel_plant.bind_runtime = bind_runtime
    R.RuleHook = lambda _name: hook
    R._job = lambda _root, _ident: dict(job)
    try:
        with WF.logged_tick(vlog):               # wraps coopfree.tick BEFORE regimeu0.worker adds the Recorder
            summary = R.worker(root, ident)
        if lc:
            from tools.coopmwm import lc_runtime
            lc_record = lc_runtime.installed_record()
    finally:
        vlog_record = vlog.close()
        R.RuleHook, R._job, accel_plant.bind_runtime = original[:3]
        if original[3] is None:
            os.environ.pop('COOPMWM_LC_RUNTIME', None)
        else:
            os.environ['COOPMWM_LC_RUNTIME'] = original[3]
    side = dict(schema=SCHEMA + '.job', id=ident, arm=job['arm'], executor=job['executor'],
                strategy_key=None if job['strategy_spec'] is None else SS.strategy_key(job['strategy_spec']),
                activations=len(getattr(hook, 'acts', []) or []),
                receipts=hook.receipts() if hasattr(hook, 'receipts') else None, lc_runtime=lc_record,
                vehicle_log=vlog_record,
                error=bool(summary.get('error')))
    _write_new(root / 'jobs' / ident / 'WMVERIFY.json', side)
    return summary


def run(root, phase=PHASE, **kw):
    """regimeu0.run (precharged ledger, cap, STOP file, load-aware dispatch); workers = this module; ledger phase
    label ``phase`` (wm_verify; hold_pin_check for tools.wzprop.holdpin)."""
    ir, R = _harness()
    import subprocess
    if _read(Path(root) / 'PLAN.json').get('dry_run'):
        raise ValueError('dry-run plan: no physics')
    real, saved_phase = subprocess.Popen, ir.PHASE

    def popen(cmd, *a, **k2):
        if isinstance(cmd, list) and 'tools.coopmwm.regimeu0' in cmd:
            cmd = [c if c != 'tools.coopmwm.regimeu0' else 'tools.wzprop.wmverify' for c in cmd]
        return real(cmd, *a, **k2)
    R.subprocess.Popen = popen
    ir.PHASE = phase
    try:
        return R.run(root, **kw)
    finally:
        R.subprocess.Popen = real
        ir.PHASE = saved_phase


# ------------------------------------------------------------------------------------------ analysis (zero physics)
def _rec(R, ir, s, base):
    """regimeu0.analyze per-arm record (verbatim)."""
    rec = dict(status='ok', J=R.control_j(s), totals=s.get('totals'), safety=ir.safety(s),
               activations=len(s.get('activations') or ()), lane_held_steps=s.get('lane_held_steps'))
    return rec


WMSEARCH_SCHEMA = 'wmc.wmsearch_u0.v1.plan'      # WM-C S-WMSEARCH roots (arm ids wm_k/str_k/rnd_k, solo2 incumbent)
WMSEARCH_SCHEMAS = (WMSEARCH_SCHEMA, 'wmc.wmsearch_u0.v2.plan')   # v2: inc (wall_g_solo2) + optional banned solo2


def arm_alias(body):
    """Arm id -> ANALYSIS family name. wm_verify roots: identity. S-WMSEARCH roots (wmsearch_u0 PLAN with 'specs'):
    the spec name, the SOLO2-keyed arm -> 'wall_h_solo2' (StrategyCore SOLO2 on the same plant; the RuleHook /
    StrategyCore parity is what the wm_verify solo2_core arm checks in SUMO)."""
    specs = body.get('specs') if body.get('schema') in WMSEARCH_SCHEMAS else None
    if not specs:
        return lambda arm: arm
    solo, inc = SS.strategy_key(dict(SS.SOLO2)), SS.strategy_key(dict(SS.INCUMBENT))
    names = {}
    for arm, spec in specs.items():
        if spec is None:
            names[arm] = BASE_ARM
        elif SS.strategy_key(spec) == inc:
            names[arm] = INCUMBENT
        elif SS.strategy_key(spec) == solo:
            names[arm] = BANNED_ARM
        else:
            names[arm] = spec['name']
    if len(set(names.values())) != len(names):
        raise ValueError('S-WMSEARCH arms do not map to distinct family names')
    return lambda arm: names.get(arm, arm)


def root_kind(body):
    """'wm_verify' | 'wmsearch'; any other root (e.g. S-HOLDPIN, cancelled) is refused."""
    if body.get('schema') == SCHEMA + '.plan':
        return 'wm_verify'
    if body.get('schema') in WMSEARCH_SCHEMAS:
        return 'wmsearch'
    raise ValueError(f"root schema {body.get('schema')!r} is not a wm_verify / S-WMSEARCH plan (refused)")


def banned_arms(body):
    """Aliased arm names that are never FULLPOL-facing: the banned comparator and EVERY arm whose spec holds the
    banned firm 15 m/s hold (strategy_search.banned), whatever its name (review REV-WMV3 blocker 2)."""
    alias = arm_alias(body)
    out = set()
    if root_kind(body) == 'wmsearch':
        for arm, spec in (body.get('specs') or {}).items():
            if spec is not None and (SS.banned(spec) or alias(arm) == BANNED_ARM):
                out.add(alias(arm))
        return out
    for g in body['groups']:
        for j in g['jobs']:
            spec = j.get('strategy_spec')
            if j['arm'] == BANNED_ARM or (spec is not None and SS.banned(spec)):
                out.add(alias(j['arm']))
    return out


def wmsearch_families(body):
    """VERIFY families of an S-WMSEARCH root: executed by strategy_hook.make_hook (G5 lane path), NOT the LCEXEC
    policy executor (closing-lane lane targets were refused by the frozen env there)."""
    out = {}
    alias = arm_alias(body)
    for arm, spec in (body.get('specs') or {}).items():
        if spec is None or alias(arm) in (BASE_ARM, INCUMBENT, BANNED_ARM) or SS.banned(spec):
            continue                             # banned specs: comparator-only, never a family
        tpl, tsha = VH.family_template(spec)
        out[alias(arm)] = dict(spec=spec, strategy_key=SS.strategy_key(spec), executor='plant', cells={},
                               template=tpl, template_sha256=tsha, offer_condition=VH.offer_condition(spec),
                               lcexec_parity=None, source_arm=arm,
                               n2_parity=None if VH.has_lane_roles(spec) else N2.parity(spec, seconds=N2_FIXTURE_S),
                               verification_note='S-WMSEARCH: WM-C strategy_hook.make_hook on the u0 plant (hold law '
                                                 'through the plant writer, lane targets through the frozen G5 path: '
                                                 'closing-lane entries refused); not the LCEXEC policy executor')
        for g in body['groups']:
            if g['stage'] == 'paired':
                out[alias(arm)]['cells'][g['cell']] = 'wmsearch'
    return out


def superseded_groups(roots):
    """{(prior plan_sha256, seed)} of the never-charged seed groups that a later root of this set re-planned
    (PLAN 'supersedes', re-plan B2); empty for the 0eee867c roots."""
    out = set()
    for root in roots:
        for sup in _read(Path(root) / 'PLAN.json').get('supersedes') or ():
            out |= {(sup['plan_sha256'], g['seed']) for g in sup['groups']}
    return out


def collect(roots):
    """regimeu0.analyze's per-seed rows (verbatim records) + identity + parity/side records."""
    ir, R = _harness()
    roots = [Path(r) for r in ([roots] if isinstance(roots, (str, Path)) else roots)]
    rows, side, identity, parity, failed, arms = [], {}, None, [], [], []
    vehicles, vlogs, identities, banned = {}, [], [], set()   # vehicles[cell][arm][seed] = WF.per_vehicle dict
    superseded = superseded_groups(roots)
    for root in roots:
        body = _read(root / 'PLAN.json')
        root_kind(body)
        if body.get('pilot'):
            raise ValueError(f'{root} is an ungated PILOT root: never pooled with gated arms (use pilot_report)')
        alias = arm_alias(body)
        banned |= banned_arms(body)

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
                    identities.append(dict(identity, root=str(root)))
                else:
                    identities.append(dict(root=str(root), equal=False, missing=True))
                continue
            if (body.get('plan_sha256'), g['jobs'][0]['world']['seed']) in superseded:
                continue                         # never-charged group of a STOPPED root, re-planned by a later root
            sums = {alias(j['arm']): load(j['id']) for j in g['jobs']}
            for j in g['jobs']:
                if alias(j['arm']) not in arms and j['arm'] != PARITY_ARM:
                    arms.append(alias(j['arm']))
                p = root / 'jobs' / j['id'] / 'WMVERIFY.json'
                if p.exists():
                    side[(g['jobs'][0]['world']['seed'], alias(j['arm']))] = _read(p)
                w0 = g['jobs'][0]['world']
                summ = root / 'jobs' / j['id'] / 'SUMMARY.json.gz'
                if summ.exists() and j['arm'] != PARITY_ARM:
                    vpath = root / VEHICLE_LOG_DIR / j['id'] / WF.VEHICLE_LOG
                    rec_v = dict(seed=w0['seed'], cell=w0['cell_key'], arm=alias(j['arm']), present=vpath.exists())
                    if rec_v['present']:
                        pv, problems = WF.per_vehicle(vpath)
                        rec_v.update(vehicles=len(pv), problems=problems, ok=not problems and bool(pv))
                        if rec_v['ok']:
                            vehicles.setdefault(w0['cell_key'], {}).setdefault(alias(j['arm']), {})[w0['seed']] = pv
                    else:
                        rec_v['ok'] = False
                    vlogs.append(rec_v)
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
    return dict(roots=roots, rows=rows, side=side, identity=identity, identities=identities, parity=parity,
                failed=failed, arms=arms, vehicles=vehicles, vehicle_logs=vlogs, banned=sorted(banned))


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
    excluded = set(NOT_FAMILIES) | set(col['banned'])
    order = [BASE_ARM] + [a for a in col['arms'] if a != BASE_ARM and a not in excluded]
    rows = [dict(r, arms={a: x for a, x in r['arms'].items() if a not in excluded}) for r in col['rows']]
    rep = dict(schema=R.SCHEMA + '.analysis', roots=[str(r) for r in col['roots']],
               spent=sum(ir.spent(r) for r in col['roots']),
               plan_sha256=[_read(Path(r) / 'PLAN.json')['plan_sha256'] for r in col['roots']], cells={},
               seeds=rows)
    if col['identity'] is not None:
        rep['identity'] = col['identity']
    rep['cells'] = cells_from_rows(rows, order)
    return rep


def fairness(col, arms):
    """Per cell per arm: N6 amendment-13 co-primary groups (wmverify_fair.co_primary_groups, labels from u0 by
    vehicle id, depart delay included) vs u0 and vs the incumbent; ALLOWED = all five co-primary groups present
    and passing vs u0 (point estimate <= 10 s); unavailable fairness = not allowed."""
    out = {}
    for cell, reports in sorted(col['vehicles'].items()):
        worlds = sorted(reports.get(BASE_ARM, {}))
        cell_out = {}
        for arm in arms:
            if arm == BASE_ARM or arm not in reports:
                continue
            vs = {ref: WF.co_primary_groups(reports, worlds, arm, ref, reference=BASE_ARM)
                  for ref in (BASE_ARM, INCUMBENT) if ref != arm}
            f0 = vs[BASE_ARM]
            cell_out[arm] = dict(fairness=vs, allowed=bool(f0.get('all_pass')),
                                 groups_vs_u0={g: v.get('increase_mean_s') for g, v in (f0.get('groups') or {}).items()},
                                 vehicles_missing=f0.get('vehicles_missing'),
                                 label='banned comparator' if arm in col['banned'] else None)
        out[cell] = cell_out
    return out


def allowed_analysis(rep, fair):
    """ANALYSIS_ALLOWED.json: the same S-REGIMEU0 schema restricted, per cell, to u0 + the families whose owner
    fairness flag is ALLOWED (missing fairness = not allowed). The banned comparator is never a family."""
    out = dict(rep, cells={}, seeds=[], allowed_rule='every co-primary group (origin:L0/L1/L2, class:hdv, class:cav; '
               'labels from u0 by vehicle id; N6 amendment-13 convention incl. depart delay) mean per-vehicle delay '
               'increase vs u0 <= 10 s (point estimate), all five groups present (owner 2026-11-07, REGISTRATION 04:03)')
    keep = {}
    for cell, arms in rep['cells'].items():
        ok = {a for a, v in (fair.get(cell) or {}).items() if v.get('allowed') is True}
        keep[cell] = {BASE_ARM} | ok
        out['cells'][cell] = {a: v for a, v in arms.items() if a in keep[cell] or (
            'minus' in a and v.get('group') in ok)}
    out['seeds'] = [dict(r, arms={a: x for a, x in r['arms'].items() if a in keep.get(r['cell'], ())})
                    for r in rep['seeds']]
    return out


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
    """Per cell per strategy: paired arm - ref (the incumbent) with the regimeu0 bootstrap; per-seed fractions;
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


LANE_EXECUTORS = ('lcexec', PILOT_EXECUTOR)


def lane_receipts(side):
    """Per arm: LCEXEC receipts summed over seeds. executed_lane_changes = env status 'issued' only (changeLane to
    another lane sent to SUMO); 'held' (same-lane hold request) is reported apart and never counted as executed."""
    lane_arms = {}
    for (_seed, arm), sd in side.items():
        if sd.get('executor') not in LANE_EXECUTORS or not sd.get('receipts'):
            continue
        rc = sd['receipts']
        pr = rc.get('protocol') or {}
        a = lane_arms.setdefault(arm, dict(seeds=0, activations=0, sessions=0, refused=0, lc_ticks=0, status={},
                                           gate_withheld={}, executed_lane_changes=0, held_requests=0,
                                           completed_lane_changes=0))
        a['seeds'] += 1
        a['activations'] += rc.get('activations', 0)
        a['sessions'] += pr.get('sessions', 0)
        a['refused'] += pr.get('n_refusals', 0)
        a['lc_ticks'] += rc.get('lc_ticks', 0)
        for k, v in (rc.get('lc_vehicle_status') or {}).items():
            a['status'][k] = a['status'].get(k, 0) + v
        for k, v in (pr.get('gate_withheld') or {}).items():
            a['gate_withheld'][k] = a['gate_withheld'].get(k, 0) + v
        a['executed_lane_changes'] += int((rc.get('lc_vehicle_status') or {}).get('issued', 0))
        a['held_requests'] += int((rc.get('lc_vehicle_status') or {}).get('held', 0))
        a['completed_lane_changes'] += int(rc.get('lc_completed', 0) or 0)
        if 'ungated' in pr:
            a['would_withhold'] = a.get('would_withhold', 0) + int(pr['ungated'].get('would_withhold', 0))
    return lane_arms


def lane_warnings(lane_arms):
    """LCEXEC refusals; lane programs that activated but executed no lane change. Re-plan (d): only 'issued' (a lane
    change requested to SUMO) is an executed lane change; 'held' (a same-lane hold request) stays a hold (0eee867c
    counted issued + held, so lc01_g22 / g2_lc01 escaped this warning)."""
    warn = []
    for arm, a in lane_arms.items():
        if a['refused']:
            warn.append(dict(name='lcexec_refused_activations', arm=arm, **a))
        if a['activations'] and not a['executed_lane_changes']:
            warn.append(dict(name='lane_program_without_executed_lane_requests', arm=arm, **a))
    return warn


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
    ids = col.get('identities') or []
    add('identity_u0_989599', bool(ids) and all(x.get('equal') for x in ids), identities=ids,
        expected=R.IDENTITY['expected_team_delay_vs'],
        rule='mandatory: at least one identity episode and every analyzed identity episode exact')
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
    charged_by_plan = {}
    for r in roots:
        charged_by_plan.setdefault(_read(Path(r) / 'PLAN.json')['plan_sha256'], set()).update(
            c['id'] for c in ir.charges(r))
    sup_rows = []
    for r in roots:
        for s in _read(Path(r) / 'PLAN.json').get('supersedes') or ():
            for g in s['groups']:
                sup_rows.append(dict(by=str(r), root=s['root'], seed=g['seed'], prior_root_analyzed=s['plan_sha256']
                                     in charged_by_plan, charged=sorted(set(g['job_ids'])
                                                                        & charged_by_plan.get(s['plan_sha256'], set()))))
    if sup_rows:                                  # re-plan B2: a superseded group must stay uncharged forever
        add('superseded_groups_never_charged', not any(x['charged'] for x in sup_rows), groups=sup_rows)
    shas, refs = set(), set()
    for r in roots:
        body = _read(Path(r) / 'PLAN.json')
        u0 = body['inputs'].get('u0.pt') or {}
        shas.add(u0.get('sha256'))
        # wmsearch_u0.plan refuses a u0 differing from the S-REGIMEU0 base at plan time (no stored reference)
        refs.add(u0.get('regimeu0_sha256') if body.get('schema') not in WMSEARCH_SCHEMAS else u0.get('sha256'))
    add('inputs_u0_identical_and_equal_regimeu0', len(shas) == 1 and shas == refs and None not in shas,
        u0_sha256=sorted(x for x in shas if x))
    if col['failed']:
        warn.append(dict(name='failed_jobs', jobs=col['failed']))
    missing = [dict(x) for x in col.get('vehicle_logs') or () if not x.get('ok')]
    add('vehicle_log_present_every_episode', not missing and bool(col.get('vehicle_logs')), missing=missing[:50],
        n_missing=len(missing))
    lane_arms = lane_receipts(col['side'])
    warn += lane_warnings(lane_arms)
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


def results_rows(rows, excluded=()):
    """REGIMEU0_RESULTS.jsonl episode rows (J_300_1200_vs = control-window J) of FAMILY arms only (u0 included; the
    parity arm, the banned comparator and every banned-spec arm excluded)."""
    out = []
    for r in rows:
        for arm, x in sorted(r['arms'].items()):
            if arm in excluded:
                continue
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
    """ANALYSIS.json + ANALYSIS_ALLOWED.json (S-REGIMEU0 schema), VERIFY.json (contrasts vs the incumbent, fairness,
    integrity, safety, families),
    RESULTS.jsonl, CHECK.json into a NEW directory (never overwritten)."""
    roots = [Path(r) for r in ([roots] if isinstance(roots, (str, Path)) else roots)]
    col = collect(roots)
    rep = analysis(roots, col)
    chk = checks(roots, col)
    families = {}
    for r in roots:
        body = _read(r / 'PLAN.json')
        for name, f in (body.get('families') or wmsearch_families(body)).items():
            if name not in col['banned'] and not SS.banned(f['spec']):
                families.setdefault(name, f)
    order = [BASE_ARM] + [a for a in col['arms'] if a != BASE_ARM]
    excluded = set(NOT_FAMILIES) | set(col['banned'])
    fair = fairness(col, order)
    ver = dict(schema=SCHEMA + '.verify', stable_id=STABLE_ID, phase=PHASE, roots=[str(r) for r in roots],
               plan_sha256=rep['plan_sha256'], incumbent=INCUMBENT, banned_comparator=BANNED_ARM,
               convention='diff = arm - reference (same seed): J over [300, 1200) s veh-s, group mean per-vehicle '
                          'delay change s/veh; negative = better. ALLOWED = N6 amendment 13 (labels from u0 by vehicle '
                          'id; all five co-primary groups <= 10 s point; depart delay included)',
               banned_arms=col['banned'],
               families=families, vs_incumbent=contrasts(col['rows'], order), fairness=fair, integrity=chk,
               parity=col['parity'])
    for cell, arms in ver['vs_incumbent'].items():
        for a in set(arms) & excluded - {PARITY_ARM}:
            arms[a]['label'] = 'banned comparator (owner 2026-11-07); never an ANALYSIS family'
    summ = {}
    for cell, arms in ver['vs_incumbent'].items():
        fc = fair.get(cell) or {}
        allowed = sorted(a for a in arms if a not in excluded and (fc.get(a) or {}).get('allowed') is True)
        beat = sorted(a for a in allowed if arms[a].get('beats_ref_ci'))
        best = min((a for a in allowed if arms[a].get('mean_diff') is not None), key=lambda a: arms[a]['mean_diff'],
                   default=None)
        summ[cell] = dict(n_strategies=len([a for a in arms if a not in excluded]), allowed=allowed,
                          not_allowed=sorted(a for a in arms if a not in excluded and a not in allowed),
                          incumbent_allowed=(fc.get(INCUMBENT) or {}).get('allowed'),
                          beat_incumbent_ci_allowed=beat, best_allowed=best,
                          best_allowed_vs_incumbent=None if best is None else {k: arms[best].get(k) for k in
                                                                               ('n', 'mean_diff', 'ci95', 'rel_diff',
                                                                                'better_share')})
    ver['summary'] = summ
    ver['analysis_written'] = bool(chk['ok'])
    if not chk['ok']:
        # FAIL CLOSED (review REV-WMV3 blocker 3): no FULLPOL-facing file when an integrity check fails
        ver['summary'] = dict(blocked='integrity failed: no ANALYSIS / ANALYSIS_ALLOWED / RESULTS written',
                              failed_checks=[c['name'] for c in chk['checks'] if not c['ok']])
        if out_dir is not None:
            out = guard_path(out_dir)
            out.mkdir(parents=True, exist_ok=False)
            _write_new(out / 'VERIFY.json', ver)
            _write_new(out / 'CHECK.json', chk)
        return None, ver
    if out_dir is not None:
        out = guard_path(out_dir)
        out.mkdir(parents=True, exist_ok=False)
        text = json.dumps(rep, indent=1, sort_keys=True, default=str) + '\n'
        (out / 'ANALYSIS.json').open('xb').write(text.encode('utf-8'))       # exact bytes (sha below)
        ver['analysis_sha256'] = hashlib.sha256(text.encode('utf-8')).hexdigest()
        allowed_text = json.dumps(allowed_analysis(rep, fair), indent=1, sort_keys=True, default=str) + '\n'
        (out / 'ANALYSIS_ALLOWED.json').open('xb').write(allowed_text.encode('utf-8'))
        ver['analysis_allowed_sha256'] = hashlib.sha256(allowed_text.encode('utf-8')).hexdigest()
        _write_new(out / 'VERIFY.json', ver)
        _write_new(out / 'CHECK.json', chk)
        with (out / 'RESULTS.jsonl').open('x', encoding='utf-8') as f:
            for row in results_rows(col['rows'], excluded):
                f.write(json.dumps(row, sort_keys=True, default=str) + '\n')
    return rep, ver


# ------------------------------------------------------------------------------------------ ungated pilot (separate)
def pilot_report(pilot_root, prior_roots):
    """PILOT_UNGATED.json body: per pilot episode the paired gated run of the same arm and the u0 / incumbent runs of
    the same seed (prior roots' charged jobs, same world), diffs (arm - reference; negative = better), safety, lane
    receipts (executed = 'issued' only; would_withhold = gap-only passes), fairness vs u0 (descriptive). Its own
    integrity (identity, ledger, inputs, pairing, CRN, vehicle logs). NEVER pooled: no ANALYSIS / ALLOWED / bank."""
    ir, R = _harness()
    proot = Path(pilot_root)
    body = _read(proot / 'PLAN.json')
    if (body.get('pilot') or {}).get('kind') != PILOT:
        raise ValueError(f'{proot} is not an ungated pilot root')
    pj = _prior_jobs(prior_roots)

    def load(root, ident):
        p = Path(root) / 'jobs' / ident / 'SUMMARY.json.gz'
        return ir._read(p) if p.exists() else None

    def side(root, ident):
        p = Path(root) / 'jobs' / ident / 'WMVERIFY.json'
        return _read(p) if p.exists() else None

    def vlog(root, ident):
        p = Path(root) / VEHICLE_LOG_DIR / ident / WF.VEHICLE_LOG
        if not p.exists():
            return None
        pv, problems = WF.per_vehicle(p)
        return pv if pv and not problems else None

    res, episodes, ids = [], [], []
    reports, sides = {}, {}
    for g in body['groups']:
        if g['stage'] == 'identity':
            s = load(proot, g['jobs'][0]['id'])
            tot = None if s is None else (s.get('totals') or {}).get('team_delay_vs')
            ids.append(dict(root=str(proot), team_delay_vs=tot, expected=R.IDENTITY['expected_team_delay_vs'],
                            equal=tot is not None and abs(tot - R.IDENTITY['expected_team_delay_vs']) < 1e-6))
            continue
        for j in g['jobs']:
            w, p = j['world'], j['pilot']
            refs = {}
            for key, name in (('gated_job', 'gated'), ('u0_job', BASE_ARM), ('incumbent_job', INCUMBENT)):
                r_root, r_job = pj.get(p[key], (None, None))
                ok_ref = r_job is not None and r_job['world'] == w
                refs[name] = (r_root, p[key], load(r_root, p[key]) if ok_ref else None, ok_ref)
            s = load(proot, j['id'])
            ep = dict(seed=w['seed'], cell=w['cell_key'], arm=j['arm'], gated_arm=p['gated_arm'], job=j['id'],
                      references={k: dict(root=None if v[0] is None else str(v[0]), id=v[1], same_world=v[3],
                                          present=v[2] is not None) for k, v in refs.items()})
            if s is None or s.get('error'):
                ep['status'] = 'not_run' if s is None else 'failed'
                episodes.append(ep)
                continue
            ep.update(status='ok', J=R.control_j(s), activations=len(s.get('activations') or ()), safety=ir.safety(s))
            for k, (_r, _i, ref, _ok) in refs.items():
                if ref is not None and not ref.get('error'):
                    ep[f'J_{k}'] = R.control_j(ref)
                    ep[f'diff_vs_{k}'] = ep['J'] - ep[f'J_{k}']
                    ep[f'safety_{k}'] = ir.safety(ref)
                    ep[f'first_divergence_vs_{k}_s'] = ir.first_divergence(ref, s)
            first_act = min((a['t'] for a in s.get('activations') or ()), default=None)
            fd = ep.get(f'first_divergence_vs_{BASE_ARM}_s')
            ep['first_activation_s'] = first_act
            ep['crn_ok'] = BASE_ARM in {k for k, v in refs.items() if v[2] is not None} and (
                fd is None or (first_act is not None and fd >= first_act))
            sd_u, sd_g = side(proot, j['id']), side(refs['gated'][0], refs['gated'][1]) if refs['gated'][0] else None
            sides[(w['seed'], j['arm'])] = sd_u or {}
            if sd_g:
                sides[(w['seed'], p['gated_arm'])] = sd_g
            for arm, (r_root, ident) in ((j['arm'], (proot, j['id'])), (p['gated_arm'], refs['gated'][:2]),
                                         (BASE_ARM, refs[BASE_ARM][:2]), (INCUMBENT, refs[INCUMBENT][:2])):
                if r_root is not None:
                    pv = vlog(r_root, ident)
                    if pv is not None:
                        reports.setdefault(w['cell_key'], {}).setdefault(arm, {})[w['seed']] = pv
            ep['vehicle_log_ok'] = (w['seed'] in reports.get(w['cell_key'], {}).get(j['arm'], {}))
            episodes.append(ep)
    lanes = lane_receipts(sides)
    fair = {}
    for cell, rep_c in reports.items():
        worlds = sorted(rep_c.get(BASE_ARM, {}))
        for arm in sorted(a for a in rep_c if a.endswith(PILOT_SUFFIX)):
            gated = arm[:-len(PILOT_SUFFIX)]
            fair.setdefault(cell, {})[arm] = dict(
                vs_u0=WF.co_primary_groups(rep_c, worlds, arm, BASE_ARM, reference=BASE_ARM),
                gated_vs_u0=WF.co_primary_groups(rep_c, worlds, gated, BASE_ARM, reference=BASE_ARM)
                if gated in rep_c else None)
    summary = {}
    for arm in body['pilot']['arms']:
        eps = [e for e in episodes if e['gated_arm'] == arm and e['status'] == 'ok']
        summary[arm + PILOT_SUFFIX] = dict(n=len(eps), **{
            f'mean_diff_vs_{k}': _mean([e[f'diff_vs_{k}'] for e in eps if f'diff_vs_{k}' in e])
            for k in ('gated', BASE_ARM, INCUMBENT)},
            per_seed={str(e['seed']): {k: e.get(f'diff_vs_{k}') for k in ('gated', BASE_ARM, INCUMBENT)} for e in eps},
            collisions=sum(float((e['safety'] or {}).get('collisions') or 0.) for e in eps),
            teleports=sum(float((e['safety'] or {}).get('teleports') or 0.) for e in eps),
            lane=lanes.get(arm + PILOT_SUFFIX), lane_gated=lanes.get(arm))
    led = dict(root=str(proot), cap=_read(proot / 'REGISTRATION.json')['cap_sim_s'], spent=ir.spent(proot),
               uncharged_dirs=sorted({p.name for p in proot.glob('jobs/*') if p.is_dir() and p.name != '_logs'}
                                     - {c['id'] for c in ir.charges(proot)}))
    u0s = {(body['inputs'].get('u0.pt') or {}).get('sha256')} | {
        ((_read(Path(r) / 'PLAN.json')['inputs'].get('u0.pt')) or {}).get('sha256') for r in prior_roots}

    def add(name, ok, **detail):
        res.append(dict(name=name, ok=bool(ok), **detail))
    add('identity_u0_989599', bool(ids) and all(x['equal'] for x in ids), identities=ids)
    add('pairing_references_present_same_world',
        all(all(v['present'] and v['same_world'] for v in e['references'].values()) for e in episodes),
        missing=[dict(job=e['job'], refs=e['references']) for e in episodes
                 if not all(v['present'] and v['same_world'] for v in e['references'].values())])
    add('pilot_episodes_ok', all(e['status'] == 'ok' for e in episodes),
        not_ok=[dict(job=e['job'], status=e['status']) for e in episodes if e['status'] != 'ok'])
    add('crn_no_divergence_before_first_activation', all(e.get('crn_ok') for e in episodes if e['status'] == 'ok'),
        violations=[e['job'] for e in episodes if e['status'] == 'ok' and not e.get('crn_ok')])
    add('ledger_within_cap_no_uncharged_jobs', led['spent'] <= led['cap'] and not led['uncharged_dirs'], **led)
    add('inputs_u0_identical_to_prior_roots', len(u0s) == 1 and None not in u0s, u0_sha256=sorted(x for x in u0s if x))
    add('vehicle_log_present_every_pilot_episode', all(e.get('vehicle_log_ok') for e in episodes if e['status'] == 'ok'))
    return dict(schema=SCHEMA + '.pilot_ungated', stable_id=STABLE_ID, phase=PHASE, label=PILOT_LABEL,
                pilot_root=str(proot), plan_sha256=body['plan_sha256'], prior_roots=[str(r) for r in prior_roots],
                convention='diff = J(ungated arm) - J(reference) on the same seed (control window [300, 1200) s, '
                           'veh-s); negative = the ungated arm is better; references = prior roots charged jobs',
                integrity=dict(ok=all(x['ok'] for x in res), checks=res), summary=summary, episodes=episodes,
                lane_receipts=lanes, fairness_vs_u0=fair, pooled_with_gated=False)


def write_pilot(out_dir, pilot_root, prior_roots):
    """PILOT_UNGATED.json (exact bytes) into ``out_dir`` (created if needed; the file is never overwritten)."""
    rep = pilot_report(pilot_root, prior_roots)
    out = guard_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(rep, indent=1, sort_keys=True, default=str) + '\n'
    (out / 'PILOT_UNGATED.json').open('xb').write(text.encode('utf-8'))
    return rep, hashlib.sha256(text.encode('utf-8')).hexdigest()


# ------------------------------------------------------------------------------------------ registration text
def budget(cells, n_candidates_by_cell, parity_seeds=2, identity=True, banned_seeds=0):
    total = EPISODE_S if identity else 0
    for cell, _d, _p, seeds in cells:
        arms = 2 + n_candidates_by_cell[cell]
        total += len(seeds) * arms * EPISODE_S + min(parity_seeds, len(seeds)) * EPISODE_S
        total += min(banned_seeds, len(seeds)) * EPISODE_S
    return total


def registration_text(*, cells, specs_by_cell, cap, parity_seeds=2, lane_executor=DEFAULT_LANE_EXECUTOR,
                      hold_executor=DEFAULT_HOLD_EXECUTOR, arms_sha256=None, regimeu0_u0_sha256=None, smoke_seeds=1,
                      banned_seeds=0, pairs_by_cell=None):
    _ir, R = _harness()
    gr = build_groups(cells, specs_by_cell, R.world_of, parity_seeds=parity_seeds, identity=True,
                      lane_executor=lane_executor, hold_executor=hold_executor, banned_seeds=banned_seeds,
                      pairs_by_cell=pairs_by_cell)
    planned = sum(j['sim_s'] for g in gr for j in g['jobs'])
    shas = module_shas()
    lines = [
        REG_HEADER,
        '- Append BEFORE any physics; `plan` refuses unless this exact header and the jobs digest, cap / planned '
        'budget, ARMS.json sha256 and module sha256 below match the run.',
        f'- Plan jobs sha256: {jobs_digest(gr)}',
        '- Question: do the WORLD-MODEL-PROPOSED strategies (WM-C ARMS.json v3: delay / bounded / fairer archives, '
        'structured, random, reference) beat the gentle incumbent wall_g_solo2 and MAPPO u0 per demand cell on fresh '
        'TRAIN-type seeds, when executed by the POLICY executor, and are they ALLOWED under the owner fairness bound '
        '(every origin-lane / HDV / CAV group mean per-vehicle delay change vs u0 <= 10 s, REGISTRATION 04:03)? '
        'Output = verified EV rows for FULLPOL (ANALYSIS.json and the fairness-filtered ANALYSIS_ALLOWED.json, '
        'S-REGIMEU0 schema) + per-family templates, offer conditions and per-group fairness (VERIFY.json).',
        '- Closest prior: S-REGIMEU0 (ANALYSIS c0b578f8): solo2 (wall_h_solo2) -24% at D3200, -14% at D2800_p60, '
        '+31-39% at D2400; walls dominated by solo2. S-WMSEARCH amendment 1 (WM/structured/random vs solo2 on '
        'REGIME seeds 989770-989785). Exact difference: fresh seeds 989950-989999, WM-proposed specs incl. lane '
        'programs executed by the LCEXEC policy executor, incumbent in every seed group, executor-parity control.',
        f'- Base: MAPPO u0 d9042cc5 = the S-REGIMEU0 inputs u0.pt (sha256 {regimeu0_u0_sha256 or "<from regimeu0 PLAN>"}'
        '), coopmwm clean4 plant, wz3to2, native warmup 0-300 s, episodes 1,200 s, J = team delay over [300, 1200) s.',
        '- Seeds: ' + ', '.join(f'{lo}-{hi}' for lo, hi in SEED_RANGES) + ' only (TRAIN-type, never evaluation; MAIN '
        'allocations 2026-11-06 (989950-989999; LSSEARCH has 989900-989949) and 2026-11-08 (991140-991199; MULTICRIT '
        'used 991100-991132) and 2026-11-08 r4 (991200-991299) and second look (991300-991347); verified unused in REGISTRATION.md and every PLAN). Registered in '
        'map_scenarios.EXTRA_TRAIN_RANGES on c4/wmverify only.',
        '- Cells / seeds: ' + '; '.join(f'{c} {s[0]}-{s[-1]} ({len(s)})' for c, _d, _p, s in cells) + '.',
        f'- Arms per seed group (paired CRN): u0; {INCUMBENT} (gentle incumbent 22 m/s / 120 s / dmax 1.0, '
        f'regimeu0.RuleHook verbatim); {PARITY_ARM} (the same rule via strategy_hook.StrategyCore on the plant path, '
        f'executor parity, first {parity_seeds} seeds per cell, VERIFY only); '
        + (f'{BANNED_ARM} (BANNED firm 15 m/s solo2, labelled comparator on the first {banned_seeds} seeds per cell, '
           'never a family); ' if banned_seeds else '') + 'candidates [source, strategy key, executor]:']
    for cell, lst in specs_by_cell.items():
        lines.append(f'  - {cell}: ' + ', '.join(
            f'{s["name"]} [{src}, {SS.strategy_key(s)}, '
            f'{VH.resolve_executor(s, lane_executor=lane_executor, hold_executor=hold_executor)}]' for src, s in lst))
    for cell, lst in (pairs_by_cell or {}).items():
        from . import wmverify_oracle as WO
        lines.append(f'  - {cell} PAIR arms (executor oracle_bank = reviewed ORACLE hook c4/oracle-arm 8bc436df, '
                     f'oracle_digest {WO.ORACLE_DIGEST[:16]}; unconditional mini-bank rows in priority order, '
                     'per-family sessions on different CAV groups): ' + ', '.join(
                         f'{name} [' + ' + '.join(f"{r['family']} {r['template_sha256']}" for r in bank['rows'])
                         + f'; bank {WO.bank_sha(bank)[:16]}]' for name, bank in lst))
    lines += [
        f'- ARMS.json sha256: {arms_sha256 or "<none>"}.',
        '- Executors: trigger, member selection and timing = WM-C strategy_hook.StrategyCore (same select_members, '
        'roles, gate, cooldown as the WM search). Hold-only strategies (executor plant): hold law clip(v_w - v, -dmax, '
        '1.0) as the absolute pre-safety request through the plant single writer (MAPPO safety downstream), own-lane '
        'G5 target = the S-REGIMEU0 solo2 path. Strategies with lane changes (executor lcexec): the LCEXEC POLICY '
        'executor (c4/lcexec 0fe9574e, COOPMWM_LC_RUNTIME=1): member_specs -> lc_roles.member_program -> '
        'template_from_program -> Protocol.start / receive / commands (own 100 m sensing, own-sensing lane gate) -> '
        'env.lc_tick, exactly the S-LCEXEC-SMOKE production path; program members bypass the MAPPO pre-safety filter '
        'as in S-LCEXEC-SMOKE (speedMode 27, direct_target bounds, env edge-end guard stay). Verified = executable '
        'by the policy executor. Hold families carry n2_parity (FULLPOL N2 hold executor e36ece92 + hold_lane_pin '
        '5cbd616e vs the plant path, through the real env writer on fixture states: identical TraCI writes with '
        'hold_lane_pin ON; without it only the plant-path laneChangeMode 513 pin differs; --hold-executor n2 runs '
        'holds without the pin). CAV rows only; HDVs never commanded. Parameter ranges: strategy_search.validate '
        '(NEWMAP2 R grid, WMDATA offsets, zones, gates) + spread {200, 300} m + one slot + cooldown <= 240 s; '
        'anything else is rejected.',
        f'- Code: c4/wmverify; module sha256 {json.dumps(shas, sort_keys=True)}.',
        f'- Budget (phase {PHASE}): cap {int(cap):,} sim-s; planned {planned:,} sim-s '
        '(identity 1,200 per root + seeds x arms x 1,200 + parity / comparator arms).',
        f'- Stage A (smoke): identity + the first {smoke_seeds} seed(s) of every cell, all arms -> '
        '`python -m tools.wzprop.wmverify check` must pass (identity 53,269.32889342696 veh-s, CRN, executor '
        'parity, ledger, inputs; warnings reviewed: LCEXEC refusals, lane programs without executed lane requests, '
        'never-activated strategies) before stage B (remaining seeds, --prior-root A; stage B repeats the identity '
        'episode). analyze writes no ANALYSIS file when any integrity check fails (fail closed).',
        '- Analysis (fixed before results): regimeu0.analyze per cell (paired diff vs u0, bootstrap 10,000 RNG '
        '20261106, 2.5/97.5%), strategy - wall_g_solo2 paired contrasts (same bootstrap), per-seed better/worse '
        'shares, beats-u0-and-incumbent share, safety (collisions, teleports, hard braking, TTC<3 s; deltas vs u0 and '
        'vs the incumbent); per-group fairness = N6 amendment 13 exactly (c4/nm-finaleval 4397d777: eval-only '
        'VehicleLog on every arm incl. u0; per-vehicle delay = window speed deficit + depart delay if inserted after '
        '300 s; labels from u0 by vehicle id: origin = first logged row at t >= 300 if upstream of the drop, else '
        'origin:undefined (reported only); class hdv/cav; missing vehicles counted both ways; ALLOWED = all five '
        'co-primary groups present and mean increase vs u0 <= 10 s (point; 95% CI reported, whole-world bootstrap '
        '10,000 seed 20261104); own-arm labels = sensitivity). Banned 15 m/s specs are never FULLPOL-facing. diff = '
        'arm - reference; negative = better. No thresholds tuned on these seeds.',
        '- Stop rules: STOP file; cap; failed integrity check -> stop and report; charged jobs are never re-run.',
        '- Outputs: ANALYSIS_ALLOWED.json (FULLPOL reads this file only; S-REGIMEU0 schema), ANALYSIS.json, '
        'VERIFY.json (families: spec, executor, LCEXEC template(s), offer condition, lcexec_parity / n2_parity; '
        'contrasts; fairness; integrity), RESULTS.jsonl (families only), CHECK.json, VEHICLES.jsonl.gz per episode.',
    ]
    return '\n'.join(lines) + '\n', planned


REPLAN_NOTE = ('- Re-plan (owner 2026-11-07, PAUSE_WMVERIFY_20261107): stage A DONE (plan 170c97c3, 33,600 sim-s, check '
               'passed); stage B (plan 1d87d4b9) STOPPED with 96,000 sim-s charged (groups 989951-989953; the 10 '
               'interrupted 989953 jobs stay charged and are never re-run). Finding: the own-sensing lane gate '
               '(lc_families.lane_gate -> LaneRules._target_gap_ok) withheld ~99% of lane-change requests in heavy '
               'traffic (22 of 2,781 passed in the A/B tapes); withheld requests are dropped, so SUMO never tried them.')


def _phase_line(prior_committed, planned, cap):
    if prior_committed is None:
        return '- Phase total: prior roots committed <from plan> + this root.'
    total = int(prior_committed) + int(planned)
    if total > int(cap):
        raise ValueError(f'phase total {total} > cap {cap}')
    return (f'- Phase total (phase {PHASE}): prior roots committed {int(prior_committed):,} sim-s + this root '
            f'{int(planned):,} sim-s = {total:,} sim-s <= cap {int(cap):,} sim-s (plan re-checks against the prior '
            'roots: STOPPED roots count their charges, others their plan).')


def b2_registration_text(*, cells, specs_by_cell, cap, arms_sha256=None, regimeu0_u0_sha256=None,
                         prior_committed=None, executors=('plant',)):
    """Registration block of the hold-arm continuation B2 (plan --only-executor plant --supersede-uncharged
    --parity-seeds 0 --prior-root A --prior-root B): the 0eee867c block text with the B2 jobs, the excluded lcexec
    arms and the re-plan rules. ``specs_by_cell`` = the full candidate list (the excluded arms are named)."""
    kept = select_executors(specs_by_cell, executors)
    text, planned = registration_text(cells=cells, specs_by_cell=kept, cap=cap, parity_seeds=0,
                                      arms_sha256=arms_sha256, regimeu0_u0_sha256=regimeu0_u0_sha256)
    dropped = {c: [s['name'] for _src, s in lst if s['name'] not in {x['name'] for _y, x in kept[c]}]
               for c, lst in specs_by_cell.items()}
    lines = text.rstrip('\n').split('\n')
    out = []
    for line in lines:
        if line.startswith('- Arms per seed group'):
            line = line.replace(f'{PARITY_ARM} (the same rule via strategy_hook.StrategyCore on the plant path, '
                                'executor parity, first 0 seeds per cell, VERIFY only); ',
                                f'({PARITY_ARM} executor parity already run in stages A/B: not repeated); ')
        if line.startswith('- Code: c4/wmverify;'):
            line = line.replace('- Code: c4/wmverify;', '- Code: c4/wmverify-replan (0eee867c + re-plan; the module '
                                'sha256 pin it);')
        if line.startswith('- Stage A (smoke)'):
            line = ('- Stage B2 (hold-arm continuation): identity + the remaining seeds, hold (plant) arms + u0 + '
                    'the incumbent only; plan --prior-root A --prior-root B --only-executor plant --supersede-uncharged '
                    '--parity-seeds 0 (B must keep its STOP file; its never-charged groups for these seeds are recorded '
                    'in PLAN supersedes, skipped by collect, and check fails if any of them is ever charged). '
                    '`check --root A B B2` must pass before analyze; analyze --root A B B2 (fail closed). The lcexec '
                    'arms keep only their A/B seeds (989950-989953) and are reported as such.')
        out.append(line)
        if line.startswith('- Plan jobs sha256:'):
            out.append(REPLAN_NOTE)
            out.append('- B2 excludes the lane-program (lcexec) arms on these seeds: ' + '; '.join(
                f'{c}: {", ".join(v)}' for c, v in dropped.items()) + '. Hold (plant) arms: ' + '; '.join(
                f'{c}: {len(v)}' for c, v in kept.items()) + '.')
            out.append(_phase_line(prior_committed, planned, cap))
        if line.startswith('- Analysis (fixed before results)'):
            out.append('- Analysis fix (d) (owner 2026-11-07, before results): the check counts only env status '
                       "'issued' (a lane change requested to SUMO) as an executed lane change; 'held' (same-lane hold "
                       'request) stays a hold. No other analysis rule changes.')
    return '\n'.join(out) + '\n', planned


def pilot_registration_text(*, cells, pilot_by_cell, cap, arms_sha256=None, regimeu0_u0_sha256=None,
                            prior_committed=None, selection_note=''):
    """Registration block of the ungated pilot root (plan --pilot-arm NAME ... --prior-root A --prior-root B
    [--prior-root B2])."""
    _ir, R = _harness()
    gr = build_pilot_groups(cells, pilot_by_cell, R.world_of)
    planned = sum(j['sim_s'] for g in gr for j in g['jobs'])
    shas = module_shas()
    lines = [
        REG_HEADER,
        '- Append BEFORE any physics; `plan` refuses unless this exact header and the jobs digest, cap / planned '
        'budget, ARMS.json sha256 and module sha256 below match the run.',
        f'- Plan jobs sha256: {jobs_digest(gr)}',
        '- Question (UNGATED PILOT, descriptive): when program lane requests that fail ONLY the own-sensing gap test '
        'are passed to SUMO (members laneChangeMode 513: SUMO\'s own lane-change safety decides), do the lane '
        'programs execute lane changes, and how do J and safety move vs the SAME arm gated (stage B, same seed), u0 '
        'and the incumbent? Output = PILOT_UNGATED.json, a separate labelled section; never pooled with the gated '
        'arms, never an ANALYSIS / ANALYSIS_ALLOWED / bank family, never FULLPOL-facing.',
        REPLAN_NOTE,
        f'- Base: MAPPO u0 d9042cc5 = the S-REGIMEU0 inputs u0.pt (sha256 {regimeu0_u0_sha256 or "<from regimeu0 PLAN>"}'
        '), coopmwm clean4 plant, wz3to2, native warmup 0-300 s, episodes 1,200 s, J = team delay over [300, 1200) s.',
        '- Cells / seeds: ' + '; '.join(f'{c} {s[0]}-{s[-1]} ({len(s)})' for c, _d, _p, s in cells) + ' (seeds already '
        'run in stage B: every pilot job pairs with the charged, DONE stage-B jobs of the same world = gated arm, u0, '
        'incumbent; those references are never re-run; plan refuses a missing / not DONE / different-world reference '
        'and any pilot job id equal to a prior job id).',
        '- Pilot arms [gated arm, strategy key, executor]: ' + '; '.join(
            f'{c}: ' + ', '.join(f'{s["name"]}{PILOT_SUFFIX} [{s["name"]}, {SS.strategy_key(s)}, {PILOT_EXECUTOR}]'
                                 for _src, s in lst) for c, lst in pilot_by_cell.items()) + '.',
        f'- Pilot arm choice (fixed before any pilot outcome): {selection_note or "<see report>"}',
        f'- ARMS.json sha256: {arms_sha256 or "<none>"}.',
        f'- Executor {PILOT_EXECUTOR} (WMVERIFY-only, opt-in per job, never a default): the LCEXEC policy executor '
        'of stages A/B unchanged except that a lane request refused by lc_families.lane_gate ONLY because of the '
        'target-lane gap test is passed on (gap-only = the same gate passes on the ego row alone); adjacency, '
        'target-lane existence (closing lane only on the approach), the LaneRules edge guard, sensing validity and the '
        'env edge-end guard still refuse. Each passed request is logged with the old verdict would_withhold (t, vid, '
        'source, target). With the option off every executor decision is identical to 0eee867c (CPU replay of the '
        'stage A/B tapes).',
        f'- Code: c4/wmverify-replan (0eee867c + re-plan; the module sha256 pin it); module sha256 '
        f'{json.dumps(shas, sort_keys=True)}.',
        f'- Budget (phase {PHASE}): cap {int(cap):,} sim-s; planned {planned:,} sim-s '
        '(identity 1,200 per root + seeds x pilot arms x 1,200).',
        _phase_line(prior_committed, planned, cap),
        '- Analysis (fixed before results): `python -m tools.wzprop.wmverify pilot --pilot-root P --prior-root A B '
        '[B2] --out DIR` -> PILOT_UNGATED.json: per episode J and diff vs the gated arm, u0 and the incumbent (same '
        'seed; negative = better), safety (collisions, teleports, hard braking, TTC<3 s) of the ungated and the '
        'references, lane receipts (executed lane changes = status issued only; held apart; would_withhold count), '
        'per-group fairness vs u0 (descriptive); integrity (identity exact, pairing, CRN, ledger, u0 input, vehicle '
        'logs). n = 2 seeds per arm: no bootstrap, no thresholds, no verdict on the arms.',
        '- Stop rules: STOP file; cap; any collision or teleport in a pilot episode -> report before anything else; '
        'failed integrity -> stop and report; charged jobs are never re-run.',
        '- Outputs: PILOT_UNGATED.json (separate labelled section), VEHICLES.jsonl.gz per episode.',
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
           hold_executor=DEFAULT_HOLD_EXECUTOR, fake_seconds=900, spec_paths=(), banned_seeds=0):
    """Dry plan (no inputs, no physics) + every arm's hook on the zero-physics fake plant for the first seed of each
    cell; the incumbent (regimeu0.RuleHook) and solo2_core must issue identical commands every tick."""
    from . import wmverify_fake as F
    ir, R = _harness()
    bind_map_for_dryrun()
    body = plan(out_dir, regimeu0_root=None, arms_path=arms_path, cells=cells, cap=10 ** 9, groups=groups, k=k,
                parity_seeds=parity_seeds, lane_executor=lane_executor, hold_executor=hold_executor,
                spec_paths=spec_paths, dry_run=True, banned_seeds=banned_seeds)
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
            a, b = traces[INCUMBENT], traces[PARITY_ARM]   # wall_g_solo2: RuleHook vs StrategyCore(SS.INCUMBENT)
            diff = [x['t'] for x, y in zip(a, b) if x['desired'] != y['desired'] or x['g5'] != y['g5']]
            par = dict(identical=not diff and len(a) == len(b), differing_ticks=diff[:10],
                       incumbent_command_ticks=arms[INCUMBENT]['command_ticks'])
        report['cells'][cell] = dict(seed=seeds[0], arms=arms, incumbent_parity=par)
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
        a.add_argument('--k-groups', help='per-group k, e.g. wm=4,wm_fairer=4,structured=8,random=8 (overrides --k)')
        a.add_argument('--parity-seeds', type=int, default=2)
        a.add_argument('--banned-seeds', type=int, default=0,
                       help='seeds per cell with the BANNED solo2 comparator arm (labelled; never a family)')
        _executor_args(a)
        if name in ('registration', 'plan'):
            a.add_argument('--only-executor', action='append', default=None,
                           help='re-plan B2: keep only candidates with this resolved executor (e.g. plant)')
            a.add_argument('--supersede-uncharged', action='store_true',
                           help='re-plan B2: seeds of a STOPPED prior root whose groups were never charged')
            a.add_argument('--pilot-arm', action='append', default=None,
                           help='ungated pilot: candidate name run with executor lcexec_ungated (pilot root only)')
        if name == 'registration':
            a.add_argument('--cap', type=int, required=True)
            a.add_argument('--regimeu0-root')
            a.add_argument('--prior-committed', type=int, help='re-plan: phase sim-s committed by the prior roots')
            a.add_argument('--selection-note', default='', help='pilot: the arm choice justification (text)')
        if name == 'dryrun':
            a.add_argument('--out', required=True)
            a.add_argument('--fake-seconds', type=int, default=900)
        if name == 'plan':
            a.add_argument('--root', required=True)
            a.add_argument('--regimeu0-root', required=True)
            a.add_argument('--cap', type=int, required=True)
            a.add_argument('--registration', required=True, help='the run REGISTRATION.md with the S-WMVERIFY block')
            a.add_argument('--prior-root', action='append', default=[])
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
    a = sub.add_parser('pilot', help='ungated pilot: PILOT_UNGATED.json (separate section, never pooled)')
    a.add_argument('--pilot-root', required=True)
    a.add_argument('--prior-root', required=True, nargs='+')
    a.add_argument('--out', required=True)
    x = ap.parse_args(argv)
    if x.cmd == 'validate':
        doc = json.loads(guard_path(x.arms).read_text()) if x.arms else {}
        specs = [json.loads(guard_path(p).read_text()) for p in x.spec]
        out = {}
        for cell, d in doc.items():
            for grp in GROUPS + OPTIONAL_GROUPS + SINGLE_GROUPS:
                items = d.get(grp)
                items = list(items.values()) if isinstance(items, dict) and grp == 'best' else items
                for s in (items if isinstance(items, list) else [items] if items else []):
                    if not s:
                        continue
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
        if x.k_groups:
            x.k = {g: int(n) for g, n in (kv.split('=') for kv in x.k_groups.split(','))}
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
        arms_sha = _sha_file(x.arms) if x.arms else None
        if x.pilot_arm:
            text, planned = pilot_registration_text(cells=cells, pilot_by_cell=pilot_specs(specs, x.pilot_arm),
                                                    cap=x.cap, arms_sha256=arms_sha, regimeu0_u0_sha256=u0,
                                                    prior_committed=x.prior_committed,
                                                    selection_note=x.selection_note)
        elif x.only_executor or x.supersede_uncharged:
            if x.parity_seeds != 0 or x.banned_seeds or not x.only_executor:
                raise SystemExit('B2 registration: --only-executor plant, --parity-seeds 0, no banned seeds')
            text, planned = b2_registration_text(cells=cells, specs_by_cell=specs, cap=x.cap, arms_sha256=arms_sha,
                                                 regimeu0_u0_sha256=u0, prior_committed=x.prior_committed,
                                                 executors=tuple(x.only_executor))
        else:
            text, planned = registration_text(cells=cells, specs_by_cell=specs, cap=x.cap,
                                              parity_seeds=x.parity_seeds, lane_executor=x.lane_executor,
                                              hold_executor=x.hold_executor, banned_seeds=x.banned_seeds,
                                              arms_sha256=arms_sha, regimeu0_u0_sha256=u0,
                                              pairs_by_cell=pairs_of(doc, cells))
        if planned > x.cap:
            raise SystemExit(f'planned {planned} > cap {x.cap}')
        sys.stdout.write(text)
    elif x.cmd == 'dryrun':
        rep = dryrun(x.out, arms_path=x.arms, cells=cells, groups=groups, k=x.k, parity_seeds=x.parity_seeds,
                     lane_executor=x.lane_executor, hold_executor=x.hold_executor, fake_seconds=x.fake_seconds,
                     spec_paths=x.spec, banned_seeds=x.banned_seeds)
        print(json.dumps({c: dict(parity=v['incumbent_parity'], arms={a: dict(acts=r['activations'], executor=r['executor'])
                                                                    for a, r in v['arms'].items()})
                          for c, v in rep['cells'].items()}, indent=1))
    elif x.cmd == 'plan':
        body = plan(x.root, regimeu0_root=x.regimeu0_root, arms_path=x.arms, cells=cells, cap=x.cap,
                    registration_md=x.registration, groups=groups, k=x.k, parity_seeds=x.parity_seeds,
                    lane_executor=x.lane_executor, hold_executor=x.hold_executor,
                    spec_paths=x.spec, prior_roots=x.prior_root, banned_seeds=x.banned_seeds,
                    only_executors=tuple(x.only_executor) if x.only_executor else None,
                    supersede_uncharged=x.supersede_uncharged, pilot_arms=x.pilot_arm)
        print(json.dumps(dict(plan_sha256=body['plan_sha256'], planned_sim_s=body['planned_sim_s'],
                              arms=sorted(body['families']) or sorted({j['arm'] for g in body['groups']
                                                                       for j in g['jobs']}),
                              groups=len(body['groups']), phase_budget=body.get('phase_budget'))))
    elif x.cmd == 'pilot':
        rep, sha = write_pilot(x.out, x.pilot_root, x.prior_root)
        print(json.dumps(dict(sha256=sha, integrity_ok=rep['integrity']['ok'], summary=rep['summary']), indent=1,
                         default=str))
        if not rep['integrity']['ok']:
            raise SystemExit(1)
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
        if not ver['integrity']['ok']:
            raise SystemExit(1)              # fail closed: no FULLPOL-facing file was written


if __name__ == '__main__':
    main()
