"""S-IMITREPLAY: end-to-end SUMO replay of the imitation-warm-started clean4 cooperation layer.

Question (MAIN 2026-11-06): independent of RL noise, does the clean4 cooperation layer (trigger head +
move library + V2V initiate/join), warm-started by imitation on the SUMO-verified B examples
(BOOST-PRE sidecar, ``coop_warmstart``), reproduce the known-good cooperation when it runs in the
clean4 runtime (frozen speed-only MAPPO u0 base, map wz3to2) on the TRAIN worlds of those moves?
The R-arm warm start on R rows is replayed the same way for comparison.

Design (zero RL, physics only in ``worker``):

* per world and arm, a BASE run = the Plan A collect job (``clean4_verify.job_fields``: MAPPO u0, same
  ``noise_seed(world, arm)``, same plant flags) and a COOP run = the same job + ``track2_backend`` with
  the warm-started worker state. The layer residual is zero-initialised and untouched by the warm
  start, and the trigger uses its own seeded streams, so both runs are byte-identical until the first
  move executes (common random numbers; checked per run pair: ``first_divergence_s``).
* r0 decodes the trigger greedily (argmax; ``feasible_proposals.decide_and_start`` and the optional
  RSU head are forced to deterministic=True in the worker process only; the MAPPO residual keeps the
  Plan A sampling). ``sampled`` decodes the trigger stochastically (the training rollout rule).
* dithers (NEWMAP2/PLANGEN replicate convention): r1 = -1 m/s^2 for 2 s on the non-member CAV nearest
  the drop at the verified onset t0, r2 = the same on a seeded random non-member CAV within 1 km of the
  drop; applied in BOTH arms at the pre-safety request boundary (never an HDV, never a coop member);
  the CAV is chosen from the r0 base rows at t0 (verified members and r0 coop move members excluded).
  A dither is INERT unless it changes the base or coop J over [t0, t0+420) by >= 50 veh-s vs r0 and
  the issued acceleration of the dithered CAV differs from r0.
* every job is precharged (own ledger ``<root>/CHARGES.jsonl``) BEFORE physics; the cap is checked
  per atomic pair; nothing is ever re-run; ``<root>/STOP`` halts dispatch.

Metrics per verified moment (cost difference = treatment - reference = coop - base; negative =
improvement): fire within +-10 s of t0 and verified type, assembly (initiate -> joins -> executed,
member counts), dJ windows [t0, t0+D+120), [t0, t0+300), [t0, t0+420), false triggers elsewhere and
their local dJ, safety totals (collisions/teleports abort the plant; TTC<3 s exposure, hard braking).

CLI (cpuV, nice 19, CUDA_VISIBLE_DEVICES=, COOPMWM_MAP_SPEC=<wz3to2 MAP_SPEC>, COOPMWM_G5_RUNTIME=1):
  python -m tools.coopmwm.imitreplay init   --root R [--cap 100000]
  python -m tools.coopmwm.imitreplay plan   --root R --config CFG.json --u0 U0.pt --sidecar-B P --sidecar-R P
        [--receipt WARMSTART_RECEIPT.json] --rows-B ... --rows-R ... [--only-rows-B ...] [--only-rows-R ...]
  python -m tools.coopmwm.imitreplay run    --root R [--workers 16] [--load-high 44]
  python -m tools.coopmwm.imitreplay status --root R
  python -m tools.coopmwm.imitreplay analyze --root R [--out ANALYSIS.json]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

SCHEMA = 'coopmwm.imitreplay.v1'
STABLE_ID = 'S-IMITREPLAY'
PHASE = 'clean4_imit_check'
CAP = 100_000
MAP = 'wz3to2'
EPISODE_S = 1200
WARMUP_S = 300
ONSET_TOL_S = 10
TAIL_S = 120
LATER_S = 420
DITHER_A = -1.0
DITHER_S = 2
DITHER_RANGE_M = 1000.0
INERT_VS = 50.0
TRAIN_RANGES = ((989200, 989599), (989000, 989099))
FORBIDDEN = ((989600, 989699), (964000, 964199), (962000, 962999), (977000, 977999))
STAGES = ('r0_B', 'r0_R', 'dither_B', 'sampled_B', 'dither_R', 'sampled_R')
SERIES_KEYS = ('team_delay_vs', 'road_delay_vs', 'source_wait_vs', 'collisions', 'teleports',
               'ttc_below_3_vs', 'hard_brake')
FAMILY_ALIASES = dict(zipper='zipper_order', discharge='discharge_sync', platoon='platoon_headway',
                      vsl_rows='harmonize_rows')
RESTRICTED = ('holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth',
              'vehicle_params', 'driver_params', 'vtype')
RULE = ('S-IMITREPLAY v1: base = Plan A collect job (MAPPO u0, noise_seed(world, arm)); coop = same job + '
        'warm-started track2 layer; r0 greedy trigger, sampled = training rule; dithers -1 m/s^2 x 2 s on '
        'the nearest-drop (r1) / seeded random <=1 km (r2) non-member CAV at t0 in both arms; fire = '
        'executed move start within +-10 s of t0; dJ = coop - base (negative = improvement); precharged '
        'own ledger, cap per atomic pair, no re-runs; TRAIN 989200-989599 and 989000-989099 only.')


# ============================================================================ small helpers

def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode())


def state_equal(a, b):
    """Two state dicts with identical names, shapes and values."""
    import torch
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def guard_path(path):
    text = str(path).replace('\\', '/').lower()
    if any(s in text for s in RESTRICTED) or 'sealed' in text.split('/'):
        raise ValueError(f'restricted input path {path}')
    return Path(path)


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, indent=1, sort_keys=True, default=str)+'\n')


def _read(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    return json.loads(path.read_text(encoding='utf-8'))


def _jsonl(path):
    return [json.loads(x) for x in guard_path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def seed_allowed(seed):
    if type(seed) is not int or any(lo <= seed <= hi for lo, hi in FORBIDDEN):
        return False
    return any(lo <= seed <= hi for lo, hi in TRAIN_RANGES)


def norm_family(name):
    name = str(name or '')
    if name.startswith('B_') or name.startswith('R_') and name != 'R_random':
        name = name[2:]
    return FAMILY_ALIASES.get(name, name)


def move_type(move):
    """Normalised move type of a runtime move dict (trigger table entry / active move)."""
    if not isinstance(move, dict):
        return None
    family = str(move.get('family') or '')
    if family in ('lane_drop_initiate', 'lane_drop_join'):
        return str(move.get('kind') or 'lane_drop')
    if family in ('wall_initiate', 'wall_join'):
        return 'v2v_wall'
    for suffix in ('_initiate', '_join'):
        if family.endswith(suffix):
            return norm_family(family[:-len(suffix)])
    return norm_family(family)


def move_role(move):
    family = str((move or {}).get('family') or '')
    return 'join' if family.endswith('_join') else 'initiate' if family.endswith('_initiate') else 'single'


def session_id(move):
    if not isinstance(move, dict):
        return None
    for key in ('wall_id', 'session_id'):
        if move.get(key) is not None:
            return str(move[key])
    return None


# ============================================================================ registration / ledger

def init(root, cap=CAP):
    root = guard_path(root)
    if int(cap) > CAP:
        raise ValueError('cap above the registered clean4_imit_check phase (100,000 sim-s)')
    root.mkdir(parents=True, exist_ok=True)
    reg = root/'REGISTRATION.json'
    if reg.exists():
        return _read(reg)
    value = dict(schema=SCHEMA, stable_id=STABLE_ID, phase=PHASE, cap_sim_s=int(cap), rule=RULE,
                 module_sha256=sha_file(__file__), train_ranges=TRAIN_RANGES, forbidden=FORBIDDEN,
                 unix=time.time())
    _write_new(reg, value)
    return value


def registration(root):
    return _read(Path(root)/'REGISTRATION.json')


def _lock(root):
    import fcntl
    f = (Path(root)/'LEDGER.lock').open('a')
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def charges(root):
    path = Path(root)/'CHARGES.jsonl'
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()] if path.exists() else []


def spent(root):
    return sum(int(c['sim_s']) for c in charges(root))


def charge(root, entries):
    """Atomically precharge a list of (job id, sim_s, stage); refuse duplicates and the cap."""
    cap = int(registration(root)['cap_sim_s'])
    lock = _lock(root) if os.name != 'nt' else None
    try:
        seen = {c['id'] for c in charges(root)}
        total = spent(root)
        want = sum(int(s) for _, s, _ in entries)
        if any(i in seen for i, _, _ in entries):
            raise ValueError('job already charged (never re-run)')
        if total+want > cap:
            return False
        with (Path(root)/'CHARGES.jsonl').open('a', encoding='utf-8') as f:
            for ident, sim_s, stage in entries:
                f.write(json.dumps(dict(id=ident, sim_s=int(sim_s), stage=stage, phase=PHASE,
                                        unix=time.time()), sort_keys=True)+'\n')
        return True
    finally:
        if lock is not None:
            lock.close()


# ============================================================================ moments / plan

def _world_of(row):
    from . import map_scenarios as ms
    w = row.get('world') or {}
    seed = int(w.get('seed', row.get('world_seed', -1)))
    if not seed_allowed(seed):
        raise ValueError(f'world seed {seed} is not an allowed TRAIN seed')
    if w.get('map', MAP) != MAP:
        raise ValueError('not a wz3to2 row')
    if 'demand' in w and 'penetration' in w:
        demand = w['demand']['main'] if isinstance(w['demand'], dict) else w['demand']
        return ms.world(seed, int(demand), float(w['penetration']))
    cell = w.get('cell_key') or w.get('cell') or row.get('cell')
    return ms.world_from_key(seed, cell)


def moments(paths, arm, *, only=None):
    """Verified moments of one arm: (world, t0, family, members) groups of mined rows (replicate seeds kept).

    only: optional set of (cell_key, seed, t0, family) keys (e.g. the warm start's admitted rows)."""
    out, excluded = {}, []
    for path in paths:
        for row in _jsonl(path):
            if row.get('arm') != arm:
                excluded.append(dict(source=Path(path).name, reason='other arm'))
                continue
            try:
                world = _world_of(row)
            except (ValueError, KeyError, StopIteration) as error:
                excluded.append(dict(source=Path(path).name, situation=row.get('situation_id'),
                                     reason=str(error)[:120]))
                continue
            family = norm_family(row.get('family'))
            t0 = int(row['t0'])
            key = (world['cell_key'], world['seed'], t0, family)
            if only is not None and key not in only:
                excluded.append(dict(source=Path(path).name, situation=row.get('situation_id'),
                                     reason='not in the restricted (warm-start admitted) set'))
                continue
            members = tuple(row.get('members') or ())
            mkey = key+(members,)
            m = out.get(mkey)
            if m is None:
                m = out[mkey] = dict(
                    id=f'{arm}_{world["cell_key"]}_s{world["seed"]}_t{t0}_{family}_{len(out):02d}',
                    arm=arm, world=world, t0=t0, family=family, variant=row.get('variant'),
                    members=list(members), duration_s=int(row.get('duration_s') or 0) or None,
                    source=Path(path).name, situation_id=row.get('situation_id'),
                    verification_id=row.get('verification_id'), plan_sha256=row.get('plan_sha256'),
                    replicate_seeds=[], synergy_veh_s={}, advantage_veh_s={})
            m['replicate_seeds'].append(row.get('seed'))
            for k in ('synergy_veh_s', 'advantage_veh_s'):
                v = row.get(k)
                if isinstance(v, dict):
                    m[k].update({str(a): b for a, b in v.items()})
                elif isinstance(v, (int, float)):
                    m[k][str(row.get('seed'))] = v
    return list(out.values()), excluded


def world_key(world):
    return f'{world["cell_key"]}_s{world["seed"]}'


def _order(ms_):
    """Pre-frozen, outcome-free order: canonical clean4 cell first, then seed, cell, t0."""
    return sorted(ms_, key=lambda m: (m['world']['cell_key'] != 'D3200_p60', m['world']['seed'],
                                      m['world']['cell_key'], m['t0'], m['family']))


def build_jobs(moments_by_arm, *, include_r=True):
    """Frozen job list. Each entry is an atomic PAIR {base, coop} charged together (or a single coop
    job for 'sampled', whose base is the r0 base)."""
    groups = []

    def add(stage, arm, world, kind, decode, stop, moment=None, dither=None, needs=()):
        ident = f'{stage}_{arm}_{world_key(world)}' + (f'_t{moment["t0"]}_{moment["family"]}' if moment else '')
        if dither:
            ident += f'_{dither}'
        return dict(id=f'{ident}_{kind}', stage=stage, arm=arm, world=world, kind=kind, decode=decode,
                    stop=int(stop), sim_s=int(stop), moment=moment['id'] if moment else None,
                    t0=moment['t0'] if moment else None, dither=dither, needs=list(needs))

    for arm in ('B', 'R') if include_r else ('B',):
        worlds = {}
        for m in _order(moments_by_arm.get(arm, ())):
            worlds.setdefault(world_key(m['world']), (m['world'], []))[1].append(m)
        for wk, (world, ms_) in worlds.items():
            base = add(f'r0_{arm}', arm, world, 'base', None, EPISODE_S)
            coop = add(f'r0_{arm}', arm, world, 'coop', 'greedy', EPISODE_S)
            groups.append(dict(stage=f'r0_{arm}', jobs=[base, coop], moments=[m['id'] for m in ms_]))
    for arm in ('B', 'R') if include_r else ('B',):
        ordered = _order(moments_by_arm.get(arm, ()))
        for m in ordered:
            r0 = [f'r0_{arm}_{world_key(m["world"])}_base', f'r0_{arm}_{world_key(m["world"])}_coop']
            stop = min(EPISODE_S, m['t0']+LATER_S)
            for d in ('r1', 'r2'):
                groups.append(dict(stage=f'dither_{arm}', jobs=[
                    add(f'dither_{arm}', arm, m['world'], 'base', None, stop, m, d, r0),
                    add(f'dither_{arm}', arm, m['world'], 'coop', 'greedy', stop, m, d, r0)], moments=[m['id']]))
        done = set()
        for m in ordered:
            wk = world_key(m['world'])
            if wk in done:
                continue
            done.add(wk)
            groups.append(dict(stage=f'sampled_{arm}', jobs=[
                add(f'sampled_{arm}', arm, m['world'], 'coop', 'sampled', EPISODE_S, needs=[f'r0_{arm}_{wk}_base'])],
                moments=[x['id'] for x in ordered if world_key(x['world']) == wk]))
    rank = {s: i for i, s in enumerate(STAGES)}
    groups.sort(key=lambda g: rank[g['stage']])          # stable inside a stage
    return groups


def plan(root, *, config, u0, sidecars, rows, only_rows=None, receipt=None, u0_sha256=None,
         include_r=True, policy_deterministic=False):
    """Zero physics: copy/verify inputs into <root>/inputs and freeze PLAN.json."""
    import shutil
    root = guard_path(root)
    reg = registration(root)
    if (root/'PLAN.json').exists():
        raise ValueError('PLAN.json already frozen (new root required)')
    inputs = root/'inputs'
    inputs.mkdir(exist_ok=True)
    files = {}

    def copy(src, name):
        src = guard_path(src)
        dst = inputs/name
        if not dst.exists():
            shutil.copyfile(src, dst)
        files[name] = dict(source=str(src), sha256=sha_file(dst))
        if sha_file(src) != files[name]['sha256']:
            raise ValueError(f'copy of {src} differs')
        return dst

    copy(config, 'CONFIG.json')
    copy(u0, 'u0.pt')
    if u0_sha256 and files['u0.pt']['sha256'] != u0_sha256:
        raise ValueError('MAPPO u0 identity differs')
    for arm, path in sidecars.items():
        copy(path, f'SIDECAR_{arm}.pt')
    if receipt:
        copy(receipt, 'WARMSTART_RECEIPT.json')
    split = {}
    if receipt:
        rec = _read(inputs/'WARMSTART_RECEIPT.json')
        for k in ('train', 'heldout'):
            for s in (rec.get('split') or {}).get(k, ()):
                split[int(s)] = k
    by_arm, excluded = {}, {}
    for arm in ('B', 'R'):
        only = None
        if only_rows and only_rows.get(arm):
            only = set()
            for p in only_rows[arm]:
                for r in _jsonl(p):
                    try:
                        w = _world_of(r)
                    except (ValueError, KeyError, StopIteration):
                        continue
                    only.add((w['cell_key'], w['seed'], int(r['t0']), norm_family(r.get('family'))))
                files[f'only_{arm}_{Path(p).name}'] = dict(source=str(p), sha256=sha_file(p))
        for p in rows.get(arm, ()):
            files[f'rows_{arm}_{Path(p).name}'] = dict(source=str(p), sha256=sha_file(p))
        ms_, ex = moments(rows.get(arm, ()), arm, only=only)
        for m in ms_:
            m['imitation_split'] = split.get(m['world']['seed'])
        by_arm[arm], excluded[arm] = _order(ms_), ex
    groups = build_jobs(by_arm, include_r=include_r)
    for g in groups:
        for j in g['jobs']:
            if not seed_allowed(j['world']['seed']):
                raise ValueError('forbidden world in plan')
    body = dict(schema=SCHEMA+'.plan', stable_id=STABLE_ID, phase=PHASE, cap_sim_s=reg['cap_sim_s'],
                inputs=files, policy_deterministic=bool(policy_deterministic), rule=RULE,
                moments=by_arm, excluded=excluded, groups=groups,
                planned_sim_s={s: sum(j['sim_s'] for g in groups if g['stage'] == s for j in g['jobs'])
                               for s in STAGES},
                module_sha256=sha_file(__file__), unix=time.time())
    body['plan_sha256'] = digest({k: v for k, v in body.items() if k != 'unix'})
    _write_new(root/'PLAN.json', body)
    return body


# ============================================================================ worker (physics)

def _load_plan(root):
    return _read(Path(root)/'PLAN.json')


def find_job(plan_body, ident):
    for g in plan_body['groups']:
        for j in g['jobs']:
            if j['id'] == ident:
                return g, j
    raise KeyError(ident)


def moment_of(plan_body, mid):
    for arm in ('B', 'R'):
        for m in plan_body['moments'].get(arm, ()):
            if m['id'] == mid:
                return m
    return None


def accel_job(root, plan_body, job, *, weights, cfg=None, worker=None):
    """The accel_plant job dict: Plan A collect fields (+ track2 layer for coop)."""
    from . import clean4_verify as c4v
    world, arm = job['world'], job['arm']
    path = Path(root)/'jobs'/job['id']/'plant'
    out = c4v.job_fields(ident=job['id'], path=path, world=world, arm=arm, weights=weights,
                         noise_seed=c4v.noise_seed(world, arm), runtime=c4v.RUNTIME_DIR,
                         checkpoint=c4v.CHECKPOINT, sim_s=job['sim_s'], v2v=c4v._v2v_width(weights),
                         deterministic=plan_body.get('policy_deterministic', False))
    out.update(stop=job['stop'], queue_root=str(Path(root)/'jobs'))
    if job['kind'] == 'coop':
        out.update(track2_backend=True, track2_config=cfg, track2_state=worker)
    return out


class Greedy:
    """Worker-process-only patch: trigger (and optional RSU) decisions decoded deterministically."""

    def __init__(self):
        self.restore = []

    def __enter__(self):
        from . import feasible_proposals as fp
        original = fp.decide_and_start

        def decide_and_start(*args, **kwargs):
            kwargs['deterministic'] = True
            return original(*args, **kwargs)

        fp.decide_and_start = decide_and_start
        self.restore.append((fp, 'decide_and_start', original))
        try:
            from . import lane_drop_rsu
            rsu_decide = lane_drop_rsu.RSU.decide

            def decide(this, now, *args, **kwargs):
                kwargs['deterministic'] = True
                return rsu_decide(this, now, *args, **kwargs)

            lane_drop_rsu.RSU.decide = decide
            self.restore.append((lane_drop_rsu.RSU, 'decide', rsu_decide))
        except (ImportError, AttributeError):
            pass
        return self

    def __exit__(self, *exc):
        for obj, name, value in reversed(self.restore):
            setattr(obj, name, value)
        return False


class Recorder:
    """controller_hook: per-second totals series (via a cf.tick wrapper), dither application, row
    captures at requested times. Reads only; the dither writes ONE pre-safety request for 2 s."""

    def __init__(self, *, capture_times=(), dither=None):
        self.capture_times = {int(t) for t in capture_times}
        self.dither = dither              # dict(vid, t0, a, seconds) or None
        self.series, self.captures, self.dither_log = [], {}, []
        self.controller = None

    def __call__(self, controller, job, prefixes):
        self.controller = controller
        if self.dither is None:
            return
        previous = controller.apply
        d = self.dither

        def apply(rows, commands, now):
            desired = previous(rows, commands, now)
            t = int(now)
            if d['t0'] <= t < d['t0']+d['seconds']:
                vid = d['vid']
                present = vid in commands
                if present:
                    desired[vid] = dict(requested_raw=float(d['a']), acceleration_residual=float(d['a'])-commands[vid],
                                        imitreplay_dither=True)
                self.dither_log.append(dict(t=t, vid=vid, present=present,
                                            mappo_command=commands.get(vid)))
            return desired

        controller.apply = apply

    def tick_wrapper(self, original):
        def tick(b, controller=None, trace=None):
            now = int(round(float(b.env.now)))
            audit = None
            try:
                audit = original(b, controller, trace)
                return audit
            finally:
                self.series.append([now]+[float(b.totals.get(k, 0.)) for k in SERIES_KEYS])
                if self.dither is not None and audit:
                    for a in audit:
                        if a.get('vid') == self.dither['vid'] and self.dither['t0'] <= now < self.dither['t0']+self.dither['seconds']+2:
                            self.dither_log.append(dict(t=now, vid=a['vid'], audit=dict(
                                issued_safe=a.get('issued_safe'), mappo_safe=a.get('mappo_safe'),
                                requested_raw=a.get('requested_raw'), clamped=a.get('clamped'))))
        return tick


def _row_view(r):
    return {k: r.get(k) for k in ('vid', 's_m', 'v_mps', 'lane', 'is_cav')}


def summarize_steps(steps, *, capture_times=()):
    """Compact telemetry from accel_plant steps (controller events): decisions, executed moves,
    protocol sessions/joins, requested row captures and the per-second reward."""
    want = {int(t) for t in capture_times}
    decisions, sessions, joins, active, captures, rewards = [], {}, [], {}, {}, []
    for e in steps:
        t = int(e['time_s'])
        rewards.append([t, float(e.get('reward', 0.))])
        if t in want or t+1 in want:
            captures[str(t)] = [_row_view(r) for r in e.get('rows') or ()]
        s = e.get('sample') or {}
        trig = s.get('track2_trigger')
        if trig:
            picks = []
            members = (trig.get('context') or {}).get('members') or []
            accepted = trig.get('accepted') or {}
            for i, ego in enumerate(members):
                try:
                    move = trig['tables'][i][int(trig['actions'][i])]
                except (IndexError, KeyError, TypeError, ValueError):
                    move = None
                if move is None or move.get('family') in (None, 'none'):
                    continue
                picks.append(dict(ego=ego, family=move.get('family'), type=move_type(move), role=move_role(move),
                                  members=list(move.get('members') or ()), session=session_id(move),
                                  kind=move.get('kind'), onset=move.get('onset'), hold=move.get('hold'),
                                  accepted=bool(accepted.get(ego))))
            decisions.append(dict(t=t, n_members=len(members), picks=picks,
                                  stats={k: v for k, v in (trig.get('proposals') or {}).items()
                                         if k in ('selected_non_none', 'rejected_non_none', 'v2v_wall',
                                                  'lane_drop', 'nm_families', 'decisions')}))
        tr = s.get('track2') or {}
        for a in tr.get('active_moves') or ():
            mv = a.get('move') or {}
            key = json.dumps([mv.get('family'), sorted(mv.get('members') or ()), a.get('start'),
                              session_id(mv)], default=str)
            rec = active.get(key)
            if rec is None:
                rec = active[key] = dict(family=mv.get('family'), type=move_type(mv), role=move_role(mv),
                                         members=list(mv.get('members') or ()), session=session_id(mv),
                                         start=a.get('start'), until=a.get('until'), first_t=t, last_t=t,
                                         onset=mv.get('onset'), hold=mv.get('hold'), kind=mv.get('kind'))
            rec['last_t'] = t
        receipts = []
        if tr.get('v2v_wall'):
            receipts.append(('v2v_wall', tr['v2v_wall']))
        if (tr.get('lane_drop') or {}).get('protocol'):
            receipts.append(('lane_drop', tr['lane_drop']['protocol']))
        if tr.get('nm_families'):
            receipts.append(('nm_families', tr['nm_families'].get('protocol', tr['nm_families'])))
        for proto, rec in receipts:
            for ss in rec.get('sessions') or ():
                sid = f'{proto}:{ss.get("wall_id", ss.get("session_id"))}'
                cur = sessions.get(sid)
                if cur is None:
                    cur = sessions[sid] = dict(protocol=proto, id=sid, initiator=ss.get('initiator'),
                                               kind=ss.get('kind'), first_t=t, last_t=t, max_members=0,
                                               released_t=None, members_by_t=[])
                cur['last_t'] = t
                n = int(ss.get('members') or 0)
                if n != (cur['members_by_t'][-1][1] if cur['members_by_t'] else None):
                    cur['members_by_t'].append([t, n])
                cur['max_members'] = max(cur['max_members'], n)
                if ss.get('released') and cur['released_t'] is None:
                    cur['released_t'] = t
            for j in rec.get('joins') or ():
                joins.append(dict(t=t, protocol=proto, join=j))
    return dict(decisions=decisions, executed=sorted(active.values(), key=lambda r: (r['first_t'], str(r['members']))),
                sessions=sorted(sessions.values(), key=lambda r: r['first_t']), joins=joins,
                captures=captures, rewards=rewards)


def worker(root, ident):
    """ONE precharged episode (base or coop); writes jobs/<id>/SUMMARY.json.gz or FAILED.json."""
    import torch
    root = Path(root)
    body = _load_plan(root)
    group, job = find_job(body, ident)
    if not any(c['id'] == ident for c in charges(root)):
        raise ValueError('job not precharged')
    work = root/'jobs'/ident
    work.mkdir(parents=True, exist_ok=False)
    started = time.time()
    _write_new(work/'CLAIM.json', dict(pid=os.getpid(), id=ident, sim_s=job['sim_s'], unix=started))
    from . import map_descriptor as md
    if os.environ.get(md.ENV) is None or os.environ.get('COOPMWM_G5_RUNTIME') != '1':
        raise RuntimeError('COOPMWM_MAP_SPEC and COOPMWM_G5_RUNTIME=1 required')
    saved = torch.load(root/'inputs'/'u0.pt', map_location='cpu', weights_only=False)
    weights = saved['policy'] if isinstance(saved, dict) and 'policy' in saved else saved
    cfg = worker_state = None
    if job['kind'] == 'coop':
        cfg = _read(root/'inputs'/'CONFIG.json')
        payload = torch.load(root/'inputs'/f'SIDECAR_{job["arm"]}.pt', map_location='cpu', weights_only=False)
        if payload.get('arm') not in (None, job['arm']):
            raise ValueError('sidecar arm mismatch')
        worker_state = payload['worker']
    capture = set()
    moments_here = [moment_of(body, mid) for mid in group['moments']]
    for m in moments_here:
        if m:
            capture.update({m['t0']-1, m['t0']})
    dither = None
    if job.get('dither'):
        dither = _read(root/'jobs'/'_dither'/f'{job["moment"]}_{job["dither"]}.json')
        if dither.get('vid') is None:
            raise ValueError('no dither CAV resolved')
        dither = dict(vid=dither['vid'], t0=int(job['t0']), a=DITHER_A, seconds=DITHER_S)
    rec = Recorder(capture_times=capture, dither=dither)
    accel = accel_job(root, body, job, weights=weights, cfg=cfg, worker=worker_state)
    from tools.coopfree import backend as cfb
    from . import accel_plant, g4_runtime
    original_tick, original_capture = cfb.tick, g4_runtime.BoundaryCapture.for_job
    cfb.tick = rec.tick_wrapper(original_tick)
    # G4 boundary capture is training-credit telemetry only (deep copies per decision); off here.
    g4_runtime.BoundaryCapture.for_job = classmethod(lambda cls, *a, **k: None)
    greedy = Greedy() if (job['kind'] == 'coop' and job['decode'] == 'greedy') else None
    result, error = None, None
    try:
        if greedy is not None:
            greedy.__enter__()
        result = accel_plant.episode(accel, controller_hook=rec)
    except Exception as exc:                                    # safety abort or failure: receipt, no retry
        import traceback
        error = dict(error=repr(exc), traceback=traceback.format_exc())
    finally:
        if greedy is not None:
            greedy.__exit__(None, None, None)
        cfb.tick, g4_runtime.BoundaryCapture.for_job = original_tick, original_capture
    steps = (result or {}).get('steps') or (rec.controller.events if rec.controller is not None else [])
    summary = dict(schema=SCHEMA+'.summary', id=ident, stage=job['stage'], arm=job['arm'], kind=job['kind'],
                   decode=job['decode'], world=job['world'], dither=dither, dither_log=rec.dither_log,
                   noise_seed=accel['noise_seed'], deterministic=accel['deterministic'],
                   totals=(result or {}).get('totals'), actual_sim_s=(result or {}).get('actual_sim_s'),
                   charged_sim_s=job['sim_s'], series_keys=list(SERIES_KEYS), series=rec.series,
                   error=error, wall_s=round(time.time()-started, 2), map_spec=os.environ.get(md.ENV),
                   **summarize_steps(steps, capture_times=capture))
    if job['kind'] == 'coop':
        summary['sidecar_sha256'] = sha_file(root/'inputs'/f'SIDECAR_{job["arm"]}.pt')
    with gzip.open(work/'SUMMARY.json.gz', 'xt', encoding='utf-8') as f:
        json.dump(summary, f, default=str)
    if error is not None:
        _write_new(work/'FAILED.json', dict(id=ident, **error, wall_s=summary['wall_s']))
    else:
        _write_new(work/'DONE.json', dict(id=ident, wall_s=summary['wall_s'], unix=time.time()))
    return summary


# ============================================================================ dither selection (zero physics)

def drop_s_m():
    from . import map_descriptor as md
    return float(md.current().bottleneck['drop_s_m'])


def resolve_dither(root, moment, which, *, drop_s=None):
    """Choose the dithered CAV from the r0 BASE rows at t0 (NEWMAP2 rule): r1 nearest the drop upstream,
    r2 seeded random within 1 km of the drop; verified members and r0-coop move members excluded."""
    root = Path(root)
    drop_s = drop_s_m() if drop_s is None else drop_s
    wk = world_key(moment['world'])
    base = _read(root/'jobs'/f'r0_{moment["arm"]}_{wk}_base'/'SUMMARY.json.gz')
    coop = _read(root/'jobs'/f'r0_{moment["arm"]}_{wk}_coop'/'SUMMARY.json.gz')
    rows = base['captures'].get(str(moment['t0'])) or []
    t0 = moment['t0']
    excluded = set(moment['members'])
    for mv in coop.get('executed') or ():
        if (mv.get('start') or mv['first_t']) <= t0+DITHER_S and (mv.get('until') or mv['last_t']) >= t0:
            excluded.update(mv.get('members') or ())
    coop_rows = {r['vid'] for r in coop['captures'].get(str(t0)) or ()}
    cands = [r for r in rows if r.get('is_cav') and r['vid'] not in excluded and r['vid'] in coop_rows
             and isinstance(r.get('s_m'), (int, float))]
    vid = None
    if which == 'r1':
        upstream = [r for r in cands if r['s_m'] <= drop_s]
        pool = upstream or cands
        if pool:
            vid = min(pool, key=lambda r: (abs(drop_s-r['s_m']), r['vid']))['vid']
    else:
        pool = sorted((r for r in cands if abs(drop_s-r['s_m']) <= DITHER_RANGE_M), key=lambda r: r['vid'])
        if pool:
            vid = random.Random(f'imitreplay_r2:{moment["id"]}').choice(pool)['vid']
    out = dict(moment=moment['id'], which=which, vid=vid, t0=t0, drop_s_m=drop_s, candidates=len(cands),
               excluded=sorted(excluded), rule='r1 nearest-drop upstream non-member CAV; r2 seeded random '
               'non-member CAV within 1 km of the drop; rows = r0 base at t0, present in r0 coop at t0')
    path = root/'jobs'/'_dither'/f'{moment["id"]}_{which}.json'
    if not path.exists():
        _write_new(path, out)
    return _read(path)


# ============================================================================ dispatcher

def _done(root, ident):
    d = Path(root)/'jobs'/ident
    return (d/'DONE.json').exists() or (d/'FAILED.json').exists()


def _ok(root, ident):
    return (Path(root)/'jobs'/ident/'DONE.json').exists()


def run(root, *, workers=16, load_high=44.0, low_workers=8, poll_s=5.0, stages=None, python=None):
    root = Path(root)
    body = _load_plan(root)
    if os.name != 'nt' and (os.getpriority(os.PRIO_PROCESS, 0) != 19 or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise ValueError('run under nice 19 with CUDA_VISIBLE_DEVICES= (CPU only)')
    python = python or sys.executable
    log = (root/'DISPATCH.jsonl').open('a', encoding='utf-8')

    def note(**kw):
        log.write(json.dumps(dict(kw, unix=time.time()), sort_keys=True, default=str)+'\n')
        log.flush()

    running = {}
    charged = {c['id'] for c in charges(root)}
    note(event='start', pid=os.getpid(), spent=spent(root), cap=registration(root)['cap_sim_s'])
    queue = [g for g in body['groups'] if stages is None or g['stage'] in stages]
    capped = False
    for g in queue:
        if (root/'STOP').exists():
            note(event='stop_file')
            break
        jobs = [j for j in g['jobs'] if j['id'] not in charged]
        if not jobs:
            continue
        # prerequisites (dither/sampled need r0) - wait until done
        needs = sorted({n for j in jobs for n in j.get('needs', ())})
        while any(not _done(root, n) for n in needs):
            _reap(running, note)
            if (root/'STOP').exists():
                break
            time.sleep(poll_s)
        if any(not _ok(root, n) for n in needs):
            note(event='skip_prereq_failed', group=[j['id'] for j in jobs])
            continue
        if g['stage'].startswith('dither'):
            m = moment_of(body, jobs[0]['moment'])
            d = resolve_dither(root, m, jobs[0]['dither'])
            if d.get('vid') is None:
                note(event='skip_no_dither_cav', group=[j['id'] for j in jobs])
                continue
        if not charge(root, [(j['id'], j['sim_s'], j['stage']) for j in jobs]):
            note(event='cap_reached', group=[j['id'] for j in jobs], spent=spent(root))
            capped = True
            break
        charged.update(j['id'] for j in jobs)
        for j in jobs:
            while True:
                _reap(running, note)
                try:
                    load = os.getloadavg()[0]
                except (AttributeError, OSError):
                    load = 0.
                limit = workers if load <= load_high else min(workers, low_workers)
                if len(running) < limit:
                    break
                time.sleep(poll_s)
            out = (root/'jobs'/'_logs')
            out.mkdir(parents=True, exist_ok=True)
            stream = (out/f'{j["id"]}.log').open('x')
            p = subprocess.Popen([python, '-B', '-m', 'tools.coopmwm.imitreplay', 'worker', '--root', str(root),
                                  '--job', j['id']], stdout=stream, stderr=subprocess.STDOUT, start_new_session=False)
            running[p.pid] = (p, j['id'], stream)
            note(event='launch', id=j['id'], pid=p.pid, load=load, limit=limit)
            with (root/'PIDS.jsonl').open('a') as f:
                f.write(json.dumps(dict(pid=p.pid, id=j['id'], unix=time.time()))+'\n')
    while running:
        _reap(running, note)
        time.sleep(poll_s)
    note(event='end', spent=spent(root), capped=capped)
    _write_new(root/f'RUN_END_{int(time.time())}.json', dict(spent=spent(root), capped=capped))
    return spent(root)


def _reap(running, note):
    for pid, (p, ident, stream) in list(running.items()):
        code = p.poll()
        if code is not None:
            stream.close()
            note(event='exit', id=ident, pid=pid, code=code)
            running.pop(pid)


def status(root):
    root = Path(root)
    body = _load_plan(root)
    out = dict(cap=registration(root)['cap_sim_s'], spent=spent(root), stages={})
    for g in body['groups']:
        st = out['stages'].setdefault(g['stage'], dict(jobs=0, charged=0, done=0, failed=0))
        ids = {c['id'] for c in charges(root)}
        for j in g['jobs']:
            st['jobs'] += 1
            st['charged'] += j['id'] in ids
            st['done'] += _ok(root, j['id'])
            st['failed'] += (root/'jobs'/j['id']/'FAILED.json').exists()
    return out


# ============================================================================ analysis (zero physics)

def _series_delay(summary):
    """{t: team-delay increment over [t, t+1)} from the per-second totals series."""
    out, prev = {}, None
    keys = summary['series_keys']
    i = keys.index('team_delay_vs')+1
    for row in summary['series']:
        t = int(row[0])
        cur = float(row[i])
        out[t] = cur-(prev if prev is not None else 0.)
        prev = cur
    return out


def _series_total(summary, key, a, b):
    keys = summary['series_keys']
    i = keys.index(key)+1
    vals = {int(r[0]): float(r[i]) for r in summary['series']}
    before = max((t for t in vals if t < a), default=None)
    last = max((t for t in vals if t < b), default=None)
    if last is None:
        return None
    return vals[last]-(vals[before] if before is not None else 0.)


def window_j(summary, a, b):
    inc = _series_delay(summary)
    ts = [t for t in inc if a <= t < b]
    if not ts:
        return None
    return sum(inc[t] for t in ts), len(ts)


def first_divergence(base, coop):
    ib, ic = _series_delay(base), _series_delay(coop)
    for t in sorted(set(ib) & set(ic)):
        if abs(ib[t]-ic[t]) > 1e-9:
            return t
    return None


def dj(base, coop, a, b):
    jb, jc = window_j(base, a, b), window_j(coop, a, b)
    if jb is None or jc is None or jb[1] != jc[1]:
        return None
    return jc[0]-jb[0]


def _starts(coop):
    """Executed starts: every executed move (first seen time) with its type, members and session."""
    return [dict(t=int(mv.get('start') if mv.get('start') is not None else mv['first_t']), type=mv['type'],
                 family=mv['family'], role=mv['role'], members=mv['members'], session=mv.get('session'),
                 last_t=mv['last_t'], until=mv.get('until')) for mv in coop.get('executed') or ()]


def _is_r(verified):
    return verified == 'R_random' or verified.startswith('R')


def assess_moment(m, base, coop, *, tol=ONSET_TOL_S):
    """fire = a non-none trigger pick or an executed move start within +-tol of t0; type = the verified
    family (R_random: any move); assembly = a verified-type group with >= 2 executing members (executed
    members of the near starts or a protocol session's member count); gain = dJ(move window) < 0."""
    t0, verified = m['t0'], norm_family(m['family'])
    D = int(m.get('duration_s') or 0) or 120
    starts = _starts(coop)
    near = [s for s in starts if abs(s['t']-t0) <= tol]
    picks_near = [dict(p, t=d['t']) for d in coop.get('decisions') or () if abs(d['t']-t0) <= tol
                  for p in d['picks']]
    decision_ticks_near = sorted({d['t'] for d in coop.get('decisions') or () if abs(d['t']-t0) <= tol})
    fired = bool(near or picks_near)

    def is_type(x):
        return _is_r(verified) or x.get('type') == verified

    typed_starts = [s for s in near if is_type(s)]
    typed_picks = [p for p in picks_near if is_type(p)]
    type_match = bool(typed_starts or typed_picks)
    members_v = set(m['members'])
    overlap = max((len(members_v & set(s['members']))/max(1, len(members_v)) for s in near), default=0.)
    sess = [s for s in coop.get('sessions') or () if abs(s['first_t']-t0) <= tol]
    typed_sess = [s for s in sess if _is_r(verified) or s.get('kind') in (None, verified)]
    group_sizes = {}
    for s in typed_starts:
        key = s['session'] or json.dumps(sorted(s['members']))
        group_sizes.setdefault(key, set()).update(s['members'])
    max_group = max((len(v) for v in group_sizes.values()), default=0)
    max_session = max((s['max_members'] for s in typed_sess), default=0)
    group_members = max(max_group, max_session)
    assembled = group_members >= 2
    windows = dict(move=(t0, t0+D+TAIL_S), h300=(t0, t0+300), h420=(t0, t0+LATER_S))
    gains = {k: dj(base, coop, a, min(b, EPISODE_S)) for k, (a, b) in windows.items()}
    gain_ok = gains['move'] is not None and gains['move'] < 0
    if not fired:
        link = 'no_fire'
    elif not type_match:
        link = 'wrong_move'
    elif not assembled:
        link = 'no_assembly'
    elif not gain_ok:
        link = 'no_gain'
    else:
        link = 'reproduced'
    nearest = min(starts, key=lambda s: abs(s['t']-t0), default=None)
    first_any = min((s['t'] for s in starts), default=None)
    return dict(moment=m['id'], arm=m['arm'], world=world_key(m['world']), t0=t0, verified_type=verified,
                verified_members=m['members'], imitation_split=m.get('imitation_split'), source=m['source'],
                fired=fired, starts_near=near, picks_near=picks_near, type_match=type_match,
                member_overlap=round(overlap, 3), decision_ticks_near=decision_ticks_near,
                sessions_near=[dict(id=s['id'], kind=s['kind'], first_t=s['first_t'], max_members=s['max_members'],
                                    released_t=s['released_t'], members_by_t=s['members_by_t']) for s in sess],
                group_members=group_members, assembled=assembled, dJ=gains,
                first_divergence_s=first_divergence(base, coop), first_executed_start_s=first_any,
                earlier_move_before_t0=first_any is not None and first_any < t0-tol,
                nearest_start=None if nearest is None else dict(t=nearest['t'], type=nearest['type'],
                                                                dt=nearest['t']-t0),
                link=link, verified_synergy_veh_s=m.get('synergy_veh_s'),
                verified_advantage_veh_s=m.get('advantage_veh_s'))


def false_triggers(ms_, base, coop, *, tol=ONSET_TOL_S):
    t0s = [m['t0'] for m in ms_]
    out = []
    for s in _starts(coop):
        if any(abs(s['t']-t0) <= tol for t0 in t0s):
            continue
        a = s['t']
        b = min(EPISODE_S, a+int(s.get('last_t', a)-a)+TAIL_S)
        out.append(dict(s, dJ_local=dj(base, coop, a, b), window=[a, b]))
    return out


def safety(summary, a=WARMUP_S, b=EPISODE_S):
    return {k: _series_total(summary, k, a, b) for k in ('collisions', 'teleports', 'ttc_below_3_vs', 'hard_brake')}


def analyze(root):
    root = Path(root)
    body = _load_plan(root)

    def load(ident):
        p = root/'jobs'/ident/'SUMMARY.json.gz'
        return _read(p) if p.exists() else None

    report = dict(schema=SCHEMA+'.analysis', plan_sha256=body['plan_sha256'], spent=spent(root),
                  cap=registration(root)['cap_sim_s'], moments=[], worlds=[], dithers=[], sampled=[])
    for g in body['groups']:
        if not g['stage'].startswith('r0_'):
            continue
        base_id, coop_id = (j['id'] for j in g['jobs'])
        base, coop = load(base_id), load(coop_id)
        ms_ = [moment_of(body, mid) for mid in g['moments']]
        w = dict(stage=g['stage'], base=base_id, coop=coop_id, ran=base is not None and coop is not None)
        if not w['ran']:
            for m in ms_:
                report['moments'].append(dict(moment=m['id'], arm=m['arm'], link='not_run'))
            report['worlds'].append(w)
            continue
        w.update(base_error=base.get('error'), coop_error=coop.get('error'),
                 first_divergence_s=first_divergence(base, coop),
                 dJ_episode=dj(base, coop, WARMUP_S, EPISODE_S),
                 safety_base=safety(base), safety_coop=safety(coop),
                 n_decision_ticks=len(coop.get('decisions') or ()),
                 n_non_none_picks=sum(len(d['picks']) for d in coop.get('decisions') or ()),
                 n_executed=len(coop.get('executed') or ()),
                 executed_types=_count(s['type'] for s in _starts(coop)),
                 sessions=len(coop.get('sessions') or ()),
                 false_triggers=false_triggers(ms_, base, coop))
        report['worlds'].append(w)
        for m in ms_:
            if coop.get('error') or base.get('error'):
                report['moments'].append(dict(moment=m['id'], arm=m['arm'], link='run_failed',
                                              base_error=(base.get('error') or {}).get('error'),
                                              coop_error=(coop.get('error') or {}).get('error')))
                continue
            report['moments'].append(assess_moment(m, base, coop))
    for g in body['groups']:
        if g['stage'].startswith('dither_'):
            bj, cj = g['jobs']
            base, coop = load(bj['id']), load(cj['id'])
            m = moment_of(body, bj['moment'])
            wk = world_key(m['world'])
            b0, c0 = load(f'r0_{m["arm"]}_{wk}_base'), load(f'r0_{m["arm"]}_{wk}_coop')
            rec = dict(moment=m['id'], which=bj['dither'], ran=base is not None and coop is not None)
            if rec['ran'] and b0 and c0 and not base.get('error') and not coop.get('error'):
                a, b = m['t0'], min(EPISODE_S, m['t0']+LATER_S)
                shift_b, shift_c = dj(b0, base, a, b), dj(c0, coop, a, b)
                applied = any(x.get('present') for x in base.get('dither_log') or ()) and any(
                    x.get('present') for x in coop.get('dither_log') or ())
                inert = not applied or (abs(shift_b or 0.) < INERT_VS and abs(shift_c or 0.) < INERT_VS)
                one = assess_moment(m, base, coop)
                rec.update(applied=applied, base_shift_vs_r0=shift_b, coop_shift_vs_r0=shift_c, inert=inert,
                           dither_cav=(base.get('dither') or {}).get('vid'), link=one['link'], dJ=one['dJ'],
                           fired=one['fired'], type_match=one['type_match'], assembled=one['assembled'],
                           safety_base=safety(base, a, b), safety_coop=safety(coop, a, b))
            elif rec['ran']:
                rec.update(error=dict(base=(base.get('error') or {}).get('error'),
                                      coop=(coop.get('error') or {}).get('error')))
            report['dithers'].append(rec)
        if g['stage'].startswith('sampled_'):
            (cj,) = g['jobs']
            coop = load(cj['id'])
            wk = world_key(cj['world'])
            base = load(f'r0_{cj["arm"]}_{wk}_base')
            rec = dict(job=cj['id'], ran=coop is not None)
            if coop is not None and base is not None and not coop.get('error'):
                ms_ = [moment_of(body, mid) for mid in g['moments']]
                rec.update(moments=[assess_moment(m, base, coop) for m in ms_],
                           dJ_episode=dj(base, coop, WARMUP_S, EPISODE_S), safety_coop=safety(coop),
                           false_triggers=false_triggers(ms_, base, coop),
                           executed_types=_count(s['type'] for s in _starts(coop)))
            elif coop is not None:
                rec.update(error=(coop.get('error') or {}).get('error'))
            report['sampled'].append(rec)
    report['links'] = {arm: _count(x['link'] for x in report['moments'] if x.get('arm') == arm) for arm in ('B', 'R')}
    return report


def _count(values):
    out = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0)+1
    return out


# ============================================================================ CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('init')
    a.add_argument('--root', required=True)
    a.add_argument('--cap', type=int, default=CAP)
    a = sub.add_parser('plan')
    a.add_argument('--root', required=True)
    a.add_argument('--config', required=True)
    a.add_argument('--u0', required=True)
    a.add_argument('--u0-sha256')
    a.add_argument('--sidecar-B', required=True)
    a.add_argument('--sidecar-R')
    a.add_argument('--receipt')
    a.add_argument('--rows-B', nargs='+', required=True)
    a.add_argument('--rows-R', nargs='*', default=())
    a.add_argument('--only-rows-B', nargs='*')
    a.add_argument('--only-rows-R', nargs='*')
    a.add_argument('--policy-deterministic', action='store_true')
    a = sub.add_parser('run')
    a.add_argument('--root', required=True)
    a.add_argument('--workers', type=int, default=16)
    a.add_argument('--low-workers', type=int, default=8)
    a.add_argument('--load-high', type=float, default=44.0)
    a.add_argument('--stages', nargs='*')
    a = sub.add_parser('worker')
    a.add_argument('--root', required=True)
    a.add_argument('--job', required=True)
    a = sub.add_parser('status')
    a.add_argument('--root', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--root', required=True)
    a.add_argument('--out')
    args = p.parse_args(argv)
    if args.cmd == 'init':
        print(json.dumps(init(args.root, args.cap), indent=1))
    elif args.cmd == 'plan':
        side = dict(B=args.sidecar_B)
        if args.sidecar_R:
            side['R'] = args.sidecar_R
        body = plan(args.root, config=args.config, u0=args.u0, u0_sha256=args.u0_sha256, sidecars=side,
                    rows=dict(B=args.rows_B, R=args.rows_R), receipt=args.receipt,
                    only_rows=dict(B=args.only_rows_B, R=args.only_rows_R),
                    include_r=bool(args.sidecar_R and args.rows_R), policy_deterministic=args.policy_deterministic)
        print(json.dumps(dict(plan_sha256=body['plan_sha256'], planned_sim_s=body['planned_sim_s'],
                              moments={k: len(v) for k, v in body['moments'].items()},
                              excluded={k: len(v) for k, v in body['excluded'].items()}), indent=1))
    elif args.cmd == 'run':
        print(run(args.root, workers=args.workers, load_high=args.load_high, low_workers=args.low_workers,
                  stages=args.stages))
    elif args.cmd == 'worker':
        s = worker(args.root, args.job)
        print(json.dumps(dict(id=s['id'], error=bool(s['error']), wall_s=s['wall_s'], totals=s['totals'])))
    elif args.cmd == 'status':
        print(json.dumps(status(args.root), indent=1))
    elif args.cmd == 'analyze':
        rep = analyze(args.root)
        text = json.dumps(rep, indent=1, sort_keys=True, default=str)
        if args.out:
            Path(args.out).write_text(text+'\n', encoding='utf-8')
        print(json.dumps(dict(links=rep['links'], spent=rep['spent']), indent=1))


if __name__ == '__main__':
    main()
