"""clean4 PRE-LAUNCH RE-VERIFICATION (owner/MAIN "Plan A", 2026-11-06 15:02 +08).

Phase ``clean4_prelaunch_verify`` (hard cap 80,000 sim-s inside the approved clean4 4M), its own
precharged ledger (``<root>/CHARGES.jsonl``), never replayed. The SUMO-only probe N3 (NEWMAP2) and
N8 admitted cooperation moments on the wz3to2 lane-drop map with a NATIVE base; their rows are not
ingestible by the coopmwm bank. This pipeline re-selects the same kind of moments by the SAME
geometry-only rule inside the coopmwm runtime on the CHOSEN clean4 base (frozen speed-only MAPPO
u0 unless the sanity check picks native) and re-verifies them with the proven merge-map native
path (mining collect2 / coop_probe probe + probe_native / wallmine), emitting what the clean3
training stack ingests: sumo_verify receipts, bank snapshots (``bank/snapshots/<sid>.pt``),
observable tapes, verify_perseed all8 confirmation records and coopmwm-mined-move-v1 rows.

Stages (every SUMO stage precharged BEFORE any physics; a stage the remaining cap cannot cover is
refused before dispatch; nothing is ever re-run):

  init     REGISTRATION.json (phase, cap, rules) BEFORE any job.
  collect  ONE base-policy 1200 s episode per (world, arm) (accel_plant.episode + Capture hook as
           mining collect2), charged 1200 sim-s before physics. At every ``every``-s mark inside the
           windows an OBSERVABLE selector (default = the N3 geometry rule) proposes a moment; per
           window the earliest moment with closing-lane pair AND abreast wall is kept, else the
           earliest pair, else the earliest wall. Saved per kept moment: CONTEXT.json (+ the
           per-second policy reference and observable track of every zone CAV for PROFILE_S s)
           and SNAPSHOT.pt (sv.capture). Never reads an outcome.
  plans    executable group plans per moment (builder protocol; B = N3 variants pair_g/pair_h/
           pair_t/wall_g/wall_h, R = the N3 random-proposal procedure) + member solos
           (replay_bank.solo_candidate).
  verify   seed 1 screen = ONE sv.verify (policy + groups + every member solo); a group continues
           only with seed-1 advantage LB95 > 0 and synergy LB95 > 0, both >= 1 veh-s, and
           collision/teleport deltas <= 0. Survivors get the native confirmation = the
           probe_native.run path (first-seed sv.verify + verify_perseed.confirm_all8 with
           extra_seeds = seeds-2) into ``<root>/minebank/bank``.
  extract  admitted & cooperative & 0 collisions/teleports & >= 2 members per (plan, seed) ->
           CLEAN4_MOVES_B/R.jsonl (coopmwm-mined-move-v1 + verified binding fields),
           CLEAN4_NEGATIVES.jsonl, POTENTIAL_INPUTS.json (clean4_potential gate inputs).
  quote    zero-physics per-moment screen/confirmation quotes (plan the 80k before spending it).
  status   cap, spent, remaining, stage counts.

CLI (cpuV: nice 19, own PIDs only, ``--workers`` bounds the branch subprocesses; ``<root>/STOP`` halts):
  python -m tools.coopmwm.clean4_verify init --root R [--cap 80000] [--base frozen_u0|native]
  python -m tools.coopmwm.clean4_verify collect --root R --arm B --weights frozen_u0.pt \
      --worlds 989200:2800:0.4,... [--leads NM2_SUMMARY.json NEWMAP_MOVES_B.jsonl] [--adapters MODULE]
  python -m tools.coopmwm.clean4_verify quote --root R --ctx DIR... [--variants pair_g,wall_g]
  python -m tools.coopmwm.clean4_verify verify --root R (--ctx DIR... | --queue FILE) [--seeds 3]
      [--workers N] [--variants ...] [--builder MODULE:FUNCTION]
  python -m tools.coopmwm.clean4_verify extract --root R [--range-m 200] [--cell D2800_p40]
  python -m tools.coopmwm.clean4_verify status --root R

Adapters that stay STUBS until N1/N2 land (see READY): ``Wz3to2GeometryStub`` (observable row ->
NEWMAP2 frame (edge, lane, x)), ``Wz3to2RuntimeStub`` (job/runtime binding + world guard of the
wz3to2 coopmwm episode and branch workers) and the N2 plan builder (``provisional_builder`` is a
clearly marked provisional open-loop stand-in).
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time

PHASE = 'clean4_prelaunch_verify'
PHASE_CAP = 80_000
SCHEMA = 'coopmwm.clean4.prelaunch_verify.v1'
MINED_SCHEMA = 'coopmwm-mined-move-v1'
MAP = 'wz3to2'
AUTHORITY = ('MAIN 2026-11-06 15:02 "clean4 ingestion blocker -> Plan A" (REGISTRATION.md): re-verify '
             'rule-selected N3/N8 moments in the coopmwm runtime on wz3to2 with the chosen base via the '
             'probe_native path; phase clean4_prelaunch_verify <= 80,000 inside the approved 4M; the '
             'pre-launch gate counts only re-verified rows')

# ---- NEWMAP2 (N3) geometry rule, verbatim constants (nm2_plan.py / REGISTRATION 14:50) -------------
X_B = 2696.0                                  # end of lane 0 on edge 'up' (the lane drop)
ZONE = (X_B-1800.0, X_B-400.0)
PAIR_DX = 100.0
WALL_SPREAD = 200.0
R_SPREAD = 300.0
WALL_LANES = (0, 1, 2)
CLOSING_LANE, PAIR_LANE = 0, 1
LANE_W = 3.2                                  # nm2_analyze V2V distance convention
WINDOWS = ((450, 550), (800, 900))
EVERY = 10
AMAX = 1.0
VARIANTS_B = {
    'pair_g': ('closing_pair', dict(v_w=22.0, hold_s=120, dmax=1.0)),
    'pair_h': ('closing_pair', dict(v_w=15.0, hold_s=90, dmax=1.5)),
    'pair_t': ('closing_pair', dict(v_w=22.0, hold_s=150, dmax=1.0, anchor_release_x=X_B-150.0)),
    'wall_g': ('abreast_wall', dict(v_w=22.0, hold_s=120, dmax=1.0)),
    'wall_h': ('abreast_wall', dict(v_w=15.0, hold_s=90, dmax=1.5)),
}
R_GRID = dict(v_w=(15.0, 18.0, 22.0), hold_s=(90, 120, 180), dmax=(1.0, 1.5))
N_R = 2
R_FAMILY = 'R_random'
FAMILIES = ('closing_pair', 'abreast_wall', R_FAMILY)

# ---- runtime conventions (mining collect2 / probe / wallmine) --------------------------------------
EPISODE_S = 1200
WARMUP_S = 300
PROFILE_S = 180                               # longest registered hold (R grid 180 s; pair_t <= 150 s)
TAIL_S = 120                                  # sv.TAIL_S; never shortened
DEFAULT_SEEDS = 3                             # CRN seeds 1..3 (confirm_all8 extra_seeds = 1)
MATERIAL_VEH_S = 1.0
NOISE_OFFSET = 9000                           # new noise streams (probe 7000, wallmine 8000)
COLLECT_SIM_S = 1200
TRAIN_RANGE = (989200, 989599)
DEMANDS = (2400, 2800, 3200)
SHARES = (0.2, 0.4, 0.6)
FORBIDDEN_SEED_RANGES = ((962000, 962999), (964000, 964199), (977000, 977999), (989600, 989699))
OBSERVABLE_KEYS = ('vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m')
PROVISIONAL = 'clean4_provisional_open_loop_v1'
BASES = ('frozen_u0', 'native')

SELECTION_RULE = ('NEWMAP2 geometry rule (never outcome): zone x in [X_B-1800, X_B-400] on edge up (X_B=2696 '
                  'lane-drop of lane 0); closing-lane pair = lane-0 CAV + lane-1 CAV |dx|<=100 m (min dx, then '
                  'most downstream); abreast wall = one CAV per lane 0/1/2, spread <= 200 m (min spread, then most '
                  'downstream); per window the earliest mark with pair AND wall, else pair, else wall; marks '
                  'every 10 s in [450,550] and [800,900]; observable CAV rows only')
SCREEN_RULE = ('seed 1 = ONE sv.verify (policy + groups + every member solo, CRN of the collection episode); '
               'continue iff advantage LB95 > 0 and >= 1 veh-s, synergy (best solo J - group J) LB95 > 0 and '
               '>= 1 veh-s (bank_statistics._stats 30 s circular block, B=2000), collision and teleport '
               'deltas <= 0')
CONFIRM_RULE = ('probe_native.run path: first-seed sv.verify of the surviving groups + verify_perseed.confirm_all8 '
                '(seed 2 + per-seed extended later-harm windows + per-admitted solos + extra seeds 3..n), every '
                'verification precharged on this ledger; whole-stage upper-bound quote must fit the remaining cap')
EXTRACT_RULE = ('a (plan, seed) is mined iff admitted (own-seed LB95 > 0, safety, later harm) AND cooperative '
                '(own-seed synergy LB95 > 0) AND 0 collisions/teleports in its group window AND >= 2 members')
POTENTIAL_RULE = ('per training cell: screened_moments = B moments whose seed-1 screen ran (incl. no-candidate '
                  'and failed screens); robust_moments = B moments with >= 1 group mined on EVERY verified '
                  'seed (admission incl. later harm); b_examples = mined B (plan, seed) rows, reachable iff '
                  'max member distance at t0 (hypot(dx, dlane*3.2)) <= range_m')


class BudgetRefused(RuntimeError):
    """A stage the remaining cap cannot cover (refused before any dispatch)."""


# ============================================================================ small helpers

def _checked(path):
    from .protocol import checked
    return checked(Path(path))


def _digest(value):
    from .protocol import digest
    return digest(value)


def _write_new(path, value):
    from .exact_grade import _write_new as write_new
    write_new(path, value)


def _write_json(path, value):
    """Atomic overwrite of a DERIVED output (extract products only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(value if isinstance(value, str) else json.dumps(value, sort_keys=True, indent=1))
    os.replace(tmp, path)


def _append(root, name, rows):
    p = Path(root)/'results'/name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True)+'\n')
        f.flush()
        os.fsync(f.fileno())


def _jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []


def seed_forbidden(n):
    return type(n) is int and any(lo <= n <= hi for lo, hi in FORBIDDEN_SEED_RANGES)


def generator_seed(salt):
    """The CRN generator seed replay_bank.replicate derives from a salt."""
    return int(_digest(dict(replication=salt))[:15], 16)


def noise_seed(world, arm):
    n = 20261106+int(world['seed'])+(0 if arm == 'B' else 500000)+NOISE_OFFSET
    if seed_forbidden(n):
        raise ValueError(f'collection noise seed {n} falls in a forbidden evaluation range')
    return n


def cell_key(demand, penetration):
    return f'D{int(demand)}_p{int(round(float(penetration)*100)):02d}'


def check_world(world, role='mining'):
    """Registered wz3to2 TRAIN world (989200-989599), never an evaluation seed.

    Uses N1's ``map_scenarios.guard`` when it is importable (exact registry dict); until N1 lands
    the local registered-range/cell check applies (READY: world guard adapter pending N1)."""
    if not isinstance(world, dict) or world.get('map') != MAP:
        raise ValueError('not a wz3to2 world')
    s = world.get('seed')
    if type(s) is not int or seed_forbidden(s) or not TRAIN_RANGE[0] <= s <= TRAIN_RANGE[1]:
        raise ValueError(f'world seed {s!r} is not a wz3to2 TRAIN seed (989200-989599)')
    if world.get('demand') not in DEMANDS or world.get('penetration') not in SHARES:
        raise ValueError('world is not a registered wz3to2 cell')
    if world.get('cell_key') != cell_key(world['demand'], world['penetration']):
        raise ValueError('world cell_key does not match its demand/penetration')
    try:
        from . import map_scenarios
    except ImportError:
        return 'local_range_check (N1 map_scenarios pending)'
    map_scenarios.guard(world, role)
    return 'map_scenarios.guard'


def _stop(root):
    return (Path(root)/'STOP').exists()


# ============================================================================ 1. ledger

def _registration_path(root):
    return Path(root)/'REGISTRATION.json'


def init(root, cap=PHASE_CAP, *, base='frozen_u0', note=None):
    """Write the sealed REGISTRATION.json BEFORE any job (idempotent for identical cap/base)."""
    from .exact_grade import _seal
    root = _checked(root)
    if type(cap) is not int or not 0 < cap <= PHASE_CAP:
        raise ValueError(f'cap must be an integer in (0, {PHASE_CAP}]')
    if base not in BASES:
        raise ValueError(f'base must be one of {BASES}')
    path = _registration_path(root)
    if path.exists():
        reg = registration(root)
        if reg['cap_sim_s'] != cap or reg['base'] != base:
            raise ValueError('registration exists with a different cap/base; never rewritten')
        return reg
    if (root/'CHARGES.jsonl').exists():
        raise ValueError('charges exist without a registration; refusing')
    src = Path(__file__).read_bytes()
    body = dict(schema=SCHEMA+'.registration', phase=PHASE, cap_sim_s=cap, phase_cap_sim_s=PHASE_CAP,
                authority=AUTHORITY, base=base, map=MAP, train_range=list(TRAIN_RANGE),
                forbidden_seed_ranges=[list(r) for r in FORBIDDEN_SEED_RANGES], windows=[list(w) for w in WINDOWS],
                every_s=EVERY, profile_s=PROFILE_S, tail_s=TAIL_S, default_seeds=DEFAULT_SEEDS,
                noise_offset=NOISE_OFFSET, variants_b={k: dict(family=f, **p) for k, (f, p) in VARIANTS_B.items()},
                r_procedure=dict(n=N_R, grid={k: list(v) for k, v in R_GRID.items()}, spread_m=R_SPREAD,
                                 seed='clean4_R_v1:{state_id}:{i}', family=R_FAMILY),
                rules=dict(selection=SELECTION_RULE, screen=SCREEN_RULE, confirmation=CONFIRM_RULE,
                           extract=EXTRACT_RULE, potential=POTENTIAL_RULE,
                           budget='precharge before physics; duplicate ids refused; never replayed; hard cap'),
                code=dict(module='tools/coopmwm/clean4_verify.py', sha256=hashlib.sha256(src).hexdigest()),
                note=note, unix=time.time())
    _write_new(path, _seal(body, 'registration_sha256'))
    return registration(root)


def registration(root):
    from .exact_grade import verify_sealed
    path = _registration_path(root)
    if not path.exists():
        raise ValueError('clean4_prelaunch_verify not registered (run init first)')
    reg = json.loads(path.read_text(encoding='utf-8'))
    verify_sealed(reg, 'registration_sha256')
    if reg.get('phase') != PHASE or not 0 < int(reg.get('cap_sim_s', 0)) <= PHASE_CAP:
        raise ValueError('registration is not a valid clean4_prelaunch_verify registration')
    return reg


@contextmanager
def _locked(root):
    try:
        import fcntl
    except ModuleNotFoundError:          # local Windows CPU tests only
        from .fcntl_compat import install
        install()
        import fcntl
    with (Path(root)/'LEDGER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def ledger_rows(root):
    return _jsonl(Path(root)/'CHARGES.jsonl')


def spent(root):
    return sum(int(r['sim_s']) for r in ledger_rows(root))


def remaining(root):
    return int(registration(root)['cap_sim_s'])-spent(root)


def charge(root, ident, sim_s, *, stage='', branches=1):
    """Durable atomic precharge on this phase's own ledger: refuses unregistered roots, duplicate ids
    (never replayed) and anything over the registered cap."""
    from .exact_grade import BudgetBlocked
    root = Path(root)
    cap = int(registration(root)['cap_sim_s'])
    if type(sim_s) is not int or sim_s <= 0:
        raise ValueError('charge must be a positive integer of sim-s')
    if not isinstance(ident, str) or not ident:
        raise ValueError('charge id required')
    with _locked(root):
        rows = ledger_rows(root)
        if any(r['id'] == ident for r in rows):
            raise BudgetBlocked('already charged; never replay '+ident)
        used = sum(int(r['sim_s']) for r in rows)
        if used+sim_s > cap:
            raise BudgetBlocked(f'{PHASE} cap {cap} reached (spent {used}, asked {sim_s})')
        with (root/'CHARGES.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(id=ident, phase=PHASE, stage=stage, sim_s=sim_s, branches=int(branches),
                                    unix=time.time()), sort_keys=True)+'\n')
            f.flush()
            os.fsync(f.fileno())


def _charger(root, prefix, stage):
    """sv.verify charge hook: sum(spec sim_s) of the WHOLE branch set, before dispatch."""
    def chg(vid, specs):
        charge(root, f'{prefix}_{vid}', sum(int(s['sim_s']) for s in specs), stage=stage, branches=len(specs))
    return chg


# ============================================================================ 2. selection (geometry only)

class Wz3to2GeometryStub:
    """Geometry adapter protocol (N1 map descriptor API, NOT READY): ``locate(row)`` maps ONE observable
    CAV row (keys OBSERVABLE_KEYS, runtime WM frame) to the NEWMAP2 frame ``(edge, lane, x)``: edge
    'up' upstream of the drop, lane = SUMO lane index on 'up' (0 = the closing lane), x = global x in
    metres (X_B = 2696 at the drop); None when the row is not on a NEWMAP2 edge. Expected N1 mapping
    (map_descriptor.load): WM lane 2 = closing lane, s = 75.44 at x = 2696."""

    name = 'wz3to2_geometry_stub'
    x_drop = X_B

    def locate(self, row):
        raise NotImplementedError('wz3to2 geometry adapter pending N1 map descriptor API (c4/nm-abstract)')


def geometry():
    return Wz3to2GeometryStub()


def observable(row):
    """Projection to the observable CAV/vehicle row keys; nothing else is ever read."""
    return {k: row[k] for k in OBSERVABLE_KEYS if k in row}


def zone_cavs(rows, geom):
    out = []
    for r in rows:
        if not r.get('is_cav'):
            continue
        loc = geom.locate(r)
        if loc is None:
            continue
        edge, lane, x = loc
        if edge == 'up' and ZONE[0] <= float(x) <= ZONE[1]:
            out.append(dict(id=r['vid'], lane=int(lane), x=round(float(x), 2), v=round(float(r['v_mps']), 3),
                            s_m=float(r['s_m']), wm_lane=int(r['lane'])))
    return out


def pick_pair(z):
    best = None
    for c0 in (c for c in z if c['lane'] == CLOSING_LANE):
        for c1 in (c for c in z if c['lane'] == PAIR_LANE):
            dx = abs(c0['x']-c1['x'])
            if dx <= PAIR_DX:
                key = (dx, -c0['x'])
                if best is None or key < best[0]:
                    best = (key, [c0, c1])
    return None if best is None else best[1]


def pick_wall(z):
    lanes = [[c for c in z if c['lane'] == ln] for ln in WALL_LANES]
    if any(not x for x in lanes):
        return None
    best = None
    for combo in itertools.product(*lanes):
        xs = [c['x'] for c in combo]
        sp = max(xs)-min(xs)
        if sp <= WALL_SPREAD:
            key = (sp, -max(xs))
            if best is None or key < best[0]:
                best = (key, list(combo))
    return None if best is None else best[1]


def n3_select(rows, t, *, geom):
    """Default observable selector: the NEWMAP2 rule at one mark. rank 0 = pair AND wall, 1 = pair,
    2 = wall. ``rows`` are observable projections; no outcome or hidden field is accessible."""
    z = zone_cavs(rows, geom)
    pair, wall = pick_pair(z), pick_wall(z)
    if not pair and not wall:
        return None
    rank = 0 if pair and wall else (1 if pair else 2)

    def ids(xs):
        return [c['id'] for c in xs] if xs else None
    members = list(dict.fromkeys((ids(pair) or [])+(ids(wall) or [])))
    return dict(t=int(t), rank=rank, kind=('pair+wall', 'pair', 'wall')[rank], pair=ids(pair), wall=ids(wall),
                members=members, zone=z, rule=SELECTION_RULE)


def selector(geom):
    return lambda rows, t: n3_select(rows, t, geom=geom)


class WindowKeeper:
    """Per window: the earliest moment of the best rank (0 pair+wall < 1 pair < 2 wall). Online: a
    snapshot is captured only when the moment could still be the window's final choice."""

    def __init__(self, windows=WINDOWS, every=EVERY):
        self.windows = [tuple(int(x) for x in w) for w in windows]
        self.every = int(every)
        if self.every <= 0 or any(not WARMUP_S <= a <= b < EPISODE_S for a, b in self.windows):
            raise ValueError('windows must lie inside the controlled episode')
        self.kept = {i: {} for i in range(len(self.windows))}

    def window_of(self, t):
        for i, (a, b) in enumerate(self.windows):
            if a <= t <= b and (t-a) % self.every == 0:
                return i
        return None

    def wants(self, wi, rank):
        return not any(r <= rank for r in self.kept[wi])

    def keep(self, wi, t, moment, snap):
        rank = int(moment.get('rank', 0))
        dropped = [x[0] for r, x in self.kept[wi].items() if r > rank]
        self.kept[wi] = {r: x for r, x in self.kept[wi].items() if r < rank}
        self.kept[wi][rank] = (int(t), moment, snap)
        return dropped

    def final(self):
        return {wi: kept[min(kept)] for wi, kept in self.kept.items() if kept}


# ============================================================================ runtime binding (stub)

def job_fields(*, ident, path, world, arm, weights, noise_seed, runtime, checkpoint, sim_s=COLLECT_SIM_S):
    """The mining collect2 base-policy job (plain episode: no guide, no intervention)."""
    return dict(id=ident, path=str(path), world=world, role='train', arm=arm, update=0, training_seed=20261101,
                weights=weights, runtime=runtime, checkpoint=checkpoint, noise_seed=noise_seed,
                deterministic=False, guided=False, b_replay=True, v2v=True, health_guards=False,
                sim_s=sim_s, stop=sim_s, warm_history=True, accel_bounds=[-3., 2.6])


class Wz3to2RuntimeStub:
    """Runtime binding protocol (N1/N4, NOT READY): ``base`` ('frozen_u0' | 'native'),
    ``job(ident=, path=, world=, arm=, weights=, noise_seed=, sim_s=) -> job dict`` (normally
    ``job_fields`` + the wz3to2 runtime/checkpoint/map binding) and optionally
    ``episode(job, controller_hook=) -> result`` (default accel_plant.episode). The bound runtime must
    (a) build the wz3to2 scenario (N1 map_scenarios), (b) guard wz3to2 worlds in
    accel_plant.bind_runtime instead of the merge registry, (c) run controller hooks also for a
    native base, and (d) bind the same map in the sumo_verify branch workers."""

    name = 'wz3to2_runtime_stub'
    base = 'frozen_u0'

    def job(self, **kw):
        raise NotImplementedError('wz3to2 coopmwm runtime binding pending N1/N4 (scenario, world guard, '
                                  'branch-worker map binding)')


def runtime():
    return Wz3to2RuntimeStub()


# ============================================================================ leads (N3 / N8 moments)

def window_index(t0, windows=WINDOWS):
    return next((i for i, (a, b) in enumerate(windows) if a <= int(t0) <= b), None)


def load_leads(paths):
    """N3/N8 admitted moments: NM2_SUMMARY.json (admitted verdicts) or mined-move JSONL rows."""
    out = []
    for path in paths:
        p = _checked(path)
        if p.suffix == '.jsonl':
            for r in _jsonl(p):
                w = r.get('world') or {}
                out.append(dict(id=r['situation_id'], seed=int(w['seed']), cell_key=w.get('cell') or r.get('cell'),
                                t0=int(r['t0']), window=window_index(r['t0']), arm=r.get('arm'),
                                family=r.get('family'), variant=r.get('variant'), source=str(p)))
        else:
            for v in json.loads(p.read_text(encoding='utf-8')).get('verdicts', []):
                if v.get('admitted'):
                    out.append(dict(id=v['sid'], seed=int(v['seed']), cell_key=v['cell'], t0=int(v['t0']),
                                    window=v.get('window', window_index(v['t0'])), arm=v.get('arm'),
                                    family=v.get('family'), variant=v.get('variant'), source=str(p)))
    seen, uniq = set(), []
    for x in out:
        key = (x['id'], x.get('variant'), x.get('arm'))
        if key not in seen:
            seen.add(key)
            uniq.append(x)
    return uniq


def match_lead(world, wi, leads):
    """N3 moment ids in the same world (seed, cell) and window (the t0 may differ: different base)."""
    hits = [x for x in leads or () if x['seed'] == world['seed'] and x['cell_key'] == world['cell_key']
            and x['window'] == wi]
    ids = list(dict.fromkeys(x['id'] for x in hits))
    return (ids[0] if ids else None), hits


# ============================================================================ 2. collect

def _reference(vid, members, action, commands):
    """Live per-second policy reference (raw command + sampled residual; base command outside the
    policy context; None when the CAV is not commanded)."""
    if vid in members and vid in commands:
        return float(commands[vid])+float(action[members.index(vid)][0])
    if vid in commands:
        return float(commands[vid])
    return None


def _track(row):
    return None if row is None else [float(row['s_m']), int(row['lane']), float(row['v_mps'])]


class _Capture:
    """accel_plant controller hook (mining collect2 Capture): per-second references/tracks and the
    window moments; reads only the controller event's observable rows, commands and sample."""

    def __init__(self, select, keeper, capture_fn, profile_s=PROFILE_S):
        self.select, self.keeper, self.capture_fn, self.profile_s = select, keeper, capture_fn, int(profile_s)
        self.recording = {}
        self.marks = 0

    def __call__(self, controller, job, prefixes):
        self.c, self.job, self.prefixes = controller, job, prefixes

    def after_step(self, t, after):
        e = self.c.events[-1]
        t = int(e['time_s'])
        members = list(e['sample']['context']['members'])
        action = e['sample']['action']
        commands = e.get('commands') or {}
        rows = [observable(r) for r in e['rows']]
        by_vid = {r['vid']: r for r in rows if 'vid' in r}
        for t0, rec in self.recording.items():
            if 0 < t-t0 < self.profile_s:
                for vid in rec['vids']:
                    rec['policy_profile'][vid].append(_reference(vid, members, action, commands))
                    rec['cav_track'][vid].append(_track(by_vid.get(vid)))
        wi = self.keeper.window_of(t)
        if wi is None:
            return
        self.marks += 1
        moment = self.select(rows, t)
        if moment is None or not self.keeper.wants(wi, int(moment.get('rank', 0))):
            return
        snap = self.capture_fn(self.c, self.prefixes, self.job, t, e['sample'], self.c.generator)
        for d in self.keeper.keep(wi, t, moment, snap):
            self.recording.pop(d, None)
        vids = [c['id'] for c in moment.get('zone') or []] or list(moment.get('members') or [])
        self.recording[t] = dict(
            vids=vids, rows_t0=rows,
            policy_profile={v: [_reference(v, members, action, commands)] for v in vids},
            cav_track={v: [_track(by_vid.get(v))] for v in vids})


def collect(root, world, arm, weights, *, select=None, windows=WINDOWS, every=EVERY, geom=None, rt=None,
            leads=None, capture_fn=None, snapshot_sha=None, profile_s=PROFILE_S, weights_sha256=None):
    """ONE base-policy episode for (world, arm), charged BEFORE physics; saves the kept moments."""
    import torch
    from . import sumo_verify as sv
    root = _checked(root)
    reg = registration(root)
    guard = check_world(world)
    if arm not in ('B', 'R'):
        raise ValueError('arm must be B or R')
    rt = rt or runtime()
    if getattr(rt, 'base', reg['base']) != reg['base']:
        raise ValueError(f'runtime base {rt.base!r} differs from the registered base {reg["base"]!r}')
    if select is None:
        geom = geom or geometry()
        geom.locate(dict(vid='__preflight__', s_m=0.0, v_mps=0.0, lane=0, is_cav=True, length_m=5.0))
        select = selector(geom)
    keeper = WindowKeeper(windows, every)
    ident = f'c4v_{arm}_{world["cell_key"]}_s{world["seed"]}'
    work = root/'collect'/ident
    ns = noise_seed(world, arm)
    job = rt.job(ident=ident, path=work, world=world, arm=arm, weights=weights, noise_seed=ns, sim_s=COLLECT_SIM_S)
    if job.get('id') != ident or int(job.get('sim_s', 0)) != COLLECT_SIM_S or job.get('noise_seed') != ns:
        raise ValueError('runtime job does not carry the registered id/sim_s/noise seed')
    if (root/'STOP').exists():
        raise RuntimeError('STOP present')
    charge(root, 'collect_'+ident, COLLECT_SIM_S, stage='collect')          # BEFORE physics
    work.mkdir(parents=True, exist_ok=False)
    hook = _Capture(select, keeper, capture_fn or sv.capture, profile_s)
    episode = getattr(rt, 'episode', None)
    if episode is None:
        from . import accel_plant
        episode = accel_plant.episode
    result = episode(job, controller_hook=hook)
    sha = snapshot_sha or sv.snapshot_sha
    saved, windows_out = [], []
    final = keeper.final()
    for wi in range(len(keeper.windows)):
        if wi not in final:
            windows_out.append(dict(window=wi, t0=None))
            continue
        t0, moment, snap = final[wi]
        rec = hook.recording[t0]
        sid = f'{ident}_t{t0}'
        lead, hits = match_lead(world, wi, leads)
        d = work/f'ctx_t{t0}'
        d.mkdir()
        torch.save(snap, d/'SNAPSHOT.pt')
        context = dict(schema=SCHEMA+'.context', phase=PHASE, state_id=sid, world=world, arm=arm, t0=t0,
                       window=wi, base=reg['base'], moment=moment, members=moment.get('members'),
                       geometry=dict(pair=moment.get('pair'), wall=moment.get('wall'), zone=moment.get('zone')),
                       selection=SELECTION_RULE, snapshot_sha256=sha(snap), policy_profile=rec['policy_profile'],
                       cav_track=rec['cav_track'], rows_t0=rec['rows_t0'], profile_s=profile_s, lead=lead,
                       lead_moments=hits, world_guard=guard, noise_seed=ns)
        _write_new(d/'CONTEXT.json', context)
        saved.append(str(d))
        windows_out.append(dict(window=wi, t0=t0, kind=moment.get('kind'), dir=str(d), lead=lead))
    _write_new(work/'COLLECTION_RECEIPT.json', dict(
        phase=PHASE, id=ident, world=world, arm=arm, base=reg['base'], noise_seed=ns, charged_sim_s=COLLECT_SIM_S,
        actual_sim_s=(result or {}).get('actual_sim_s'), windows=windows_out, marks=hook.marks,
        weights_sha256=weights_sha256, unix=time.time()))
    return saved


# ============================================================================ 3. plans

def r_proposals(ctx, n=N_R):
    """N3 R procedure (seeded per state id): 2-3 random zone CAVs, spread <= 300 m, never a B set."""
    z = sorted((ctx.get('geometry') or {}).get('zone') or [], key=lambda c: c['id'])
    geo = ctx.get('geometry') or {}
    excl = {frozenset(x) for x in (geo.get('pair'), geo.get('wall')) if x}
    out = []
    for i in range(n):
        rng = random.Random(f'clean4_R_v1:{ctx["state_id"]}:{i}')
        k = rng.choice([2, 3])
        prop = None
        for kk in (k, 2):
            if len(z) < kk:
                continue
            for _ in range(400):
                mem = rng.sample(z, kk)
                xs = [c['x'] for c in mem]
                ids = frozenset(c['id'] for c in mem)
                if max(xs)-min(xs) <= R_SPREAD and ids not in excl:
                    prop = mem
                    break
            if prop:
                break
        if not prop:
            continue
        excl.add(frozenset(c['id'] for c in prop))
        params = dict(v_w=rng.choice(R_GRID['v_w']), hold_s=rng.choice(R_GRID['hold_s']),
                      dmax=rng.choice(R_GRID['dmax']))
        out.append(dict(name=f'R{i}', family=R_FAMILY, members=[c['id'] for c in prop],
                        params=dict(params, amax=AMAX, release_x=X_B)))
    return out


def variants_for(ctx):
    """Registered variants of the context's arm (B: N3 B variants; R: R procedure)."""
    if ctx['arm'] == 'R':
        return r_proposals(ctx)
    geo = ctx.get('geometry') or {}
    out = []
    for name, (family, prm) in VARIANTS_B.items():
        members = geo.get('pair') if family == 'closing_pair' else geo.get('wall')
        if not members:
            continue
        params = dict(prm, amax=AMAX, release_x=X_B)
        if 'anchor_release_x' in params:
            params['anchor'] = members[0]            # the closing-lane CAV
        out.append(dict(name=name, family=family, members=list(members), params=params))
    return out


def _policy_rows(ctx, vid, duration):
    """Seed-1 policy reference row (the collection episode = the branch CRN). None seconds are
    allowed only after the CAV left the observed rows (inert placeholder, never applied)."""
    prof = (ctx.get('policy_profile') or {}).get(vid)
    track = (ctx.get('cav_track') or {}).get(vid) or []
    if prof is None or len(prof) < duration:
        return None
    out = []
    for k in range(duration):
        x = prof[k]
        if x is None:
            present = k < len(track) and track[k] is not None
            if present or not out:
                return None
            x = out[-1]
        out.append(float(x))
    return out


def provisional_builder(ctx, members, variant):
    """PROVISIONAL (until the N2 move library on c4/nm-moves provides the real builder).

    Speed-target variant -> open-loop per-second acceleration on top of the seed-1 policy reference:
    each member's NEWMAP2-frame (x, v) at t0 is propagated under the N3 law
    a = clip((v_w - v)/1 s, -dmax, +amax) for hold_s seconds; a member is released (offset 0, policy
    reference) once its predicted x reaches release_x (the drop) or, for pair_t, once the anchor
    (closing-lane CAV) reaches anchor_release_x; afterwards the offset is zero. duration_s = hold_s.
    Returns the probe.ease_plan shape (id, family, members, duration_s, policy_profile, profile,
    perturbation_profile) or None when a member has no complete reference."""
    p = variant['params']
    duration = int(p['hold_s'])
    zone = {c['id']: c for c in (ctx.get('geometry') or {}).get('zone') or []}
    if len(members) < 2 or len(set(members)) != len(members) or any(m not in zone for m in members):
        return None
    pol = [_policy_rows(ctx, m, duration) for m in members]
    if any(r is None for r in pol):
        return None
    v_w, dmax, amax = float(p['v_w']), float(p['dmax']), float(p.get('amax', AMAX))
    release_x = float(p.get('release_x', X_B))
    anchor, anchor_x = p.get('anchor'), p.get('anchor_release_x')
    if anchor is not None and anchor not in members:
        return None
    x = {m: float(zone[m]['x']) for m in members}
    v = {m: float(zone[m]['v']) for m in members}
    rel = {m: None for m in members}
    prof = [list(r) for r in pol]
    for k in range(duration):
        anchor_out = anchor is not None and x[anchor] >= float(anchor_x)
        acc = {}
        for i, m in enumerate(members):
            if rel[m] is not None:
                continue
            if x[m] >= release_x or anchor_out:
                rel[m] = k
                continue
            a = min(amax, max(-dmax, (v_w-v[m])/1.0))
            prof[i][k] = acc[m] = a
        for m, a in acc.items():
            v_new = max(0.0, v[m]+a)
            x[m] += 0.5*(v[m]+v_new)
            v[m] = v_new
        if all(r is not None for r in rel.values()):
            break
    pert = [[round(a-b, 12) for a, b in zip(pr, po)] for pr, po in zip(prof, pol)]
    return dict(id=f'c4v_{variant["name"]}_{"_".join(members)}', family=variant['family'], members=list(members),
                duration_s=duration, origin=PHASE, builder=PROVISIONAL, provisional=True,
                variant=dict(name=variant['name'], params=dict(p)), policy_profile=pol, profile=prof,
                perturbation_profile=pert, release_s=rel)


def solos(group):
    from . import replay_bank as rb
    return [rb.solo_candidate(group, m) for m in group['members']]


def max_member_distance(ctx, members):
    zone = {c['id']: c for c in (ctx.get('geometry') or {}).get('zone') or []}
    pts = [(zone[m]['x'], zone[m]['lane']) for m in members if m in zone]
    if len(pts) != len(members):
        return None
    return max((math.hypot(a[0]-b[0], (a[1]-b[1])*LANE_W) for a, b in itertools.combinations(pts, 2)), default=0.0)


def plans(ctx, variants=None, *, builder=provisional_builder):
    """Executable group plans of one moment + member solos. Builder protocol:
    ``builder(ctx, members, variant) -> plan dict | None`` (id, family, members, duration_s,
    policy_profile, profile, perturbation_profile; members x duration matrices)."""
    variants = variants_for(ctx) if variants is None else variants
    groups, meta, skipped, out_solos = [], {}, [], {}
    for var in variants:
        plan = builder(ctx, list(var['members']), var)
        if plan is None:
            skipped.append(dict(variant=var['name'], members=list(var['members']),
                                reason='builder returned None (incomplete reference or geometry)'))
            continue
        if plan['id'] in meta:
            raise ValueError('duplicate plan id '+plan['id'])
        if plan.get('family') not in FAMILIES or list(plan['members']) != list(var['members']):
            raise ValueError('builder changed the family/members of a registered variant')
        groups.append(plan)
        out_solos[plan['id']] = solos(plan)
        meta[plan['id']] = dict(variant=var['name'], family=plan['family'], params=var['params'],
                                members=list(plan['members']),
                                max_member_dist_t0_m=max_member_distance(ctx, plan['members']),
                                builder=plan.get('builder', getattr(builder, '__name__', str(builder))),
                                provisional=bool(plan.get('provisional')))
    return dict(groups=groups, solos=out_solos, meta=meta, skipped=skipped)


# ============================================================================ 4. verify

def _runner_factory(vroot, workers):
    from . import sumo_verify as sv
    return sv.SubprocessBranchRunner(Path(vroot)/'branches', workers=workers, timeout_s=3600)


def screen_quote(t0, cands, tail_s=TAIL_S, end_s=EPISODE_S):
    from . import sumo_verify as sv
    return (1+len(cands))*sv.branch_sim_s(sv.window_end(t0, cands, end_s, tail_s))


def confirm_quote(t0, plans_, extra_seeds, tail_s=TAIL_S, end_s=EPISODE_S):
    """Upper bound of the whole native confirmation stage: first seed + every candidate admitted on
    seeds 1/2 + every candidate cooperative on every extra seed (verify_perseed.quote_decision)."""
    from . import sumo_verify as sv, verify_perseed as vp
    durations = [int(p['duration_s']) for p in plans_]
    members = [list(p['members']) for p in plans_]
    idx = list(range(len(plans_)))
    q = vp.quote_decision(int(t0), durations, members, admitted={1: idx, 2: idx}, end_s=end_s,
                          extra_seeds=int(extra_seeds), cooperative=idx)
    first = (len(plans_)+1)*sv.branch_sim_s(sv.window_end(t0, plans_, end_s, tail_s))
    return first+q['all8_confirmation_sim_s']+int(q.get('extra_seeds_sim_s', 0))


def quote_moment(ctx, groups, *, seeds=DEFAULT_SEEDS, tail_s=TAIL_S, end_s=EPISODE_S):
    """Zero-physics planning quote of one moment: the seed-1 screen (groups + member solos) and the
    confirmation upper bound if ONE / ALL groups survive the screen."""
    t0 = int(ctx['t0'])
    cands = list(groups)+[s for g in groups for s in solos(g)]
    one = max((confirm_quote(t0, [g], seeds-2, tail_s, end_s) for g in groups), default=0)
    return dict(t0=t0, n_groups=len(groups), n_solos=len(cands)-len(groups),
                screen_sim_s=screen_quote(t0, cands, tail_s, end_s) if groups else 0,
                confirm_one_max_sim_s=one,
                confirm_all_sim_s=confirm_quote(t0, groups, seeds-2, tail_s, end_s) if groups else 0)


def check_salts(episode_id, t0, plans_, extra_seeds):
    """Seed-2 salt (the first-seed verification id) and extra-seed salts must never derive a CRN
    generator seed in a forbidden evaluation range (incl. wz3to2 held-out 989600-989699)."""
    from . import sumo_verify as sv, verify_perseed as vp
    vid = sv.verification_id(episode_id, t0, plans_)
    salts = {2: vid}
    for k in vp.extra_seed_indices(extra_seeds):
        salts[k] = vp.extra_seed_salt(vid, k)
    bad = {k: s for k, s in salts.items() if seed_forbidden(generator_seed(s))}
    if bad:
        raise ValueError(f'CRN salt(s) derive a forbidden evaluation seed: {sorted(bad)}')
    return vid, salts


def _screen_rows(rec, groups, owner, bdir, base):
    from . import bank_statistics as bs
    from . import replay_bank as rb
    br = rec['branches']
    idx = {c: i+1 for i, c in enumerate(rec['candidate_ids'])}
    tapes = {}

    def tape(i):
        if i not in tapes:
            tapes[i] = bs._read_result(bdir, rec['verification_id'], br[i])
        return tapes[i]
    Jp = rec['J_policy_veh_s']
    rows = []
    for g in groups:
        gi = idx[g['id']]
        Jg = br[gi]['J_veh_s']
        sj = {s: br[idx[s]]['J_veh_s'] for s in owner[g['id']]}
        row = dict(base, group_id=g['id'], plan_sha256=_digest(g), family=g['family'], members=g['members'],
                   n_members=len(g['members']), J_policy=Jp, J_group=Jg, J_solos=sj,
                   group_adv=None if Jg is None else Jp-Jg,
                   safety={k: (None if Jg is None else float(br[gi]['window'][k])-float(br[0]['window'][k]))
                           for k in rb.SAFETY_KEYS})
        if Jg is None or any(x is None for x in sj.values()):
            row.update(screen='incomplete', passed=False, reason='group or solo branch ungraded; never imputed')
            rows.append(row)
            continue
        best = min(sj, key=lambda k: (sj[k], k))
        row.update(best_solo=best, synergy=sj[best]-Jg)
        try:
            adv = bs._stats([(tape(0), tape(gi), rec['window'])])
            syn = bs._stats([(tape(idx[best]), tape(gi), rec['window'])])
        except Exception as e:      # missing/tampered tape: never imputed
            row.update(screen='data_missing', passed=False, reason=repr(e))
            rows.append(row)
            continue
        row.update(adv_lb95=adv['lower95_one_sided_veh_s'], synergy_lb95=syn['lower95_one_sided_veh_s'],
                   adv_stats={k: v for k, v in adv.items() if not isinstance(v, list)},
                   synergy_stats={k: v for k, v in syn.items() if not isinstance(v, list)})
        safe = all((row['safety'].get(k) or 0) <= 0 for k in ('collisions', 'teleports'))
        checks = [(row['adv_lb95'] > 0, 'advantage LB95 <= 0'), (row['group_adv'] >= MATERIAL_VEH_S, 'advantage < 1 veh-s'),
                  (row['synergy_lb95'] > 0, 'synergy LB95 <= 0'), (row['synergy'] >= MATERIAL_VEH_S, 'synergy < 1 veh-s'),
                  (safe, 'collision/teleport delta > 0')]
        fails = [msg for ok, msg in checks if not ok]
        row.update(screen='pass' if not fails else 'fail', passed=not fails, reason='; '.join(fails))
        rows.append(row)
    return rows


@contextmanager
def _confirm_charge_bound(root, tag):
    """Scoped probe_native binding: verify_perseed's confirmation verifications charge this ledger
    (kind-prefixed ids); the original replay_bank._confirm_charge is always restored."""
    from . import replay_bank as rb
    original = rb._confirm_charge
    rb._confirm_charge = lambda run_root, phase, kind: _charger(root, f'{tag}_{kind}', kind)
    try:
        yield
    finally:
        rb._confirm_charge = original


def bank_dir(root):
    return Path(root)/'minebank'/'bank'


def _native(root, ctx, snap, passing, extra, workers, runner_factory, tail_s, health):
    """probe_native.run: first-seed sv.verify of the survivors + verify_perseed.confirm_all8."""
    import torch
    from . import replay_bank as rb, sumo_verify as sv, verify_perseed as vp
    from .storage import write as swrite
    arm, sid = ctx['arm'], ctx['state_id']
    BANK = bank_dir(root)
    bank = Path(rb.bank_root(BANK, arm))
    (bank/'snapshots').mkdir(parents=True, exist_ok=True)
    if not (bank/'snapshots'/(sid+'.pt')).exists():
        torch.save(snap, bank/'snapshots'/(sid+'.pt'))
        swrite(bank/'snapshots'/(sid+'.json'), dict(
            state_id=sid, arm=arm, job=sid.rsplit('_t', 1)[0], world=ctx['world'], t0=int(ctx['t0']),
            group=dict(members=ctx.get('members')), regime=None, policy_update=0,
            record_sha256=ctx['snapshot_sha256'], unix=time.time(),
            mining=dict(phase=PHASE, selection=SELECTION_RULE, moment=(ctx.get('moment') or {}).get('kind'),
                        lead=ctx.get('lead'), base=ctx.get('base'), map=MAP)))
    tag = f'nat_{sid}'
    r = sv.verify(snap, passing, runner_factory(bank/'sumo_verification', workers), _charger(root, f'{tag}_first', 'first'),
                  'wm', workers=workers, root=bank/'sumo_verification', tail_s=tail_s)
    _write_new(bank/'decisions'/(sid+'.json'), dict(state_id=sid, verification_id=r['verification_id'],
               status=r['status'], policy_update=0, arm=arm, available_k=len(passing), issued_unix=time.time(),
               matched_B=None, phase=PHASE))
    out = dict(state_id=sid, tag=tag, verification_id=r['verification_id'], first_status=r['status'])
    if r['status'] not in ('accepted', 'rejected'):
        out.update(status='first_'+r['status'], reason=r.get('reason'))
        return out
    receipt = dict(r, _branches=str(bank/'sumo_verification'/'branches'), _policy_version='u0')
    with _confirm_charge_bound(root, tag):
        conf = vp.confirm_all8(BANK, arm, receipt, snap, lambda work: runner_factory(work, workers), str(BANK), PHASE,
                               workers=workers, version='u0', candidates=passing, slim_rejected_tapes=False,
                               extra_seeds=int(extra), health=health)
    out.update(status='confirmed', admitted=conf['admitted'], confirmation_sha256=conf['confirmation_sha256'],
               charged_sim_s=conf.get('charged_sim_s'))
    return out


def load_context(ctx_dir):
    import torch
    from . import sumo_verify as sv
    ctx_dir = _checked(ctx_dir)
    ctx = json.loads((ctx_dir/'CONTEXT.json').read_text(encoding='utf-8'))
    snap = torch.load(ctx_dir/'SNAPSHOT.pt', weights_only=False)
    if sv.snapshot_sha(snap) != ctx['snapshot_sha256']:
        raise ValueError('snapshot differs from its CONTEXT.json binding')
    return ctx, snap


def verify(root, ctx_dir, groups, *, seeds=DEFAULT_SEEDS, workers=4, runner_factory=None, tail_s=TAIL_S, health=None):
    """Staged, budget-aware verification of one moment (claimed once; never replayed)."""
    from . import sumo_verify as sv
    root = _checked(root)
    registration(root)
    if type(seeds) is not int or seeds < 2:
        raise ValueError('seeds must be an integer >= 2 (confirm_all8 always measures seeds 1 and 2)')
    if int(tail_s) < TAIL_S:
        raise ValueError('verification windows are never shortened below sv.TAIL_S')
    runner_factory = runner_factory or _runner_factory
    ctx, snap = load_context(ctx_dir)
    check_world(ctx['world'])
    if snap['episode']['world'] != ctx['world'] or int(snap['t0']) != int(ctx['t0']):
        raise ValueError('snapshot episode/t0 differ from the context')
    sid, t0, arm = ctx['state_id'], int(ctx['t0']), ctx['arm']
    base = dict(phase=PHASE, state_id=sid, ctx_dir=str(ctx_dir), arm=arm, world=ctx['world'],
                cell_key=ctx['world']['cell_key'], t0=t0, window=ctx.get('window'), lead=ctx.get('lead'),
                seed_index=1, unix=time.time())
    claim = root/'claims'/f'verify_{sid}.claim'
    claim.parent.mkdir(parents=True, exist_ok=True)

    def take_claim():
        try:
            with claim.open('x') as f:
                f.write(json.dumps(dict(pid=os.getpid(), unix=time.time())))
            return True
        except FileExistsError:
            return False
    if claim.exists():
        return dict(state_id=sid, status='skipped_claimed')
    if not groups:
        if not take_claim():
            return dict(state_id=sid, status='skipped_claimed')
        _append(root, 'contexts.jsonl', [dict(base, screen_status='no_candidates', n_groups=0, passing=[])])
        return dict(state_id=sid, status='no_candidates')
    ids = [g['id'] for g in groups]
    if len(set(ids)) != len(ids) or any(len(g['members']) < 2 for g in groups):
        raise ValueError('groups need distinct ids and >= 2 members')
    cands, owner = list(groups), {}
    for g in groups:
        for s in solos(g):
            if s['id'] not in {c['id'] for c in cands}:
                cands.append(s)
            owner.setdefault(g['id'], []).append(s['id'])
    quote = screen_quote(t0, cands, tail_s, int(snap.get('end_s', EPISODE_S)))
    left = remaining(root)
    if quote > left or _stop(root):
        # refused before any dispatch: no charge, no claim (a later run is not a replay)
        why = 'STOP' if _stop(root) else f'screen quote {quote} > remaining {left}'
        _append(root, 'contexts.jsonl', [dict(base, screen_status='refused', reason=why, n_groups=len(groups),
                                              screen_quote_sim_s=quote, passing=[])])
        return dict(state_id=sid, status='refused_screen', reason=why)
    if not take_claim():
        return dict(state_id=sid, status='skipped_claimed')
    work = root/'screen'/sid
    rec = sv.verify(snap, cands, runner_factory(work, workers), _charger(root, f'scr_{sid}', 'screen'), 'wm',
                    workers=workers, root=work, tail_s=tail_s)
    head = dict(base, verification_id=rec['verification_id'], status=rec['status'], window_s=rec.get('window'),
                charged_sim_s=rec.get('charged_sim_s'), screen_quote_sim_s=quote, work=str(work))
    if rec['status'] not in ('accepted', 'rejected'):
        _append(root, 'screen.jsonl', [dict(head, group_id=g['id'], plan_sha256=_digest(g), family=g['family'],
                                            members=g['members'], screen='incomplete', passed=False,
                                            reason=rec.get('reason')) for g in groups])
        _append(root, 'contexts.jsonl', [dict(base, screen_status=rec['status'], n_groups=len(groups), passing=[],
                                              verification_id=rec['verification_id'])])
        return dict(state_id=sid, status='screen_'+rec['status'])
    rows = _screen_rows(rec, groups, owner, work/'branches', head)
    _append(root, 'screen.jsonl', rows)
    passing = [g for g, r in zip(groups, rows) if r['passed']]
    ctx_row = dict(base, screen_status=rec['status'], verification_id=rec['verification_id'], n_groups=len(groups),
                   passing=[g['id'] for g in passing])
    _append(root, 'contexts.jsonl', [ctx_row])
    out = dict(state_id=sid, status='screened', screened=len(rows), passing=len(passing))
    if not passing:
        return out
    extra = seeds-2
    try:
        check_salts(snap['episode']['id'], t0, passing, extra)
    except ValueError as e:
        _append(root, 'native.jsonl', [dict(base, status='refused_salt', reason=str(e))])
        return dict(out, native='refused_salt')
    cq = confirm_quote(t0, passing, extra, tail_s, int(snap.get('end_s', EPISODE_S)))
    left = remaining(root)
    if cq > left or _stop(root):
        why = 'STOP' if _stop(root) else f'confirmation quote {cq} > remaining {left}'
        _append(root, 'native.jsonl', [dict(base, status='refused_budget', reason=why, confirm_quote_sim_s=cq,
                                            plan_ids=[g['id'] for g in passing])])
        return dict(out, native='refused_budget')

    def stop_health():
        if _stop(root):
            raise ValueError('clean4_prelaunch_verify STOP')
    res = _native(root, ctx, snap, passing, extra, workers, runner_factory, tail_s, health or stop_health)
    _append(root, 'native.jsonl', [dict(base, **res, confirm_quote_sim_s=cq, seeds=seeds,
                                        plan_ids=[g['id'] for g in passing])])
    return dict(out, native=res.get('status'), admitted=res.get('admitted'))


def run_context(root, ctx_dir, *, builder=provisional_builder, seeds=DEFAULT_SEEDS, workers=4, runner_factory=None,
                tail_s=TAIL_S, variants=None, names=None):
    """plans -> results/plans.jsonl (binding metadata for extract) -> verify. ``names`` restricts the
    registered slate (e.g. ('pair_g', 'wall_g')) to fit the cap; selection is by name, never outcome."""
    ctx, _ = load_context(ctx_dir)
    if names is not None:
        variants = [v for v in (variants_for(ctx) if variants is None else variants) if v['name'] in set(names)]
    built = plans(ctx, variants, builder=builder)
    rows = [dict(phase=PHASE, state_id=ctx['state_id'], ctx_dir=str(ctx_dir), arm=ctx['arm'], world=ctx['world'],
                 cell_key=ctx['world']['cell_key'], t0=ctx['t0'], window=ctx.get('window'), lead=ctx.get('lead'),
                 plan_id=g['id'], plan_sha256=_digest(g), plan=g, **built['meta'][g['id']]) for g in built['groups']]
    rows += [dict(phase=PHASE, state_id=ctx['state_id'], arm=ctx['arm'], skipped=True, **s) for s in built['skipped']]
    if rows and not (Path(root)/'claims'/f'verify_{ctx["state_id"]}.claim').exists():
        _append(root, 'plans.jsonl', rows)
    return verify(root, ctx_dir, built['groups'], seeds=seeds, workers=workers, runner_factory=runner_factory,
                  tail_s=tail_s)


# ============================================================================ 5. extract

def mined_ok(example, plan, families=FAMILIES):
    """probe_native.extract / wallmine rule for one per-seed all8 example."""
    syn = example.get('synergy') or {}
    win = (example.get('tape') or {}).get('branch', {}).get('window') or {}
    lb = syn.get('lower95_one_sided_veh_s')
    return bool(plan and plan.get('family') in families and len(plan['members']) >= 2 and example.get('admitted')
                and example.get('cooperative') is True and example.get('safety_ok') is not False
                and not win.get('collisions') and not win.get('teleports') and lb is not None and lb > 0)


def potential_inputs(contexts, mined, examples, *, range_m, cell=None):
    """Per training cell gate inputs (pure). contexts: results/contexts.jsonl rows; mined: mined rows
    (with arm, cell, situation_id, plan_sha256, seed, max_member_dist_t0_m); examples: every per-seed
    example row (situation_id, plan_sha256, seed, ok, n_reps)."""
    by_plan = {}
    for e in examples:
        by_plan.setdefault((e['arm'], e['situation_id'], e['plan_sha256']), []).append(e)
    robust_plans = {k for k, xs in by_plan.items()
                    if xs and all(x['ok'] for x in xs) and {int(x['seed']) for x in xs} == set(range(1, int(xs[0]['n_reps'])+1))}
    robust_sids = {(a, s) for a, s, _ in robust_plans}
    cells = {}
    final = {}
    for c in contexts:      # one status per moment: a later screen supersedes an earlier refusal
        key = (c['arm'], c['state_id'])
        if key not in final or final[key].get('screen_status') == 'refused':
            final[key] = c
    for key, c in final.items():
        d = cells.setdefault(c['cell_key'], dict(screened_moments=0, robust_moments=0, b_examples=[],
                                                 screened_with_lead=0, refused_or_unscreened=0,
                                                 r_screened_moments=0, r_robust_moments=0, r_examples=0))
        if c.get('screen_status') == 'refused':
            if c['arm'] == 'B':
                d['refused_or_unscreened'] += 1
            continue
        if c['arm'] == 'B':
            d['screened_moments'] += 1
            d['screened_with_lead'] += bool(c.get('lead'))
            d['robust_moments'] += key in robust_sids
        else:
            d['r_screened_moments'] += 1
            d['r_robust_moments'] += key in robust_sids
    for r in mined:
        d = cells.setdefault(r['cell'], dict(screened_moments=0, robust_moments=0, b_examples=[],
                                             screened_with_lead=0, refused_or_unscreened=0,
                                             r_screened_moments=0, r_robust_moments=0, r_examples=0))
        if r['arm'] != 'B':
            d['r_examples'] += 1
            continue
        dist = r.get('max_member_dist_t0_m')
        d['b_examples'].append(dict(id=f"{r['verification_id']}#{r['plan_sha256'][:16]}#s{r['seed']}",
                                    reachable=dist is not None and dist <= float(range_m), max_member_dist_t0_m=dist,
                                    robust=(r['arm'], r['situation_id'], r['plan_sha256']) in robust_plans))
    out = dict(schema=SCHEMA+'.potential_inputs', rule=POTENTIAL_RULE, range_m=float(range_m), cells=cells,
               denominator_note=('screened_moments counts every re-screened B moment of the cell; if collection was '
                                 'restricted to N3-lead windows the share is conditional on N3 admission '
                                 '(screened_with_lead reports how many)'))
    if cell is not None:
        out['chosen_cell'] = cell
        out['chosen'] = cells.get(cell, dict(screened_moments=0, robust_moments=0, b_examples=[]))
    return out


def extract(root, *, range_m=200.0, cell=None, families=FAMILIES):
    """Bank records -> CLEAN4_MOVES_B/R.jsonl, CLEAN4_NEGATIVES.jsonl, POTENTIAL_INPUTS.json."""
    from . import replay_bank as rb, verify_perseed as vp
    from .exact_grade import verify_sealed
    root = _checked(root)
    reg = registration(root)
    BANK = bank_dir(root)
    meta = {(r['state_id'], r['plan_sha256']): r for r in _jsonl(root/'results'/'plans.jsonl') if not r.get('skipped')}
    contexts = _jsonl(root/'results'/'contexts.jsonl')
    mined, negatives, examples = [], [], []
    for arm in ('B', 'R'):
        bank = Path(rb.bank_root(BANK, arm))
        if not (bank/vp.RECORDS).exists():
            continue
        for f in sorted((bank/vp.RECORDS).glob('*.json')):
            c = json.loads(f.read_text(encoding='utf-8'))
            verify_sealed(c, 'confirmation_sha256')
            vid = c['verification_id']
            first = json.loads((bank/'sumo_verification'/'verifications'/(vid+'.json')).read_text(encoding='utf-8'))
            verify_sealed(first, 'receipt_sha256')
            ep = first.get('episode') or {}
            sid = f"{ep.get('id')}_t{c['t0']}"
            exs = c['per_seed_v1']['examples']
            n_reps = 2+max([0]+[int(e['seed_index'])-2 for e in exs])
            for e in exs:
                plan = e['plan']
                psha = _digest(plan)
                m = meta.get((sid, psha)) or {}
                syn = e.get('synergy') or {}
                win = (e.get('tape') or {}).get('branch', {}).get('window') or {}
                s_ = str(e['seed_index'])
                ok = mined_ok(e, plan, families)
                cellk = (ep.get('world') or {}).get('cell_key')
                examples.append(dict(arm=arm, situation_id=sid, plan_sha256=psha, seed=int(e['seed_index']), ok=ok,
                                     n_reps=n_reps))
                common = dict(arm=arm, run_root=str(BANK), verification_id=vid, plan_sha256=psha, situation_id=sid,
                              world=ep.get('world'), t0=int(c['t0']), family=plan.get('family'),
                              members=list(plan['members']), map=MAP, source=PHASE, lead=m.get('lead'),
                              variant=m.get('variant'), cell=cellk, max_member_dist_t0_m=m.get('max_member_dist_t0_m'),
                              provisional_builder=m.get('provisional'), n_reps=n_reps)
                if ok:
                    dist = m.get('max_member_dist_t0_m')
                    mined.append(dict(common, schema=MINED_SCHEMA, replay_sources={}, seed=int(e['seed_index']),
                        duration_s=int(plan['duration_s']), profile=plan['profile'],
                        policy_profile=plan['policy_profile'], perturbation_profile=plan['perturbation_profile'],
                        synergy_veh_s={s_: syn.get('pooled_total_veh_s')},
                        synergy_lower95={s_: syn.get('lower95_one_sided_veh_s')},
                        advantage_veh_s={s_: e.get('advantage_veh_s')}, adv_lower95={s_: e.get('lower95_one_sided_veh_s')},
                        best_solo=syn.get('best_solo_member'), reachable=dist is not None and dist <= float(range_m),
                        range_m=float(range_m),
                        statistic='per-seed one-sided 95% circular-block (30 s) paired bootstrap B=2000, best solo '
                                  'minus group on the common window (verify_perseed._cooperation)',
                        base=reg['base'],
                        binding_note='group_lane_bank.binding() still requires world seed 986000-986071; wz3to2 '
                                     'TRAIN 989200-989599 needs that range extended before ingestion'))
                else:
                    negatives.append(dict(common, stage='native', seed=int(e['seed_index']), plan_id=e['plan_id'],
                        admitted=e.get('admitted'), status=e.get('status'), reason=e.get('reason'),
                        cooperative=e.get('cooperative'), cooperative_status=e.get('cooperative_status'),
                        advantage_veh_s=e.get('advantage_veh_s'), adv_lower95=e.get('lower95_one_sided_veh_s'),
                        synergy_veh_s=syn.get('pooled_total_veh_s'), synergy_lower95=syn.get('lower95_one_sided_veh_s'),
                        collisions=win.get('collisions'), teleports=win.get('teleports')))
    for r in _jsonl(root/'results'/'screen.jsonl'):
        if r.get('passed'):
            continue        # went to native confirmation; its per-seed rows are above
        m = meta.get((r['state_id'], r.get('plan_sha256'))) or {}
        negatives.append(dict(arm=r['arm'], stage='screen', seed=1, situation_id=r['state_id'], world=r['world'],
                              t0=r['t0'], plan_id=r['group_id'], plan_sha256=r.get('plan_sha256'), family=r.get('family'),
                              members=r.get('members'), map=MAP, source=PHASE, lead=r.get('lead'),
                              variant=m.get('variant'), cell=r.get('cell_key'), screen=r.get('screen'),
                              reason=r.get('reason'), group_adv=r.get('group_adv'), adv_lower95=r.get('adv_lb95'),
                              synergy_veh_s=r.get('synergy'), synergy_lower95=r.get('synergy_lb95'),
                              safety=r.get('safety'), screen_verification_id=r.get('verification_id')))
    pot = potential_inputs(contexts, mined, examples, range_m=range_m, cell=cell)
    for arm in ('B', 'R'):
        _write_json(root/f'CLEAN4_MOVES_{arm}.jsonl', ''.join(json.dumps(r, sort_keys=True)+'\n'
                                                                for r in mined if r['arm'] == arm))
    _write_json(root/'CLEAN4_NEGATIVES.jsonl', ''.join(json.dumps(r, sort_keys=True)+'\n' for r in negatives))
    pot['sources'] = [dict(path=str(root/name), sha256=hashlib.sha256((root/name).read_bytes()).hexdigest())
                      for name in ('CLEAN4_MOVES_B.jsonl', 'CLEAN4_NEGATIVES.jsonl', 'CHARGES.jsonl')
                      if (root/name).exists()]
    pot.update(spent_sim_s=spent(root), cap_sim_s=reg['cap_sim_s'])
    _write_json(root/'POTENTIAL_INPUTS.json', pot)
    return dict(mined_B=sum(r['arm'] == 'B' for r in mined), mined_R=sum(r['arm'] == 'R' for r in mined),
                negatives=len(negatives), examples=len(examples), cells=sorted(pot['cells']))


def status(root):
    root = _checked(root)
    reg = registration(root)
    out = dict(phase=PHASE, cap_sim_s=reg['cap_sim_s'], spent_sim_s=spent(root), base=reg['base'])
    out['remaining_sim_s'] = out['cap_sim_s']-out['spent_sim_s']
    for name in ('contexts', 'screen', 'native'):
        out[name] = len(_jsonl(root/'results'/f'{name}.jsonl'))
    out['screen_pass'] = sum(bool(r.get('passed')) for r in _jsonl(root/'results'/'screen.jsonl'))
    out['native_admitted'] = sum(bool(r.get('admitted')) for r in _jsonl(root/'results'/'native.jsonl'))
    out['stop'] = _stop(root)
    return out


# ============================================================================ 6. CLI

def _load_attr(spec):
    mod, _, attr = spec.partition(':')
    m = importlib.import_module(mod)
    return getattr(m, attr) if attr else m


def _nice():
    if os.name == 'posix':
        os.nice(max(0, 19-os.nice(0)))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python -m tools.coopmwm.clean4_verify', description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('init')
    p.add_argument('--root', required=True)
    p.add_argument('--cap', type=int, default=PHASE_CAP)
    p.add_argument('--base', default='frozen_u0', choices=BASES)
    p.add_argument('--note')
    p = sub.add_parser('collect')
    p.add_argument('--root', required=True)
    p.add_argument('--arm', required=True, choices=('B', 'R'))
    p.add_argument('--weights', required=True, help="torch file with a 'policy' state dict (frozen u0)")
    p.add_argument('--worlds', required=True, help='seed:demand:penetration[,...] (wz3to2 TRAIN)')
    p.add_argument('--leads', nargs='*', default=(), help='NM2_SUMMARY.json and/or mined-move JSONL (N3/N8)')
    p.add_argument('--adapters', default='tools.coopmwm.clean4_verify',
                   help='module exposing geometry() and runtime() (default: the pending stubs)')
    p = sub.add_parser('verify')
    p.add_argument('--root', required=True)
    p.add_argument('--ctx', nargs='*', default=())
    p.add_argument('--queue', help='file with one ctx dir per line (re-read before every context)')
    p.add_argument('--seeds', type=int, default=DEFAULT_SEEDS)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--tail', type=int, default=TAIL_S)
    p.add_argument('--builder', default='tools.coopmwm.clean4_verify:provisional_builder')
    p.add_argument('--variants', help='comma list restricting the slate (B: pair_g,pair_h,pair_t,wall_g,wall_h; R: R0,R1)')
    p = sub.add_parser('quote', help='per-moment screen/confirmation sim-s quotes (no physics, no charge)')
    p.add_argument('--root', required=True)
    p.add_argument('--ctx', nargs='+', required=True)
    p.add_argument('--seeds', type=int, default=DEFAULT_SEEDS)
    p.add_argument('--variants')
    p = sub.add_parser('extract')
    p.add_argument('--root', required=True)
    p.add_argument('--range-m', type=float, default=200.0)
    p.add_argument('--cell')
    p = sub.add_parser('status')
    p.add_argument('--root', required=True)
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.cmd == 'init':
        print(json.dumps(dict(registration_sha256=init(root, a.cap, base=a.base, note=a.note)['registration_sha256'])))
    elif a.cmd == 'collect':
        import torch
        _nice()
        adapters = _load_attr(a.adapters)
        geom, rt = adapters.geometry(), adapters.runtime()
        weights = torch.load(a.weights, weights_only=False, map_location='cpu')['policy']
        wsha = hashlib.sha256(Path(a.weights).read_bytes()).hexdigest()
        leads = load_leads(a.leads) if a.leads else []
        try:
            from .map_scenarios import world as make_world
        except ImportError:
            def make_world(seed, demand, pen):
                return dict(seed=seed, cell=DEMANDS.index(demand)*len(SHARES)+SHARES.index(pen),
                            repeat=seed-TRAIN_RANGE[0], penetration=pen, demand=demand, map=MAP,
                            cell_key=cell_key(demand, pen))
        for spec in a.worlds.split(','):
            if _stop(root):
                break
            s, d, pen = spec.split(':')
            w = make_world(int(s), int(d), float(pen))
            try:
                saved = collect(root, w, a.arm, weights, geom=geom, rt=rt, leads=leads, weights_sha256=wsha)
                print(json.dumps(dict(world=spec, saved=saved)), flush=True)
            except Exception as e:
                import traceback
                print(json.dumps(dict(world=spec, error=repr(e), tb=traceback.format_exc()[-2000:])), flush=True)
                if 'cap' in str(e) or isinstance(e, NotImplementedError):
                    break
    elif a.cmd == 'verify':
        _nice()
        builder = _load_attr(a.builder)
        done = set()
        while not _stop(root):
            todo = list(a.ctx)
            if a.queue:
                todo += [x.strip() for x in Path(a.queue).read_text().splitlines() if x.strip()]
            todo = [d for d in todo if d not in done]
            if not todo:
                break
            d = todo[0]
            done.add(d)
            try:
                res = run_context(root, d, builder=builder, seeds=a.seeds, workers=a.workers, tail_s=a.tail,
                                  names=a.variants.split(',') if a.variants else None)
                print(json.dumps(res), flush=True)
            except Exception as e:
                import traceback
                print(json.dumps(dict(ctx=d, error=repr(e), tb=traceback.format_exc()[-3000:])), flush=True)
                if 'cap' in str(e):
                    break
    elif a.cmd == 'quote':
        names = set(a.variants.split(',')) if a.variants else None
        for d in a.ctx:
            ctx, _ = load_context(d)
            vs = [v for v in variants_for(ctx) if names is None or v['name'] in names]
            print(json.dumps(dict(ctx=d, **quote_moment(ctx, plans(ctx, vs)['groups'], seeds=a.seeds),
                                  remaining_sim_s=remaining(root))))
    elif a.cmd == 'extract':
        print(json.dumps(extract(root, range_m=a.range_m, cell=a.cell)))
    elif a.cmd == 'status':
        print(json.dumps(status(root)))


if __name__ == '__main__':
    main(sys.argv[1:])
