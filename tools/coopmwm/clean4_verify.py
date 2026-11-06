"""clean4 PRE-LAUNCH RE-VERIFICATION (owner/MAIN "Plan A", 2026-11-06 15:02 +08).

Phase ``clean4_prelaunch_verify`` (cap configurable at init, hard maximum 400,000 sim-s - MAIN 16:0x
raised it from 80,000 inside the approved clean4 4M), its own precharged ledger
(``<root>/CHARGES.jsonl``), never replayed. N3 (NEWMAP2), N8 (COOPKINDS), PLANGEN, LOWCAV and PLANGEN2
admitted cooperation moves on the wz3to2 lane-drop map with a NATIVE base; their rows are not
ingestible by the coopmwm bank. This pipeline re-verifies them inside the coopmwm runtime on the
CHOSEN clean4 base (sanity verdict: frozen speed-only MAPPO u0) with the proven merge-map native path
(mining collect2 / coop_probe probe + probe_native / wallmine), emitting what the clean3 training
stack ingests: sumo_verify receipts, bank snapshots (``bank/snapshots/<sid>.pt``), observable tapes,
verify_perseed all8 confirmation records and coopmwm-mined-move-v1 rows.

Stages (every SUMO stage precharged BEFORE any physics; a stage the remaining cap cannot cover is
refused before dispatch; nothing is ever re-run):

  init     REGISTRATION.json (phase, cap, rules) BEFORE any job.
  freeze   MOVE_LIST.json (sealed, source file SHAs, order rule) BEFORE any physics: every admitted B
           move of the given sources in the training cells, then every R move (MAIN freeze rule).
  collect  ONE base-policy 1200 s episode per (world, arm) (accel_plant.episode + Capture hook as
           mining collect2), charged 1200 sim-s before physics. Captures (a) requested contexts at the
           frozen moves' t0 (``--from-moves``), (b) optionally the N3 geometry-rule moments (earliest
           pair+wall > pair > wall per window) and (c) sensor-whitelisted frames {t0-1, t0} for every
           context and for extra requested frame times (zero extra sim). Never reads an outcome.
  verify   ``--next``: the next unrun executable move of the frozen list (nothing else runs once a
           list is frozen). Seed 1 screen = ONE sv.verify (policy + group + every member solo); the
           group continues only with advantage LB95 > 0 and synergy LB95 > 0, both >= 1 veh-s, and
           collision/teleport deltas <= 0; survivors get the native confirmation = the
           probe_native.run path (first-seed sv.verify + verify_perseed.confirm_all8, extra_seeds =
           seeds-2) into ``<root>/minebank/bank``. Every outcome is written to results/moves.jsonl.
  extract  admitted & cooperative & 0 collisions/teleports & >= 2 members per (plan, seed) ->
           CLEAN4_MOVES_B/R.jsonl (coopmwm-mined-move-v1 + verified binding fields),
           CLEAN4_NEGATIVES.jsonl, CLEAN4_NEGATIVE_FRAMES.jsonl (W2 frame format: lost groups by the
           W2 per-seed rule + every requested frame time), POTENTIAL_INPUTS.json.
  quote    zero-physics screen/confirmation quotes (``--next`` = the next frozen move).
  status   cap, spent, remaining, stage and move counts.

CLI (cpuV: nice 19, own PIDs only, ``--workers`` bounds the branch subprocesses; ``<root>/STOP`` halts;
env COOPMWM_MAP_SPEC=<staged wz3to2 MAP_SPEC.json> COOPMWM_G5_RUNTIME=1 CUDA_VISIBLE_DEVICES=):
  python -m tools.coopmwm.clean4_verify init --root R [--cap 400000] [--base frozen_u0|native]
  python -m tools.coopmwm.clean4_verify freeze --root R --sources B1.jsonl ... R1.jsonl ... [--cells ...]
  python -m tools.coopmwm.clean4_verify collect --root R --arm B --weights frozen_u0.pt --from-moves \
      [--ident c4v_B_D3200_p60_s989200] [--runtime /home/ubuntu/e3final] [--checkpoint .../policy_00169.pt]
      [--frames FRAMES.json]
  python -m tools.coopmwm.clean4_verify verify --root R (--move MOVE_ID | --next [--count N]) [--seeds 3]
      [--workers N]
  python -m tools.coopmwm.clean4_verify quote --root R --next
  python -m tools.coopmwm.clean4_verify extract --root R [--range-m 200] [--cell D3200_p60]
  python -m tools.coopmwm.clean4_verify status --root R

Adapters: geometry = ``DescriptorGeometry`` on the N1 descriptor (``map_descriptor.current()``);
runtime = ``Wz3to2Runtime`` (map_runtime.guard_world + accel_plant.bind_runtime map install; branch
workers inherit COOPMWM_MAP_SPEC / COOPMWM_G5_RUNTIME / C4NM_V2V); plan builder =
``provisional_builder`` (origin ``clean4_provisional_speed_target``): N2's lane_drop.LaneDropMove
registers holds 20-90 s and depth <= 1.0 and releases 30 m before the drop, so it cannot express the
N3/PLANGEN variants (holds 90-180 s, dmax 1.5, release at the drop / pair_t anchor).
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
PHASE_CAP = 400_000                       # MAIN 16:0x: 80,000 -> 400,000 (REGISTRATION.md, BUDGET_CLEAN4 requote)
SCHEMA = 'coopmwm.clean4.prelaunch_verify.v1'
MOVE_LIST_SCHEMA = 'coopmwm.clean4.move_list.v1'
NEG_FRAMES_SCHEMA = 'coopmwm.clean4.negative_frames.v1'
MINED_SCHEMA = 'coopmwm-mined-move-v1'
MAP = 'wz3to2'
AUTHORITY = ('MAIN 2026-11-06 15:02 "clean4 ingestion blocker -> Plan A" (REGISTRATION.md): re-verify '
             'N3/N8 (and PLANGEN/LOWCAV/PLANGEN2) moves in the coopmwm runtime on wz3to2 with the chosen base via '
             'the probe_native path; phase clean4_prelaunch_verify inside the approved 4M (80,000 -> 400,000: move '
             'list frozen before the run, all outcomes reported, no reruns of failed moves, R matched per-move '
             'procedure, no criteria change); the pre-launch gate counts only re-verified rows')

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
MINED_FAMILIES = FAMILIES+('group_hold',)    # N2 plan families (R hold rows -> group_hold)
SPEED_HOLD_FAMILIES = FAMILIES                # executable by the speed-target builder in this runtime
SPEED_HOLD_VARIANTS = ('pair_g', 'pair_h', 'pair_t', 'wall_g', 'wall_h')
NOT_EXECUTABLE = 'not_executable_in_runtime'
ORIGIN_PROVISIONAL = 'clean4_provisional_speed_target'
# MAIN freeze rules (verbatim, message to N4 ~16:4x) -------------------------------------------------
TRAIN_CELL_DEMANDS = (2400, 2800, 3200)
TRAIN_CELL_SHARES = (0.4, 0.6)
OFF_FAMILIES = ('discharge_sync', 'discharge', 'platoon_headway', 'harmonize_rows',
                'platoon', 'vsl_rows')        # last two = the COOPKINDS row names of platoon_headway / harmonize_rows
N9_FAMILIES = ('lane_balance', 'early_merge', 'zipper', 'zipper_order')
N9_DEADLINE = '2026-11-06T20:00:00+08:00'
FAMILY_BUILDERS = {}                          # N9 integration registers {family: builder}; empty = not merged
MAIN_RULES = (
    'Sources, in this order, all in E:/research-dense/runs/active/C4-V7FT-20261017: NEWMAP_MOVES_B.jsonl (N3), '
    'PLANGEN_MOVES_B.jsonl, PLANGEN2_MOVES_B.jsonl (final: 24 rows / 8 moves), COOPKINDS_MOVES_B.jsonl, '
    'LOWCAV_MOVES_B.jsonl. Then R: NEWMAP_MOVES_R.jsonl, PLANGEN_MOVES_R.jsonl, PLANGEN2_MOVES_R.jsonl '
    '(18 rows / 6 moves), COOPKINDS_MOVES_R.jsonl (empty).',
    'A move = distinct (source, world seed, cell, t0, family, variant). Rows are per seed; collapse them to moves.',
    'Training-cell rule: eligible iff demand in {2400, 2800, 3200} and CAV share in {0.4, 0.6}. Other cells '
    "(LOWCAV 0.1-0.3, any 0.2 cell) are listed with status 'excluded_cell', reported, and never run.",
    "discharge_sync / discharge, platoon_headway and harmonize_rows are OFF families: list them with status "
    "'excluded_family'.",
    "lane_balance, early_merge and zipper(_order) stay in the list with status 'pending_n9' and run only if N9 "
    "merges before 20:00. Otherwise they become 'excluded_n9_deadline'.",
    'Each move is re-verified in its OWN cell/world, on the clean4 base (MAPPO u0). R moves get the same per-move '
    'procedure on their own rows; do not pad R.',
    'Every outcome is reported, including failures. A failed move is never re-run.',
    'Budget: cap 400,000, phase clean4_prelaunch_verify.',
    'Executable now: N3/PLANGEN/PLANGEN2 walls and pairs (wall_g, wall_h, pair_g, pair_h, pair_t) and R_random '
    "moves (the R random-procedure variants: build the R plan from the row's members/params with the same "
    'builder).',
    # MAIN update (N4 URGENT, N2 exact converter c4/nm-moves c7983467) - supersedes the pending_n9 rule above:
    'Eligible = wall/pair/hold rows (pair_g, pair_h, pair_t, wall_g, wall_h, and R group_hold) in training cells '
    'p40/p60.',
    "Every lane-change row (early_merge, lane_balance, zipper_order, R lane_balance) gets status 'excluded: no "
    "exact executor' (lane_drop.NotRepresentable). Same for the pending_n9/COOPKINDS lane-family rows.",
    'Route plan building through N2 with lane_drop_macro set; moves.from_plan(plan) must return a LaneDropMove '
    '(zero legacy projection); what SUMO executes in the branch must be the N2 law.')
EFFORT_MATCHED = ('PLANGEN_MOVES_B.jsonl', 'PLANGEN_MOVES_R.jsonl', 'PLANGEN2_MOVES_B.jsonl', 'PLANGEN2_MOVES_R.jsonl')
EXCLUDED_F3 = 'excluded: effort-matched sources only (FAIRAUDIT F3)'
EXCLUDED_F4 = 'excluded: budget cut (FAIRAUDIT F4)'
FAIRAUDIT_RULES = (
    'F3 (sources): freeze ONLY the effort-matched sources: PLANGEN_MOVES_B/R.jsonl and PLANGEN2_MOVES_B/R.jsonl. '
    "NEWMAP (N3) and COOPKINDS rows are not in clean4: listed as 'excluded: effort-matched sources only "
    "(FAIRAUDIT F3)'.",
    'F4 (order): the frozen order interleaves B and R moves alternately (B1, R1, B2, R2, ...; the longer arm '
    "tail continues alternation-free at the end).",
    'F4 (budget): per-arm budget reserved pro rata to the number of executable moves per arm (share_B = '
    'nB/(nB+nR) of the cap minus the collections). If the cap cannot cover all moves of both arms, BOTH arms are '
    'cut by the same rule (same fraction of each arm ordered list, from the tail), never only R tail; the run '
    'list is written here before any physics and verify refuses any move outside it.')
EXCLUDED_R_A = 'excluded: R zero-guidance (option a)'
OPTION_A_FACT = ('MAIN 2026-11-06 option (a): group_hold is not a runtime proposal family and not guidance; the '
                 'matched-effort random procedure (R; PLANGEN/PLANGEN2 R mining ledgers = matched effort) yielded 0 '
                 'runtime-executable rows; R trains with zero guidance rows (RL + team reward + G4 only).')
OPTION_A_SCOPE = ('MAIN N4 FINAL scope (option a): Plan A re-verifies ONLY the B wall/pair moves from PLANGEN_MOVES_B + '
                  'PLANGEN2_MOVES_B in training cells p40/p60. R moves are NOT run. F4 interleaving is moot.')
R_PROCEDURE_FACT = ('BOOST-PRE/MAIN 2026-11-06: no R_random row maps to a runtime closing_pair/abreast_wall (15/18 lack '
                    'the geometry, 3 are unobservable); R guidance = negatives only')
NO_EXACT_EXECUTOR = 'excluded: no exact executor'
ORIGIN_N2 = 'clean4_n2_law_verified_plan'
N2_BUILDER_NOTE = ('N2 law (lane_drop.member_command: a = clip((v_w - v)/1 s, -dmax, 1.0), release at the member own '
                   'drop crossing, hold end or the pair_t anchor 150 m before the drop) applied second by second on the '
                   'member speed/position trajectory under the law, starting from the recorded t0 state of the '
                   'collection episode (seed-1 CRN); bound as lane_drop.verified_plan (per-member release el, policy '
                   'reference after release). The sv.verify branch Controller executes plan profiles open loop '
                   '(plans.acceleration_at) - it has no closed-loop lane-drop executor - so the branch runs exactly '
                   'this N2-law profile. Base-recorded speeds are NOT used as the law input: the open-loop law on them '
                   'would brake every second for the whole hold (towards a stop).')

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
FRAME_KEYS = OBSERVABLE_KEYS                  # = wall_negatives.FRAME_KEYS (W2 sensor whitelist)
PROVISIONAL = 'clean4_provisional_open_loop_v1'
BASES = ('frozen_u0', 'native')
RUNTIME_DIR = '/home/ubuntu/e3final'
CHECKPOINT = '/home/ubuntu/mappospeed/production2/policy_00169.pt'

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
NEGATIVE_RULE = ('W2 per-seed rule (wall_negatives.RULE): a verified (group, seed) LOST iff its own-seed synergy '
                 '(best solo J - group J) < 0; where no solos ran on that seed, iff its own-seed advantage vs '
                 'policy-continues < 0. Frames = sensor-whitelisted rows {t0-1, t0} of the collection episode '
                 '(= the shared prefix of every branch). Requested extra frame times are emitted as kind '
                 '"requested" without a label.')
ORDER_RULE = ('MAIN freeze rule: MOVE_LIST.json is frozen before any physics of this phase. Order = every admitted '
              'B move of the given sources (in the order given, rows in file order, one entry per (situation, '
              'variant, members) with all its admitted seeds) in the training cells, then every R move of the given '
              'sources the same way (identical per-move procedure). Moves run strictly in this order (verify '
              '--next); every outcome incl. failures is reported (results/moves.jsonl); a claimed move is never '
              're-run. Families without an executable builder in this runtime are listed with status '
              'not_executable_in_runtime and never run.')


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
    """Append under an exclusive cross-process lock (concurrent moves share results/*.jsonl)."""
    p = Path(root)/'results'/name
    p.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(r, sort_keys=True, default=str)+'\n' for r in rows)
    with _locked(root, 'RESULTS.lock'):
        with p.open('a', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())


def _jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []


def _finite(x):
    return x is not None and not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


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


def make_world(seed, demand, penetration):
    """The N1 registry world dict (map_scenarios.world)."""
    from . import map_scenarios
    return map_scenarios.world(int(seed), int(demand), float(penetration))


def check_world(world, role='mining'):
    """Registered wz3to2 TRAIN world (989200-989599), never an evaluation seed (N1 map_scenarios.guard)."""
    from . import map_scenarios
    if not isinstance(world, dict) or world.get('map') != MAP:
        raise ValueError('not a wz3to2 world')
    s = world.get('seed')
    if type(s) is not int or seed_forbidden(s) or map_scenarios.seed_scope(s) != 'train':
        raise ValueError(f'world seed {s!r} is not a registered wz3to2 TRAIN seed')
    if world.get('demand') not in DEMANDS or world.get('penetration') not in SHARES:
        raise ValueError('world is not a registered wz3to2 cell')
    if world.get('cell_key') != cell_key(world['demand'], world['penetration']):
        raise ValueError('world cell_key does not match its demand/penetration')
    map_scenarios.guard(world, role)
    return 'map_scenarios.guard'


def production_env(world):
    """Process binding of a real wz3to2 run (collector AND inherited by every branch subprocess):
    COOPMWM_MAP_SPEC must select the world's map (map_descriptor.for_world) and COOPMWM_G5_RUNTIME=1."""
    from . import map_descriptor
    if world.get('map') == MAP and not os.environ.get(map_descriptor.ENV):
        raise RuntimeError(f'{map_descriptor.ENV} must point at the staged wz3to2 MAP_SPEC.json')
    if os.environ.get('COOPMWM_G5_RUNTIME') != '1':
        raise RuntimeError('COOPMWM_G5_RUNTIME=1 required (clean3 G5 runtime overlay)')
    desc = map_descriptor.for_world(world)
    return dict(map_spec=os.environ.get(map_descriptor.ENV), g5_runtime='1', descriptor_sha256=desc.digest,
                descriptor=desc.name)


def _stop(root):
    return (Path(root)/'STOP').exists()


def whitelist_rows(rows):
    """W2 sensor whitelist (wall_negatives.whitelist): exactly FRAME_KEYS, sorted by vid. Rows lacking a key
    are dropped and counted (never filled)."""
    out = [{k: r[k] for k in FRAME_KEYS} for r in rows if all(k in r for k in FRAME_KEYS)]
    return sorted(out, key=lambda r: r['vid']), len(rows)-len(out)


def frames_record(world, t0, prev_rows, rows):
    frames, dropped = {}, 0
    for t, rr in ((int(t0)-1, prev_rows), (int(t0), rows)):
        if rr is not None:
            frames[str(t)], d = whitelist_rows(rr)
            dropped += d
    complete = set(frames) == {str(int(t0)-1), str(int(t0))}
    return dict(world_seed=int(world['seed']), world=world, t0=int(t0), frames=frames,
                frames_sha256=_digest(frames), complete=complete, dropped_rows=dropped)


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
                           extract=EXTRACT_RULE, potential=POTENTIAL_RULE, negatives=NEGATIVE_RULE,
                           order=ORDER_RULE,
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
def _locked(root, name='LEDGER.lock'):
    """Exclusive flock (Linux: fcntl, safe across the concurrent move processes)."""
    try:
        import fcntl
    except ModuleNotFoundError:          # local Windows CPU tests only
        from .fcntl_compat import install
        install()
        import fcntl
    Path(root).mkdir(parents=True, exist_ok=True)
    with (Path(root)/name).open('a') as lock:
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


def arm_budget(root, arm):
    """FAIRAUDIT F4 per-arm move budget (None without a frozen MOVE_LIST)."""
    p = move_list_path(root)
    if arm not in ('B', 'R') or not p.exists():
        return None
    fair = json.loads(p.read_text(encoding='utf-8')).get('fairaudit') or {}
    return fair.get(f'budget_{arm}')


def arm_spent(root, arm, rows=None):
    """Move spend of one arm (collections excluded: they are reserved before the pro-rata split)."""
    return sum(int(r['sim_s']) for r in (ledger_rows(root) if rows is None else rows)
               if r.get('arm') == arm and r.get('stage') != 'collect')


def left_for(root, arm):
    """Remaining sim-s a stage of ``arm`` may use: min(global cap, the arm's F4 budget)."""
    left = remaining(root)
    budget = arm_budget(root, arm)
    return left if budget is None else min(left, int(budget)-arm_spent(root, arm))


def charge(root, ident, sim_s, *, stage='', branches=1, arm=None):
    """Durable atomic precharge on this phase's own ledger: refuses unregistered roots, duplicate ids
    (never replayed), anything over the registered cap and (moves) anything over the arm's F4 budget."""
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
        budget = arm_budget(root, arm) if stage != 'collect' else None
        if budget is not None and arm_spent(root, arm, rows)+sim_s > int(budget):
            raise BudgetBlocked(f'arm {arm} F4 budget {budget} reached')
        with (root/'CHARGES.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(id=ident, phase=PHASE, stage=stage, sim_s=sim_s, branches=int(branches),
                                    arm=arm, unix=time.time()), sort_keys=True)+'\n')
            f.flush()
            os.fsync(f.fileno())


def _charger(root, prefix, stage, arm=None):
    """sv.verify charge hook: sum(spec sim_s) of the WHOLE branch set, before dispatch."""
    def chg(vid, specs):
        charge(root, f'{prefix}_{vid}', sum(int(s['sim_s']) for s in specs), stage=stage, branches=len(specs),
               arm=arm)
    return chg


# ============================================================================ 2. geometry (observable only)

class Wz3to2GeometryStub:
    """Geometry adapter protocol: ``locate(row)`` maps ONE observable CAV row (keys OBSERVABLE_KEYS, runtime
    WM frame) to the NEWMAP2 frame ``(edge, lane, x)``: edge 'up' upstream of the drop, lane 0 = closing
    lane, x = global x (X_B = 2696 at the drop); None when unknown. Used when the process map is not a
    lane drop (COOPMWM_MAP_SPEC unset): refuses."""

    name = 'wz3to2_geometry_stub'
    x_drop = X_B

    def locate(self, row):
        raise NotImplementedError('process map is not the wz3to2 lane drop (set COOPMWM_MAP_SPEC)')


class DescriptorGeometry:
    """N1 map-descriptor frame -> NEWMAP2 frame (the N3 rule runs verbatim on it).

    WM frame (map_descriptor): closing lane = bottleneck['closing_lane'] (WM 2), adjacent through lane
    = bottleneck['adjacent_lane'] (WM 1), far lane = the other through lane (WM 0); the drop is at
    s = bottleneck['drop_s_m'] (75.44) = x bottleneck['drop_x_m'] (2696). NEWMAP2 lane 0 = closing,
    1 = adjacent, 2 = far; x = s - drop_s + drop_x; edge 'up' upstream of the drop. Hence the N3 zone
    x in [X_B-1800, X_B-400] is s in [drop_s-1800, drop_s-400], the pair is WM 2 + WM 1 and the wall
    one CAV in each of WM 2/1/0."""

    name = 'n1_descriptor_geometry'

    def __init__(self, desc):
        neck = getattr(desc, 'bottleneck', None) or {}
        if getattr(desc, 'kind', None) != 'lane_drop' or 'drop_s_m' not in neck or 'closing_lane' not in neck:
            raise ValueError('descriptor has no lane-drop bottleneck in the model frame')
        self.drop_s = float(neck['drop_s_m'])
        self.x_drop = float(neck.get('drop_x_m', X_B))
        if abs(self.x_drop-X_B) > 1e-6:
            raise ValueError('descriptor drop x differs from the NEWMAP2 X_B')
        closing = int(neck['closing_lane'])
        adjacent = int(neck.get('adjacent_lane', 1))
        through = [int(x) for x in neck.get('through_lanes', [ln for ln in range(3) if ln != closing])]
        far = [ln for ln in through if ln != adjacent]
        if len(far) != 1 or adjacent not in through or closing in through:
            raise ValueError('descriptor lanes are not a 3-to-2 lane drop')
        self.lane_map = {closing: CLOSING_LANE, adjacent: PAIR_LANE, far[0]: 2}
        self.desc_name = getattr(desc, 'name', None)

    def locate(self, row):
        s, lane = float(row['s_m']), int(row['lane'])
        if lane not in self.lane_map:
            return None
        return ('up' if s < self.drop_s else 'past_drop', self.lane_map[lane], s-self.drop_s+self.x_drop)


def geometry():
    """The process map's geometry (COOPMWM_MAP_SPEC -> N1 descriptor); the stub on any other map."""
    from . import map_descriptor
    desc = map_descriptor.current()
    if getattr(desc, 'kind', None) == 'lane_drop':
        return DescriptorGeometry(desc)
    return Wz3to2GeometryStub()


def observable(row):
    """Projection to the observable CAV/vehicle row keys; nothing else is ever read."""
    return {k: row[k] for k in OBSERVABLE_KEYS if k in row}


def located_cavs(rows, geom):
    out = []
    for r in rows:
        if not r.get('is_cav'):
            continue
        loc = geom.locate(r)
        if loc is None:
            continue
        edge, lane, x = loc
        out.append(dict(id=r['vid'], edge=edge, lane=int(lane), x=round(float(x), 2), v=round(float(r['v_mps']), 3),
                        s_m=float(r['s_m']), wm_lane=int(r['lane']), in_zone=edge == 'up' and ZONE[0] <= x <= ZONE[1]))
    return out


def zone_cavs(rows, geom):
    return [c for c in located_cavs(rows, geom) if c['in_zone']]


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


# ============================================================================ runtime binding

def _v2v_width(weights):
    w = weights.get('actor.0.weight') if isinstance(weights, dict) else None
    return True if w is None else int(w.shape[1]) > 33            # newmap_sanity.run_clean3_u0 rule


def job_fields(*, ident, path, world, arm, weights, noise_seed, runtime, checkpoint, sim_s=COLLECT_SIM_S,
               v2v=True, deterministic=False):
    """The mining collect2 base-policy job (plain episode: no guide, no intervention). deterministic=False
    (default) matches the sv.verify branches, which sample the residual after t0 under CRN."""
    return dict(id=ident, path=str(path), world=world, role='train', arm=arm, update=0, training_seed=20261101,
                weights=weights, runtime=runtime, checkpoint=checkpoint, noise_seed=noise_seed,
                deterministic=bool(deterministic), guided=False, b_replay=True, v2v=bool(v2v), health_guards=False,
                sim_s=sim_s, stop=sim_s, warm_history=True, accel_bounds=[-3., 2.6])


class Wz3to2RuntimeStub:
    """Runtime binding protocol: ``base``, ``job(ident=, path=, world=, arm=, weights=, noise_seed=,
    sim_s=) -> job dict`` and optionally ``episode(job, controller_hook=)`` (default
    accel_plant.episode). The stub refuses (no runtime/checkpoint configured)."""

    name = 'wz3to2_runtime_stub'
    base = 'frozen_u0'

    def job(self, **kw):
        raise NotImplementedError('no wz3to2 runtime configured (Wz3to2Runtime(runtime_dir, checkpoint))')


class Wz3to2Runtime:
    """The real binding: frozen e3final runtime + frozen speed-only MAPPO checkpoint, the clean3 training
    episode path (accel_plant.episode -> bind_runtime -> map_runtime.install(job) binds wz3to2 from
    COOPMWM_MAP_SPEC; accel_protocol.guard dispatches map worlds to map_scenarios.guard)."""

    name = 'wz3to2_runtime'

    def __init__(self, runtime_dir=RUNTIME_DIR, checkpoint=CHECKPOINT, *, base='frozen_u0', deterministic=False):
        self.runtime_dir, self.checkpoint, self.base, self.deterministic = runtime_dir, checkpoint, base, deterministic
        self.env = None

    def job(self, *, ident, path, world, arm, weights, noise_seed, sim_s):
        from . import map_runtime
        self.env = production_env(world)
        map_runtime.guard_world(world, 'mining')
        return job_fields(ident=ident, path=path, world=world, arm=arm, weights=weights, noise_seed=noise_seed,
                          runtime=self.runtime_dir, checkpoint=self.checkpoint, sim_s=sim_s,
                          v2v=_v2v_width(weights), deterministic=self.deterministic)


def runtime(runtime_dir=None, checkpoint=None, **kw):
    if runtime_dir is None and checkpoint is None:
        return Wz3to2RuntimeStub()
    return Wz3to2Runtime(runtime_dir or RUNTIME_DIR, checkpoint or CHECKPOINT, **kw)


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


# ============================================================================ 3. collect

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
    """accel_plant controller hook (mining collect2 Capture): per-second references/tracks, requested
    contexts, window moments and {t-1, t} frames; reads only the controller event's observable rows,
    commands and sample."""

    def __init__(self, select, keeper, capture_fn, profile_s=PROFILE_S, *, geom=None, requests=None, obs_sink=None,
                 frame_times=None):
        self.select, self.keeper, self.capture_fn, self.profile_s = select, keeper, capture_fn, int(profile_s)
        self.geom = geom
        self.requests = {int(r['t0']): r for r in requests or ()}
        self.frame_want = {int(t)+d for t in frame_times or () for d in (-1, 0)}
        self.obs_sink = obs_sink
        self.obs_times = {}
        for r in requests or ():
            for d in (-1, 0):
                self.obs_times.setdefault(int(r['t0'])+d, []).append(
                    (int(r['t0']), 'move', list(r.get('move_ids') or []), list(r.get('members') or [])))
        for t0 in sorted({int(x) for x in frame_times or ()}):
            for d in (-1, 0):
                self.obs_times.setdefault(t0+d, []).append((t0, 'negative', [], []))
        self.frames = {}
        self.recording = {}          # key -> record ; key = t0 (window moment) or ('req', t0)
        self.marks = 0
        self.prev = None

    def __call__(self, controller, job, prefixes):
        self.c, self.job, self.prefixes = controller, job, prefixes

    def _start(self, key, t, vids, rows, by_vid, members, action, commands, extra=None):
        prev = self.prev[1] if self.prev is not None and self.prev[0] == t-1 else None
        self.recording[key] = dict(
            t0=t, vids=vids, rows_t0=rows, frames=frames_record(self.job['world'], t, prev, rows),
            policy_profile={v: [_reference(v, members, action, commands)] for v in vids},
            cav_track={v: [_track(by_vid.get(v))] for v in vids}, **(extra or {}))

    def after_step(self, t, after):
        e = self.c.events[-1]
        t = int(e['time_s'])
        members = list(e['sample']['context']['members'])
        action = e['sample']['action']
        commands = e.get('commands') or {}
        rows = [observable(r) for r in e['rows']]
        by_vid = {r['vid']: r for r in rows if 'vid' in r}
        for rec in self.recording.values():
            if 0 < t-rec['t0'] < self.profile_s:
                for vid in rec['vids']:
                    rec['policy_profile'][vid].append(_reference(vid, members, action, commands))
                    rec['cav_track'][vid].append(_track(by_vid.get(vid)))
        if self.obs_sink is not None and t in self.obs_times:
            for t0, kind, move_ids, member_ids in self.obs_times[t]:
                self.obs_sink(obs_row(self.job, e, t0=t0, time_s=t, kind=kind, move_ids=move_ids,
                                      member_ids=member_ids))
        if t in self.frame_want:
            self.frames[t] = rows
        if t in self.requests:
            req = self.requests[t]
            located = located_cavs(rows, self.geom)
            upstream = [c for c in located if c['edge'] == 'up']
            vids = list(dict.fromkeys(list(req.get('members') or [])+[c['id'] for c in upstream if c['in_zone']]))
            snap = self.capture_fn(self.c, self.prefixes, self.job, t, e['sample'], self.c.generator)
            self._start(('req', t), t, vids, rows, by_vid, members, action, commands,
                        dict(snap=snap, located=located, upstream=upstream, request=req))
        wi = self.keeper.window_of(t)
        if wi is not None:
            self.marks += 1
            moment = self.select(rows, t)
            if moment is not None and self.keeper.wants(wi, int(moment.get('rank', 0))):
                snap = self.capture_fn(self.c, self.prefixes, self.job, t, e['sample'], self.c.generator)
                for d in self.keeper.keep(wi, t, moment, snap):
                    self.recording.pop(d, None)
                vids = [c['id'] for c in moment.get('zone') or []] or list(moment.get('members') or [])
                self._start(t, t, vids, rows, by_vid, members, action, commands)
        self.prev = (t, rows)


OBS_FORBIDDEN = ('hetdata', 'driver', 'vtype', 'v_type', 'vehicle_param', 'truth')


def _obs_ok(key):
    k = str(key).lower()
    return not any(f in k for f in OBS_FORBIDDEN)


def _jsonable(x, dropped, depth=0):
    """Tensors/arrays -> lists; dict keys whitelist-checked (OBS_FORBIDDEN dropped and recorded)."""
    if depth > 12:
        return repr(x)
    if hasattr(x, 'detach') and hasattr(x, 'tolist'):
        return x.detach().cpu().tolist()
    if hasattr(x, 'tolist') and not isinstance(x, (str, bytes)):
        return x.tolist()
    if isinstance(x, dict):
        out = {}
        for k, v in dict.items(x):
            if _obs_ok(k):
                out[str(k)] = _jsonable(v, dropped, depth+1)
            else:
                dropped.add(str(k))
        return out
    if isinstance(x, (list, tuple)):
        return [_jsonable(v, dropped, depth+1) for v in x]
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return repr(x)


def obs_row(job, event, *, t0, time_s, kind, move_ids, member_ids):
    """One OBS_TABLES row: the controller's exact observation rows of every CAV (incl. detected HDVs) and
    the per-CAV sample context the base policy saw (members, MAPPO observation, V2V slots, RSU block -
    whatever the event exposes), sensor-field whitelist-checked (no hetdata/driver/vType/truth keys)."""
    dropped = set()
    rows = [_jsonable(r, dropped) for r in event.get('rows') or []]
    sample = event.get('sample') or {}
    context = _jsonable(sample.get('context'), dropped)
    extra = {k: _jsonable(v, dropped) for k, v in dict.items(sample)
             if k not in ('context', 'action') and _obs_ok(k) and k in ('v2v', 'rsu', 'obs', 'x', 'slots')}
    return dict(schema=SCHEMA+'.obs_table', world=job['world'], arm=job['arm'], t0=int(t0), time_s=int(time_s),
                kind=kind, move_ids=list(move_ids), member_ids=list(member_ids), rows=rows, sample_context=context,
                sample_extra=extra, lane_drop_features='not_available',
                lane_drop_features_reason=('lane_drop.lane_drop_features needs per-CAV SensorReport packets '
                                           '(own + V2V partner reports), not observation rows'),
                dropped_keys=sorted(dropped), whitelist=list(OBS_FORBIDDEN))


def _append_file(root, name, rows):
    """Locked append to a root-level JSONL (OBS_TABLES.jsonl)."""
    text = ''.join(json.dumps(r, sort_keys=True, default=str)+'\n' for r in rows)
    with _locked(root, 'RESULTS.lock'):
        with (Path(root)/name).open('a', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())


def collect_ident(world, arm):
    return f'c4v_{arm}_{world["cell_key"]}_s{world["seed"]}'


def collect(root, world, arm, weights, *, select=None, windows=WINDOWS, every=EVERY, geom=None, rt=None,
            leads=None, capture_fn=None, snapshot_sha=None, profile_s=PROFILE_S, weights_sha256=None,
            requests=None, frame_times=None):
    """ONE base-policy episode for (world, arm), charged BEFORE physics. Saves (a) requested contexts
    ``req_t<t0>`` (requests = [{t0, members, move_ids}]), (b) the kept window moments ``ctx_t<t0>``
    (windows=() disables them), each with FRAMES.json, and (c) frames/frames_t<t0>.json for frame_times."""
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
    geom = geom or geometry()
    geom.locate(dict(vid='__preflight__', s_m=0.0, v_mps=0.0, lane=0, is_cav=True, length_m=5.0))
    select = select or selector(geom)
    keeper = WindowKeeper(windows, every)
    for r in requests or ():
        if not WARMUP_S < int(r['t0']) < EPISODE_S:
            raise ValueError('requested t0 outside the controlled episode')
    for t in frame_times or ():
        if not WARMUP_S < int(t) < EPISODE_S:
            raise ValueError('frame time outside the controlled episode')
    ident = collect_ident(world, arm)
    work = root/'collect'/ident
    ns = noise_seed(world, arm)
    job = rt.job(ident=ident, path=work, world=world, arm=arm, weights=weights, noise_seed=ns, sim_s=COLLECT_SIM_S)
    if job.get('id') != ident or int(job.get('sim_s', 0)) != COLLECT_SIM_S or job.get('noise_seed') != ns:
        raise ValueError('runtime job does not carry the registered id/sim_s/noise seed')
    if (root/'STOP').exists():
        raise RuntimeError('STOP present')
    charge(root, 'collect_'+ident, COLLECT_SIM_S, stage='collect', arm=arm)          # BEFORE physics
    work.mkdir(parents=True, exist_ok=False)
    hook = _Capture(select, keeper, capture_fn or sv.capture, profile_s, geom=geom, requests=requests,
                    obs_sink=lambda row: _append_file(root, 'OBS_TABLES.jsonl', [row]),
                    frame_times=frame_times)
    episode = getattr(rt, 'episode', None)
    if episode is None:
        from . import accel_plant
        episode = accel_plant.episode
    result = episode(job, controller_hook=hook)
    sha = snapshot_sha or sv.snapshot_sha
    common = dict(schema=SCHEMA+'.context', phase=PHASE, world=world, arm=arm, base=reg['base'],
                  selection=SELECTION_RULE, profile_s=profile_s, world_guard=guard, noise_seed=ns,
                  v2v=bool(job.get('v2v')), deterministic=bool(job.get('deterministic')),
                  runtime_env=getattr(rt, 'env', None))

    def save(d, snap, context, rec):
        d.mkdir()
        torch.save(snap, d/'SNAPSHOT.pt')
        _write_new(d/'FRAMES.json', rec['frames'])
        _write_new(d/'CONTEXT.json', dict(common, snapshot_sha256=sha(snap), policy_profile=rec['policy_profile'],
                                          cav_track=rec['cav_track'], rows_t0=rec['rows_t0'],
                                          frames_sha256=rec['frames']['frames_sha256'], **context))
        return str(d)
    saved, windows_out, requested_out = [], [], []
    for t0, req in sorted(hook.requests.items()):
        rec = hook.recording.get(('req', t0))
        if rec is None:
            requested_out.append(dict(t0=t0, captured=False, move_ids=req.get('move_ids')))
            continue
        sid = f'{ident}_r{t0}'
        lead, hits = match_lead(world, window_index(t0), leads)
        d = save(work/f'req_t{t0}', rec['snap'], dict(
            state_id=sid, kind='requested', t0=t0, window=window_index(t0), members=list(req.get('members') or []),
            move_ids=list(req.get('move_ids') or []), moment=dict(kind='requested', t=t0, members=req.get('members')),
            geometry=dict(pair=None, wall=None, zone=rec['upstream'], located=rec['located']),
            lead=lead, lead_moments=hits), rec)
        saved.append(d)
        requested_out.append(dict(t0=t0, captured=True, dir=d, move_ids=req.get('move_ids')))
    final = keeper.final()
    for wi in range(len(keeper.windows)):
        if wi not in final:
            windows_out.append(dict(window=wi, t0=None))
            continue
        t0, moment, snap = final[wi]
        rec = hook.recording[t0]
        lead, hits = match_lead(world, wi, leads)
        d = save(work/f'ctx_t{t0}', snap, dict(
            state_id=f'{ident}_t{t0}', kind='moment', t0=t0, window=wi, moment=moment, members=moment.get('members'),
            geometry=dict(pair=moment.get('pair'), wall=moment.get('wall'), zone=moment.get('zone')),
            lead=lead, lead_moments=hits), rec)
        saved.append(d)
        windows_out.append(dict(window=wi, t0=t0, kind=moment.get('kind'), dir=d, lead=lead))
    frames_out = []
    for t0 in sorted({int(t) for t in frame_times or ()}):
        rec = frames_record(world, t0, hook.frames.get(t0-1), hook.frames.get(t0))
        _write_new(work/'frames'/f'frames_t{t0}.json', dict(rec, arm=arm, kind='requested', state_ident=ident))
        frames_out.append(dict(t0=t0, complete=rec['complete']))
    _write_new(work/'COLLECTION_RECEIPT.json', dict(
        phase=PHASE, id=ident, world=world, arm=arm, base=reg['base'], noise_seed=ns, charged_sim_s=COLLECT_SIM_S,
        actual_sim_s=(result or {}).get('actual_sim_s'), windows=windows_out, requested=requested_out,
        frames=frames_out, marks=hook.marks, weights_sha256=weights_sha256, v2v=common['v2v'],
        deterministic=common['deterministic'], runtime_env=common['runtime_env'], unix=time.time()))
    return saved


# ============================================================================ 4. plans

def r_proposals(ctx, n=N_R):
    """N3 R procedure (seeded per state id): 2-3 random zone CAVs, spread <= 300 m, never a B set."""
    z = sorted([c for c in (ctx.get('geometry') or {}).get('zone') or [] if c.get('in_zone', True)],
               key=lambda c: c['id'])
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
    """Registered variants of a window moment (B: N3 B variants; R: R procedure)."""
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
    """PROVISIONAL speed-target builder (plan.origin = 'clean4_provisional_speed_target').

    N2's lane_drop.LaneDropMove cannot express the N3/PLANGEN variants (registered hold 20-90 s, depth
    <= 1.0, release 30 m before the drop), so the N3 law is applied open loop on top of the seed-1 policy
    reference: each member's NEWMAP2-frame (x, v) at t0 is propagated under a = clip((v_w - v)/1 s,
    -dmax, +amax) for hold_s seconds; a member is released (offset 0, policy reference) once its
    predicted x reaches release_x (the drop) or, for pair_t, once the anchor (closing-lane CAV) reaches
    anchor_release_x. duration_s = hold_s. Returns the probe.ease_plan shape or None when a member has
    no complete reference or is not upstream of the drop at t0."""
    p = variant['params']
    duration = int(p['hold_s'])
    zone = {c['id']: c for c in (ctx.get('geometry') or {}).get('zone') or [] if c.get('edge', 'up') == 'up'}
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
                duration_s=duration, origin=ORIGIN_PROVISIONAL, builder=PROVISIONAL, provisional=True,
                variant=dict(name=variant['name'], params=dict(p)), policy_profile=pol, profile=prof,
                perturbation_profile=pert, release_s=rel)


def _descriptor():
    """The process map descriptor (COOPMWM_MAP_SPEC -> N1 wz3to2); refuses any other map."""
    from . import map_descriptor
    desc = map_descriptor.current()
    if getattr(desc, 'kind', None) != 'lane_drop':
        raise RuntimeError('process map is not the wz3to2 lane drop (set COOPMWM_MAP_SPEC)')
    return desc


def n2_builder(ctx, members, variant):
    """N2-exact builder for frozen hold moves (variant['lane_drop_macro'] from
    lane_drop.move_from_mined_row at freeze). See N2_BUILDER_NOTE. Returns a lane_drop.verified_plan
    (lane_drop_macro set; moves.from_plan returns the LaneDropMove) or None when a member is not
    upstream of the drop at t0 / lacks a complete policy reference / releases at once."""
    from . import lane_drop as ld, moves as mv
    macro = variant.get('lane_drop_macro')
    if not isinstance(macro, dict):
        return None
    move = ld.LaneDropMove.from_dict(macro)
    desc = variant.get('descriptor') or _descriptor()
    view = ld.MapView.from_descriptor(desc)
    move.check_roles(view)
    zone = {c['id']: c for c in (ctx.get('geometry') or {}).get('zone') or [] if c.get('edge', 'up') == 'up'}
    if sorted(members) != sorted(move.members) or any(v not in zone for v in move.members):
        return None
    hold_end = move.onset+move.hold
    pol = {v: _policy_rows(ctx, v, hold_end) for v in move.members}
    if any(r is None for r in pol.values()):
        return None
    s = {v: float(zone[v]['s_m']) for v in move.members}
    speed = {v: float(zone[v]['v']) for v in move.members}
    prof = {v: list(pol[v]) for v in move.members}
    rel = {}
    for k in range(hold_end):
        known = {v: s[v] for v in move.members}
        acc = {}
        for v in move.members:
            if v in rel:
                continue
            a = ld.member_command(move, v, k, view, speed=speed[v], positions=known)
            if a is None:
                if k >= move.onset:
                    rel[v] = k
                continue
            prof[v][k] = acc[v] = float(a)
        for v, a in acc.items():
            v_new = max(0.0, speed[v]+a)
            s[v] += 0.5*(speed[v]+v_new)
            speed[v] = v_new
        if len(rel) == len(move.members):
            break
    for v in move.members:
        rel.setdefault(v, hold_end)
    dur = max(rel.values())
    if dur < 1:
        return None
    plan = ld.verified_plan(move, view, duration_s=dur, release_el=rel,
                            profile={v: prof[v][:dur] for v in move.members},
                            policy_profile={v: pol[v][:dur] for v in move.members},
                            plan_id=f'c4v_{variant["name"]}_{"_".join(move.members)}',
                            source=dict(origin=ORIGIN_N2, builder='clean4_verify.n2_builder', move_id=variant.get('move_id'),
                                        state_id=ctx.get('state_id'), rule=N2_BUILDER_NOTE))
    plan['origin'] = ORIGIN_N2
    if not isinstance(mv.from_plan(plan), ld.LaneDropMove):
        raise ValueError('N2 plan did not bind to a LaneDropMove (legacy projection refused)')
    return plan


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
    """Executable group plans of one context + member solos. Builder protocol:
    ``builder(ctx, members, variant) -> plan dict | None`` (id, family, members, duration_s,
    policy_profile, profile, perturbation_profile; members x duration matrices)."""
    variants = variants_for(ctx) if variants is None else variants
    groups, meta, skipped, out_solos = [], {}, [], {}
    for var in variants:
        plan = builder(ctx, list(var['members']), var)
        if plan is None:
            skipped.append(dict(variant=var['name'], members=list(var['members']),
                                reason='builder returned None (incomplete reference or member not upstream at t0)'))
            continue
        if plan['id'] in meta:
            raise ValueError('duplicate plan id '+plan['id'])
        if (plan.get('family') not in FAMILIES+('group_hold',)
                or sorted(plan['members']) != sorted(var['members'])):
            raise ValueError('builder changed the family/members of a registered variant')
        groups.append(plan)
        out_solos[plan['id']] = solos(plan)
        meta[plan['id']] = dict(variant=var['name'], family=plan['family'], params=var['params'],
                                members=list(plan['members']), move_id=var.get('move_id'),
                                max_member_dist_t0_m=max_member_distance(ctx, plan['members']),
                                builder=plan.get('builder', getattr(builder, '__name__', str(builder))),
                                origin=plan.get('origin'), provisional=bool(plan.get('provisional')))
    return dict(groups=groups, solos=out_solos, meta=meta, skipped=skipped)


# ============================================================================ 5. freeze (MAIN freeze rule)

def _move_of(row):
    mv = row.get('move')
    if isinstance(mv, str):
        mv = json.loads(mv)
    return mv if isinstance(mv, dict) else {}


def _executable(family, params):
    if family not in SPEED_HOLD_FAMILIES:
        return False, f'family {family!r} has no executable builder in this runtime (speed-hold families only)'
    if not params or any(not isinstance(p, dict) or any(k not in p for k in ('v_w', 'hold_s', 'dmax')) for p in params):
        return False, 'member params lack v_w/hold_s/dmax'
    core = [{k: p[k] for k in p if k != 'anchor'} for p in params]
    if any(c != core[0] for c in core):
        return False, 'members carry different speed-hold params'
    if type(core[0]['hold_s']) is not int or not 0 < core[0]['hold_s'] <= PROFILE_S:
        return False, f'hold_s must be an integer in (0, {PROFILE_S}]'
    return True, ''


def move_list_path(root):
    return Path(root)/'MOVE_LIST.json'


def _cell_eligible(demand, share, cells):
    ok = int(demand) in TRAIN_CELL_DEMANDS and any(abs(float(share)-s) < 1e-9 for s in TRAIN_CELL_SHARES)
    return ok and (cells is None or cell_key(demand, share) in cells)


def _row_status(row, family, variant, params, members, eligible, desc):
    """MAIN freeze rules -> (status, reason, lane_drop_macro). Exactness = N2's converter
    (lane_drop.move_from_mined_row): NotRepresentable -> 'excluded: no exact executor'."""
    from . import lane_drop as ld
    if not eligible:
        return 'excluded_cell', 'cell outside the training-cell rule (demand 2400/2800/3200 x share 0.4/0.6)', None
    kind = row.get('r_kind') if family == R_FAMILY else family
    if family in OFF_FAMILIES or kind in OFF_FAMILIES:
        return 'excluded_family', f'{kind} is an OFF family', None
    if family in ('closing_pair', 'abreast_wall') and variant not in SPEED_HOLD_VARIANTS:
        return NOT_EXECUTABLE, f'variant {variant!r} is not a registered N3/PLANGEN hold variant', None
    try:
        move = ld.move_from_mined_row(dict(row, move=_move_of(row)), desc)
    except ld.NotRepresentable as e:
        return NO_EXACT_EXECUTOR, f'lane_drop.NotRepresentable: {e}', None
    except (ValueError, KeyError, TypeError) as e:
        return NOT_EXECUTABLE, f'N2 converter refused the row: {e!r}', None
    if family in N9_FAMILIES or kind in N9_FAMILIES:
        return NO_EXACT_EXECUTOR, f'{kind} is a lane-change family', None
    return 'pending', '', move.to_dict()


def freeze(root, sources, cells=None, *, descriptor=None, effort_matched=EFFORT_MATCHED, budget_model='worst',
           r_zero_guidance=True):
    """Sealed MOVE_LIST.json BEFORE any physics (MAIN freeze rules, MAIN_RULES verbatim): B moves of
    ``sources`` (given order, file order), then R moves the same way; a move = distinct (source, world
    seed, cell, t0, family, variant) with its per-seed rows collapsed. Every row is listed with a status:
    pending (N2-exact executable: lane_drop_macro stored), excluded_cell, excluded_family,
    'excluded: no exact executor' or not_executable_in_runtime. ``cells`` (optional) further restricts the
    training-cell rule; ``descriptor`` defaults to the process map (COOPMWM_MAP_SPEC)."""
    from .exact_grade import _seal
    root = _checked(root)
    registration(root)
    desc = descriptor or _descriptor()
    if move_list_path(root).exists():
        raise ValueError('MOVE_LIST.json already frozen; never rewritten')
    if ledger_rows(root):
        raise ValueError('charges exist: the move list must be frozen before any physics')
    cells = None if cells is None else list(cells)
    srcs = []
    for path in sources:
        p = _checked(path)
        data = p.read_bytes()
        rows = [json.loads(x) for x in data.decode('utf-8').splitlines() if x.strip()]
        srcs.append(dict(path=str(p), name=p.name, sha256=hashlib.sha256(data).hexdigest(), rows=rows))
    moves, seen = [], {}
    for arm in ('B', 'R'):
        for s in srcs:
            for row in s['rows']:
                if row.get('arm') != arm:
                    continue
                w = row.get('world') or {}
                ck = w.get('cell') if isinstance(w.get('cell'), str) else (w.get('cell_key') or row.get('cell'))
                mv = _move_of(row)
                family = row.get('family') or mv.get('family')
                variant = row.get('variant') or mv.get('variant')
                key = (s['path'], int(w['seed']), ck, int(row['t0']), family, variant)
                if key in seen:
                    m = moves[seen[key]]
                    m['seeds'] = sorted(set(m['seeds']) | {int(row.get('seed', 0))})
                    m['n_rows'] += 1
                    m['source_plan_sha256'] = sorted(set(m['source_plan_sha256']) | {row.get('plan_sha256')})
                    continue
                members = list(row.get('members') or [x['id'] for x in mv.get('members', [])])
                params = [x.get('params') for x in mv.get('members', [])] if mv.get('members') else []
                eligible = _cell_eligible(w['demand'], w['penetration'], cells)
                status, why, macro = _row_status(row, family, variant, params, members, eligible, desc)
                if eligible:
                    world = make_world(int(w['seed']), int(w['demand']), float(w['penetration']))
                    check_world(world)
                    if world['cell_key'] != ck:
                        raise ValueError(f'move world cell {ck} differs from its registry cell {world["cell_key"]}')
                else:
                    world = dict(w)
                if effort_matched is not None and s['name'] not in effort_matched:
                    status, why, macro = EXCLUDED_F3, 'source not effort-matched', None
                if arm == 'R' and r_zero_guidance:
                    status, why, macro = EXCLUDED_R_A, OPTION_A_FACT, None
                seen[key] = len(moves)
                moves.append(dict(arm=arm,
                                  source=s['path'], source_name=s['name'], source_sha256=s['sha256'],
                                  situation_id=row['situation_id'], world=world, cell_key=ck, t0=int(row['t0']),
                                  window=window_index(int(row['t0'])), family=family, variant=variant,
                                  r_kind=row.get('r_kind'), members=members, member_lanes=row.get('member_lanes'),
                                  params=params, seeds=[int(row.get('seed', 0))], n_rows=1,
                                  source_plan_sha256=[row.get('plan_sha256')], lane_drop_macro=macro,
                                  status=status, executable=status == 'pending', reason=why))
    moves, fair = _fair_order(moves, cap=int(registration(root)['cap_sim_s']), budget_model=budget_model)
    counts = {}
    for m in moves:
        counts[m['status']] = counts.get(m['status'], 0)+1
    body = dict(schema=MOVE_LIST_SCHEMA, phase=PHASE, order_rule=ORDER_RULE, main_rules=list(MAIN_RULES),
                fairaudit=fair, run_list=fair['run_list'], effort_matched_sources=list(effort_matched or ()),
                option_a=dict(fact=OPTION_A_FACT, scope=OPTION_A_SCOPE, r_zero_guidance=bool(r_zero_guidance)),
                training_cell_rule=dict(demands=list(TRAIN_CELL_DEMANDS), shares=list(TRAIN_CELL_SHARES),
                                        cells=cells),
                off_families=list(OFF_FAMILIES), n9_families=list(N9_FAMILIES),
                sources=[dict(path=s['path'], name=s['name'], sha256=s['sha256'], rows=len(s['rows']),
                              b_rows=sum(r.get('arm') == 'B' for r in s['rows']),
                              r_rows=sum(r.get('arm') == 'R' for r in s['rows'])) for s in srcs],
                moves=moves, n_moves=len(moves), n_executable=sum(m['executable'] for m in moves),
                status_counts=counts, r_procedure_fact=R_PROCEDURE_FACT,
                builder=dict(name='clean4_verify.n2_builder', origin=ORIGIN_N2, note=N2_BUILDER_NOTE,
                             converter='lane_drop.move_from_mined_row (c4/nm-moves c7983467)'),
                descriptor=dict(name=desc.name, digest=desc.digest), frozen_unix=time.time())
    _write_new(move_list_path(root), _seal(body, 'move_list_sha256'))
    return move_list(root)


def interleave(b, r):
    """FAIRAUDIT F4: B1, R1, B2, R2, ...; the longer arm's tail continues at the end."""
    out = []
    for i in range(max(len(b), len(r))):
        out += ([b[i]] if i < len(b) else [])+([r[i]] if i < len(r) else [])
    return out


def fair_cut(costs_b, costs_r, available):
    """FAIRAUDIT F4 (deterministic): per-arm budgets pro rata to the number of executable moves
    (budget_arm = available * n_arm / (n_B + n_R)); the LARGEST common fraction f such that the first
    floor(f * n_arm) moves of EACH arm fit that arm's budget; both arms are cut from the tail by the same
    fraction (never only one arm)."""
    nb, nr = len(costs_b), len(costs_r)
    n = nb+nr
    bud_b = available*nb/n if n else 0.
    bud_r = available*nr/n if n else 0.
    cands = sorted({1.0, 0.0} | {i/nb for i in range(nb+1) if nb} | {j/nr for j in range(nr+1) if nr},
                   reverse=True)
    for f in cands:
        kb, kr = math.floor(f*nb+1e-9), math.floor(f*nr+1e-9)
        cb, cr = sum(costs_b[:kb]), sum(costs_r[:kr])
        if cb <= bud_b and cr <= bud_r:
            return dict(fraction=f, k_B=kb, k_R=kr, n_B=nb, n_R=nr, budget_B=int(bud_b), budget_R=int(bud_r),
                        cost_B=int(cb), cost_R=int(cr), full_cost_B=int(sum(costs_b)), full_cost_R=int(sum(costs_r)),
                        cut=kb < nb or kr < nr)
    raise AssertionError('f = 0 always fits')


def move_quote(m, model='worst', seeds=DEFAULT_SEEDS):
    """Zero-physics upper bound of one frozen hold move (duration = its registered hold)."""
    macro = m['lane_drop_macro']
    dur = int(macro['onset'])+int(macro['hold'])
    plan = dict(id='q', members=list(macro['members']), duration_s=dur)
    solo = [dict(id=f'q__solo_{v}', members=[v], duration_s=dur) for v in macro['members']]
    cost = screen_quote(m['t0'], [plan]+solo)
    return cost+(confirm_quote(m['t0'], [plan], seeds-2) if model == 'worst' else 0)


def _fair_order(moves, *, cap, budget_model='worst'):
    """Interleaved frozen order + the FAIRAUDIT F4 run list and per-arm budgets."""
    if budget_model not in ('worst', 'screen'):
        raise ValueError('budget_model must be worst or screen')
    ex = {a: [m for m in moves if m['arm'] == a and m['status'] == 'pending'] for a in ('B', 'R')}
    worlds = {(m['arm'], collect_ident(m['world'], m['arm'])) for a in ex for m in ex[a]}
    collections = COLLECT_SIM_S*len(worlds)
    costs = {a: [move_quote(m, budget_model) for m in ex[a]] for a in ex}
    cut = fair_cut(costs['B'], costs['R'], cap-collections)
    keep = {'B': cut['k_B'], 'R': cut['k_R']}
    for a in ex:
        for i, (m, c) in enumerate(zip(ex[a], costs[a])):
            m['quote_sim_s'] = int(c)
            if i >= keep[a]:
                m['status'], m['reason'] = EXCLUDED_F4, f'budget cut by the common fraction {cut["fraction"]:.4f}'
    run = {a: [m for m in ex[a] if m['status'] == 'pending'] for a in ex}
    cut_moves = {a: [m for m in ex[a] if m['status'] != 'pending'] for a in ex}
    rest = {a: [m for m in moves if m['arm'] == a and m not in ex[a]] for a in ('B', 'R')}
    ordered = interleave(run['B'], run['R'])+interleave(cut_moves['B'], cut_moves['R'])+rest['B']+rest['R']
    for i, m in enumerate(ordered):
        m.update(index=i, move_id=f'{m["arm"]}{i:03d}_{m["situation_id"]}_{m["variant"]}',
                 executable=m['status'] == 'pending', in_run_list=m['status'] == 'pending')
    ordered = [{k: m[k] for k in ['index', 'move_id']+[k for k in m if k not in ('index', 'move_id')]}
               for m in ordered]
    fair = dict(rules=list(FAIRAUDIT_RULES), budget_model=budget_model, cap_sim_s=cap,
                collections=dict(worlds=len(worlds), sim_s=collections), available_sim_s=cap-collections,
                **cut, run_list=[m['move_id'] for m in ordered if m['in_run_list']])
    return ordered, fair


def move_list(root):
    from .exact_grade import verify_sealed
    p = move_list_path(root)
    if not p.exists():
        return None
    ml = json.loads(p.read_text(encoding='utf-8'))
    verify_sealed(ml, 'move_list_sha256')
    if ml.get('schema') != MOVE_LIST_SCHEMA:
        raise ValueError('not a clean4 MOVE_LIST')
    return ml


def move_requests(root, arm):
    """Per world (move-list order) the requested contexts {t0, members, move_ids} of executable moves."""
    ml = move_list(root)
    if ml is None:
        raise ValueError('no frozen MOVE_LIST.json')
    out = {}
    for m in ml['moves']:
        if m['arm'] != arm or not m['executable']:
            continue
        w = out.setdefault(collect_ident(m['world'], arm), dict(world=m['world'], requests={}))
        r = w['requests'].setdefault(m['t0'], dict(t0=m['t0'], members=[], move_ids=[]))
        r['members'] = list(dict.fromkeys(r['members']+m['members']))
        r['move_ids'].append(m['move_id'])
    return [dict(ident=k, world=v['world'], requests=[v['requests'][t] for t in sorted(v['requests'])])
            for k, v in out.items()]


# ============================================================================ 6. verify

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
    """Zero-physics planning quote of one context: the seed-1 screen (groups + member solos) and the
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
def _confirm_charge_bound(root, tag, arm=None):
    """Scoped probe_native binding: verify_perseed's confirmation verifications charge this ledger
    (kind-prefixed ids); the original replay_bank._confirm_charge is always restored."""
    from . import replay_bank as rb
    original = rb._confirm_charge
    rb._confirm_charge = lambda run_root, phase, kind: _charger(root, f'{tag}_{kind}', kind, arm)
    try:
        yield
    finally:
        rb._confirm_charge = original


def bank_dir(root):
    return Path(root)/'minebank'/'bank'


def _native(root, ctx, snap, passing, extra, workers, runner_factory, tail_s, health, tag):
    """probe_native.run: first-seed sv.verify of the survivors + verify_perseed.confirm_all8."""
    import torch
    from . import replay_bank as rb, sumo_verify as sv, verify_perseed as vp
    from .storage import write as swrite
    arm, sid = ctx['arm'], ctx['state_id']
    BANK = bank_dir(root)
    bank = Path(rb.bank_root(BANK, arm))
    (bank/'snapshots').mkdir(parents=True, exist_ok=True)
    with _locked(root, 'BANK.lock'):            # concurrent moves on one context share its bank snapshot
        if not (bank/'snapshots'/(sid+'.pt')).exists():
            tmp = bank/'snapshots'/f'.{sid}.{os.getpid()}.pt.tmp'
            torch.save(snap, tmp)
            os.replace(tmp, bank/'snapshots'/(sid+'.pt'))
        if not (bank/'snapshots'/(sid+'.json')).exists():
            swrite(bank/'snapshots'/(sid+'.json'), dict(
                state_id=sid, arm=arm, job=sid.rsplit('_', 1)[0], world=ctx['world'], t0=int(ctx['t0']),
                group=dict(members=ctx.get('members')), regime=None, policy_update=0,
                record_sha256=ctx['snapshot_sha256'], unix=time.time(),
                mining=dict(phase=PHASE, selection=SELECTION_RULE, moment=(ctx.get('moment') or {}).get('kind'),
                            lead=ctx.get('lead'), base=ctx.get('base'), map=MAP, move_ids=ctx.get('move_ids'))))
    ntag = f'nat_{tag}'
    r = sv.verify(snap, passing, runner_factory(bank/'sumo_verification', workers), _charger(root, f'{ntag}_first', 'first', arm),
                  'wm', workers=workers, root=bank/'sumo_verification', tail_s=tail_s)
    decision = dict(state_id=sid, verification_id=r['verification_id'], status=r['status'], policy_update=0,
                    arm=arm, available_k=len(passing), issued_unix=time.time(), matched_B=None, phase=PHASE)
    try:
        _write_new(bank/'decisions'/(sid+'.json'), decision)
    except FileExistsError:                      # another move on the same context decided first
        _write_new(bank/'decisions'/f'{sid}__{r["verification_id"]}.json', decision)
    out = dict(state_id=sid, tag=ntag, verification_id=r['verification_id'], first_status=r['status'])
    if r['status'] not in ('accepted', 'rejected'):
        out.update(status='first_'+r['status'], reason=r.get('reason'))
        return out
    receipt = dict(r, _branches=str(bank/'sumo_verification'/'branches'), _policy_version='u0')
    with _confirm_charge_bound(root, ntag, arm):
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


def _engine(root, ctx_dir, ctx, snap, groups, *, tag, seeds, workers, runner_factory, tail_s, health, take_claim,
            extra=None):
    """Screen -> gate -> native confirmation of ``groups`` on one context (budget-checked before the
    claim; the claim precedes any charge)."""
    from . import bank_v2v, sumo_verify as sv
    sid, t0, arm = ctx['state_id'], int(ctx['t0']), ctx['arm']
    base = dict(phase=PHASE, state_id=sid, ctx_dir=str(ctx_dir), arm=arm, world=ctx['world'],
                cell_key=ctx['world']['cell_key'], t0=t0, window=ctx.get('window'), lead=ctx.get('lead'),
                seed_index=1, unix=time.time(), **(extra or {}))
    if runner_factory is None:
        production_env(ctx['world'])
        runner_factory = _runner_factory
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
    left = left_for(root, arm)
    if quote > left or _stop(root):
        # refused before any dispatch: no charge, no claim (a later run is not a replay)
        why = 'STOP' if _stop(root) else f'screen quote {quote} > remaining {left}'
        _append(root, 'contexts.jsonl', [dict(base, screen_status='refused', reason=why, n_groups=len(groups),
                                              screen_quote_sim_s=quote, passing=[])])
        return dict(state_id=sid, status='refused_screen', reason=why, screen_quote_sim_s=quote)
    if not take_claim():
        return dict(state_id=sid, status='skipped_claimed')
    work = root/'screen'/tag
    with bank_v2v.enabled(bool(ctx.get('v2v', True))):     # branch subprocesses inherit C4NM_V2V
        rec = sv.verify(snap, cands, runner_factory(work, workers), _charger(root, f'scr_{tag}', 'screen', arm), 'wm',
                        workers=workers, root=work, tail_s=tail_s)
    head = dict(base, verification_id=rec['verification_id'], status=rec['status'], window_s=rec.get('window'),
                charged_sim_s=rec.get('charged_sim_s'), screen_quote_sim_s=quote, work=str(work))
    if rec['status'] not in ('accepted', 'rejected'):
        _append(root, 'screen.jsonl', [dict(head, group_id=g['id'], plan_sha256=_digest(g), family=g['family'],
                                            members=g['members'], screen='incomplete', passed=False,
                                            reason=rec.get('reason')) for g in groups])
        _append(root, 'contexts.jsonl', [dict(base, screen_status=rec['status'], n_groups=len(groups), passing=[],
                                              verification_id=rec['verification_id'])])
        return dict(state_id=sid, status='screen_'+rec['status'], reason=rec.get('reason'))
    rows = _screen_rows(rec, groups, owner, work/'branches', head)
    _append(root, 'screen.jsonl', rows)
    passing = [g for g, r in zip(groups, rows) if r['passed']]
    _append(root, 'contexts.jsonl', [dict(base, screen_status=rec['status'], verification_id=rec['verification_id'],
                                          n_groups=len(groups), passing=[g['id'] for g in passing])])
    out = dict(state_id=sid, status='screened', screened=len(rows), passing=len(passing),
               screen=[dict(group_id=r['group_id'], screen=r['screen'], reason=r.get('reason'),
                            adv=r.get('group_adv'), adv_lb95=r.get('adv_lb95'), synergy=r.get('synergy'),
                            synergy_lb95=r.get('synergy_lb95')) for r in rows])
    if not passing:
        return out
    ext = seeds-2
    try:
        check_salts(snap['episode']['id'], t0, passing, ext)
    except ValueError as e:
        _append(root, 'native.jsonl', [dict(base, status='refused_salt', reason=str(e))])
        return dict(out, native='refused_salt')
    cq = confirm_quote(t0, passing, ext, tail_s, int(snap.get('end_s', EPISODE_S)))
    left = left_for(root, arm)
    if cq > left or _stop(root):
        why = 'STOP' if _stop(root) else f'confirmation quote {cq} > remaining {left}'
        _append(root, 'native.jsonl', [dict(base, status='refused_budget', reason=why, confirm_quote_sim_s=cq,
                                            plan_ids=[g['id'] for g in passing])])
        return dict(out, native='refused_budget', reason=why)

    def stop_health():
        if _stop(root):
            raise ValueError('clean4_prelaunch_verify STOP')
    with bank_v2v.enabled(bool(ctx.get('v2v', True))):
        res = _native(root, ctx, snap, passing, ext, workers, runner_factory, tail_s, health or stop_health, tag)
    _append(root, 'native.jsonl', [dict(base, **res, confirm_quote_sim_s=cq, seeds=seeds,
                                        plan_ids=[g['id'] for g in passing])])
    return dict(out, native=res.get('status'), admitted=res.get('admitted'))


def _check_args(seeds, tail_s):
    if type(seeds) is not int or seeds < 2:
        raise ValueError('seeds must be an integer >= 2 (confirm_all8 always measures seeds 1 and 2)')
    if int(tail_s) < TAIL_S:
        raise ValueError('verification windows are never shortened below sv.TAIL_S')


def _claim(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x') as f:
            f.write(json.dumps(dict(pid=os.getpid(), unix=time.time())))
        return True
    except FileExistsError:
        return False


def verify(root, ctx_dir, groups, *, seeds=DEFAULT_SEEDS, workers=4, runner_factory=None, tail_s=TAIL_S, health=None):
    """Staged, budget-aware verification of one window moment (claimed once; never replayed). Refused once a
    MOVE_LIST is frozen (then only verify --next runs)."""
    root = _checked(root)
    registration(root)
    if move_list_path(root).exists():
        raise ValueError('MOVE_LIST.json is frozen: only verify --next runs (frozen order)')
    _check_args(seeds, tail_s)
    ctx, snap = load_context(ctx_dir)
    check_world(ctx['world'])
    if snap['episode']['world'] != ctx['world'] or int(snap['t0']) != int(ctx['t0']):
        raise ValueError('snapshot episode/t0 differ from the context')
    claim = root/'claims'/f'verify_{ctx["state_id"]}.claim'
    if claim.exists():
        return dict(state_id=ctx['state_id'], status='skipped_claimed')
    return _engine(root, ctx_dir, ctx, snap, groups, tag=ctx['state_id'], seeds=seeds, workers=workers,
                   runner_factory=runner_factory, tail_s=tail_s, health=health, take_claim=lambda: _claim(claim))


def run_context(root, ctx_dir, *, builder=provisional_builder, seeds=DEFAULT_SEEDS, workers=4, runner_factory=None,
                tail_s=TAIL_S, variants=None, names=None):
    """plans -> results/plans.jsonl (binding metadata for extract) -> verify. ``names`` restricts the
    registered slate (e.g. ('pair_g', 'wall_g')) to fit the cap; selection is by name, never outcome."""
    if move_list_path(root).exists():
        raise ValueError('MOVE_LIST.json is frozen: only verify --next runs (frozen order)')
    ctx, _ = load_context(ctx_dir)
    if names is not None:
        variants = [v for v in (variants_for(ctx) if variants is None else variants) if v['name'] in set(names)]
    built = plans(ctx, variants, builder=builder)
    _record_plans(root, ctx, ctx_dir, built)
    return verify(root, ctx_dir, built['groups'], seeds=seeds, workers=workers, runner_factory=runner_factory,
                  tail_s=tail_s)


def _record_plans(root, ctx, ctx_dir, built, extra=None):
    rows = [{**dict(phase=PHASE, state_id=ctx['state_id'], ctx_dir=str(ctx_dir), arm=ctx['arm'], world=ctx['world'],
                    cell_key=ctx['world']['cell_key'], t0=ctx['t0'], window=ctx.get('window'), lead=ctx.get('lead'),
                    plan_id=g['id'], plan_sha256=_digest(g), plan=g), **built['meta'][g['id']], **(extra or {})}
            for g in built['groups']]
    rows += [{**dict(phase=PHASE, state_id=ctx['state_id'], arm=ctx['arm'], skipped=True), **s, **(extra or {})}
             for s in built['skipped']]
    if rows:
        _append(root, 'plans.jsonl', rows)


def move_status_path(root, m):
    """Each move's OWN status file (atomic: created with O_EXCL at start, replaced at the outcome)."""
    return Path(root)/'move_status'/f'{m["index"]:04d}.json'


def move_status(root, m):
    p = move_status_path(root, m)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def _kind(m):
    return m.get('r_kind') if m['family'] == R_FAMILY else m['family']


def _runnable(m):
    """Executable now: pending speed-hold moves; pending_n9 only with a registered N9 builder."""
    return m['status'] == 'pending' or (m['status'] == 'pending_n9' and _kind(m) in FAMILY_BUILDERS)


def next_move(root):
    """The next unstarted RUNNABLE move of the frozen list (list order) or None."""
    ml = move_list(root)
    if ml is None:
        raise ValueError('no frozen MOVE_LIST.json (freeze first)')
    for m in ml['moves']:
        if _runnable(m) and not move_status_path(root, m).exists():
            return ml, m
    return ml, None


def move_ctx_dir(root, m):
    return Path(root)/'collect'/collect_ident(m['world'], m['arm'])/f'req_t{m["t0"]}'


def move_variant(m):
    params = dict(m['params'][0])
    params.setdefault('amax', AMAX)
    params.setdefault('release_x', X_B)
    if 'anchor_release_x' in params and not params.get('anchor'):
        params['anchor'] = m['members'][0]
    macro = m.get('lane_drop_macro')
    return dict(name=m['variant'], family=m['family'], params=params, move_id=m['move_id'], lane_drop_macro=macro,
                members=list(macro['members']) if macro else list(m['members']))


def _start_move(root, m):
    """Atomic claim = O_EXCL creation of the move's status file; False if the move already started."""
    p = move_status_path(root, m)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with p.open('x', encoding='utf-8') as f:
            f.write(json.dumps(dict(move_id=m['move_id'], index=m['index'], status='started', pid=os.getpid(),
                                    unix=time.time()), sort_keys=True))
            f.flush()
            os.fsync(f.fileno())
        return True
    except FileExistsError:
        return False


def _move_outcome(root, m, status, **kw):
    row = dict(phase=PHASE, index=m['index'], move_id=m['move_id'], arm=m['arm'], situation_id=m['situation_id'],
               world=m['world'], cell_key=m['cell_key'], t0=m['t0'], family=m['family'], variant=m['variant'],
               members=m['members'], status=status, unix=time.time(), **kw)
    _write_json(move_status_path(root, m), json.dumps(row, sort_keys=True, default=str))
    _append(root, 'moves.jsonl', [json.loads(json.dumps(row, default=str))])
    return row


def resolve_n9(root):
    """pending_n9 moves -> excluded_n9_deadline once every executable move has finished and no N9 builder
    is registered (MAIN: they join only if N9 merges before the last executable move finishes)."""
    ml = move_list(root)
    if ml is None:
        return []
    if any((move_status(root, m) or {}).get('status') in (None, 'started') for m in ml['moves'] if m['status'] == 'pending'):
        return []
    out = []
    for m in ml['moves']:
        if m['status'] == 'pending_n9' and not _runnable(m) and _start_move(root, m):
            out.append(_move_outcome(root, m, 'excluded_n9_deadline',
                                     reason='no N9 builder merged before the last executable move finished'))
    return out


def run_move(root, move_id, *, builder=None, seeds=DEFAULT_SEEDS, workers=4, runner_factory=None,
             tail_s=TAIL_S, health=None):
    """verify --move: ONE specific frozen move (refuses unknown, excluded/not-runnable and already started
    moves). Refusals before the claim (not collected yet, budget, STOP) leave the move unstarted; everything
    after the claim is a reported outcome in the move's own status file and the move is never re-run."""
    root = _checked(root)
    registration(root)
    _check_args(seeds, tail_s)
    ml = move_list(root)
    if ml is None:
        raise ValueError('no frozen MOVE_LIST.json (freeze first)')
    m = next((x for x in ml['moves'] if x['move_id'] == move_id), None)
    if m is None:
        raise ValueError(f'unknown move {move_id!r}')
    if not _runnable(m):
        raise ValueError(f'move {move_id} has status {m["status"]}: never run')
    if move_status_path(root, m).exists():
        raise ValueError(f'move {move_id} already started; a move is never re-run')
    if _stop(root):
        return dict(status='refused', reason='STOP', move_id=move_id)
    ident = collect_ident(m['world'], m['arm'])
    if not (Path(root)/'collect'/ident/'COLLECTION_RECEIPT.json').exists():
        return dict(status='refused', reason=f'world not collected yet: collect --ident {ident} first',
                    move_id=move_id)
    d = move_ctx_dir(root, m)
    if not (d/'CONTEXT.json').exists():
        if not _start_move(root, m):
            return dict(status='skipped_claimed', move_id=move_id)
        return _move_outcome(root, m, 'context_not_captured', reason='collection exists without this t0 context')
    ctx, snap = load_context(d)
    check_world(ctx['world'])
    if ctx['world'] != m['world'] or int(ctx['t0']) != int(m['t0']) or ctx['arm'] != m['arm']:
        raise ValueError('requested context does not match its frozen move')
    up = {c['id'] for c in (ctx.get('geometry') or {}).get('zone') or []}
    missing = [v for v in m['members'] if v not in up]
    if missing:
        if not _start_move(root, m):
            return dict(status='skipped_claimed', move_id=move_id)
        return _move_outcome(root, m, 'member_absent_or_past_drop', missing=missing,
                             reason='frozen member not upstream of the drop at t0 on this base')
    use = (builder or n2_builder) if m['status'] == 'pending' else FAMILY_BUILDERS[_kind(m)]
    built = plans(ctx, [move_variant(m)], builder=use)
    extra = dict(move_id=m['move_id'], move_index=m['index'])
    if not built['groups']:
        if not _start_move(root, m):
            return dict(status='skipped_claimed', move_id=move_id)
        _record_plans(root, ctx, d, built, extra)
        return _move_outcome(root, m, 'unbuildable', reason=built['skipped'][0]['reason'])

    def take():
        ok = _start_move(root, m)
        if ok:
            _record_plans(root, ctx, d, built, extra)
        return ok
    res = _engine(root, d, ctx, snap, built['groups'], tag=f'{ctx["state_id"]}__m{m["index"]:04d}', seeds=seeds,
                  workers=workers, runner_factory=runner_factory, tail_s=tail_s, health=health, take_claim=take,
                  extra=extra)
    if res['status'] in ('refused_screen', 'skipped_claimed'):
        return dict(res, move_id=move_id)                 # not claimed: the move stays unstarted
    status = (res.get('native') and f"native_{res['native']}") or res['status']
    if res.get('native') == 'confirmed':
        status = 'confirmed_admitted' if res.get('admitted') else 'confirmed_not_admitted'
    elif res['status'] == 'screened' and not res.get('passing'):
        status = 'screen_rejected'
    out = _move_outcome(root, m, status, plan_id=built['groups'][0]['id'], plan_sha256=_digest(built['groups'][0]),
                        origin=built['groups'][0].get('origin'), result=res)
    resolve_n9(root)
    return out


def run_next(root, **kw):
    """verify --next: the next unstarted runnable move of the frozen list (list order)."""
    ml, m = next_move(root)
    if m is None:
        resolve_n9(root)
        return dict(status='no_moves_left', n_moves=ml['n_moves'])
    return run_move(root, m['move_id'], **kw)


def moves_summary(root):
    ml = move_list(root)
    if ml is None:
        return None
    st = {}
    for m in ml['moves']:
        s = (move_status(root, m) or {}).get('status') or ('unstarted' if _runnable(m) else m['status'])
        st[s] = st.get(s, 0)+1
    return dict(move_list_sha256=ml['move_list_sha256'], n_moves=ml['n_moves'], n_executable=ml['n_executable'],
                frozen_status_counts=ml.get('status_counts'), by_status=st,
                not_run=[(m['move_id'], m['status']) for m in ml['moves'] if not _runnable(m)])


# ============================================================================ 7. extract

def mined_ok(example, plan, families=MINED_FAMILIES):
    """probe_native.extract / wallmine rule for one per-seed all8 example."""
    syn = example.get('synergy') or {}
    win = (example.get('tape') or {}).get('branch', {}).get('window') or {}
    lb = syn.get('lower95_one_sided_veh_s')
    return bool(plan and plan.get('family') in families and len(plan['members']) >= 2 and example.get('admitted')
                and example.get('cooperative') is True and example.get('safety_ok') is not False
                and not win.get('collisions') and not win.get('teleports') and lb is not None and lb > 0)


def lost(synergy, advantage):
    """W2 per-seed rule -> (basis, value) when the group LOST on that seed, else None."""
    if _finite(synergy):
        return ('synergy', float(synergy)) if synergy < 0 else None
    if synergy is None and _finite(advantage):
        return ('advantage', float(advantage)) if advantage < 0 else None
    return None


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

    def cell_of(k):
        return cells.setdefault(k, dict(screened_moments=0, robust_moments=0, b_examples=[], screened_with_lead=0,
                                        refused_or_unscreened=0, r_screened_moments=0, r_robust_moments=0,
                                        r_examples=0))
    for key, c in final.items():
        d = cell_of(c['cell_key'])
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
        d = cell_of(r['cell'])
        if r['arm'] != 'B':
            d['r_examples'] += 1
            continue
        dist = r.get('max_member_dist_t0_m')
        d['b_examples'].append(dict(id=f"{r['verification_id']}#{r['plan_sha256'][:16]}#s{r['seed']}",
                                    reachable=dist is not None and dist <= float(range_m), max_member_dist_t0_m=dist,
                                    robust=(r['arm'], r['situation_id'], r['plan_sha256']) in robust_plans))
    out = dict(schema=SCHEMA+'.potential_inputs', rule=POTENTIAL_RULE, range_m=float(range_m), cells=cells,
               denominator_note=('screened_moments counts every re-screened B context of the cell; with a frozen move '
                                 'list the contexts are the N3/PLANGEN-admitted moments, so the share is conditional '
                                 'on that earlier admission (screened_with_lead reports how many carry an N3 lead)'))
    if cell is not None:
        out['chosen_cell'] = cell
        out['chosen'] = cells.get(cell, dict(screened_moments=0, robust_moments=0, b_examples=[]))
    return out


def _frames_for(ctx_dir):
    p = Path(ctx_dir)/'FRAMES.json' if ctx_dir else None
    if p is None or not p.exists():
        return None
    rec = json.loads(p.read_text(encoding='utf-8'))
    return rec if rec.get('complete') else None


def _neg_frame_row(*, kind, frames, arm, world, t0, members, family, **kw):
    return dict(schema=NEG_FRAMES_SCHEMA, kind=kind, world_seed=int(world['seed']), world=world, t0=int(t0),
                frames=frames['frames'], frames_sha256=frames['frames_sha256'], members=list(members or []),
                family=family, arm=arm, source=PHASE, **kw)


def extract(root, *, range_m=100.0, cell=None, families=MINED_FAMILIES):
    """Bank records -> CLEAN4_MOVES_B/R.jsonl, CLEAN4_NEGATIVES.jsonl, CLEAN4_NEGATIVE_FRAMES.jsonl,
    POTENTIAL_INPUTS.json."""
    from . import replay_bank as rb, verify_perseed as vp
    from .exact_grade import verify_sealed
    root = _checked(root)
    reg = registration(root)
    BANK = bank_dir(root)
    meta = {(r['state_id'], r['plan_sha256']): r for r in _jsonl(root/'results'/'plans.jsonl') if not r.get('skipped')}
    contexts = _jsonl(root/'results'/'contexts.jsonl')
    ctx_dirs = {c['state_id']: c.get('ctx_dir') for c in contexts}
    mined, negatives, examples, neg_frames = [], [], [], []
    frames_missing = 0
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
            exs = c['per_seed_v1']['examples']
            n_reps = 2+max([0]+[int(e['seed_index'])-2 for e in exs])
            for e in exs:
                plan = e['plan']
                psha = _digest(plan)
                m = next((v for (s, p), v in meta.items() if p == psha and s.startswith(str(ep.get('id')))), {})
                sid = m.get('state_id') or f"{ep.get('id')}_t{c['t0']}"
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
                              variant=m.get('variant'), move_id=m.get('move_id'), cell=cellk,
                              max_member_dist_t0_m=m.get('max_member_dist_t0_m'),
                              provisional_builder=m.get('provisional'), origin=plan.get('origin'), n_reps=n_reps)
                if ok:
                    dist = m.get('max_member_dist_t0_m')
                    mined.append(dict(common, schema=MINED_SCHEMA, replay_sources={}, seed=int(e['seed_index']),
                        duration_s=int(plan['duration_s']), profile=plan['profile'],
                        lane_drop_macro=plan.get('lane_drop_macro'), member_release_el=plan.get('member_release_el'),
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
                    continue
                negatives.append(dict(common, stage='native', seed=int(e['seed_index']), plan_id=e['plan_id'],
                    admitted=e.get('admitted'), status=e.get('status'), reason=e.get('reason'),
                    cooperative=e.get('cooperative'), cooperative_status=e.get('cooperative_status'),
                    advantage_veh_s=e.get('advantage_veh_s'), adv_lower95=e.get('lower95_one_sided_veh_s'),
                    synergy_veh_s=syn.get('pooled_total_veh_s'), synergy_lower95=syn.get('lower95_one_sided_veh_s'),
                    collisions=win.get('collisions'), teleports=win.get('teleports')))
                lab = lost(syn.get('pooled_total_veh_s') if syn else None, e.get('advantage_veh_s'))
                if lab:
                    fr = _frames_for(ctx_dirs.get(sid))
                    if fr is None:
                        frames_missing += 1
                    else:
                        neg_frames.append(_neg_frame_row(
                            kind='lost', frames=fr, arm=arm, world=ep.get('world'), t0=c['t0'], members=plan['members'],
                            family=plan.get('family'), stage='native', basis=lab[0], seed=int(e['seed_index']),
                            situation_id=sid, verification_id=vid, plan_sha256=psha, move_id=m.get('move_id'),
                            synergy_veh_s={s_: syn['pooled_total_veh_s']} if _finite(syn.get('pooled_total_veh_s')) else {},
                            advantage_veh_s={s_: e['advantage_veh_s']} if _finite(e.get('advantage_veh_s')) else {}))
    for r in _jsonl(root/'results'/'screen.jsonl'):
        if r.get('passed'):
            continue        # went to native confirmation; its per-seed rows are above
        m = meta.get((r['state_id'], r.get('plan_sha256'))) or {}
        negatives.append(dict(arm=r['arm'], stage='screen', seed=1, situation_id=r['state_id'], world=r['world'],
                              t0=r['t0'], plan_id=r['group_id'], plan_sha256=r.get('plan_sha256'), family=r.get('family'),
                              members=r.get('members'), map=MAP, source=PHASE, lead=r.get('lead'),
                              variant=m.get('variant'), move_id=r.get('move_id'), cell=r.get('cell_key'),
                              screen=r.get('screen'), reason=r.get('reason'), group_adv=r.get('group_adv'),
                              adv_lower95=r.get('adv_lb95'), synergy_veh_s=r.get('synergy'),
                              synergy_lower95=r.get('synergy_lb95'), safety=r.get('safety'),
                              screen_verification_id=r.get('verification_id')))
        lab = lost(r.get('synergy'), r.get('group_adv'))
        if lab:
            fr = _frames_for(r.get('ctx_dir'))
            if fr is None:
                frames_missing += 1
            else:
                neg_frames.append(_neg_frame_row(
                    kind='lost', frames=fr, arm=r['arm'], world=r['world'], t0=r['t0'], members=r.get('members'),
                    family=r.get('family'), stage='screen', basis=lab[0], seed=1, situation_id=r['state_id'],
                    verification_id=r.get('verification_id'), plan_sha256=r.get('plan_sha256'), move_id=r.get('move_id'),
                    synergy_veh_s={'1': r['synergy']} if _finite(r.get('synergy')) else {},
                    advantage_veh_s={'1': r['group_adv']} if _finite(r.get('group_adv')) else {}))
    for p in sorted((root/'collect').glob('*/frames/frames_t*.json')) if (root/'collect').exists() else ():
        rec = json.loads(p.read_text(encoding='utf-8'))
        if not rec.get('complete'):
            frames_missing += 1
            continue
        neg_frames.append(_neg_frame_row(kind='requested', frames=rec, arm=rec.get('arm'), world=rec['world'],
                                         t0=rec['t0'], members=[], family=None, label=None,
                                         situation_id=f"{rec.get('state_ident')}_f{rec['t0']}"))
    pot = potential_inputs(contexts, mined, examples, range_m=range_m, cell=cell)
    for arm in ('B', 'R'):
        _write_json(root/f'CLEAN4_MOVES_{arm}.jsonl', ''.join(json.dumps(r, sort_keys=True)+'\n'
                                                                for r in mined if r['arm'] == arm))
    _write_json(root/'CLEAN4_NEGATIVES.jsonl', ''.join(json.dumps(r, sort_keys=True)+'\n' for r in negatives))
    _write_json(root/'CLEAN4_NEGATIVE_FRAMES.jsonl', ''.join(json.dumps(r, sort_keys=True)+'\n' for r in neg_frames))
    pot['sources'] = [dict(path=str(root/name), sha256=hashlib.sha256((root/name).read_bytes()).hexdigest())
                      for name in ('CLEAN4_MOVES_B.jsonl', 'CLEAN4_NEGATIVES.jsonl', 'CHARGES.jsonl', 'MOVE_LIST.json')
                      if (root/name).exists()]
    resolve_n9(root)
    pot['moves'] = moves_summary(root)
    pot.update(spent_sim_s=spent(root), cap_sim_s=reg['cap_sim_s'])
    _write_json(root/'POTENTIAL_INPUTS.json', pot)
    return dict(mined_B=sum(r['arm'] == 'B' for r in mined), mined_R=sum(r['arm'] == 'R' for r in mined),
                negatives=len(negatives), negative_frames=len(neg_frames), frames_missing=frames_missing,
                examples=len(examples), cells=sorted(pot['cells']), moves=pot['moves'])


def contexts(root, arm='B'):
    """Every run-list move of ``arm`` (FROZEN order) with a completed first-seed verification = its seed-1
    screen sv.verify (every run move gets one, whatever its outcome: metadata only, never an outcome
    filter). Items: seed, cell_key, t0, move_id, state_id, snapshot_path (the decision snapshot of the
    move context - the same state the bank snapshot copies), receipt_path (the sealed seed-1 sv.verify
    receipt: episode/window/verification_id), group_plan (the exact frozen N2 plan that was verified).
    Checks: snapshot binding sha, episode role 'train', world seed = the move world seed."""
    import torch
    from . import sumo_verify as sv
    from .exact_grade import verify_sealed
    root = _checked(root)
    ml = move_list(root)
    if ml is None:
        raise ValueError('no frozen MOVE_LIST.json')
    plans_by = {r['move_id']: r for r in _jsonl(root/'results'/'plans.jsonl') if r.get('move_id') and not r.get('skipped')}
    screen_by = {}
    for r in _jsonl(root/'results'/'screen.jsonl'):
        if r.get('move_id'):
            screen_by.setdefault(r['move_id'], r)
    out = []
    for m in ml['moves']:
        if m['arm'] != arm or not m.get('in_run_list', m['status'] == 'pending'):
            continue
        scr, plan_row = screen_by.get(m['move_id']), plans_by.get(m['move_id'])
        if scr is None or plan_row is None or scr.get('status') not in ('accepted', 'rejected'):
            continue                                       # no completed first-seed verification
        receipt_path = Path(scr['work'])/'verifications'/(scr['verification_id']+'.json')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        verify_sealed(receipt, 'receipt_sha256')
        if receipt['status'] not in ('accepted', 'rejected') or plan_row['plan']['id'] not in receipt['candidate_ids']:
            raise ValueError(f'{m["move_id"]}: screen receipt does not bind its frozen plan')
        d = move_ctx_dir(root, m)
        ctx = json.loads((d/'CONTEXT.json').read_text(encoding='utf-8'))
        snap = torch.load(d/'SNAPSHOT.pt', weights_only=False, map_location='cpu')
        if sv.snapshot_sha(snap) != ctx['snapshot_sha256'] or receipt['snapshot_sha256'] != ctx['snapshot_sha256']:
            raise ValueError(f'{m["move_id"]}: snapshot binding mismatch')
        if snap['episode'].get('role') != 'train' or snap['episode']['world']['seed'] != m['world']['seed']:
            raise ValueError(f'{m["move_id"]}: snapshot is not a TRAIN state of its world')
        out.append(dict(seed=int(m['world']['seed']), cell_key=m['cell_key'], t0=int(m['t0']), move_id=m['move_id'],
                        state_id=ctx['state_id'], snapshot_path=str(d/'SNAPSHOT.pt'), receipt_path=str(receipt_path),
                        group_plan=plan_row['plan'], screen_status=receipt['status']))
    return out


def status(root):
    root = _checked(root)
    reg = registration(root)
    out = dict(phase=PHASE, cap_sim_s=reg['cap_sim_s'], spent_sim_s=spent(root), base=reg['base'])
    out['remaining_sim_s'] = out['cap_sim_s']-out['spent_sim_s']
    for name in ('contexts', 'screen', 'native', 'moves'):
        out[name] = len(_jsonl(root/'results'/f'{name}.jsonl'))
    out['screen_pass'] = sum(bool(r.get('passed')) for r in _jsonl(root/'results'/'screen.jsonl'))
    out['native_admitted'] = sum(bool(r.get('admitted')) for r in _jsonl(root/'results'/'native.jsonl'))
    ml = move_list(root)
    if ml is not None:
        nxt = next_move(root)[1]
        out.update(moves_summary=moves_summary(root), next_move=None if nxt is None else nxt['move_id'],
                   runnable_unstarted=[m['move_id'] for m in ml['moves']
                                       if _runnable(m) and not move_status_path(root, m).exists()],
                   arms={a: arm_progress(root, a, ml) for a in ('B', 'R')})
    out['stop'] = _stop(root)
    return out


def arm_progress(root, arm, ml=None):
    """Run-list progress of one arm: done iff every run-list move of the arm has a final outcome."""
    ml = ml or move_list(root)
    run = [m for m in ml['moves'] if m['arm'] == arm and m.get('in_run_list', m['status'] == 'pending')]
    st = [(move_status(root, m) or {}).get('status') for m in run]
    return dict(run_list=len(run), finished=sum(s not in (None, 'started') for s in st),
                started=sum(s == 'started' for s in st), unstarted=sum(s is None for s in st),
                done=all(s not in (None, 'started') for s in st), budget=arm_budget(root, arm),
                spent=arm_spent(root, arm))


# ============================================================================ 8. CLI

def _load_attr(spec):
    mod, _, attr = spec.partition(':')
    m = importlib.import_module(mod)
    return getattr(m, attr) if attr else m


def _nice():
    if os.name == 'posix':
        os.nice(max(0, 19-os.nice(0)))


def _frame_times(path):
    """FRAMES file: JSON list of {seed, cell_key, t0} -> {(seed, cell_key): [t0, ...]}."""
    out = {}
    for r in json.loads(Path(path).read_text(encoding='utf-8')) if path else []:
        out.setdefault((int(r['seed']), r['cell_key']), []).append(int(r['t0']))
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python -m tools.coopmwm.clean4_verify', description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('init')
    p.add_argument('--root', required=True)
    p.add_argument('--cap', type=int, default=PHASE_CAP)
    p.add_argument('--base', default='frozen_u0', choices=BASES)
    p.add_argument('--note')
    p = sub.add_parser('freeze')
    p.add_argument('--root', required=True)
    p.add_argument('--sources', nargs='+', required=True, help='move JSONL files (B first-pass order, R second)')
    p.add_argument('--cells', nargs='+', help='optional extra cell-key restriction (default: the MAIN training-cell '
                                              'rule, demand 2400/2800/3200 x share 0.4/0.6)')
    p = sub.add_parser('collect')
    p.add_argument('--root', required=True)
    p.add_argument('--arm', required=True, choices=('B', 'R'))
    p.add_argument('--weights', required=True, help="torch file with a 'policy' state dict (frozen u0)")
    p.add_argument('--from-moves', action='store_true', help='worlds + requested t0 of the frozen move list')
    p.add_argument('--worlds', help='seed:demand:penetration[,...] (window-moment collection, no move list)')
    p.add_argument('--frames', help='JSON list of {seed, cell_key, t0}: extra whitelisted frame times')
    p.add_argument('--leads', nargs='*', default=(), help='NM2_SUMMARY.json and/or mined-move JSONL (N3/N8)')
    p.add_argument('--runtime', default=RUNTIME_DIR)
    p.add_argument('--checkpoint', default=CHECKPOINT)
    p.add_argument('--deterministic', action='store_true', help='deterministic residual (branches still sample)')
    p.add_argument('--limit', type=int, default=0, help='at most N worlds this invocation (0 = all)')
    p.add_argument('--ident', help='with --from-moves: collect only this world ident (c4v_<arm>_<cell>_s<seed>)')
    p = sub.add_parser('verify')
    p.add_argument('--root', required=True)
    p.add_argument('--move', help='run ONE specific frozen move id (refuses unknown/excluded/started)')
    p.add_argument('--next', action='store_true', help='the next unrun move of the frozen list')
    p.add_argument('--count', type=int, default=1, help='with --next: run up to N moves in order')
    p.add_argument('--ctx', nargs='*', default=())
    p.add_argument('--queue', help='file with one ctx dir per line (no move list only)')
    p.add_argument('--seeds', type=int, default=DEFAULT_SEEDS)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--tail', type=int, default=TAIL_S)
    p.add_argument('--builder', help='MODULE:FUNCTION (default: n2_builder for --move/--next, provisional for --ctx)')
    p.add_argument('--variants', help='comma list restricting the window-moment slate')
    p = sub.add_parser('quote', help='screen/confirmation sim-s quotes (no physics, no charge)')
    p.add_argument('--root', required=True)
    p.add_argument('--next', action='store_true')
    p.add_argument('--ctx', nargs='*', default=())
    p.add_argument('--seeds', type=int, default=DEFAULT_SEEDS)
    p.add_argument('--variants')
    p = sub.add_parser('extract')
    p.add_argument('--root', required=True)
    p.add_argument('--range-m', type=float, default=100.0, help='V2V range (MAIN registered 100 m)')
    p.add_argument('--cell')
    p = sub.add_parser('contexts', help='frozen-order Plan A contexts with a completed first-seed verification')
    p.add_argument('--root', required=True)
    p.add_argument('--arm', default='B', choices=('B', 'R'))
    p.add_argument('--out', required=True)
    p = sub.add_parser('status')
    p.add_argument('--root', required=True)
    p.add_argument('--arm', choices=('B', 'R'))
    p.add_argument('--done', action='store_true', help='with --arm: exit 3 unless every run-list move finished')
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.cmd == 'init':
        print(json.dumps(dict(registration_sha256=init(root, a.cap, base=a.base, note=a.note)['registration_sha256'])))
    elif a.cmd == 'freeze':
        ml = freeze(root, a.sources, a.cells)
        print(json.dumps(dict(move_list_sha256=ml['move_list_sha256'], n_moves=ml['n_moves'],
                              n_executable=ml['n_executable'],
                              moves=[(m['move_id'], m['status']) for m in ml['moves']])))
    elif a.cmd == 'collect':
        import torch
        _nice()
        rt = runtime(a.runtime, a.checkpoint, deterministic=a.deterministic)
        geom = geometry()
        saved = torch.load(a.weights, weights_only=False, map_location='cpu')
        weights = saved['policy'] if isinstance(saved, dict) and 'policy' in saved else saved   # = newmap_sanity
        wsha = hashlib.sha256(Path(a.weights).read_bytes()).hexdigest()
        leads = load_leads(a.leads) if a.leads else []
        frames = _frame_times(a.frames)
        if a.from_moves:
            jobs = [(j['world'], j['requests'], ()) for j in move_requests(root, a.arm)
                    if a.ident is None or j['ident'] == a.ident]
            if a.ident and not jobs:
                raise SystemExit(f'{a.ident} is not a world of a runnable {a.arm} move in the frozen list')
        else:
            if move_list_path(root).exists():
                raise SystemExit('MOVE_LIST.json is frozen: collect --from-moves only')
            jobs = []
            for spec in (a.worlds or '').split(','):
                if spec:
                    s, d, pen = spec.split(':')
                    jobs.append((make_world(int(s), int(d), float(pen)), None, WINDOWS))
        done = 0
        for w, reqs, wins in jobs:
            if _stop(root) or (a.limit and done >= a.limit):
                break
            if (root/'collect'/collect_ident(w, a.arm)/'COLLECTION_RECEIPT.json').exists():
                continue
            try:
                saved = collect(root, w, a.arm, weights, geom=geom, rt=rt, leads=leads, weights_sha256=wsha,
                                requests=reqs, windows=wins, frame_times=frames.get((w['seed'], w['cell_key'])))
                done += 1
                print(json.dumps(dict(world=collect_ident(w, a.arm), saved=saved)), flush=True)
            except Exception as e:
                import traceback
                print(json.dumps(dict(world=collect_ident(w, a.arm), error=repr(e),
                                      tb=traceback.format_exc()[-2000:])), flush=True)
                break
    elif a.cmd == 'verify':
        _nice()
        builder = _load_attr(a.builder) if a.builder else None
        if a.move:
            print(json.dumps(run_move(root, a.move, builder=builder, seeds=a.seeds, workers=a.workers, tail_s=a.tail),
                             default=str), flush=True)
            return
        if a.next:
            for _ in range(max(1, a.count)):
                res = run_next(root, builder=builder, seeds=a.seeds, workers=a.workers, tail_s=a.tail)
                print(json.dumps(res, default=str), flush=True)
                if res.get('status') in ('no_moves_left', 'refused', 'refused_screen', 'skipped_claimed'):
                    break
            return
        if move_list_path(root).exists():
            raise SystemExit('MOVE_LIST.json is frozen: only verify --next runs (frozen order)')
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
                res = run_context(root, d, builder=builder or provisional_builder, seeds=a.seeds, workers=a.workers, tail_s=a.tail,
                                  names=a.variants.split(',') if a.variants else None)
                print(json.dumps(res), flush=True)
            except Exception as e:
                import traceback
                print(json.dumps(dict(ctx=d, error=repr(e), tb=traceback.format_exc()[-3000:])), flush=True)
                if 'cap' in str(e):
                    break
    elif a.cmd == 'quote':
        if a.next:
            ml, m = next_move(root)
            if m is None:
                print(json.dumps(dict(status='no_moves_left')))
            elif not (move_ctx_dir(root, m)/'CONTEXT.json').exists():
                print(json.dumps(dict(move_id=m['move_id'], status='not_collected')))
            else:
                ctx, _ = load_context(move_ctx_dir(root, m))
                groups = plans(ctx, [move_variant(m)], builder=n2_builder)['groups']
                print(json.dumps(dict(move_id=m['move_id'], **quote_moment(ctx, groups, seeds=a.seeds),
                                      remaining_sim_s=remaining(root))))
        names = set(a.variants.split(',')) if a.variants else None
        for d in a.ctx:
            ctx, _ = load_context(d)
            vs = [v for v in variants_for(ctx) if names is None or v['name'] in names]
            print(json.dumps(dict(ctx=d, **quote_moment(ctx, plans(ctx, vs)['groups'], seeds=a.seeds),
                                  remaining_sim_s=remaining(root))))
    elif a.cmd == 'extract':
        print(json.dumps(extract(root, range_m=a.range_m, cell=a.cell)))
    elif a.cmd == 'contexts':
        rows = contexts(root, a.arm)
        _write_json(a.out, json.dumps(rows, sort_keys=True, default=str))
        print(json.dumps(dict(out=a.out, n=len(rows), seeds=sorted({r['seed'] for r in rows}))))
    elif a.cmd == 'status':
        if a.arm:
            prog = arm_progress(root, a.arm)
            print(json.dumps(dict(arm=a.arm, **prog)))
            if a.done and not prog['done']:
                raise SystemExit(3)
            return
        print(json.dumps(status(root)))


if __name__ == '__main__':
    main(sys.argv[1:])
