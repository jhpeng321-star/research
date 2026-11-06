"""BOOST-PRE: offline imitation warm start of the clean4 cooperation-layer trigger (DEFAULT OFF).

Owner 2026-11-06 (every lever that raises the chance of learning group cooperation): before
RL update 1, pre-train the cooperation layer's trigger (move-choice) logits by imitation so B
starts from a layer that already knows when/how to cooperate instead of from zero.

* B imitates its OWN SUMO-verified examples (``*_MOVES_B`` rows), R imitates its OWN
  random-procedure examples (``*_MOVES_R`` rows). Both learn "do not act" (target ``none``)
  from ONE shared negatives set. B and R get identical effort: the same number of optimizer
  steps, minibatch size, loss, optimizer, hyperparameters, sampling seed and mass rule; the
  receipt's ``effort`` blocks must be equal (``effort_equal``).
* Only the trigger LOGIT parameters train (``trigger_head`` plus any W2/extension logit
  columns). The frozen MAPPO base (``layer.base`` == ``trigger.base_policy``), the
  cooperation residual and the trigger value head are untouched; the base is hashed
  before/after and the hash travels with the checkpoint.
* Observable inputs only. Decision contexts come either from recorded bank pair contexts via
  the same reader as clean3/clean4 supervision (``bank_breadth.visit`` +
  ``observable_contexts``) or from sensor-whitelisted frames (vid, s_m, v_mps, lane, is_cav,
  length_m) replayed through the online causal observer (``EpisodePolicy.observe`` at t0-1
  then t0), exactly like W2 ``wall_negatives``. Row metadata (cell, demand, penetration,
  synergy, advantage, statistics) only SELECTS/LABELS examples and never enters a tensor.
* Rows: wz3to2 TRAIN seeds 989200-989599 and the registered training cells only. Any
  989600-989699 (HELD-OUT) row refuses the whole run. Split by TRAIN world seed with ONE
  assignment common to B, R and the negatives; held-out imitation accuracy is reported.
* DEFAULT OFF: a config without ``coop_warmstart`` leaves prepare/runtime byte-identical
  (``sidecar_state`` returns exactly the clean3 inline sidecar dict).

Negative "do not act" rule (RULE_NEGATIVE): a row is a do-not-act label iff its own measured
evaluation FAILED (group not better than base on the screen, no synergy vs the best solo, or
a failed replicate). Rows that were only outranked ("not the best-T variant", "r0_not_picked",
"not among the top-2 T random proposals") or never evaluated ("r0_only_budget") are NOT
failures and are excluded (counted).

Modes
  bank: N4 Plan A re-verified coopmwm-bank rows (CLEAN4_MOVES_{B,R}.jsonl; banks readable at
        their run_root) + CLEAN4_NEGATIVE_FRAMES.jsonl (W2 frame format) + negatives files.
  dry:  raw N3-funnel rows before Plan A. Observations are SYNTHETIC frames built from row
        metadata and every move is the stand-in family 'wall'. Plumbing only: dry
        checkpoints are refused by ``apply`` unless the registration says allow_dry.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch

from .protocol import digest

SCHEMA = 'coopmwm-coop-warmstart-v1'
CHECKPOINT_SCHEMA = 'coopmwm-coop-warmstart-checkpoint-v1'
CONFIG_KEY = 'coop_warmstart'
RULE = ('BOOST-PRE warm start v1: trigger-logit imitation before u1; B from its own *_MOVES_B '
        'examples, R from its own *_MOVES_R examples, one shared do-not-act negatives set; '
        'identical steps/minibatch/loss/optimizer/hyperparameters/sampling seed for B and R; '
        'frozen MAPPO, residual and trigger value untouched; observable inputs only; TRAIN '
        'seeds 989200-989599 and registered training cells only; seed-level held-out split.')
RULE_NEGATIVE = ('do-not-act iff the row\'s own evaluation failed: screen group-vs-base '
                 '(LB95(T)<=0 or T<1), solos synergy (LB95(S)<=0 or S<1), or a failed '
                 'replicate r1/r2; outranked or unevaluated rows are excluded and counted')
MAP = 'wz3to2'
TRAIN_SEEDS = (989200, 989599)
HELDOUT_EVAL_SEEDS = (989600, 989699)
# N1 crossed wz3to2 registry (map_scenarios): demand x penetration. Labels select rows only.
TRAIN_CELLS = tuple((d, p) for d in (2400, 2800, 3200) for p in (.2, .4, .6))
FRAME_KEYS = ('vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m')
POSITIVE_SCHEMA = 'coopmwm-mined-move-v1'
DEFAULT_SOURCES = dict(
    B=('NEWMAP_MOVES_B.jsonl', 'COOPKINDS_MOVES_B.jsonl', 'PLANGEN_MOVES_B.jsonl', 'LOWCAV_MOVES_B.jsonl'),
    R=('NEWMAP_MOVES_R.jsonl', 'COOPKINDS_MOVES_R.jsonl', 'PLANGEN_MOVES_R.jsonl'),
    negatives=('NEWMAP_NEGATIVES.jsonl', 'COOPKINDS_NEGATIVES.jsonl', 'PLANGEN_NEGATIVES.jsonl',
               'LOWCAV_NEGATIVES.jsonl'))
# N8 (coopkinds) names -> N9 (nm_families) names. Labels only.
FAMILY_ALIASES = dict(platoon='platoon_headway', zipper='zipper_order', discharge='discharge_sync',
                      vsl_rows='harmonize_rows')
LANE_DROP_JOINT = ('closing_pair', 'abreast_wall')
DRY_FAMILY = 'wall'
# Registered optimisation (identical for B and R; a per-arm key is refused).
TRAINING = dict(steps=800, batch_size=32, learning_rate=1e-3, max_grad_norm=.5,
                positive_mass=.5, sample_seed=20261106, heldout_fraction=.25,
                split_salt='boost-pre-split-v1', slate_seed=20261106)
TRAINING_KEYS = frozenset(TRAINING)


class Unrepresentable(ValueError):
    """The row's move has no registered runtime representation (excluded and counted)."""


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _guard(value):
    from .track2_backend import _path
    return _path(value)


def state_sha256(module):
    """Exact bytes of a module's state (names, dtypes, shapes, values), order-stable."""
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.float().numpy().tobytes())
    return h.hexdigest()


def training_settings(raw=None):
    raw = dict(raw or {})
    unknown = set(raw)-TRAINING_KEYS
    if unknown:
        raise ValueError(f'unregistered warm-start training keys {sorted(unknown)} (no per-arm keys)')
    out = dict(TRAINING, **raw)
    for key in ('steps', 'batch_size', 'sample_seed', 'slate_seed'):
        if type(out[key]) is not int or out[key] < (0 if key.endswith('seed') else 1):
            raise ValueError(f'{key} must be a registered positive integer')
    for key in ('learning_rate', 'max_grad_norm'):
        if not isinstance(out[key], (int, float)) or not math.isfinite(out[key]) or out[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('positive_mass', 'heldout_fraction'):
        if not isinstance(out[key], (int, float)) or not 0 <= out[key] < 1:
            raise ValueError(f'{key} must be in [0, 1)')
    if not isinstance(out['split_salt'], str) or not out['split_salt']:
        raise ValueError('split_salt must be a registered string')
    return out


# ---------------------------------------------------------------- rows

def read_rows(path):
    path = _guard(path)
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path.name}:{number} is not an object')
            rows.append(dict(row, _source=path.name, _line=number))
    return rows


def world_seed(row):
    world = row.get('world') or {}
    seed = world.get('seed', row.get('world_seed'))
    return seed if type(seed) is int else None


def admission(row, cells=TRAIN_CELLS):
    """None when admitted, else the exclusion reason. A HELD-OUT seed refuses the run."""
    seed = world_seed(row)
    if seed is None:
        return 'no_world_seed'
    if HELDOUT_EVAL_SEEDS[0] <= seed <= HELDOUT_EVAL_SEEDS[1]:
        raise ValueError(f'HELD-OUT evaluation seed {seed} in warm-start input ({row.get("_source")})')
    if not TRAIN_SEEDS[0] <= seed <= TRAIN_SEEDS[1]:
        return 'not_train_seed'
    world = row.get('world') or {}
    if world.get('map', MAP) != MAP:
        return 'other_map'
    if cells is not None:
        cell = (world.get('demand'), world.get('penetration'))
        if not any(cell[0] == d and isinstance(cell[1], (int, float)) and abs(cell[1]-p) < 1e-9
                   for d, p in cells):
            return 'not_training_cell'
    return None


def do_not_act(row):
    """(label, reason): evaluated failures only (RULE_NEGATIVE)."""
    schema = row.get('schema')
    if schema == 'coopmwm-wall-negative-v1' or row.get('label') in ('do_not_wall', 'do_not_act'):
        return True, 'labelled_'+str(row.get('basis') or row.get('label'))
    if row.get('admitted') is not False:
        return False, 'not_rejected'
    if schema in ('newmap2-negative-v1', 'lowcav-negative-v1'):
        stage, reason = row.get('stage'), str(row.get('reason') or '')
        if stage == 'screen' and reason.startswith('group not better than base'):
            return True, 'screen_group_vs_base_failed'
        if stage == 'solos':
            return True, 'synergy_failed'
        if stage in ('r1', 'r2'):
            return True, 'replicate_failed'
        return False, 'outranked_or_unevaluated:'+str(stage)
    if schema == 'coopkinds-negative-v1':
        status = row.get('status')
        mapped = dict(screen_fail_LB95T='screen_group_vs_base_failed', r0_no_synergy='synergy_failed',
                      r1_fail='replicate_failed', r2_fail='replicate_failed').get(status)
        return (True, mapped) if mapped else (False, 'outranked_or_unevaluated:'+str(status))
    return False, 'unknown_schema:'+str(schema)


def members_of(row):
    members = row.get('members')
    if not isinstance(members, list) or not members:
        raise Unrepresentable('row has no members')
    return [str(v) for v in members]


def move_detail(row):
    """The row's recorded move description (N8 rows store it as a JSON string)."""
    move = row.get('move')
    if isinstance(move, str):
        try:
            move = json.loads(move)
        except ValueError:
            return {}
    return move if isinstance(move, dict) else {}


def family_of(row):
    family = str(row.get('family') or move_detail(row).get('family') or '')
    return FAMILY_ALIASES.get(family, family)


def params_of(row):
    params = dict(row.get('params') or {})
    for key, value in (move_detail(row).get('params') or {}).items():
        params.setdefault(key, value)
    for member in move_detail(row).get('members') or ():
        if isinstance(member, dict):
            for key, value in (member.get('params') or {}).items():
                params.setdefault(key, value)
    return params


def split_assignment(seeds, *, salt, fraction):
    """{seed: 'train'|'heldout'}: hash rank, common to B, R and negatives; >=1 per side."""
    seeds = sorted(set(seeds))
    if not seeds:
        return {}
    ranked = sorted(seeds, key=lambda s: hashlib.sha256(f'{salt}:{s}'.encode()).hexdigest())
    count = int(round(fraction*len(ranked)))
    if fraction > 0 and len(ranked) >= 2:
        count = min(max(count, 1), len(ranked)-1)
    held = set(ranked[:count])
    return {s: ('heldout' if s in held else 'train') for s in seeds}


# ---------------------------------------------------------------- observable frames

def whitelist(rows):
    out = []
    for r in rows:
        if any(k not in r for k in FRAME_KEYS):
            raise ValueError('frame row lacks the sensor whitelist keys')
        out.append({k: r[k] for k in FRAME_KEYS})
    return sorted(out, key=lambda r: r['vid'])


def frame_entries(rows):
    """{(world_seed, t0): {t0-1: rows, t0: rows}} from W2-format frame rows (Plan A)."""
    index = {}
    for row in rows:
        seed, t0 = world_seed(row), row.get('t0')
        frames = row.get('frames')
        if seed is None or type(t0) is not int or not isinstance(frames, dict):
            raise ValueError('negative-frame row needs world seed, integer t0 and frames')
        if set(frames) != {str(t0-1), str(t0)}:
            raise ValueError('negative-frame row needs exactly the t0-1 and t0 frames')
        if row.get('frames_sha256') is not None and row['frames_sha256'] != digest(frames):
            raise ValueError('negative-frame frames_sha256 mismatch')
        value = {t: whitelist(frames[str(t)]) for t in (t0-1, t0)}
        key = (seed, t0)
        if key in index and index[key] != value:
            raise ValueError(f'conflicting frames for world {seed} t0 {t0}')
        index[key] = value
    return index


def dry_frames(row):
    """SYNTHETIC whitelisted frames for dry mode from the row's own (observable) metadata.

    Members sit in their recorded lanes (raw lane l -> frame lane 2-l, the N1 projection:
    closing lane -> WM lane 2, lane-drop end at s=75.44) at their recorded x_t0 when present,
    else evenly over the recorded member spread max_dist_t0_m, at the recorded approach
    speed. Deterministic; never used for a deployable checkpoint.
    """
    t0 = int(row['t0'])
    members = members_of(row)
    detail = {m.get('id'): m for m in move_detail(row).get('members') or () if isinstance(m, dict)}
    lanes = row.get('member_lanes') or [detail.get(v, {}).get('lane', 1) for v in members]
    reach = (row.get('v2v_range') or row.get('v2v') or {}).get('max_dist_t0_m')
    spread = float(reach) if isinstance(reach, (int, float)) and math.isfinite(reach) else 30.
    speed = (row.get('situation') or {}).get('approach_speed_t0')
    speed = float(speed) if isinstance(speed, (int, float)) and speed > 0 else 20.
    vehicles = []
    for i, vid in enumerate(members):
        info = detail.get(vid, {})
        if isinstance(info.get('x_t0'), (int, float)):
            s = float(info['x_t0'])-2700.+75.44
        else:
            s = -500.-(spread*i/max(1, len(members)-1))
        v = float(info['v_t0']) if isinstance(info.get('v_t0'), (int, float)) else speed
        lane = int(lanes[i]) if i < len(lanes) and isinstance(lanes[i], int) else 1
        vehicles.append(dict(vid=vid, s_m=s, v_mps=v, lane=2-min(2, max(0, lane)), is_cav=True, length_m=5.))
    anchor = min(r['s_m'] for r in vehicles)
    vehicles += [dict(vid='dry_hdv_a', s_m=anchor+40., v_mps=speed, lane=1, is_cav=False, length_m=5.),
                 dict(vid='dry_hdv_b', s_m=anchor-35., v_mps=speed, lane=0, is_cav=False, length_m=5.)]
    previous = [dict(r, s_m=r['s_m']-r['v_mps']) for r in vehicles]
    return {t0-1: whitelist(previous), t0: whitelist(vehicles)}


# ---------------------------------------------------------------- policy pieces

def build_policy(base, cfg, arm):
    """The clean4 EpisodePolicy for one arm; B and R start from the SAME initialisation."""
    from .track2_backend import EpisodePolicy
    if arm not in ('B', 'R'):
        raise ValueError('A has no cooperation layer')
    torch.manual_seed(int(cfg['training_seed']))
    return EpisodePolicy(base, cfg, arm)


def logit_parameters(policy):
    """Named trigger logit parameters: never base, trigger value, residual or layer gain."""
    out = []
    for name, p in policy.trigger.named_parameters():
        if name.startswith('base_policy.') or name.startswith('trigger_value.') or not p.requires_grad:
            continue
        out.append((name, p))
    if not out:
        raise ValueError('no trainable trigger logit parameters')
    return out


def observer(policy):
    """A causal observer copy with NO episode history (as W2 wall_negatives.observer)."""
    reader = deepcopy(policy)
    reader.executor.active = []
    reader.reports, reader.trigger_board = {}, {}
    if hasattr(reader, 'wall_ends'):
        reader.wall_ends = {}
    return reader


def _attach_extensions(context, reader, members):
    try:
        from . import v2v_wall
    except ImportError:
        return context
    attach = getattr(v2v_wall, 'attach_context', None)
    return context if attach is None else attach(context, reader, members)


def _wrap(move):
    from . import moves
    from .track2_backend import TriggerMove
    return TriggerMove(move) if isinstance(move, moves.Move) else move


def _slate(policy, context, states, visible, move, target, seed):
    """The deterministic online slate with the labelled move in slot 1 (clean3/W2 rule)."""
    from .trigger_policy import TriggerBatch
    batch = policy.trigger.decide(context, obs_states=states, observable_cavs=visible,
                                  rng=np.random.default_rng(seed),
                                  generator=torch.Generator().manual_seed(seed), deterministic=True)
    tables, descriptors, mask = list(batch.tables), batch.descriptors.clone(), batch.mask.clone()
    targets = []
    for i, vid in enumerate(context['members']):
        if move is not None:
            wrapped = _wrap(move)
            table = list(tables[i])
            table[1] = wrapped
            tables[i] = tuple(table)
            descriptors[i, 1] = torch.tensor(policy.trigger._descriptor(wrapped, visible[vid]))
            mask[i, 1] = True
        targets.append(int(target))
    return TriggerBatch(context, tuple(tables), descriptors, mask, torch.tensor(targets, dtype=torch.long),
                        batch.logp, batch.values)


def _ego_context(reader, ego, t0):
    from tools.coopnative.policy import encode
    state = reader.local_states[ego]
    encoded = encode(state['rows'])
    index = encoded['members'].index(ego)
    context = dict(encoded, members=[ego], x=[encoded['x'][index]], anchors=[encoded['anchors'][index]],
                   rsu=[0.]*len(encoded['rsu']), decision_s=int(t0), observed_at_s=int(t0))
    if getattr(reader.layer.base, 'intent_horizon', 0):
        context['trigger_v2v'] = {ego: reader.local_intents[ego]}
    pressure = getattr(reader, 'pressure_blocks', None)
    if pressure and ego in pressure:
        context['merge_pressure'] = {ego: pressure[ego]}
    return _attach_extensions(context, reader, [ego])


def _clip(value, lo, hi):
    return min(hi, max(lo, value))


def runtime_move(row, reader, ego, *, dry):
    """(move, clipped_fields) in the runtime action space, or raise Unrepresentable."""
    from . import moves
    members = members_of(row)
    params = params_of(row)
    hold = params.get('hold_s', row.get('duration_s'))
    hold = int(hold) if isinstance(hold, (int, float)) and math.isfinite(hold) else int(moves._HOLD_HI)
    depth = params.get('dmax', params.get('depth', moves._DEPTH_HI))
    depth = float(depth) if isinstance(depth, (int, float)) and math.isfinite(depth) else moves._DEPTH_HI
    clipped = []
    h = int(_clip(hold, moves._HOLD_LO, moves._HOLD_HI))
    d = float(_clip(depth, moves._DEPTH_LO, moves._DEPTH_HI))
    if h != hold:
        clipped.append('hold')
    if abs(d-depth) > 1e-12:
        clipped.append('depth')
    if dry:
        if not 2 <= len(members) <= moves.MOVE_MAX_MEMBERS:
            raise Unrepresentable('dry stand-in wall needs 2..4 members')
        return moves.Move(DRY_FAMILY, tuple(members), 0, h, d), clipped
    family = family_of(row)
    detail = move_detail(row)
    if family in LANE_DROP_JOINT:
        try:
            from . import lane_drop
        except ImportError as error:
            raise Unrepresentable('lane_drop (N2) not merged') from error
        ld = getattr(reader, 'lane_drop', None)
        if not ld or not ld.get('moves'):
            raise Unrepresentable('lane-drop moves are off in this configuration')
        view = ld['view']
        if 'target_mps' in detail:
            # Plan A / W2-format rows carry the exact runtime move (LaneDropMove.to_dict).
            try:
                move = lane_drop.LaneDropMove.from_dict(detail)
                move.check_roles(view)
            except (KeyError, TypeError, ValueError) as error:
                raise Unrepresentable(f'recorded lane-drop move: {error}') from error
            return move, []
        rows = {r['vid']: r for r in reader.local_states[ego]['rows']}
        if any(v not in rows for v in members):
            raise Unrepresentable('member not in the ego observation')
        group = [rows[v] for v in members]
        if family == 'abreast_wall':
            group = sorted(group, key=lambda r: int(r['lane']))
        target = params.get('v_w')
        if not isinstance(target, (int, float)) or not math.isfinite(target):
            raise Unrepresentable('lane-drop row without a target speed')
        t = float(_clip(float(target), *lane_drop.TARGET_WINDOW_MPS))
        if t != float(target):
            clipped.append('target_mps')
        schedule = lane_drop.clip_schedule(0, h, lane_drop._arrival(view, group))
        if schedule is None:
            raise Unrepresentable('release point too close for the registered minimum hold')
        if schedule[1] != h:
            clipped.append('hold_arrival')
        try:
            move = lane_drop.LaneDropMove(family, tuple(r['vid'] for r in group), schedule[0], schedule[1], d,
                                          t, 0., False, tuple(int(r['lane']) for r in group))
            move.check_roles(view)
        except ValueError as error:
            raise Unrepresentable(f'lane-drop roles/geometry: {error}') from error
        return move, clipped
    if {'family', 'members', 'onset', 'hold', 'depth'} <= set(detail) and detail['family'] in moves.ALL_FAMILIES:
        # W2-format rows (coopmwm-wall-negative-v1) carry an exact shared-vocabulary move.
        try:
            return moves.Move.from_dict(detail), []
        except (KeyError, TypeError, ValueError) as error:
            raise Unrepresentable(f'recorded move: {error}') from error
    raise Unrepresentable(f'no runtime adapter for family {family!r}')


def frame_records(policy, row, frames, *, target, kind, seed, dry, slate_seed):
    """One single-ego record per member whose own slots contain every member (W2 rule)."""
    t0 = int(row['t0'])
    members = members_of(row)
    intent = getattr(policy.layer.base, 'intent_horizon', 0)
    if not intent:
        raise ValueError('frame-built warm-start contexts require the intent trigger observer (v2v_intent)')
    reader = observer(policy)
    for t in (t0-1, t0):
        reader.observe(whitelist(frames[t]), t)
    out, reasons = [], Counter()
    for ego in sorted(members):
        visible = reader.visible.get(ego)
        if visible is None or not set(members) <= set(visible):
            reasons['not_observable_from_ego'] += 1
            continue
        if intent and ego not in reader.local_intents:
            reasons['no_local_intent'] += 1
            continue
        try:
            move, clipped = runtime_move(row, reader, ego, dry=dry)
        except Unrepresentable as error:
            reasons['unrepresentable:'+str(error)[:60]] += 1
            continue
        context = _ego_context(reader, ego, t0)
        states = {ego: dict(reader.local_states[ego], acquirable_cav_ids=list(visible))}
        batch = _slate(policy, context, states, {ego: visible}, move, target, slate_seed)
        out.append(dict(batch=batch, kind=kind, family=family_of(row) or DRY_FAMILY, seed=seed, situation_id=row.get('situation_id'), source=row.get('_source'),
                        line=row.get('_line'), members=[ego], clipped=clipped, time_s=t0))
    return out, reasons


# ---------------------------------------------------------------- bank mode positives

def bank_records(cfg, arm, policy, mined_rows, *, slate_seed, frame_index=None):
    """Positives of ONE arm through the clean3/clean4 bank reader (observable contexts only)."""
    from . import bank_breadth, bank_v2v, group_lane
    if not group_lane.enabled(cfg):
        raise ValueError('bank mode reads mined rows only with group_lane_moves=True (clean4 config)')
    by_vid = {r['verification_id']: r for r in mined_rows if r.get('arm') == arm}
    records, reasons = [], Counter()

    def consume(*, cfg, receipt, example, plan, pairs, trace):
        row = by_vid.get(receipt['verification_id'])
        if row is None:
            # Only this arm's admitted registered rows; never an inherited (e.g. merge-map) bank.
            reasons['not_an_admitted_own_arm_row'] += 1
            return
        seed = world_seed(row)
        if seed is None:
            reasons['no_world_seed'] += 1
            return
        frames = bank_v2v.trace_frames(trace)
        t0 = int(receipt['t0'])
        if frame_index is not None and t0-1 in frames and t0 in frames:
            frame_index.setdefault((seed, t0), {t: whitelist(frames[t]['vehicles']) for t in (t0-1, t0)})
        for context, label, reader, weight, time in bank_breadth.observable_contexts(
                policy, plan, pairs, frames, t0=t0, certification=cfg.get('_executable_lesson')):
            pressure = getattr(reader, 'pressure_blocks', None)
            if pressure:
                context = dict(context, merge_pressure={v: pressure[v] for v in context['members']
                                                        if v in pressure})
            states = {v: (dict(reader.local_states[v], acquirable_cav_ids=list(reader.visible[v]))
                          if 'acquirable_cav_ids' not in reader.local_states[v] else reader.local_states[v])
                      for v in context['members']}
            none = getattr(label, 'family', None) == 'none'
            batch = _slate(policy, context, states, reader.visible, None if none else label, 0 if none else 1,
                           slate_seed)
            records.append(dict(batch=batch, kind='positive', family=FAMILY_ALIASES.get(label.family, label.family),
                                seed=seed, situation_id=row.get('situation_id'),
                                source=row.get('_source'), line=row.get('_line'),
                                verification_id=receipt['verification_id'], members=list(context['members']),
                                clipped=[], time_s=int(time)))

    bank_breadth.visit(cfg, arm, consume)
    return records, reasons


# ---------------------------------------------------------------- training

def _record_ce(policy, record):
    b = record['batch']
    logits = policy.trigger.distribution(b.context, b.descriptors, b.mask)[0]
    return torch.nn.functional.cross_entropy(logits, b.actions)


def _content_key(record):
    b = record['batch']
    return digest(dict(context=json.loads(json.dumps(b.context, default=str)),
                       descriptors=b.descriptors.tolist(), mask=b.mask.tolist(),
                       actions=b.actions.tolist(), kind=record['kind']))


def deduplicate(records):
    """One record per identical (context, slate, target); CRN seed replicates collapse."""
    seen, out = {}, []
    for r in records:
        key = _content_key(r)
        if key in seen:
            seen[key]['replicates'] += 1
            continue
        r = dict(r, replicates=1)
        seen[key] = r
        out.append(r)
    return out


def mass_weights(records, positive_mass):
    """Registered mass rule: positives `positive_mass`, negatives the rest (all when one side
    is empty); uniform over families within a side, uniform over records within a family."""
    sides = {k: [r for r in records if r['kind'] == k] for k in ('positive', 'negative')}
    present = [k for k, v in sides.items() if v]
    share = {}
    if len(present) == 2:
        share = dict(positive=float(positive_mass), negative=1.-float(positive_mass))
    elif present:
        share = {present[0]: 1.}
    weights = []
    for r in records:
        side = sides[r['kind']]
        families = Counter(x['family'] for x in side)
        weights.append(share[r['kind']]/len(families)/families[r['family']])
    return torch.tensor(weights, dtype=torch.float64)


@torch.no_grad()
def evaluate(policy, records):
    """Top-1 imitation accuracy / CE / act probability, by kind and family."""
    rows = []
    for r in records:
        b = r['batch']
        logits = policy.trigger.distribution(b.context, b.descriptors, b.mask)[0]
        prob = logits.softmax(-1)
        for i in range(len(b.actions)):
            target = int(b.actions[i])
            rows.append(dict(kind=r['kind'], family=r['family'], correct=int(logits[i].argmax()) == target,
                             ce=float(-logits[i].log_softmax(-1)[target]), act=float(1-prob[i, 0])))
    def summary(subset):
        if not subset:
            return dict(n=0, accuracy=None, mean_ce=None, mean_act_probability=None)
        return dict(n=len(subset), accuracy=sum(x['correct'] for x in subset)/len(subset),
                    mean_ce=sum(x['ce'] for x in subset)/len(subset),
                    mean_act_probability=sum(x['act'] for x in subset)/len(subset))
    out = dict(all=summary(rows))
    for kind in ('positive', 'negative'):
        part = [x for x in rows if x['kind'] == kind]
        out[kind] = summary(part)
        out[kind+'_by_family'] = {f: summary([x for x in part if x['family'] == f])
                                  for f in sorted({x['family'] for x in part})}
    return out


def train(policy, records, settings):
    """Fixed-effort weighted-sampling CE on the trigger logits only; returns the effort/log."""
    if not records:
        raise ValueError('no warm-start training records')
    named = logit_parameters(policy)
    parameters = [p for _, p in named]
    optimizer = torch.optim.Adam(parameters, lr=settings['learning_rate'])
    weights = mass_weights(records, settings['positive_mass'])
    generator = torch.Generator().manual_seed(int(settings['sample_seed']))
    base_before = state_sha256(policy.layer.base)
    frozen_before = dict(residual=state_sha256(policy.layer.residual),
                         trigger_value=state_sha256(policy.trigger.trigger_value))
    log = []
    for step in range(int(settings['steps'])):
        index = torch.multinomial(weights, int(settings['batch_size']), replacement=True, generator=generator)
        loss = torch.stack([_record_ce(policy, records[int(i)]) for i in index]).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('nonfinite warm-start loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, settings['max_grad_norm'], error_if_nonfinite=True)
        optimizer.step()
        if step % 50 == 0 or step == settings['steps']-1:
            log.append(dict(step=step, loss=float(loss.detach()), grad_norm=float(norm)))
    optimizer.zero_grad(set_to_none=True)
    for p in policy.parameters():
        p.grad = None
    policy.layer._check_frozen()
    if state_sha256(policy.layer.base) != base_before:
        raise ValueError('frozen MAPPO base changed during warm start')
    if (state_sha256(policy.layer.residual) != frozen_before['residual']
            or state_sha256(policy.trigger.trigger_value) != frozen_before['trigger_value']):
        raise ValueError('residual/trigger value changed during warm start')
    effort = dict(optimizer='torch.optim.Adam', loss='cross_entropy(trigger logits, target slot)',
                  sampling='torch.multinomial(mass_weights, batch_size, replacement, seeded)',
                  optimizer_steps=int(settings['steps']), **{k: settings[k] for k in sorted(settings)},
                  trained_parameters=[n for n, _ in named],
                  trained_parameter_count=sum(p.numel() for p in parameters))
    return dict(effort=effort, log=log, base_sha256=base_before)


# ---------------------------------------------------------------- run

def _negatives(policy, rows, frame_index, *, dry, slate_seed, split):
    records, reasons, labels = [], Counter(), Counter()
    for row in rows:
        try:
            label, why = do_not_act(row)
        except Unrepresentable:
            label, why = False, 'unrepresentable'
        labels[why] += 1
        if not label:
            continue
        seed = world_seed(row)
        if dry:
            frames = dry_frames(row)
        elif isinstance(row.get('frames'), dict):
            frames = frame_entries([row])[(seed, int(row['t0']))]   # the row's own recorded frames
        else:
            frames = frame_index.get((seed, int(row['t0'])))
        if frames is None:
            reasons['no_observable_frame'] += 1
            continue
        try:
            out, why = frame_records(policy, row, frames, target=0, kind='negative', seed=seed, dry=dry,
                                     slate_seed=slate_seed)
        except Unrepresentable as error:
            reasons['unrepresentable:'+str(error)[:60]] += 1
            continue
        reasons.update(why)
        if not out:
            reasons['row_without_usable_ego'] += 1
        records += out
    return records, reasons, labels


def run(*, cfg, base, out_dir, positives, negatives, mode, training=None, cells=TRAIN_CELLS,
        negative_frames=(), allow_existing=False):
    """Build, train and save B/R warm-start checkpoints plus WARMSTART_RECEIPT.json.

    positives: {'B': [paths], 'R': [paths]} (each arm reads ONLY its own rows)
    negatives: [paths] (one shared set). negative_frames: [paths] (bank mode frame index).
    """
    if mode not in ('bank', 'dry'):
        raise ValueError("mode must be 'bank' or 'dry'")
    dry = mode == 'dry'
    settings = training_settings(training)
    out_dir = _guard(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {arm: out_dir/f'WARMSTART_{arm}.pt' for arm in ('B', 'R')}
    if not allow_existing and any(p.exists() for p in targets.values()):
        raise ValueError('warm-start checkpoints already exist; use a new output directory')
    inputs, exclusions = {}, {}
    pos_rows = {}
    for arm in ('B', 'R'):
        rows = []
        for path in positives.get(arm, ()):
            path = _guard(path)
            inputs[str(path)] = dict(sha256=_sha_file(path), role=f'positives_{arm}')
            for row in read_rows(path):
                if row.get('schema') != POSITIVE_SCHEMA or row.get('arm') != arm:
                    raise ValueError(f'{path.name}:{row["_line"]} is not an own-arm {arm} mined row')
                why = admission(row, cells)
                if why is not None:
                    exclusions.setdefault(f'positives_{arm}', Counter())[why] += 1
                    continue
                rows.append(row)
        pos_rows[arm] = rows
    neg_rows = []
    for path in negatives:
        path = _guard(path)
        inputs[str(path)] = dict(sha256=_sha_file(path), role='negatives')
        for row in read_rows(path):
            why = admission(row, cells)
            if why is not None:
                exclusions.setdefault('negatives', Counter())[why] += 1
                continue
            neg_rows.append(row)
    frame_rows = []
    for path in negative_frames:
        path = _guard(path)
        inputs[str(path)] = dict(sha256=_sha_file(path), role='negative_frames')
        for row in read_rows(path):
            why = admission(row, cells)
            if why is not None:
                exclusions.setdefault('negative_frames', Counter())[why] += 1
                continue
            frame_rows.append(row)
            if move_detail(row):
                # Plan A (a): a re-verified lost group in W2 format is itself a negative example.
                neg_rows.append(row)
    frame_index = frame_entries(frame_rows)
    rows_sha256 = digest(sorted((k, v['sha256'], v['role']) for k, v in inputs.items()))
    seeds = {world_seed(r) for r in pos_rows['B']+pos_rows['R']+neg_rows}
    seeds |= {s for s, _ in frame_index}
    split = split_assignment(seeds, salt=settings['split_salt'], fraction=settings['heldout_fraction'])
    config_sha256 = digest({k: v for k, v in cfg.items() if k != CONFIG_KEY})
    policies, records, notes = {}, {}, {}
    # Positives first (bank mode also indexes their decision frames for the negatives).
    for arm in ('B', 'R'):
        policies[arm] = build_policy(base, cfg, arm)
    initial = {arm: state_sha256(policies[arm]) for arm in policies}
    if initial['B'] != initial['R']:
        raise ValueError('B and R must start from the identical initial cooperation layer')
    for arm in ('B', 'R'):
        if dry:
            recs, why = [], Counter()
            for row in pos_rows[arm]:
                try:
                    out, reasons = frame_records(policies[arm], row, dry_frames(row), target=1, kind='positive',
                                                 seed=world_seed(row), dry=True, slate_seed=settings['slate_seed'])
                except Unrepresentable as error:
                    why['unrepresentable:'+str(error)[:60]] += 1
                    continue
                why.update(reasons)
                recs += out
        else:
            local = dict(cfg, mined_moves=str(_mined_file(out_dir, arm, pos_rows[arm])))
            recs, why = bank_records(local, arm, policies[arm], pos_rows[arm], slate_seed=settings['slate_seed'],
                                     frame_index=frame_index)
        records[arm] = recs
        notes[arm] = dict(positive_exclusions=dict(why))
    # ONE shared negatives set, built once on the (identical) initial B layer.
    shared, neg_reasons, neg_labels = _negatives(policies['B'], neg_rows, frame_index, dry=dry,
                                                 slate_seed=settings['slate_seed'], split=split)
    shared_keys = sorted(_content_key(r) for r in deduplicate(shared))
    report = dict(schema=SCHEMA, rule=RULE, negative_rule=RULE_NEGATIVE, mode=mode, dry=dry,
                  map=MAP, train_seeds=list(TRAIN_SEEDS), refused_seeds=list(HELDOUT_EVAL_SEEDS),
                  training_cells=[list(c) for c in cells] if cells is not None else None,
                  inputs=inputs, rows_sha256=rows_sha256, config_sha256=config_sha256,
                  exclusions={k: dict(v) for k, v in exclusions.items()},
                  split=dict(rule='sha256(salt:seed) rank; common to B, R and negatives',
                             salt=settings['split_salt'], fraction=settings['heldout_fraction'],
                             heldout=sorted(s for s, v in split.items() if v == 'heldout'),
                             train=sorted(s for s, v in split.items() if v == 'train')),
                  negatives=dict(rows=len(neg_rows), labels=dict(neg_labels), reasons=dict(neg_reasons),
                                 records=len(shared_keys), records_sha256=digest(shared_keys)),
                  arms={})
    for arm in ('B', 'R'):
        policy = policies[arm]
        everything = deduplicate(records[arm]+shared)
        part = {k: [r for r in everything if split.get(r['seed']) == k] for k in ('train', 'heldout')}
        before = dict(train=evaluate(policy, part['train']), heldout=evaluate(policy, part['heldout']))
        trained = train(policy, part['train'], settings) if part['train'] else None
        if trained is None:
            raise ValueError(f'{arm}: no TRAIN-split warm-start records')
        after = dict(train=evaluate(policy, part['train']), heldout=evaluate(policy, part['heldout']))
        worker = policy.worker_state()
        payload = dict(schema=CHECKPOINT_SCHEMA, arm=arm, dry=dry, mode=mode, worker=worker,
                       effort=trained['effort'], rows_sha256=rows_sha256, config_sha256=config_sha256,
                       base_sha256=trained['base_sha256'], initial_layer_sha256=initial[arm],
                       negatives_sha256=report['negatives']['records_sha256'], rule=RULE)
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        with open(targets[arm], 'wb' if allow_existing else 'xb') as stream:
            stream.write(buffer.getvalue())
        families = Counter((r['kind'], r['family']) for r in everything)
        report['arms'][arm] = dict(
            checkpoint=str(targets[arm]), checkpoint_sha256=_sha_file(targets[arm]),
            positive_rows=len(pos_rows[arm]), positive_records=sum(r['kind'] == 'positive' for r in everything),
            negative_records=sum(r['kind'] == 'negative' for r in everything),
            records_by_kind_family={f'{k}:{f}': n for (k, f), n in sorted(families.items())},
            split_records={k: len(v) for k, v in part.items()},
            clipped=dict(Counter(c for r in everything for c in r.get('clipped', ()))),
            effort=trained['effort'], loss_log=trained['log'], base_sha256=trained['base_sha256'],
            base_unchanged=state_sha256(policy.layer.base) == trained['base_sha256'],
            before=before, after=after, **notes[arm])
    report['effort_equal'] = report['arms']['B']['effort'] == report['arms']['R']['effort']
    report['base_equal'] = report['arms']['B']['base_sha256'] == report['arms']['R']['base_sha256']
    if not report['effort_equal'] or not report['base_equal']:
        raise ValueError('B/R warm-start effort or frozen base differ')
    report['coop_warmstart'] = registration(report)
    text = json.dumps(report, indent=1, sort_keys=True, default=str)
    (out_dir/'WARMSTART_RECEIPT.json').write_text(text+'\n', encoding='utf-8')
    return report


def _mined_file(out_dir, arm, rows):
    """The admitted own-arm rows only (clean rows without the reader bookkeeping fields)."""
    path = out_dir/f'WARMSTART_MINED_{arm}.jsonl'
    clean = [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]
    path.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in clean), encoding='utf-8')
    return path


def registration(report):
    """The cfg['coop_warmstart'] block for the clean4 prepare (B and R together)."""
    arms = report['arms']
    return dict(B=dict(path=arms['B']['checkpoint'], sha256=arms['B']['checkpoint_sha256']),
                R=dict(path=arms['R']['checkpoint'], sha256=arms['R']['checkpoint_sha256']),
                steps=arms['B']['effort']['optimizer_steps'], rows_sha256=report['rows_sha256'],
                rule=RULE, allow_dry=False)


# ---------------------------------------------------------------- clean4 prepare (default OFF)

def settings(cfg):
    """None when OFF (absent/None/False); else the validated registration block."""
    value = cfg.get(CONFIG_KEY)
    if value is None or value is False:
        return None
    if not isinstance(value, dict):
        raise ValueError('coop_warmstart must be a registration mapping')
    required = {'B', 'R', 'steps', 'rows_sha256', 'rule'}
    if not required <= set(value) or set(value)-required-{'allow_dry'}:
        raise ValueError(f'coop_warmstart requires exactly {sorted(required)} (+ allow_dry)')
    for arm in ('B', 'R'):
        spec = value[arm]
        if (not isinstance(spec, dict) or set(spec) != {'path', 'sha256'}
                or not isinstance(spec['sha256'], str) or len(spec['sha256']) != 64):
            raise ValueError(f'coop_warmstart.{arm} needs path and sha256')
    if type(value['steps']) is not int or value['steps'] < 1:
        raise ValueError('coop_warmstart.steps must be a positive integer')
    if value.get('allow_dry', False) not in (True, False):
        raise ValueError('coop_warmstart.allow_dry must be boolean')
    if value['rule'] != RULE:
        raise ValueError('coop_warmstart rule mismatch')
    return value


def load(cfg, arm):
    """The verified checkpoint payload of one arm (sha, arm, effort parity, dry guard)."""
    value = settings(cfg)
    if value is None:
        return None
    if arm not in ('B', 'R'):
        raise ValueError('A never warm-starts (plain MAPPO)')
    payloads = {}
    for name in ('B', 'R'):
        spec = value[name]
        path = _guard(spec['path'])
        if _sha_file(path) != spec['sha256']:
            raise ValueError(f'coop_warmstart.{name} checkpoint sha256 mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('schema') != CHECKPOINT_SCHEMA or payload.get('arm') != name:
            raise ValueError(f'coop_warmstart.{name} is not an own-arm {name} warm-start checkpoint')
        if payload.get('dry') and not value.get('allow_dry', False):
            raise ValueError('dry warm-start checkpoints are plumbing only (allow_dry not registered)')
        if payload.get('rows_sha256') != value['rows_sha256']:
            raise ValueError(f'coop_warmstart.{name} rows_sha256 mismatch')
        if payload['effort'].get('optimizer_steps') != value['steps']:
            raise ValueError(f'coop_warmstart.{name} step count differs from the registration')
        payloads[name] = payload
    if payloads['B']['effort'] != payloads['R']['effort']:
        raise ValueError('coop_warmstart B and R training effort differ')
    if payloads['B']['base_sha256'] != payloads['R']['base_sha256']:
        raise ValueError('coop_warmstart B and R were trained on different frozen bases')
    return payloads[arm]


def apply(cfg, arm, policy):
    """Restore the warm-start worker state into a freshly built EpisodePolicy (OFF -> None)."""
    payload = load(cfg, arm)
    if payload is None:
        return None
    if state_sha256(policy.layer.base) != payload['base_sha256']:
        raise ValueError('coop_warmstart frozen base differs from this run\'s MAPPO u0')
    try:
        policy.restore_worker(payload['worker'])
    except RuntimeError as error:
        raise ValueError(f'coop_warmstart layer/trigger shapes differ from this configuration: {error}') from error
    policy.layer._check_frozen()
    if state_sha256(policy.layer.base) != payload['base_sha256']:
        raise ValueError('coop_warmstart changed the frozen base')
    return dict(arm=arm, checkpoint_sha256=settings(cfg)[arm]['sha256'], rows_sha256=payload['rows_sha256'],
                steps=payload['effort']['optimizer_steps'], base_sha256=payload['base_sha256'], rule=RULE)


def sidecar_state(cfg, arm, runtime):
    """The clean4 prepare's {arm}_track2_000.pt payload; byte-identical to clean3 when OFF.

    Optimizers stay the runtime's fresh ones (no step was taken on them).
    """
    receipt = apply(cfg, arm, runtime.policy)
    state = dict(arm=arm, update=0, worker=runtime.policy.worker_state(),
                 layer_optimizer=runtime.policy.layer.optimizer.state_dict(),
                 trigger_optimizer=runtime.policy.trigger_optimizer.state_dict())
    if receipt is not None:
        if any(runtime.policy.layer.optimizer.state.values()) or any(
                runtime.policy.trigger_optimizer.state.values()):
            raise ValueError('coop_warmstart requires fresh optimizers')
        state['coop_warmstart'] = receipt
    return state


# ---------------------------------------------------------------- CLI

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--mode', choices=('bank', 'dry'), required=True)
    parser.add_argument('--config', help='clean4 B/R config JSON (track2 keys); dry default = CPU fixture')
    parser.add_argument('--base-checkpoint', help='frozen MAPPO u0 (dry default = fresh IntentPolicy)')
    parser.add_argument('--base-sha256')
    parser.add_argument('--rows-dir', help='directory with the default *_MOVES_* / *_NEGATIVES files')
    parser.add_argument('--positives-B', nargs='*', default=None)
    parser.add_argument('--positives-R', nargs='*', default=None)
    parser.add_argument('--negatives', nargs='*', default=None)
    parser.add_argument('--negative-frames', nargs='*', default=())
    parser.add_argument('--training', help='JSON object overriding registered TRAINING keys')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.threads > 4:
        raise SystemExit('at most 4 CPU threads (registered small-model budget)')
    torch.set_num_threads(args.threads)
    root = Path(args.rows_dir) if args.rows_dir else None

    def files(given, key):
        if given is not None:
            return [Path(p) for p in given]
        if root is None:
            raise SystemExit(f'--rows-dir or explicit {key} required')
        return [root/name for name in DEFAULT_SOURCES[key] if (root/name).exists()]

    if args.config:
        cfg = json.loads(_guard(args.config).read_text(encoding='utf-8'))
    elif args.mode == 'dry':
        cfg = dry_config()
    else:
        raise SystemExit('bank mode requires the clean4 --config')
    if args.base_checkpoint:
        path = _guard(args.base_checkpoint)
        if args.base_sha256 and _sha_file(path) != args.base_sha256:
            raise SystemExit('base checkpoint sha256 mismatch')
        from .v2v_runtime import load_checkpoint
        base = load_checkpoint(path, config=cfg)[0]
    elif args.mode == 'dry':
        from .v2v_intent import IntentPolicy
        torch.manual_seed(int(cfg['training_seed']))
        base = IntentPolicy()
    else:
        raise SystemExit('bank mode requires --base-checkpoint')
    report = run(cfg=cfg, base=base, out_dir=args.out, mode=args.mode,
                 positives=dict(B=files(args.positives_B, 'B'), R=files(args.positives_R, 'R')),
                 negatives=files(args.negatives, 'negatives'), negative_frames=args.negative_frames,
                 training=json.loads(args.training) if args.training else None)
    print(json.dumps(dict(mode=report['mode'], effort_equal=report['effort_equal'],
                          coop_warmstart=report['coop_warmstart'],
                          heldout={a: report['arms'][a]['after']['heldout'] for a in ('B', 'R')}),
                     indent=1, default=str))
    return report


def dry_config():
    """CPU-fixture track2 configuration for dry plumbing runs (merge-map base geometry)."""
    return dict(track2_backend=True, run_root='.', training_seed=123, v2v=True, v2v_dim=48, v2v_intent=True,
                merge_features=True, host='cpuV',
                layer_config=dict(hidden_dim=64, learning_rate=.0003, kl_budget=.045, clip_ratio=.2,
                                  max_grad_norm=.5),
                trigger_candidates=8, trigger_partner_slots=4, decision_interval_s=60, trigger_lr=.0003,
                layer_epochs=4, trigger_ppo=dict(clip_ratio=.2, value_weight=.5, entropy_weight=.01,
                                                 max_grad_norm=.5))


if __name__ == '__main__':
    main()
