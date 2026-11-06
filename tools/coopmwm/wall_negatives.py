"""Per-seed losing wall situations as explicit "do not wall" trigger labels (W2, default OFF).

COOP_PROBE_REPORT: a verified wall is a per-seed bet (lead wall positive on 3/6 CRN
seeds, hp wall 1/6, down to -505 veh-s). The positive seeds become MINED_MOVES rows; the
losing seeds were dropped. With ``wall_negative_labels`` the trigger head is also shown
those losing (wall, seed) examples, with the target ``none`` and the losing wall inserted
as the competing candidate, through the SAME bank supervision CE (separate KL budget).

Rule (RULE, per seed, never pooled): an example is a do-not-wall label iff its own seed's
measured synergy (best member-solo J minus group J, CRN, common window) is < 0; where the
native verifier ran no solos on that seed (the wall was not admitted against policy), iff
its own seed's group advantage vs policy-continues is < 0. ``basis`` records which.

B gets its own B-verified negatives, R its own R-verified negatives (``arm`` field), with
the identical rule, observer and weighting. Note: the decision-time observable state of
one situation is the SAME on every CRN seed (seeds differ only after t0), so negative and
positive labels of one situation share their trigger input; the CE then teaches the
situation's wall probability, not a per-seed oracle.

Row schema ``coopmwm-wall-negative-v1`` (one JSONL line per (wall plan, seed)):
  arm, label='do_not_wall', basis, situation_id, verification_id, plan_sha256, seed, t0,
  world{seed in TRAIN 986000..986071}, members, move (moves.Move dict, wall), and the
  per-seed value dicts keyed by the row's own seed: synergy_veh_s, synergy_lower95,
  advantage_veh_s; frames {str(t0-1), str(t0)} -> whitelisted sensor rows
  (vid, s_m, v_mps, lane, is_cav, length_m) of the policy-continues branch tape (the
  shared prefix), frames_sha256 = protocol.digest(frames); source. The trigger context is
  rebuilt from those rows by the SAME causal observer as online execution (t-1 then t).
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch

from .protocol import digest

SCHEMA = 'coopmwm-wall-negative-v1'
LABEL = 'do_not_wall'
FAMILY_KEY = 'wall_negative'
FRAME_KEYS = ('vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m')
RULE = ('W2 negative rule: a (wall plan, CRN seed) example is a do-not-wall label iff that '
        "seed's measured synergy (best solo J - group J) < 0; where no solos ran on that seed "
        "(not admitted vs policy), iff that seed's group advantage vs policy-continues < 0. "
        'Labels bind their own seed; values are never pooled; B and R use only their own rows.')
FIELDS = ('schema', 'arm', 'label', 'basis', 'situation_id', 'verification_id', 'plan_sha256',
          'seed', 't0', 'world', 'members', 'move', 'synergy_veh_s', 'synergy_lower95',
          'advantage_veh_s', 'frames', 'frames_sha256', 'source')


def _finite(x):
    return x is not None and not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


def whitelist(rows):
    """Sensor-whitelisted rows only (drops issued commands and anything else a tape carries)."""
    out = []
    for r in rows:
        if any(k not in r for k in FRAME_KEYS):
            raise ValueError('negative frame row lacks the sensor whitelist keys')
        out.append({k: r[k] for k in FRAME_KEYS})
    return sorted(out, key=lambda r: r['vid'])


def frames_from_trace(trace_path, t0):
    """{str(t0-1), str(t0)} whitelisted frames from a branch's observable.jsonl.gz."""
    from .bank_v2v import trace_frames
    from .track2_backend import _path
    frames = trace_frames(_path(trace_path))
    if int(t0)-1 not in frames or int(t0) not in frames:
        raise ValueError('policy branch tape lacks the t0-1/t0 decision frames')
    return {str(t): whitelist(frames[t].get('vehicles', [])) for t in (int(t0)-1, int(t0))}


def example(*, arm, situation_id, verification_id, plan, seed, t0, world, frames, source,
            synergy=None, synergy_lower95=None, advantage=None):
    """One negative row, or None when the per-seed rule does not label it do-not-wall."""
    from . import moves
    move = moves.from_plan(plan)
    if move.family != 'wall' or plan.get('family', 'wall') != 'wall':
        return None
    # The label must name the wall that was actually measured: the move's offsets must
    # reproduce the verified plan's policy-relative offsets exactly (e.g. a 6-CAV probe wall
    # projects onto a 4-member sub-wall and is NOT relabelled as that sub-wall).
    members, recorded = moves._plan_offsets(plan)
    exact = np.array([[moves.offsets(move, t).get(v, 0.) for t in range(recorded.shape[1])]
                      for v in members], float)
    if sorted(move.members) != sorted(members) or not np.allclose(recorded, exact, atol=1e-8, rtol=0):
        return None
    if _finite(synergy):
        basis, value = 'synergy', float(synergy)
    elif synergy is None and _finite(advantage):
        basis, value = 'advantage', float(advantage)
    else:
        return None
    if not value < 0:
        return None
    s = str(int(seed))
    row = dict(schema=SCHEMA, arm=arm, label=LABEL, basis=basis, situation_id=situation_id,
               verification_id=verification_id, plan_sha256=digest(plan), seed=int(seed), t0=int(t0),
               world=dict(world), members=list(move.members), move=move.to_dict(),
               synergy_veh_s={s: float(synergy)} if _finite(synergy) else {},
               synergy_lower95={s: float(synergy_lower95)} if _finite(synergy_lower95) else {},
               advantage_veh_s={s: float(advantage)} if _finite(advantage) else {},
               frames=frames, frames_sha256=digest(frames), source=source)
    return validate(row)


def validate(row):
    """Fail-closed schema/rule check of one negative row; returns it unchanged."""
    from . import moves
    if not isinstance(row, dict) or set(row) != set(FIELDS):
        raise ValueError('negative row must carry exactly the registered fields')
    if row['schema'] != SCHEMA or row['label'] != LABEL or row['arm'] not in ('B', 'R'):
        raise ValueError('negative row schema/label/arm')
    seed, t0 = row['seed'], row['t0']
    if type(seed) is not int or seed < 1 or type(t0) is not int or t0 < 300:   # sv.WARMUP_S
        raise ValueError('negative row seed/t0')
    world = row['world']
    if not isinstance(world, dict) or type(world.get('seed')) is not int \
            or not 986000 <= world['seed'] <= 986071:
        raise ValueError('negative rows come from registered TRAIN worlds only')
    move = moves.Move.from_dict(row['move'])
    if move.family != 'wall' or list(move.members) != row['members'] or len(move.members) < 2:
        raise ValueError('negative row must bind a multi-CAV wall move')
    s = str(seed)
    for key in ('synergy_veh_s', 'synergy_lower95', 'advantage_veh_s'):
        value = row[key]
        if not isinstance(value, dict) or set(value)-{s} or not all(_finite(v) for v in value.values()):
            raise ValueError(f'negative {key} must be keyed by the row OWN seed only')
    if row['basis'] == 'synergy':
        if s not in row['synergy_veh_s'] or not row['synergy_veh_s'][s] < 0:
            raise ValueError('synergy-basis negative requires its own-seed synergy < 0')
    elif row['basis'] == 'advantage':
        if row['synergy_veh_s'] or s not in row['advantage_veh_s'] or not row['advantage_veh_s'][s] < 0:
            raise ValueError('advantage-basis negative requires no measured synergy and own-seed advantage < 0')
    else:
        raise ValueError('negative basis must be synergy or advantage')
    frames = row['frames']
    if not isinstance(frames, dict) or set(frames) != {str(t0-1), str(t0)}:
        raise ValueError('negative row needs exactly the t0-1 and t0 frames')
    for rows in frames.values():
        if not isinstance(rows, list) or any(not isinstance(r, dict) or set(r) != set(FRAME_KEYS)
                                             for r in rows):
            raise ValueError('negative frame rows must be sensor-whitelisted')
    if row['frames_sha256'] != digest(frames):
        raise ValueError('negative frames_sha256 mismatch')
    for key in ('situation_id', 'verification_id', 'plan_sha256', 'source'):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError(f'negative row needs {key}')
    return row


def path(cfg):
    from .track2_backend import _path
    explicit = cfg.get('wall_negatives')
    value = _path(explicit if explicit else Path(cfg['run_root'])/'WALL_NEGATIVES.jsonl')
    if explicit and not value.exists():
        raise ValueError('configured wall_negatives file missing')
    return value


def load(cfg):
    value = path(cfg)
    if not value.exists():
        return []
    out = []
    for number, line in enumerate(value.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            try:
                out.append(validate(json.loads(line)))
            except ValueError as error:
                raise ValueError(f'WALL_NEGATIVES row {number}: {error}') from error
    return out


def observer(policy):
    """A causal observer copy with NO episode history (wall memory/commitments cleared)."""
    reader = deepcopy(policy)
    reader.executor.active = []
    reader.wall_ends = {}
    reader.reports, reader.trigger_board = {}, {}
    return reader


def trigger_records(cfg, arm, policy):
    """[(TriggerBatch, weight)], contexts, receipt for this arm's OWN negatives.

    One single-ego context per wall member whose own observable slots contain every wall
    member (the positives' acquirability rule), built from the t-1/t frames by the online
    observer. Target index 0 (none); the losing wall occupies candidate slot 1.
    """
    from tools.coopnative.policy import encode
    from . import moves
    from .track2_backend import TriggerMove
    from .trigger_policy import TriggerBatch
    if arm not in ('B', 'R'):
        return [], [], dict(rule=RULE, rows=0, contexts=0)
    rows = [r for r in load(cfg) if r['arm'] == arm]
    records, contexts = [], []
    receipt = dict(rule=RULE, schema=SCHEMA, arm=arm, rows=len(rows), contexts=0,
                   unobservable_rows=0, basis={'synergy': 0, 'advantage': 0}, path=str(path(cfg)))
    intent = getattr(policy.layer.base, 'intent_horizon', 0)
    if rows and not intent:
        raise ValueError('negative labels require the intent trigger observer (v2v_intent)')
    pressure = getattr(policy, 'w2', {}).get('wall_pressure_features', False)
    for row in rows:
        move = moves.Move.from_dict(row['move'])
        t0 = row['t0']
        reader = observer(policy)
        for t in (t0-1, t0):
            reader.observe(row['frames'][str(t)], t)
        used = 0
        for ego in sorted(move.members):
            visible = reader.visible.get(ego)
            if (visible is None or not set(move.members) <= set(visible)
                    or ego not in reader.local_intents):
                continue
            state = reader.local_states[ego]
            encoded = encode(state['rows'])
            index = encoded['members'].index(ego)
            context = dict(encoded, members=[ego], x=[encoded['x'][index]],
                           anchors=[encoded['anchors'][index]], rsu=[0.]*len(encoded['rsu']),
                           trigger_v2v={ego: reader.local_intents[ego]}, decision_s=t0,
                           observed_at_s=t0)
            if pressure:
                context['merge_pressure'] = {ego: reader.pressure_blocks[ego]}
            batch = policy.trigger.decide(context,
                obs_states={ego: dict(state, acquirable_cav_ids=list(visible))},
                observable_cavs={ego: visible}, rng=np.random.default_rng(cfg['training_seed']),
                generator=torch.Generator().manual_seed(cfg['training_seed']), deterministic=True)
            table = list(batch.tables[0])
            target = TriggerMove(move)
            table[1] = target
            descriptors, mask = batch.descriptors.clone(), batch.mask.clone()
            descriptors[0, 1] = torch.tensor(policy.trigger._descriptor(target, visible))
            mask[0, 1] = True
            records.append((TriggerBatch(context, (tuple(table),), descriptors, mask,
                                         torch.tensor([0]), batch.logp, batch.values), 1.))
            contexts.append(dict(verification_id=row['verification_id'],
                example_id=f"{row['verification_id']}#s{row['seed']}#neg", family=FAMILY_KEY,
                label='none', negative=True, basis=row['basis'], seed=row['seed'],
                situation_id=row['situation_id'], time_s=t0, members=[ego]))
            used += 1
        receipt['contexts'] += used
        receipt['unobservable_rows'] += int(used == 0)
        receipt['basis'][row['basis']] += 1
    return records, contexts, receipt
