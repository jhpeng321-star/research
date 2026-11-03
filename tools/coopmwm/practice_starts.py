"""practice_starts (owner 2026-11-04 proposal, DEFAULT OFF): practice PPO starts from the bank.

For arms B-v2v and R-v2v a registered fraction (practice_fraction, proposal 0.25 => 6 of the
24 episodes of one update) of each update's PPO episodes does NOT start fresh from t=300: it
restarts from the decision snapshot of an ADMITTED within-seed COOPERATIVE bank example of the
arm's OWN bank (replay_bank links, never imported u3 labels), with

- the SAME snapshot and CRN seed as the example: the sealed prefix tape is replayed through the
  registered VerifyController ('policy continues' branch semantics) and at t0 the episode's
  generator is restored from the example's state (seed-1 examples: the snapshot generator state;
  seed-2 examples: replay_bank.replicate(snapshot, verification_id) - the exact seed-2 CRN),
- the CURRENT training policy controlling every CAV from t0 (the t0 command is part of the
  sealed snapshot, exactly as in every bank branch; committed snapshot plans continue; no new
  plan is ever proposed - training episodes are plain-PPO b_replay episodes),
- the NORMAL horizon (1200 s stop) and the NORMAL team-J per-second reward, charged 1200 sim-s
  through the existing episode hooks (reserve/admission/episode), so a practice update costs
  exactly what a normal update costs (per-update sim-s delta = 0).

Examples are sampled uniformly (deterministic per arm/update/training-seed RNG) with a
per-example repeat cap per update (practice_repeat_cap); the sampled plan is sealed once
(PRACTICE_PLAN_<ARM>_<UUU>.json) and frozen - restarts resume the same job ids, never re-sample.
R falls back to its OTHER admitted examples when it has fewer cooperative examples than the
fraction requires (same practice count as B); B never falls back. An empty bank (or an
unresolvable one) yields normal starts. A-v2v is unchanged.
"""
from pathlib import Path
import json

import numpy as np
import torch

from .protocol import digest
from . import sumo_verify as sv

PRACTICE_RULE = (
    'owner 2026-11-04 proposal practice_starts: a registered fraction of each B/R update PPO '
    'episodes start from the arm\'s OWN admitted within-seed cooperative bank examples - same '
    'decision snapshot and CRN seed as the example (seed 2 = replicate(snapshot, vid)), current '
    'policy controls the CAVs from t0 (snapshot prefix replayed through the registered '
    'VerifyController, committed snapshot plans continue, no new plan), normal 1200 s horizon and '
    'team-J reward, charged 1200 sim-s like every training episode via the existing episode hooks '
    '(per-update sim-s delta 0); uniform sampling with a per-example repeat cap per update, frozen '
    'sealed plan; empty bank -> normal starts; A unchanged')
R_FALLBACK_RULE = (
    'R fallback (documented): when R has fewer cooperative examples than the fraction requires, '
    'the remaining practice starts come from R\'s other ADMITTED examples (fallback=True rows); '
    'the practice count is always B\'s count; B never falls back')
CHARGED_SIM_S = 1200          # identical to every training episode: same hooks, same phase


def enabled(cfg, arm):
    """practice_starts applies to the bank arms B and R only; A-v2v is unchanged."""
    return bool(cfg.get('practice_starts')) and arm in ('B', 'R')


def practice_count(episodes, fraction):
    """The registered fraction of one update's PPO episodes (proposal 0.25 of 24 -> 6)."""
    fraction = float(fraction)
    if not 0. < fraction <= 1.:
        raise ValueError(f'practice_fraction must be in (0, 1], not {fraction!r}')
    return min(max(int(round(int(episodes)*fraction)), 0), int(episodes))


# ---------------------------------------------------------------- eligible examples (own bank)

def _links(root, arm):
    """verification_id -> decision link of the arm's OWN bank (own snapshots only)."""
    out = {}
    for p in sorted((Path(root)/'replay_bank'/arm/'decisions').glob('*.json')):
        v = json.loads(p.read_text(encoding='utf-8'))
        out[v['verification_id']] = v
    return out


def examples(root, arm, cooperative_only=True):
    """Eligible practice rows of the arm's own bank: ADMITTED examples with their within-seed
    cooperation label (per_seed_v1 (plan, seed) examples; legacy records: the first-seed entry).
    Importable u3 label sources are NOT the arm's own bank and are never practice starts.
    A bank without evaluable data (BankDataMissing) is an empty bank, never a crash."""
    from . import replay_bank as rb
    try:
        conf = rb.confirmations(root, arm)
    except ValueError:                                   # includes bank_statistics.BankDataMissing
        return []
    links = _links(root, arm)
    rows = []
    for vid, c in conf.items():
        link = links.get(vid)
        if link is None:
            continue                                     # imported/foreign label: never practice
        per = (c.get('per_seed_v1') or {}).get('examples') if c.get('bank_rule') == 'per_seed_v1' else None
        if per is None:
            co = c.get('cooperation') or {}
            if c.get('admitted') and co.get('cooperative') is True:
                rows.append(dict(verification_id=vid, example_id=f'{vid}#s1', seed_index=1,
                                 salt=vid, state_id=link['state_id'], fallback=False))
            continue
        for ex in per:
            if ex['admitted'] and (ex['cooperative'] is True or not cooperative_only):
                rows.append(dict(verification_id=vid, example_id=ex['example_id'],
                                 seed_index=int(ex['seed_index']), salt=vid,
                                 state_id=link['state_id'],
                                 fallback=not (ex['cooperative'] is True)))
    # one row per example, decision order (confirmations are built in sealed-file order)
    seen, out = set(), []
    for row in rows:
        if row['example_id'] not in seen:
            seen.add(row['example_id'])
            out.append(row)
    return out


# ---------------------------------------------------------------- frozen per-update plan

def plan_path(root, arm, update):
    return Path(root)/'replay_bank'/arm/f'PRACTICE_PLAN_{arm}_{int(update):03}.json'


def plan_for_update(root, arm, update, *, fraction, repeat_cap, count, training_seed):
    """The frozen practice plan of one (arm, update): uniformly sampled examples with a
    per-example repeat cap; sealed once, never re-sampled (restarts reuse the same job ids)."""
    from . import exact_grade as xg
    cap = int(repeat_cap)
    if cap < 1:
        raise ValueError(f'practice_repeat_cap must be >= 1, not {cap!r}')
    count = int(count)
    path = plan_path(root, arm, update)
    if path.exists():
        record = json.loads(path.read_text(encoding='utf-8'))
        xg.verify_sealed(record, 'practice_plan_sha256')
        if record['arm'] != arm or record['update'] != int(update):
            raise ValueError('practice plan belongs to another arm/update')
        return record
    from .warm_protocol import SEED
    coop = examples(root, arm, cooperative_only=True)
    pool = [dict(r) for r in coop]
    fallback = False
    if arm == 'R' and len(pool) < count:
        # R only (documented): fill from R's other admitted examples, same count as B.
        extra = examples(root, arm, cooperative_only=False)
        have = {r['example_id'] for r in pool}
        additions = [dict(r, fallback=True) for r in extra if r['example_id'] not in have
                     and r['fallback']]
        if additions:
            pool += additions
            fallback = True
    rng = np.random.default_rng(int(digest(dict(practice='plan', arm=arm, update=int(update),
                                                training_seed=int(training_seed),
                                                fraction=float(fraction), cap=cap))[:15], 16))
    rows, used = [], {}
    while len(rows) < count:
        pickable = [r for r in pool if used.get(r['example_id'], 0) < cap]
        if not pickable:
            break                                        # cap exhausted: fewer practice episodes
        row = pickable[int(rng.integers(0, len(pickable)))]
        used[row['example_id']] = used.get(row['example_id'], 0)+1
        rows.append(dict(row))
    import time
    record = dict(arm=arm, update=int(update), training_seed=int(training_seed),
                  fraction=float(fraction), repeat_cap=cap, requested=count, rows=rows,
                  registry=dict(cooperative_examples=len(coop), pool_examples=len(pool),
                                fallback_used=fallback, empty_bank=not pool),
                  sampling=('uniform over eligible examples, per-example repeat cap %d, '
                            'RNG digest(practice=plan|arm|update|training_seed|fraction|cap)' % cap),
                  rule=PRACTICE_RULE, fallback_rule=R_FALLBACK_RULE, unix=time.time())
    sealed = xg._seal(record, 'practice_plan_sha256')
    xg._write_new(path, sealed)
    return sealed


# ---------------------------------------------------------------- snapshot + CRN (the example's twin state)

def load_snapshot(root, arm, row):
    """The example's decision snapshot with the example's CRN seed: seed-1 examples restart from
    the stored snapshot state; seed-2 examples from replicate(snapshot, vid) - the same
    independent seed-2 generator the example's own seed-2 branches ran with."""
    from . import replay_bank as rb
    snap = torch.load(rb.bank_root(root, arm)/'snapshots'/(row['state_id']+'.pt'), weights_only=False)
    if int(row['seed_index']) == 2:
        snap = rb.replicate(snap, row['salt'])
    elif int(row['seed_index']) != 1:
        raise ValueError(f'unknown practice seed_index {row["seed_index"]!r}')
    return snap


def wave_call(root, arm, ident, phase, row, index, base_extra):
    """(ident, worlds, extras) of ONE practice episode for original_wave: the snapshot's world,
    the exact prefix branch (CRN check), the snapshot's recorded runtime/checkpoint/noise and the
    practice meta. Charging stays with the caller (original_wave reserves sim_s=1200 like every
    training episode)."""
    snap = load_snapshot(root, arm, row)
    t0 = int(snap['t0'])
    sumo = snap['sumo']
    meta = dict(practice=True, verification_id=row['verification_id'], example_id=row['example_id'],
                seed_index=int(row['seed_index']), t0=t0, world=snap['episode']['world'],
                snapshot_sha256=sv.snapshot_sha(snap),
                crn_seed=('example seed-1 decision snapshot generator state' if int(row['seed_index']) == 1
                          else 'example seed-2 CRN: replicate(snapshot, vid) generator state'),
                fallback=bool(row.get('fallback')),
                horizon_end_s=int(snap.get('end_s', sv.EPISODE_END_S)), charged_sim_s=CHARGED_SIM_S,
                policy='current training policy from t0; snapshot prefix replayed; committed snapshot '
                       'plans continue; no new plan (plain-PPO b_replay episode)',
                rule=PRACTICE_RULE)
    extras = dict(base_extra,
                  practice=dict(meta, snapshot=snap),
                  branch=dict(time_s=t0, takeover_s=t0, revert=[],
                              prefixes={str(t): snap['prefixes'][str(t)] for t in range(sv.WARMUP_S, t0+1)}),
                  noise_seed=int(sumo.get('noise_seed') or 0),
                  runtime=sumo['runtime'], checkpoint=sumo['checkpoint'],
                  warm_history=bool(sumo['warm_history']), accel_bounds=sumo['accel_bounds'],
                  sim_s=CHARGED_SIM_S, deterministic=False)
    return dict(ident=f'{ident}_p{int(index)+1:02d}', worlds=[snap['episode']['world']], extras=extras)


# ---------------------------------------------------------------- training-side consumers

def strip_steps(result):
    """A practice episode enters PPO/viability/routing as an episode starting after t0: the
    replayed prefix seconds (<= t0, including the snapshot's own t0 command) carry recorded
    behaviour policy data and never become current-policy training steps."""
    t0 = int(result['practice']['t0'])
    return dict(result, steps=[s for s in result['steps'] if int(s['time_s']) > t0])


def update_report(eps):
    """Per-update practice block: practice episodes, examples used and returns vs normal
    (per-episode team-J reward sums; the per-control-second variant normalises the shorter
    practice horizons against the 900-step normal episodes)."""
    practice = [e for e in eps if e.get('practice')]
    normal = [e for e in eps if not e.get('practice')]
    def ret(e):
        return float(sum(float(s['reward']) for s in e['steps']))
    rows = [dict(id=e['id'], verification_id=e['practice']['verification_id'],
                 example_id=e['practice']['example_id'], seed_index=int(e['practice']['seed_index']),
                 t0=int(e['practice']['t0']), world=e['world']['seed'], steps=len(e['steps']),
                 fallback=bool(e['practice'].get('fallback')), return_veh_s=ret(e)) for e in practice]
    used = {}
    for r in rows:
        used[r['example_id']] = used.get(r['example_id'], 0)+1
    returns_practice = [r['return_veh_s']/r['steps'] for r in rows if r['steps']]
    returns_normal = [ret(e)/len(e['steps']) for e in normal if e['steps']]
    return dict(practice=len(practice), normal=len(normal),
                practice_return_veh_s=(float(np.mean([r['return_veh_s'] for r in rows])) if rows else None),
                normal_return_veh_s=(float(np.mean([ret(e) for e in normal])) if normal else None),
                practice_return_per_control_s=(float(np.mean(returns_practice)) if returns_practice else None),
                normal_return_per_control_s=(float(np.mean(returns_normal)) if returns_normal else None),
                examples_used=used, examples=sorted(used), episodes=rows,
                rule=PRACTICE_RULE, fallback_rule=R_FALLBACK_RULE)
