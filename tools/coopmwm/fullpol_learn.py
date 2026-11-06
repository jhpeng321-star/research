"""FULLPOL learning signals for free_program selection (clean5, DEFAULT OFF; CPU only).

Signals (FULLPOL_DESIGN.md section 4; MAIN 18:4x approval with conditions 1-3):

* exact-program BC: every verified free_program example (Plan A / bank plan carrying an
  LCEXEC ``free_program`` dict) yields trigger CE contexts - each member whose OWN observable
  slots contain every program member, at the earliest observed second before any member's
  first primitive (LCEXEC tick label ``event=activate``, else the earliest member start_el),
  with the exact program (shifted by the elapsed seconds, nothing else) inserted in slot 1 of
  the CAV's own slate. No projection onto any other vocabulary (LEARNAUDIT finding 1).
  Optional own-arm verified NEGATIVE examples -> target ``none`` with the program in slot 1.
* regime EV-BC and EV-credit (owner pivot 2026-11-06 18:21): S-REGIME rows {arm, regime,
  family, mean, lb95, n}. Only families whose LCEXEC converter/parity test PASSES
  (``parity_families``) are used (condition 1). The regime key is an OFFLINE episode label
  (world cell) - it never enters the policy input, slate or descriptor (condition 2). R uses
  ONLY its own matched-effort random-strategy sweep rows; none -> no R terms (condition 3).
* PPO on own choices is the existing per-CAV trigger PPO (free_program decision rows are not
  ``executed``); EV-credit is an additive proposer advantage like G4's.
* G4: ``register_g4_adapter`` plugs free_program into G4LD's ADAPTERS (same estimator).
* Report-only receipts (values unchanged): P1 BC KL measured on UNLABELLED on-policy
  contexts; P2 seed-split validation AUC of the BC procedure (shadow fit, TRAIN seeds only).
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from statistics import fmean

import numpy as np
import torch

from . import fullpol_programs as fp

SCHEMA = 'fullpol-learning-v1'
GUIDANCE_SCHEMA = 'fullpol-regime-guidance-v1'
LEARN_KEYS = ('bc', 'negatives', 'ev_bc', 'ev_credit', 'report_kl_unlabelled', 'report_valsplit_auc')
EV_RULE = ('EV-BC target = registered-admission convention: best parity-passed family with lb95 > 0 in '
           'the episode regime (offline label), else none; EV-credit = clip(weight*mean/UNIT) like G4')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     default=str).encode()).hexdigest()


def learn_settings(cfg):
    """None when FULLPOL is OFF; else the validated learning switches (all default False)."""
    settings = fp.settings(cfg)
    if settings is None:
        return None
    learn = dict.fromkeys(LEARN_KEYS, False)
    raw = settings.get('learn') or {}
    if set(raw)-set(LEARN_KEYS) or any(type(v) is not bool for v in raw.values()):
        raise ValueError(f'fullpol learn switches are booleans {LEARN_KEYS}')
    learn.update(raw)
    return learn


# ---------------------------------------------------------------- regime guidance (offline labels)

def guidance(cfg):
    """Validated regime guidance or None (absent)."""
    settings = fp.settings(cfg)
    if settings is None or settings.get('regime_guidance') is None:
        return None
    g = settings['regime_guidance']
    if (not isinstance(g, dict) or g.get('schema') != GUIDANCE_SCHEMA or g.get('regime_rule') != 'cell'
            or not isinstance(g.get('rows'), list) or not isinstance(g.get('provenance'), str)
            or not g.get('source', {}).get('sha256')):
        raise ValueError('regime guidance needs schema, regime_rule=cell, rows, source sha and provenance')
    for row in g['rows']:
        if (row.get('arm') not in ('B', 'R') or not isinstance(row.get('regime'), str)
                or not isinstance(row.get('family'), str)
                or not all(type(row.get(k)) in (int, float) and math.isfinite(row[k])
                           for k in ('mean_veh_s', 'lb95_veh_s'))
                or type(row.get('n')) is not int or row['n'] < 1):
            raise ValueError('regime guidance rows: arm, regime, family, mean_veh_s, lb95_veh_s, n')
    for key in ('weight', 'clip'):
        if type(g.get(key)) not in (int, float) or not math.isfinite(g[key]) or g[key] <= 0:
            raise ValueError('regime guidance weight/clip must be registered positive numbers')
    return deepcopy(g)


def regime_key(episode):
    """OFFLINE regime label of a training episode (world cell). Never a policy input."""
    world = episode['world']
    return f"D{int(world['demand'])}_p{int(round(float(world['penetration'])*100))}"


def ev_table(cfg, arm):
    """{regime: {family: row}} for this arm's OWN sweep rows, parity-passed families only."""
    g = guidance(cfg)
    if g is None:
        return None, dict(enabled=False)
    settings = fp.settings(cfg)
    parity = set(settings.get('parity_families') or ())
    table, dropped = {}, Counter()
    for row in g['rows']:
        if row['arm'] != arm:
            continue
        if row['family'] not in parity:
            dropped[row['family']] += 1        # condition 1: no exact executor -> no EV term
            continue
        table.setdefault(row['regime'], {})[row['family']] = row
    receipt = dict(enabled=True, arm=arm, rule=EV_RULE, source=g['source'], regimes=sorted(table),
                   families=sorted({f for r in table.values() for f in r}),
                   dropped_without_parity=dict(dropped),
                   r_without_matched_sweep=(arm == 'R' and not table))
    return table, receipt


def best_family(rows):
    """Registered admission convention: lb95 > 0; best by lb95 (ties: mean, name)."""
    ok = [r for r in rows.values() if r['lb95_veh_s'] > 0]
    if not ok:
        return None
    return max(ok, key=lambda r: (r['lb95_veh_s'], r['mean_veh_s'], r['family']))['family']


def option_labels(entry):
    """Family labels of one recorded table entry (dict): legacy family, or the program's
    structured family / variant / verified row family (provenance metadata, offline only)."""
    if entry is None:
        return ()
    if entry.get('family') != fp.FAMILY:
        return (entry['family'],)
    src = entry.get('source') or {}
    return tuple(x for x in (src.get('variant'), src.get('family')) if x)


def ev_credit_terms(cfg, arm, episodes, *, eligible):
    """({(episode_id, time_s, vid): term}, receipt|None). Disabled -> ({}, None) exactly."""
    learn = learn_settings(cfg)
    if learn is None or not learn['ev_credit'] or arm == 'A':
        return {}, None
    table, receipt = ev_table(cfg, arm)
    if table is None:
        return {}, None
    from .accel_learner import UNIT
    g = guidance(cfg)
    terms = {}
    counts = Counter()
    for episode in episodes:
        if episode.get('practice'):
            continue
        rows = table.get(regime_key(episode), {})
        for step in episode['steps']:
            key = (episode['id'], step['time_s'])
            trigger = step['sample'].get('track2_trigger')
            if key not in eligible or trigger is None:
                continue
            for i, vid in enumerate(trigger['context']['members']):
                entry = trigger['tables'][i][trigger['actions'][i]]
                if entry is None or not trigger.get('accepted', {}).get(vid, False):
                    continue
                hits = [rows[f] for f in option_labels(entry) if f in rows]
                if not hits:
                    counts['no_guidance'] += 1
                    continue
                value = g['weight']*hits[0]['mean_veh_s']/float(UNIT)
                terms[(*key, vid)] = max(-g['clip'], min(g['clip'], value))
                counts['credited'] += 1
    receipt.update(credited_rows=counts['credited'], uncredited_selected=counts['no_guidance'],
                   unit=float(UNIT), weight=g['weight'], clip=g['clip'])
    return terms, receipt


def ev_bc_records(cfg, arm, episodes):
    """EV-BC targets on RECORDED slates (the slate never depends on the regime label).

    Per accepted-or-not decision row: target = slot of the best parity-passed lb95>0 family
    of the episode's regime when that family is in the recorded slate; ``none`` when no
    family qualifies; skipped when the qualifying family was not offered."""
    learn = learn_settings(cfg)
    if learn is None or not learn['ev_bc'] or arm == 'A':
        return [], None
    table, receipt = ev_table(cfg, arm)
    if table is None:
        return [], None
    from .track2_backend import _batch
    records, counts = [], Counter()
    for episode in episodes:
        if episode.get('practice'):
            continue
        best = best_family(table.get(regime_key(episode), {}))
        for step in episode['steps']:
            trigger = step['sample'].get('track2_trigger')
            if trigger is None:
                continue
            batch = _batch(trigger)
            targets = []
            for i in range(len(trigger['context']['members'])):
                if best is None:
                    targets.append(0)
                    continue
                slots = [j for j, entry in enumerate(trigger['tables'][i])
                         if trigger['mask'][i][j] and best in option_labels(entry)]
                targets.append(slots[0] if slots else None)
            for i, target in enumerate(targets):
                if target is None:
                    counts['skipped_not_offered'] += 1
                    continue
                counts['none' if target == 0 else 'act'] += 1
                same = [target == 0 and j == 0 or target > 0 and trigger['mask'][i][j] and best in option_labels(e)
                        for j, e in enumerate(trigger['tables'][i])]
                records.append(dict(batch=_row(batch, i, target), weight=1., equivalent=torch.tensor([same]),
                                    meta=dict(kind='ev_bc', episode_id=episode['id'], time_s=step['time_s'],
                                              seed=episode['world']['seed'], family=best or 'none')))
    receipt.update(records=len(records), targets=dict(counts))
    return records, receipt


def _row(batch, i, target):
    from .trigger_policy import TriggerBatch
    context = dict(batch.context, members=[batch.context['members'][i]],
                   x=[batch.context['x'][i]])
    for key, value in batch.context.items():
        if isinstance(value, dict) and set(value) >= {batch.context['members'][i]}:
            context[key] = {batch.context['members'][i]: value[batch.context['members'][i]]}
    return TriggerBatch(context, (batch.tables[i],), batch.descriptors[i:i+1].clone(),
                        batch.mask[i:i+1].clone(), torch.tensor([int(target)]), batch.logp[i:i+1],
                        batch.values[i:i+1])


# ---------------------------------------------------------------- exact-program BC contexts

def first_activation_el(prog, labels=None):
    """Earliest move-clock second at which any member issues a primitive (exactness window)."""
    els = [int(r['el']) for r in labels or () if r.get('event') == 'activate']
    starts = [int(p['start_el'])+int(j) for p, j in zip(prog['programs'], prog['join_el'])]
    return min(els) if els else min(starts)


def shift(prog, elapsed):
    """Same program observed ``elapsed`` seconds later, before any member acted (exact)."""
    e = int(elapsed)
    if e < 0 or e > first_activation_el(prog):
        raise ValueError('shift only inside the pre-activation window')
    out = deepcopy(prog)
    for p in out['programs']:
        p['start_el'] = int(p['start_el'])-e if int(p['start_el']) >= e else 0
        p['lon']['from_el'] = max(0, int(p['lon']['from_el'])-e)
        if p.get('lane') is not None:
            p['lane']['from_el'] = max(0, int(p['lane']['from_el'])-e)
    out['duration_s'] = max(1, int(prog['duration_s'])-e)
    return out


def _context(reader, source, keep):
    from . import v2v_wall
    context = dict(source, members=list(keep), x=[source['x'][source['members'].index(v)] for v in keep],
                   trigger_v2v={v: reader.local_intents[v] for v in keep})
    return v2v_wall.attach_context(context, reader, keep)


def program_contexts(policy, plan, pairs, frames, *, t0, labels=None):
    """Yield (context, option, reader, weight, time) for one verified free_program example.

    Mirrors bank_breadth.observable_contexts: causal reader observes t-1 and t; each member
    is supervised once, at the earliest observed second where ALL program members are in its
    own observable slots, inside the exact pre-activation window."""
    prog = fp.lc_checked(plan['free_program'])
    members = list(prog['members'])
    window = first_activation_el(prog, labels)
    reader = deepcopy(policy)
    observed, remaining = set(), set(members)
    for sample, weight in sorted(pairs, key=lambda pair: int(pair[0]['intent_link']['t'])):
        time = int(sample['intent_link']['t'])
        elapsed = time-int(t0)
        if elapsed < 0 or elapsed > window or time not in frames:
            continue
        for t in (time-1, time):
            if t in frames and t not in observed:
                reader.observe(frames[t]['vehicles'], t)
                observed.add(t)
        source = sample['context']
        keep = [v for v in source['members'] if v in remaining and set(members) <= set(reader.visible.get(v, ()))
                and v in reader.local_intents]
        for vid in keep:
            rows = {r['vid']: r for r in reader.local_states[vid]['rows']}
            if not set(members) <= set(rows):
                continue
            shifted = shift(prog, elapsed)
            order = [vid]+[v for v in members if v != vid]
            index = [members.index(v) for v in order]
            shifted = dict(shifted, members=order, programs=[shifted['programs'][j] for j in index],
                           join_el=[shifted['join_el'][j] for j in index])
            option = fp.FreeProgramOption(tuple(order), shifted, 'verified',
                                          dict(plan.get('fullpol_source') or {}, exact=True),
                                          {v: (int(rows[v]['lane']), float(rows[v]['s_m'])) for v in order})
            yield _context(reader, source, [vid]), option, reader, float(weight), time
        remaining.difference_update(keep)
        if not remaining:
            break


def supervised_batch(policy, cfg, context, option, reader, *, target_none=False):
    """The acting CAV's own (deterministic) slate with ``option`` in slot 1; target 1 or 0."""
    from .trigger_policy import TriggerBatch
    vid = context['members'][0]
    states = {vid: dict(reader.local_states[vid], acquirable_cav_ids=list(reader.visible[vid]))}
    batch = policy.trigger.decide(context, obs_states=states, observable_cavs=reader.visible,
                                  rng=np.random.default_rng(cfg['training_seed']),
                                  generator=torch.Generator().manual_seed(cfg['training_seed']), deterministic=True)
    table = list(batch.tables[0])
    table[1] = option
    descriptors, mask = batch.descriptors.clone(), batch.mask.clone()
    descriptors[0, 1] = torch.tensor(policy.trigger._descriptor(option, reader.visible[vid]))
    mask[0, 1] = True
    out = TriggerBatch(context, (tuple(table),), descriptors, mask,
                       torch.tensor([0 if target_none else 1]), batch.logp, batch.values)
    return out, equivalent_slots(out, 0 if target_none else 1)


def equivalent_slots(batch, target):
    """Slots executing the same thing as ``target`` (identical descriptor; none only itself).

    The acting CAV's own slate may already contain the labelled program (e.g. its verified
    class re-binds the same template): choosing any copy executes the same program, so the
    CE target is the SET (-log sum p) instead of one arbitrary copy (no ln k floor)."""
    if target == 0:
        same = [j == 0 for j in range(batch.mask.shape[1])]
    else:
        ref = batch.descriptors[0, target]
        same = [bool(batch.mask[0, j]) and j > 0 and bool(torch.equal(batch.descriptors[0, j], ref))
                for j in range(batch.mask.shape[1])]
    return torch.tensor([same])


def bank_records(policy, cfg, arm):
    """Exact-program BC (+ own negatives) records from the arm's verified bank examples."""
    learn = learn_settings(cfg)
    if learn is None or not learn['bc'] or arm == 'A':
        return [], None
    from . import bank_breadth, bank_v2v
    records, counts = [], Counter()

    def consume(*, cfg, receipt, example, plan, pairs, trace):
        if plan.get('free_program') is None:
            return
        negative = bool(plan.get('fullpol_negative'))
        if negative and not learn['negatives']:
            return
        frames = bank_v2v.trace_frames(trace)
        found = 0
        for context, option, reader, weight, time in program_contexts(
                policy, plan, pairs, frames, t0=receipt['t0'], labels=plan.get('free_program_labels')):
            batch, same = supervised_batch(policy, cfg, context, option, reader, target_none=negative)
            hdv = sum(1 for r in reader.local_states[context['members'][0]]['rows'] if not r['is_cav'])
            records.append(dict(batch=batch, weight=float(weight), equivalent=same, meta=dict(
                kind='negative' if negative else 'program', verification_id=receipt['verification_id'],
                seed=int((receipt.get('world') or plan.get('world') or {}).get('seed', -1)),
                time_s=time, members=list(option.members), hdv_rows=hdv)))
            found += 1
        counts['examples'] += 1
        counts['examples_without_observable_context'] += int(found == 0)
    bank_breadth.visit(cfg, arm, consume)
    receipt = dict(examples=counts['examples'], records=len(records),
                   unobservable_examples=counts['examples_without_observable_context'],
                   kinds=dict(Counter(r['meta']['kind'] for r in records)),
                   contexts_with_hdv_rows=sum(r['meta']['hdv_rows'] > 0 for r in records))
    return records, receipt


# ---------------------------------------------------------------- the BC update (+ report-only receipts)

def _ce(policy, record):
    """-log P(target set): plain cross entropy when the set is the single labelled slot."""
    b = record['batch']
    logits = policy.trigger.distribution(b.context, b.descriptors, b.mask)[0]
    same = record.get('equivalent')
    if same is None or int(same.sum()) <= 1:
        return torch.nn.functional.cross_entropy(logits, b.actions)
    return (torch.logsumexp(logits, -1)-torch.logsumexp(logits.masked_fill(~same, -torch.inf), -1)).mean()


def _mass(records):
    """Equal mass per record kind (program / negative / ev_bc) and then per record."""
    kinds = Counter(r['meta']['kind'] for r in records)
    return [r['weight']/kinds[r['meta']['kind']]/len(kinds) for r in records]


@torch.no_grad()
def _logits(policy, batches):
    return [policy.trigger.distribution(b.context, b.descriptors, b.mask)[0] for b in batches]


def _mean_kl(before, after):
    values = [torch.distributions.kl_divergence(torch.distributions.Categorical(logits=a),
              torch.distributions.Categorical(logits=b)).sum() for a, b in zip(before, after)]
    return float(torch.stack(values).mean()) if values else 0.


def bc_steps(policy, records, *, epochs, kl_budget, max_grad_norm):
    """Registered rule (identical to bank_move_supervision): each step is kept only if the mean
    KL over the supervised contexts stays <= kl_budget; otherwise reverted and stop."""
    if not records:
        return dict(initial_ce=None, final_ce=None, passes=[])
    weights = _mass(records)
    reference = _logits(policy, [r['batch'] for r in records])

    def loss():
        return torch.stack([w*_ce(policy, r) for r, w in zip(records, weights)]).sum()/sum(weights)
    initial = float(loss().detach())
    passes = []
    for _ in range(int(epochs)):
        state = deepcopy(policy.trigger.state_dict())
        opt = deepcopy(policy.trigger_optimizer.state_dict())
        objective = loss()
        policy.trigger_optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(policy.trigger.learning_parameters(), max_grad_norm,
                                       error_if_nonfinite=True)
        policy.trigger_optimizer.step()
        kl = _mean_kl(reference, _logits(policy, [r['batch'] for r in records]))
        accepted = math.isfinite(kl) and kl <= kl_budget
        if not accepted:
            policy.trigger.load_state_dict(state)
            policy.trigger_optimizer.load_state_dict(opt)
        passes.append(dict(accepted=accepted, kl=kl))
        if not accepted:
            break
    policy.trigger_optimizer.zero_grad(set_to_none=True)
    return dict(initial_ce=initial, final_ce=float(loss().detach()), passes=passes)


def unlabelled_batches(episodes):
    """Recorded on-policy decision rows (P1 report: drift where no label exists)."""
    from .track2_backend import _batch
    out = []
    for episode in episodes:
        for step in episode['steps']:
            if 'track2_trigger' in step['sample']:
                out.append(_batch(step['sample']['track2_trigger']))
    return out


def auc(pos, neg):
    """Mann-Whitney AUC (ties = 1/2); None without both classes."""
    if not pos or not neg:
        return None
    wins = sum((p > q)+.5*(p == q) for p in pos for q in neg)
    return wins/(len(pos)*len(neg))


@torch.no_grad()
def _slot1(policy, record):
    """P(the offered program) = probability mass of every slot equivalent to slot 1."""
    b = record['batch']
    probs = policy.trigger.distribution(b.context, b.descriptors, b.mask)[0].softmax(-1)[0]
    same = equivalent_slots(b, 1)[0]
    return float(probs[same].sum())


def valsplit_auc(policy, records, *, epochs, kl_budget, max_grad_norm, folds=2):
    """P2 (report-only): shadow-fit the BC procedure on seed folds, AUC on the held-back fold.

    Folds by world-seed hash (TRAIN seeds only - the bank never contains evaluation seeds).
    Positives = program records, negatives = negative records. The live policy is untouched."""
    labelled = [r for r in records if r['meta']['kind'] in ('program', 'negative')]
    seeds = sorted({r['meta']['seed'] for r in labelled})
    if len(seeds) < folds:
        return dict(auc=None, reason='fewer seeds than folds', seeds=len(seeds))
    fold = {s: int(_digest(['fullpol-valsplit-v1', s]), 16) % folds for s in seeds}
    out = []
    for k in range(folds):
        train = [r for r in labelled if fold[r['meta']['seed']] != k]
        test = [r for r in labelled if fold[r['meta']['seed']] == k]
        shadow = deepcopy(policy)
        before_pos = [_slot1(shadow, r) for r in test if r['meta']['kind'] == 'program']
        before_neg = [_slot1(shadow, r) for r in test if r['meta']['kind'] == 'negative']
        fit = bc_steps(shadow, train, epochs=epochs, kl_budget=kl_budget, max_grad_norm=max_grad_norm)
        pos = [_slot1(shadow, r) for r in test if r['meta']['kind'] == 'program']
        neg = [_slot1(shadow, r) for r in test if r['meta']['kind'] == 'negative']
        train_pos = [_slot1(shadow, r) for r in train if r['meta']['kind'] == 'program']
        train_neg = [_slot1(shadow, r) for r in train if r['meta']['kind'] == 'negative']
        out.append(dict(fold=k, train=len(train), test=len(test), auc_before=auc(before_pos, before_neg),
                        auc_heldback=auc(pos, neg), auc_train=auc(train_pos, train_neg),
                        mean_p_slot1_heldback_pos=fmean(pos) if pos else None,
                        fit_final_ce=fit['final_ce']))
    vals = [f['auc_heldback'] for f in out if f['auc_heldback'] is not None]
    return dict(auc=fmean(vals) if vals else None, folds=out, seeds=len(seeds),
                rule='seed-hash folds over TRAIN-seed bank contexts; shadow copy; report only')


def supervise(runtime, episodes=()):
    """FULLPOL BC update for one arm/update (after PPO, before save). Disabled -> None."""
    cfg, arm, policy = runtime.cfg, runtime.arm, runtime.policy
    learn = learn_settings(cfg)
    if learn is None or arm == 'A':
        return None
    records, bank = bank_records(policy, cfg, arm)
    ev, ev_receipt = ev_bc_records(cfg, arm, episodes)
    records = records+ev
    unlabelled = unlabelled_batches(episodes) if learn['report_kl_unlabelled'] else []
    before = _logits(policy, unlabelled) if unlabelled else []
    valsplit = None
    epochs, budget = cfg['imitation_epochs'], policy.layer.config.kl_budget
    clip = cfg['trigger_ppo']['max_grad_norm']
    if learn['report_valsplit_auc']:
        valsplit = valsplit_auc(policy, records, epochs=epochs, kl_budget=budget, max_grad_norm=clip)
    fit = bc_steps(policy, records, epochs=epochs, kl_budget=budget, max_grad_norm=clip)
    policy.layer._check_frozen()
    out = dict(schema=SCHEMA, arm=arm, records=len(records), bank=bank, ev_bc=ev_receipt, fit=fit,
               kl_rule='registered: mean KL over the supervised contexts <= kl_budget (unchanged)',
               kl_budget=budget)
    if unlabelled:
        out['report_kl_unlabelled'] = dict(kl=_mean_kl(before, _logits(policy, unlabelled)),
            contexts=len(unlabelled), note='P1 report only: drift on unlabelled on-policy contexts')
    if valsplit is not None:
        out['report_valsplit_auc'] = valsplit
    return out


# ---------------------------------------------------------------- G4 (G4LD estimator) adapter

def solo_program(prog, vid):
    """The G4 solo branch: only ``vid``'s member program (other members keep the base policy)."""
    i = prog['members'].index(vid)
    return dict(deepcopy(prog), members=[vid], programs=[deepcopy(prog['programs'][i])], join_el=[0],
                duration_s=int(prog['programs'][i]['start_el'])+int(prog['programs'][i]['hold_s'] or 0)
                or int(prog['duration_s']))


def _g4_row(episode, steps, index, ego, selected, trigger, members, pending):
    from .sparse_synergy_credit import digest
    option = fp.option_from_dict(selected)
    if option.members[0] != ego or not set(option.members) <= set(members):
        raise ValueError('FULLPOL G4 free_program proposer/member mismatch')
    if len(option.members) < 2:
        return None              # a one-member primitive is not a joint start
    start = steps[index]['time_s']
    row = dict(episode_id=episode['id'], time_s=start, proposer=ego, move=option.to_dict(),
               trigger_sha256=digest(trigger))
    row['opportunity_id'] = digest(row)
    row['g4ld'] = dict(kind='fp_joint', family=fp.FAMILY, members=list(option.members),
                       schedule=[dict(vid=v, start_s=start) for v in option.members],
                       credit=[dict(vid=ego, time_s=start)])
    return row


def _g4_check(start):
    g = start['g4ld']
    option = fp.option_from_dict(start['move'])
    if (g['kind'] != 'fp_joint' or g['members'] != list(option.members) or len(option.members) < 2
            or start['proposer'] != option.members[0]
            or g['schedule'] != [dict(vid=v, start_s=start['time_s']) for v in option.members]
            or g['credit'] != [dict(vid=start['proposer'], time_s=start['time_s'])]):
        raise ValueError('FULLPOL G4 free_program schedule/credit mismatch')


def register_g4_adapter():
    """Plug free_program into G4LD's estimator (no-op when the G4LD extension is absent)."""
    from . import sparse_synergy_credit as g4
    adapters = getattr(g4, 'ADAPTERS', None)
    if adapters is None:
        return False
    adapters[fp.FAMILY] = dict(row=_g4_row, check=_g4_check, solo=solo_program)
    return True


# ---------------------------------------------------------------- reproduction receipt

def reproduction(verified_labels, executed_labels):
    """Per-second agreement of executed primitives with the verified LCEXEC tick labels."""
    key = lambda r: (r['vid'], int(r['el']))
    ref, got = {key(r): r for r in verified_labels}, {key(r): r for r in executed_labels}
    common = sorted(set(ref) & set(got))
    if not common:
        return dict(seconds=0, lane_agreement=None, lon_agreement=None, accel_mae=None)
    lane = fmean(float(ref[k]['lane_target'] == got[k]['lane_target']) for k in common)
    lon = fmean(float(ref[k]['lon'] == got[k]['lon']) for k in common)
    errs = [abs(ref[k]['accel_mps2']-got[k]['accel_mps2']) for k in common
            if ref[k]['accel_mps2'] is not None and got[k]['accel_mps2'] is not None]
    return dict(seconds=len(common), missing=len(set(ref)-set(got)), lane_agreement=lane, lon_agreement=lon,
                accel_mae=fmean(errs) if errs else None)


__all__ = ['learn_settings', 'guidance', 'regime_key', 'ev_table', 'best_family', 'ev_credit_terms',
           'ev_bc_records', 'program_contexts', 'supervised_batch', 'bank_records', 'bc_steps', 'supervise',
           'valsplit_auc', 'auc', 'shift', 'first_activation_el', 'solo_program', 'register_g4_adapter',
           'reproduction']
