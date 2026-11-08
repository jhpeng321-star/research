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


def active_slot(cfg):
    """The enabled learning slot: clean5a ``clean5_ev`` (N2 EV mode) or ``fullpol`` (programs)."""
    from . import fullpol_ev
    ev = fullpol_ev.settings(cfg)
    if ev is not None:
        return ev, 'clean5_ev'
    settings = fp.settings(cfg)
    return (settings, 'fullpol') if settings is not None else (None, None)


def learn_settings(cfg):
    """None when both clean5a and FULLPOL are OFF; else the validated switches (default False)."""
    slot, _ = active_slot(cfg)
    if slot is None:
        return None
    learn = dict.fromkeys(LEARN_KEYS, False)
    raw = slot.get('learn') or {}
    if set(raw)-set(LEARN_KEYS) or any(type(v) is not bool for v in raw.values()):
        raise ValueError(f'fullpol learn switches are booleans {LEARN_KEYS}')
    learn.update(raw)
    return learn


# ---------------------------------------------------------------- regime guidance (offline labels)

BASELINES = ('mappo', 'native')     # MAIN 19:2x: estimates relative to the MAPPO base where available
GAIN_SIGN = ('gain = J(base) - J(strategy) in veh-s per episode (positive = lower team delay); S-REGIME '
             'rg_analyze reports difference = strategy - base, so gain mean = -mean_diff and '
             'lb95(gain) = -ub95(difference)')


def guidance(cfg):
    """Validated regime guidance or None (absent)."""
    slot, _ = active_slot(cfg)
    if slot is None or slot.get('regime_guidance') is None:
        return None
    g = slot['regime_guidance']
    if (not isinstance(g, dict) or g.get('schema') != GUIDANCE_SCHEMA or g.get('regime_rule') != 'cell'
            or not isinstance(g.get('rows'), list) or not isinstance(g.get('provenance'), str)
            or not g.get('source', {}).get('sha256')):
        raise ValueError('regime guidance needs schema, regime_rule=cell, rows, source sha and provenance')
    for row in g['rows']:
        if (row.get('arm') not in ('B', 'R') or not isinstance(row.get('regime'), str)
                or not isinstance(row.get('family'), str) or row.get('baseline', 'native') not in BASELINES
                or not all(type(row.get(k)) in (int, float) and math.isfinite(row[k])
                           for k in ('mean_veh_s', 'lb95_veh_s'))
                or type(row.get('n')) is not int or row['n'] < 1):
            raise ValueError('regime guidance rows: arm, regime, family, baseline, mean_veh_s, lb95_veh_s, n')
    for key in ('weight', 'clip'):
        if type(g.get(key)) not in (int, float) or not math.isfinite(g[key]) or g[key] <= 0:
            raise ValueError('regime guidance weight/clip must be registered positive numbers')
    return deepcopy(g)


def regime_key(episode):
    """OFFLINE regime label of a training episode (world cell). Never a policy input."""
    world = episode['world']
    if world.get('cell_key'):
        return world['cell_key']                 # map_scenarios.world: the registered cell key
    return f"D{int(world['demand'])}_p{int(round(float(world['penetration'])*100))}"


def ev_table(cfg, arm):
    """{regime: {family: row}} from this arm's OWN sweep rows, parity-passed families only;
    per (regime, family) the MAPPO-relative row when present, else the native-relative row."""
    g = guidance(cfg)
    if g is None:
        return None, dict(enabled=False)
    slot, _ = active_slot(cfg)
    parity = set(slot.get('parity_families') or ())
    # MAIN 21:5x: per REGIME (cell) the MAPPO-relative (S-REGIMEU0) rows are used whenever that
    # cell has any; native S-REGIME rows only for cells without MAPPO data (never mixed in a cell).
    table, dropped, used = {}, Counter(), Counter()
    mine = [r for r in g['rows'] if r['arm'] == arm]
    chosen = {}
    for regime in sorted({r['regime'] for r in mine}):
        rows = [r for r in mine if r['regime'] == regime]
        for baseline in BASELINES:
            pick = [r for r in rows if r.get('baseline', 'native') == baseline]
            if pick:
                chosen[regime] = (baseline, pick)
                break
    from .fullpol_fair import settings as fair_settings
    fair = fair_settings(cfg)
    unfair = Counter()
    for regime, (baseline, rows) in sorted(chosen.items()):
        for row in rows:
            if fair is not None and fair.get('ev_filter') and row.get('fair') is not True:
                unfair[row['family']] += 1           # clean5b-F: group delay increase > 10 s, banned or unchecked
                continue
            if row['family'] not in parity:
                dropped[row['family']] += 1        # condition 1: no exact executor -> no EV term
                continue
            table.setdefault(regime, {})[row['family']] = row
            used[baseline] += 1
    table = {k: v for k, v in table.items() if v}
    receipt = dict(enabled=True, arm=arm, rule=EV_RULE, sign=GAIN_SIGN, source=g['source'],
                   regimes=sorted(table), families=sorted({f for r in table.values() for f in r}),
                   baselines_used=dict(used), regime_baseline={k: b for k, (b, _) in sorted(chosen.items())},
                   dropped_without_parity=dict(dropped),
                   **(dict(dropped_unfair=dict(unfair)) if fair is not None else {}),
                   r_without_matched_sweep=(arm == 'R' and not table))
    return table, receipt


FAIRNESS_ADMISSION = 'p20-fairness-priority-v1'   # owner 2026-11-08: D2400_p20 fairness targets (guidance row flag)


def qualifying(rows):
    """Registered admission convention: families with lb95 > 0, best first (lb95, mean, name). A row flagged
    admission = FAIRNESS_ADMISSION (p20 D2400 fairness priority, set only by the p20 guidance builder) also
    qualifies; it ranks after every lb95 > 0 family by the same key. Unflagged docs: unchanged."""
    ok = [r for r in rows.values() if r['lb95_veh_s'] > 0 or r.get('admission') == FAIRNESS_ADMISSION]
    return [r['family'] for r in sorted(ok, key=lambda r: (-r['lb95_veh_s'], -r['mean_veh_s'], r['family']))]


def best_family(rows):
    q = qualifying(rows)
    return q[0] if q else None


def option_labels(entry, cfg=None):
    """Strategy labels of one recorded table entry (dict); provenance metadata, offline only."""
    if entry is None:
        return ()
    if entry.get('family') == fp.FAMILY:
        src = entry.get('source') or {}
        return tuple(x for x in (src.get('variant'), src.get('family')) if x)
    if entry.get('family') in ('free_program_initiate', 'free_program_join'):
        # clean5b (LCEXEC ProgramOption): the template's own strategy labels (fullpol_templates)
        src = (entry.get('template') or {}).get('source') or {}
        return tuple(x for x in (src.get('variant'), src.get('family')) if x)
    from . import fullpol_ev, lane_drop
    ld = lane_drop.settings(cfg) if cfg is not None else None
    if ld is not None and entry.get('family') in lane_drop.JOINT_FAMILIES+lane_drop.PROTOCOL_FAMILIES:
        return fullpol_ev.labels(entry, ld['view'], fullpol_ev.settings(cfg))
    return (entry['family'],)


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
                hits = [rows[f] for f in option_labels(entry, cfg) if f in rows]
                if not hits:
                    counts['no_guidance'] += 1
                    continue
                value = g['weight']*hits[0]['mean_veh_s']/float(UNIT)
                terms[(*key, vid)] = max(-g['clip'], min(g['clip'], value))
                counts['credited'] += 1
    receipt.update(credited_rows=counts['credited'], uncredited_selected=counts['no_guidance'],
                   unit=float(UNIT), weight=g['weight'], clip=g['clip'])
    return terms, receipt


def ev_targets(cfg, trigger, rows, admit=None):
    """Per decision row: (target slot, equivalent-slot mask) or None (the CAV had no choice).

    Target = the highest-lb95 qualifying (lb95 > 0, parity-passed) family OFFERED in this
    recorded slate; ``none`` when no qualifying family is offered. Equivalent slots = every
    masked slot carrying that family label (choosing any of them executes that strategy).
    ``admit(row, i)`` (CLEAN5C4 privileged teacher, B only): a family whose row it rejects for decision row i
    is skipped (default None: unchanged). Deterministic; no RNG."""
    order = qualifying(rows)
    out = []
    for i in range(len(trigger['context']['members'])):
        table, mask = trigger['tables'][i], trigger['mask'][i]
        if not any(mask[1:]):
            out.append(None)
            continue
        target, same = 0, [j == 0 for j in range(len(table))]
        for family in order:
            if admit is not None and not admit(rows[family], i):
                continue
            slots = [j for j, entry in enumerate(table) if j and mask[j] and family in option_labels(entry, cfg)]
            if slots:
                target, same = slots[0], [j in slots for j in range(len(table))]
                break
        out.append((target, same))
    return out


def _learnfix_targets(cfg, trigger, rows, verdict):
    """learnfix labels per decision row: (target slot, equivalent mask, kind) with kind 'act' / 'none' /
    'unlabelled' (see learnfix module doc), or None (no choice). ``verdict(row, i)`` -> teacher verdict."""
    order = qualifying(rows)
    out = []
    for i in range(len(trigger['context']['members'])):
        table, mask = trigger['tables'][i], trigger['mask'][i]
        if not any(mask[1:]):
            out.append(None)
            continue
        found, rejected = None, False
        for family in order:
            slots = [j for j, entry in enumerate(table) if j and mask[j] and family in option_labels(entry, cfg)]
            if not slots:
                continue
            v = 'n/a' if verdict is None else verdict(rows[family], i)
            if v in ('n/a', 'admit'):
                found = slots
                break
            rejected = rejected or v == 'reject'
        if found:
            out.append((found[0], [j in found for j in range(len(table))], 'act'))
        elif rejected:
            out.append((0, [j == 0 for j in range(len(table))], 'none'))
        else:
            out.append((0, [j == 0 for j in range(len(table))], 'unlabelled'))
    return out


def _learnfix_teacher(cfg, episode, rows, counts):
    """Per-step verdict function (None without a privileged situational row)."""
    if not any((r.get('situational') or {}).get('mode') == 'privileged' for r in rows.values()):
        return None
    from pathlib import Path
    from .fullpol_situational import TapeTeacher, teacher_verdict
    tape = Path(cfg['run_root'])/'jobs'/episode['id']/'observable.jsonl.gz'
    if not tape.exists():
        raise ValueError(f'CLEAN5C4 teacher tape missing for episode {episode["id"]}: {tape}')
    teacher = TapeTeacher(tape)

    def per_step(trigger, time_s):
        members = trigger['context']['members']

        def verdict(row, i):
            v = teacher_verdict(row, teacher, members[i], time_s)
            counts['teacher_'+v] += 1
            return v
        return verdict
    return per_step


def _learnfix_records(cfg, arm, episodes, table, receipt, fix):
    from .track2_backend import _batch
    from .trigger_policy import TriggerBatch
    from . import learnfix
    pending, counts, families = [], Counter(), Counter()
    for episode in episodes:
        if episode.get('practice'):
            continue
        rows = table.get(regime_key(episode), {})
        if not rows:
            counts['episodes_without_regime_rows'] += 1
            continue
        per_step = _learnfix_teacher(cfg, episode, rows, counts)
        for step in episode['steps']:
            trigger = step['sample'].get('track2_trigger')
            if trigger is None:
                continue
            targets = _learnfix_targets(cfg, trigger, rows, None if per_step is None else per_step(trigger, step['time_s']))
            if all(t is None or t[2] == 'unlabelled' for t in targets):
                counts['unlabelled'] += sum(t is not None for t in targets)
                counts['no_choice'] += sum(t is None for t in targets)
                continue
            pending.append((episode, step, trigger, targets))
    kinds = [t[2] for _, _, _, ts in pending for t in ts if t is not None and t[2] != 'unlabelled']
    w = learnfix.balance(kinds, fix['row_weight_cap'])
    records = []
    for episode, step, trigger, targets in pending:
        batch = _batch(trigger)
        width = batch.mask.shape[1]
        actions = torch.tensor([0 if t is None else t[0] for t in targets])
        same = torch.tensor([[j == 0 for j in range(width)] if t is None else t[1] for t in targets])
        weight = torch.tensor([0. if t is None or t[2] == 'unlabelled' else w[t[2]] for t in targets])
        for t, table_row in zip(targets, trigger['tables']):
            counts['no_choice' if t is None else t[2]] += 1
            if t is not None and t[2] == 'act':
                labels = option_labels(table_row[t[0]], cfg)
                families[next((f for f in labels if f in table.get(regime_key(episode), {})), labels[0])] += 1
        total = float(weight.sum())
        if total <= 0:
            continue
        records.append(dict(batch=TriggerBatch(batch.context, batch.tables, batch.descriptors, batch.mask,
                                               actions, batch.logp, batch.values),
                            weight=total, equivalent=same, row_weight=weight,
                            meta=dict(kind='ev_bc', episode_id=episode['id'], time_s=step['time_s'],
                                      seed=episode['world']['seed'], regime=regime_key(episode))))
    receipt.update(records=len(records), rows=dict(counts), target_families=dict(families),
                   learnfix=dict(rule=learnfix.RULE, row_weight=w, act_rows=kinds.count('act'),
                                 none_rows=kinds.count('none'),
                                 weighting='record weight = sum of its row weights (update-level act:none balance)'))
    return records, receipt


def ev_bc_records(cfg, arm, episodes):
    """EV-BC records on RECORDED slates (slates never depend on the regime label): one record
    per decision step (all CAV rows at once), per-row targets, equivalent-slot sets, row weights."""
    learn = learn_settings(cfg)
    if learn is None or not learn['ev_bc'] or arm == 'A':
        return [], None
    table, receipt = ev_table(cfg, arm)
    if table is None:
        return [], None
    from . import learnfix
    fix = learnfix.settings(cfg)
    if fix is not None:
        return _learnfix_records(cfg, arm, episodes, table, receipt, fix)
    from .track2_backend import _batch
    from .trigger_policy import TriggerBatch
    records, counts, families = [], Counter(), Counter()
    for episode in episodes:
        if episode.get('practice'):
            continue
        rows = table.get(regime_key(episode), {})
        if not rows:
            counts['episodes_without_regime_rows'] += 1
            continue
        admit_step = _teacher(cfg, episode, rows, counts)
        for step in episode['steps']:
            trigger = step['sample'].get('track2_trigger')
            if trigger is None:
                continue
            targets = ev_targets(cfg, trigger, rows, None if admit_step is None else admit_step(trigger, step['time_s']))
            if all(t is None for t in targets):
                continue
            batch = _batch(trigger)
            width = batch.mask.shape[1]
            actions = torch.tensor([0 if t is None else t[0] for t in targets])
            same = torch.tensor([[j == 0 for j in range(width)] if t is None else t[1] for t in targets])
            weight = torch.tensor([0. if t is None else 1. for t in targets])
            for t, table_row in zip(targets, trigger['tables']):
                if t is None:
                    counts['no_choice'] += 1
                    continue
                counts['none' if t[0] == 0 else 'act'] += 1
                if t[0]:
                    labels = option_labels(table_row[t[0]], cfg)
                    families[next((f for f in labels if f in rows), labels[0])] += 1
            records.append(dict(batch=TriggerBatch(batch.context, batch.tables, batch.descriptors, batch.mask,
                                                   actions, batch.logp, batch.values),
                                weight=1., equivalent=same, row_weight=weight,
                                meta=dict(kind='ev_bc', episode_id=episode['id'], time_s=step['time_s'],
                                          seed=episode['world']['seed'], regime=regime_key(episode))))
    receipt.update(records=len(records), rows=dict(counts), target_families=dict(families))
    return records, receipt


def _teacher(cfg, episode, rows, counts):
    """CLEAN5C4: when a guidance row carries a privileged situational block, the episode's recorded observable
    tape (run_root/jobs/<id>/observable.jsonl.gz) gives the teacher bin at each decision second. None otherwise."""
    if not any((r.get('situational') or {}).get('mode') == 'privileged' for r in rows.values()):
        return None
    from pathlib import Path
    from .fullpol_situational import TapeTeacher, teacher_admit
    teacher = TapeTeacher(Path(cfg['run_root'])/'jobs'/episode['id']/'observable.jsonl.gz')

    def per_step(trigger, time_s):
        members = trigger['context']['members']

        def admit(row, i):
            ok = teacher_admit(row, teacher, members[i], time_s)
            if (row.get('situational') or {}).get('mode') == 'privileged':
                counts['teacher_admitted' if ok else 'teacher_blocked'] += 1
            return ok
        return admit
    return per_step


GUIDANCE_RULE = ('S-REGIME registered analysis (rg_analyze.py cell_stats): per strategy x cell paired seed '
                 'differences of episode team delay J_total, seed bootstrap 10,000 resamples RNG 20261106, '
                 '2.5/97.5% quantiles')


J_FIELDS = ('J_total', 'J_300_1200_vs', 'team_delay_vs')   # S-REGIME episode J / S-REGIMEU0 control-window J


def _j_field(rows):
    for key in J_FIELDS:
        if all(key in r for r in rows):
            return key
    raise ValueError(f'episode rows carry none of the team-delay fields {J_FIELDS}')


def guidance_rows(results, *, arm='B', baseline='native', base_arm='base', strategies=None):
    """S-REGIME / S-REGIMEU0 episode rows -> guidance rows (gain sign, see GAIN_SIGN).

    Reproduces rg_analyze.cell_stats' mean and CI (same paired differences, same bootstrap,
    RNG and quantiles), then flips the sign: gain = -(strategy - base)."""
    ok = [r for r in results if r.get('status') == 'ok']
    E = {(r['cell'], r['seed'], r['arm']): r for r in ok}
    J = _j_field(ok) if ok else 'J_total'
    out = []
    for cell in sorted({r['cell'] for r in ok}):
        seeds = sorted({s for (c, s, a) in E if c == cell and a == base_arm})
        arms = sorted({a for (c, s, a) in E if c == cell and a != base_arm})
        for a in arms:
            if strategies is not None and a not in strategies:
                continue
            ss = [s for s in seeds if (cell, s, a) in E]
            if len(ss) < 2:
                continue
            d = np.asarray([E[(cell, s, a)][J]-E[(cell, s, base_arm)][J] for s in ss], float)
            rng = np.random.default_rng(20261106)
            idx = rng.integers(0, len(d), (10000, len(d)))
            m = d[idx].mean(1)
            lo, hi = float(np.quantile(m, .025)), float(np.quantile(m, .975))
            out.append(dict(arm=arm, regime=cell, family=a, baseline=baseline, n=len(ss),
                            mean_veh_s=-float(d.mean()), lb95_veh_s=-hi, ub95_veh_s=-lo,
                            harm_rate=float((d > 0).mean()), j_field=J))
    return out


def guidance_doc(rows, *, sources, provenance, weight=1., clip=1.):
    """Config value for ``regime_guidance`` (sources = [{name, sha256, baseline}])."""
    sha = _digest(dict(rows=rows, sources=sources))
    return dict(schema=GUIDANCE_SCHEMA, regime_rule='cell', provenance=provenance, weight=float(weight),
                clip=float(clip), source=dict(sources=sources, sha256=sha, rule=GUIDANCE_RULE, sign=GAIN_SIGN),
                rows=rows)


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
        if p.get('tau_from_el') is not None:                 # LCEXEC MemberProgram (LCMOVES tau_from)
            p['tau_from_el'] = max(0, int(p['tau_from_el'])-e)
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


def program_protocol_contexts(policy, plan, pairs, frames, *, t0):
    """clean5b (LCEXEC lc_families): initiate/join BC contexts for one verified free_program example.

    ``plan['free_program']`` = lc_convert.from_mined_row output (concrete vids, initiator first).
    Same rule as the N2/W1 protocol BC: the initiator imitates ``free_program_initiate`` (its
    template = lc_families.template_from_program at its own decision second) from its own causal
    observation; every other member imitates ``free_program_join`` at the earliest second its OWN
    receive step offers the reconstructed session at its TRUE slot (exactness: an offer for another
    slot is not a label). Yields (context, option, reader, weight, time)."""
    from . import lc_families as lf, lc_program as lp
    if getattr(policy, 'lc_families', None) is None:
        raise ValueError('clean5b program BC needs the lc_families protocol')
    prog = lp.FreeProgram.from_dict(plan['free_program'])
    members = list(prog.members)
    by_t = {}
    for sample, weight in sorted(pairs, key=lambda pair: int(pair[0]['intent_link']['t'])):
        by_t.setdefault(int(sample['intent_link']['t']), (sample, weight))
    reader = deepcopy(policy)
    reader.lc_families.reset()
    t0, initiator, sid, remaining = int(t0), members[0], None, set(members[1:])
    window = first_activation_el(plan['free_program'], plan.get('free_program_labels'))
    for t in range(t0-1, t0+int(prog.duration_s)):
        if t not in frames:
            continue
        reader.observe(frames[t]['vehicles'], t)
        if t < t0:
            continue
        pair = by_t.get(t)
        if sid is None:
            if pair is None or t-t0 > window or initiator not in pair[0]['context']['members']                     or initiator not in reader.local_intents:
                continue
            rows = {r['vid']: r for r in frames[t]['vehicles']}
            shifted = lp.FreeProgram.from_dict(shift(plan['free_program'], t-t0))
            template = lf.template_from_program(shifted, reader.lc_families.frame, rows)
            sid = f'{lf.NAME}:{initiator}@{t}'
            option = lf.ProgramOption(f'{lf.NAME}_initiate', (initiator,), sid, initiator, 0, template.to_dict(), 0,
                                      int(template.duration_s), 0., 1, reader.lc_families.cap, 0.,
                                      reader.lc_families.range_m, int(rows[initiator]['lane']))
            yield _context(reader, pair[0]['context'], [initiator]), option, reader, float(pair[1]), t
            reader.lc_families.open_offline(option, t)
        elif pair is not None and remaining:
            for vid in sorted(remaining):
                slot = members.index(vid)
                offer = next((o for o in reader.lc_families.offers.get(vid, ())
                              if o.session_id == sid and o.slot == slot), None)
                if (offer is None or vid not in pair[0]['context']['members'] or vid not in reader.local_intents
                        or not reader.lc_families.admit_offline(offer, t)):
                    continue
                remaining.discard(vid)
                yield _context(reader, pair[0]['context'], [vid]), offer, reader, float(pair[1]), t
        if sid is not None:
            reader.lc_families.publish(t, {v: r.source for v, r in reader.reports.items()})
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
        if getattr(policy, 'lc_families', None) is not None:      # clean5b: LCEXEC initiate/join
            contexts = program_protocol_contexts(policy, plan, pairs, frames, t0=receipt['t0'])
        else:                                                     # FULLPOL group option (superseded)
            contexts = program_contexts(policy, plan, pairs, frames, t0=receipt['t0'],
                                        labels=plan.get('free_program_labels'))
        for context, option, reader, weight, time in contexts:
            batch, same = supervised_batch(policy, cfg, context, option, reader, target_none=negative)
            hdv = sum(1 for r in reader.local_states[context['members'][0]]['rows'] if not r['is_cav'])
            records.append(dict(batch=batch, weight=float(weight), equivalent=same, meta=dict(
                kind='negative' if negative else 'program', verification_id=receipt['verification_id'],
                option_family=getattr(option, 'family', None),
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
    """Weighted mean over rows of -log P(target set) (plain CE for a single labelled slot)."""
    b = record['batch']
    logits = policy.trigger.distribution(b.context, b.descriptors, b.mask)[0]
    same = record.get('equivalent')
    if same is None:
        same = torch.nn.functional.one_hot(b.actions, logits.shape[-1]).bool()
    nll = torch.logsumexp(logits, -1)-torch.logsumexp(logits.masked_fill(~same, -torch.inf), -1)
    w = record.get('row_weight')
    if w is None:
        return nll.mean()
    return (nll*w).sum()/w.sum().clamp_min(1e-12)


def _mass(records):
    """Equal mass per record kind (program / negative / ev_bc) and then per record."""
    kinds = Counter(r['meta']['kind'] for r in records)
    return [r['weight']/kinds[r['meta']['kind']]/len(kinds) for r in records]


@torch.no_grad()
def _logits(policy, batches):
    return [policy.trigger.distribution(b.context, b.descriptors, b.mask)[0] for b in batches]


def _mean_kl(before, after, row_weights=None):
    """Mean KL over decision CONTEXTS = CAV decision rows (one row per bank/program record, every
    labelled row of a multi-CAV EV step); unlabelled rows (weight 0) are not supervised contexts."""
    rows, weights = [], []
    for i, (a, b) in enumerate(zip(before, after)):
        kl = torch.distributions.kl_divergence(torch.distributions.Categorical(logits=a),
                                               torch.distributions.Categorical(logits=b))
        w = torch.ones_like(kl) if row_weights is None or row_weights[i] is None else (row_weights[i] > 0).float()
        rows.append(kl*w)
        weights.append(w)
    if not rows:
        return 0.
    total = torch.cat(weights).sum()
    return float(torch.cat(rows).sum()/total) if float(total) > 0 else 0.


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
        kl = _mean_kl(reference, _logits(policy, [r['batch'] for r in records]),
                      [r.get('row_weight') for r in records])
        accepted = math.isfinite(kl) and kl <= kl_budget
        if not accepted:
            policy.trigger.load_state_dict(state)
            policy.trigger_optimizer.load_state_dict(opt)
        passes.append(dict(accepted=accepted, kl=kl))
        if not accepted:
            break
    policy.trigger_optimizer.zero_grad(set_to_none=True)
    return dict(initial_ce=initial, final_ce=float(loss().detach()), passes=passes)


def unlabelled_batches(episodes, labelled_keys=()):
    """Recorded on-policy decision steps NOT used by any BC record (P1: drift where no label exists)."""
    from .track2_backend import _batch
    labelled = set(labelled_keys)
    out = []
    for episode in episodes:
        for step in episode['steps']:
            if 'track2_trigger' in step['sample'] and (episode['id'], step['time_s']) not in labelled:
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


@torch.no_grad()
def _ev_rows(policy, record):
    """(P(act), P(target set), target is act) for every weighted EV row of one decision step."""
    b = record['batch']
    probs = policy.trigger.distribution(b.context, b.descriptors, b.mask)[0].softmax(-1)
    out = []
    for i in range(len(b.actions)):
        if float(record['row_weight'][i]) <= 0:
            continue
        out.append((float(1.-probs[i, 0]), float(probs[i][record['equivalent'][i]].sum()), int(b.actions[i]) > 0))
    return out


def _scores(policy, records):
    prog = [_slot1(policy, r) for r in records if r['meta']['kind'] == 'program']
    neg = [_slot1(policy, r) for r in records if r['meta']['kind'] == 'negative']
    rows = [x for r in records if r['meta']['kind'] == 'ev_bc' for x in _ev_rows(policy, r)]
    return dict(program_auc=auc(prog, neg),
                ev_act_auc=auc([a for a, _, t in rows if t], [a for a, _, t in rows if not t]),
                ev_target_nll=fmean(-math.log(max(p, 1e-12)) for _, p, _ in rows) if rows else None,
                ev_rows=len(rows), program=len(prog), negative=len(neg))


def valsplit_auc(policy, records, *, epochs, kl_budget, max_grad_norm, folds=2):
    """P2 (report-only): shadow-fit the BC procedure on seed folds; score the held-back fold.

    Folds by world-seed hash (TRAIN seeds only). Program records: AUC of P(slot-1 program)
    positives vs own-arm negatives. EV records: held-back target NLL and AUC of P(act) for
    act-target rows vs none-target rows. The live policy is untouched (deepcopy)."""
    seeds = sorted({r['meta']['seed'] for r in records})
    if len(seeds) < folds:
        return dict(auc=None, reason='fewer seeds than folds', seeds=len(seeds))
    fold = {s: int(_digest(['fullpol-valsplit-v1', s]), 16) % folds for s in seeds}
    out = []
    for k in range(folds):
        train = [r for r in records if fold[r['meta']['seed']] != k]
        test = [r for r in records if fold[r['meta']['seed']] == k]
        if not train or not test:
            out.append(dict(fold=k, train=len(train), test=len(test), skipped='empty fold'))
            continue
        shadow = deepcopy(policy)
        before = _scores(shadow, test)
        fit = bc_steps(shadow, train, epochs=epochs, kl_budget=kl_budget, max_grad_norm=max_grad_norm)
        out.append(dict(fold=k, train=len(train), test=len(test), heldback_before=before,
                        heldback_after=_scores(shadow, test), train_after=_scores(shadow, train),
                        fit_final_ce=fit['final_ce'], accepted_steps=sum(p['accepted'] for p in fit['passes'])))
    vals = [f['heldback_after']['program_auc'] for f in out if 'heldback_after' in f
            and f['heldback_after']['program_auc'] is not None]
    ev = [f['heldback_after']['ev_act_auc'] for f in out if 'heldback_after' in f
          and f['heldback_after']['ev_act_auc'] is not None]
    return dict(auc=fmean(vals) if vals else None, ev_act_auc=fmean(ev) if ev else None, folds=out,
                seeds=len(seeds), rule='seed-hash folds over TRAIN-seed contexts; shadow copy; report only')


def supervise(runtime, episodes=()):
    """FULLPOL / clean5a BC update for one arm/update (after PPO, before save). Disabled -> None."""
    cfg, arm, policy = runtime.cfg, runtime.arm, runtime.policy
    learn = learn_settings(cfg)
    if learn is None or arm == 'A':
        return None
    records, bank = bank_records(policy, cfg, arm)
    ev, ev_receipt = ev_bc_records(cfg, arm, episodes)
    records = records+ev
    demo_receipt = None
    from . import demo_bc
    demo = demo_bc.settings(cfg) if arm == 'B' else None
    if demo is not None:
        # c4/learnfix: the fixed verified-seed demo set, its own record kind (equal mass vs online targets)
        doc = demo_bc.load(demo['path'], demo['sha256'])
        demo_records = demo_bc.demo_records(doc['items'])
        records = records+demo_records
        demo_receipt = dict(rule=demo_bc.RULE, sha256=demo['sha256'], records=len(demo_records),
                            act=sum(r['meta']['label'] == 'act' for r in demo_records),
                            none=sum(r['meta']['label'] == 'none' for r in demo_records))
    labelled = {(r['meta']['episode_id'], r['meta']['time_s']) for r in ev}
    unlabelled = unlabelled_batches(episodes, labelled) if learn['report_kl_unlabelled'] else []
    before = _logits(policy, unlabelled) if unlabelled else []
    valsplit = None
    slot, _ = active_slot(cfg)
    # clean5a registered EV-BC steps (MAIN ~21:1x: 60); otherwise the shared imitation_epochs
    epochs = slot.get('ev_bc_steps', cfg['imitation_epochs'])
    budget = policy.layer.config.kl_budget
    clip = cfg['trigger_ppo']['max_grad_norm']
    if learn['report_valsplit_auc']:
        valsplit = valsplit_auc(policy, records, epochs=epochs, kl_budget=budget, max_grad_norm=clip)
    fit = bc_steps(policy, records, epochs=epochs, kl_budget=budget, max_grad_norm=clip)
    policy.layer._check_frozen()
    accepted = [p['kl'] for p in fit['passes'] if p['accepted']]
    out = dict(schema=SCHEMA, arm=arm, records=len(records), bank=bank, ev_bc=ev_receipt, fit=fit,
               **({} if demo_receipt is None else dict(demo_bc=demo_receipt)),
               kl_rule='registered: mean KL over the supervised contexts <= kl_budget (unchanged)',
               kl_budget=budget, epochs=epochs,
               report_bc_kl=dict(kl_supervised=accepted[-1] if accepted else 0., accepted_steps=len(accepted),
                                 note='report only: cumulative KL on the supervised contexts after the update'))
    if learn['report_kl_unlabelled']:
        out['report_kl_unlabelled'] = dict(
            kl=_mean_kl(before, _logits(policy, unlabelled)) if unlabelled else None, contexts=len(unlabelled),
            note='P1 report only: drift on on-policy decision steps that carry no BC label')
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


# ---------------------------------------------------------------- G4 for clean5b program sessions

PROGRAM_INITIATE, PROGRAM_JOIN = 'free_program_initiate', 'free_program_join'


def _program_session_joins(steps, index, option):
    """Accepted joins of ONE LCEXEC program session from the recorded slates, in decision order."""
    end = steps[index]['time_s']+int(option['hold'])
    out = []
    for step in steps[index+1:]:
        if step['time_s'] >= end:
            break
        trigger = step['sample'].get('track2_trigger')
        if trigger is None:
            continue
        for i, vid in enumerate(trigger['context']['members']):
            sel = trigger['tables'][i][trigger['actions'][i]]
            if (sel is None or sel.get('family') != PROGRAM_JOIN or sel.get('session_id') != option['session_id']
                    or not trigger.get('accepted', {}).get(vid, False)):
                continue
            if sel['members'] != [vid] or vid in [m['vid'] for m in out] or vid == option['initiator']:
                raise ValueError('FULLPOL G4 recorded program join does not match its session')
            out.append(dict(vid=vid, start_s=step['time_s'], option=sel, slot=int(sel['slot']),
                            eligible=not step['sample'].get('deterministic')
                            and not step['sample'].get('behaviour', {}).get('executed')))
    return out


def _g4_program_row(episode, steps, index, ego, selected, trigger, members, pending):
    """G4LD adapter row for an accepted LCEXEC free_program_initiate (session of >= 2 members)."""
    from .sparse_synergy_credit import digest
    start = steps[index]['time_s']
    if (selected.get('family') != PROGRAM_INITIATE or selected['members'] != [ego] or selected['initiator'] != ego
            or selected['session_id'] != f'free_program:{ego}@{start}'):
        raise ValueError('FULLPOL G4 program initiation mismatch')
    joins = [] if pending else _program_session_joins(steps, index, selected)
    schedule = [dict(vid=ego, start_s=start, slot=0)]+[dict(vid=j['vid'], start_s=j['start_s'], slot=j['slot'])
                                                       for j in joins]
    if not pending and len(schedule) < 2:
        return None              # nobody joined: a single-actor program, not a joint start
    row = dict(episode_id=episode['id'], time_s=start, proposer=ego, move=selected, trigger_sha256=digest(trigger))
    row['opportunity_id'] = digest(row)
    row['g4ld'] = dict(kind='fp_session', family=PROGRAM_INITIATE, members=[s['vid'] for s in schedule],
                       schedule=schedule, credit=[dict(vid=ego, time_s=start)]+[
                           dict(vid=j['vid'], time_s=j['start_s']) for j in joins if j['eligible']])
    return row


def _g4_program_check(start):
    g = start['g4ld']
    members, schedule = g['members'], g['schedule']
    if (g['kind'] != 'fp_session' or g['family'] != PROGRAM_INITIATE or len(members) < 2
            or members != [s['vid'] for s in schedule] or len(set(members)) != len(members)
            or schedule[0] != dict(vid=start['proposer'], start_s=start['time_s'], slot=0)
            or len({s['slot'] for s in schedule}) != len(schedule)
            or any(s['start_s'] < start['time_s'] for s in schedule)
            or not g['credit'] or g['credit'][0] != dict(vid=start['proposer'], time_s=start['time_s'])):
        raise ValueError('FULLPOL G4 program session schedule/credit mismatch')


def _g4_program_macro(job, observer):
    raise NotImplementedError(
        'G4 branch execution of LCEXEC program sessions needs the program executor + lc_tick lane writer '
        'in g4_runtime (LCEXEC/G4LD); keep free_program_initiate out of g4_extension.families until it lands')


def register_g4_program_adapter():
    """Register the clean5b session adapter (rows/check; macro fails closed until G4 branches exist)."""
    from . import sparse_synergy_credit as g4
    adapters = getattr(g4, 'ADAPTERS', None)
    if adapters is None:
        return False
    adapters[PROGRAM_INITIATE] = dict(row=_g4_program_row, check=_g4_program_check, macro=_g4_program_macro)
    return True


# ---------------------------------------------------------------- reproduction receipt

def reproduction(verified_labels, executed_labels):
    """Per-second agreement of executed primitives with the verified LCEXEC tick labels."""
    def key(r):
        return r['vid'], int(r['el'])
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


# ---------------------------------------------------------------- guidance builder (CPU, read-only inputs)

def _base_arm(results, base_arm):
    if base_arm is not None:
        return base_arm
    arms = {r.get('arm') for r in results}
    return 'u0' if 'u0' in arms else 'base'      # S-REGIMEU0 names its MAPPO base arm 'u0'


BASE_ENTRIES = ('base', 'u0')                       # base arms (mean_J only), never a guidance family
DERIVED_ENTRIES = ('group_minus_best_solo',)          # derived contrasts (point/ci95/solos), never a family


def guidance_rows_from_analysis(analysis, *, arm='B', baseline):
    """rg_analyze / S-REGIMEU0 ANALYSIS ``cells`` summaries (difference = strategy - base) -> gain rows.

    Only STRATEGY entries (dicts carrying mean_diff, ci95 and n) become rows; the base entry
    ('base' / 'u0') and derived contrasts ('group_minus_best_solo') are excluded explicitly."""
    out = []
    for cell, arms in sorted(analysis['cells'].items()):
        for a, v in sorted(arms.items()):
            if a in BASE_ENTRIES or a in DERIVED_ENTRIES or 'minus' in a:
                continue
            if not isinstance(v, dict) or not {'mean_diff', 'ci95'} <= set(v) or not {'n', 'n_paired'} & set(v):
                raise ValueError(f'unrecognised ANALYSIS entry {cell}/{a}: strategy rows need mean_diff, ci95, n')
            lo, hi = v['ci95']
            out.append(dict(arm=arm, regime=cell, family=a, baseline=baseline, n=int(v.get('n_paired', v.get('n'))),
                            mean_veh_s=-float(v['mean_diff']), lb95_veh_s=-float(hi), ub95_veh_s=-float(lo),
                            harm_rate=float(v.get('harm_rate', float('nan')))))
    return out


def build_guidance(*, native=None, mappo=None, provenance, arm='B', base_arm=None):
    """S-REGIME (native-relative) and/or S-REGIMEU0 (MAPPO-relative) REGIME_RESULTS files ->
    the ``regime_guidance`` config value. ev_table prefers the MAPPO-relative row per
    (regime, family). Only TRAIN-type REGIME seeds are read; no evaluation data."""
    from pathlib import Path
    rows, sources = [], []
    for path, baseline in ((native, 'native'), (mappo, 'mappo')):
        if path is None:
            continue
        raw = Path(path).read_bytes()
        text = raw.decode('utf-8')
        if Path(path).suffix == '.json':       # an rg_analyze ANALYSIS.json summary
            rows += guidance_rows_from_analysis(json.loads(text), arm=arm, baseline=baseline)
            count = None
        else:                                  # episode rows (regime_episode/1)
            results = [json.loads(line) for line in text.splitlines() if line.strip()]
            rows += guidance_rows(results, arm=arm, baseline=baseline, base_arm=_base_arm(results, base_arm))
            count = len(results)
        sources.append(dict(name=Path(path).name, sha256=hashlib.sha256(raw).hexdigest(), baseline=baseline,
                            episodes=count))
    if not sources:
        raise ValueError('at least one S-REGIME results file required')
    return guidance_doc(rows, sources=sources, provenance=provenance)


def main(argv=None):
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description='FULLPOL clean5a guidance builder (CPU, no physics)')
    ap.add_argument('command', choices=['guidance'])
    ap.add_argument('--native', help='S-REGIME REGIME_RESULTS.jsonl or ANALYSIS.json (native base)')
    ap.add_argument('--mappo', help='S-REGIMEU0 results jsonl or ANALYSIS.json (MAPPO u0 base)')
    ap.add_argument('--provenance', required=True)
    ap.add_argument('--arm', default='B', choices=['B', 'R'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    doc = build_guidance(native=a.native, mappo=a.mappo, provenance=a.provenance, arm=a.arm)
    out = Path(a.out)
    if out.exists():
        raise SystemExit('refusing to overwrite '+str(out))
    out.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding='utf-8')
    print(json.dumps(dict(out=str(out), rows=len(doc['rows']), sha256=doc['source']['sha256'])))


if __name__ == '__main__':
    main()
