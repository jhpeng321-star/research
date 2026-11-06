"""CLEAN4 read-only per-update diagnostic (N4 operator); CPU only, no SUMO/GPU, no torch.

Reads a clean2/clean3/clean4 training run root:
  <root>/{A,B,R}_UPDATE_<u:03>.json   sealed update receipts (launch_training.update)
  <root>/LEARNING_MONITOR_u<u:03>.json
  <root>/g4/ENVELOPE_<u:03>.json       G4 measured envelope (g4_runtime.Coordinator)
  <root>/CHARGES.jsonl                 G4 plan rows only (reservation journal)
  <root>/jobs/<id>/RESULT.json(.gz)    per-episode tapes (optional; B/R trigger records)
  <root>/updates/, <root>/track2_diagnostics/  presence/listing only
and reports per update u and per arm:
  (1) imitation/guidance loss: learning.bank_move_ce / bank_trigger (trigger CE on the verified
      bank rows), learning.bank (effective-space AWR imitation), clean3_mined_usage fit CE;
  (2) trigger fire rate (selected non-none / decisions): all decisions from
      metrics.move_proposals, and from tapes the lesson moment (first trigger tick at/after a
      practice t0), practice, mined-practice and ordinary episodes;
  (3) group moves initiated / joined / executed by move type (tapes; receipt fallback);
  (4) G4 measured synergy credit count / sum / sign (receipt learning.g4_synergy_credit,
      g4/ENVELOPE, clean3_mined_usage.g4_credit, CHARGES g4_plan);
  (5) mean per-episode team delay (totals.team_delay_vs, ordinary episodes) of B/R minus A on
      the same update, paired by world seed. Positive = more delay than A = worse.
Anything absent is reported as 'not logged'; nothing is computed from truth files. The tool
never writes under the run root and refuses forbidden paths (sealed/holdout/... tokens).

CLI: python -m tools.coopmwm.clean4_diag <root> [--update N] [--status-file PATH]
     [--no-tapes] [--json]
With --status-file one 'CLEAN4 diag u<u> ...' line per COMPLETE update (all three arm receipts)
is appended once; reported updates are tracked in <status-file>.state.json (idempotent).
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

NL = 'not logged'
ARMS = ('A', 'B', 'R')
FORBIDDEN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
             'hetdata_truth', 'vehicle_params', 'vtype', 'truth')
SINGLE_ACTOR = ('ease_off', 'gap_open', 'open_gap_merge')   # moves.Move.actors == 1
NO_ACTOR = ('none', 'yield_lane')
INITIATE = ('wall_initiate', 'lane_drop_initiate')            # W1 / N2 protocol initiations
JOIN = ('wall_join', 'lane_drop_join')                        # W1 / N2 protocol joins
PROTOCOL_STATS = ('v2v_wall', 'lane_drop')                    # feasible_proposals per-step stats
PROTOCOL_KEYS = ('offered_join', 'selected_initiate', 'accepted_initiate',
                 'selected_join', 'accepted_join')
DELAY_SIGN = 'arm minus A, mean per ordinary episode, veh-s; positive = more delay than A = worse'
SYNERGY_SIGN = 'synergy_veh_s = min solo J - group J (veh-s); positive = group beat best solo'
TAPE_ENCODING = 'coopmwm-lossless-gzip-v1'
UPDATE_RE = re.compile(r'^([ABR])_UPDATE_(\d{3})\.json$')
STATE_SCHEMA = 'coopmwm-clean4-diag-state-v1'


class ForbiddenPath(ValueError):
    pass


def guard(path):
    text = str(path).replace('\\', '/').lower()
    for token in FORBIDDEN:
        if token in text:
            raise ForbiddenPath(f'refused forbidden path token {token!r}: {path}')
    return Path(path)


def read_json(path):
    """Read-only JSON (optionally .gz); None when absent."""
    path = guard(path)
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        data = stream.read()
    if path.suffix == '.gz':
        data = gzip.decompress(data)
    return json.loads(data)


def digest(value):
    """tools.coopbandit.protocol.digest (canonical sorted compact JSON sha256), no import."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def seal_ok(record, key):
    if not isinstance(record, dict) or key not in record:
        return NL
    return digest({k: v for k, v in record.items() if k != key}) == record[key]


def _num(value):
    if type(value) is bool or not isinstance(value, (int, float)):
        return NL
    if isinstance(value, float) and not math.isfinite(value):
        return NL
    return value


def _get(mapping, *keys):
    for key in keys:
        if not isinstance(mapping, dict) or key not in mapping:
            return NL
        mapping = mapping[key]
    return mapping


def _rate(num, den, source):
    if num == NL or den == NL:
        return NL
    return dict(non_none=num, decisions=den, rate=(num/den if den else None), source=source)


def _stats(values):
    values = [float(v) for v in values]
    total = sum(values)
    return dict(count=len(values), sum=total, mean=(total/len(values) if values else None),
                n_pos=sum(v > 0 for v in values), n_neg=sum(v < 0 for v in values),
                n_zero=sum(v == 0 for v in values),
                sign=('none' if not values else '+' if total > 0 else '-' if total < 0 else '0'))


# ------------------------------------------------------------------ discovery


def updates(root):
    """{u: set(arms with a published UPDATE receipt)} from a root listing."""
    found = {}
    for path in guard(root).iterdir():
        match = UPDATE_RE.match(path.name)
        if match:
            found.setdefault(int(match.group(2)), set()).add(match.group(1))
    return found


def read_tape(root, episode_id):
    """Episode job tape (accel_storage pointer or plain JSON); None when absent."""
    if not isinstance(episode_id, str) or Path(episode_id).name != episode_id:
        raise ValueError('invalid episode id')
    folder = Path(root)/'jobs'/episode_id
    value = read_json(folder/'RESULT.json')
    if isinstance(value, dict) and value.get('encoding') == TAPE_ENCODING:
        if value.get('file') != 'RESULT.json.gz':
            raise ValueError('invalid tape payload pointer')
        path = guard(folder/'RESULT.json.gz')
        if not path.is_file():
            return None
        with path.open('rb') as stream:
            payload = gzip.decompress(stream.read())
        if hashlib.sha256(payload).hexdigest() != value.get('sha256'):
            raise ValueError('tape payload digest mismatch')
        return json.loads(payload)
    return value


def g4_plans(root):
    """{update: plan summary} from CHARGES.jsonl g4_plan rows (absent -> {})."""
    path = guard(Path(root)/'CHARGES.jsonl')
    if not path.is_file():
        return {}
    out = {}
    with path.open('rb') as stream:
        for line in stream:
            if b'"g4_plan"' not in line:
                continue
            try:
                plan = json.loads(line)['g4_plan']
            except (ValueError, KeyError):
                continue   # incomplete live tail; never guessed
            starts = Counter(s.get('arm') for s in plan.get('starts', ()))
            out[plan.get('update')] = dict(quote_sim_s=plan.get('quote_sim_s'),
                starts={a: starts.get(a, 0) for a in ('B', 'R')}, jobs=len(plan.get('jobs', ())),
                plan_sha256=plan.get('plan_sha256'))
    return out


# ------------------------------------------------------------------ (1) imitation


def imitation(receipt):
    learning = receipt.get('learning') if isinstance(receipt.get('learning'), dict) else {}
    trigger = learning.get('bank_trigger') if isinstance(learning.get('bank_trigger'), dict) else None
    bank = learning.get('bank') if isinstance(learning.get('bank'), dict) else None
    mined = receipt.get('clean3_mined_usage')
    out = dict(bank_move_ce=_num(learning.get('bank_move_ce')))
    if trigger is None or 'final_ce' not in trigger:
        out.update(bank_move_ce_initial=NL, bank_move_examples=NL, bank_move_family_counts=NL,
                   bank_move_ce_by_family=NL)
    else:
        out.update(bank_move_ce_initial=_num(trigger.get('initial_ce')),
                   bank_move_examples=_num(trigger.get('examples')),
                   bank_move_family_counts=trigger.get('family_counts', NL))
        contexts, ce = trigger.get('contexts'), trigger.get('context_final_ce')
        if isinstance(contexts, list) and isinstance(ce, list) and len(ce) == len(contexts):
            groups = {}
            for context, value in zip(contexts, ce):
                if _num(value) != NL:
                    groups.setdefault(context.get('family'), []).append(float(value))
            out['bank_move_ce_by_family'] = {f: dict(mean_ce=sum(v)/len(v), contexts=len(v))
                                             for f, v in sorted(groups.items())}
        else:
            out['bank_move_ce_by_family'] = NL
    epochs = bank.get('epochs') if bank else None
    if isinstance(epochs, list) and epochs:
        out.update(bank_imitation_loss_before=_num(epochs[0].get('awr_loss_before')),
                   bank_imitation_loss_after=_num(epochs[-1].get('awr_loss_after')),
                   bank_imitation_epochs_accepted=sum(bool(e.get('accepted')) for e in epochs))
    else:
        out.update(bank_imitation_loss_before=NL, bank_imitation_loss_after=NL,
                   bank_imitation_epochs_accepted=NL)
    for key in ('action_error_before_mps2', 'action_error_after_mps2', 'kl', 'pairs_total'):
        out['bank_imitation_'+key] = _num(bank.get(key)) if bank else NL
    if isinstance(mined, dict):
        out.update(mined_imitation_fit_ce=_num(mined.get('imitation_fit_ce')),
                   mined_imitation_fit_contexts=_num(mined.get('imitation_fit_contexts')),
                   mined_imitation_contexts=(sum(mined['imitation'].values())
                                             if isinstance(mined.get('imitation'), dict) else NL))
    else:
        out.update(mined_imitation_fit_ce=NL, mined_imitation_fit_contexts=NL,
                   mined_imitation_contexts=NL)
    return out


# ------------------------------------------------------------------ (2)/(3) trigger + group


def move_type(move):
    family = move.get('family')
    if family in ('lane_drop_initiate', 'lane_drop_join') and move.get('kind'):
        return f'{family}:{move["kind"]}'
    return family


def group_role(move):
    """'initiated' / 'joined' / None (single-actor or none). Mirrors moves.Move.actors."""
    family = move.get('family')
    if family in INITIATE:
        return 'initiated'
    if family in JOIN:
        return 'joined'
    if family in NO_ACTOR or family in SINGLE_ACTOR:
        return None
    return 'initiated' if len(move.get('members') or ()) >= 2 else None


def _bucket():
    return dict(decisions=0, non_none=0, accepted=0, demo_decisions=0,
                lesson_family_selected=0, initiated={}, joined={},
                protocol={k: None for k in PROTOCOL_STATS})


def _add_protocol(bucket, proposals):
    for name in PROTOCOL_STATS:
        stats = (proposals or {}).get(name)
        if not isinstance(stats, dict):
            continue
        if bucket['protocol'][name] is None:
            bucket['protocol'][name] = dict.fromkeys(PROTOCOL_KEYS, 0)
        for key in PROTOCOL_KEYS:
            bucket['protocol'][name][key] += int(stats.get(key, 0))


def _count_step(bucket, trigger, demo, lesson_family=None):
    members = (trigger.get('context') or {}).get('members') or []
    accepted = trigger.get('accepted') or {}
    for i, ego in enumerate(members):
        if demo:
            bucket['demo_decisions'] += 1
            continue
        action, table = trigger['actions'][i], trigger['tables'][i]
        move = table[action] if type(action) is int and 0 <= action < len(table) else None
        bucket['decisions'] += 1
        if move is None or move.get('family') == 'none':
            continue
        bucket['non_none'] += 1
        ok = bool(accepted.get(ego, False))
        bucket['accepted'] += ok
        if lesson_family is not None and move.get('family') == lesson_family:
            bucket['lesson_family_selected'] += 1
        role = group_role(move)
        if role is not None:
            row = bucket[role].setdefault(move_type(move), dict(selected=0, accepted=0))
            row['selected'] += 1
            row['accepted'] += ok
    _add_protocol(bucket, trigger.get('proposals'))


def _finish(bucket, source):
    out = dict(bucket, rate=(bucket['non_none']/bucket['decisions'] if bucket['decisions'] else None),
               source=source)
    out['protocol'] = {k: (v if v is not None else NL) for k, v in bucket['protocol'].items()}
    return out


def _mined_examples(receipt):
    mined = receipt.get('clean3_mined_usage')
    starts = mined.get('practice_starts') if isinstance(mined, dict) else None
    if not isinstance(starts, dict):
        return None
    return {key.split('|', 1)[1] for key in starts if '|' in key}


KINDS = ('all', 'ordinary', 'practice', 'practice_mined', 'lesson_moment', 'non_lesson')


def scan_tapes(root, receipt):
    """Per-episode trigger records from the B/R job tapes (read-only).

    Kinds: ordinary / practice episodes (receipt practice block, else the tape's own
    'practice' key), practice_mined (example in clean3_mined_usage.practice_starts),
    lesson_moment (first trigger tick at/after the practice t0) and non_lesson (every other
    decision). A kind whose expected episodes have no readable tape is 'not logged'.
    Decisions on plan-executed (behaviour.executed, demonstration) ticks are counted apart.
    """
    physical = receipt.get('physical') if isinstance(receipt.get('physical'), list) else []
    episodes = _get(receipt, 'practice', 'episodes')
    practice = ({e['id']: e for e in episodes if isinstance(e, dict) and 'id' in e}
                if isinstance(episodes, list) else {})
    mined = _mined_examples(receipt)
    buckets = {k: _bucket() for k in KINDS}
    expected, seen = Counter(), Counter()
    read, missing, errors = 0, [], []
    for row in physical:
        episode_id = row.get('id')
        meta = practice.get(episode_id)
        failed = False
        try:
            tape = read_tape(root, episode_id)
        except (ValueError, OSError) as error:
            tape, failed = None, True
            errors.append(dict(id=episode_id, error=str(error)[:200]))
        if meta is None and isinstance(tape, dict) and isinstance(tape.get('practice'), dict):
            meta = tape['practice']
        kinds = ['all', 'non_lesson', 'practice' if meta else 'ordinary']
        if meta:
            kinds.append('lesson_moment')
            if mined is not None and meta.get('example_id') in mined:
                kinds.append('practice_mined')
        expected.update(kinds)
        if tape is None:
            if not failed:
                missing.append(episode_id)
            continue
        read += 1
        seen.update(kinds)
        t0 = meta.get('t0') if meta else None
        family = (meta.get('breadth') or {}).get('family') if meta else None
        steps = sorted((s for s in tape.get('steps', ()) if isinstance((s.get('sample') or {}).get(
            'track2_trigger'), dict)), key=lambda s: s['time_s'])
        lesson_time = next((s['time_s'] for s in steps if t0 is not None and s['time_s'] >= t0), None)
        for step in steps:
            sample = step['sample']
            demo = bool((sample.get('behaviour') or {}).get('executed'))
            lesson = lesson_time is not None and step['time_s'] == lesson_time
            for kind in kinds:
                if kind == 'lesson_moment' and not lesson or kind == 'non_lesson' and lesson:
                    continue
                _count_step(buckets[kind], sample['track2_trigger'], demo,
                            family if kind == 'lesson_moment' else None)
    out = dict(read=read, expected=len(physical), missing=len(missing), errors=errors,
               source=f'jobs/<id>/RESULT.json tapes {read}/{len(physical)}')
    for kind, bucket in buckets.items():
        if kind == 'practice_mined' and mined is None or expected[kind] and not seen[kind]:
            out[kind] = NL
        else:
            out[kind] = _finish(bucket, f'tapes {seen[kind]}/{expected[kind]} episodes')
            out[kind]['episodes'] = seen[kind]
    return out


def trigger_rates(receipt, tapes):
    metrics = receipt.get('metrics') if isinstance(receipt.get('metrics'), dict) else {}
    proposals = metrics.get('move_proposals')
    if isinstance(proposals, dict):
        overall = _rate(_num(proposals.get('selected_non_none')), _num(proposals.get('decisions')),
                        'metrics.move_proposals')
        if isinstance(overall, dict):
            overall['rejected_non_none'] = _num(proposals.get('rejected_non_none'))
    elif isinstance(metrics.get('trigger_counts'), dict) and metrics.get('decisions'):
        counts = metrics['trigger_counts']
        overall = _rate(metrics['decisions']-counts.get('none', 0), metrics['decisions'],
                        'metrics.trigger_counts')
    else:
        overall = NL
    mined = receipt.get('clean3_mined_usage')
    out = dict(all=overall, mined_trigger_rate_by_family=(mined.get('trigger_rate_by_family', NL)
               if isinstance(mined, dict) else NL))
    for kind in ('lesson_moment', 'non_lesson', 'practice', 'practice_mined', 'ordinary'):
        bucket = tapes.get(kind, NL) if isinstance(tapes, dict) else NL
        out[kind] = (NL if bucket == NL else dict(non_none=bucket['non_none'], decisions=bucket['decisions'],
                     rate=bucket['rate'], accepted=bucket['accepted'], demo_decisions=bucket['demo_decisions'],
                     episodes=bucket['episodes'], source=bucket['source'],
                     **(dict(lesson_family_selected=bucket['lesson_family_selected'])
                        if kind == 'lesson_moment' else {})))
    out['tapes'] = (f"{tapes['read']}/{tapes['expected']}" if isinstance(tapes, dict) else NL)
    return out


def group_moves(receipt, tapes):
    metrics = receipt.get('metrics') if isinstance(receipt.get('metrics'), dict) else {}
    counts = metrics.get('trigger_counts')
    proposals = metrics.get('move_proposals') if isinstance(metrics.get('move_proposals'), dict) else None
    out = dict(selected_by_family=({f: n for f, n in counts.items() if f != 'none'}
                                   if isinstance(counts, dict) else NL),
               proposed_by_family=(proposals.get('proposed_by_family', NL) if proposals else NL),
               accepted_moves_total=_num(_get(receipt, 'execution', 'accepted_moves')))
    full = tapes.get('all') if isinstance(tapes, dict) else NL
    if full not in (None, NL):
        out.update(initiated=full['initiated'], joined=full['joined'], source=full['source'])
        for name in PROTOCOL_STATS:
            out[name] = full['protocol'][name]
        out['by_episode_kind'] = {k: dict(initiated=tapes[k]['initiated'], joined=tapes[k]['joined'])
                                  for k in ('ordinary', 'practice') if tapes.get(k) not in (None, NL)}
    else:
        # Receipt-only fallback: protocol families are unambiguous by name (selected only;
        # the receipt keeps no per-family acceptance and no joint-macro arity).
        for role, names in (('initiated', INITIATE), ('joined', JOIN)):
            present = {f: dict(selected=counts[f], accepted=NL) for f in names
                       if isinstance(counts, dict) and f in counts}
            out[role] = present or NL
        out['source'] = 'receipt metrics.trigger_counts (protocol families only)'
        for name in PROTOCOL_STATS:
            out[name] = proposals.get(name, NL) if proposals else NL
    mined = receipt.get('clean3_mined_usage')
    if isinstance(mined, dict) and isinstance(mined.get('executed_joint_moves'), dict):
        executed = Counter()
        for key, n in mined['executed_joint_moves'].items():
            executed[key.split('|', 1)[0]] += n
        out['mined_executed_joint_moves'] = dict(sorted(executed.items()))
    else:
        out['mined_executed_joint_moves'] = NL
    rates = mined.get('trigger_rate_by_family') if isinstance(mined, dict) else None
    decisions = mined.get('trigger_decisions') if isinstance(mined, dict) else None
    if isinstance(rates, dict) and isinstance(decisions, int):
        out['mined_joint_selected'] = {f[:-len(':joint')]: int(round(r*decisions))
                                       for f, r in sorted(rates.items()) if f.endswith(':joint')}
    else:
        out['mined_joint_selected'] = NL
    return out


# ------------------------------------------------------------------ (4) G4


def g4(arm, receipt, envelope, plan):
    out = {}
    credit = _get(receipt, 'learning', 'g4_synergy_credit')
    if isinstance(credit, dict):
        rows = credit.get('rows') or []
        out['receipt'] = dict(_stats(r['synergy_veh_s'] for r in rows),
            advantage_sum=sum(float(r['advantage']) for r in rows),
            measured_starts=credit.get('measured_starts', NL),
            eligible_starts=credit.get('eligible_starts', NL),
            unsampled_starts=credit.get('unsampled_starts', NL),
            source='learning.g4_synergy_credit')
    else:
        out['receipt'] = NL
    if isinstance(envelope, dict) and isinstance(envelope.get('rows'), list):
        rows = [r for r in envelope['rows'] if (r.get('start') or {}).get('arm') == arm]
        out['envelope'] = dict(_stats(r['synergy_veh_s'] for r in rows),
            advantage_sum=sum(float(r['advantage']) for r in rows), source='g4/ENVELOPE')
    else:
        out['envelope'] = NL
    mined = receipt.get('clean3_mined_usage')
    rows = mined.get('g4_credit') if isinstance(mined, dict) else None
    if isinstance(rows, list):
        out['mined'] = dict(_stats(r['measured_synergy_veh_s'] for r in rows),
            advantage_sum=sum(float(r['normalized_advantage']) for r in rows),
            used_in_ppo=sorted({bool(r.get('used_in_ppo')) for r in rows}),
            source='clean3_mined_usage.g4_credit (mining verification)')
    else:
        out['mined'] = NL
    out['plan'] = (dict(starts=plan['starts'].get(arm, 0), quote_sim_s=plan['quote_sim_s'],
                        source='CHARGES.jsonl g4_plan') if isinstance(plan, dict) else NL)
    out['sign_convention'] = SYNERGY_SIGN
    return out


# ------------------------------------------------------------------ (5) team delay


def _ordinary(receipt):
    """(ordinary rows with a numeric team_delay_vs, practice rows excluded) or None."""
    physical = receipt.get('physical')
    if not isinstance(physical, list):
        return None, 0
    episodes = _get(receipt, 'practice', 'episodes')
    practice = {e.get('id') for e in episodes if isinstance(e, dict)} if isinstance(episodes, list) else set()
    rows = [r for r in physical if r.get('id') not in practice]
    return ([r for r in rows if _num(_get(r, 'totals', 'team_delay_vs')) != NL],
            len(physical)-len(rows))


def _mean_delay(rows):
    return sum(r['totals']['team_delay_vs'] for r in rows)/len(rows) if rows else None


def team_delay(receipt, a_receipt):
    """Mean ordinary-episode team delay and the seed-paired difference vs A (same update)."""
    rows, excluded = _ordinary(receipt)
    if rows is None:
        return NL
    out = dict(mean_veh_s=_mean_delay(rows), episodes=len(rows), practice_excluded=excluded,
               sign_convention=DELAY_SIGN)
    reference = _ordinary(a_receipt)[0] if isinstance(a_receipt, dict) else None
    if reference is None:
        out.update(a_mean_veh_s=NL, a_episodes=NL, diff_vs_a_veh_s=NL, paired_n=NL)
        return out

    def by_seed(items):
        grouped = {}
        for r in items:
            grouped.setdefault(_get(r, 'world', 'seed'), []).append(r)
        return grouped
    mine, theirs = by_seed(rows), by_seed(reference)
    diffs, per_seed = [], {}
    for seed, own in sorted(mine.items(), key=lambda kv: str(kv[0])):
        other = theirs.get(seed, [])
        # Same world only: one ordinary episode per arm and an identical world record.
        if seed == NL or len(own) != 1 or len(other) != 1 or own[0].get('world') != other[0].get('world'):
            continue
        diff = own[0]['totals']['team_delay_vs']-other[0]['totals']['team_delay_vs']
        diffs.append(diff)
        per_seed[str(seed)] = diff
    out.update(a_mean_veh_s=_mean_delay(reference), a_episodes=len(reference),
               diff_vs_a_veh_s=(sum(diffs)/len(diffs) if diffs else None), paired_n=len(diffs),
               unpaired_n=len(rows)-len(diffs), per_seed_diff_veh_s=per_seed)
    return out


# ------------------------------------------------------------------ assembly


def monitor(root, u):
    value = read_json(Path(root)/f'LEARNING_MONITOR_u{u:03}.json')
    if not isinstance(value, dict):
        return NL
    rows = {r.get('arm'): dict(action=r.get('action'), diagnose=r.get('diagnose'))
            for r in value.get('arms', ()) if isinstance(r, dict)}
    return dict(seal_ok=seal_ok(value, 'receipt_sha256'), arms=rows)


def diagnose_update(root, u, *, tapes=True, plans=None):
    root = guard(Path(root))
    receipts = {a: read_json(root/f'{a}_UPDATE_{u:03}.json') for a in ARMS}
    envelope = read_json(root/'g4'/f'ENVELOPE_{u:03}.json')
    plan = (plans if plans is not None else g4_plans(root)).get(u)
    arms = {}
    for arm in ARMS:
        receipt = receipts[arm]
        if not isinstance(receipt, dict):
            arms[arm] = dict(arm=arm, update=u, receipt='missing')
            continue
        scanned = scan_tapes(root, receipt) if tapes and arm != 'A' else NL
        arms[arm] = dict(arm=arm, update=u, receipt='present',
            seal_ok=seal_ok(receipt, 'update_sha256'), imitation=imitation(receipt),
            trigger=trigger_rates(receipt, scanned), group_moves=group_moves(receipt, scanned),
            g4=g4(arm, receipt, envelope, plan),
            team_delay=team_delay(receipt, receipts['A'] if arm != 'A' else None),
            tape_errors=(scanned['errors'] if isinstance(scanned, dict) else NL))
    started = sorted(a for a in ARMS if guard(root/'updates'/f'{a}_{u:03}_STARTED.json').is_file())
    folder = guard(root/'track2_diagnostics')
    diagnostics = (sorted(p.name for p in folder.iterdir() if p.suffix == '.json'
                          and f'_u{u:03}' in p.name) if folder.is_dir() else NL)
    report = dict(update=u, root=str(root), complete=all(isinstance(r, dict) for r in receipts.values()),
                  arms=arms, monitor=monitor(root, u), started=started,
                  track2_diagnostics=diagnostics, g4_plan=plan if plan is not None else NL,
                  tapes_read=bool(tapes))
    report['line'] = status_line(report)
    return report


# ------------------------------------------------------------------ status line


def _f(value, digits=4):
    if value == NL or value is None:
        return NL if value == NL else 'n/a'
    if isinstance(value, float):
        return f'{value:.{digits}g}'
    return str(value)


def _r(rate):
    if rate == NL:
        return NL
    if rate['rate'] is None:
        return f"n/a(0/{rate['decisions']})"
    return f"{rate['rate']:.3f}({rate['non_none']}/{rate['decisions']})"


def _moves(rows):
    if rows == NL:
        return NL
    if not rows:
        return 'none'
    return ','.join(f"{k}:{v['selected']}/{_f(v['accepted'])}" for k, v in sorted(rows.items()))


def _protocol(value):
    if value == NL:
        return NL
    return '/'.join(str(value.get(k, NL)) for k in PROTOCOL_KEYS[1:])


def _g4(value):
    for key in ('receipt', 'envelope'):
        stats = value[key]
        if stats != NL:
            break
    if stats != NL:
        text = f"n={stats['count']} sum={_f(stats['sum'])} sign={stats['sign']} [{stats['source']}]"
    elif value['plan'] != NL:
        text = f"plan starts={value['plan']['starts']} envelope {NL}"
    else:
        text = NL
    mined = value['mined']
    if mined != NL:
        text += f" mined n={mined['count']} sum={_f(mined['sum'])} sign={mined['sign']}"
    return text


def status_line(report):
    parts = [f"CLEAN4 diag u{report['update']}"]
    if not report['complete']:
        parts.append('INCOMPLETE receipts=' + ''.join(a for a in ARMS
                                                     if report['arms'][a].get('receipt') == 'present'))
    for arm in ('B', 'R'):
        a = report['arms'][arm]
        if a.get('receipt') != 'present':
            parts.append(f'{arm}: receipt missing')
            continue
        im, tr, gm, td = a['imitation'], a['trigger'], a['group_moves'], a['team_delay']
        imit = (f"imit bank_move_ce={_f(im['bank_move_ce'])} (init {_f(im['bank_move_ce_initial'])}, "
                f"n={_f(im['bank_move_examples'])}) eff_awr={_f(im['bank_imitation_loss_before'])}->"
                f"{_f(im['bank_imitation_loss_after'])} mined_fit_ce={_f(im['mined_imitation_fit_ce'])}")
        fire = (f"fire all={_r(tr['all'])} lesson={_r(tr['lesson_moment'])} "
                f"non_lesson={_r(tr['non_lesson'])} practice={_r(tr['practice'])} "
                f"mined_practice={_r(tr['practice_mined'])} ordinary={_r(tr['ordinary'])} tapes={tr['tapes']}")
        group = (f"group init={_moves(gm['initiated'])} join={_moves(gm['joined'])} "
                 f"exec_total={_f(gm['accepted_moves_total'])} v2v_wall={_protocol(gm['v2v_wall'])} "
                 f"lane_drop={_protocol(gm['lane_drop'])} mined_exec="
                 + (NL if gm['mined_executed_joint_moves'] == NL else
                    ','.join(f'{k}:{n}' for k, n in gm['mined_executed_joint_moves'].items()) or 'none'))
        delay = (NL if td == NL else
                 f"delay-A={_f(td['diff_vs_a_veh_s'], 6)} veh-s (paired n={_f(td['paired_n'])}, "
                 f"mean {_f(td['mean_veh_s'], 6)} vs A {_f(td['a_mean_veh_s'], 6)})")
        parts.append(f'{arm}: ' + '; '.join([imit, fire, group, 'g4 '+_g4(a['g4']), delay]))
    mon = report['monitor']
    parts.append('monitor=' + (NL if mon == NL else '/'.join(
        f"{a}:{(mon['arms'].get(a) or {}).get('action', NL)}" for a in ARMS)))
    parts.append('sign: delay-A>0 = more delay than A = worse; g4 sum>0 = group beat best solo')
    return ' | '.join(parts)


# ------------------------------------------------------------------ CLI


def _outside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or root in path.parents:
        raise ValueError('status/state files must be outside the run root (read-only tool)')
    guard(path)
    return path


def _state_path(status):
    return status.with_name(status.name+'.state.json')


def _load_state(path):
    if not path.is_file():
        return dict(schema=STATE_SCHEMA, roots={})
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema') != STATE_SCHEMA:
        raise ValueError('unknown clean4_diag state schema')
    return value


def _save_state(path, value):
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('root')
    parser.add_argument('--update', type=int)
    parser.add_argument('--status-file')
    parser.add_argument('--no-tapes', action='store_true',
                        help='skip per-episode job tapes (lesson/ordinary split -> not logged)')
    parser.add_argument('--json', action='store_true', help='print full per-update dicts as JSON lines')
    args = parser.parse_args(argv)
    root = guard(Path(args.root))
    if not root.is_dir():
        parser.error('run root not found: '+str(root))
    status = state_path = state = None
    key = str(root.resolve())
    if args.status_file:
        status = _outside(args.status_file, root)
        state_path = _outside(_state_path(status), root)
        state = _load_state(state_path)
    found = updates(root)
    wanted = [args.update] if args.update is not None else sorted(found)
    plans = g4_plans(root)
    for u in wanted:
        complete = found.get(u) == set(ARMS)
        if status is not None and (not complete or u in state['roots'].get(key, [])):
            print(f'clean4_diag u{u}: ' + ('already reported' if complete else
                  'incomplete receipts ' + ''.join(sorted(found.get(u, ())))) + '; not appended',
                  file=sys.stderr)
            continue
        report = diagnose_update(root, u, tapes=not args.no_tapes, plans=plans)
        print(json.dumps(report, sort_keys=True, default=str) if args.json else report['line'])
        if status is not None:
            status.parent.mkdir(parents=True, exist_ok=True)
            with status.open('a', encoding='utf-8') as stream:
                stream.write(report['line']+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            state['roots'][key] = sorted(set(state['roots'].get(key, [])) | {u})
            _save_state(state_path, state)
    return 0


if __name__ == '__main__':
    sys.exit(main())
