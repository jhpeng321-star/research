"""Strict per-update learning receipts. Pure diagnostics never pause a safe run."""
import json
import math
from pathlib import Path

from . import rerun_cpus as rc
from .exact_grade import _seal, verify_sealed

METRICS = ('group_credit_wm', 'group_credit_checked', 'dense_ce', 'dense_accuracy',
           'dense_filtered_share', 'dense_checked_accuracy',
           'bank_move_ce', 'layer_kl', 'merge_missing_share', 'reward')
COUNTS = ('decisions', 'credit_rows', 'credit_checked_rows', 'dense_labels',
          'dense_eligible_labels', 'dense_checked_labels',
          'augmented_examples', 'weighted_starts')
SAFETY = ('collisions', 'teleports', 'hard_brakes', 'ttc_events')


def checked_metrics(arm, metrics):
    missing = (set(METRICS+COUNTS+('trigger_counts', 'families', 'safety', 'safety_stop'))
               - set(metrics))
    if missing:
        raise ValueError('REFUSED: missing monitor metrics: '+', '.join(sorted(missing)))
    for name in METRICS:
        value = metrics[name]
        if value is None and arm == 'A' and name != 'reward':
            continue
        if value is None and name in ('dense_ce', 'dense_accuracy') and metrics['dense_eligible_labels'] == 0:
            continue
        if value is None and name == 'dense_checked_accuracy' and metrics['dense_checked_labels'] == 0:
            continue
        if value is None and name == 'group_credit_checked' and metrics['credit_checked_rows'] == 0:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('REFUSED: invalid monitor metric: '+name)
        if name in ('dense_accuracy', 'dense_filtered_share', 'dense_checked_accuracy', 'merge_missing_share') and not 0 <= value <= 1:
            raise ValueError('REFUSED: monitor share outside [0,1]: '+name)
        if name in ('dense_ce', 'bank_move_ce', 'layer_kl') and value < 0:
            raise ValueError('REFUSED: negative monitor loss: '+name)
    for name in COUNTS:
        if type(metrics[name]) is not int or metrics[name] < 0:
            raise ValueError('REFUSED: invalid monitor count: '+name)
    families = metrics['families']
    triggers = metrics['trigger_counts']
    if not families or 'none' not in families or set(triggers) != set(families):
        raise ValueError('REFUSED: monitor needs every move family')
    if (any(type(n) is not int or n < 0 for n in triggers.values())
            or sum(triggers.values()) != metrics['decisions']):
        raise ValueError('REFUSED: inconsistent monitor trigger counts')
    if arm != 'A' and any(metrics[k] <= 0 for k in
                          ('decisions', 'credit_rows', 'dense_labels')):
        raise ValueError('REFUSED: missing learning component observations')
    if metrics['credit_checked_rows'] > metrics['credit_rows']:
        raise ValueError('REFUSED: inconsistent credit verification count')
    if (metrics['dense_eligible_labels'] > metrics['dense_labels'] or
            metrics['dense_checked_labels'] > metrics['dense_labels']):
        raise ValueError('REFUSED: inconsistent dense counts')
    if arm != 'A' and not math.isclose(metrics['dense_filtered_share'],
            1-metrics['dense_eligible_labels']/metrics['dense_labels']):
        raise ValueError('REFUSED: inconsistent dense filtered share')
    for name in SAFETY:
        n = metrics['safety'].get(name)
        if type(n) is not int or n < 0:
            raise ValueError('REFUSED: missing/invalid monitor safety counter: '+name)
    if type(metrics['safety_stop']) is not bool:
        raise ValueError('REFUSED: safety_stop must be explicit')


def monitor_row(arm, update, metrics, history=()):
    if arm not in ('A', 'B', 'R') or type(update) is not int or update < 1:
        raise ValueError('REFUSED: invalid monitor arm/update')
    checked_metrics(arm, metrics)
    if [r['update'] for r in history] != list(range(1, update)):
        raise ValueError('REFUSED: monitor history must be contiguous, no skipped or repeated updates')
    for r in history:
        verify_sealed(r, 'monitor_sha256')
        if r['arm'] != arm:
            raise ValueError('REFUSED: monitor history arm mismatch')
    decisions = metrics['decisions']
    rates = {f: n/decisions if decisions else None for f, n in metrics['trigger_counts'].items()}
    diagnose = []
    if arm != 'A':
        active_rate = 1-rates['none']
        if active_rate in rc.LEARNING_MONITOR['trigger_saturation']:
            diagnose.append('trigger_saturated')
        if metrics['layer_kl'] <= rc.LEARNING_MONITOR['layer_kl_near_zero']:
            diagnose.append('layer_kl_near_zero')
        window = [r['metrics'] for r in history[-2:]]+[metrics]
        if len(window) == rc.LEARNING_MONITOR['history_updates']:
            if all(b['group_credit_wm'] <= a['group_credit_wm'] for a, b in zip(window, window[1:])):
                diagnose.append('group_credit_not_rising_3_updates')
            accuracy = [r['dense_accuracy'] for r in window]
            if all(v is not None for v in accuracy) and max(accuracy)-min(accuracy) <= rc.LEARNING_MONITOR['accuracy_tolerance']:
                diagnose.append('dense_accuracy_flat_3_updates')
        if metrics['dense_eligible_labels'] == 0:
            diagnose.append('no_eligible_dense_labels')
    return _seal(dict(schema='coopmwm.t2c.learning_monitor.v1', arm=arm, update=update,
        metrics=metrics, trigger_rates=rates, diagnose=diagnose,
        action='PAUSE_SAFETY' if metrics['safety_stop'] else ('DIAGNOSE' if diagnose else 'CONTINUE'),
        previous_monitor_sha256=history[-1]['monitor_sha256'] if history else None), 'monitor_sha256')


def monitor_history(root, arm):
    folder = Path(rc.guard_path(root))/'learning_monitor'/arm
    rc.guard_path(folder.resolve())
    rows = []
    for path in sorted(folder.glob('u*.json')):
        rc.guard_path(path.resolve())
        row = json.loads(path.read_text(encoding='utf-8'))
        expected = monitor_row(arm, len(rows)+1, row['metrics'], rows)
        if row != expected:
            raise ValueError('REFUSED: invalid monitor history receipt')
        rows.append(row)
    return rows


def write_monitor(root, arm, update, metrics):
    history = monitor_history(root, arm)
    row = monitor_row(arm, update, metrics, history)
    folder = Path(rc.guard_path(root))/'learning_monitor'/arm
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f'u{update:03d}.json'
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
    return row


def read_monitor(root):
    rows = {a: monitor_history(root, a) for a in ('A', 'B', 'R')}
    return dict(schema='coopmwm.t2c.monitor_status.v1',
                latest={a: rs[-1] if rs else None for a, rs in rows.items()},
                updates={a: len(rs) for a, rs in rows.items()})
