"""Identity gate for retiring the two already-superseded, stopped run trees.

No process discovery by name, no training restart, and no filesystem deletion.
The caller pins a metadata inventory before registration and revalidates every
target immediately before sending a signal. Unattributed orphans are excluded.
"""

PARENTS = {
    2146725: ('54595438', '/home/ubuntu/miniconda3/bin/python -B -u -m '
             'tools.coopmwm.volume_runner /home/ubuntu/coopmwm20261101_v2 --only-A '),
    2477255: ('55545604', '/home/ubuntu/miniconda3/bin/python -B -u -m '
             'tools.coopmwm.accel_runner /home/ubuntu/coopmwm20261101_accel1 '),
}
SPAWN = '/home/ubuntu/miniconda3/bin/python -B -c from multiprocessing.spawn import spawn_main; '
TRACKER = '/home/ubuntu/miniconda3/bin/python -B -c from multiprocessing.resource_tracker import main;'


def targets(rows, *, expected_boot, actual_boot):
    if not expected_boot or expected_boot != actual_boot:
        raise ValueError('boot identity changed')
    by_pid = {r['pid']: r for r in rows}
    if len(by_pid) != len(rows):
        raise ValueError('duplicate PID')
    chosen = []
    for pid, (start, cmd) in PARENTS.items():
        r = by_pid[pid]
        if (r['start_ticks'] != start or r['cmd'] != cmd or r['state'] != 'T'
                or r['ppid'] != 1 or r['uid'].split() != ['1000'] * 4):
            raise ValueError('superseded parent identity/state changed')
        chosen.append(r)
    descendants = set(PARENTS)
    while True:
        extra = {r['pid'] for r in rows if r['ppid'] in descendants} - descendants
        if not extra:
            break
        descendants.update(extra)
    for pid in sorted(descendants - PARENTS.keys()):
        r = by_pid[pid]
        if (r['ppid'] in PARENTS and r['uid'].split() == ['1000'] * 4
                and r['cmd'].startswith(TRACKER) and r['state'] == 'S'
                and r['cwd'] == by_pid[r['ppid']]['cwd']):
            continue  # trackers drain after parked clients exit; never signal them
        if r['ppid'] != 2477255 or r['uid'].split() != ['1000'] * 4:
            raise ValueError('unexpected descendant; no live grandchild may be orphaned')
        if (not r['cmd'].startswith(SPAWN) or not r['cmd'].endswith('--multiprocessing-fork ')
                or r['state'] != 'T'
                or r['cwd'] != '/home/ubuntu/coopmwm20261101_accel1'):
            raise ValueError('descendant is not a parked owned worker')
        chosen.append(r)
    return sorted(chosen, key=lambda r: (r['pid'] in PARENTS, r['pid']))


def unchanged(expected, actual):
    """Reject PID reuse, resumed work, changed ownership or changed ancestry."""
    keys = ('pid', 'ppid', 'start_ticks', 'uid', 'cmd', 'cwd', 'state', 'cpu_ticks')
    if any(expected[k] != actual[k] for k in keys):
        raise ValueError('process changed after registration; no signal sent')
    if actual['state'] != 'T':
        raise ValueError('only stopped processes may be retired')
