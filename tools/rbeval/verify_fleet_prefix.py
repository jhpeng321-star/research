"""Check saved clock/event/front-trace prefixes when extending an inference horizon.

These saved diagnostics, not complete vehicle trajectories, are checked bit-exact.
The short run is retained and is not counted as another independent world/future.
"""
import argparse
import hashlib
import json
from pathlib import Path

from tools.rbeval.full_fleet import safe
from tools.rbeval.summarize_full_fleet import load_panel


def compare(short, extended):
    failures = []
    if len(short) != len(extended) or {r['seed'] for r in short} != {r['seed'] for r in extended}:
        return ['different world panel']
    by_seed = {r['seed']: r for r in extended}
    invariant = ('arm', 'stream', 'cf_model', 'native_model', 'settings', 'entry_phase',
                 'lc_memory', 'cpu_fast', 'episode_sha256', 'tape_sha256', 'inferred_settings', 'source_observations')
    for old in short:
        new = by_seed[old['seed']]
        seed = old['seed']
        h = old['horizon_s']
        if new['horizon_s'] <= h:
            failures.append(f'{seed}: not an extended horizon')
        for key in invariant:
            if old.get(key) != new.get(key):
                failures.append(f'{seed}: changed input contract {key}')
        if old['checkpoints'][str(h)] != new['checkpoints'].get(str(h)):
            failures.append(f'{seed}: changed saved checkpoint {h}')
        for label, a, b in (
                ('merge events', old['merges']['events'], new['merges']['events']),
                ('birth events', old['births'], new['births']),
                ('front trace', old['merge_front_trace'], new['merge_front_trace'])):
            if a != [r for r in b if r['time_s'] <= 300+h]:
                failures.append(f'{seed}: changed {label} prefix')
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--short', required=True)
    p.add_argument('--extended', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    short = safe(args.short)
    extended = safe(args.extended)
    a, b = load_panel(short), load_panel(extended)
    failures = compare(a, b)
    ia = json.loads((short/'IDENTITY.json').read_text())
    ib = json.loads((extended/'IDENTITY.json').read_text())
    for key in ('sources', 'validator_sha256', 'encoder_checkpoint_sha256', 'targets'):
        if ia.get(key) != ib.get(key):
            failures.append(f'identity: changed {key}')
    if sorted(ia.get('auxiliary_sources', {}).values()) != sorted(ib.get('auxiliary_sources', {}).values()):
        failures.append('identity: changed settings adapter')
    result = dict(status='FAIL' if failures else 'PASS', worlds=len(a),
        contract='saved clock/event/front-trace prefixes only; no full trajectory-parity claim',
        failures=failures, source_collections={str(d): hashlib.sha256((d/'COLLECTION.json').read_bytes()).hexdigest()
                                              for d in (short, extended)})
    with safe(args.output).open('x', newline='\n') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
