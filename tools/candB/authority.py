"""Candidate-B launch authority: full source/config, ranking, WM and caprel bindings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.candA.authority import (clean_tree, effective_config, resolve_chain,
                                   source_commit, verify_cct2)
from tools.candB.rankcheck import (WM_SHA, binding, runtime_sources, sha256,
                                   verify_receipt)
from tools.distill.runtime import configure_execution, safe_path

SCHEMA = 'candB_launch_authority/1'
ENTRY = 'configs/c4_policy14_candB.toml'


def inspect(inputs):
    raw = effective_config(Path.cwd(), ENTRY)
    raw['horizon']['gate_file'] = inputs['horizon_gate']
    if raw['trainer']['entrant_learned_mix'] is not True:
        raise ValueError('candidate B requires actual entrant mix')
    if sha256(inputs['wm']) != WM_SHA:
        raise ValueError('candidate B requires WM2')
    ranking = json.loads(safe_path(inputs['rankcheck']).read_text())
    verify_receipt(ranking, raw, wm=WM_SHA, wm_config=sha256(inputs['wm_config']))
    for key, name in (('current_policy_sha256', 'current_policy'),
                      ('boundary_sha256', 'boundary'), ('pairing_receipt_sha256', 'pairing_receipt'),
                      ('cct2_sha256', 'cct2'), ('horizon_gate_sha256', 'horizon_gate')):
        if ranking.get(key) != sha256(inputs[name]):
            raise ValueError('ranking input mismatch: '+name)
    cct2 = json.loads(safe_path(inputs['cct2']).read_text())
    if verify_cct2(cct2, WM_SHA, learned_mix=True) != 300:
        raise ValueError('candidate B requires objective-qualified H_G=300')
    if cct2.get('objective_contract') != ranking['runtime_contract']['objective']:
        raise ValueError('trained objective differs from WM qualification')
    expected_cct2 = raw['long_credit']['cct2_file'].format(wm_sha256=WM_SHA)
    if sha256(expected_cct2) != sha256(inputs['cct2']):
        raise ValueError('preset CCT-2 differs from bound qualification')
    configure_execution('configs/execparity_cav_idm_v1.json')
    from wms.control.execution import verify_parity
    verify_parity(inputs['parity'], residual_reference='ctrl0')
    return raw


def manifest():
    path = Path('SOURCE_MANIFEST.json')
    if path.is_file():
        entries = json.loads(path.read_text())
        for name, digest in entries.items():
            if sha256(name) != digest:
                raise ValueError('staged source mismatch: '+name)
        return entries
    files = set(runtime_sources()) | set(resolve_chain(Path.cwd(), ENTRY)) | {
        'tools/candB/authority.py', 'tools/candB/launch.sh', 'tools/candB/stage.sh',
        'tools/candB/bundle.py'}
    return {name: sha256(name) for name in sorted(files)}


def build(inputs, out):
    if not clean_tree(Path.cwd()):
        raise ValueError('commit the source tree before binding launch authority')
    raw = inspect(inputs)
    doc = dict(schema=SCHEMA, source_commit=source_commit(Path.cwd()), inputs=inputs,
        input_sha256={name: sha256(path) for name, path in inputs.items()},
        config_sha256=binding(raw), runtime_manifest=manifest(),
        execution=dict(mode=2, residual_reference='ctrl0', leader_rule=True),
        fresh_zero_start=True, training_launched=False)
    with safe_path(out).open('x', encoding='utf-8') as stream:
        json.dump(doc, stream, indent=1, allow_nan=False)
        stream.write('\n')
    return doc


def verify(path):
    doc = json.loads(safe_path(path).read_text())
    if doc.get('schema') != SCHEMA or source_commit(Path.cwd()) != doc['source_commit']:
        raise ValueError('launch source/schema mismatch')
    raw = inspect(doc['inputs'])
    if binding(raw) != doc['config_sha256'] or manifest() != doc['runtime_manifest']:
        raise ValueError('launch runtime/config manifest mismatch')
    for name, digest in doc['input_sha256'].items():
        if sha256(doc['inputs'][name]) != digest:
            raise ValueError('launch input hash mismatch: '+name)
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    for name in ('wm', 'wm-config', 'current-policy', 'cct2', 'horizon-gate',
                 'boundary', 'pairing-receipt', 'parity', 'rankcheck',
                 'real-returns', 'native-returns', 'gate-panel', 'out'):
        b.add_argument('--'+name, required=True)
    sub.add_parser('verify').add_argument('authority')
    a = p.parse_args()
    if a.command == 'build':
        inputs = {name: str(safe_path(value)) for name, value in vars(a).items()
                  if name not in ('command', 'out')}
        doc = build(inputs, a.out)
    else:
        doc = verify(a.authority)
    print(json.dumps(dict(verified=True, source_commit=doc['source_commit']), indent=1))


if __name__ == '__main__':
    main()
