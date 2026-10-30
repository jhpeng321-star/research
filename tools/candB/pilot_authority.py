"""Owner-20261030 exploratory authority; never manufactures a qualification PASS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.candA.authority import effective_config, source_commit
from tools.candB.authority import ENTRY, manifest
from tools.candB.rankcheck import binding, canonical, WM_SHA
from tools.distill.runtime import configure_execution, safe_path, sha256

FLAG = '--exploratory-owner-20261030'
SCHEMA = 'candB_EXPLORATORY_owner_20261030/1'
INPUTS = ('wm', 'wm_config', 'current_policy', 'current_policy_driver', 'cct2',
          'horizon_gate', 'boundary', 'pairing_receipt', 'parity', 'real_returns',
          'native_returns', 'gate_panel')


def require_opt_in(enabled):
    if enabled is not True:
        raise ValueError('EXPLORATORY authority requires explicit '+FLAG)


def config(inputs):
    raw = effective_config(Path.cwd(), ENTRY)
    raw['horizon']['gate_file'] = inputs['horizon_gate']
    from wms.control.looplong import read_boundary
    boundary = read_boundary(inputs['boundary'])
    if boundary.get('coop_credit') is False:
        for key in ('w_individual', 'w_group', 'w_coop', 'w_team'):
            raw['counterfactual'][key] = [0.] * len(raw['counterfactual'][key])
    return raw


def check_identity(raw):
    long = raw['long_credit']
    if (raw['policy']['residual_reference'] != 'ctrl0'
            or raw['trainer']['entrant_learned_mix'] is not True
            or raw['trainer']['future_samples'] != 8
            or raw['trainer']['pessimistic_credit'] is not True
            or raw['trainer']['deterministic_crn'] is not True
            or raw['trainer']['entrant_control'] is not True
            or raw['policy14']['zero_residual_output'] is not True
            or raw['counterfactual']['headroom_starts'] is not True
            or not all(long[k] is True for k in ('enabled', 'ctrl0_reference',
                                                'score_training_horizon', 'require_objective_match'))
            or long['mode'] != 'gain_only' or long['estimator'] != 'antithetic'
            or long['es_directions'] != 2 or long['k16'] is not False):
        raise ValueError('EXPLORATORY candidate-B runtime/config/mix identity mismatch')


def inspect(inputs):
    raw = config(inputs)
    check_identity(raw)
    if sha256(inputs['wm']) != WM_SHA:
        raise ValueError('EXPLORATORY requires exact WM2')
    from wms.control.looplong import read_boundary
    boundary = read_boundary(inputs['boundary'])
    if (boundary['wm']['sha256'] != WM_SHA
            or sha256(boundary['wm']['config']) != sha256(inputs['wm_config'])):
        raise ValueError('EXPLORATORY boundary/WM-config mismatch')
    cct2 = json.loads(safe_path(inputs['cct2']).read_text())
    if cct2.get('wm_sha256') != WM_SHA:
        raise ValueError('EXPLORATORY historical CCT-2 WM mismatch')
    if sha256(raw['long_credit']['cct2_file'].format(wm_sha256=WM_SHA)) != sha256(inputs['cct2']):
        raise ValueError('EXPLORATORY preset CCT-2 identity mismatch')
    if safe_path(inputs['current_policy_driver']).read_bytes().replace(b'\r\n', b'\n') != safe_path(
            'configs/execparity_cav_idm_v1.json').read_bytes():
        raise ValueError('historical execution driver differs beyond CRLF normalization')
    configure_execution('configs/execparity_cav_idm_v1.json')
    from wms.control.execution import verify_parity
    verify_parity(inputs['parity'], residual_reference='ctrl0')
    from wms.control.nativeinit import PANEL_SHA256
    if sha256(inputs['gate_panel']) != PANEL_SHA256:
        raise ValueError('EXPLORATORY frozen gate-panel mismatch')
    return raw


def build(inputs, out, *, exploratory=False):
    require_opt_in(exploratory)
    raw = inspect(inputs)
    doc = dict(schema=SCHEMA, label='EXPLORATORY', policy_authorized=False,
        qualification_passed=False, owner='2026-10-30 19:45 S-PILOTB-EXPLORE',
        waived_evidence=['objective-matched CCT-2 qualification', 'R2 ranking evidence'],
        source_commit=source_commit(Path.cwd()), inputs=inputs,
        input_sha256={k: sha256(v) for k, v in inputs.items()},
        config_sha256=binding(raw), runtime_manifest=manifest(),
        fresh_zero_start=True, training_launched=False)
    for key in ('current_policy', 'boundary', 'pairing_receipt'):
        doc[key+'_sha256'] = doc['input_sha256'][key]
    with safe_path(out).open('x', encoding='utf-8') as stream:
        json.dump(doc, stream, indent=1, allow_nan=False)
        stream.write('\n')
    return doc


def verify(path, *, exploratory=False):
    require_opt_in(exploratory)
    doc = json.loads(safe_path(path).read_text())
    if (doc.get('schema') != SCHEMA or doc.get('label') != 'EXPLORATORY'
            or doc.get('policy_authorized') is not False
            or doc['source_commit'] != source_commit(Path.cwd())):
        raise ValueError('EXPLORATORY source/schema/label mismatch')
    raw = inspect(doc['inputs'])
    if binding(raw) != doc['config_sha256'] or manifest() != doc['runtime_manifest']:
        raise ValueError('EXPLORATORY runtime/config/source mismatch')
    for key, digest in doc['input_sha256'].items():
        if sha256(doc['inputs'][key]) != digest:
            raise ValueError('EXPLORATORY input identity mismatch: '+key)
    return doc


def verify_runtime(doc, raw, *, wm, wm_config, trainer=None):
    """Replacement only for the explicitly exploratory process's ranking evidence gate."""
    if (doc.get('schema') != SCHEMA or wm != WM_SHA
            or wm_config != doc['input_sha256']['wm_config']
            or binding(raw) != doc['config_sha256']):
        raise ValueError('EXPLORATORY launch config/WM identity mismatch')
    check_identity(raw)
    if trainer is not None:
        from dataclasses import asdict
        from wms.control.dreamer import DreamerConfig
        from wms.control.gated300 import LongCreditConfig
        if (asdict(trainer.config) != asdict(DreamerConfig(**raw['trainer']))
                or trainer.policy.residual_reference != 'ctrl0'
                or trainer.gated300['H_long'] != 300
                or trainer.gated300['config'] != asdict(LongCreditConfig(**raw['long_credit']))):
            raise ValueError('EXPLORATORY effective runtime differs from bound preset')
        from tools.distill.runtime import check_metadata
        check_metadata(trainer.start_states.metadata.values())
    for name, digest in doc['runtime_manifest'].items():
        if sha256(name) != digest:
            raise ValueError('EXPLORATORY source identity mismatch: '+name)
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(FLAG, action='store_true', required=True)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    for key in INPUTS:
        b.add_argument('--'+key.replace('_', '-'), required=True)
    b.add_argument('--out', required=True)
    sub.add_parser('verify').add_argument('authority')
    a = p.parse_args()
    if a.command == 'build':
        doc = build({k: str(safe_path(getattr(a, k))) for k in INPUTS}, a.out, exploratory=True)
    else:
        doc = verify(a.authority, exploratory=True)
    print(json.dumps(dict(label='EXPLORATORY', verified=True,
                         source_commit=doc['source_commit'], authority_sha256=canonical(doc))))


if __name__ == '__main__':
    main()
