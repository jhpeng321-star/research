"""Bind a newly accepted WM's measured legacy gate and CCT-1 to a fresh boundary.

This is artifact verification only. CCT-1 computation and beyond-90s credit belong
to S-GATED300; this authority never enables dense credit beyond 90s.
"""
import argparse
import hashlib
import json
import math

from .c4launch_authority import verified
from .start_states import checked_path

SCHEMA = 'c4launch_measured_horizon/1'
LEGACY_THRESHOLD = 0.868421052631579  # unchanged registered 13E-b2 lower bound


def digest(path):
    return hashlib.sha256(checked_path(path).read_bytes()).hexdigest()


def measured_gate(doc, wm_sha256s):
    if doc.get('schema') != SCHEMA or set(wm_sha256s) != {doc.get('wm_sha256')}:
        raise ValueError('measured authority schema / WM mismatch')
    wm, config = doc['wm_sha256'], doc['wm_config_sha256']
    if digest(doc['wm_config']) != config:
        raise ValueError('measured authority WM config changed')
    accepted = verified(doc['acceptance'], doc['acceptance_sha256'])
    if (accepted.get('decision') != 'ACCEPT' or
            accepted.get('inputs', {}).get('candidate_sha256') != wm or
            accepted.get('inputs', {}).get('config_sha256') != config):
        raise ValueError('measured authority requires ACCEPT for exact WM/config')
    threshold = verified(doc['threshold_file'], doc['threshold_sha256'])
    gate = verified(doc['measured_gate'], doc['measured_gate_sha256'])
    if (gate.get('schema') != 'coopw_horizon_gate/1' or gate.get('wm_sha256') != wm or
            gate.get('wm_config_sha256') != config or
            gate.get('threshold') != LEGACY_THRESHOLD or
            threshold.get('threshold') != LEGACY_THRESHOLD or
            gate.get('threshold_source', {}).get('sha256') != doc['threshold_sha256']):
        raise ValueError('measured gate WM/config/registered threshold mismatch')
    if doc.get('trusted_horizon_s') != 90:
        raise ValueError('measured dense authority is exactly90s; extension belongs to S-GATED300')
    for h in (10, 30, 60, 90):
        row = gate.get('horizons', {}).get(str(h), {})
        sign = row.get('sign_agreement')
        if (type(sign) not in (int, float) or not math.isfinite(sign) or
                not LEGACY_THRESHOLD <= sign <= 1 or
                row.get('verdict') != ('BASE' if h == 10 else 'PASS')):
            raise ValueError('measured legacy short/90s gate failed')
    cct = verified(doc['cct1_file'], doc['cct1_sha256'])
    if (cct.get('schema') != 'gated300_cct1/1' or cct.get('wm_sha256') != wm or
            cct.get('wm_config_sha256') != config or cct.get('panel') != 'pairs300' or
            cct.get('policy_authorized') is not True or
            cct.get('source_sha256') != doc['measured_gate_sha256'] or
            cct.get('legacy_threshold') != gate['threshold'] or
            cct.get('legacy_horizons') != gate['horizons'] or
            type(cct.get('H_C')) is not int or cct['H_C'] not in (0, 90, 120, 180, 300)):
        raise ValueError('CCT-1 must bind the same measured pairs300 WM/config/gate')
    for name in ('entrant_control', 'deterministic_crn', 'samples'):
        if cct.get(name) != gate.get(name):
            raise ValueError('CCT-1 / measured gate inference mismatch')
    if gate.get('entrant_control') is not doc.get('entrant_control'):
        raise ValueError('measured gate entrant setting mismatch')
    # H_C is reported, not interpreted as dense horizon authority here.
    return gate


def build(args):
    template = json.loads(checked_path(args.boundary_template).read_text(encoding='utf-8'))
    if (template.get('schema') != 'looplong_boundary/1' or
            template.get('horizon_s') != 90 or template.get('imagination_horizon_s', 90) != 90):
        raise ValueError('supply the existing90s boundary; horizon changes are not an auditfix')
    wm, config = digest(args.wm_ckpt), digest(args.wm_config)
    gate = json.loads(checked_path(args.gate).read_text(encoding='utf-8'))
    authority = dict(schema=SCHEMA, wm_sha256=wm, wm_config_sha256=config,
        wm_config=str(checked_path(args.wm_config)), trusted_horizon_s=90,
        entrant_control=gate.get('entrant_control'))
    for key, path in (('acceptance', args.acceptance), ('threshold_file', args.threshold),
                      ('measured_gate', args.gate), ('cct1_file', args.cct1)):
        authority[key] = str(checked_path(path))
        hash_key = {'threshold_file': 'threshold_sha256', 'cct1_file': 'cct1_sha256'}.get(key, key + '_sha256')
        authority[hash_key] = digest(path)
    measured_gate(authority, [wm])
    # Refuse output collisions before creating either immutable artifact.
    authority_path, boundary_path = checked_path(args.authority_out), checked_path(args.out)
    if authority_path == boundary_path or authority_path.exists() or boundary_path.exists():
        raise ValueError('authority/boundary outputs must be distinct NEW files')
    text = json.dumps(authority, indent=2, allow_nan=False) + '\n'
    boundary = {**template, 'wm': {'checkpoint': str(checked_path(args.wm_ckpt)),
        'config': str(checked_path(args.wm_config)), 'sha256': wm},
        'gate_file': str(authority_path), 'gate_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'threshold_sha256': authority['threshold_sha256'],
        'acceptance': authority['acceptance'], 'acceptance_sha256': authority['acceptance_sha256'],
        'cct1_file': authority['cct1_file'], 'cct1_sha256': authority['cct1_sha256']}
    with authority_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    from .looplong import read_boundary
    with boundary_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(boundary, stream, indent=2, allow_nan=False)
    read_boundary(boundary_path, swap=True)
    return boundary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('boundary-template', 'wm-ckpt', 'wm-config', 'gate', 'cct1',
                 'acceptance', 'threshold', 'authority-out', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
