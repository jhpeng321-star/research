"""Explicit MAIN inherited horizon authority, distinct from a measured gate."""
import hashlib
import json

from .start_states import checked_path

WM3 = '41e18f09b6060a1f4e4039f02655fcbcd497e32a360cffb71be716fffeaadc84'
WM2_GATE = 'be0297ab09a988c5ced1a007432effdea36dfb80be4e86b33ac99605824fd152'
AUTHORITY = '2b0c43a7f8157e8402df98932359582e6138d054fefc965b566f5ddcfb83598e'


def verified(path, digest):
    raw = checked_path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('C4 launch authority/input hash mismatch')
    return json.loads(raw)


def inherited_gate(doc, wm_sha256s):
    if doc.get('schema') != 'c4launch_inherited_horizon/1':
        raise ValueError('not a C4 launch horizon authority')
    authority = verified(doc['authority'], AUTHORITY)
    if (authority['authority'] != 'MAIN2026-10-28T15:00' or set(wm_sha256s) != {WM3}
            or doc['wm_sha256'] != WM3 or authority['wm_sha256'] != WM3
            or doc['trusted_horizon_s'] != authority['trusted_horizon_s']
            or doc['trusted_horizon_s'] != 90 or not doc['entrant_control']):
        raise ValueError('C4 launch MAIN/WM/horizon/entrant binding mismatch')
    accepted = verified(doc['acceptance'], doc['acceptance_sha256'])
    if accepted['decision'] != 'ACCEPT' or accepted['inputs']['candidate_sha256'] != WM3:
        raise ValueError('C4 launch requires accepted WM3')
    if accepted['inputs']['config_sha256'] != hashlib.sha256(checked_path(doc['wm_config']).read_bytes()).hexdigest():
        raise ValueError('C4 launch WM config changed')
    gate = verified(doc['inherited_gate'], WM2_GATE)
    if gate['horizons']['90']['verdict'] != 'PASS':
        raise ValueError('inherited WM2 horizon did not pass')
    return gate
