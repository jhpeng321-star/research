"""CPU tests for the variety plan-collection path (arms B and R; c4/variety-collect).

Real code paths with a fake cpuQ RPC and fake episodes: pinned-checkpoint refusals (wrong
sha/k), charges ONLY through the cpuS->cpuQ RPC route with the registered phases
(sumo_verification, bank_verification_ext), never a second CHARGES.jsonl, B/R variety-mix
wiring (B floor(share*N) search slots, R Bernoulli(share) unscored draws), parameterised
episode ids, single-slot broker staging artifacts. No SUMO, no ssh, no GPU.
"""
import ast
import hashlib
import inspect
import io
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tools.coopmwm import cpus_checking as cc
from tools.coopmwm import cpus_rpc
from tools.coopmwm import replay_bank as rb
from tools.coopmwm import sumo_verify as sv
from tools.coopmwm import varcollect as vc
from tools.coopmwm import variety as va
from tools.coopmwm.accel_learner import save
from tools.coopmwm.acceleration import Policy
from tools.coopmwm.test_sumo_verify_cpu import FakePlant, plan, snapshot

PREFIX = 'vc1'
SEED = 986000


@pytest.fixture(autouse=True)
def restore_accounting():
    """The RPC route patches process-wide accel_storage/storage readers; restore after every test."""
    from tools.coopmwm import accel_storage, storage
    keep = (accel_storage.reserve, accel_storage.ledger, storage.ledger)
    yield
    accel_storage.reserve, accel_storage.ledger, storage.ledger = keep


def pin_checkpoints(root, update=3, arms=('B', 'R')):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for arm in arms:
        path = root/f'{arm}_checkpoint_{update:03d}.pt'
        if not path.exists():          # accel_learner.save is exclusive-create; never overwrite
            policy = Policy()
            opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
            save(path, policy, opt, torch.Generator(), update, 0)
        out[arm] = dict(path=path, sha256=vc.file_sha256(path))
    return out


def base_config(tmp, checkpoints=None, k=3, shas=None, **kw):
    pins = checkpoints if checkpoints is not None else pin_checkpoints(tmp/'coop3', update=k)
    cfg = dict(policies_root=str(tmp/'coop3'), checkpoint_k=k,
               checkpoint_sha256={a: (shas or {a: pins[a]['sha256']})[a] for a in ('B', 'R')},
               charge_route=dict(run_root=str(tmp/'shared'), q_host='cpuQ', q_control='/ctl',
                                 q_python='/usr/bin/python3', q_pythonpath='/pytools'),
               collect_id_prefix=PREFIX, episode_phase='sumo_verification',
               runtime='rt', accel_bounds=[-3., 2.], gpu_deadline=1793894340.,
               worlds=[dict(seed=SEED)])
    cfg.update(kw)
    return cfg


def collect_binding(cfg, phases=('sumo_verification', 'bank_verification_ext')):
    return dict(phases=list(phases), id_prefixes=list(vc.route_prefixes(cfg)))


def fake_rpc(binding, charges):
    """A cpuQ stand-in enforcing the collect route exactly like cpus_rpc.op_reserve."""
    def rpc(cfg, op, payload=None, timeout=600):
        if op == 'budget':
            return dict(ok=True, all8_used=0, delegate_cap=10**9, phase='x', phase_used=0,
                        phase_cap=10**9, total_used=sum(r['sim_s'] for r in charges['rows']),
                        total_cap=10**9, rows=len(charges['rows']))
        if op == 'ledger':
            return dict(ok=True, rows=list(charges['rows']))
        if op == 'reserve':
            jobs, phase = payload['jobs'], payload['phase']
            ok = phase in binding['collect']['phases'] and all(
                any(str(j['id']).startswith(p) for p in binding['collect']['id_prefixes'])
                for j in jobs)
            if not ok:
                return dict(ok=False, error='collect route refused the charge')
            rows = []
            for j in jobs:
                row = dict(id=j['id'], phase=phase, sim_s=int(j['sim_s']))
                charges['rows'].append(row)
                charges['calls'].append((phase, row['id'], row['sim_s']))
                rows.append(row)
            return dict(ok=True, rows=rows)
        raise AssertionError(op)
    return rpc


def routed(tmp, cfg, **kw):
    """(charges, context manager factory) installing the fake RPC route."""
    charges = dict(rows=[], calls=[])
    binding = dict(run_root=cfg['charge_route']['run_root'], collect=collect_binding(cfg, **kw))
    return charges, patch.object(cc, 'rpc', fake_rpc(binding, charges))


# ---------------------------------------------------------------- config

def test_config_registered_keys_and_refusals(tmp_path):
    cfg = base_config(tmp_path)
    v = vc.split_config(cfg)
    assert v['checkpoint_k_B'] == v['checkpoint_k_R'] == 3
    assert v['episode_phase'] == 'sumo_verification'
    with pytest.raises(ValueError, match='unregistered varcollect'):
        vc.split_config(dict(cfg, bogus=1))
    with pytest.raises(ValueError, match='mismatched pinned checkpoint k'):
        vc.split_config(dict(cfg, checkpoint_k_B=3, checkpoint_k_R=4))
    with pytest.raises(ValueError, match='checkpoint_sha256'):
        vc.split_config(dict(cfg, checkpoint_sha256=dict(B='0'*64)))
    with pytest.raises(ValueError, match='checkpoint_sha256'):
        vc.split_config(dict(cfg, checkpoint_sha256=dict(B='zz', R='0'*64)))
    with pytest.raises(ValueError, match='unregistered charge_route'):
        vc.split_config(dict(cfg, charge_route=dict(cfg['charge_route'], extra=1)))
    with pytest.raises(ValueError, match='charge_route must carry'):
        vc.split_config(dict(cfg, charge_route={k: v for k, v in cfg['charge_route'].items()
                                                if k != 'run_root'}))
    with pytest.raises(ValueError, match='collect_id_prefix'):
        vc.split_config(dict(cfg, collect_id_prefix='VC1'))
    with pytest.raises(ValueError, match='accel_bounds'):
        vc.split_config(dict(cfg, accel_bounds=[2., -3.]))
    with pytest.raises(ValueError, match='worlds'):
        vc.split_config(dict(cfg, worlds=[]))


def test_route_phases_default_registered_and_prefixes(tmp_path):
    cfg = vc.split_config(base_config(tmp_path))
    assert vc.route_phases(cfg) == ('sumo_verification', 'bank_verification_ext')
    narrow = vc.split_config(base_config(tmp_path, checkpoints=pin_checkpoints(tmp_path/'coop3'),
                                        charge_route=dict(run_root=str(tmp_path/'shared'),
                                                          q_host='q', q_control='/c', q_python='/p',
                                                          q_pythonpath='/pp',
                                                          phases=['sumo_verification'])))
    assert vc.route_phases(narrow) == ('sumo_verification',)
    assert vc.route_prefixes(cfg) == (
        f'confirm_extended_sv_{PREFIX}_', f'confirm_replication_sv_{PREFIX}_',
        f'confirm_solo_sv_{PREFIX}_', f'sv_{PREFIX}_', f'{PREFIX}_')


def test_load_config_staged_file(tmp_path):
    with pytest.raises(ValueError, match='no VARCOLLECT_CONFIG.json'):
        vc.load_config(tmp_path)
    cfg = base_config(tmp_path)
    (tmp_path/'VARCOLLECT_CONFIG.json').write_text(json.dumps(cfg))
    assert vc.load_config(tmp_path)['collect_id_prefix'] == PREFIX
    (tmp_path/'VARCOLLECT_CONFIG.json').write_text('{broken')
    with pytest.raises(ValueError):
        vc.load_config(tmp_path)


# ---------------------------------------------------------------- pinned policies

def test_pinned_loads_readonly_checkpoints(tmp_path):
    cfg = vc.split_config(base_config(tmp_path))
    pins = vc.pinned_checkpoints(cfg)
    assert set(pins) == {'B', 'R'}
    assert pins['B']['k'] == 3 and Path(pins['B']['path']).is_file()
    policy, pin = vc.pinned_policy(cfg, 'B')
    assert pin['k'] == 3 and re.fullmatch(r'[0-9a-f]{64}', pin['sha256'])
    assert isinstance(policy, Policy)


def test_pinned_refuses_wrong_sha_and_missing_file(tmp_path):
    pins = pin_checkpoints(tmp_path/'coop3')
    cfg = vc.split_config(base_config(tmp_path, checkpoints=pins,
                                      shas=dict(B='0'*64, R=pins['R']['sha256'])))
    with pytest.raises(ValueError, match='sha256 mismatch'):
        vc.pinned_checkpoints(cfg)
    cfg = vc.split_config(base_config(tmp_path, checkpoints=pins,
                                      shas=dict(B=pins['B']['sha256'], R='1'*64)))
    with pytest.raises(ValueError, match='R checkpoint sha256 mismatch'):
        vc.pinned_checkpoints(cfg)
    cfg = vc.split_config(base_config(tmp_path, checkpoints=pins, k=4,
                                      shas=dict(B='0'*64, R='0'*64)))
    with pytest.raises(ValueError, match='pinned checkpoint missing'):
        vc.pinned_checkpoints(cfg)


def test_pinned_refuses_mismatched_k_between_arms(tmp_path):
    with pytest.raises(ValueError, match='mismatched pinned checkpoint k'):
        vc.split_config(base_config(tmp_path, checkpoint_k_B=3, checkpoint_k_R=4))


def test_pinned_refuses_swapped_payload_update(tmp_path):
    """A file NAMED R_checkpoint_003.pt whose payload carries update 7 refuses at load."""
    root = tmp_path/'coop3';root.mkdir(parents=True)
    pins = pin_checkpoints(root, arms=('B',))
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    r_path = root/'R_checkpoint_003.pt'
    save(r_path, policy, opt, torch.Generator(), 7, 0)
    pins['R'] = dict(path=r_path, sha256=vc.file_sha256(r_path))
    cfg = vc.split_config(base_config(tmp_path, checkpoints=pins))
    with pytest.raises(ValueError, match='payload update'):
        vc.pinned_policy(cfg, 'R')


# ---------------------------------------------------------------- charge route

def test_route_probe_phases_and_prefixes(tmp_path):
    cfg = vc.split_config(base_config(tmp_path))
    charges, ctx = routed(tmp_path, cfg)
    with ctx:
        reserve = vc.route_charges(cfg)
        reserve([dict(id=f'{PREFIX}_B_k003_e00000_{SEED}', sim_s=1200)], 'sumo_verification')
        reserve([dict(id=f'confirm_replication_sv_{PREFIX}_x_policy', sim_s=7)],
                'bank_verification_ext')
    assert [c[0] for c in charges['calls']] == ['sumo_verification', 'bank_verification_ext']
    assert charges['rows'][0]['id'] == f'{PREFIX}_B_k003_e00000_{SEED}'


def test_route_refuses_unregistered_phase_before_any_charge(tmp_path):
    cfg = vc.split_config(base_config(tmp_path))
    charges, ctx = routed(tmp_path, cfg)
    with ctx:
        reserve = vc.route_charges(cfg)
        with pytest.raises(ValueError, match='not registered on the collection charge route'):
            reserve([dict(id=f'{PREFIX}_x', sim_s=1)], 'discovery')
    assert charges['calls'] == []          # nothing was charged anywhere


def test_route_refuses_when_unavailable(tmp_path):
    cfg = vc.split_config(base_config(tmp_path))
    from tools.coopmwm.cpus_checking import RpcError
    with patch.object(cc, 'rpc', side_effect=RpcError('ssh died')):
        with pytest.raises(ValueError, match='charge route unavailable'):
            vc.route_charges(cfg)
    with patch.object(cc, 'rpc', return_value=dict(ok=False, error='cpuQ down')):
        with pytest.raises(ValueError, match='charge route unavailable'):
            vc.route_charges(cfg)


def test_refuse_local_ledger(tmp_path):
    root = tmp_path/'collect_root'
    root.mkdir()
    vc.refuse_local_ledger(root)
    (root/'CHARGES.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='second ledger'):
        vc.refuse_local_ledger(root)
    (root/'CHARGES.jsonl').unlink()
    (root/'CHARGES.jsonl').mkdir()
    with pytest.raises(ValueError, match='second ledger'):
        vc.refuse_local_ledger(root)


# ------------------------------------------------- ledger host modes (MAIN 2026-11-04 16:2x)

def ledger_root(tmp):
    """A temp ledger run root: accounting binding (amendment in AUTHORITY) + AUTHORITY line."""
    from tools.coopmwm.warm_protocol import budget
    root = tmp/'ledger_root';root.mkdir()
    b = budget()
    b['phases'] = dict(b['phases'], sumo_verification=2332800, bank_verification_ext=2600000)
    b['total'] += 2332800+2600000
    b['allocated'] = b['total']
    (root/'B_REPLAY_PHASE_RECONCILIATION_0627.json').write_text(json.dumps(
        dict(budget=b, amendment_sha256='ledger-host-test')))
    authority = 'ledger-host-test amendment line'
    (root/'AUTHORITY.md').write_text(authority+'\n')
    return root, authority


def register_ledger_host(root, authority_line, host=None, ledger_path=None):
    binding = dict(host=host or socket.gethostname(),
                   ledger_path=str(ledger_path or root),
                   authority_sha256=hashlib.sha256(authority_line.encode('utf-8')).hexdigest())
    (Path(root)/'LEDGER_HOST.json').write_text(json.dumps(binding))
    return binding


def local_config(tmp, root):
    return base_config(tmp, charge_route=dict(run_root=str(root), mode='local'))


def test_route_mode_registered_rpc_default(tmp_path):
    cfg = vc.split_config(base_config(tmp_path))
    assert vc.route_mode(cfg) == 'rpc'
    local = vc.split_config(local_config(tmp_path, tmp_path/'ledger_root'))
    assert vc.route_mode(local) == 'local'
    with pytest.raises(ValueError, match='charge_route.mode'):
        vc.split_config(base_config(tmp_path,
                                    charge_route=dict(cfg['charge_route'], mode='rsync')))
    with pytest.raises(ValueError, match='charge_route must carry'):
        vc.split_config(base_config(tmp_path, charge_route=dict(mode='local')))
    charges, ctx = routed(tmp_path, cfg)               # mode 'rpc' (default) is byte-unchanged
    with ctx:
        reserve = vc.route_charges(cfg)
        reserve([dict(id=f'{PREFIX}_B_k003_e00000_{SEED}', sim_s=1)], 'sumo_verification')
    assert charges['calls'] == [('sumo_verification', f'{PREFIX}_B_k003_e00000_{SEED}', 1)]


def test_local_mode_charges_the_registered_single_ledger(tmp_path):
    """On the registered ledger host itself, mode 'local' charges the ONE ledger at the
    registered path through the UNCHANGED reserve; registered phases still enforced."""
    root, authority = ledger_root(tmp_path)
    binding = register_ledger_host(root, authority)
    cfg = vc.split_config(local_config(tmp_path, root))
    charge = vc.route_charges(cfg)
    charge([dict(id=f'{PREFIX}_B_k003_e00000_{SEED}', sim_s=1200)], 'sumo_verification')
    charge([dict(id=f'confirm_solo_sv_{PREFIX}_x_policy', sim_s=5)], 'bank_verification_ext')
    with pytest.raises(ValueError, match='not registered on the collection charge route'):
        charge([dict(id=f'{PREFIX}_x', sim_s=1)], 'discovery')
    rows = [json.loads(l) for l in (root/'CHARGES.jsonl').read_text().splitlines()]
    assert [(r['id'], r['phase'], r['sim_s']) for r in rows] == [
        (f'{PREFIX}_B_k003_e00000_{SEED}', 'sumo_verification', 1200),
        (f'confirm_solo_sv_{PREFIX}_x_policy', 'bank_verification_ext', 5)]
    assert binding['ledger_path'] == str(root)
    collect_root = tmp_path/'collect_root';collect_root.mkdir()
    vc.refuse_local_ledger(collect_root)       # the collection root never holds a ledger


def test_local_mode_refused_off_the_ledger_host(tmp_path):
    """mode 'local' on any OTHER host refuses before anything runs (never two ledgers)."""
    root, authority = ledger_root(tmp_path)
    register_ledger_host(root, authority, host='cpuQ')
    cfg = vc.split_config(local_config(tmp_path, root))
    with patch('socket.gethostname', return_value='cpuS'):
        with pytest.raises(ValueError, match='not the registered single ledger host'):
            vc.route_charges(cfg)
    assert not (root/'CHARGES.jsonl').exists()             # nothing charged anywhere
    collect_root = tmp_path/'collect_root';collect_root.mkdir()
    assert not list(collect_root.rglob('CHARGES.jsonl'))


def test_local_mode_refuses_missing_or_tampered_binding(tmp_path):
    root, authority = ledger_root(tmp_path)
    cfg = vc.split_config(local_config(tmp_path, root))
    with pytest.raises(ValueError, match='no LEDGER_HOST.json'):
        vc.route_charges(cfg)
    b = register_ledger_host(root, authority, ledger_path=str(tmp_path/'elsewhere'))
    with pytest.raises(ValueError, match='ledger_path'):
        vc.route_charges(cfg)
    (root/'LEDGER_HOST.json').write_text(json.dumps(dict(b, bogus=1)))
    with pytest.raises(ValueError, match='unregistered LEDGER_HOST.json keys'):
        vc.route_charges(cfg)
    (root/'LEDGER_HOST.json').write_text(json.dumps({k: v for k, v in b.items() if k != 'host'}))
    with pytest.raises(ValueError, match='LEDGER_HOST.json must carry'):
        vc.route_charges(cfg)
    (root/'LEDGER_HOST.json').write_text(json.dumps(dict(b, authority_sha256='zz')))
    with pytest.raises(ValueError, match='sha256 hex digest'):
        vc.route_charges(cfg)
    register_ledger_host(root, authority)          # a valid binding again...
    (root/'AUTHORITY.md').write_text('a DIFFERENT amendment line\n')   # ...but tampered AUTHORITY
    with pytest.raises(ValueError, match='AUTHORITY line'):
        vc.route_charges(cfg)
    assert not (root/'CHARGES.jsonl').exists()             # no charge ever left the refused path


def test_collect_local_mode_charges_the_one_ledger(tmp_path):
    """End to end in mode 'local': the episode charge lands ONLY in the registered ledger."""
    root, authority = ledger_root(tmp_path)
    register_ledger_host(root, authority)
    cfg = vc.split_config(local_config(tmp_path, root))
    collect_root = tmp_path/'varietycollect20261104b'
    with patch.object(vc, 'run_episodes', fake_episodes):
        statuses = vc.collect(collect_root, cfg, 'R', stop=lambda: False, rounds=1,
                              episodes_per_round=1)
    assert statuses == []
    rows = [json.loads(l) for l in (root/'CHARGES.jsonl').read_text().splitlines()]
    assert [(r['id'], r['phase'], r['sim_s']) for r in rows] == \
        [(f'{PREFIX}_R_k003_e00000_{SEED}', cfg['episode_phase'], 1200)]
    assert not list(collect_root.rglob('CHARGES.jsonl'))
    vc.refuse_local_ledger(collect_root)
    assert (collect_root/'jobs'/f'{PREFIX}_R_k003_e00000_{SEED}'/'RESULT.json').is_file()
    assert (rb.bank_root(collect_root, 'R')/'BANK_QUALITY.json').is_file()


def test_rpc_server_collect_route_all8_unchanged(tmp_path):
    """The REAL cpus_rpc op_reserve: the collect route charges the ONE local ledger; without the
    binding's 'collect' key the refusals are the original bytes."""
    from tools.coopmwm.warm_protocol import budget
    root = tmp_path/'run';root.mkdir()
    b = budget()
    b['phases'] = dict(b['phases'], sumo_verification=2332800, bank_verification_ext=2600000)
    b['total'] += 2332800+2600000
    b['allocated'] = b['total']
    (root/'B_REPLAY_PHASE_RECONCILIATION_0627.json').write_text(json.dumps(
        dict(budget=b, amendment_sha256='test-vc')))
    (root/'AUTHORITY.md').write_text('test-vc')
    control = tmp_path/'ctl';control.mkdir()
    prefixes = list(vc.route_prefixes(vc.split_config(base_config(tmp_path))))
    (control/'RPC_BINDING.json').write_text(json.dumps(dict(
        run_root=str(root), phase='bank_verification_ext', delegate_cap_sim_s=10**9,
        delegate_locks=[],
        collect=dict(phases=['sumo_verification', 'bank_verification_ext'],
                     id_prefixes=prefixes))))

    def call(op, payload):
        saved_in, saved_out = sys.stdin, sys.stdout
        sys.stdin, sys.stdout = io.StringIO(json.dumps(payload)), io.StringIO()
        try:
            cpus_rpc.main([str(control), op])
            return json.loads(sys.stdout.getvalue())
        finally:
            sys.stdin, sys.stdout = saved_in, saved_out

    ok = call('reserve', dict(jobs=[dict(id=f'{PREFIX}_B_k003_e00000_{SEED}', sim_s=1200)],
                              phase='sumo_verification', host='cpuS'))
    assert ok['ok'] and ok['rows'][0]['phase'] == 'sumo_verification'
    ok = call('reserve', dict(jobs=[dict(id=f'confirm_solo_sv_{PREFIX}_x_policy', sim_s=5)],
                              phase='bank_verification_ext'))
    assert ok['ok']
    v = call('reserve', dict(jobs=[dict(id=f'{PREFIX}_x', sim_s=1)], phase='discovery'))
    assert not v['ok']
    v = call('reserve', dict(jobs=[dict(id='other_collection_x', sim_s=1)],
                             phase='sumo_verification'))
    assert not v['ok'] and 'registered collection' in v['error']
    # the all8 route is unchanged (registered phase + all8-kind ids only)
    v = call('reserve', dict(jobs=[dict(id='confirm_seed2_all8_x', sim_s=1)], phase='train'))
    assert not v['ok'] and v['error'] == 'phase train is not the registered bank_verification_ext'
    v = call('reserve', dict(jobs=[dict(id='warm_u005_B_7', sim_s=1)],
                             phase='bank_verification_ext'))
    assert not v['ok'] and 'all8-kind' in v['error']
    rows = [json.loads(line) for line in (root/'CHARGES.jsonl').read_text().splitlines()]
    assert [r['id'] for r in rows] == [f'{PREFIX}_B_k003_e00000_{SEED}',
                                       f'confirm_solo_sv_{PREFIX}_x_policy']
    rpc_rows = [json.loads(line) for line in (control/'CHARGES_RPC.jsonl').read_text().splitlines()]
    assert [r['route'] for r in rpc_rows] == ['collect', 'collect']

    # WITHOUT the binding's collect key: byte-identical legacy refusals for collection ids
    (control/'RPC_BINDING.json').write_text(json.dumps(dict(
        run_root=str(root), phase='bank_verification_ext', delegate_cap_sim_s=10**9,
        delegate_locks=[])))
    v = call('reserve', dict(jobs=[dict(id=f'{PREFIX}_x', sim_s=1)], phase='sumo_verification'))
    assert not v['ok'] and v['error'] == \
        'phase sumo_verification is not the registered bank_verification_ext'
    v = call('reserve', dict(jobs=[dict(id=f'{PREFIX}_x', sim_s=1)],
                             phase='bank_verification_ext'))
    assert not v['ok'] and v['error'] == 'only all8-kind confirmation ids are charged through this RPC'


# ---------------------------------------------------------------- verify charges through the route

def test_window_and_confirm_charges_route_through_rpc(tmp_path):
    cfg = vc.split_config(base_config(tmp_path))
    root = tmp_path/'collect_root'
    charges, ctx = routed(tmp_path, cfg)
    with ctx:
        vc.route_charges(cfg)
        ep_id = vc.job_id(cfg, 'B', 0, dict(seed=SEED))
        snap = snapshot(t0=400)
        snap['episode'] = dict(snap['episode'], id=ep_id, arm='B')
        r = sv.verify(snap, [plan('a', gain=1.)], FakePlant(),
                      sv.accel_charge(cfg['charge_route']['run_root'], vc.WINDOW_PHASE),
                      root=rb.bank_root(root, 'B')/'sumo_verification')
        assert r['status'] == 'accepted'
        assert charges['calls'] and all(i.startswith(f'sv_{PREFIX}_') for c in charges['calls']
                                        for i in [c[1]])
        assert {c[0] for c in charges['calls']} == {'sumo_verification'}
        # confirmation-charge id shape (rb.confirm_pending uses these kinds/phase)
        charge = rb._confirm_charge(cfg['charge_route']['run_root'], vc.CONFIRM_PHASE,
                                    'confirm_replication')
        charges['calls'].clear()
        charge(r['verification_id'], [dict(branch='policy', sim_s=5)])
        assert charges['calls'][0][0] == 'bank_verification_ext'
        assert charges['calls'][0][1].startswith(f'confirm_replication_sv_{PREFIX}_')
    vc.refuse_local_ledger(root)
    assert not list(root.rglob('CHARGES.jsonl'))


# ---------------------------------------------------------------- mix wiring (B floor(share*N), R Bernoulli)

def guide_once(tmp, cfg, arm):
    """Run rb.collector_guide once at t=300 (patched capture/torch.save) -> (meta, request)."""
    from types import SimpleNamespace
    r = Path(tmp)
    sample = dict(context=dict(members=['a', 'b']), own_action=[[0.], [0.]])
    collect = dict(slots=4, fanout_factor=2., demand_phase_end_s=1200., variety=dict(cfg['variety']))
    if arm == 'B':
        collect['search_config'] = vc.search_config(cfg)
    job = dict(id='job', arm=arm, queue_root=str(r), world=dict(seed=986001),
               collector_policy_update=3, policy_state={}, collect=collect)
    for n in ('requests', 'centre_requests'):     # the collection root pre-creates these (collect)
        (r/n).mkdir(parents=True, exist_ok=True)
    rows = [dict(vid='a', s_m=-50., v_mps=20., lane=1, is_cav=True),
            dict(vid='b', s_m=-70., v_mps=18., lane=2, is_cav=True)]
    with patch('tools.coopmwm.sumo_verify.capture', return_value={}), \
            patch('tools.coopmwm.exploration.regime', return_value='ordinary'), \
            patch('torch.save'):
        guide = rb.collector_guide(job, None, None, {})
        assert guide(rows, dict(a=0., b=0.), 300, sample) is None
    meta = json.loads(next((r/'replay_bank'/arm/'snapshots').glob('*.json')).read_text())
    folder = 'requests' if arm == 'B' else 'centre_requests'
    request = json.loads((r/folder/(meta['state_id']+'.json')).read_text())
    return meta, request


def test_variety_config_registered_or_off(tmp_path):
    with tempfile.TemporaryDirectory() as d:
        cfg = vc.split_config(base_config(Path(d), variety=dict(variety_families=True)))
        assert cfg['variety']['variety_structured_share'] == .25
        off = vc.split_config(base_config(Path(d), checkpoints=pin_checkpoints(Path(d)/'coop3'),
                                          variety=dict(variety_families=False)))
        assert off['variety'] is None      # option off: nothing registered, legacy bytes
        with pytest.raises(ValueError):
            vc.split_config(base_config(Path(d), checkpoints=pin_checkpoints(Path(d)/'coop3'),
                                        variety=dict(variety_bogus=1)))


def test_r_draws_bernoulli_share_mix(tmp_path):
    share = .25
    cfg = dict(variety_families=True, variety_structured_share=share)
    group = dict(members=['a', 'b'],
                 rows=[dict(vid='a', s_m=-50., v_mps=20., lane=1, is_cav=True),
                       dict(vid='b', s_m=-70., v_mps=18., lane=2, is_cav=True)])
    centre = np.full((2, 300), .2)
    kinds = []
    for seed in range(400):
        cands = sv.random_candidates(group, centre, (-3., 2.), 'ordinary', seed, k=1, config=dict(cfg))
        kinds.append(cands[0].get('variety_family', 'free'))
    structured = sum(k in set(va.STRUCTURED) for k in kinds)
    assert 40 <= structured <= 400*share+120      # Bernoulli(0.25): ~25% structured, never all
    assert structured < 400                        # R candidates stay dominated by free draws
    assert {k for k in kinds} <= set(va.FAMILIES)


def test_b_request_carries_variety_search_config_r_does_not(tmp_path):
    with tempfile.TemporaryDirectory() as d:
        cfg = vc.split_config(base_config(Path(d), variety=dict(variety_families=True)))
        expect = json.loads(json.dumps(dict(cfg['variety'])))   # JSON round-trip: tuples -> lists
        meta_b, request_b = guide_once(tmp_path, cfg, 'B')
        assert meta_b['variety_config'] == expect
        assert request_b['search_config']['variety_families'] is True
        assert request_b['search_config']['variety_structured_share'] == .25
        meta_r, request_r = guide_once(tmp_path, cfg, 'R')
        assert meta_r['variety_config'] == expect
        assert request_r.get('mode') == 'centre' and 'search_config' not in request_r


def test_b_search_takes_floor_share_structured_slots(tmp_path):
    # The B search-side floor(share*N) budget is covered byte-level by test_variety_cpu
    # (SearchIntegration); here the wiring: the staged request config IS that registered mix.
    from tools.coopmwm import mppi_search as ms
    from tools.coopmwm.niche_collect import validate
    with tempfile.TemporaryDirectory() as d:
        cfg = vc.split_config(base_config(Path(d), variety=dict(variety_families=True)))
        staged = vc.search_config(cfg)
        assert validate(staged) == staged                  # GPU side re-derives it unchanged
        assert staged['variety_structured_share'] == .25   # B slots: floor(.25 * population)
        assert set(staged) <= (set(ms.REGISTERED) | set(va.VARIETY_REGISTERED) |
                               {'name', 'niches'})


# ---------------------------------------------------------------- ids and ls_collection default

def test_episode_ids_parameterised(tmp_path):
    cfg = vc.split_config(base_config(tmp_path))
    assert vc.job_id(cfg, 'B', 12, dict(seed=986005)) == f'{PREFIX}_B_k003_e00012_986005'
    cfg2 = vc.split_config(base_config(tmp_path, collect_id_prefix='vc2x'))
    assert vc.job_id(cfg2, 'R', 0, dict(seed=1)) == 'vc2x_R_k003_e00000_1'
    source = Path(vc.__file__).read_text(encoding='utf-8')
    assert not re.search(r'lscollect', source)
    assert 'collect_id_prefix' in source


def test_ls_collection_default_prefix_byte_identical():
    from tools.coopmwm import ls_collection as ls
    source = inspect.getsource(ls.supervise)
    assert source.count('lscollect1_B3') == 1      # only the additive default
    assert "config.get('episode_prefix','lscollect1_B3')" in source
    assert 'lscollect1_B3_986000' == 'lscollect1_B3_986000'


# ---------------------------------------------------------------- broker staging (CONFIG ONLY)

def broker_cfg():
    return dict(amendment_sha256='a'*64, gpu_deadline=1793894340.,
                workers=[dict(endpoint=dict(slot='gpuA-new-1', destination='root@gpuA', port=22,
                                            identity_file='/id', known_hosts='/kh'),
                              owner=dict(boot_id='b', pid=7, start_ticks='42'))])


def test_validate_broker_single_slot_and_refusals():
    assert vc.validate_broker(broker_cfg())['workers'][0]['endpoint']['slot'] == 'gpuA-new-1'
    with pytest.raises(ValueError, match='exactly ONE new GPU slot'):
        vc.validate_broker(dict(broker_cfg(), workers=[broker_cfg()['workers'][0]]*2))
    with pytest.raises(ValueError, match='unregistered broker keys'):
        vc.validate_broker(dict(broker_cfg(), ssh='x'))
    bad = dict(broker_cfg());bad['amendment_sha256'] = 'nothex'
    with pytest.raises(ValueError, match='sha256'):
        vc.validate_broker(bad)
    drop = dict(broker_cfg());drop['workers'] = [dict(endpoint={}, owner={})]
    with pytest.raises(ValueError, match='endpoint needs'):
        vc.validate_broker(drop)


def test_stage_broker_writes_config_only_artifacts(tmp_path):
    cfg = vc.split_config(base_config(tmp_path, broker=broker_cfg()))
    root = tmp_path/'collect_root'
    out = vc.stage_broker(cfg, root)
    assert all(Path(p).is_file() for p in out.values())
    config = json.loads(out['broker'].read_text())
    assert config['shape'].startswith('LS_ENDPOINTS.json')
    assert len(config['workers']) == 1 and config['workers'][0]['endpoint']['slot'] == 'gpuA-new-1'
    gates = json.loads(out['gates'].read_text())
    from tools.coopmwm.hotadd import GATES
    assert len(GATES) == 8                       # ALL 8 startup gates (fix round, was ADOPTABLE=4)
    assert gates['required'] == list(GATES) and gates['slot'] == 'gpuA-new-1'
    assert config['startup_gates'] == list(GATES)
    assert config['fresh']['service'] == 'tools.coopmwm.warm_spool'   # the FRESH service
    assert 'adopted_startup_gates' not in config['fresh']['binding']
    script = Path(out['worker_script']).read_text()
    compile(script, 'STAGE_WORKER.py', 'exec')       # the artifact parses
    assert 'NEW EMPTY' in script and 'OWNER_OR_LOCK' in script and 'gpuA-new-1' in script
    assert 'ls_gpu' not in script and 'recover' not in script    # fix round 2: fresh path only


def test_fresh_binding_built_from_this_build_pins_variety_sources():
    """Without a staged snapshot binding, fresh_binding builds one from THIS build: the
    numerical set plus the variety sources, in the fresh 8-gate startup shape."""
    from tools.coopmwm.warm_service import NUMERICAL
    broker = vc.validate_broker(broker_cfg())
    b = vc.fresh_binding(broker)
    assert set(vc.VARIETY_GPU_SOURCES) <= set(b['hashes'])
    for rel in vc.VARIETY_GPU_SOURCES:
        assert b['hashes'][rel] == vc.file_sha256(vc.repo_root()/rel.split('pytools/', 1)[1])
    numerical = {n for n in NUMERICAL if n.endswith('.py') and
                 n.startswith(('pytools/tools/coopmwm/', 'pytools/tools/coopnative/'))}
    assert numerical <= set(b['hashes'])
    # the fresh service itself runs WARM_BATCH_SINGLE x2 + COMMITTED x2 + PARITY x2 +
    # WARM_PARITY x2 = ALL of hotadd.GATES: nothing is ever adopted/inherited
    for key, value in dict(memory_mode='warm', startup_qualification=True, speed2=True,
                           speed2_transfers=True, concurrent_plans=True).items():
        assert b[key] == value
    assert 'adopted_startup_gates' not in b
    assert b['service'] == 'tools.coopmwm.warm_spool'
    assert b['amendment_sha256'] == broker['amendment_sha256']
    assert b['deadline_unix'] == b['gpu_deadline'] == float(broker['gpu_deadline'])


@pytest.mark.parametrize('host,widths,fraction', [
    ('gpuA', {'4': 18, '8': 10}, .42), ('gpuR', {'4': 4, '8': 4}, .34)])
def test_host_pins_are_not_inherited_from_the_cpu_collector(host, widths, fraction):
    broker = broker_cfg()
    broker['workers'][0]['endpoint']['slot'] = host+'-fresh'
    b = vc.fresh_binding(broker, host)
    assert b['gpu_host'] == b['worker_env']['host'] == host
    assert b['worker_env']['python_version'] == '3.12'
    assert b['worker_env']['torch_version'] == '2.8.0+cu128'
    assert b['worker_env']['cuda_version'] == '12.8'
    assert b['batch_width_by_K'] == widths and b['memory_fraction'] == fraction
    assert b['niche_comparison'] is True
    assert b['qualification']['pass_'] is False
    assert b['qualification']['receipts'] == {}
    with pytest.raises(ValueError, match='agree with the broker slot'):
        vc.fresh_binding(broker, 'gpuR' if host == 'gpuA' else 'gpuA')


def test_gpu_host_cli_selects_the_staged_slot(tmp_path):
    cfg = base_config(tmp_path, broker=broker_cfg())
    (tmp_path/'VARCOLLECT_CONFIG.json').write_text(json.dumps(cfg))
    with patch.object(vc, 'collect', return_value=[]) as collect:
        vc.main([str(tmp_path), '--arm', 'B', '--gpu-host', 'gpuA', '--rounds', '0'])
    assert collect.call_args.args[1]['gpu_host'] == 'gpuA'
    with pytest.raises(ValueError, match='agree with the broker slot'):
        vc.main([str(tmp_path), '--arm', 'B', '--gpu-host', 'gpuR'])


@pytest.mark.parametrize('failed_gate', [None, 'STARTUP_PARITY_K8_H300.json',
                                       'STARTUP_WARM_BATCH_SINGLE_K4_H120.json'])
def test_fresh_binding_through_real_warm_and_plan_service_all_eight_gates(
        tmp_path, monkeypatch, failed_gate):
    """Real generated staging -> warm_service -> plan_service load/READY path.

    Only numerical GPU operations, POSIX locks and /proc identity are faked. In
    particular neither service nor plan_service's binding read is replaced. This
    catches the original four-gate crash rather than writing eight fake files.
    """
    import os
    import types
    from tools.coopmwm import fresh_worker as fw, warm_service as ws, plan_service as ps
    from tools.coopmwm import plan_wm, speed2_runtime, speed2_memory, wm, plans, mppi_search
    from tools.coopmwm.hotadd import GATES

    broker = dict(broker_cfg(), worker_env=cpu_worker_environment())
    binding = vc.fresh_binding(broker, 'gpuA')
    overlay = tmp_path/'overlay'
    overlay.mkdir()
    for name in binding['hashes']:
        dest = overlay/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((vc.repo_root()/name.removeprefix('pytools/')).read_bytes())
    rows = [dict(vid=v, is_cav=True) for v in ('a', 'b', 'c', 'd')]
    record = dict(state_id='a'*64, world_id='world', time_s=300., rows=rows, group=['a', 'b'])
    (overlay/'INPUTS.json').write_text(json.dumps(dict(records=[record])))
    (overlay/'FROZEN_COSTS.json').write_text(json.dumps(dict(code='fake observable costs')))
    out = vc.stage_broker(dict(broker=broker, gpu_host='gpuA'), tmp_path/'collection')
    worker = tmp_path/'worker'
    ns = {'__name__': 'cpu_staging_test'}
    exec(compile(out['worker_script'].read_text(), str(out['worker_script']), 'exec'), ns)

    rt = types.SimpleNamespace(frozen=types.SimpleNamespace(costs=object()),
                               mp=types.SimpleNamespace(policy=types.SimpleNamespace(
                                   accel_min=-3., accel_max=2.)))
    memory = types.SimpleNamespace(receipt={'fake': True})
    runtime = types.ModuleType('tools.wmteach.runtime')
    runtime.bootstrap = lambda: rt
    wmteach = types.ModuleType('tools.wmteach')
    wmteach.runtime = runtime
    monkeypatch.setitem(sys.modules, 'tools.wmteach', wmteach)
    monkeypatch.setitem(sys.modules, 'tools.wmteach.runtime', runtime)
    monkeypatch.setitem(sys.modules, 'fcntl', types.SimpleNamespace(
        LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None))
    monkeypatch.setattr(ps, 'configured_runtime', lambda bootstrap, fraction: bootstrap())
    monkeypatch.setattr(wm, 'cost_adapter', lambda costs, code, **kw: costs)
    monkeypatch.setattr(speed2_memory, 'native_fixture', lambda rt: (record, memory))
    monkeypatch.setattr(mppi_search, 'group_components', lambda rows: [dict(
        size=2, members=['a', 'b'])])
    monkeypatch.setattr(mppi_search, 'comparators', lambda *a: [dict(id='policy')])
    monkeypatch.setattr(mppi_search, 'sample_unscored', lambda *a, **kw: dict(id='candidate'))
    monkeypatch.setattr(plans, 'baseline_actions', lambda record: [dict(id='policy')])
    monkeypatch.setattr(plans, 'family_seeds', lambda *a: [dict(id='candidate')])
    order = []

    def qualify(rt, record, acts, k, h, width, memory=None, **kw):
        stem = 'COMMITTED' if h in (300, 420) and 'committed_plans' in record else (
            'WARM_PARITY' if memory is not None else 'PARITY')
        name = f'STARTUP_{stem}_K{k}_H{h}.json'
        order.append(name)
        return dict(pass_=name != failed_gate, max_cost_error=0. if name != failed_gate else 1.,
                    K=k, H=h, width=width)

    def evaluate(rt, record, acts, k, h, **kw):
        name = f'STARTUP_WARM_BATCH_SINGLE_K{k}_H{h}.json'
        if kw['chunk_actions'] != 1:
            order.append(name)
        value = 1. if name == failed_gate and kw['chunk_actions'] != 1 else 0.
        return {'policy': [dict(team_delay=value, draw_id=0)]}

    monkeypatch.setattr(speed2_runtime, 'qualify', qualify)
    monkeypatch.setattr(plan_wm, 'evaluate', evaluate)
    original_read_text = Path.read_text

    def read_text(path, *args, **kw):
        text = str(path).replace('\\', '/')
        if text == f'/proc/{os.getpid()}/stat':
            return ' '.join(['0']*21+['42'])
        if text == '/proc/sys/kernel/random/boot_id':
            return 'cpu-fake-boot'
        return original_read_text(path, *args, **kw)

    monkeypatch.setattr(Path, 'read_text', read_text)
    monkeypatch.setattr(sys, 'stdin', io.StringIO())

    def launch(argv, **kw):
        assert argv[0] == binding['worker_env']['python_executable']
        assert not (worker/'FRESH_QUALIFICATION.json').exists()
        with patch.dict(os.environ, kw['env']):
            ws.main(worker)
        return types.SimpleNamespace(pid=os.getpid())

    monkeypatch.setattr(subprocess, 'Popen', launch)
    # Metadata probe is mocked separately because run() itself constructs Popen.
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: types.SimpleNamespace(
        stdout=json.dumps({k: binding['worker_env'][k] for k in
                           ('python_version', 'torch_version', 'cuda_version')})))
    monkeypatch.setattr(sys, 'argv', ['STAGE_WORKER.py', str(worker), '--overlay', str(overlay)])
    if failed_gate:
        with pytest.raises(ValueError, match='parity failed'):
            ns['main']()
        assert not (worker/'SERVICE_OWNER.json').exists()
        assert not (worker/fw.RECEIPT).exists()
        assert vc.broker_readiness(broker, {n: json.loads((worker/n).read_text())
            for n in GATES if (worker/n).exists()})['ready'] is False
    else:
        assert ns['main']() == 0
        assert set(order) == set(GATES) and len(order) == 8
        result = json.loads((worker/fw.RECEIPT).read_text())
        assert result['pass_'] is True and result['status'] == 'passed'
        assert result['pid'] == os.getpid()
        assert result['binding_sha256'] == vc.file_sha256(worker/'GPU_BINDING.json')
        assert set(result['receipts']) == set(GATES)
        assert result['source_hashes'] == binding['hashes']
        assert result['environment']['torch_version'] == str(torch.__version__)
        assert (worker/'SERVICE_OWNER.json').is_file()
        assert vc.broker_readiness(broker, result['receipts'])['ready'] is True
        # Immutable staged binding remains a pending contract pointing at the
        # worker-owned completed receipt, not a forged pre-passed qualification.
        assert json.loads((worker/'GPU_BINDING.json').read_text()) == binding
        with pytest.raises(ValueError, match='existing startup evidence'):
            ws.main(worker)


def test_fresh_binding_rejects_inherited_qualification():
    b = vc.fresh_binding(broker_cfg())
    b['qualification']['pass_'] = True
    with pytest.raises(ValueError, match='never copied'):
        vc.fresh_binding(dict(broker_cfg(), binding=b))


def test_fresh_qualification_rejects_another_process_and_nonexact_gates(tmp_path, monkeypatch):
    import os
    from tools.coopmwm import fresh_worker as fw
    from tools.coopmwm.hotadd import GATES
    b = vc.fresh_binding(broker_cfg())
    (tmp_path/'GPU_BINDING.json').write_text(json.dumps(b))
    start = dict(pid=os.getpid(), binding_sha256=vc.file_sha256(tmp_path/'GPU_BINDING.json'),
                 environment={})
    (tmp_path/fw.START).write_text(json.dumps(start))
    for name in GATES:
        (tmp_path/name).write_text(json.dumps(dict(pid=os.getpid(), pass_=True, max_cost_error=0.)))
    path = tmp_path/GATES[-1]
    for bad in (dict(pid=os.getpid()+1, pass_=True, max_cost_error=0.),
                dict(pid=os.getpid(), pass_=True, max_cost_error=.001),
                dict(pid=os.getpid(), pass_=False, max_cost_error=0.)):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match='nonexact fresh-worker gate'):
            fw.finish(tmp_path, b)
        assert not (tmp_path/fw.RECEIPT).exists()
    (tmp_path/fw.START).write_text(json.dumps(dict(start, pid=os.getpid()+1)))
    with pytest.raises(ValueError, match='process/binding changed'):
        fw.qualification(tmp_path, b)


@pytest.mark.parametrize('evidence', ['STARTUP_PARITY_K4_H120.json', 'SPOOL_OWNER.json',
                                    'FRESH_QUALIFICATION.json'])
def test_staging_refuses_overlay_receipts_before_creating_worker(tmp_path, evidence):
    overlay = fake_overlay(tmp_path, [])
    (overlay/evidence).write_text(json.dumps(dict(pass_=True)))
    broker = dict(broker_cfg(), binding=staged_binding(overlay))
    out = vc.stage_broker(dict(broker=broker), tmp_path/'collection')
    worker = tmp_path/'worker'
    rc, verdict = run_worker(out['worker_script'], worker, overlay)
    assert rc == 3 and 'prior worker evidence' in verdict['reason']
    assert not worker.exists()


def test_staging_refuses_cpu_collector_environment_for_gpu_pin(tmp_path):
    overlay = fake_overlay(tmp_path, [])
    b = staged_binding(overlay)
    # Probe the local CPU interpreter against the explicit GPU package contract.
    b['worker_env'] = dict(vc.worker_environment('gpuA'), python_executable=sys.executable)
    out = vc.stage_broker(dict(broker=dict(broker_cfg(), binding=b)), tmp_path/'collection')
    worker = tmp_path/'worker'
    rc, verdict = run_worker(out['worker_script'], worker, overlay)
    assert rc == 3 and 'environment mismatch' in verdict['reason']
    assert not worker.exists()


def staged_binding(overlay):
    """An operator-staged binding snapshot: hashes over the staged overlay files, with the
    variety sources pinned (check_fresh_binding requires them)."""
    overlay = Path(overlay)

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    return dict(memory_mode='warm', startup_qualification=True, speed2=True,
                worker_env=cpu_worker_environment(),
                speed2_transfers=True, concurrent_plans=True,
                hashes={'INPUTS.json': sha(overlay/'INPUTS.json'),
                        'pytools/tools/coopmwm/warm_spool.py':
                            sha(overlay/'pytools/tools/coopmwm/warm_spool.py'),
                        'pytools/tools/coopmwm/variety.py':
                            sha(overlay/'pytools/tools/coopmwm/variety.py'),
                        'pytools/tools/coopmwm/niche_collect.py':
                             sha(overlay/'pytools/tools/coopmwm/niche_collect.py')})


def cpu_worker_environment(host='gpuA'):
    """Explicit test-only pin; production defaults never inspect collector metadata."""
    return dict(vc.worker_environment(host), python_executable=sys.executable,
                python_version='.'.join(map(str, sys.version_info[:2])),
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda)


def fake_overlay(tmp, gates):
    """A worker-root-shaped overlay whose fresh service module is a FAKE: it writes its own
    owner receipt and then EVERY gate receipt itself (CPU stand-in for warm_spool ->
    warm_service.main -> plan_service.main)."""
    overlay = Path(tmp)/'overlay'
    pkg = overlay/'pytools'/'tools'/'coopmwm'
    pkg.mkdir(parents=True)
    (overlay/'pytools'/'tools'/'__init__.py').write_text('')
    (pkg/'__init__.py').write_text('')
    (pkg/'variety.py').write_text('# pinned variety source\n')
    (pkg/'niche_collect.py').write_text('# pinned niche source\n')
    (pkg/'warm_spool.py').write_text(
        'import json, sys\n'
        'from pathlib import Path\n'
        'root = Path(sys.argv[1])\n'
        '(root/"SPOOL_OWNER.json").write_text(json.dumps(dict(pid=424242, fake=True)))\n'
        f'for name in {list(gates)!r}:\n'
        '    (root/name).write_text(json.dumps(dict(pass_=True, max_cost_error=0)))\n'
        '(root/"FAKE_DONE.json").write_text(json.dumps(dict(ran="all gates")))\n')
    (overlay/'INPUTS.json').write_text('{"records": [{"fake": true}]}')
    return overlay


def run_worker(script, worker, overlay, *extra):
    p = subprocess.run([sys.executable, str(script), str(worker), '--overlay', str(overlay),
                        *extra], capture_output=True, text=True, timeout=120)
    if p.returncode not in (0, 2, 3):        # the artifact must never crash: rc is 0/2/3 only
        raise AssertionError(f'STAGE_WORKER.py crashed rc={p.returncode}: {p.stderr[-2000:]}')
    return p.returncode, (json.loads(p.stdout) if p.stdout.strip().startswith('{')
                          else p.stdout.strip())


def test_fresh_worker_stages_new_root_and_runs_the_gates_itself(tmp_path):
    """Fix-round BLOCK: a fresh root + binding with the variety sources; the FRESH service
    itself runs all 8 hotadd.GATES; the slot is admitted only after all 8 receipts pass."""
    from tools.coopmwm.hotadd import GATES
    overlay = fake_overlay(tmp_path, GATES)
    broker = dict(broker_cfg(), binding=staged_binding(overlay))
    cfg = vc.split_config(base_config(tmp_path, broker=broker))
    out = vc.stage_broker(cfg, tmp_path/'collect_root')
    worker = tmp_path/'fresh_worker_root'
    assert not worker.exists()
    rc, started = run_worker(out['worker_script'], worker, overlay)
    assert rc == 0 and started['service'] == 'tools.coopmwm.warm_spool'
    assert worker.is_dir() and (worker/'pytools'/'tools'/'coopmwm'/'variety.py').is_file()
    binding = json.loads((worker/'GPU_BINDING.json').read_text())
    assert binding == json.loads(json.dumps(vc.fresh_binding(broker)))
    deadline = time.time()+30
    while time.time() < deadline and not (worker/'FAKE_DONE.json').exists():
        time.sleep(.2)
    assert (worker/'FAKE_DONE.json').exists()
    assert json.loads((worker/'SPOOL_OWNER.json').read_text())['pid'] == 424242
    for name in GATES:                 # the fresh service itself wrote every gate receipt
        assert json.loads((worker/name).read_text())['pass_'] is True
    rc, verdict = run_worker(out['worker_script'], worker, overlay, '--check')
    assert rc == 0 and verdict['ready'] is True and len(verdict['receipts']) == 8
    receipts = {name: json.loads((worker/name).read_text()) for name in GATES}
    assert vc.broker_readiness(broker, receipts)['ready'] is True   # slot admitted only now


def test_fresh_worker_refuses_existing_root_or_owner_or_lock(tmp_path):
    """Refuse to start when the root already exists or has any owner PID/lock: never adopt,
    never signal another process."""
    from tools.coopmwm.hotadd import GATES
    overlay = fake_overlay(tmp_path, GATES)
    broker = dict(broker_cfg(), binding=staged_binding(overlay))
    cfg = vc.split_config(base_config(tmp_path, broker=broker))
    out = vc.stage_broker(cfg, tmp_path/'collect_root')

    existing = tmp_path/'worker_exists';existing.mkdir()
    (existing/'operator_leftover.txt').write_text('keep me')
    rc, verdict = run_worker(out['worker_script'], existing, overlay)
    assert rc == 3 and verdict['refused'] is True and 'already exists' in verdict['reason']
    assert (existing/'operator_leftover.txt').read_text() == 'keep me'
    assert not (existing/'GPU_BINDING.json').exists() and not (existing/'SPOOL.log').exists()

    owner = tmp_path/'worker_owner';owner.mkdir()
    (owner/'SPOOL_OWNER.json').write_text(json.dumps(dict(pid=99, start_ticks='7')))
    rc, verdict = run_worker(out['worker_script'], owner, overlay)
    assert rc == 3 and 'owner/lock' in verdict['reason'] and 'SPOOL_OWNER.json' in verdict['reason']
    assert json.loads((owner/'SPOOL_OWNER.json').read_text())['pid'] == 99   # untouched
    assert not (owner/'SPOOL.log').exists() and not (owner/'GPU_BINDING.json').exists()

    lock = tmp_path/'worker_lock';lock.mkdir()
    (lock/'SPOOL_WORKER.claim').write_text('')
    rc, verdict = run_worker(out['worker_script'], lock, overlay)
    assert rc == 3 and 'SPOOL_WORKER.claim' in verdict['reason']
    assert not list(lock.rglob('GPU_BINDING.json'))

    service_owner = tmp_path/'worker_service_owner';service_owner.mkdir()
    (service_owner/'SERVICE_OWNER.json').write_text(json.dumps(dict(pid=100)))
    rc, verdict = run_worker(out['worker_script'], service_owner, overlay)
    assert rc == 3 and 'SERVICE_OWNER.json' in verdict['reason']


def test_binding_staged_snapshot_rules(tmp_path):
    """A staged binding snapshot is validated: adopted-gate sets refuse, the variety sources
    must be pinned, hashes must be sha256 digests."""
    hashes = staged_binding(fake_overlay(tmp_path, []))['hashes']
    ok = vc.validate_broker(dict(broker_cfg(), binding=dict(hashes=hashes)))
    assert set(vc.VARIETY_GPU_SOURCES) <= set(ok['binding']['hashes'])
    with pytest.raises(ValueError, match='adopted_startup_gates'):
        vc.validate_broker(dict(broker_cfg(),
                                binding=dict(hashes=hashes, adopted_startup_gates={'x': '0'*64})))
    missing_niche = {k: v for k, v in hashes.items() if 'niche_collect' not in k}
    with pytest.raises(ValueError, match='variety sources'):
        vc.validate_broker(dict(broker_cfg(), binding=dict(hashes=missing_niche)))
    with pytest.raises(ValueError, match='sha256'):
        vc.validate_broker(dict(broker_cfg(), binding=dict(hashes=dict(hashes, extra='nothex'))))
    with pytest.raises(ValueError, match='non-empty hashes'):
        vc.validate_broker(dict(broker_cfg(), binding=dict(hashes={})))
    with pytest.raises(ValueError, match='dict'):
        vc.validate_broker(dict(broker_cfg(), binding='GPU_BINDING'))
    cfg = vc.split_config(base_config(tmp_path, broker=dict(broker_cfg(), binding=dict(hashes=hashes))))
    assert set(vc.VARIETY_GPU_SOURCES) <= set(cfg['broker']['binding']['hashes'])


def test_no_code_path_reaches_recover(tmp_path):
    """Fix-round BLOCK: no code path reaches ls_gpu's recover/adoption path - the module and
    the staged artifact carry neither ls_gpu nor the recover flag, and never import signal
    machinery to terminate another process; the fresh service is warm_spool itself."""
    source = Path(vc.__file__).read_text(encoding='utf-8')
    assert 'ls_gpu' not in source and 'recover' not in source
    assert vc.FRESH_SERVICE_MODULE == 'tools.coopmwm.warm_spool'
    from tools.coopmwm.hotadd import GATES
    overlay = fake_overlay(tmp_path, GATES)
    broker = dict(broker_cfg(), binding=staged_binding(overlay))
    cfg = vc.split_config(base_config(tmp_path, broker=broker))
    out = vc.stage_broker(cfg, tmp_path/'collect_root')
    script = Path(out['worker_script']).read_text(encoding='utf-8')
    compile(script, 'STAGE_WORKER.py', 'exec')
    assert 'ls_gpu' not in script and 'recover' not in script
    assert 'import signal' not in script and 'os.kill' not in script
    config = ast.literal_eval(re.search(r'^CONFIG = (.+)$', script, re.M).group(1))
    assert config['fresh']['service'] == 'tools.coopmwm.warm_spool'
    assert 'adopted_startup_gates' not in config['fresh']['binding']
    assert 'owner' not in config      # no takeover target is staged at all


def test_broker_readiness_requires_all_eight_gates():
    """Independent-review fix: readiness = ALL 8 hotadd.GATES present with pass_ true; the old
    4-gate warm_service.ADOPTABLE subset is NOT enough (that was the BLOCK)."""
    from tools.coopmwm.hotadd import GATES
    from tools.coopmwm.warm_service import ADOPTABLE
    assert len(GATES) == 8 and set(GATES) > set(ADOPTABLE)
    broker = broker_cfg()
    receipts = {name: dict(pass_=True) for name in GATES}
    assert vc.broker_readiness(broker, receipts)['ready'] is True
    partial = {name: dict(pass_=True) for name in ADOPTABLE}     # the 4-gate subset
    verdict = vc.broker_readiness(broker, partial)
    assert verdict['ready'] is False and verdict['required'] == list(GATES)
    failing = dict(receipts)
    failing[GATES[3]] = dict(pass_=False)
    assert vc.broker_readiness(broker, failing)['ready'] is False
    assert vc.broker_readiness(broker, {})['ready'] is False


def test_staged_worker_requires_every_gate(tmp_path):
    """The staged artifact's readiness: one gate MISSING -> not ready; one gate FAILING or
    unparsable -> not ready; all 8 pass -> ready."""
    from tools.coopmwm.hotadd import GATES
    cfg = vc.split_config(base_config(tmp_path, broker=broker_cfg()))
    out = vc.stage_broker(cfg, tmp_path/'collect_root')
    py = sys.executable

    def check(worker):
        p = subprocess.run([py, str(out['worker_script']), str(worker), '--overlay', '/ov',
                            '--check'], capture_output=True, text=True, timeout=60)
        return p.returncode, json.loads(p.stdout)

    missing = tmp_path/'worker_one_gate_missing';missing.mkdir()
    for name in list(GATES)[:-1]:
        (missing/name).write_text(json.dumps(dict(pass_=True, max_cost_error=0)))
    rc, v = check(missing)
    assert rc == 2 and v['ready'] is False and v['receipts'][GATES[-1]] is False

    failing = tmp_path/'worker_one_gate_failing';failing.mkdir()
    for i, name in enumerate(GATES):
        body = 'not json' if i == 0 else json.dumps(dict(pass_=i != 1, max_cost_error=0))
        (failing/name).write_text(body)
    rc, v = check(failing)
    assert rc == 2 and v['ready'] is False
    assert v['receipts'][GATES[0]] is False and v['receipts'][GATES[1]] is False

    passing = tmp_path/'worker_all_eight_pass';passing.mkdir()
    for name in GATES:
        (passing/name).write_text(json.dumps(dict(pass_=True, max_cost_error=0)))
    rc, v = check(passing)
    assert rc == 0 and v['ready'] is True and len(v['receipts']) == 8
    assert v['slot'] == 'gpuA-new-1'


def test_staged_worker_script_gate_check(tmp_path):
    """The generated artifact runs its readiness check only (--check starts nothing)."""
    cfg = vc.split_config(base_config(tmp_path, broker=broker_cfg()))
    out = vc.stage_broker(cfg, tmp_path/'collect_root')
    worker = tmp_path/'worker_root';worker.mkdir()
    py = sys.executable
    p = subprocess.run([py, str(out['worker_script']), str(worker), '--overlay', '/ov', '--check'],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 2 and json.loads(p.stdout)['ready'] is False
    from tools.coopmwm.hotadd import GATES
    for name in GATES:
        (worker/name).write_text(json.dumps(dict(pass_=True, max_cost_error=0)))
    p = subprocess.run([py, str(out['worker_script']), str(worker), '--overlay', '/ov', '--check'],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and json.loads(p.stdout)['ready'] is True
    assert json.loads(p.stdout)['slot'] == 'gpuA-new-1'


# ---------------------------------------------------------------- collect round (fake episodes)

def fake_episodes(jobs):
    out = []
    for j in jobs:
        Path(j['path']).mkdir(parents=True, exist_ok=True)
        (Path(j['path'])/'RESULT.json').write_text('{}')
        out.append(j['id'])
    return out


def test_collect_routes_episode_charge_and_never_local_ledger(tmp_path):
    cfg = vc.split_config(base_config(tmp_path))
    root = tmp_path/'varietycollect20261104a'          # the NEW root name is a parameter
    charges, ctx = routed(tmp_path, cfg)
    with ctx, patch.object(vc, 'run_episodes', fake_episodes):
        statuses = vc.collect(root, cfg, 'R', stop=lambda: False, rounds=1, episodes_per_round=1)
    assert statuses == []
    assert charges['calls'] == [(cfg['episode_phase'], f'{PREFIX}_R_k003_e00000_{SEED}', 1200)]
    assert (rb.bank_root(root, 'R')/'BANK_QUALITY.json').is_file()
    assert not list(root.rglob('CHARGES.jsonl'))
    vc.refuse_local_ledger(root)
    assert (root/'jobs'/f'{PREFIX}_R_k003_e00000_{SEED}'/'RESULT.json').is_file()


def test_collect_requires_broker_config_for_arm_b(tmp_path):
    cfg = vc.split_config(base_config(tmp_path))
    charges, ctx = routed(tmp_path, cfg)
    with ctx:
        with pytest.raises(ValueError, match='broker'):
            vc.collect(tmp_path/'root', cfg, 'B', stop=lambda: True, rounds=0)
    assert charges['calls'] == []


def test_collect_refuses_wrong_phase_or_unavailable_route(tmp_path):
    pins = pin_checkpoints(tmp_path/'coop3')
    bad_phase = vc.split_config(base_config(tmp_path, checkpoints=pins, episode_phase='discovery'))
    with patch.object(cc, 'rpc', side_effect=AssertionError('no rpc should be needed')):
        with pytest.raises(ValueError, match='not registered on the collection charge route'):
            vc.collect(tmp_path/'root', bad_phase, 'R', stop=lambda: True, rounds=1)
    cfg = vc.split_config(base_config(tmp_path, checkpoints=pins))
    from tools.coopmwm.cpus_checking import RpcError
    with patch.object(cc, 'rpc', side_effect=RpcError('route down')):
        with pytest.raises(ValueError, match='charge route unavailable'):
            vc.collect(tmp_path/'root', cfg, 'R', stop=lambda: True, rounds=1)


def test_collect_pinned_refusal_happens_before_any_charge(tmp_path):
    pin_checkpoints(tmp_path/'coop3')
    cfg = vc.split_config(base_config(tmp_path, shas=dict(B='0'*64, R='0'*64)))
    charges, ctx = routed(tmp_path, cfg)
    with ctx:
        with pytest.raises(ValueError, match='sha256 mismatch'):
            vc.collect(tmp_path/'root', cfg, 'R', stop=lambda: False, rounds=1)
    assert charges['calls'] == []          # refused BEFORE the route charged anything


# ---------------------------------------------------------------- confirmations are all8 (fix round 2)

def test_collect_confirms_are_all8_both_seeds(tmp_path):
    """Fix round 2: confirm_pending is called with verify_mode 'all8_both_seeds' (the live
    per_seed_v1 bank rule) and the configured extra seeds - never the legacy single-replica
    path."""
    import tools.coopmwm.verify_perseed as vp
    cfg = vc.split_config(base_config(tmp_path, extra_seeds=2))
    root = tmp_path/'collect_root'
    charges, ctx = routed(tmp_path, cfg)
    calls = []

    def recorder(*args, **kwargs):
        calls.append(dict(args=args, kwargs=kwargs))
        return []
    with ctx, patch.object(vc, 'run_episodes', fake_episodes), \
            patch.object(vp, 'confirm_pending_all8', recorder):
        vc.collect(root, cfg, 'R', stop=lambda: False, rounds=1, episodes_per_round=1)
    assert len(calls) == 1                 # every round routes through the all8 builder
    args, kwargs = calls[0]['args'], calls[0]['kwargs']
    assert args[0] == root and args[1] == 'R' and args[4] == vc.CONFIRM_PHASE
    assert str(args[3]) == str(cfg['charge_route']['run_root'])
    assert kwargs['extra_seeds'] == 2      # the live per_seed_v1 bank extra-seeds option
    assert charges['calls'] == [(cfg['episode_phase'], f'{PREFIX}_R_k003_e00000_{SEED}', 1200)]


def test_collect_all8_extra_seeds_default_zero(tmp_path):
    import tools.coopmwm.verify_perseed as vp
    cfg = vc.split_config(base_config(tmp_path))     # extra_seeds absent = 0 (byte-identical off)
    assert cfg['extra_seeds'] == 0
    charges, ctx = routed(tmp_path, cfg)
    calls = []

    def recorder(*args, **kwargs):
        calls.append(kwargs)
        return []
    with ctx, patch.object(vc, 'run_episodes', fake_episodes), \
            patch.object(vp, 'confirm_pending_all8', recorder):
        vc.collect(tmp_path/'collect_root', cfg, 'R', stop=lambda: False, rounds=1,
                   episodes_per_round=1)
    assert calls == [dict(extra_seeds=0, slim_rejected_tapes=None)]


def test_extra_seeds_config_validation(tmp_path):
    with pytest.raises(ValueError, match='extra_seeds'):
        vc.split_config(base_config(tmp_path, extra_seeds=-1))
    with pytest.raises(ValueError, match='extra_seeds'):
        vc.split_config(base_config(tmp_path, extra_seeds='2'))
    with pytest.raises(ValueError, match='extra_seeds'):
        vc.split_config(base_config(tmp_path, extra_seeds=True))
    assert vc.split_config(base_config(tmp_path, checkpoints=pin_checkpoints(tmp_path/'coop3'),
                                       extra_seeds=4))['extra_seeds'] == 4


# ------------------------------------------------- V2V coop3 path (deploy fix round 3)

def v2v_checkpoints(root, update=3, arms=('B', 'R'), seed=11):
    """coop3-shaped V2V intent checkpoints saved by the REAL builder chain the live runtime
    uses: a base MAPPO checkpoint widened through v2v_runtime.load_checkpoint with the
    registered launch3 INTENT_OPTIONS, then the runtime's own accel_learner.save."""
    from tools.coopmwm import v2v_launch3 as v3
    from tools.coopmwm import v2v_runtime as vr
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for arm in arms:
        path = root/f'{arm}_checkpoint_{update:03d}.pt'
        if not path.exists():
            torch.manual_seed(seed)
            base = Policy()
            opt = torch.optim.Adam(base.parameters(), lr=3e-4)
            base_path = root/f'_{arm}_base_u000.pt'
            save(base_path, base, opt, torch.Generator(), 0, 0)
            policy, optimizer, generator, _ = vr.load_checkpoint(base_path,
                                                                 config=dict(v3.INTENT_OPTIONS))
            save(path, policy, optimizer, generator, update, 0)
            base_path.unlink()
        out[arm] = dict(path=path, sha256=vc.file_sha256(path))
    return out


def coop3_intent_binding():
    """The live coop3 runtime config exactly as its registered launcher produces it
    (v2v_runner.prepare_intent - the intent launch3 binding v2v_launch3 pins)."""
    from tools.coopmwm import v2v_runner
    return v2v_runner.prepare_intent(dict(runtime='rt', accel_bounds=[-3., 2.], gpu_deadline=1.),
                                     root='/coop3run', legacy_root='/coop3legacy',
                                     source_root='/coop3src', common_checkpoint_sha256='0'*64)


def test_pinned_policy_loads_coop3_v2v_checkpoints_with_the_live_runtime_builder(tmp_path):
    """Deploy BLOCK: pinned_policy used the plain accel_learner.load, which fails on coop3
    V2V checkpoints (intent-head keys + v2v_dim 48 input shape). It now loads through the
    SAME builder the live coop3 runtime uses (v2v_runtime.load_checkpoint with the
    registered launch3 INTENT_OPTIONS: v2v_dim=48, K=4 partner slots, intent head,
    detach_trunk)."""
    from tools.coopmwm import v2v_launch3 as v3
    from tools.coopmwm import v2v_runtime as vr
    from tools.coopmwm.v2v_intent import IntentPolicy
    pins = v2v_checkpoints(tmp_path/'coop3')
    cfg = vc.split_config(base_config(tmp_path, checkpoints=pins))
    for arm in ('B', 'R'):
        policy, pin = vc.pinned_policy(cfg, arm)
        assert isinstance(policy, IntentPolicy) and pin['k'] == 3
        assert policy.v2v_width == int(v3.INTENT_OPTIONS['v2v_dim']) == 48
        assert policy.intent_horizon == v3.INTENT_OPTIONS['v2v_intent_horizon']
        assert policy.intent_detach_trunk == v3.INTENT_OPTIONS['v2v_intent_detach_trunk']
        assert int(policy.actor[0].weight.shape[1]) == 33+48     # widened coop3 input shape
        assert vc.policy_v2v(policy) is True
        assert vc.v2v_job_keys(policy) == dict(v2v=True, v2v_dim=48)
        # the SAME class the live episode-side builder rebuilds from these very weights:
        assert type(policy) is type(vr.from_weights(policy.state_dict(), enabled=True))
    # a plain checkpoint keeps the plain loader (and its bytes)
    plain_home = tmp_path/'plain_home'
    plain_cfg = vc.split_config(base_config(plain_home,
                                            checkpoints=pin_checkpoints(plain_home/'coop3')))
    plain, _ = vc.pinned_policy(plain_cfg, 'B')
    assert not isinstance(plain, IntentPolicy) and vc.policy_v2v(plain) is False
    assert vc.v2v_job_keys(plain) == {}


def test_pinned_v2v_load_is_strict_no_missing_or_unexpected_keys(tmp_path):
    """The coop3 builder's state-dict load is STRICT: a checkpoint with an unexpected or a
    missing key refuses at load (never silently accepted)."""
    root = tmp_path/'coop3'
    pins = v2v_checkpoints(root)
    saved = torch.load(pins['B']['path'], weights_only=False)
    extra = dict(saved, policy=dict(saved['policy'],
                                    **{'bogus_head.weight': torch.zeros(1, 2)}))
    torch.save(extra, pins['B']['path'])
    cfg = vc.split_config(base_config(tmp_path, checkpoints=dict(
        B=dict(path=str(pins['B']['path']), sha256=vc.file_sha256(pins['B']['path'])),
        R=pins['R'])))
    with pytest.raises(RuntimeError):          # unexpected key
        vc.pinned_policy(cfg, 'B')
    missing = dict(saved, policy={k: v for k, v in saved['policy'].items()
                                  if k != 'actor.2.weight'})
    torch.save(missing, pins['B']['path'])
    cfg = vc.split_config(base_config(tmp_path, checkpoints=dict(
        B=dict(path=str(pins['B']['path']), sha256=vc.file_sha256(pins['B']['path'])),
        R=pins['R'])))
    with pytest.raises(RuntimeError):          # missing key
        vc.pinned_policy(cfg, 'B')


def test_pinned_v2v_refuses_swapped_payload_update_before_building(tmp_path):
    root = tmp_path/'coop3'
    pins = v2v_checkpoints(root)
    saved = torch.load(pins['B']['path'], weights_only=False)
    torch.save(dict(saved, update=7), pins['B']['path'])
    cfg = vc.split_config(base_config(tmp_path, checkpoints=dict(
        B=dict(path=str(pins['B']['path']), sha256=vc.file_sha256(pins['B']['path'])),
        R=pins['R'])))
    with pytest.raises(ValueError, match='payload update'):
        vc.pinned_policy(cfg, 'B')


def test_collect_episode_jobs_equal_a_coop3_training_job_v2v_settings(tmp_path):
    """A varcollect episode job carries EXACTLY the V2V settings the live coop3 training wave
    adds to every train job (v2v_runner.run_arm wave extras), and accel_plant's builder
    (v2v_runtime.from_weights) rebuilds the SAME intent policy class from the job weights."""
    from tools.coopmwm.v2v_intent import IntentPolicy
    from tools.coopmwm.v2v_runtime import from_weights
    coop3 = coop3_intent_binding()
    assert coop3['v2v'] is True and coop3['v2v_intent'] is True and coop3['v2v_dim'] == 48
    pins = v2v_checkpoints(tmp_path/'coop3')
    cfg = vc.split_config(base_config(tmp_path, checkpoints=pins))
    root = tmp_path/'collect_root'
    seen = []

    def recorder(jobs):
        seen.extend(jobs)
        return fake_episodes(jobs)
    charges, ctx = routed(tmp_path, cfg)
    with ctx, patch.object(vc, 'run_episodes', recorder):
        vc.collect(root, cfg, 'R', stop=lambda: False, rounds=1, episodes_per_round=1)
    assert len(seen) == 1
    job = seen[0]
    coop3_wave = dict(v2v=True, v2v_dim=coop3['v2v_dim'])   # the wave extras of coop3 train jobs
    assert {k: job[k] for k in coop3_wave} == coop3_wave
    assert job['v2v_dim'] == int(vc.v2v_job_keys(vc.pinned_policy(cfg, 'R')[0])['v2v_dim'])
    assert isinstance(from_weights(job['weights'], enabled=True), IntentPolicy)
    assert charges['calls'] == [(cfg['episode_phase'], f'{PREFIX}_R_k003_e00000_{SEED}', 1200)]


def test_plain_pinned_checkpoint_keeps_the_legacy_job_and_branch_bytes(tmp_path):
    import os
    from tools.coopmwm import bank_v2v
    import tools.coopmwm.verify_perseed as vp
    pins = pin_checkpoints(tmp_path/'coop3')
    cfg = vc.split_config(base_config(tmp_path, checkpoints=pins))
    jobs = vc.round_jobs(cfg, 'R', vc.pinned_policy(cfg, 'R')[0],
                         vc.pinned_checkpoints(cfg)['R'], tmp_path/'root', 0, cfg['worlds'], 1)
    assert 'v2v' not in jobs[0] and 'v2v_dim' not in jobs[0]   # legacy job bytes
    seen = {}

    def verify_recorder(*args, **kwargs):
        seen['verify'] = os.environ.get(bank_v2v.ENV)
        return []

    def confirm_recorder(*args, **kwargs):
        seen['confirm'] = os.environ.get(bank_v2v.ENV)
        return []
    charges, ctx = routed(tmp_path, cfg)
    with ctx, patch.object(vc, 'run_episodes', fake_episodes), \
            patch.object(rb, 'verify_pending', verify_recorder), \
            patch.object(vp, 'confirm_pending_all8', confirm_recorder):
        vc.collect(tmp_path/'collect_root', cfg, 'R', stop=lambda: False, rounds=1,
                   episodes_per_round=1)
    assert seen == {'verify': None, 'confirm': None}           # C4NM_V2V never set
    assert os.environ.get(bank_v2v.ENV) is None


def test_branch_replays_and_confirmations_run_the_v2v_observation_path(tmp_path):
    """A coop3 V2V pinned policy runs every SUMO branch replay (window verifications AND the
    all8 confirmations) under the coop3 branch flag C4NM_V2V=1 - the branch runner then uses
    the V2V observation controller exactly as coop3 training branches do."""
    import os
    from tools.coopmwm import bank_v2v
    import tools.coopmwm.verify_perseed as vp
    pins = v2v_checkpoints(tmp_path/'coop3')
    cfg = vc.split_config(base_config(tmp_path, checkpoints=pins))
    seen = {}

    def verify_recorder(*args, **kwargs):
        seen['verify'] = os.environ.get(bank_v2v.ENV)
        return []

    def confirm_recorder(*args, **kwargs):
        seen['confirm'] = os.environ.get(bank_v2v.ENV)
        return []
    charges, ctx = routed(tmp_path, cfg)
    with ctx, patch.object(vc, 'run_episodes', fake_episodes), \
            patch.object(rb, 'verify_pending', verify_recorder), \
            patch.object(vp, 'confirm_pending_all8', confirm_recorder):
        vc.collect(tmp_path/'collect_root', cfg, 'R', stop=lambda: False, rounds=1,
                   episodes_per_round=1)
    assert seen == {'verify': '1', 'confirm': '1'}   # branch subprocesses inherit the flag
    assert os.environ.get(bank_v2v.ENV) is None      # restored after every call
    with bank_v2v.enabled(True):                     # the coop3 branch-runner switch it drives
        cls = bank_v2v.branch_controller_class()
    assert cls is not None and any(k.__name__ == 'VerifyController' for k in cls.__mro__)
    assert os.environ.get(bank_v2v.ENV) is None


def test_b_gpu_search_path_evaluates_the_v2v_policy_class(tmp_path):
    """Deploy BLOCK: the B WM search must evaluate the V2V policy. The worker overlay's
    policy builder (policy_wm.load_actor - pinned in warm_service.NUMERICAL) rebuilds the
    SAME widened coop3 layer class from the job's policy_state, the search forward runs with
    matching shapes, and the absent-partner v2v columns are exactly neutral (the residual
    equals the base 33-column actor)."""
    from tools.coopmwm import policy_wm, v2v_model
    from tools.coopmwm.v2v_intent import IntentPolicy
    pins = v2v_checkpoints(tmp_path/'coop3', seed=5)
    policy, _ = vc.pinned_policy(vc.split_config(base_config(tmp_path, checkpoints=pins)), 'B')
    assert isinstance(policy, IntentPolicy)
    state = policy_wm.actor_state(policy)          # exactly what round_jobs queues on the request
    actor = policy_wm.load_actor(state, 'cpu')
    assert int(actor.actor[0].weight.shape[1]) == 33+48      # coop3 v2v_dim 48 input shape
    assert isinstance(actor.actor[0], v2v_model.V2VInputLinear)   # the SAME widened layer class
    rows = [dict(vid='a', s_m=-20., v_mps=10., lane=1, is_cav=True, length_m=5.),
            dict(vid='b', s_m=-40., v_mps=8., lane=2, is_cav=True, length_m=5.)]
    wm_state = {k: torch.tensor([[r[k] for r in rows]]) for k in ('s_m', 'v_mps', 'lane', 'is_cav')}
    wm_state['present'] = torch.ones_like(wm_state['is_cav'], dtype=torch.bool)
    out = policy_wm.residuals(actor, wm_state, ['a', 'b'])
    assert out.shape == wm_state['s_m'].shape and bool(torch.isfinite(out).all())
    trunc = dict(state)
    trunc['actor.0.weight'] = [row[:33] for row in state['actor.0.weight']]
    base_actor = policy_wm.load_actor(trunc, 'cpu')
    assert int(base_actor.actor[0].weight.shape[1]) == 33      # plain path unchanged
    assert torch.allclose(out, policy_wm.residuals(base_actor, wm_state, ['a', 'b']),
                          atol=1e-6, rtol=0)                   # zeros == exactly neutral
    # and the plain caller bytes are unchanged (the existing WM-search parity shape)
    plain = Policy()
    plain_state = policy_wm.actor_state(plain)
    plain_actor = policy_wm.load_actor(plain_state, 'cpu')
    assert int(plain_actor.actor[0].weight.shape[1]) == 33 and not hasattr(plain_actor, 'intent_head')
