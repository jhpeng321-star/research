"""CPU tests for the variety plan-collection path (arms B and R; c4/variety-collect).

Real code paths with a fake cpuQ RPC and fake episodes: pinned-checkpoint refusals (wrong
sha/k), charges ONLY through the cpuS->cpuQ RPC route with the registered phases
(sumo_verification, bank_verification_ext), never a second CHARGES.jsonl, B/R variety-mix
wiring (B floor(share*N) search slots, R Bernoulli(share) unscored draws), parameterised
episode ids, single-slot broker staging artifacts. No SUMO, no ssh, no GPU.
"""
import hashlib
import inspect
import io
import json
import re
import socket
import subprocess
import sys
import tempfile
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
    script = Path(out['worker_script']).read_text()
    compile(script, 'STAGE_WORKER.py', 'exec')       # the artifact parses
    assert '_VARCOLLECT' in script and 'gpuA-new-1' in script


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
