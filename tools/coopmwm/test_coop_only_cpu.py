"""CPU tests: imitation_bank_filter (owner 2026-11-04 00:55). Default 'all_admitted' is byte-identical;
'cooperative_only' (the V2V runtime value for B and R alike) keeps only within-seed cooperative examples."""
import contextlib
import json

import pytest
import torch

from tools.coopmwm import bank_imitation as bi
from tools.coopmwm import coopfix_exact_gate as gate
from tools.coopmwm import replay_bank as rb
from tools.coopmwm import v2v_launch3 as launch
from tools.coopmwm import v2v_runner
from tools.coopmwm.protocol import digest
from tools.coopmwm.test_bank_boot_cpu import noisy
from tools.coopmwm.test_bank_imitation_cpu import Policy
from tools.coopmwm.test_bank_perseed_cpu import D, bs, make, ps, win  # noqa: F401


def mixed_bank(tmp_path):
    """c1only: seed1 cooperative + seed2 not_cooperative; crep: both cooperative; nocoop: both no_solos."""
    make(tmp_path, 'c1only', win(), win(), syn1=noisy(3., .5, 104), syn2=noisy(0., 1., 105))
    make(tmp_path, 'crep', win(), win(), syn1=noisy(3., .5, 106), syn2=noisy(2., .5, 107))
    make(tmp_path, 'nocoop', win(), win(), cooperation=False)


def test_default_is_byte_identical_and_unknown_filter_refused(tmp_path, ps):
    mixed_bank(tmp_path)
    p0, s0 = rb.bank_pairs(tmp_path, 'B', 12345)
    p1, s1 = rb.bank_pairs(tmp_path, 'B', 12345, imitation_bank_filter='all_admitted')
    assert digest([(a, b) for a, b in p0]) == digest([(a, b) for a, b in p1])
    assert json.dumps(s0, sort_keys=True, default=str) == json.dumps(s1, sort_keys=True, default=str)
    assert 'imitation_filter' not in s0 and len(s0['entries']) == 6
    with pytest.raises(ValueError, match='imitation_bank_filter'):
        rb.bank_pairs(tmp_path, 'B', 12345, imitation_bank_filter='bogus')


def test_cooperative_only_keeps_cooperative_and_excludes_no_solos(tmp_path, ps):
    mixed_bank(tmp_path)
    pairs, s = rb.bank_pairs(tmp_path, 'B', 12345, imitation_bank_filter='cooperative_only')
    keys = {(e['verification_id'], e['seed_index']) for e in s['entries']}
    assert keys == {('c1only', 1), ('crep', 1), ('crep', 2)}
    assert all(e['cooperative'] and e['cooperative_status'] == 'cooperative' for e in s['entries'])
    assert len(pairs) == 3*D
    f = s['imitation_filter']
    assert f['filter'] == 'cooperative_only' and f['excluded_examples'] == 3 and not f['empty_bank']
    assert f['excluded_by_status'] == {'not_cooperative': 1, 'no_solos': 2}
    assert f['imitated_cooperative_by_seed'] == {'1': 2, '2': 1}
    assert f['imitated_cooperative_examples'] == 3 and f['imitated_pairs'] == len(pairs)
    # oversampling is a no-op: no non-cooperative samples remain
    assert s['exposure']['duplicated_pairs'] == 0 and s['exposure']['non_cooperative_samples'] == 0
    assert s['exposure']['cooperative_share_of_samples'] == 1.
    assert 'no-op' in s['exposure']['oversampling']
    assert s['counts']['per_seed_v1']['cooperative_by_seed'] == {'1': 2, '2': 1}
    # excluded examples stay in the bank records (confirmations) untouched
    conf = rb.confirmations(tmp_path, 'B')
    assert {e['cooperative_status'] for e in conf['nocoop']['per_seed_v1']['examples']} == {'no_solos'}
    # the same key applies to R
    make(tmp_path, 'c1only', win(), win(), arm='R', syn1=noisy(3., .5, 104), syn2=noisy(0., 1., 105))
    _, sr = rb.bank_pairs(tmp_path, 'R', 12345, imitation_bank_filter='cooperative_only')
    assert len(sr['entries']) == 1 and sr['imitation_filter']['filter'] == 'cooperative_only'


def test_r_lag_cap_counts_examples_after_the_filter(tmp_path, ps):
    mixed_bank(tmp_path)
    pairs, s = rb.bank_pairs(tmp_path, 'B', 12345, max_decisions=2, imitation_bank_filter='cooperative_only')
    assert len(s['entries']) == 2 and all(e['cooperative'] for e in s['entries'])
    assert s['counts']['admitted_excess_not_used'] == 1 and s['counts']['available_admitted'] == 3
    assert rb.imitation_decision_cap(dict(exact_imitation=dict(entries=s['entries']))) == 2
    # unfiltered, non-cooperative examples count toward availability
    _, u = rb.bank_pairs(tmp_path, 'B', 12345, max_decisions=2)
    assert u['counts']['available_admitted'] == 6


def test_empty_filtered_bank_skips_imitation_cleanly(tmp_path, ps):
    make(tmp_path, 'nocoop', win(), win(), cooperation=False)
    pairs, s = rb.bank_pairs(tmp_path, 'B', 12345, imitation_bank_filter='cooperative_only')
    assert pairs == [] and s['entries'] == []
    assert s['imitation_filter']['empty_bank'] is True and s['imitation_filter']['excluded_examples'] == 2
    assert s['exposure']['cooperative_entries'] == 0
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    report = bi.run_separate(policy, optimizer, pairs, torch.Generator().manual_seed(0), Policy(), passes=5)
    assert report['stop_reason'] == 'empty statistically qualified bank' and report['optimizer_steps'] == 0
    assert report['epochs_completed'] == 0 and not report['imitation_failed']
    assert all(torch.equal(before[k], v) for k, v in policy.state_dict().items())


def test_empty_bank_with_an_intent_policy_does_not_crash():
    from tools.coopmwm import v2v_intent as vi
    policy = vi.IntentPolicy()
    assert vi.refresh_bank_intents(policy, []) is not None
    assert vi.bank_stats(policy, [])['pairs'] == 0
    assert vi.coverage([], policy.intent_horizon, policy.intent_k)['partner_slots_present'] == 0
    vi.weight_norms(policy)


# ---------------------------------------------------------------- launcher / runtime / gate

def _cfg(tmp_path, **kw):
    cfg = v2v_runner.prepare_intent(dict(budget={}), root=tmp_path/'v2v', legacy_root=tmp_path/'legacy',
                                    source_root=tmp_path/'src', common_checkpoint_sha256='a'*64, **kw)
    live = dict(cfg, launch_authorized=True, phase_funding_authorized=True,
                budget=dict(total=launch.CEILING_SIM_S, phases=dict(launch.V2V_PHASES)))
    return cfg, live


def test_runtime_pins_cooperative_only_and_refuses_everything_else(tmp_path):
    assert launch.BANK_OPTIONS['imitation_bank_filter'] == 'cooperative_only'
    cfg, live = _cfg(tmp_path)
    assert cfg['imitation_bank_filter'] == 'cooperative_only'
    assert launch.bind_runtime(dict(config=live), root=tmp_path/'v2v', source_root=tmp_path/'src') is live
    for bad in (dict(imitation_bank_filter='all_admitted'), dict(imitation_bank_filter=None),
                dict(imitation_bank_filter='bogus'),
                dict(imitation_bank_filter_by_arm=dict(B='cooperative_only', R='all_admitted')),
                dict(imitation_bank_filter_by_arm=dict(B='all_admitted', R='all_admitted'))):
        with pytest.raises(launch.Refused):
            launch.bind_runtime(dict(config=dict(live, **bad)), root=tmp_path/'v2v', source_root=tmp_path/'src')
    ok = dict(live, imitation_bank_filter_by_arm=dict(B='cooperative_only', R='cooperative_only'))
    assert launch.bind_runtime(dict(config=ok), root=tmp_path/'v2v', source_root=tmp_path/'src') is ok
    with pytest.raises(launch.Refused):
        launch.imitation_filter_for(live, 'A')
    with pytest.raises(ValueError):                       # producer refuses a non-pinned value
        _cfg(tmp_path, imitation_bank_filter='all_admitted')


def test_hooks_apply_the_same_filter_to_b_and_r_and_refuse_mismatch(tmp_path, monkeypatch):
    from tools.coopmwm import v2v_live
    seen = []
    monkeypatch.setattr(bs, 'install', lambda rb_, bank_rule='boot_v2', sources=None: None)
    monkeypatch.setattr(rb, 'bank_pairs',
                        lambda root, arm, seed, **k: seen.append((arm, k.get('imitation_bank_filter'))))
    h = v2v_live.hooks(tmp_path, dict(bank_statistics='per_seed_v1', imitation_bank_filter='cooperative_only'))
    h['bank_pairs'](tmp_path, 'B', 1)
    h['bank_pairs'](tmp_path, 'R', 1, v2v=True)
    assert seen == [('B', 'cooperative_only'), ('R', 'cooperative_only')]
    seen.clear()
    h = v2v_live.hooks(tmp_path, dict(bank_statistics='per_seed_v1'))     # default: no kwarg added
    h['bank_pairs'](tmp_path, 'B', 1)
    assert seen == [('B', None)]
    with pytest.raises(launch.Refused):
        v2v_live.hooks(tmp_path, dict(bank_statistics='per_seed_v1', imitation_bank_filter='cooperative_only',
                                      imitation_bank_filter_by_arm=dict(B='cooperative_only', R='all_admitted')))


def test_gate_config_and_bank_use_the_runtime_filter(tmp_path, monkeypatch):
    cfg, _ = _cfg(tmp_path)
    assert gate.runtime_gate_config(cfg) is not None
    with pytest.raises(gate.GateIdentityError, match='imitation_bank_filter'):
        gate.runtime_gate_config(dict(cfg, imitation_bank_filter='all_admitted'))
    stripped = {k: v for k, v in cfg.items() if k != 'imitation_bank_filter'}
    with pytest.raises(gate.GateIdentityError, match='imitation_bank_filter'):
        gate.runtime_gate_config(stripped)
    # the gate's offline bank is built with the runtime's filter
    seen = {}
    monkeypatch.setattr(gate, 'gate_identity', lambda *a, **k: (
        None, dict(commit='c', runtime_sha256='r', source_manifest_sha256=None,
                   imitation_bank_filter='cooperative_only'), dict(replay_sources={})))

    def fake(root, arm, seed, sources=None, config=None, **kw):
        seen.update(kw)
        raise gate.bank_statistics.BankDataMissing('stop')
    monkeypatch.setattr(gate, 'statistical_bank_pairs', fake)
    (tmp_path/'ctl').mkdir()
    out = gate.plan(tmp_path, 'B', tmp_path/'ctl')
    assert seen == dict(imitation_bank_filter='cooperative_only') and out is not None


def test_statistical_bank_pairs_forwards_the_filter(tmp_path, monkeypatch):
    got = {}
    monkeypatch.setattr(gate.rb, 'bank_pairs', lambda root, arm, seed, **kw: got.update(kw) or ([], {}))
    monkeypatch.setattr(gate, 'bank_rule', lambda sources: contextlib.nullcontext())
    gate.statistical_bank_pairs(tmp_path, 'B', 1, imitation_bank_filter='cooperative_only')
    assert got['imitation_bank_filter'] == 'cooperative_only'
    got.clear()
    gate.statistical_bank_pairs(tmp_path, 'B', 1, imitation_bank_filter='all_admitted')
    assert 'imitation_bank_filter' not in got
