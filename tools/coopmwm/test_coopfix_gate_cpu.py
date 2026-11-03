"""CPU tests for the COOPFIX GATE (tools/coopmwm/coopfix_gate.py).

Offline only: a synthetic MAPPO u0 checkpoint (accel_learner schema), a synthetic bank of
pairs modelled like the live one (plan actions = u0 means + noise + a small consistent
cooperative delta, with v2v partner features), and the registered bank_imitation loop.
No SUMO, no GPU, no writes outside tmp_path. The v2v flag default-off path is asserted
byte-identical.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

from tools.coopnative.test_cpu import rows
from tools.coopmwm import coopfix_gate as gate
from tools.coopmwm import v2v_model
from tools.coopmwm.accel_learner import save
from tools.coopmwm.acceleration import Policy, encode
from tools.coopmwm.v2v_obs import V2V_DIM

COOP_MEMBERS = ['m0', 'm1']
SOLO_MEMBER = 'm0'
SIGNAL, NOISE = 0.03, 0.01
READY_FILES_DIR = gate.READY_FILES['imitation_trust'].parent


def u0_checkpoint(tmp_path, lr=1e-4, seed=5):
    """A trained-u0 stand-in: random weights with a NON-zero action head (the registered
    zero-init head would block all gradient flow on the first optimizer step)."""
    torch.manual_seed(seed)
    policy = Policy()
    with torch.no_grad():
        policy.actor[-1].weight.normal_(0, .5)
        policy.actor[-1].bias.normal_(0, .5)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    path = tmp_path/'B_checkpoint_000.pt'
    save(path, policy, optimizer, torch.Generator().manual_seed(9), 0, {})
    return path, policy


def diverse_context(rng, members, v2v=True):
    """A whitelisted sensor context (3-5 CAVs, random lanes/positions/speeds); the v2v block
    carries partner intent (each plan member's block contains the partner's intended delta
    in the last-command channel - the feature the gate says the policy must learn to use)."""
    n = int(rng.integers(3, 6))
    rows_out = [dict(vid=f'v{i}', s_m=float(rng.uniform(-100, 70)), v_mps=float(rng.uniform(6, 24)),
                     lane=int(rng.integers(0, 3)), is_cav=True, length_m=5.) for i in range(n)]
    for i, m in enumerate(members):
        rows_out[i]['vid'] = m
    context = encode(rows_out)
    if v2v:
        block = {}
        for ego in context['members']:
            hint = ((.3 if ego == members[0] else -.3) if len(members) > 1 else 0.)
            out = [float(rng.normal()), 0., float(rng.normal()), float(rng.normal()), hint, 1.]
            block[ego] = out+[0.]*(V2V_DIM-len(out))
        context['v2v'] = block
    return context


def build_bank(tmp_path, coop_entries=6, solo=40, v2v=True, lr=1e-4, signal=SIGNAL, noise=NOISE):
    """Pairs modelled like the live bank: every plan action = the u0 mean for that context
    row + per-pair execution noise + (cooperative pairs only) a small consistent
    cooperative delta on each plan member; flattened in the oversampled order of
    replay_bank.bank_pairs (other + coop*(1+full) + coop[:rem])."""
    path, _ = u0_checkpoint(tmp_path, lr=lr)
    reference = gate.load_gate_policy(path, V2V_DIM)[0]

    def make(plan_id, start_s, members):
        rng = np.random.default_rng(start_s)
        context = diverse_context(rng, members, v2v)
        means, _ = reference(dict(context, v2v={}))
        cooperative = len(members) > 1
        action = []
        for i, m in enumerate(context['members']):
            delta = signal*(1 if m == members[0] else -1) if cooperative and m in members else 0.
            action.append([float(means[i, 0].detach())+delta+float(rng.normal())*noise])
        sample = dict(context=context, action=action,
                      behaviour=dict(route='plan', executed=True, plan_start=False,
                                     proposal_members=list(members), plan_id=plan_id),
                      plan_executions=[dict(plan_id=plan_id, proposal_members=list(members),
                                            start_s=start_s)],
                      previous_noise=[[0.]]*len(members), deterministic=True, rho=.5, window=1,
                      logp=0., value=0.)
        return sample, 1.
    coop = [make(f'p{k}', 400+2*k, COOP_MEMBERS) for k in range(coop_entries) for _ in range(2)]
    other = [make(f's{k}', 500+k, [SOLO_MEMBER]) for k in range(solo)]
    raw = 2*coop_entries
    full, rem = divmod(max(0, solo-raw), raw)
    pairs = other+coop*(1+full)+coop[:rem]
    summary = dict(
        entries=[dict(plan_id=f'p{k}', duration_s=2, cooperative=True, start_s=400+2*k)
                 for k in range(coop_entries)]
        +[dict(plan_id=f's{k}', duration_s=1, cooperative=False, start_s=500+k) for k in range(solo)],
        exposure=dict(cooperative_samples_raw=raw, non_cooperative_samples=solo,
                      duplicated_pairs=len(pairs)-solo-raw),
        counts=dict(used_admitted=coop_entries+solo, cooperative=coop_entries))
    return path, pairs, summary


def synthetic_pairs(n_coop_entries=2, n_solo=10, v2v=True):
    """Small fixed-rows bank for the metric helpers (no optimizer involved)."""
    coop = [make_metric_pair(f'p{k}', 400+2*k, COOP_MEMBERS, v2v)
            for k in range(n_coop_entries) for _ in range(2)]
    other = [make_metric_pair(f's{k}', 500+k, [SOLO_MEMBER], v2v) for k in range(n_solo)]
    raw = 2*n_coop_entries
    full, rem = divmod(max(0, n_solo-raw), raw) if raw else (0, 0)
    pairs = other+coop*(1+full)+coop[:rem]
    summary = dict(
        entries=[dict(plan_id=f'p{k}', duration_s=2, cooperative=True, start_s=400+2*k)
                 for k in range(n_coop_entries)]
        +[dict(plan_id=f's{k}', duration_s=1, cooperative=False, start_s=500+k)
          for k in range(n_solo)],
        exposure=dict(cooperative_samples_raw=raw, non_cooperative_samples=n_solo,
                      duplicated_pairs=len(pairs)-n_solo-raw),
        counts=dict(used_admitted=n_coop_entries+n_solo, cooperative=n_coop_entries))
    return pairs, summary


def make_metric_pair(plan_id, start_s, members, v2v=True):
    context = encode(rows())
    proposal = list(context['members'])[:2] if len(members) > 1 else [context['members'][-1]]
    targets = dict(zip(proposal, [[0.4], [-0.3]])) if len(members) > 1 else {proposal[0]: [-0.1]}
    if v2v:
        plan = dict(zip(proposal, [0.4, -0.3]))
        block = {}
        for ego in context['members']:
            partner = next((m for m in proposal if m != ego), members[0])
            out = [float(np.random.default_rng(start_s).normal()), 0., 1., -1.,
                   float(plan.get(partner, 0.)), 1.]
            block[ego] = out+[0.]*(V2V_DIM-len(out))
        context['v2v'] = block
    sample = dict(context=context,
                  action=[targets.get(m, [0.05]) for m in context['members']],
                  behaviour=dict(route='plan', executed=True, plan_start=False,
                                 proposal_members=proposal, plan_id=plan_id),
                  plan_executions=[dict(plan_id=plan_id, proposal_members=proposal,
                                        start_s=start_s)],
                  previous_noise=[[0.]]*len(members), deterministic=True, rho=.5, window=1,
                  logp=0., value=0.)
    return sample, 1.


def widened_block(policy, head):
    extra = policy.v2v_width()
    return getattr(policy, head)[0].weight.detach()[:, 29:29+extra]


def test_gate_policy_extra_zero_forward_is_byte_identical():
    torch.manual_seed(11)
    base, gated = Policy(), gate.GatePolicy()
    gated.load_state_dict(base.state_dict())
    assert gated.v2v_width() == 0
    context = diverse_context(np.random.default_rng(1), COOP_MEMBERS, v2v=True)
    for ctx in (context, dict(context, v2v={})):            # flag off ignores any v2v payload
        m1, v1 = base(ctx)
        m2, v2 = gated(ctx)
        assert torch.equal(m1, m2) and torch.equal(v1, v2)


def test_extended_policy_without_v2v_context_matches_base_exactly():
    torch.manual_seed(11)
    base = Policy()
    gated = gate.GatePolicy()
    gated.load_state_dict(base.state_dict())
    v2v_model.extend_for_v2v(gated, V2V_DIM)
    context = diverse_context(np.random.default_rng(2), COOP_MEMBERS, v2v=False)
    m1, v1 = base(context)
    m2, v2 = gated(context)
    assert torch.equal(m1, m2) and torch.equal(v1, v2)          # zero-init: any input, same output
    wild = dict(context, v2v={m: [9e3]*V2V_DIM for m in context['members']})
    m3, v3 = gated(wild)
    assert torch.equal(m1, m3) and torch.equal(v1, v3)


def test_load_gate_policy_zero_init_round_trip(tmp_path):
    path, base = u0_checkpoint(tmp_path)
    policy, optimizer, generator, payload = gate.load_gate_policy(path, V2V_DIM)
    assert isinstance(policy, gate.GatePolicy)
    assert policy.v2v_width() == V2V_DIM
    for head in ('actor', 'critic'):
        assert torch.equal(widened_block(policy, head), torch.zeros_like(widened_block(policy, head)))
    context = diverse_context(np.random.default_rng(3), COOP_MEMBERS, v2v=True)
    m0, v0 = base(context)
    m1, v1 = policy(context)
    assert torch.equal(m0, m1) and torch.equal(v0, v1)          # warm start is output-identical
    assert generator.get_state().numel() > 0 and payload['update'] == 0


def test_synthetic_pairs_and_summary_shapes():
    pairs, summary = synthetic_pairs()
    assert len(pairs) == 10+4+4+2 and summary['exposure']['duplicated_pairs'] == 6
    coop, raw = gate.cooperative_pairs(pairs, summary)
    assert raw == 4 and len(coop) == 4
    assert all(gate.is_cooperative(s) for s, _ in coop)
    assert all(not gate.is_cooperative(s) for s, _ in pairs[:10])
    entries, coop2, raw2 = gate.cooperative_entries(pairs, summary)
    assert raw2 == 4 and [len(e) for e in entries] == [2, 2]
    assert all(e[0][0]['plan_executions'][0]['plan_id'] == f'p{k}' for k, e in enumerate(entries))
    # without a summary: consecutive (plan_id, start_s) runs group the duplicated blocks apart
    entries2, coop3, raw3 = gate.cooperative_entries(pairs, None)
    assert raw3 is None and len(coop3) == 10
    assert [len(e) for e in entries2] == [2]*5


class FixedPolicy:
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, context):
        mean = torch.zeros((len(context['members']), 1))
        for i, m in enumerate(context['members']):
            if m in ('downstream', 'ramp'):
                mean[i] = self.shift.get(m, 0.)
        return mean, torch.zeros(())


def test_entry_toward_fractions_exact():
    pairs, _ = synthetic_pairs(n_coop_entries=1, n_solo=2)
    entries, coop, _ = gate.cooperative_entries(pairs, None)
    before = FixedPolicy({'downstream': 0.9, 'ramp': -0.9})     # both start away from the plan
    after = FixedPolicy({'downstream': 0.3, 'ramp': -0.9})      # downstream toward .4, ramp not
    report = gate.entry_toward_fractions(before, after, entries)
    assert len(report) == 1 and report[0]['member_steps'] == 4
    assert report[0]['toward_steps'] == 2 and report[0]['toward_fraction'] == .5
    assert gate.entry_toward_fractions(after, before, entries)[0]['toward_steps'] == 0   # reversed: all away


def test_coop_fall_bootstrap_direction():
    pairs, _ = synthetic_pairs(n_coop_entries=1, n_solo=2)
    entries, coop, _ = gate.cooperative_entries(pairs, None)

    class Means:
        def __init__(self, err):
            self.err = err

        def __call__(self, context):
            mean = torch.zeros((len(context['members']), 1))
            for i, m in enumerate(context['members']):
                mean[i] = (0.4 if m == 'downstream' else -0.3 if m == 'ramp' else 0.)+self.err
            return mean, torch.zeros(())

    mean, ci, n = gate.coop_fall_bootstrap(Means(.2), Means(0.), coop)
    assert n == 2 and mean == pytest.approx(-.2) and ci[1] < 0
    mean, ci, n = gate.coop_fall_bootstrap(Means(0.), Means(.2), coop)
    assert mean == pytest.approx(.2) and ci[0] > 0              # wrong direction never passes


def test_run_gate_passes_on_cooperative_bank(tmp_path):
    path, pairs, summary = build_bank(tmp_path)
    report = gate.run_gate(path, pairs, summary, dict(v2v=True, passes=5))
    assert report['verdict'] == 'PASS', report['reasons']
    assert report['imitation']['passes_completed'] == 5
    assert report['imitation']['kl'] <= report['imitation']['kl_limit']
    coop = report['cooperative']
    assert coop['error_after_mps2'] < coop['error_before_mps2']
    assert coop['fall_ci95'][1] < 0. and coop['fall_mean_mps2'] < 0.
    fractions = [e['toward_fraction'] for e in coop['entries']]
    assert sum(fractions)/len(fractions) > .5
    assert report['v2v_weights']['moved'] is True
    assert report['v2v_weights']['heads']['actor']['nonzero'] is True
    assert report['bank']['v2v_feature_fraction'] == 1.
    assert report['bank']['cooperative_pairs'] == 12
    assert report['options']['status']['imitation_trust']['ready_file'].endswith('IMITFIX_READY.json')
    json.loads(json.dumps(report, default=float))               # report is JSON-clean


def test_run_gate_fails_when_pairs_carry_no_v2v_features(tmp_path):
    path, pairs, summary = build_bank(tmp_path, coop_entries=2, solo=10, v2v=False)
    report = gate.run_gate(path, pairs, summary, dict(v2v=True, passes=5))
    assert report['verdict'] == 'FAIL' and 'v2v_weights_moved' in report['reasons']
    assert report['v2v_weights']['moved'] is False
    assert report['bank']['v2v_feature_fraction'] == 0.


def test_run_gate_fails_with_v2v_flag_off(tmp_path):
    path, pairs, summary = build_bank(tmp_path, coop_entries=2, solo=10)
    report = gate.run_gate(path, pairs, summary, dict(v2v=False, passes=5))
    assert report['verdict'] == 'FAIL' and 'v2v_weights_moved' in report['reasons']
    assert report['v2v_weights']['extra_dim'] == 0


def test_run_gate_fails_on_kl_breach(tmp_path):
    path, pairs, summary = build_bank(tmp_path, coop_entries=2, solo=10, lr=2.0)
    report = gate.run_gate(path, pairs, summary, dict(v2v=True, passes=5))
    assert report['verdict'] == 'FAIL'
    assert not report['checks']['full_pass'] or not report['checks']['kl_within_limit']
    assert report['imitation']['stop_reason'] and 'complete' not in report['imitation']['stop_reason']


def test_option_stubs_identity_and_ready_gating(monkeypatch, tmp_path):
    absent = tmp_path/'absent.json'
    monkeypatch.setitem(gate.READY_FILES, 'imitation_trust', absent)
    monkeypatch.setitem(gate.READY_FILES, 'bank_filter', absent)
    pairs, summary = synthetic_pairs(n_solo=3)
    opts = gate.normalize_options(None)
    assert opts['v2v'] is False and opts['passes'] == 5 and opts['coefficient'] == 1.
    stage = gate.filter_report(opts, summary)
    assert stage['enabled'] is False and stage['stage'].startswith('identity')
    kwargs, report = gate.trust_kwargs(opts)
    assert kwargs == {} and report['enabled'] is False
    status = gate.option_status()
    assert set(status) == {'imitation_trust', 'bank_filter'}
    with pytest.raises(ValueError):
        gate.normalize_options(dict(bank_filter='unregistered'))
    with pytest.raises(ValueError):
        gate.normalize_options(dict(imitation_trust='unregistered'))
    with pytest.raises(ValueError):
        gate.filter_report(dict(bank_filter='statistical'), summary)
    with pytest.raises(ValueError):
        gate.trust_kwargs(dict(imitation_trust='own_pass'))
    ready = tmp_path/'BANKFILT_READY.json'
    ready.write_text(json.dumps(dict(branch='c4/bank-stat-filter')))
    monkeypatch.setitem(gate.READY_FILES, 'bank_filter', ready)
    applied = dict(mode='statistical', kept=1, dropped=1)
    stage = gate.filter_report(dict(bank_filter='statistical'), dict(summary, bank_filter=applied))
    assert stage['active'] == 'ready' and stage['report'] == applied
    with pytest.raises(ValueError):                          # a bare pairs file cannot apply the rule
        gate.filter_report(dict(bank_filter='statistical'), summary)


def test_imitation_trust_own_pass_is_forwarded_with_cooperative_flags(monkeypatch):
    absent = Path('/nonexistent/IMITFIX_READY.json')
    monkeypatch.setitem(gate.READY_FILES, 'imitation_trust', absent)
    with pytest.raises(ValueError):
        gate.trust_kwargs(dict(imitation_trust='own_pass'), [], None)
    monkeypatch.setitem(gate.READY_FILES, 'imitation_trust', READY_FILES_DIR/'IMITFIX_READY.json')
    pairs, summary = synthetic_pairs(n_coop_entries=2, n_solo=4)
    kwargs, report = gate.trust_kwargs(dict(imitation_trust='own_pass'), pairs, summary)
    assert report['enabled'] is True and kwargs['imitation_trust'] == 'own_pass'
    assert kwargs.get('cooperative') is None                 # no pair flags in this summary
    flags = [gate.is_cooperative(s) for s, _ in pairs]
    summary['pair_cooperative'] = flags
    kwargs, report = gate.trust_kwargs(dict(imitation_trust='own_pass'), pairs, summary)
    assert kwargs['cooperative'] == flags and report['cooperative_flags'] is True
    kwargs, report = gate.trust_kwargs(dict(imitation_trust='own_pass'), pairs[:-1], summary)
    assert kwargs.get('cooperative') is None                 # misaligned flags are not forwarded


def test_run_gate_with_own_pass_trust_region(tmp_path):
    path, pairs, summary = build_bank(tmp_path)
    summary['pair_cooperative'] = [gate.is_cooperative(s) for s, _ in pairs]
    report = gate.run_gate(path, pairs, summary, dict(v2v=True, passes=5, imitation_trust='own_pass'))
    assert report['verdict'] == 'PASS', report['reasons']
    assert report['imitation']['passes_completed'] == 5
    assert report['imitation']['kl'] <= report['imitation']['kl_limit']
    assert report['trust']['enabled'] is True and report['trust']['cooperative_flags'] is True
    coop = report['cooperative']
    assert coop['fall_ci95'][1] < 0. and coop['fall_mean_mps2'] < 0.
    assert report['v2v_weights']['moved'] is True


def test_bank_filter_option_is_forwarded_to_bank_pairs(monkeypatch, tmp_path):
    seen = {}

    def fake_bank_pairs(root, arm, seed, beta=1., max_decisions=None, v2v=True, v2v_cfg=None,
                        bank_filter=None, pair_flags=False):
        seen.update(bank_filter=bank_filter, v2v=v2v, pair_flags=pair_flags)
        pairs, base = synthetic_pairs(n_coop_entries=1, n_solo=2)
        base['bank_filter'] = dict(mode='statistical', kept=2, dropped=1)
        return pairs, base
    from tools.coopmwm import replay_bank
    monkeypatch.setattr(replay_bank, 'bank_pairs', fake_bank_pairs)
    pairs, summary = gate.bank_inputs('root', 'B', 7, bank_filter='statistical')
    assert seen == dict(bank_filter='statistical', v2v=True, pair_flags=False) and len(pairs) == 4
    path, _ = u0_checkpoint(tmp_path)
    report = gate.run_gate(path, pairs, summary, dict(v2v=True, passes=5, bank_filter='statistical'))
    assert report['filter']['enabled'] is True and report['filter']['report']['mode'] == 'statistical'


def test_cli_run_writes_report(tmp_path):
    path, pairs, summary = build_bank(tmp_path, coop_entries=2, solo=20)
    pf = tmp_path/'pairs.json'
    pf.write_text(json.dumps(dict(pairs=[[s, w] for s, w in pairs], summary=summary)))
    out = tmp_path/'gate_report.json'
    report = gate.main(['run', '--checkpoint', str(path), '--pairs-file', str(pf),
                        '--v2v', '--passes', '5', '--out', str(out)])
    saved = json.loads(out.read_text())
    assert saved['verdict'] == report['verdict'] == 'PASS'
    assert saved['imitation']['passes_completed'] == 5
    flag_off = tmp_path/'gate_report_off.json'
    off = gate.main(['run', '--checkpoint', str(path), '--pairs-file', str(pf),
                     '--passes', '5', '--out', str(flag_off)])
    assert off['verdict'] == 'FAIL' and 'v2v_weights_moved' in off['reasons']
    status = gate.main(['status'])
    assert status['bank_filter']['ready_file'].endswith('BANKFILT_READY.json')


def test_run_gate_reports_insufficient_coop_clusters(tmp_path):
    path, pairs, summary = build_bank(tmp_path, coop_entries=1, solo=6)
    one = [p for p in pairs if gate.is_cooperative(p[0])][:1]
    pairs = [p for p in pairs if not gate.is_cooperative(p[0])]+one
    report = gate.run_gate(path, pairs, None, dict(v2v=True, passes=5))
    assert report['verdict'] == 'FAIL' and 'cooperative_error_falls' in report['reasons']
    assert report['cooperative']['fall_ci95'] is None and report['cooperative']['fall_clusters'] == 1
