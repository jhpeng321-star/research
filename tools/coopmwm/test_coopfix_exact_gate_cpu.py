"""CPU tests for the exact gate runner (synthetic banks and synthetic job outputs, no SUMO):
the offline fixed update through bank_imitation.run_separate on the live V2V loader, the
cooperative-pair slice of the bank layout, the v2v / intent block layouts, per_seed_v1
certification (per seed; replicated / seed1_only / data_missing; all8 per-seed plans), per-arm
sources, planning on each certified seed's own CRN with u3 references, the single-seed
per_seed_v1 paired-bootstrap synergy (>= 1 certified seed must pass on 986066), the ledger
precharge / admission / running hooks of execute, and the PASS/FAIL verdict."""
import json

import numpy as np
import pytest
import torch

from tools.coopmwm import bank_imitation, bank_statistics as bs
from tools.coopmwm import coopfix_exact_gate as ceg
from tools.coopmwm import policy_synergy as ps, replay_bank as rb, sumo_verify as sv, v2v_obs
from tools.coopmwm.accel_learner import save as ckpt_save
from tools.coopmwm.accel_storage import write as awrite
from tools.coopmwm.acceleration import Policy, encode
from tools.coopmwm.exact_grade import _seal, _write_new, verify_sealed
from tools.coopmwm.protocol import digest
from tools.coopmwm.v2v_model import extend_for_v2v, FEATURES
from tools.coopmwm.v2v_obs import V2VConfig, v2v_features
from .test_replay_bank_confirm_cpu import snapshot, plan as group_plan, fp, SumRunner

MANDATORY_COPY = tuple(ceg.MANDATORY_STATES)


@pytest.fixture(autouse=True)
def bank_budget(tmp_path):
    """Same authorised additive binding as the confirmation-stage tests (never the shared
    historical quote)."""
    from tools.coopmwm.warm_protocol import budget
    b = budget()
    b['total'] += 2600000
    b['allocated'] += 2600000
    b['phases'] = dict(b['phases'], bank_verification_ext=2600000)
    (tmp_path/'B_REPLAY_PHASE_RECONCILIATION_0627.json').write_text(json.dumps(
        dict(budget=b, amendment_sha256='test-bank-budget')))
    (tmp_path/'AUTHORITY.md').write_text('test-bank-budget')


@pytest.fixture(autouse=True)
def mandatory_states(monkeypatch):
    caps = (ceg.MAX_GROUP, ceg.ISOLATED_STATE_CAP_SIM_S, ceg.MAX_WINDOW_END_S)
    monkeypatch.setattr(rb, 'confirmations', rb.confirmations)     # restored after each test
    ceg.MANDATORY_STATES = (986066, 986048, 986003)
    yield
    ceg.MANDATORY_STATES = MANDATORY_COPY
    ceg.MAX_GROUP, ceg.ISOLATED_STATE_CAP_SIM_S, ceg.MAX_WINDOW_END_S = caps


# ---------------------------------------------------------------- offline fixed update

def synth_rows():
    return [dict(vid='cav0', s_m=-30., v_mps=8., lane=0, is_cav=True, length_m=5.),
            dict(vid='cav1', s_m=-12., v_mps=9.5, lane=0, is_cav=True, length_m=5.),
            dict(vid='cav2', s_m=25., v_mps=10., lane=1, is_cav=True, length_m=5.),
            dict(vid='hdv9', s_m=-50., v_mps=7., lane=2, is_cav=False, length_m=5.)]


def blocks_for(state, members, present=True):
    cfg = V2VConfig(v2v=True)
    table = {row['vid']: row for row in state}
    out = {}
    for vid in members:
        row = [float(t) for t in v2v_features(table, vid, {'cav0': .4, 'cav1': -1.2, 'cav2': 0.}, cfg)]
        if not present:                       # carry the block, no partner present
            row = [0.]*len(row)
        out[vid] = row
    return out


def pair(members=('cav0', 'cav1'), blocks=None, action=(.3, .5, .1), executed=True):
    context = encode(synth_rows())
    if blocks is not None:
        context['v2v'] = blocks
    return dict(context=context, action=[[a] for a in action],
                behaviour=dict(proposal_members=list(members), executed=executed))


def fixture_checkpoint(path):
    torch.set_num_threads(1)
    torch.manual_seed(47)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    mean, value = policy(encode(synth_rows()))
    (mean.sum()+value.sum()).backward()
    optimizer.step()
    ckpt_save(path, policy, optimizer, torch.Generator().manual_seed(9), 0, {'charged': 1})
    return policy


def test_fixed_update_runs_run_separate_on_the_live_v2v_loader(tmp_path, monkeypatch):
    ckpt = tmp_path/'B_checkpoint_000.pt'
    fixture_checkpoint(ckpt)
    present = blocks_for(synth_rows(), ['cav0', 'cav1', 'cav2'])
    absent = blocks_for(synth_rows(), ['cav0', 'cav1', 'cav2'], present=False)
    other = [(pair(members=('cav0',), blocks=present), 1.), (pair(blocks=absent), 1.),
             (pair(blocks=present, executed=False), 1.)]
    coop = [(pair(members=('cav0', 'cav2'), blocks=present, action=(.9, .1, .8)), 1.)]
    pairs = other+coop*3                                    # bank layout: other + coop*dup
    summary = dict(exposure=dict(cooperative_samples_raw=1, non_cooperative_samples=3))
    calls = []
    real = bank_imitation.run_separate

    def spy(*args, **kwargs):
        calls.append(type(args[0]).__module__)
        return real(*args, **kwargs)
    monkeypatch.setattr(bank_imitation, 'run_separate', spy)
    monkeypatch.setattr(bank_imitation, 'run', lambda *a, **k: pytest.fail('legacy run used'))
    state, offline = ceg.fixed_update(ckpt, pairs, summary, coefficient=.01, passes=5, intent='off')
    assert calls == ['tools.coopmwm.v2v_runtime']          # live V2V policy class, one run
    assert offline['imitation_runner'] == 'bank_imitation.run_separate'
    assert offline['config'] == dict(v2v=True, v2v_dim=24)
    assert int(state['actor.0.weight'].shape[1]) == 29+4+24 and offline['v2v_dim'] == 24
    imitation = offline['imitation']
    assert imitation['pairs_filtered_unexecuted'] == 1 and imitation['imitation_failed'] is False
    assert imitation['imitation_error'] is None
    assert isinstance(imitation['pass1_effective'], bool) and isinstance(imitation['backtracks'], int)
    assert len(imitation['epochs']) >= 1 and imitation['epochs'][0]['epoch'] == 1
    assert offline['passes']['requested'] == 5 and offline['passes']['completed'] >= 1
    assert set(offline['checks']) == {'full_pass', 'kl_within_limit', 'pass1_effective', 'imitation_ok',
                                      'cooperative_error_falls', 'partner_intent_weights_moved'}
    assert offline['checks']['pass1_effective'] == bool(imitation['pass1_effective']
                                                        and imitation['pass1_loss_drop'] > 0)
    # unique cooperative slice of the layout, not the structural multi-CAV rule
    assert offline['cooperative_pairs']['count'] == 1
    assert offline['cooperative_pairs']['slice'] == [3, 4]
    # intent pairs fall back to partner presence on the 24-wide layout
    assert offline['intent_pairs']['count'] == 5
    weights = offline['partner_intent_weights']
    assert weights['required'] == 'v2v columns' and weights['moved'] is True
    assert weights['heads']['actor']['intent']['columns'] == 0


def test_fixed_update_refuses_pairs_without_v2v_blocks(tmp_path):
    ckpt = tmp_path/'B_checkpoint_000.pt'
    fixture_checkpoint(ckpt)
    with pytest.raises(ValueError, match='lack recorded v2v'):
        ceg.fixed_update(ckpt, [(pair(), 1.)], None, intent='off')


def test_offline_checks_require_effective_pass1_and_no_failure():
    def offline(ci95=(-2., -1.), moved=True, completed=5, kl=.001, effective=True, drop=.3, failed=False):
        return dict(passes=dict(completed=completed), kl=kl, kl_limit=.045,
                    imitation=dict(pass1_effective=effective, pass1_loss_drop=drop, imitation_failed=failed),
                    cooperative_pairs=dict(change_ci95=list(ci95)),
                    partner_intent_weights=dict(moved=moved))
    assert all(ceg.offline_checks(offline()).values())
    assert not ceg.offline_checks(offline(effective=False))['pass1_effective']
    assert not ceg.offline_checks(offline(drop=0.))['pass1_effective']
    assert not ceg.offline_checks(offline(drop=None))['pass1_effective']
    assert not ceg.offline_checks(offline(failed=True))['imitation_ok']
    assert not ceg.offline_checks(offline(ci95=(-2., 0.)))['cooperative_error_falls']
    assert not ceg.offline_checks(offline(moved=False))['partner_intent_weights_moved']
    assert not ceg.offline_checks(offline(kl=.9))['kl_within_limit']
    assert not ceg.offline_checks(offline(kl=None))['kl_within_limit']
    assert not ceg.offline_checks(offline(completed=0))['full_pass']
    mean, ci, n = ceg.paired_fall_ci([1., 1., 1.], [.5, .6, .4])
    assert abs(mean+.5) < 1e-12 and ci[1] < 0 and n == 3


def test_cooperative_pair_slice_follows_the_bank_layout():
    def tag(i, members):
        return dict(id=i, behaviour=dict(proposal_members=list(members))), 1.
    other = [tag('o0', ('a', 'b')), tag('o1', ('a',)), tag('o2', ('a', 'b', 'c'))]
    coop = [tag('c0', ('a', 'b')), tag('c1', ('b', 'c'))]
    pairs = other+coop+coop+coop[:1]                         # other + coop*2 + coop[:rem]
    summary = dict(exposure=dict(non_cooperative_samples=3, cooperative_samples_raw=2))
    unique, record = ceg.coop_pairs(pairs, summary)
    assert [s['id'] for s, _ in unique] == ['c0', 'c1']
    assert record == dict(source='bank_exposure_slice', n_non_coop=3, raw=2, slice=[3, 5])
    with pytest.raises(ValueError, match='exceeds'):
        ceg.coop_pairs(pairs[:4], summary)
    fallback, rec = ceg.coop_pairs(pairs, None)              # structural rule, content-deduplicated
    assert [s['id'] for s, _ in fallback] == ['o0', 'o2', 'c0', 'c1']
    assert rec['source'] == 'structural_fallback_no_summary'


# ---------------------------------------------------------------- v2v / intent layouts

def intent_layout(monkeypatch):
    """The c4/intent-claude v2v_obs intent API (K*(6+H+1); presence index 5)."""
    width = len(v2v_obs.V2V_PARTNER_FEATURES)
    monkeypatch.setattr(v2v_obs, 'v2v_intent_dim', lambda k=4, horizon=5: int(k)*(width+int(horizon)+1),
                        raising=False)
    monkeypatch.setattr(v2v_obs, 'intent_columns', lambda k=4, horizon=5: [
        s*(width+int(horizon)+1)+width+j for s in range(int(k)) for j in range(int(horizon)+1)], raising=False)


def test_block_layouts_24_and_intent_48(monkeypatch):
    base = ceg.block_layout(24)
    assert base['stride'] == 6 and base['present'] == v2v_obs.V2V_PARTNER_FEATURES.index('present')
    assert base['intent'] == () and base['kind'] == 'v2v'
    with pytest.raises(ValueError):
        ceg.block_layout(47)
    intent_layout(monkeypatch)
    lay = ceg.block_layout(48)
    assert lay['stride'] == 12 and lay['present'] == 5 and lay['horizon'] == 5
    assert lay['intent'] == tuple(s*12+6+j for s in range(4) for j in range(6))
    assert lay['intent_valid'] == (11, 23, 35, 47)
    row = [0.]*48
    row[12+5] = 1.                                          # slot-1 partner present, no intent
    sample = dict(context=dict(v2v={'a': row}))
    assert ceg._present_bits(sample) and not ceg._present_bits(sample, 'intent_valid')
    assert ceg.intent_pairs([(sample, 1.)]) == []
    row2 = list(row)
    row2[23] = 1.                                           # slot-1 intent valid
    assert len(ceg.intent_pairs([(dict(context=dict(v2v={'a': row2})), 1.)])) == 1
    r24 = [0.]*24                                           # 24-wide rows: base presence index
    r24[6+5] = 1.
    assert ceg._present_bits(dict(context=dict(v2v={'a': r24})))


def test_weight_report_requires_intent_columns_when_present(monkeypatch):
    intent_layout(monkeypatch)
    for width in (24, 48):
        torch.manual_seed(1)
        policy = Policy()
        extend_for_v2v(policy, width)
        report = ceg.weight_report(policy)
        assert report['extra_dim'] == width and report['moved'] is False
        with torch.no_grad():
            policy.actor[0].weight[0, FEATURES+0] = .1          # a partner gap column
        report = ceg.weight_report(policy)
        assert report['v2v_columns_moved'] is True
        if width == 24:
            assert report['required'] == 'v2v columns' and report['moved'] is True
        else:
            assert report['required'] == 'intent columns' and report['moved'] is False
            assert report['heads']['actor']['intent']['columns'] == 24
            with torch.no_grad():
                policy.critic[0].weight[0, FEATURES+12+6+2] = -.2   # slot-1 intent_3
            report = ceg.weight_report(policy)
            assert report['intent_columns_moved'] is True and report['moved'] is True


def test_gate_config_feature_detects_the_intent_option():
    module = ceg.intent_module()
    assert ceg.gate_config(intent='off') == dict(v2v=True, v2v_dim=24)
    assert ceg.gate_config(v2v=False, intent='off') is None
    if module is None:
        assert ceg.gate_config(intent='auto') == dict(v2v=True, v2v_dim=24)
        with pytest.raises(ValueError, match='absent'):
            ceg.gate_config(intent='on')
    else:
        cfg = ceg.gate_config(intent='auto')
        assert cfg['v2v_intent'] is True and cfg == module.config_keys()


# ---------------------------------------------------------------- synthetic boot_v2 bank

class TapeSumRunner(SumRunner):
    """The cooperative SumRunner plant plus the branch RESULT.json one-second tape with
    per-second team_delay_vs fingerprints over the window (what boot_v2 resamples)."""

    def run(self, spec, snap):
        r = super().run(spec, snap)
        w = snap['episode']['world']['seed']
        t0, t_end = spec['window']['start_s'], spec['window']['end_s']
        base = 3.*w+.5*t0*t0/100.
        J = self.J(spec['plan'])
        prefixes = dict(r['prefixes'])
        prefixes.update({str(t): fp(t, base+J*(t-t0)/(t_end-t0)) for t in range(t0+1, t_end)})
        tape = dict(prefixes=prefixes, final=r['final'])
        awrite(self.work/'branches'/spec['verification_id']/spec['branch']/'RESULT.json', tape)
        r['result_sha256'] = digest(tape)
        return r


def overlay(root, bank, receipt, snap, runner, replicate_solos):
    """The MAIN04:26 overlay (registered confirmations_safety0426 path); with
    replicate_solos the second-seed solo verifications are run and linked as
    cooperation['replication_solo'] (bank_statistics._cooperation layout)."""
    from tools.coopmwm import bank_safety
    vid = receipt['verification_id']
    out = bank_safety.reevaluate(json.loads((bank/'confirmations'/f'{vid}.json').read_text()))
    if replicate_solos:
        rep_root = bank/'confirm_solo_replication'
        rows = []
        for m in receipt['chosen_plan']['members']:
            r = sv.verify(rb.replicate(snap, vid), [rb.solo_candidate(receipt['chosen_plan'], m)],
                          runner(rep_root), rb._confirm_charge(root, 'bank_verification_ext',
                                                               'confirm_solo_replication'),
                          receipt['proposer'], workers=2, root=rep_root)
            rows.append(dict(member=m, verification_id=r['verification_id'], status=r['status'],
                             receipt_sha256=r['receipt_sha256']))
        out['cooperation']['replication_solo'] = rows
    _write_new(bank/'confirmations_safety0426'/f'{vid}.json', _seal(out, 'confirmation_sha256'))


def coop_state(root, bank, seed, pid, members, runner=TapeSumRunner, replicate_solos=True, t0=420):
    snap = snapshot(t0)
    snap['episode']['world']['seed'] = seed
    snap['prefixes'] = {str(t): fp(t, 3.*seed+.5*t*t/100.) for t in range(300, t0+1)}
    first = sv.verify(snap, [group_plan(pid, 1., members)], runner(bank/'sumo_verification'),
                      lambda vid, specs: None, root=bank/'sumo_verification')
    assert first['status'] == 'accepted', first.get('reason')
    receipt = {k: v for k, v in first.items() if not k.startswith('_')}
    out = rb.confirm(root, 'B', receipt, snap, lambda work: runner(work), str(root),
                     'bank_verification_ext', workers=2, version='u3')
    assert out['admitted'] is True and out['cooperation']['cooperative'] is True, out
    overlay(root, bank, receipt, snap, runner, replicate_solos)
    state_id = f'st_{seed}'                                 # rb.collect decisions layout
    (bank/'snapshots').mkdir(parents=True, exist_ok=True)
    with (bank/'snapshots'/f'{state_id}.pt').open('xb') as f:
        torch.save(snap, f)
    (bank/'decisions').mkdir(parents=True, exist_ok=True)
    (bank/'decisions'/f'{state_id}.json').write_text(json.dumps(
        dict(state_id=state_id, verification_id=receipt['verification_id'], status='accepted')))
    return receipt, snap


SIX = tuple(f'c{i}' for i in range(6))
STATES = ((986066, 'six', SIX), (986048, 'duo48', ('a1', 'a2')), (986003, 'duo03', ('b1', 'b2')))


def build_bank(root, runner=TapeSumRunner, single_seed=()):
    bank = rb.bank_root(root, 'B')
    bank.mkdir(parents=True, exist_ok=True)
    bs._CACHE.clear()
    return {seed: coop_state(root, bank, seed, pid, members, runner, seed not in single_seed)
            for seed, pid, members in STATES}


@pytest.fixture
def gate_bank(tmp_path):
    build_bank(tmp_path)
    return tmp_path


def offline_fixture():
    offline = dict(gate=ceg.GATE, passes=dict(completed=5, requested=5, optimizer_steps=9, pairs_seen=12,
                                              stop_reason='all registered imitation epochs complete'),
                   imitation=dict(pass1_effective=True, pass1_loss_drop=.2, imitation_failed=False),
                   kl=.001, kl_limit=.045, v2v=True, v2v_dim=24,
                   cooperative_pairs=dict(change_ci95=[-2., -1.]),
                   partner_intent_weights=dict(moved=True))
    offline['checks'] = ceg.offline_checks(offline)
    return offline


def planned(root, control):
    return ceg.plan(root, 'B', control, Policy().state_dict(), Policy().state_dict(), offline=offline_fixture())


def test_per_seed_certifies_states_and_restores_the_reader(gate_bank, monkeypatch):
    monkeypatch.setattr(rb, 'confirmations', rb.confirmations)
    original = rb.confirmations
    candidates, statuses = ceg.certified_candidates(gate_bank, 'B')
    assert rb.confirmations is original                    # installed here -> restored
    assert sorted(r['episode']['world']['seed'] for r, _, _ in candidates) == [986003, 986048, 986066]
    assert {v['cooperative_status'] for v in statuses.values()} == {'replicated'}
    for _, c, seeds in candidates:
        assert c['admitted'] is True and c['bank_rule'] == 'per_seed_v1' and seeds == [1, 2]
        for e in c['per_seed_v1']['examples']:
            assert e['cooperative'] is True and e['synergy']['lower95_one_sided_veh_s'] > 0
    bs.install(rb, bank_rule='per_seed_v1')                 # pre-installed: left installed
    installed = rb.confirmations
    ceg.certified_candidates(gate_bank, 'B')
    assert rb.confirmations is installed
    monkeypatch.setattr(rb, 'confirmations', installed.original)
    bs.install(rb)                                          # a different rule: unwrapped, then restored
    boot = rb.confirmations
    assert len(ceg.certified_candidates(gate_bank, 'B')[0]) == 3 and rb.confirmations is boot


def test_state_cooperative_on_one_seed_is_certified_and_planned_on_that_seed_only(tmp_path):
    build_bank(tmp_path, single_seed=(986003,))              # no second-seed solos for 986003
    candidates, statuses = ceg.certified_candidates(tmp_path, 'B')
    assert {r['episode']['world']['seed']: s for r, _, s in candidates} == {
        986003: [1], 986048: [1, 2], 986066: [1, 2]}
    st = {v['seed']: v for v in statuses.values()}[986003]
    assert st['cooperative_status'] == 'seed1_only' and st['certified_seeds'] == [1]
    assert [p['cooperative_status'] for p in st['per_seed']] == ['cooperative', 'no_solos']
    record = planned(tmp_path, tmp_path/'control')
    three = next(s for s in record['states'] if s['seed'] == 986003)
    assert three['seeds'] == [1] and {j['seed'] for j in three['jobs']} == {1}
    assert list(three['quotes']) == ['seed1'] and three['certification']['replicated'] is False
    assert three['cost']['total_sim_s'] == three['quotes']['seed1']['sim_s']


def test_data_missing_bank_fails_the_gate(tmp_path):
    build_bank(tmp_path, runner=SumRunner)                  # no RESULT.json tapes
    candidates, statuses = ceg.certified_candidates(tmp_path, 'B')
    assert candidates == []
    assert {v['cooperative_status'] for v in statuses.values()} == {'data_missing'}
    report = planned(tmp_path, tmp_path/'c1')
    assert report['verdict'] == 'FAIL' and report['stage'] == 'certification'
    assert 'mandatory_states_certified' in report['reasons']
    # the CLI path: the boot_v2 bank reader itself cannot evaluate the bank -> FAIL, no physics
    ckpt = tmp_path/'u0.pt'
    fixture_checkpoint(ckpt)
    out = ceg.main(['plan', '--root', str(tmp_path), '--control', str(tmp_path/'c2'), '--u0', str(ckpt),
                    '--u3', str(ckpt), '--intent', 'off', '--sources', 'B='+str(tmp_path/'no_labels')])
    assert out['verdict'] == 'FAIL' and out['stage'] == 'bank'
    assert 'BankDataMissing' in out['reasons'][-1]


def test_sources_are_threaded_per_arm(gate_bank, monkeypatch):
    seen = {}
    real_install, real_decisions = bs.install, rb.bank_decisions

    def install(r, **kwargs):
        seen['install'] = kwargs
        return real_install(r, **kwargs)

    def decisions(root, arm, *args, **kwargs):
        seen.setdefault('decisions', []).append(kwargs.get('sources'))
        return real_decisions(root, arm, *args, **kwargs)
    monkeypatch.setattr(bs, 'install', install)
    monkeypatch.setattr(rb, 'bank_decisions', decisions)
    sources = {'B': [str(gate_bank/'srcB')], 'R': [str(gate_bank/'srcR')]}
    ceg.certified_candidates(gate_bank, 'B', sources)
    assert seen['install'] == dict(bank_rule='per_seed_v1', sources=sources)
    assert set(seen['decisions']) == {(str(gate_bank/'srcB'),)}
    calls = []
    monkeypatch.setattr(rb, 'bank_pairs', lambda *a, **k: calls.append(k) or ([], {}))
    ceg.statistical_bank_pairs(gate_bank, 'R', 7, sources, dict(v2v=True, v2v_dim=24))
    assert calls == [dict(v2v=True, sources=(str(gate_bank/'srcR'),))]
    assert ceg.parse_sources(['B=/a,/b', 'R=/c', 'B=/d']) == {'B': ['/a', '/b', '/d'], 'R': ['/c']}
    cfg = gate_bank/'V2V_RUNTIME.json'
    cfg.write_text(json.dumps(dict(config=dict(replay_sources={'B': ['/x'], 'R': []}))))
    assert ceg.parse_sources([str(cfg)]) == {'B': ['/x'], 'R': []}


# ---------------------------------------------------------------- CRN-paired planning

def test_plan_two_crn_seeds_with_u3_references_and_quoted_overage(gate_bank, tmp_path, capsys):
    root = gate_bank
    torch.manual_seed(3)
    new, old = Policy().state_dict(), Policy().state_dict()
    control = tmp_path/'control'
    record = ceg.plan(root, 'B', control, new, old, offline=offline_fixture())
    verify_sealed(record, 'gate_plan_sha256')
    assert {s['seed'] for s in record['states']} == {986066, 986048, 986003}
    six = next(s for s in record['states'] if s['seed'] == 986066)
    names = ['all', 'background', 'reference']+[f'solo{i}' for i in range(6)]
    for seed in ceg.SEEDS:                                  # decision snapshot policy is not u3
        jobs = [j for j in six['jobs'] if j['seed'] == seed]
        assert sorted(j['branch'] for j in jobs) == sorted(names)
        assert len({j['generator_state_sha256'] for j in jobs}) == 1      # CRN within a seed
    g1 = {j['generator_state_sha256'] for j in six['jobs'] if j['seed'] == 1}
    g2 = {j['generator_state_sha256'] for j in six['jobs'] if j['seed'] == 2}
    assert g1 != g2
    # each seed is the VERIFIED example's own CRN: seed 1 = decision snapshot, seed 2 = the
    # confirmation replication salt (replay_bank.replicate(snapshot, verification_id))
    snap = torch.load(rb.bank_root(root, 'B')/'snapshots'/'st_986066.pt', weights_only=False)
    assert g1 == {(snap.get('rng') or {}).get('generator_state_sha256')}
    assert g2 == {rb.replicate(snap, six['verification_id'])['rng']['generator_state_sha256']}
    assert six['seeds'] == [1, 2] and six['certification']['certified_seeds'] == [1, 2]
    assert six['reference_adopted'] is None
    sim = six['jobs'][0]['sim_s']
    assert six['cost']['total_sim_s'] == 18*sim == sum(q['sim_s'] for q in six['quotes'].values())
    assert six['cost']['isolated_n_plus_2_sim_s'] == dict(seed1=8*sim, seed2=8*sim)
    assert six['cost']['overage_sim_s'] == max(0, 18*sim-ceg.ISOLATED_STATE_CAP_SIM_S)
    assert record['totals']['jobs'] == 18+10+10
    assert record['totals']['per_state_sim_s']['986066'] == 18*sim
    ids = []
    for state in record['states']:
        for job in state['jobs']:
            spec = json.loads((control/'jobs'/f'{job["verification_id"]}.{job["branch"]}.spec.json').read_text())
            verify_sealed(spec, 'branch_sha256')
            assert spec['branch_sha256'] == job['spec_sha256']
            snap = torch.load(control/'jobs'/f'{job["verification_id"]}.SNAPSHOT.pt', weights_only=False)
            assert sv.snapshot_sha(snap) == job['snapshot_sha256']
            cfg = snap['policy_synergy']
            if job['branch'] == 'reference':                # all CAVs u3
                assert cfg['new_state_sha256'] == cfg['u3_state_sha256'] == sv._state_sha(old)
            else:
                assert cfg['new_state_sha256'] == sv._state_sha(new)
                assert cfg['background_old'] is (job['branch'] != 'background')
            ids.append(job['verification_id'])
    assert len(set(ids)) == len(ids)
    # caps: overage printed and recorded, never trimmed or refused
    ceg.ISOLATED_STATE_CAP_SIM_S = 100
    capped = ceg.plan(root, 'B', tmp_path/'c2', new, old, offline=offline_fixture())
    assert capped['totals']['jobs'] == record['totals']['jobs']
    assert capped['totals']['overage_sim_s'] == sum(s['cost']['total_sim_s']-100 for s in capped['states'])
    assert 'exceeds the registered isolated cap' in capsys.readouterr().err
    assert ceg.plan(root, 'B', control, new, old, offline=offline_fixture()) == record   # idempotent
    ceg.MAX_GROUP = 2
    with pytest.raises(ValueError, match='group size'):
        ceg.plan(root, 'B', tmp_path/'c3', new, old, offline=offline_fixture())


def test_u3_snapshot_adopts_the_receipt_reference_on_seed_1_only(gate_bank):
    old = Policy().state_dict()
    r = next(x for x in rb.bank_decisions(gate_bank, 'B') if x['episode']['world']['seed'] == 986048)
    snap = dict(torch.load(rb.bank_root(gate_bank, 'B')/'snapshots'/'st_986048.pt', weights_only=False),
                weights=old)
    per_seed, adopted = ceg.state_jobs(r, snap, 0, 'B', Policy().state_dict(), old)
    assert [s['branch'] for s, _ in per_seed[1]] == ['all', 'solo0', 'solo1', 'background']
    assert [s['branch'] for s, _ in per_seed[2]][-1] == 'reference'
    assert adopted['J_veh_s'] == r['branches'][0]['J_veh_s'] and adopted['branch']['branch'] == 'policy'
    tape = bs._read_result(adopted['branches_dir'], adopted['verification_id'], adopted['branch'])
    assert 'final' in tape


def test_offline_failure_fails_before_planning(gate_bank, tmp_path):
    offline = offline_fixture()
    offline['imitation']['pass1_effective'] = False
    offline['checks'] = ceg.offline_checks(offline)
    report = ceg.plan(gate_bank, 'B', tmp_path/'c', Policy().state_dict(), Policy().state_dict(),
                      offline=offline)
    assert report['verdict'] == 'FAIL' and report['stage'] == 'offline'
    assert report['reasons'] == ['offline_checks'] and report['sim_s_planned'] == 0


# ---------------------------------------------------------------- outputs, synergy, verdict

def bfp(t, cum):
    return dict(t_s=float(t), metrics=dict(team_delay_vs=float(cum).hex(), collisions=(0.).hex()))


def write_branch(control, job, window, J, noise):
    """A graded receipt + RESULT.json tape whose per-second increments = the seed's common
    (CRN) noise + J/n, so within-seed differences are exact and cross-seed ones are not."""
    t0, t1 = window['start_s'], window['end_s']
    inc = noise[job['seed']]+J/(t1-t0)
    cum = np.concatenate([[1000.], 1000.+np.cumsum(inc)])
    tape = dict(prefixes={str(t0+i): bfp(t0+i, cum[i]) for i in range(t1-t0)}, final=bfp(t1, cum[-1]))
    awrite(control/'branches'/job['verification_id']/job['branch']/'RESULT.json', tape)
    path = control/'branches'/job['verification_id']/f'{job["branch"]}.graded.json'
    path.write_text(json.dumps(dict(branch=job['branch'], index=0, status='graded',
                                    J_veh_s=float(cum[-1]-cum[0]), branch_sha256=job['spec_sha256'],
                                    result_sha256=digest(tape))))


def crn_noise(n, seed):
    x = np.random.default_rng(seed).normal(0., 3., n)
    return x-x.mean()


def write_outputs(control, record, gains, skip=()):
    """gains: per state seed {(branch, crn seed): gain vs the reference} (reference J = 500)."""
    for state in record['states']:
        w = state['window']
        n = w['end_s']-w['start_s']
        noise = {1: crn_noise(n, 1), 2: crn_noise(n, 2)}
        g = gains[state['seed']]
        for job in state['jobs']:
            if (job['branch'], job['seed']) in skip:
                continue
            gain = 0. if job['branch'] == 'reference' else g.get((job['branch'], job['seed']), 0.)
            write_branch(control, job, w, 500.-gain, noise)


def gains_for(members, all_gain, solo1, solo2, background=1.):
    out = {('all', 1): all_gain[0], ('all', 2): all_gain[1], ('background', 1): background,
           ('background', 2): background}
    for i in range(members):
        out[(f'solo{i}', 1)], out[(f'solo{i}', 2)] = solo1[i], solo2[i]
    return out


PASS_DUOS = {986048: gains_for(2, (10., 9.), (5., 2.), (4., 3.)),
             986003: gains_for(2, (10., 9.), (5., 2.), (4., 3.))}


def test_collect_crn_paired_bootstrap_synergy_pass(gate_bank, tmp_path):
    control = tmp_path/'control'
    record = planned(gate_bank, control)
    write_outputs(control, record, {
        986066: gains_for(6, (20., 18.), [5.+.2*i for i in range(6)], [6.-.5*i for i in range(6)]),
        **PASS_DUOS})
    report = ceg.collect(control, record)
    verify_sealed(json.loads((control/ceg.GATE_FILE).read_text()), 'gate_report_sha256')
    six = {s['seed']: s for s in report['states']}[986066]
    assert six['complete'] is True and six['status'] == 'graded'
    # best solo is chosen per seed on that seed's own branches
    assert six['per_seed']['1']['best_solo_member'] == 'c5' and six['per_seed']['2']['best_solo_member'] == 'c0'
    assert six['per_seed']['1']['synergy_veh_s'] == pytest.approx(20.-6.)
    assert six['per_seed']['2']['synergy_veh_s'] == pytest.approx(18.-6.)
    s1, s2 = six['per_seed']['1'], six['per_seed']['2']
    for s, best, total in ((s1, 'solo5', 14.), (s2, 'solo0', 12.)):   # never pooled across seeds
        syn = s['synergy_v2']
        assert syn['unit_source'] == bs.UNIT_PER_STEP and syn['B'] == bs.BOOT_B and len(syn['per_seed']) == 1
        assert syn['best_solo_branch'] == best and syn['pooled_total_veh_s'] == pytest.approx(total)
        # CRN pairing within the seed cancels the common noise exactly: the bound is the total
        assert s['synergy_lower95_veh_s'] == pytest.approx(total)
    assert s1['group_only_v2']['pooled_total_veh_s'] == pytest.approx(20.)
    assert s2['group_only_v2']['pooled_total_veh_s'] == pytest.approx(18.)
    assert s1['per_vehicle']['c1']['pooled_total_veh_s'] == pytest.approx(5.2)
    assert six['background_only_delta_veh_s'] == pytest.approx(1.)
    assert six['synergy_lower95_veh_s_per_seed'] == pytest.approx({'1': 14., '2': 12.})
    assert six['synergy_lower95_veh_s'] == pytest.approx(14.) and six['seeds_synergy_lower95_gt_0'] == [1, 2]
    assert report['verdict'] == 'PASS' and report['reasons'] == []
    assert report['gate_state_synergy_lower_bound'] == pytest.approx(14.)
    assert report['gate_state_synergy_lower_bounds_per_seed'] == pytest.approx({'1': 14., '2': 12.})
    assert report['gate_state_seeds_tested'] == [1, 2]


def test_both_seeds_tested_and_one_positive_seed_passes(gate_bank, tmp_path):
    control = tmp_path/'control'
    record = planned(gate_bank, control)
    # seed 1: group 20 vs best solo 22 (synergy -2); seed 2: group 18 vs best solo 6 (+12)
    write_outputs(control, record, {986066: gains_for(6, (20., 18.), [22.]*6, [6.]*6), **PASS_DUOS})
    report = ceg.collect(control, record)
    assert report['gate_state_synergy_lower_bounds_per_seed'] == pytest.approx({'1': -2., '2': 12.})
    assert report['verdict'] == 'PASS' and report['gate_state_synergy_lower_bound'] == pytest.approx(12.)


def test_state_certified_on_seed2_only_is_tested_on_its_own_seed2(gate_bank, tmp_path, monkeypatch):
    real = ceg.certified_candidates

    def seed2_only(root, arm, sources=None):
        cands, statuses = real(root, arm, sources)
        return [(r, c, [2] if r['episode']['world']['seed'] == 986066 else s) for r, c, s in cands], statuses
    monkeypatch.setattr(ceg, 'certified_candidates', seed2_only)
    control = tmp_path/'control'
    record = planned(gate_bank, control)
    six = next(s for s in record['states'] if s['seed'] == 986066)
    assert six['seeds'] == [2] and {j['seed'] for j in six['jobs']} == {2} and list(six['quotes']) == ['seed2']
    snap = torch.load(rb.bank_root(gate_bank, 'B')/'snapshots'/'st_986066.pt', weights_only=False)
    assert {j['generator_state_sha256'] for j in six['jobs']} == {
        rb.replicate(snap, six['verification_id'])['rng']['generator_state_sha256']}
    write_outputs(control, record, {986066: gains_for(6, (0., 18.), [9.]*6, [6.]*6), **PASS_DUOS})
    report = ceg.collect(control, record)
    assert report['gate_state_seeds_tested'] == [2]
    assert report['gate_state_synergy_lower_bounds_per_seed'] == pytest.approx({'2': 12.})
    assert report['verdict'] == 'PASS'


def test_collect_fails_on_negative_or_missing_gate_state_data(gate_bank, tmp_path):
    control = tmp_path/'c1'
    record = planned(gate_bank, control)
    write_outputs(control, record, {986066: gains_for(6, (20., 20.), [22.]*6, [21.]*6), **PASS_DUOS})
    report = ceg.collect(control, record)
    assert report['verdict'] == 'FAIL' and report['gate_state_synergy_lower_bound'] < 0
    assert report['reasons'] == ['gate_state_synergy_lower95_gt_0']
    # a missing per-second tape is data_missing -> FAIL, never imputed
    control2 = tmp_path/'c2'
    record2 = planned(gate_bank, control2)
    write_outputs(control2, record2, {986066: gains_for(6, (20., 18.), [5.]*6, [4.]*6), **PASS_DUOS})
    job = next(j for s in record2['states'] if s['seed'] == 986066 for j in s['jobs']
               if j['branch'] == 'all' and j['seed'] == 2)
    for p in (control2/'branches'/job['verification_id']/'all').iterdir():
        p.unlink()
    report2 = ceg.collect(control2, record2)
    six = {s['seed']: s for s in report2['states']}[986066]
    assert six['status'] == 'data_missing' and six['complete'] is False
    assert report2['verdict'] == 'FAIL' and 'gate_state_complete' in report2['reasons']


def test_incomplete_receipts_are_disclosed_never_imputed(gate_bank, tmp_path):
    control = tmp_path/'control'
    record = planned(gate_bank, control)
    report = ceg.collect(control, record)                  # no outputs at all
    six = {s['seed']: s for s in report['states']}[986066]
    assert six['status'] == 'incomplete' and six['synergy_lower95_veh_s'] is None
    assert report['verdict'] == 'FAIL'
    control2 = tmp_path/'control2'
    record2 = planned(gate_bank, control2)
    write_outputs(control2, record2, {986066: gains_for(6, (20., 18.), [5.]*6, [4.]*6), **PASS_DUOS},
                  skip={('reference', 2)})
    six2 = {s['seed']: s for s in ceg.collect(control2, record2)['states']}[986066]
    assert six2['ungraded_branches'] == ['seed2/reference'] and '1' in six2['per_seed']
    assert '2' not in six2['per_seed'] and six2['complete'] is False


def test_receipts_from_raw_results_are_regraded_without_sumo(gate_bank, tmp_path):
    control = tmp_path/'control'
    record = planned(gate_bank, control)
    state = next(s for s in record['states'] if s['seed'] == 986048)
    t0, end_s = state['window']['start_s'], state['window']['end_s']
    job = next(j for j in state['jobs'] if j['branch'] == 'all' and j['seed'] == 1)
    snap = torch.load(control/'jobs'/f'{job["verification_id"]}.SNAPSHOT.pt', weights_only=False)
    base = 3.*986048+.5*t0*t0/100.
    raw = dict(prefixes=dict(snap['prefixes']), final=fp(end_s, base+490.), actual_sim_s=end_s)
    path = control/'branches'/job['verification_id']/'all.result.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw))
    receipt = ceg._receipt(control, job)
    assert receipt['status'] == 'graded' and receipt['J_veh_s'] == pytest.approx(490.)


# ---------------------------------------------------------------- execute: admission and ledger hooks

def test_execute_admits_precharges_and_checks_running(gate_bank, tmp_path, monkeypatch):
    from tools.coopmwm import warm_health
    control = tmp_path/'control'
    record = planned(gate_bank, control)
    log = []

    class FakeBranchRunner:
        blocked = None

        def __init__(self, work, workers=1):
            self.work = work

        def admit(self, specs):
            log.append(('admit', len(specs)))
            return self.blocked

        def prepare(self, specs, snapshot):
            (self.work/specs[0]['verification_id']).mkdir(parents=True, exist_ok=True)

    def fake_charge(run_root, phase, kind):
        def charge(vid, rows):
            log.append(('charge', str(run_root), phase, kind, vid, [r['branch'] for r in rows],
                        sum(r['sim_s'] for r in rows)))
        return charge

    def fake_run_branch(spec, snap, runner):
        log.append(('run', spec['verification_id'], spec['branch']))
        return dict(branch=spec['branch'], index=0, status='graded', J_veh_s=1.,
                    branch_sha256=spec['branch_sha256'])

    monkeypatch.setattr(ps, 'BranchRunner', FakeBranchRunner)
    monkeypatch.setattr(sv, 'run_branch', fake_run_branch)
    monkeypatch.setattr(rb, '_confirm_charge', fake_charge)
    monkeypatch.setattr(warm_health, 'assert_running', lambda root, arm: log.append(('running', arm)))
    FakeBranchRunner.blocked = 'free disk below guard'
    with pytest.raises(RuntimeError, match='free disk'):
        ceg.execute(control)
    assert not any(x[0] == 'charge' for x in log)          # admission precedes any charge
    FakeBranchRunner.blocked = None
    log.clear()
    quotes = ceg.execute(control, root=gate_bank)
    runs = [x for x in log if x[0] == 'run']
    assert len(runs) == record['totals']['jobs'] == sum(x[0] == 'running' for x in log)
    for state in record['states']:
        i = next(k for k, x in enumerate(log) if x[0] == 'charge' and x[4] == state['verification_id'])
        first = next(k for k, x in enumerate(log) if x[0] == 'run'
                     and x[1] in {j['verification_id'] for j in state['jobs']})
        assert log[i-1][0] == 'admit' and i < first        # precharged before the first branch
        _, run_root, phase, kind, vid, branches, sim_s = log[i]
        assert run_root == str(gate_bank) and phase == ceg.PHASE and kind == 'coopfix_exact_u000_B'
        assert len(branches) == len(set(branches)) == len(state['jobs'])   # seed-qualified ids
        assert sim_s == state['cost']['total_sim_s']
    assert [q['sim_s'] for q in quotes] == [s['cost']['total_sim_s'] for s in record['states']]
    log.clear()                                             # claims: never replayed
    again = ceg.execute(control)
    assert all(q['status'] == 'claimed_previously_not_replayed' for q in again) and not log
    report = ceg.collect(control, record)                  # graded receipts feed collect
    assert all(s['ungraded_branches'] == [] for s in report['states'])


def test_all8_examples_certify_each_seed_with_its_own_plan(gate_bank):
    r = next(x for x in rb.bank_decisions(gate_bank, 'B') if x['episode']['world']['seed'] == 986048)
    alt = dict(r['chosen_plan'], id='alt', members=list(reversed(r['chosen_plan']['members'])))
    c = dict(verify_mode='all8_both_seeds', per_seed_v1=dict(examples=[
        dict(seed_index=1, candidate_index=0, admitted=True, cooperative=False, plan=r['chosen_plan']),
        dict(seed_index=2, candidate_index=3, admitted=True, cooperative=True, plan=alt),
        dict(seed_index=2, candidate_index=5, admitted=True, cooperative=True, plan=r['chosen_plan'])]))
    assert ceg.certified_seeds(c) == [2, 2] and ceg.certified_plans(r, c) == {2: alt}   # lowest index
    snap = torch.load(rb.bank_root(gate_bank, 'B')/'snapshots'/'st_986048.pt', weights_only=False)
    per_seed, adopted = ceg.state_jobs(r, snap, 0, 'B', Policy().state_dict(), Policy().state_dict(),
                                       [2], {2: alt})
    assert list(per_seed) == [2] and adopted is None
    assert [s['branch'] for s, _ in per_seed[2]] == ['all', 'solo0', 'solo1', 'background', 'reference']
    assert all(sn['policy_synergy'].get('group', alt['members']) == alt['members'] for _, sn in per_seed[2])
