"""COOPFIX INTEGRATION end-to-end CPU test (c4/coopfix-int).

Full chain on ONE synthetic bank, the way the live B-v2v/R-v2v runs are configured:
  synthetic recorded bank (replay_bank.confirm) -> bank_pairs(v2v=True,
  bank_filter='statistical', pair_flags=True) -> COOPFIX gate (coopfix_gate.run_gate via
  its CLI, imitation_trust='own_pass') -> verdict PASS;
plus the hard rule: with every new option off, the outputs are byte-identical to the
pre-fix registered behaviour (bank_pairs off == explicit-off == old summary keys; the
gate's options-off imitation sub-report equals a direct registered bank_imitation.run).

The bank entries reuse the c4/bank-stat-filter fixture semantics (clear/zero/noisy/lin/
solo1 plants with controlled per-branch per-vehicle speeds), extended so every branch
also writes the committed-plan RESULT tape with FULL sample contexts/actions and the
observable trace of every plan second plus the previous one with CAV partner commands
(the c4/v2v bank fixture), so the statistical filter, the v2v attach and the imitation
runner all consume the same recorded bank. Plan actions are the frozen u0 stand-in's own
means plus a small consistent cooperative delta on the duo entries and per-pair noise -
the gate test's learnability pattern - so the verdicts measure the integration, not the
optimiser. No SUMO, no GPU; CPU only.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

from tools.coopmwm import bank_v2v
from tools.coopmwm import coopfix_gate as gate
from tools.coopmwm import replay_bank as rb
from tools.coopmwm import sumo_verify as sv
from tools.coopmwm import v2v_model
from tools.coopmwm.accel_learner import save
from tools.coopmwm.acceleration import encode
from tools.coopmwm.protocol import digest
from tools.coopnative.storage import canonical
from tools.coopmwm.v2v_obs import V2V_DIM

SIGNAL, NOISE = 0.03, 0.01
DUO = ('c1', 'c2')
HD = 'hd1'


@pytest.fixture(autouse=True)
def bank_budget(tmp_path):
    from tools.coopmwm.warm_protocol import budget
    b = budget()
    b['total'] += 2600000
    b['allocated'] += 2600000
    b['phases'] = dict(b['phases'], bank_verification_ext=2600000)
    (tmp_path/'B_REPLAY_PHASE_RECONCILIATION_0627.json').write_text(json.dumps(
        dict(budget=b, amendment_sha256='test-e2e-budget')))
    (tmp_path/'AUTHORITY.md').write_text('test-e2e-budget')


def fp(t, delay):
    hexf = lambda x: float(x).hex()
    return dict(t_s=t, metrics=dict(team_delay_vs=hexf(delay), arrived=hexf(t), collisions=hexf(0.),
                                    teleports=hexf(0.), ttc_below_3_vs=hexf(0.), hard_brake=hexf(0.)))


def snapshot(t0=420):
    return dict(episode=dict(id='warm_u005_B_7', world=dict(seed=7), arm='B', update=5, role='train'),
                policy=dict(state_dict_sha256='p'), sumo=dict(runtime='rt', checkpoint='ck', warm_history=True,
                                                             accel_bounds=[-3, 2], noise_seed=11),
                t0=t0, end_s=1200, prefixes={str(t): fp(t, 3.*7+0.5*t*t/100.) for t in range(300, t0+1)},
                samples={}, rng=dict(generator_state_sha256='g'), committed_plans=[])


def plan(pid, members=DUO, g1=1., g2=.5):
    profile = [g1, g2] if len(members) == 2 else [g1]
    return dict(id=pid, members=list(members), duration_s=120,
                profile=[[g]*120 for g in profile],
                saving_vs_mappo=sum(profile)/len(profile), saving_vs_policy=sum(profile)/len(profile))


def duo_plan(pid, g1=1., g2=.5):
    return plan(pid, DUO, g1, g2)


# per-entry branch speeds [c1, c2(duo only), hd1]; every slot differs between the policy/
# solo side and the saved-duo side exactly like the members' own speeds, so the paired
# per-vehicle deficit differences are uniformly positive on saved entries (the HD row is
# just one more observed vehicle for the filter, in its own lane - no closing pair)
SLOW = [0., 0., 0.]
FAST = [31.29, 31.29, 31.29]


def checkpoint(tmp_path, lr=1e-4, seed=5):
    """A trained-u0 stand-in with a non-zero action head (the registered zero-init head
    would block gradient flow on the first step)."""
    from tools.coopmwm.acceleration import Policy as BasePolicy
    torch.manual_seed(seed)
    policy = BasePolicy()
    with torch.no_grad():
        policy.actor[-1].weight.normal_(0, .5)
        policy.actor[-1].bias.normal_(0, .5)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    path = tmp_path/'B_checkpoint_000.pt'
    save(path, policy, optimizer, torch.Generator().manual_seed(9), 0, {})
    return path


def pair_context(plan_id, t, members):
    """Deterministic sensory context of one committed plan second (3 rows: the member CAVs
    first, then one human-driven vehicle); encode sorts by vid, so members stay ordered."""
    rng = np.random.default_rng(int(abs(hash((plan_id, t))) % (2**31)))
    rows = []
    for i, vid in enumerate(list(members)+[HD]):
        rows.append(dict(vid=vid, s_m=float(rng.uniform(-100, 70)), v_mps=float(rng.uniform(6, 24)),
                         lane=i % 2, is_cav=vid != HD, length_m=5.))
    return rows, encode(rows)


class E2ERunner:
    """Fake plant: the bank-stat-filter speed-controlled J() plus full RESULT tapes
    (sample context/action per committed plan second) and observable traces covering every
    plan second plus the previous one, with CAV partner rows carrying issued commands (the
    v2v bank fixture). The policy branch keeps every vehicle at its recorded speed; plan
    branches use the entry's speed list, so the statistical filter's paired per-vehicle
    deficit differences are exactly controlled."""

    def __init__(self, work, speeds, actions, contexts, syn=0., J_policy=1000.):
        from tools.coopmwm.accel_storage import write as awrite
        self.work = Path(work)
        self.awrite = awrite
        self.speeds, self.actions, self.contexts = speeds, actions, contexts
        self.syn, self.J_policy = syn, J_policy

    def J(self, p):
        if p is None:
            return self.J_policy
        base = float(sum(x for row in p['profile'] for x in row))
        return self.J_policy-base-self.syn*max(0, len(p['members'])-1)

    def speeds_for(self, p):
        return self.speeds.get('policy' if p is None else p['id'], self.speeds['policy'])

    def run(self, spec, snap):
        t0, t_end = spec['window']['start_s'], spec['window']['end_s']
        d = self.work/'branches'/spec['verification_id']/spec['branch']
        d.mkdir(parents=True, exist_ok=True)
        p = spec['plan']
        members = list(p['members']) if p is not None else list(DUO)
        sp = self.speeds_for(p)
        first = t0-1                      # the previous second: v2v last-command channel
        with gzip.open(d/'observable.jsonl.gz', 'wt', encoding='utf-8') as f:
            for t in range(first, t0+120):
                i = t-first
                vehicles = []
                for k, vid in enumerate(members+[HD]):
                    vehicles.append(dict(vid=vid, lane=k, is_cav=vid != HD, length_m=5.,
                                         s_m=10.+k*14.+i, v_mps=float(sp[k % len(sp)]),
                                         issued_acceleration_mps2=(.5+.1*i if vid != HD else None),
                                         command_present=vid != HD))
                f.write(canonical(dict(time_s=t, vehicles=vehicles))+'\n')
        w = snap['episode']['world']['seed']
        r = dict(prefixes={str(t): fp(t, 3.*w+0.5*t*t/100.) for t in range(300, t0+1)},
                 final=fp(t_end, 3.*w+0.5*t0*t0/100.+self.J(p)), actual_sim_s=t_end-300)
        if p is not None:
            steps = []
            for t in range(t0, t0+p['duration_s']):
                rows, context = self.contexts[(p['id'], t)]
                steps.append(dict(time_s=t, sample=dict(
                    context=context, action=self.actions[(p['id'], t)],
                    behaviour=dict(executed=True, plan_id=p['id']),
                    previous_noise=[[0.]]*len(context['members']), deterministic=True, rho=.5,
                    window=1, logp=0., value=0., absolute_accels={v: .5 for v in p['members']},
                    plan_executions=[dict(plan_id=p['id'], proposal_members=list(p['members']),
                                          start_s=t0)])))
            tape = dict(steps=steps)
            self.awrite(d/'RESULT.json', tape)
            r['result_sha256'] = digest(tape)
        return r


def build_bank(root, ckpt_path):
    """Five admitted entries with the bank-stat-filter plants, extended with tapes/traces:
    clear (cooperative), zero (dropped), noisy (dropped), lin (kept, synergy vetoed),
    solo1 (single-CAV). Plan actions = frozen means + consistent coop delta + noise."""
    ref, _, _, _ = gate.load_gate_policy(ckpt_path, V2V_DIM)
    actions, contexts = {}, {}
    # every plan id the confirm machinery executes: the five entries plus one solo branch
    # per duo member (ids '<pid>__solo_<member>'), over every second any window may tape
    pids = [('clear', DUO, True), ('zero', DUO, False), ('noisy', DUO, False),
            ('lin', DUO, True), ('solo1', ('c1',), False)]
    plan_ids = [(pid, members, coop) for pid, members, coop in pids]
    for pid, members, _ in pids:
        if len(members) == 2:
            plan_ids += [(f'{pid}__solo_{m}', (m,), False) for m in members]
    for pid, members, coop in plan_ids:
        for t in range(420, 1140):
            rows, context = pair_context(pid, t, members)
            contexts[(pid, t)] = (rows, context)
            with torch.no_grad():
                means, _ = ref(dict(context, v2v={}))   # zero-init v2v columns: the u0 mean
            rng = np.random.default_rng(abs(hash((pid, 'a', t))) % (2**31))
            action = []
            for i, vid in enumerate(context['members']):
                delta = SIGNAL*(1. if vid == members[0] else -1.) if (coop and vid in members) else 0.
                action.append([float(means[i, 0].detach())+delta+float(rng.normal())*NOISE])
            actions[(pid, t)] = action
    bank = rb.bank_root(root, 'B')
    bank.mkdir(parents=True)

    def confirm(t0, runner):
        first = sv.verify(snapshot(t0), [runner[1]()], runner[0](bank/'sumo_verification'),
                          lambda vid, specs: None, root=bank/'sumo_verification')
        assert first['status'] == 'accepted', first.get('reason')
        receipt = {k: v for k, v in first.items() if not k.startswith('_')}
        out = rb.confirm(root, 'B', receipt, snapshot(t0), lambda work: runner[0](work), str(root),
                         'bank_verification_ext', workers=2, version='u4')
        assert out['admitted'] is True, out
        return out

    def factory(syn, speeds):
        return lambda work: E2ERunner(work, speeds, actions, contexts, syn=syn)

    confirm(420, (factory(50., dict(policy=SLOW, clear=FAST,
                                    **{f'clear__solo_{m}': SLOW for m in DUO})), lambda: duo_plan('clear')))
    confirm(460, (factory(0., dict(policy=SLOW, zero=SLOW,
                                   **{f'zero__solo_{m}': SLOW for m in DUO})), lambda: duo_plan('zero')))
    confirm(500, (factory(0., dict(policy=[20.]*3, noisy=[31.29, 0., 31.29],
                                   **{f'noisy__solo_{m}': [31.29, 0., 31.29] for m in DUO})),
                   lambda: duo_plan('noisy')))
    confirm(540, (factory(0., dict(policy=SLOW, lin=FAST,
                                   **{f'lin__solo_{m}': SLOW for m in DUO})), lambda: duo_plan('lin')))
    confirm(580, (factory(0., dict(policy=SLOW, solo1=FAST)), lambda: plan('solo1', ('c1',))))
    return root


# ---------------------------------------------------------------- hard rule: options off

def test_all_options_off_is_byte_identical_to_the_registered_behaviour(tmp_path):
    ckpt = checkpoint(tmp_path)
    root = build_bank(tmp_path, ckpt)
    p_off, s_off = rb.bank_pairs(root, 'B', 12345)
    p_none, s_none = rb.bank_pairs(root, 'B', 12345, bank_filter=None, pair_flags=False, v2v=False)
    assert json.dumps([p_off, s_off]) == json.dumps([p_none, s_none])
    assert 'bank_filter' not in s_off and 'pair_cooperative' not in s_off and 'v2v' not in s_off
    assert 'filter_dropped' not in s_off['counts']
    # the gate's options-off imitation sub-report equals the registered runner directly
    import copy
    from tools.coopmwm import bank_imitation
    ref, opt, gen, _ = v2v_model.load_checkpoint(ckpt, v2v_dim=0)
    direct = bank_imitation.run(ref, opt, [(s, w) for s, w in p_off], gen, copy.deepcopy(ref))
    report = gate.main(['run', '--checkpoint', str(ckpt), '--bank-root', str(root), '--arm', 'B',
                        '--training-seed', '12345', '--passes', '5', '--out', str(tmp_path/'off.json')])
    imi = report['imitation']
    assert (imi['passes_completed'], imi['pairs_seen'], imi['optimizer_steps'], imi['kl'],
            imi['stop_reason']) == (direct['epochs_completed'], direct['pairs_seen'],
                                    direct['optimizer_steps'], direct['kl'], direct['stop_reason'])
    # with the options off the gate is the registered rule: no v2v inputs exist, so the
    # v2v criterion fails structurally (verdict FAIL, nothing else leaks in)
    assert report['verdict'] == 'FAIL' and report['checks']['v2v_weights_moved'] is False
    assert 'v2v_weights_moved' in report['reasons']
    assert report['filter']['enabled'] is False and report['trust']['enabled'] is False
    assert report['bank']['pairs_with_v2v_features'] == 0


# ---------------------------------------------------------------- the full chain, options on

def test_e2e_bank_filter_v2v_ownpass_gate_pass_and_off_identity(tmp_path):
    ckpt = checkpoint(tmp_path)
    root = build_bank(tmp_path, ckpt)
    # ---- bank_pairs: statistical filter + v2v features + imitation flags, in one call
    pairs, s = rb.bank_pairs(root, 'B', 12345, v2v=True, bank_filter='statistical', pair_flags=True)
    f = s['bank_filter']
    assert f['counts'] == dict(kept=3, dropped=2)
    assert s['counts']['used_admitted'] == 3 and s['counts']['filter_dropped'] == 2
    assert s['v2v']['counts']['kept'] == 360 and s['v2v']['counts']['trace_missing'] is False
    assert all('v2v' in p[0]['context'] and sorted(p[0]['context']['v2v']) for p in pairs)
    # structural: other = lin + solo1 (both statistically NOT cooperative), then the clear
    # entry duplicated to the 50% cooperative share - the flags carry the STATISTICAL label
    # (oversampled duplicates included), aligned with the returned pairs
    assert len(pairs) == 480 and s['exposure']['duplicated_pairs'] == 120
    assert s['exposure']['cooperative_share_of_samples'] == 0.5
    assert s['pair_cooperative'] == [False]*240+[True]*240 and sum(s['pair_cooperative']) == 240
    # ---- the gate through its own CLI, exactly the live B-v2v/R-v2v invocation
    report = gate.main(['run', '--checkpoint', str(ckpt), '--bank-root', str(root), '--arm', 'B',
                        '--training-seed', '12345', '--v2v', '--bank-filter', '--imitation-trust',
                        '--passes', '5', '--out', str(tmp_path/'on.json')])
    assert report['verdict'] == 'PASS' and report['reasons'] == []
    assert all(report['checks'].values())
    assert report['imitation']['passes_completed'] == 5                 # full first pass and beyond
    assert report['imitation']['kl'] <= report['imitation']['kl_limit']
    assert report['filter']['enabled'] is True and report['filter']['report']['counts'] == dict(
        kept=3, dropped=2)                                              # the bank's own filter report
    assert report['trust']['enabled'] is True and report['trust']['cooperative_flags'] is True
    # 360 of 480 pairs carry non-zero v2v features: every duo pair (clear, lin, the 50%
    # oversampled duplicates) does; the single-CAV entry has no partner -> zero block
    assert report['bank']['pairs'] == 480 and report['bank']['pairs_with_v2v_features'] == 360
    assert report['v2v_weights']['moved'] is True and report['v2v_weights']['extra_dim'] == V2V_DIM
    fall = report['cooperative']['fall_ci95']
    assert fall is not None and fall[1] < 0.                            # cooperative error falls
    assert json.loads((tmp_path/'on.json').read_text())['verdict'] == 'PASS'
