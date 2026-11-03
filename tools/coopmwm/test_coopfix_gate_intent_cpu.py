"""CPU tests for the COOPFIX gate intent extension (tools/coopmwm/coopfix_gate.py).

The ``v2v_intent`` option (DEFAULT OFF) trains the zero-initialised intent head on the
pairs' plan-future targets during the gate's offline imitation phase and the report then
also carries (a) the masked plan-future intent error before/after on the cooperative
pairs and (b) whether the PARTNER-INTENT input columns of the widened actor moved.
Option off: the report and verdict checks are exactly the pre-intent gate's.
"""
import numpy as np
import torch

torch.set_num_threads(1)

from tools.coopmwm import coopfix_gate as gate
from tools.coopmwm.accel_learner import save
from tools.coopmwm.acceleration import Policy, encode
from tools.coopmwm.v2v_model import INTENT_BOUNDS, V2V_INTENT_DIM, extend_for_v2v, planned_commands
from tools.coopmwm.v2v_obs import V2VConfig, V2VIntentBuffer, V2V_DIM, intent_block_width, publish_intents

COOP_MEMBERS = ['m0', 'm1']
SIGNAL, NOISE = 0.03, 0.01
PLAN_ACTIONS = {'m0': 0.4, 'm1': -0.3}


def u0_checkpoint(tmp_path, lr=1e-4, seed=5):
    torch.manual_seed(seed)
    policy = Policy()
    with torch.no_grad():
        policy.actor[-1].weight.normal_(0, .5)
        policy.actor[-1].bias.normal_(0, .5)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    path = tmp_path/'B_checkpoint_000.pt'
    save(path, policy, optimizer, torch.Generator().manual_seed(9), 0, {})
    return path, policy


def intent_block(rng, ego, members):
    """48-wide partner slot blocks: the plain six features carry the learnable hint in the
    last-command channel; the five partner-intent channels carry the partner's planned
    command (the signal the intent input columns must learn to use); intent_valid = 1."""
    hint = (.3 if ego == members[0] else -.3) if len(members) > 1 else 0.
    intent = [PLAN_ACTIONS.get(ego, 0.)]*5
    out = []
    for slot in range(4):                     # K=4 partner slots, 6 plain + 5 intent + 1 valid
        out += [float(rng.normal()), 0., float(rng.normal()), float(rng.normal()), hint, 1.]
        out += list(intent) + [1.]
    assert len(out) == intent_block_width(V2VConfig(v2v=True, v2v_intent=True)) == V2V_INTENT_DIM
    return [float(v) for v in out]


def intent_bank(tmp_path, coop_entries=6, solo=40, lr=1e-4):
    """Their build_bank shape, with 48-wide intent v2v blocks and plan-future
    intent targets on every pair (the winning plan's absolute commands)."""
    path, _ = u0_checkpoint(tmp_path, lr=lr)
    policy, _, _, _ = gate.load_gate_policy(path, V2V_INTENT_DIM)
    reference = policy

    def make(plan_id, start_s, members):
        rng = np.random.default_rng(start_s)
        n = 4
        rows_out = [dict(vid=f'v{i}', s_m=float(rng.uniform(-100, 70)),
                         v_mps=float(rng.uniform(6, 24)), lane=int(rng.integers(0, 3)),
                         is_cav=True, length_m=5.) for i in range(n)]
        for i, m in enumerate(members):
            rows_out[i]['vid'] = m
        context = encode(rows_out)
        context['v2v'] = {ego: intent_block(rng, ego, members) for ego in context['members']}
        means, _ = reference(dict(context, v2v={}))
        cooperative = len(members) > 1
        action = []
        for i, m in enumerate(context['members']):
            delta = SIGNAL*(1 if m == members[0] else -1) if cooperative and m in members else 0.
            action.append([float(means[i, 0].detach())+delta+float(rng.normal())*NOISE])
        context['intent_target'] = {m: dict(target=[PLAN_ACTIONS[m]]*5, mask=[1.]*5)
                                    for m in members}
        sample = dict(context=context, action=action,
                      behaviour=dict(route='plan', executed=True, plan_start=False,
                                     proposal_members=list(members), plan_id=plan_id),
                      plan_executions=[dict(plan_id=plan_id, proposal_members=list(members),
                                            start_s=start_s)],
                      previous_noise=[[0.]]*len(members), deterministic=True, rho=.5, window=1,
                      logp=0., value=0.)
        return sample, 1.
    coop = [make(f'p{k}', 400+2*k, COOP_MEMBERS) for k in range(coop_entries) for _ in range(2)]
    other = [make(f's{k}', 500+k, ['m0']) for k in range(solo)]
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


def test_intent_columns_and_weight_report_shapes(tmp_path):
    path, _ = u0_checkpoint(tmp_path)
    plain, _, _, _ = gate.load_gate_policy(path, V2V_DIM)
    assert gate.intent_columns(plain) is None
    assert gate.intent_weight_report(plain) is None
    ext, _, _, _ = gate.load_gate_policy(path, V2V_INTENT_DIM, intent=True)
    columns = gate.intent_columns(ext)
    assert len(columns) == 4*5 and columns[0] == 29+6 and columns[-1] == 29+4*(6+5+1)-2
    report = gate.intent_weight_report(ext)
    assert report['columns'] == 20 and report['horizon'] == 5
    assert report['moved'] is False and report['heads']['actor']['l2'] == 0.   # zero at load


def test_gate_intent_option_off_report_is_pre_intent(tmp_path):
    path, pairs, summary = intent_bank(tmp_path, coop_entries=2, solo=10)
    report = gate.run_gate(path, pairs, summary, dict(v2v=True, v2v_dim=V2V_DIM, passes=5))
    assert 'intent' not in report
    assert 'partner_intent' not in report['v2v_weights']
    assert set(report['checks']) == {'full_pass', 'kl_within_limit',
                                     'cooperative_error_falls', 'v2v_weights_moved'}


def test_gate_intent_on_trains_head_reports_error_fall_and_weight_movement(tmp_path):
    path, pairs, summary = intent_bank(tmp_path)
    report = gate.run_gate(path, pairs, summary,
                           dict(v2v=True, v2v_dim=V2V_INTENT_DIM, v2v_intent=True, passes=5))
    intent = report['intent']
    assert intent['enabled'] is True and intent['horizon'] == 5
    assert intent['pairs_with_intent_targets'] > 0
    assert intent['error_before_mps2'] > 0.       # planned = 0 at load vs 0.4/-0.3 targets
    assert intent['error_after_mps2'] < intent['error_before_mps2']
    partner = report['v2v_weights']['partner_intent']
    assert partner['moved'] is True and partner['columns'] == 20
    assert report['v2v_weights']['moved'] is True
    assert report['checks']['intent_error_falls'] is True
    assert report['checks']['partner_intent_weights_moved'] is True
    assert report['verdict'] == 'PASS'


def test_gate_intent_head_used_by_planned_commands(tmp_path):
    path, _ = u0_checkpoint(tmp_path)
    ext, _, _, _ = gate.load_gate_policy(path, V2V_INTENT_DIM, intent=True)
    context = encode([dict(vid='m0', s_m=-30., v_mps=9., lane=0, is_cav=True, length_m=5.)])
    planned = planned_commands(ext, context)
    assert tuple(planned.shape) == (1, 5)
    assert float(planned.abs().max()) == 0.       # zero-initialised head: neutral plan at load
