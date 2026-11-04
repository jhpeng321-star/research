"""c4/launch-final integration gates (CPU only; no SUMO, GPU or ssh).

Audit fixes (E:/agent-work/_audit_20261103/AUDIT_REPORT.md) and the final A/B/R-v2v configuration:
  I1  v2v/intent inputs divided by fixed physical scales inside the model + fresh Adam state
      (step 0) for the widened columns; one PPO minibatch on a u0-like model stays near the
      frozen-column KL;
  I2  intent-ON launch: bank rows never enter the PPO phase (separate imitation only);
  I4  registered producer of the intent-ON runtime accepted by v2v_launch3.bind_runtime;
  I5  lesson action error with an intent policy (intent blocks + refreshed partner intents);
  B1  gate fixed_update refreshes the partner intents so the intent columns can move;
  B2  gate plan/report seal commit + runtime_sha256 (+ source manifest) and the launcher
      read_gate accepts exactly that report;
  B3  gate config = the deployed V2V_RUNTIME.json keys, refused on any mismatch;
  B5  every V2V dispatch needs the launch3 binding; V2V-only route and V2V_STARTED collisions refused;
  seed-2 later harm measured for new all8 verifications (failure rejects), own-seed safety
      deltas required;
  END-TO-END: MAPPO-u0 -> intent loader (v2v_dim 48, identical keys for A/B/R) -> one decision
      with 8 candidates verified on 2 seeds, different winners per seed -> per_seed_v1 bank
      examples from each seed's own tape (B and R identical rule) -> one exact_update through
      run_separate (pass1_effective, intent loss falls, partner-intent columns move); A has no
      bank. No imitation pair uses another seed's tape; no partner-intent input equals the
      recorded future commands.
"""
import copy
import gzip
import hashlib
import json
import math
import subprocess

import pytest
import torch

from tools.coopnative.storage import canonical
from tools.coopmwm import bank_imitation, coopfix_exact_gate as ceg, exact_grade as xg
from tools.coopmwm import replay_bank as rb, sumo_verify as sv, v2v_intent as vi
from tools.coopmwm import v2v_launch3 as launch, v2v_live, v2v_model as vm, v2v_runner, verify_perseed as vp
from tools.coopmwm.acceleration import Policy as BasePolicy, encode
from tools.coopmwm.accel_learner import save, update as a_update
from tools.coopmwm.accel_protocol import SEED
from tools.coopmwm.accel_storage import write as awrite
from tools.coopmwm.protocol import digest
from tools.coopmwm.v2v_obs import V2V_K, intent_columns
from tools.coopmwm.v2v_runtime import from_weights, load_checkpoint
from tools.coopmwm.test_replay_bank_confirm_cpu import bank_budget, fp, snapshot  # noqa: F401 (autouse budget)
from tools.coopmwm.test_v2v_intent_cpu import CFG, H, cooperative_bank, mappo_u0, rollout, rows
from tools.coopmwm.test_v2v_intent_cpu import single_thread  # noqa: F401 (autouse)
from tools.coopmwm.test_verify_perseed_cpu import SeedPlant, seed_of

PHASE = 'bank_verification_ext'
W = 48
INTENT_COLS = [vm.FEATURES+c for c in intent_columns(V2V_K, H)]
NEW_COLS = list(range(vm.FEATURES, vm.FEATURES+W))


@pytest.fixture
def restore_process_globals(monkeypatch):
    """v2v_live.hooks installs the bank rule and run_separate process-wide (audit M4)."""
    monkeypatch.setattr(rb, 'confirmations', rb.confirmations)
    monkeypatch.setattr(bank_imitation, 'run', bank_imitation.run)


def warmed_u0(path, steps=300, seed=7):
    """MAPPO-u0-shaped base checkpoint whose Adam has a long history (step = ``steps``)."""
    torch.manual_seed(seed)
    policy = BasePolicy()
    torch.nn.init.normal_(policy.actor[-1].weight, std=.3)
    torch.nn.init.normal_(policy.actor[-1].bias, std=.1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    g = torch.Generator().manual_seed(seed)
    for i in range(steps):
        mean, value = policy(encode(rows(i % 7)))
        target = torch.randn(mean.shape, generator=g)*.5
        loss = ((mean-target)**2).mean()+.1*value**2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    save(path, policy, optimizer, torch.Generator().manual_seed(9), 0, 0)
    return policy


# ---------------------------------------------------------------- I1 input scales + fresh Adam

def test_v2v_input_scales_layout_and_scaled_first_layer(tmp_path):
    assert vm.v2v_input_scales(W)[:12] == [100., 1., 30., 3., 3., 1., 3., 3., 3., 3., 3., 1.]
    assert len(vm.v2v_input_scales(W)) == W and vm.v2v_input_scales(24)[:6] == [100., 1., 30., 3., 3., 1.]
    assert vm.v2v_input_scales(4) == [1.]*4                          # dev stub width: unscaled
    path = tmp_path/'u0.pt'
    mappo_u0(path)
    policy, _, _, _ = load_checkpoint(path, config=dict(CFG))
    actor, critic = policy.actor[0], policy.critic[0]
    assert isinstance(actor, vm.V2VInputLinear) and isinstance(critic, vm.V2VInputLinear)
    assert (actor.v2v_tail, critic.v2v_tail) == (4, 12)
    assert not [k for k in policy.state_dict() if 'scale' in k]     # checkpoint keys unchanged
    torch.manual_seed(1)
    with torch.no_grad():
        actor.weight[:, vm.FEATURES:vm.FEATURES+W] = torch.randn(actor.weight.shape[0], W)
    x = torch.randn(3, actor.weight.shape[1])*50.
    scale = actor._input_scale()
    assert scale[vm.FEATURES] == pytest.approx(1/100.) and scale[vm.FEATURES+2] == pytest.approx(1/30.)
    assert scale[vm.FEATURES+6] == pytest.approx(1/3.) and scale[vm.FEATURES+11] == 1. and scale[-1] == 1.
    assert torch.equal(scale[:vm.FEATURES], torch.ones(vm.FEATURES))
    manual = torch.nn.functional.linear(x*scale, actor.weight, actor.bias)
    assert torch.allclose(actor(x), manual)
    # worker reconstruction carries the same scaled layer
    worker = from_weights({k: v.clone() for k, v in policy.state_dict().items()}, enabled=True)
    assert isinstance(worker.actor[0], vm.V2VInputLinear) and torch.allclose(worker.actor[0](x), manual)


def _grads(policy, actor_grad):
    for p in policy.parameters():
        p.grad = torch.zeros_like(p)
    policy.actor[0].weight.grad = actor_grad.clone()


def test_widened_columns_get_a_fresh_adam_state(tmp_path):
    path = tmp_path/'u0.pt'
    warmed_u0(path, steps=300)
    policy, opt, gen, _ = load_checkpoint(path, config=dict(CFG))
    weight = policy.actor[0].weight
    st = opt.state[weight]
    assert float(st['step']) == 300.
    assert torch.equal(st['v2v_column_offset'][NEW_COLS], torch.full((W,), 300.))
    assert float(st['v2v_column_offset'][:vm.FEATURES].abs().sum()) == 0. and float(st['v2v_column_offset'][-4:].sum()) == 0.
    assert float(st['exp_avg'][:, NEW_COLS].abs().sum()) == 0.
    # reference: the new columns as their own tensor under a FRESH torch Adam (step 0)
    ref = torch.nn.Parameter(weight.detach()[:, NEW_COLS].clone())
    ref_opt = torch.optim.Adam([ref], lr=3e-4)
    # reference for the old columns: the unmodified warmed Adam state (no correction)
    old = torch.nn.Parameter(weight.detach().clone())
    old_opt = torch.optim.Adam([old], lr=3e-4)
    old_opt.load_state_dict(dict(state={0: copy.deepcopy({k: v for k, v in st.items()
                                                          if k in ('step', 'exp_avg', 'exp_avg_sq')})},
                                 param_groups=[dict(opt.state_dict()['param_groups'][0], params=[0])]))
    g = torch.Generator().manual_seed(3)
    for _ in range(3):
        grad = torch.randn(weight.shape, generator=g)
        _grads(policy, grad)
        opt.step()
        ref.grad = grad[:, NEW_COLS].clone()
        ref_opt.step()
        old.grad = grad.clone()
        old_opt.step()
        assert torch.allclose(weight.detach()[:, NEW_COLS], ref.detach(), atol=1e-7, rtol=0)
        assert torch.allclose(weight.detach()[:, :vm.FEATURES], old.detach()[:, :vm.FEATURES], atol=1e-7, rtol=0)
    # the bookkeeping survives a checkpoint round trip (intent resume path re-installs the hook)
    save(tmp_path/'u1.pt', policy, opt, gen, 1, 0)
    again, opt2, _, _ = load_checkpoint(tmp_path/'u1.pt', config=dict(CFG))
    st2 = opt2.state[again.actor[0].weight]
    assert torch.equal(st2['v2v_column_offset'], st['v2v_column_offset'])
    assert getattr(opt2, '_v2v_fresh_columns_hook', None) is not None


def _kl_one_minibatch(path, mode):
    """KL (new vs rollout policy) after ONE PPO minibatch on a u0-like intent policy."""
    torch.manual_seed(0)
    policy, opt, gen, _ = load_checkpoint(path, config=dict(CFG))
    if mode == 'legacy':                       # pre-I1: unscaled inputs, warmed step count
        for head in ('actor', 'critic'):
            layer = getattr(policy, head)[0]
            layer.__dict__['_v2v_scale_cache'] = {int(layer.weight.shape[1]): torch.ones(layer.weight.shape[1])}
            opt.state[layer.weight].pop('v2v_column_offset', None)
    worker = from_weights({k: v.clone() for k, v in policy.state_dict().items()}, enabled=True)
    episodes = [rollout(worker, 9, seed) for seed in range(4)]   # 9 s: no regime lookups (multiples of 10)
    handles = []
    if mode == 'frozen':
        for head in ('actor', 'critic'):
            w = getattr(policy, head)[0].weight
            mask = torch.ones_like(w)
            mask[:, NEW_COLS] = 0.
            handles.append(w.register_hook(lambda grad, m=mask: grad*m))
    xg._ppo_exact_update(policy, opt, episodes, gen, [], 1., epochs=1)
    for h in handles:
        h.remove()
    with torch.no_grad():
        d = torch.stack([policy.score(s['sample'])[0]-torch.tensor(s['sample']['logp'])
                         for s in episodes[0]['steps']])
    return float((d.exp()-1-d).mean())


def test_first_ppo_minibatch_kl_stays_near_the_frozen_column_level(tmp_path):
    path = tmp_path/'u0.pt'
    warmed_u0(path, steps=300)
    kl = {mode: _kl_one_minibatch(path, mode) for mode in ('frozen', 'fixed', 'legacy')}
    print('first-minibatch KL', json.dumps(kl))
    assert kl['fixed'] <= 2.*kl['frozen']+5e-3, kl
    assert kl['legacy'] > kl['fixed'], kl


# ---------------------------------------------------------------- I2 bank rows out of PPO

def test_intent_launch_keeps_bank_rows_out_of_ppo(tmp_path, monkeypatch):
    policy, opt, gen, _ = load_checkpoint(_u0(tmp_path), config=dict(CFG))
    pairs = cooperative_bank(tmp_path/'bank')
    episodes = [rollout(from_weights({k: v.clone() for k, v in policy.state_dict().items()}, enabled=True))]
    seen = []
    real = xg._ppo_exact_update

    def spy(policy_, optimizer_, episodes_, generator_, pairs_, *a, **k):
        seen.append(list(pairs_))
        return real(policy_, optimizer_, episodes_, generator_, pairs_, *a, **k)
    monkeypatch.setattr(xg, '_ppo_exact_update', spy)
    monkeypatch.setattr(bank_imitation, 'run', bank_imitation.run_separate)
    metrics = xg.exact_update(policy, opt, episodes, gen, pairs, 1., epochs=1, imitation_epochs=5)
    assert seen == [[]] and metrics[-1]['bank_imitation']['pairs_total'] == len(pairs)
    assert all(row.get('exact_imitation_steps', 0) == 0 for row in metrics if 'bank_imitation' not in row)


def _u0(tmp_path):
    path = tmp_path/'u0.pt'
    if not path.exists():
        mappo_u0(path)
    return path


# ---------------------------------------------------------------- I4 runtime producer

def _runtime_cfg(tmp_path):
    root, legacy, source = tmp_path/'v2v', tmp_path/'legacy', tmp_path/'src'
    cfg = v2v_runner.prepare_intent(dict(budget={}), root=root, legacy_root=legacy, source_root=source,
                                    common_checkpoint_sha256='a'*64)
    return cfg, root, source


def test_intent_runtime_producer_is_accepted_by_the_launcher(tmp_path):
    cfg, root, source = _runtime_cfg(tmp_path)
    for key, want in dict(launch.INTENT_OPTIONS, **launch.BANK_OPTIONS).items():
        assert cfg[key] == want and type(cfg[key]) is type(want)
    assert cfg['v2v_dim'] == W and cfg['bank_statistics'] == 'per_seed_v1'
    assert cfg['arm_names'] == dict(A='A-v2v', B='B-v2v', R='R-v2v') and cfg['bank_arms'] == dict(B='B', R='R')
    assert set(cfg['start_checkpoint_sha256'].values()) == {'a'*64}
    assert cfg['r_concurrent_lag1_quota'] is True       # registered fairness control (kept, MAIN)
    live = dict(cfg, launch_authorized=True, phase_funding_authorized=True,
                budget=dict(total=launch.CEILING_SIM_S, phases=dict(launch.V2V_PHASES)))
    assert launch.bind_runtime(dict(config=live), root=root, source_root=source) is live
    for bad in (dict(bank_statistics='boot_v2'), dict(v2v_intent_detach_trunk=False), dict(v2v_dim=24)):
        with pytest.raises(launch.Refused):
            launch.bind_runtime(dict(config=dict(live, **bad)), root=root, source_root=source)


# ---------------------------------------------------------------- I5 lesson action error

def test_lesson_action_error_with_an_intent_policy(tmp_path, monkeypatch):
    plan = dict(id='p', members=['c1', 'c2'], duration_s=12, profile=[[.1]*12]*2, phase=0.)
    d = tmp_path/'branches'
    tape = write_branch(d/'v1'/'cand1', plan, seed=1, t0=420, t_end=440)
    write_branch(d/'v1'/'policy', None, seed=1, t0=420, t_end=440)
    r = dict(branches=[dict(branch='policy'), dict(branch='cand1', result_sha256=digest(tape))], chosen=0,
             _branches=str(d), verification_id='v1', t0=420)
    policy = vi_nonzero(load_checkpoint(_u0(tmp_path), config=dict(CFG))[0])
    seen = []
    real = vi.refresh_bank_intents
    monkeypatch.setattr(vi, 'refresh_bank_intents', lambda p, pairs: seen.append(pairs) or real(p, pairs))
    err = rb._action_error(policy, r, plan)
    assert isinstance(err, float) and math.isfinite(err)
    assert seen and all(len(b) == W for s, _ in seen[0] for b in s['context']['v2v'].values())


def vi_nonzero(policy, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        policy.intent_head.weight.copy_(torch.randn(policy.intent_head.weight.shape, generator=g)*.3)
    return policy


# ---------------------------------------------------------------- B1 gate refresh

def test_gate_fixed_update_refreshes_partner_intents_so_intent_columns_move(tmp_path, monkeypatch):
    pairs = cooperative_bank(tmp_path/'bank')
    _, offline = ceg.fixed_update(_u0(tmp_path), copy.deepcopy(pairs), None, passes=5, config=dict(CFG))
    assert offline['partner_intents_refreshed']['partner_slots_valid'] > 0
    assert offline['partner_intent_weights']['intent_columns_moved'] is True
    assert offline['checks']['partner_intent_weights_moved'] is True
    # the audit reproduction: without the refresh the intent inputs stay zero and never move
    monkeypatch.setattr(vi, 'refresh_bank_intents', lambda policy, pairs: None)
    _, stale = ceg.fixed_update(_u0(tmp_path), copy.deepcopy(pairs), None, passes=5, config=dict(CFG))
    assert stale['checks']['partner_intent_weights_moved'] is False


# ---------------------------------------------------------------- B2/B3 gate identity

def _write_runtime(root, cfg):
    root.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(dict(config=cfg), sort_keys=True).encode()
    (root/'V2V_RUNTIME.json').write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _git_source(tmp_path):
    src = tmp_path/'gitsrc'
    (src/'tools').mkdir(parents=True)
    (src/'tools'/'x.py').write_text('X = 1\n')
    env = dict(GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@t')
    import os
    env = dict(os.environ, **env)
    for args in (('init', '-q'), ('add', '-A'), ('commit', '-q', '-m', 'x')):
        subprocess.run(['git', '-C', str(src), *args], check=True, capture_output=True, env=env)
    head = subprocess.run(['git', '-C', str(src), 'rev-parse', 'HEAD'], capture_output=True).stdout.decode().strip()
    return src, head


def test_gate_config_is_the_deployed_runtime_and_refuses_mismatch(tmp_path):
    cfg, _, _ = _runtime_cfg(tmp_path)
    runtime_sha = _write_runtime(tmp_path/'run', cfg)
    sha, loaded = ceg.load_runtime(tmp_path/'run')
    # the lane keys (c4/lane-policy) ride the same identity; an intent-only runtime has none
    assert sha == runtime_sha and ceg.runtime_gate_config(loaded) == {k: cfg[k] for k in ceg.RUNTIME_KEYS if k in cfg}
    for bad, kw in ((dict(v2v_intent_loss_weight=.1), {}), (dict(v2v_intent_detach_trunk=False), {}),
                    (dict(bank_statistics='boot_v2'), {}), ({}, dict(intent='off')), ({}, dict(v2v=False))):
        with pytest.raises(ceg.GateIdentityError):
            ceg.runtime_gate_config(dict(cfg, **bad), **kw)
    with pytest.raises(ceg.GateIdentityError, match='unavailable'):
        ceg.load_runtime(tmp_path/'missing')


def test_gate_seals_commit_and_runtime_and_the_launcher_reads_it(tmp_path, monkeypatch):
    cfg, _, _ = _runtime_cfg(tmp_path)
    root = tmp_path/'run'
    runtime_sha = _write_runtime(root, cfg)
    src, head = _git_source(tmp_path)
    ident = ceg.source_identity(src)
    assert ident['commit'] == head and ident['source_manifest_sha256'] is None
    (src/'tools'/'x.py').write_text('X = 2\n')                 # tracked python change -> refuse
    with pytest.raises(ceg.GateIdentityError, match='differ from HEAD'):
        ceg.source_identity(src)
    (src/'tools'/'x.py').write_text('X = 1\n')
    manifest = tmp_path/'SOURCE_PROVENANCE.json'
    manifest.write_text(json.dumps(dict(commit='b'*40, complete_source_inventory=True, files={})))
    packaged = ceg.source_identity(src, manifest)
    assert packaged['commit'] == 'b'*40
    assert packaged['source_manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    # an offline FAIL report (before any physics) already carries the identity
    report = ceg.plan(root, 'B', tmp_path/'control', new_state={}, old_state={},
                      offline=dict(checks=dict(full_pass=False)), source_root=src)
    assert report['verdict'] == 'FAIL' and report['commit'] == head and report['runtime_sha256'] == runtime_sha
    assert report['identity']['runtime_config']['v2v_dim'] == W
    # collect refuses a plan without the sealed identity and binds the plan's identity into the report
    with pytest.raises(ceg.GateIdentityError):
        ceg.collect(tmp_path/'c2', dict(arm='B', update=0, states=[], offline=dict(checks={}), totals={}, caps={}))
    plan_record = dict(arm='B', update=0, states=[], offline=dict(checks=dict(ok=True)), totals={}, caps={},
                       identity=report['identity'], commit=head, runtime_sha256=runtime_sha,
                       source_manifest_sha256=None)
    gate = ceg.collect(tmp_path/'c3', plan_record)
    assert gate['commit'] == head and gate['runtime_sha256'] == runtime_sha
    # the launcher's read_gate accepts exactly these fields (verdict forced PASS for the interface check)
    gate_path = tmp_path/'GATE_PASS.json'
    gate_path.write_text(json.dumps(dict(gate, verdict='PASS')))
    binding = dict(gate_path=str(gate_path), gate_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
                   runtime_sha256=runtime_sha, gate_not_before_unix=gate['unix']-10, gate_max_age_s=3600)
    ok = launch.read_gate(binding, dict(commit=head), now=gate['unix']+1)
    assert ok['verdict'] == 'PASS' and ok['commit'] == head
    with pytest.raises(launch.Refused, match='commit'):
        launch.read_gate(binding, dict(commit='c'*40), now=gate['unix']+1)
    with pytest.raises(launch.Refused, match='runtime'):
        launch.read_gate(dict(binding, runtime_sha256='d'*64), dict(commit=head), now=gate['unix']+1)
    # a runtime changed after planning is refused at collect
    _write_runtime(root, dict(cfg, note='changed'))
    with pytest.raises(ceg.GateIdentityError, match='changed'):
        ceg.collect(tmp_path/'c4', plan_record)


# ---------------------------------------------------------------- B5 dispatch path

def test_dispatch_requires_launch3_and_never_collides_with_another_start(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(v2v_live, 'configuration', lambda root: dict(config=dict(v2v=True, v2v_dim=24)))
    monkeypatch.setattr(v2v_live, 'launch3_check', lambda *a: calls.append(a))
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('Popen without launch3'))
    with pytest.raises(ValueError, match='V2V-only route is disabled'):
        v2v_live.spawn(tmp_path, 'smoke', 'B')
    assert calls == [] and not list(tmp_path.glob('*_DISPATCH.json'))
    with pytest.raises(ValueError, match='V2V-only route is disabled'):
        v2v_live._require_launch3(tmp_path, dict(v2v=True))
    v2v_live._require_launch3(tmp_path, dict(CFG))
    (tmp_path/'V2V_STARTED.json').write_text('{}')
    with pytest.raises(ValueError, match='another launcher'):
        v2v_live._require_launch3(tmp_path, dict(CFG))
    # intent configs: the launch3 preflight runs before EVERY Popen and a refusal aborts
    monkeypatch.setattr(v2v_live, 'configuration', lambda root: dict(config=dict(CFG)))
    monkeypatch.setattr(v2v_live, 'launch3_check', lambda *a: (_ for _ in ()).throw(launch.Refused('HOLD')))
    for stage, arm in (('smoke', 'A'), ('train', 'B'), ('test8', None), ('test24', None)):
        with pytest.raises(launch.Refused):
            v2v_live.spawn(tmp_path, stage, arm)


# ---------------------------------------------------------------- per-seed plant (shared)

def plan_cmd(plan, vid, i):
    if plan is None or vid not in plan['members']:
        return 0.
    return 1.5*math.sin(.7*i+float(plan.get('phase', 0.)))


def seed_rows(i, seed):
    """Synthetic sensor rows; speeds carry the CRN seed so each seed's tape is distinguishable."""
    return rows(i, speed={v: s+.25*seed for v, s in (('c0', 10.), ('c1', 10.5), ('c2', 11.), ('c3', 9.),
                                                   ('h0', 9.5))})


def write_branch(d, plan, seed, t0, t_end, prefixes=None, final=None):
    """RESULT.json (plan steps with full samples) + per-second observable trace of one branch."""
    d.mkdir(parents=True, exist_ok=True)
    D = int(plan['duration_s']) if plan else 0
    frames, steps = [], []
    for t in range(t0-1, t_end):
        sensor = seed_rows(t-t0, seed)
        cmd = {v: plan_cmd(plan, v, t-t0) for v in plan['members']} if plan and t0 <= t < t0+D else {}
        frames.append(dict(time_s=t, vehicles=[dict(r, issued_acceleration_mps2=cmd.get(r['vid'], 0.)
                                                     if r['is_cav'] else None) for r in sensor]))
        if plan is not None and t0 <= t < t0+D:
            context = encode(sensor)
            steps.append(dict(time_s=t, commands={r['vid']: 0. for r in sensor if r['is_cav']},
                              requests={v: dict(requested_raw=x) for v, x in cmd.items()},
                              sample=dict(context=context, action=[[cmd.get(v, 0.)] for v in context['members']],
                                          absolute_accels=cmd, behaviour=dict(executed=True, plan_id=plan['id']),
                                          plan_executions=[dict(plan_id=plan['id'],
                                                                proposal_members=list(plan['members']))])))
    with gzip.open(d/'observable.jsonl.gz', 'wt', encoding='utf-8') as f:
        for frame in frames:
            f.write(canonical(frame)+'\n')
    tape = dict(prefixes=prefixes or {}, final=final or {}, steps=steps)
    awrite(d/'RESULT.json', tape)
    return tape


class E2EPlant(SeedPlant):
    """SeedPlant fingerprints (effect per (plan, seed)) + full plan-step samples and traces.
    ``later_harm`` = {(plan_id, seed): veh-s/s} added AFTER the ordinary window (only an
    extended later-harm window sees it)."""

    def __init__(self, work, gains, later_harm=None, drop_trace=None):
        super().__init__(work, gains)
        self.later_harm, self.drop_trace = later_harm or {}, drop_trace

    def rate(self, spec, snap, t):
        r = super().rate(spec, snap, t)
        p = spec['plan']
        if p is not None and t >= spec['window']['start_s']+int(p['duration_s'])+sv.TAIL_S:
            r += self.later_harm.get((p['id'], seed_of(snap)), 0.)
        return r

    def run(self, spec, snap):
        k = seed_of(snap)
        w = snap['episode']['world']['seed']
        t0, t_end = spec['window']['start_s'], spec['window']['end_s']
        bdir = self.work/'branches'/spec['verification_id']
        bdir.mkdir(parents=True, exist_ok=True)
        sp = bdir/f'{spec["branch"]}.spec.json'
        if not sp.exists():
            sp.write_text(json.dumps(spec))
        prefixes = {str(t): fp(t, 3.*w+0.5*t*t/100.) for t in range(300, t0+1)}
        cum = 3.*w+0.5*t0*t0/100.
        for t in range(t0, t_end):
            cum += self.rate(spec, snap, t)
            if t+1 < t_end:
                prefixes[str(t+1)] = fp(t+1, cum)
        final = fp(t_end, cum)
        tape = write_branch(bdir/spec['branch'], spec['plan'], k, t0, t_end, prefixes, final)
        if self.drop_trace and self.drop_trace(spec, k):
            (bdir/spec['branch']/'observable.jsonl.gz').unlink()
        return dict(prefixes=prefixes, final=final, actual_sim_s=t_end-300, result_sha256=digest(tape))


def plans8(duration=120):
    return [dict(id=f'p{i+1}', members=['c1', 'c2'], duration_s=duration, profile=[[.1]*duration]*2,
                 phase=.4*i, saving_vs_mappo=.1, saving_vs_policy=.1) for i in range(8)]


def decision(root, arm, gains, t0=600, **plant):
    bank = rb.bank_root(root, arm)
    snap = snapshot(t0)
    snap['episode'] = dict(snap['episode'], id=f'warm_u005_{arm}_7', arm=arm)
    plans = plans8()
    r = sv.verify(snap, plans, E2EPlant(bank/'sumo_verification', gains, **plant), sv.local_charge(root/f'first_{arm}'),
                  'wm', root=bank/'sumo_verification')
    (bank/'snapshots').mkdir(parents=True, exist_ok=True)
    torch.save(snap, bank/'snapshots'/'s1.pt')
    (bank/'decisions').mkdir(exist_ok=True)
    (bank/'decisions'/'s1.json').write_text(json.dumps(dict(state_id='s1', verification_id=r['verification_id'],
                                                            status=r['status'], policy_update=5)))
    return r, snap, plans


def confirm(root, arm, gains, **plant):
    out = rb.confirm_pending(root, arm, lambda work: E2EPlant(work, gains, **plant), str(root), PHASE,
                             verify_mode=vp.ALL8)
    recs = vp.records(root, arm)
    assert len(recs) == 1
    return out, next(iter(recs.values()))


def example(c, i, k):
    return next(e for e in c['per_seed_v1']['examples'] if e['candidate_index'] == i and e['seed_index'] == k)


# ---------------------------------------------------------------- seed-2 later harm / own-seed safety

def test_seed2_later_harm_is_measured_and_a_failure_rejects(tmp_path):
    gains = {('p3', 2): 1.5}
    r, _, _ = decision(tmp_path, 'B', gains, later_harm={('p3', 2): 2.})
    confirm(tmp_path, 'B', gains, later_harm={('p3', 2): 2.})
    c = vp.records(tmp_path, 'B')[r['verification_id']]
    e = example(c, 2, 2)
    assert e['lower95_one_sided_veh_s'] > 0 and e['later_harm'] == 'fail' and not e['admitted']
    assert 'seed-2 later-harm check failed' in e['reason'] and c['extended_seed2']['candidate_indices'] == [2]
    assert c['later_harm_measured'] == dict(seed1=True, seed2=True)
    q = vp.quote_decision(600, [120]*8, [['c1', 'c2']]*8, {1: [], 2: [2]})
    assert q['all8_extended_sim_s'] == 2*sv.branch_sim_s(min(600+120+rb.EXTENDED_TAIL_S, 1200))


def test_seed2_later_harm_opt_out_reproduces_not_measured_and_safety_needs_own_deltas(tmp_path):
    gains = {('p3', 2): 1.5}
    r, snap, _ = decision(tmp_path, 'B', gains)
    receipt = {k: v for k, v in r.items()}
    bank = rb.bank_root(tmp_path, 'B')
    out = vp.confirm_all8(tmp_path, 'B', receipt, snap, lambda work: E2EPlant(work, gains), str(tmp_path), PHASE,
                          version=5, branches_dir=bank/'sumo_verification'/'branches', later_harm_seed2=False)
    e = example(out, 2, 2)
    assert e['admitted'] and e['later_harm'] == 'not_measured' and out['extended_seed2'] is None
    # a missing seed-2 trace (own-seed safety deltas) is data_missing, never another seed's window
    root2 = tmp_path/'r2'
    root2.mkdir()
    from tools.coopmwm.test_verify_perseed_cpu import _budget
    _budget(root2)
    drop = (lambda spec, k: k == 2 and spec['branch'] == 'cand3')
    r2, snap2, _ = decision(root2, 'B', gains, drop_trace=drop)
    confirm(root2, 'B', gains, drop_trace=drop)
    e2 = example(vp.records(root2, 'B')[r2['verification_id']], 2, 2)
    assert not e2['admitted'] and e2['status'] == 'data_missing' and 'safety deltas unavailable' in e2['reason']


# ---------------------------------------------------------------- END-TO-END final configuration

def test_end_to_end_final_configuration(tmp_path, monkeypatch, restore_process_globals):
    root = tmp_path
    gains = {('p1', 1): 1.5, ('p3', 2): 1.5, ('p1', 1, 'solo'): .2, ('p3', 2, 'solo'): .2}
    # one decision, 8 candidates, verified on both CRN seeds; different winners per seed (B and R banks)
    vids = {}
    for arm in ('B', 'R'):
        r, _, _ = decision(root, arm, gains)
        res, c = confirm(root, arm, gains)
        assert res == [True] and c['winners'] == {'1': ['p1'], '2': ['p3']} and c['distinct_winners_across_seeds']
        assert c['K'] == 8 and len(c['per_seed_v1']['examples']) == 16
        vids[arm] = r['verification_id']
    # the final A/B/R-v2v runtime: identical intent keys, per_seed_v1 bank for B and R, A has no bank
    cfg = v2v_runner.prepare_intent(dict(budget={}, replay_sources={}), root=root, legacy_root=tmp_path/'legacy',
                                    source_root=tmp_path/'src', common_checkpoint_sha256='a'*64)
    assert {k: cfg[k] for k in launch.INTENT_OPTIONS} == launch.INTENT_OPTIONS and cfg['v2v_dim'] == W
    assert cfg['bank_statistics'] == 'per_seed_v1' and set(cfg['bank_arms']) == {'B', 'R'}
    hooks = v2v_live.hooks(root, cfg)
    assert rb.confirmations.bank_rule == 'per_seed_v1' and bank_imitation.run is bank_imitation.run_separate
    banks = {arm: hooks['bank_pairs'](root, arm, SEED, v2v=True) for arm in ('B', 'R')}
    for arm, (pairs, summary) in banks.items():
        ps1 = summary['counts']['per_seed_v1']
        assert ps1['by_seed'] == {'1': 1, '2': 1} and ps1['cooperative_by_seed'] == {'1': 1, '2': 1}
        by_seed = {e['seed_index']: e for e in summary['entries']}
        rec = vp.records(root, arm)[vids[arm]]
        assert by_seed[1]['plan_id'] == 'p1' and by_seed[2]['plan_id'] == 'p3'
        assert by_seed[1]['tape'] == f"{vids[arm]}/cand1" and by_seed[2]['tape'] == f"{rec['seed2']['verification_id']}/cand3"
        assert by_seed[2]['seed_advantage_veh_s'] > 0 and by_seed[1]['cooperative'] and by_seed[2]['cooperative']
        assert len(pairs) == 2*120 and all(len(b) == W for s, _ in pairs for b in s['context']['v2v'].values())
    assert banks['B'][1]['rule'] == banks['R'][1]['rule'] and len(banks['B'][0]) == len(banks['R'][0])
    pairs, summary = banks['B']
    # every imitation pair comes from ITS OWN seed's tape (context, action and plan of that example)
    record = vp.records(root, 'B')[vids['B']]
    tapes = {}
    for k, i in ((1, 0), (2, 2)):
        e = example(record, i, k)
        tape, _ = vp.imitation_entry(e, record['t0'], 1.)
        tapes[e['tape']['intent_entry']] = (k, e['plan'], {s['time_s']: s for s in tape['steps']})
    assert len(tapes) == 2
    for s, _ in pairs:
        link = s['intent_link']
        k, plan, steps = tapes[link['entry']]
        step = steps[link['t']]
        assert s['context']['x'] == step['sample']['context']['x']
        other = next(v for v in tapes.values() if v[0] != k)[2][link['t']]['sample']['context']['x']
        assert s['context']['x'] != other                                    # never the other seed's tape
        assert s['action'] == [[plan_cmd(plan, v, link['t']-600)] for v in s['context']['members']]
        assert s['behaviour']['proposal_members'] == plan['members']
    # MAPPO-u0 -> intent loader -> one exact_update through run_separate (B)
    u0 = tmp_path/'B_checkpoint_000.pt'
    warmed_u0(u0, steps=50)
    policy, opt, gen, _ = hooks['load_checkpoint'](u0, config=cfg)
    assert isinstance(policy, vi.IntentPolicy) and policy.actor[0].weight.shape[1] == vm.FEATURES+W+4
    episodes = [rollout(from_weights({k: v.clone() for k, v in policy.state_dict().items()}, enabled=True))]
    ppo_pairs = []
    real = xg._ppo_exact_update

    def spy(policy_, optimizer_, episodes_, generator_, pairs_, *a, **k):
        ppo_pairs.append(len(pairs_))
        return real(policy_, optimizer_, episodes_, generator_, pairs_, *a, **k)
    monkeypatch.setattr(xg, '_ppo_exact_update', spy)
    norms_before = vi.weight_norms(policy)
    torch.manual_seed(0)
    metrics = xg.exact_update(policy, opt, episodes, gen, pairs, 1., epochs=1, imitation_epochs=5)
    report = metrics[-1]['bank_imitation']
    assert ppo_pairs == [0]                                                   # bank rows never in PPO
    assert report['pass1_effective'] is True and report['imitation_failed'] is False and report['kl'] <= .045
    intent = report['v2v_intent']
    assert intent['after']['intent_loss'] < intent['before']['intent_loss']   # intent loss drops
    assert norms_before == dict(actor=0., critic=0.) and vi.weight_norms(policy)['actor'] > 0.  # columns move
    # partner-intent inputs = current head on t-1 contexts, never the recorded future commands
    checked = 0
    for s, _ in pairs:
        link = s['intent_link']
        _, plan, _ = tapes[link['entry']]
        for ego, slots in link['slots'].items():
            for slot, vid in enumerate(slots):
                got = s['context']['v2v'][ego][slot*12+6:slot*12+12]
                if vid not in plan['members'] or got[H] != 1.:
                    continue
                recorded = [plan_cmd(plan, vid, link['t']+j-600) for j in range(1, H+1)]
                assert got[:H] != pytest.approx(recorded, abs=1e-3)
                checked += 1
    assert checked > 0
    json.dumps(metrics)
    # R: identical rule and update path on its own bank
    policy_r, opt_r, gen_r, _ = hooks['load_checkpoint'](u0, config=cfg)
    report_r = xg.exact_update(policy_r, opt_r, episodes, gen_r, banks['R'][0], 1., epochs=1,
                               imitation_epochs=5)[-1]['bank_imitation']
    assert report_r['pass1_effective'] is True and vi.weight_norms(policy_r)['actor'] > 0.
    # A: PPO only (accel_learner.update), no bank, no imitation
    policy_a, opt_a, gen_a, _ = hooks['load_checkpoint'](u0, config=cfg)
    monkeypatch.setattr(rb, 'bank_pairs', lambda *a, **k: pytest.fail('A must never read a bank'))
    metrics_a = a_update(policy_a, opt_a, episodes, gen_a)
    assert not any('bank_imitation' in m for m in metrics_a)
    assert cfg['diagnostic_schedule']['A'] == dict(lessons=[], isolated=[]) and 'A' not in cfg['bank_arms']
