"""Deterministic CPU plants and numerical sibling substitutes (never production defaults)."""
from copy import deepcopy
from dataclasses import dataclass
import json
import random
import sys
import threading
import time
import types

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tools.coopmwm import rerun_cpus as rc
from tools.coopmwm.rerun_tracks_runtime import Runtime, WorkerLimit, diagnostic_identity, run, state_sha
from tools.coopmwm.rerun_tracks_monitor import read_monitor


@dataclass(frozen=True)
class Move:
    family: str
    members: tuple = ('c1', 'c2')
    depth: float = 0.


def install_moves(monkeypatch):
    import tools.coopmwm as pkg
    mod = types.ModuleType('tools.coopmwm.moves')
    mod.FAMILIES = ('none', 'gap_open')
    mod.candidates = lambda obs, rng, n: [Move('none'), Move('gap_open', depth=.1+rng.random()*.1)]
    mod.offsets = lambda move, t: {v: move.depth for v in move.members}
    mod.from_plan = lambda p: Move(p['family'], tuple(p['members']), p['depth'])
    monkeypatch.setitem(sys.modules, 'tools.coopmwm.moves', mod)
    monkeypatch.setattr(pkg, 'moves', mod, raising=False)
    return mod


class Trigger(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))
        self.calls = 0

    def choose(self, obs, features, proposals, rng):
        self.calls += 1
        # Both families occur before learning, then policy probabilities can learn.
        return proposals[self.calls % 2]


class Layer(nn.Module):
    def __init__(self, trigger):
        super().__init__()
        self.correction = nn.Parameter(torch.zeros(()))
        self.optimizer = torch.optim.SGD([self.correction, *trigger.parameters()], lr=.03)

    def residual(self, obs, features):
        return {v: float(self.correction.detach()) for v in obs['cav_ids']}

    def update(self, *, rows, bank_targets, group_advantages, dense_targets, trigger, kl_cap):
        old = trigger.logits.detach().softmax(0)
        label = torch.tensor([int(m.family != 'none') for m in dense_targets])
        bank_label = torch.tensor([int(m.family != 'none') for m in bank_targets])
        logits = trigger.logits.expand(len(rows), -1)
        selected = torch.tensor([int(r['family'] != 'none') for r in rows])
        advantage = torch.tensor(group_advantages)
        # Actual autograd/optimizer updates: group credit enters ONLY layer loss.
        ppo = -(logits.log_softmax(-1).gather(1, selected[:, None]).squeeze(1)*advantage).mean()
        loss = ppo+F.cross_entropy(logits, label)+F.cross_entropy(
            trigger.logits.expand(len(bank_label), -1), bank_label)-self.correction*advantage.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            p = trigger.logits.softmax(0)
            return dict(dense_ce=float(F.cross_entropy(trigger.logits.expand(len(label), -1), label)),
                dense_accuracy=float((trigger.logits.argmax() == label).float().mean()),
                bank_move_ce=float(F.cross_entropy(trigger.logits.expand(len(bank_label), -1), bank_label)),
                layer_kl=float((old*(old.log()-p.log())).sum()))


class Features:
    def encode(self, obs):
        assert all(r['time_s'] <= obs['time_s']-1 for r in obs['v2v_reports'])
        return dict(values=[obs['own_sensing']['queue']], missing_share=float(not obs['v2v_reports']))


class Credit:
    def estimate(self, obs, move, model):
        return .2+move.depth*obs['own_sensing']['queue']


class Teacher:
    def __init__(self):
        self.kinds, self.slates = [], []

    def label(self, obs, proposals, *, kind, rng):
        self.kinds.append(kind)
        self.slates.append(proposals)
        return proposals[1] if kind == 'wm_dense' else rng.choice(proposals)


class Augment:
    def augment(self, bank, *, count, arm):
        return [dict(deepcopy(bank[i % len(bank)]), id=f'{arm}_aug_{i}') for i in range(count)]


class Weighting:
    def select(self, bank, augmented, *, count, repeat_cap):
        assert repeat_cap == 2
        return [deepcopy((bank+augmented)[i % (len(bank)+len(augmented))]) for i in range(count)]


class Plant:
    def __init__(self, backend, arm, index):
        self.backend, self.arm, self.index, self.t = backend, arm, index, 0
        self.commands = []

    def observe(self):
        return dict(time_s=self.t, own_sensing=dict(queue=2.),
            v2v_reports=[dict(time_s=self.t-1), dict(time_s=self.t)],
            cav_ids=['c1', 'c2'], bounds={v: [-3., 1.2] for v in ('c1', 'c2')})

    def full_window(self):
        return dict(start_s=0, end_s=1200)

    def step(self, commands, *, speed_mode):
        assert speed_mode == 27 and set(commands) == {'c1', 'c2'}
        assert len(self.commands) == self.t  # single writer, exactly once per second
        assert all(-3 <= v <= 1.2 for v in commands.values())
        self.commands.append(commands)
        self.t += 1

    def finish(self):
        assert self.t == 1200
        self.backend.plants.append(self)
        return dict(reward=sum(c['c1'] for c in self.commands)/1200,
            safety=dict(collisions=0, teleports=0, hard_brakes=0, ttc_events=0),
            training=torch.tensor([1.]), safety_stop=False)


class CPUBackend:
    cpu_plant = True

    def __init__(self):
        self.plants, self.built, self.branch_calls, self.ppo_calls = [], {}, [], 0

    def load_base(self, path, *, arm, cfg):
        model = nn.Linear(1, 1)
        model.load_state_dict(torch.load(path, weights_only=True))
        return model

    def adapters(self, model):
        factories = dict(merge_features=lambda c: Features(), trigger_head=lambda c: Trigger(),
            coop_layer=lambda c: Layer(c['trigger_head']), group_credit=lambda c: Credit(),
            dense_teacher=lambda c: Teacher(), plan_augment=lambda c: Augment(),
            practice_weighting=lambda c: Weighting())
        def build(name):
            def factory(*, cfg, arm, moves, components):
                value = factories[name](components)
                self.built.setdefault(arm, {})[name] = value
                return value
            return factory
        return {n: build(n) for n in factories}

    def base_actions(self, model, obs):
        with torch.no_grad():
            value = float(model(torch.ones(1)).squeeze())
        return {v: value for v in obs['cav_ids']}

    def new_episode(self, *, arm, index, horizon_s, options, **kwargs):
        assert horizon_s == 1200
        assert bool(options) == (arm != 'A')
        return Plant(self, arm, index)

    def ppo_update(self, model, batches):
        self.ppo_calls += 1
        opt = torch.optim.SGD(model.parameters(), lr=.01)
        opt.zero_grad()
        loss = (model(torch.stack(batches))-1).square().mean()
        loss.backward()
        opt.step()

    def bank_examples(self, arm, update, *, cfg):
        return [dict(id=f'{arm}_{i}', arm=arm, plan=dict(family='gap_open', members=['c1', 'c2'], depth=.2),
                     window=dict(start_s=0, end_s=1200), seed=986000+i) for i in range(24)]

    def expand_verification(self, kind, payload, *, cfg):
        return [(dict(sim_s=1200, window=payload['window'], branch=kind), dict(payload))]

    def run_branch(self, spec, snapshot):
        self.branch_calls.append((deepcopy(spec), deepcopy(snapshot)))
        return dict(status='graded', J_veh_s={'reference': 10., 'all': 7., 'solo0': 9.,
            'solo1': 9.}.get(spec['branch'], 5.),
            safety=dict(collisions=0, teleports=0, hard_brakes=0))

    def grade_verification(self, kind, payload, results):
        return dict(status='graded', advantage=.4)

    def practice_reference(self, start, *, common_weights):
        return dict(sim_s=1200, window=start['window'], branch='reference'), dict(start)

    def save_checkpoint(self, path, model, components):
        torch.save(dict(base=model.state_dict(), layer=components['coop_layer'].state_dict()
                        if components else None), path)

    def diagnostic_weights(self, model, components):
        weights = deepcopy(model.state_dict())
        if components:
            weights['trigger_logits'] = components['trigger_head'].logits.detach().clone()
            weights['correction'] = components['coop_layer'].correction.detach().clone()
        return weights

    def resolve_states(self, job):
        from tools.coopmwm.test_replay_bank_confirm_cpu import snapshot, plan
        from tools.coopmwm.sumo_verify import build_branches
        states = []
        for i in range(3):
            snap = snapshot(600)
            snap['episode']['world']['seed'] = job['world_seed']
            p = plan('p1', .1)
            spec = build_branches(snap, [p], 'wm')[0]
            receipt = dict(verification_id=f'state_{i}', chosen_plan=p, proposer='wm', window=spec['window'])
            states.append((snap, receipt))
        return states

    def expand_scheduled(self, kind, *, arm, update, cfg):
        n = rc.TEST_EPISODES if kind == 'fixed_test' else 1
        return [(dict(sim_s=1200, branch=kind, window=dict(start_s=0, end_s=1200)), {}) for _ in range(n)]

    def record_scheduled(self, kind, *, results, **kwargs):
        return dict(status='graded', kind=kind, count=len(results))


def cpu_backend():
    """CLI smoke test adapter: installs test-only move module, never used by production."""
    class Patch:
        def setitem(self, d, k, v): d[k] = v
        def setattr(self, obj, key, value, **kw): setattr(obj, key, value)
    install_moves(Patch())
    return CPUBackend()


def registration(backend):
    rows = []
    for r in range(8):
        job = dict(world_seed=986000+r)
        rows.append(dict(replicate=r, **job, **diagnostic_identity(backend.resolve_states(job))))
    return dict(role='train', metric=rc.ISOLATED_TEST['metric'], replicates=rows)


def prepare(tmp_path, monkeypatch):
    install_moves(monkeypatch)
    root = tmp_path/'run'
    rc.init_root(root)
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    torch.save(model.state_dict(), root/'A_checkpoint_000.pt')
    data = (root/'A_checkpoint_000.pt').read_bytes()
    for arm in ('B', 'R'):
        (root/f'{arm}_checkpoint_000.pt').write_bytes(data)
    cfg = rc.runtime_config({}, root=str(root), reference_root=str(tmp_path/'refs'),
        source_root=str(tmp_path/'src'), common_checkpoint_sha256=rc.sha(root/'A_checkpoint_000.pt'), track2=True)
    backend = CPUBackend()
    return cfg, registration(backend), backend


def test_smoke_executes_all_consumers_updates_and_full_isolated_expansion(tmp_path, monkeypatch):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    rt = Runtime(cfg, reg, backend, smoke=True)
    initial = {a: state_sha(m) for a, m in rt.models.items()}
    result = rt.run()
    assert result['status'] == 'CPU_PLANT_SMOKE_PASS' and result['new_sim_s'] == 0
    assert result['episodes'] == 6 and result['isolated_dry_cells'] == 96
    assert result['virtual_sim_s'] <= cfg['smoke']['total_sim_s']
    assert (rt.root/'CHARGES.jsonl').read_bytes() == b''
    assert backend.ppo_calls == 1 and len(backend.plants) == 6
    assert state_sha(rt.models['A']) != initial['A']
    assert all(state_sha(rt.models[a]) == initial[a] for a in ('B', 'R'))
    assert all(not p.requires_grad for a in ('B', 'R') for p in rt.models[a].parameters())
    for a in ('B', 'R'):
        c = backend.built[a]
        assert c['coop_layer'].correction.item() != 0
        assert c['trigger_head'].logits.abs().sum() > 0
        assert set(c['dense_teacher'].kinds) == {cfg['teacher'][a]['kind']}
    assert sorted(s[1].depth for s in backend.built['B']['dense_teacher'].slates) == sorted(
        s[1].depth for s in backend.built['R']['dense_teacher'].slates)
    latest = read_monitor(rt.root)['latest']
    assert all(latest[a]['metrics']['dense_labels'] == 40 for a in ('B', 'R'))
    assert latest['A']['metrics']['dense_labels'] == 0
    assert latest['B']['metrics']['merge_missing_share'] < .01
    # Real expanded diagnostic path (existing policy_synergy specs and decomposition).
    graded = rt.diagnostics(2)
    assert graded['cells'] == 3 and graded['branches'] == 36
    assert all(r['metric'] == 2. for r in graded['records'])
    for spec, snap in backend.branch_calls:
        if 'policy_synergy' in snap:
            assert snap['u3_weights'].keys() == rt.common_weights.keys()
            assert all(torch.equal(snap['u3_weights'][k], v) for k, v in rt.common_weights.items())
    with pytest.raises(FileExistsError):
        Runtime(cfg, reg, CPUBackend(), smoke=True)


def test_worker_limit_is_aggregate_across_arms_and_diagnostics():
    from concurrent.futures import ThreadPoolExecutor
    limiter = WorkerLimit()
    release = threading.Event()
    entered = threading.Event()
    def job(kind):
        with limiter.permit(kind):
            if limiter.active == 40:
                entered.set()
            assert release.wait(10)
    with ThreadPoolExecutor(max_workers=60) as pool:
        futures = [pool.submit(job, 'episode' if i < 30 else 'diagnostic') for i in range(60)]
        assert entered.wait(10)
        assert limiter.active == limiter.peak == 40
        release.set()
        for f in futures: f.result()
    assert limiter.active == 0 and limiter.peak == 40


def test_off_path_action_reward_rng_and_checkpoint_identity(tmp_path, monkeypatch):
    cfg, _, _ = prepare(tmp_path, monkeypatch)
    cfg = rc.runtime_config({}, root=str(tmp_path/'other'), reference_root='refs', source_root='src',
                            common_checkpoint_sha256='a'*64)
    def legacy(c):
        model = nn.Linear(2, 1)
        x = torch.randn(4, 2)
        reward = torch.randn(4)
        action = model(x)
        loss = (action.flatten()-reward).square().mean()
        opt = torch.optim.SGD(model.parameters(), lr=.01)
        opt.zero_grad(); loss.backward(); opt.step()
        return action.detach(), reward, deepcopy(model.state_dict()), torch.get_rng_state()
    torch.manual_seed(17)
    before = torch.get_rng_state()
    expected = legacy(cfg)
    torch.set_rng_state(before)
    actual = rc.run_tracks(cfg, None, object(), legacy_run=legacy)
    assert torch.equal(expected[0], actual[0]) and torch.equal(expected[1], actual[1])
    assert all(torch.equal(expected[2][k], actual[2][k]) for k in expected[2])
    assert torch.equal(expected[3], actual[3])


def test_runtime_refuses_foreign_bank_short_window_and_optimizer_overlap(tmp_path, monkeypatch):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    rt = Runtime(cfg, reg, backend, smoke=True)
    original = backend.bank_examples
    backend.bank_examples = lambda *a, **k: [dict(original(*a, **k)[0], arm='R')]
    with pytest.raises(ValueError, match='own nonempty banks'):
        rt.update('B', 1, episodes=2)
    backend.bank_examples = original
    def shortened(kind, payload, **kw):
        return [(dict(sim_s=1, window=dict(start_s=0, end_s=1)), {})]
    backend.expand_verification = shortened
    with pytest.raises(ValueError, match='shortened/changed'):
        rt.update('B', 1, episodes=2)
    comp = rt.components['B']['coop_layer']
    comp.optimizer.param_groups[0]['params'].extend(rt.models['B'].parameters())
    with pytest.raises(ValueError, match='separate optimizer'):
        rt._check_optimizer('B')


def test_zero_initialized_layer_identity_and_credit_changes_only_layer():
    trigger = Trigger()
    layer = Layer(trigger)
    assert layer.residual(dict(cav_ids=['c1']), {}) == {'c1': 0.}
    base = torch.tensor([.125, -.25])
    assert torch.equal(base, base+layer.correction.detach())


def test_executes_complete_paired_grid_and_scheduled_tests(tmp_path, monkeypatch):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    rt = Runtime(cfg, reg, backend)
    for update in (6, 12, 18, 24):
        assert rt.diagnostics(update)['cells'] == 24
    assert len(rt.isolated_records) == 96
    analysis = rc.evaluate_isolated_test(rt.isolated_records, registration=reg)
    assert analysis['cells'] == 96 and analysis['verdict'] == 'fail'  # no fabricated learning claim
    for update in (6, 8, 12, 18, 24):
        rt.scheduled_checks(update)
    assert rt.spend['v2v_interim'] == rt.spend['v2v_final'] == 3*96*1200
    assert rt.spend['v2v_lessons'] == 4*1200
    assert all(rt.spend[p] <= cap for p, cap in cfg['budget']['phases'].items())


def test_smoke_and_one_command_monitor_cli(tmp_path, monkeypatch):
    import os
    import subprocess
    cfg, reg, _ = prepare(tmp_path, monkeypatch)
    config_file, reg_file = tmp_path/'config.json', tmp_path/'registration.json'
    config_file.write_text(json.dumps(cfg))
    reg_file.write_text(json.dumps(reg))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    completed = subprocess.run([sys.executable, '-m', 'tools.coopmwm.rerun_cpus', 'smoke',
        '--config', str(config_file), '--registration', str(reg_file), '--adapter',
        'tools.coopmwm.test_t2c_runtime_cpu:cpu_backend'], env=env, text=True,
        capture_output=True, check=True, timeout=60)
    result = json.loads(completed.stdout)
    assert result['status'] == 'CPU_PLANT_SMOKE_PASS' and result['isolated_dry_cells'] == 96
    completed = subprocess.run([sys.executable, '-m', 'tools.coopmwm.rerun_cpus', 'monitor',
        '--root', cfg['run_root']], env=env, text=True, capture_output=True, check=True, timeout=30)
    assert json.loads(completed.stdout)['updates'] == dict(A=1, B=1, R=1)


def test_admission_and_runtime_detect_frozen_base_mutation_and_kl_violation(tmp_path, monkeypatch):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    rt = Runtime(cfg, reg, backend, smoke=True)
    layer = rt.components['B']['coop_layer']
    original = layer.update
    def excessive(**kwargs):
        return original(**kwargs) | dict(layer_kl=1.)
    layer.update = excessive
    with pytest.raises(ValueError, match='KL budget exceeded'):
        rt.update('B', 1, episodes=2)
    assert not (rt.root/'B_checkpoint_001.pt').exists()
    def mutate(**kwargs):
        with torch.no_grad():
            rt.models['R'].bias.add_(1.)
        return original(**kwargs)
    rt.components['R']['coop_layer'].update = mutate
    with pytest.raises(ValueError, match='frozen MAPPO base changed'):
        rt.update('R', 1, episodes=2)
    assert not (rt.root/'R_checkpoint_001.pt').exists()
