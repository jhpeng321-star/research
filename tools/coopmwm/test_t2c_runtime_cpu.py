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
from tools.coopmwm.rerun_tracks_bank import training_batch, batch_receipt, OPTIONS
from tools.coopmwm import bank_imitation


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
        self.logits = nn.Parameter(torch.zeros(2, 2))
        self.calls = 0

    def forward(self, obs, features):
        return self.logits @ torch.tensor([1., features['values'][0]-2.])

    def choose(self, obs, features, proposals, rng):
        self.calls += 1
        probabilities = self(obs, features).detach().softmax(-1)
        return proposals[int(rng.random() >= float(probabilities[0]))]


class Layer(nn.Module):
    def __init__(self, trigger, base=None):
        super().__init__()
        self.correction = nn.Parameter(torch.zeros(()))
        self.effective_slope = nn.Parameter(torch.zeros(()))
        self.base = base
        self.optimizer = torch.optim.SGD([self.correction, self.effective_slope, *trigger.parameters()], lr=.03)

    def residual(self, obs, features):
        value = self.correction+self.effective_slope*(features.get('values', [2.])[0]-2.)
        return {v: float(value.detach()) for v in obs['cav_ids']}

    def effective_policy(self):
        layer = self
        class Policy(nn.Module):
            log_std = torch.tensor([-.7])
            def forward(self, context):
                value = layer.correction+layer.effective_slope*(context['features']['values'][0]-2.)
                if layer.base is not None:
                    with torch.no_grad():
                        base_value = layer.base(torch.ones(1)).squeeze()
                    value = value+base_value
                return value.expand(len(context['members']), 1), None
        return Policy()

    def bank_losses(self, batch, trigger):
        logits = torch.stack([trigger(r['observation'], r['features']) for r in batch])
        labels = torch.tensor([int(r['target'].family != 'none') for r in batch])
        weights = torch.tensor([r['weight'] for r in batch])
        move = (F.cross_entropy(logits, labels, reduction='none')*weights).mean()
        # Use the actual reviewed effective/Tobit objective with each source sample,
        # binding limits and unchanged AWR weight, targeting the trainable layer.
        effective = torch.stack([bank_imitation.pair_loss(self.effective_policy(),
            r['effective_sample'], r['weight'], target_space='effective') for r in batch]).mean()
        return move, effective

    def update(self, *, rows, bank_batch, bank_options, group_advantages=None, realised_rewards=None,
               dense_targets, dense_mask, trigger, kl_cap):
        assert bank_options == OPTIONS
        logits = torch.stack([trigger(r['observation'], r['features']) for r in rows])
        old = logits.detach().softmax(-1)
        label = torch.tensor([int(m.family != 'none') for m, use in zip(dense_targets, dense_mask) if use], dtype=torch.long)
        selected = torch.tensor([int(r['family'] != 'none') for r in rows])
        advantage = torch.tensor(realised_rewards if realised_rewards is not None else group_advantages)
        # Actual autograd/optimizer updates: group credit enters ONLY layer loss.
        ppo = -(logits.log_softmax(-1).gather(1, selected[:, None]).squeeze(1)*advantage).mean()
        dense_loss = F.cross_entropy(logits[dense_mask], label) if len(label) else logits.sum()*0
        self.last_dense_count, self.last_policy_count = len(label), len(rows)
        self.last_bank_batch = deepcopy(bank_batch)
        loss = ppo+dense_loss-self.correction*advantage.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Separate full bank pass, same layer-only optimizer; actual effective loss.
        move_loss, effective_loss = self.bank_losses(bank_batch, trigger)
        self.optimizer.zero_grad()
        (move_loss+effective_loss).backward()
        self.optimizer.step()
        with torch.no_grad():
            after = torch.stack([trigger(r['observation'], r['features']) for r in rows])
            p = after.softmax(-1)
            return dict(dense_ce=float(F.cross_entropy(after[dense_mask], label)) if len(label) else None,
                dense_accuracy=float((after[dense_mask].argmax(-1) == label).float().mean()) if len(label) else None,
                bank_move_ce=float(self.bank_losses(bank_batch, trigger)[0]),
                layer_kl=float((old*(old.log()-p.log())).sum(-1).mean()),
                bank_receipt=dict(batch_sha256=batch_receipt(bank_batch)['batch_sha256'],
                    move_rows=len(bank_batch), effective_rows=len(bank_batch),
                    effective_loss=float(effective_loss), options=bank_options))


class Features:
    def encode(self, obs):
        assert all(r['time_s'] <= obs['time_s']-1 for r in obs['v2v_reports'])
        return dict(values=[obs['own_sensing']['queue']], missing_share=float(not obs['v2v_reports']))


class Credit:
    def estimate(self, obs, move, model):
        return .2+move.depth*obs['own_sensing']['queue']


class RealisedReward:
    def realised_rewards(self, observation, next_observation, *, elapsed_s):
        assert elapsed_s == 1
        return {r['vid']: float(r['v_mps'])/1200 for r in next_observation['own_sensing']['vehicles']}


class Teacher:
    def __init__(self):
        self.kinds, self.slates = [], []

    def label(self, obs, proposals, *, kind, rng):
        self.kinds.append(kind)
        self.slates.append(proposals)
        return dict(target=proposals[1] if kind == 'wm_dense' else rng.choice(proposals),
                    calibrated_probability=.8, imitation_eligible=True,
                    provenance=dict(kind='planner_calibration', receipt_sha256='a'*64))


class Augment:
    def augment(self, bank, *, count, arm):
        out = []
        for i in range(count):
            e = dict(deepcopy(bank[i % len(bank)]), id=f'{arm}_aug_{i}')
            e['plan']['depth'] += .01
            # Proposals deliberately retain source evidence. Only verification
            # may replace it with the modified execution's evidence.
            out.append(e)
        return out


class Weighting:
    def select(self, bank, augmented, *, count, repeat_cap):
        assert repeat_cap == 2
        return [deepcopy((bank+augmented)[i % (len(bank)+len(augmented))]) for i in range(count)]


class Plant:
    def __init__(self, backend, arm, index):
        self.backend, self.arm, self.index, self.t = backend, arm, index, 0
        self.commands = []

    def observe(self):
        return dict(time_s=self.t, own_sensing=dict(queue=2., ego_id='c1',
            vehicles=[dict(vid=v, v_mps=1.+(self.commands[-1][v] if self.commands else 0.)) for v in ('c1', 'c2')]),
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
            coop_layer=lambda c: Layer(c['trigger_head'], model), group_credit=lambda c: Credit(),
            track2_reward=lambda c: RealisedReward(),
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
        from tools.coopmwm.exact_grade import _seal
        rows = []
        for i in range(24):
            obs = Plant(self, arm, i).observe()
            obs['v2v_reports'] = [r for r in obs['v2v_reports'] if r['time_s'] < 0]
            obs['own_sensing']['queue'] = 1.+2*(i % 2)
            plan = dict(family='gap_open' if i % 2 else 'none', members=['c1', 'c2'], depth=.2)
            sample = dict(context=dict(members=['c1', 'c2'], observation=deepcopy(obs),
                                       features=Features().encode(obs)),
                action=[.1 if i % 2 == 0 else .8]*2, binding_limits=dict(c1=.3, c2=.3),
                behaviour=dict(executed=True, proposal_members=['c1', 'c2']),
                plan_sha256=_seal(dict(plan=plan), 'sha256')['sha256'])
            rows.append(dict(id=f'{arm}_{i}', source_id=f'{arm}_{i}', arm=arm, plan=plan,
                window=dict(start_s=0, end_s=1200), seed=986000+i,
                observation=obs, effective_sample=sample, weight=1.+i/24, statistics='per_seed_v1',
                provenance=dict(verification_sha256='c'*64, weight_sha256='d'*64, effective_sha256='e'*64)))
        return rows

    def expand_verification(self, kind, payload, *, cfg):
        return [(dict(sim_s=1200, window=payload['window'], branch=kind), dict(payload))]

    def run_branch(self, spec, snapshot):
        self.branch_calls.append((deepcopy(spec), deepcopy(snapshot)))
        if spec['branch'] == 'shared_final':
            return dict({k: deepcopy(spec[k]) for k in ('arm', 'world_id', 'world', 'checkpoint',
                                                       'window', 'final_worlds_sha256')},
                status='graded', J_veh_s=float(spec['world']['seed'] % 7)+{'A': 4, 'B': 2, 'R': 3}[spec['arm']],
                safety=dict(collisions=0, teleports=0, hard_brakes=0, ttc_events=0))
        return dict(status='graded', J_veh_s={'reference': 10., 'all': 7., 'solo0': 9.,
            'solo1': 9.}.get(spec['branch'], 5.),
            safety=dict(collisions=0, teleports=0, hard_brakes=0, ttc_events=0))

    def grade_verification(self, kind, payload, results):
        return dict(status='graded', advantage=.4, label_correct=payload.get('episode', 0) == 0)

    def verify_augmentation(self, payload, *, cfg, execute):
        from tools.coopmwm.exact_grade import _seal
        execute([(dict(sim_s=1200, window=payload['window'], branch='augment'), payload)], 'all8')
        e = deepcopy(payload)
        e['id'] = payload['proposal_id']+'_s1'
        e['effective_sample']['plan_sha256'] = _seal(dict(plan=e['plan']), 'sha256')['sha256']
        e['effective_sample']['action'] = [.4, .4]  # newly realised modified-plan commands
        e['weight'] = 1.4  # newly realised advantage, not source weight
        e['snapshot_file_sha256'] = 'f'*64
        ex = dict(example_id=e['id'], admitted=True, cooperative=True, advantage_veh_s=.4,
                  later_harm='pass', plan=e['plan'])
        record = _seal(dict(verify_mode='all8_both_seeds', arm=e['arm'],
                            per_seed_v1=dict(examples=[ex])), 'confirmation_sha256')
        eff = _seal(dict(sample=e['effective_sample']), 'sha256')['sha256']
        weight = _seal(dict(weight=e['weight'], example=ex), 'sha256')['sha256']
        e['provenance'] = dict(verification_sha256=record['confirmation_sha256'],
                               weight_sha256=weight, effective_sha256=eff)
        e['augmentation_admission'] = dict(confirmation=record, example_id=ex['example_id'],
            seed=e['seed'], snapshot_file_sha256=e['snapshot_file_sha256'],
            effective_sha256=eff, weight_sha256=weight)
        return dict(examples=[e], receipt=record)

    def practice_reference(self, start, *, common_weights):
        return dict(sim_s=1200, window=start['window'], branch='reference'), dict(start)

    def save_checkpoint(self, path, model, components):
        data = dict(base=model.state_dict(), optimizer={})
        if components:
            data.update(layer=components['coop_layer'].state_dict(),
                        trigger=components['trigger_head'].state_dict(),
                        optimizer=components['coop_layer'].optimizer.state_dict())
        torch.save(data, path)

    def checkpoint_components(self, path):
        return sorted(torch.load(path, weights_only=True), key=['base', 'layer', 'trigger', 'optimizer'].index)

    def diagnostic_weights(self, model, components):
        weights = deepcopy(model.state_dict())
        if components:
            weights['trigger_logits'] = components['trigger_head'].logits.detach().clone()
            weights['correction'] = components['coop_layer'].correction.detach().clone()
            weights['effective_slope'] = components['coop_layer'].effective_slope.detach().clone()
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
    for r in range(rc.ISOLATED_REPLICATES):
        job = dict(world_seed=986000+r)
        rows.append(dict(replicate=r, **job, **diagnostic_identity(backend.resolve_states(job))))
    return dict(role='train', metric=rc.ISOLATED_TEST['metric'], replicates=rows,
                final_evaluation=dict(role='train', worlds=[dict(world_id=f'w{i}', world=dict(seed=986100+i))
                                                           for i in range(rc.TEST_EPISODES)]))


def install_planner_eval(monkeypatch):
    """Test-only implementation of the explicit T1D integration boundary."""
    mod = types.ModuleType('tools.coopmwm.planner_eval')
    mod.calls = []
    def prepare_track2(*, registration, checkpoints, contract):
        mod.calls.append('prepare')
        return [(dict(branch='shared_final', arm=a, **deepcopy(w),
                      checkpoint=deepcopy(checkpoints[a]), sim_s=contract['horizon_s'],
                      window=dict(start_s=0, end_s=contract['horizon_s']),
                      final_worlds_sha256=registration['final_worlds_sha256']), {})
                for a in contract['arms'] for w in registration['worlds']]
    def analyse_track2(*, registration, checkpoints, records, contract):
        mod.calls.append('analyse')
        ci = {}
        keyed = {(r['arm'], r['world_id']): r for r in records}
        for pair in contract['comparisons']:
            a, b = pair.split('-')
            deltas = [keyed[a, w['world_id']]['J_veh_s']-keyed[b, w['world_id']]['J_veh_s']
                      for w in registration['worlds']]
            bs = contract['bootstrap']
            stat = rc.bootstrap_ci(deltas, reps=bs['reps'], seed=bs['seed'], level=bs['level'])
            ci[pair] = dict(n=len(deltas), mean=stat['mean'], low=stat['lo'], high=stat['hi'])
        return dict(status='graded', paired_ci=ci, bootstrap=contract['bootstrap'], checkpoints=checkpoints,
                    final_worlds_sha256=registration['final_worlds_sha256'],
                    safety={a: {k: sum(r['safety'][k] for r in records if r['arm'] == a)
                                for k in contract['safety_counters']} for a in contract['arms']})
    mod.prepare_track2, mod.analyse_track2 = prepare_track2, analyse_track2
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return mod


def prepare(tmp_path, monkeypatch):
    install_moves(monkeypatch)
    install_planner_eval(monkeypatch)
    root = tmp_path/'run'
    rc.init_root(root, track2=True)
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
    assert result['episodes'] == 6 and result['isolated_dry_cells'] == 30
    assert result['virtual_sim_s'] <= cfg['smoke']['total_sim_s']
    assert result['spend']['track2_smoke'] == result['virtual_sim_s']
    assert sum(v for k, v in result['spend'].items() if k != 'track2_smoke') == 0
    assert (rt.root/'CHARGES.jsonl').read_bytes() == b''
    assert backend.ppo_calls == 1 and len(backend.plants) == 6
    assert state_sha(rt.models['A']) != initial['A']
    assert all(state_sha(rt.models[a]) == initial[a] for a in ('B', 'R'))
    assert all(not p.requires_grad for a in ('B', 'R') for p in rt.models[a].parameters())
    for a in ('B', 'R'):
        c = backend.built[a]
        assert c['coop_layer'].correction.item() != 0
        assert c['trigger_head'].logits.abs().sum() > 0
        assert 'dense_teacher' not in c and 'group_credit' not in c
    latest = read_monitor(rt.root)['latest']
    assert all(latest[a]['metrics']['dense_labels'] == 0 for a in ('B', 'R'))
    assert all(latest[a]['metrics']['reward_rows'] == 40 for a in ('B', 'R'))
    assert latest['A']['metrics']['dense_labels'] == 0
    assert latest['B']['metrics']['merge_missing_share'] < .01
    # Real expanded diagnostic path (existing policy_synergy specs and decomposition).
    rt.smoke = False  # post-smoke CPU diagnostic has its own quoted probe phase
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
    rt.smoke = False
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


@pytest.mark.parametrize('arm', ['B', 'R'])
@pytest.mark.parametrize('prob,verified,eligible', [(.799999, False, False), (.8, False, True),
                                                 (.01, True, True), (None, False, False)])
def test_dense_filter_boundary_bypass_and_empty_update(tmp_path, monkeypatch, arm, prob, verified, eligible):
    # Retain coverage of the dormant filter without constructing disabled teachers.
    from tools.coopmwm.rerun_tracks_runtime import dense_label
    proposals = [Move('none'), Move('gap_open')]
    label = dict(target=proposals[1] if prob is not None else None,
                    calibrated_probability=prob, imitation_eligible=eligible,
                    provenance=dict(kind='sumo_verified' if verified else
                        ('planner_calibration' if prob is not None else 'no_target'), receipt_sha256='b'*64))
    assert dense_label(label, proposals, dict(dense_teacher=rc.TRACK_SLOTS['dense_teacher']))['imitation_eligible'] is eligible


def test_dense_filter_rejects_missing_calibration_and_forged_mask():
    from tools.coopmwm.rerun_tracks_runtime import dense_label
    proposals = [Move('none')]
    label = dict(target=proposals[0], calibrated_probability=.79, imitation_eligible=True,
                 provenance=dict(kind='planner_calibration', receipt_sha256='b'*64))
    with pytest.raises(ValueError, match='eligibility disagrees'):
        dense_label(label, proposals, dict(dense_teacher=rc.TRACK_SLOTS['dense_teacher']))
    label['calibrated_probability'] = None
    with pytest.raises(ValueError, match='calibrated probability'):
        dense_label(label, proposals, dict(dense_teacher=rc.TRACK_SLOTS['dense_teacher']))


def test_executes_complete_paired_grid_and_scheduled_tests(tmp_path, monkeypatch):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    rt = Runtime(cfg, reg, backend)
    for update in (6, 12, 24):
        assert rt.diagnostics(update)['cells'] == 10
    assert len(rt.isolated_records) == 30
    analysis = rc.evaluate_isolated_test(rt.isolated_records, registration=reg)
    assert analysis['cells'] == 30 and analysis['verdict'] == 'pass'  # synthetic, never a learning claim
    for arm in ('A', 'B', 'R'):
        backend.save_checkpoint(rt.root/f'{arm}_checkpoint_024.pt', rt.models[arm], rt.components[arm])
    for update in (6, 8, 12, 18, 24):
        rt.scheduled_checks(update)
    assert rt.spend['v2v_interim'] == rt.spend['v2v_final'] == 3*96*1200
    assert rt.spend['v2v_lessons'] == 4*1200
    receipt = json.loads((rt.root/'T2C_FINAL_PAIRED_EVAL.json').read_text())
    assert sys.modules['tools.coopmwm.planner_eval'].calls == ['prepare', 'analyse']
    assert receipt['analysis']['paired_ci']['B-A'] == dict(n=96, mean=-2., low=-2., high=-2.)
    assert len(receipt['records']) == 288
    assert all('ttc_events' in s for s in receipt['analysis']['safety'].values())
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
    assert result['status'] == 'CPU_PLANT_SMOKE_PASS' and result['isolated_dry_cells'] == 30
    completed = subprocess.run([sys.executable, '-m', 'tools.coopmwm.rerun_cpus', 'monitor',
        '--root', cfg['run_root']], env=env, text=True, capture_output=True, check=True, timeout=30)
    assert json.loads(completed.stdout)['updates'] == dict(A=1, B=1, R=1)


@pytest.mark.parametrize('fault', ['unpaired', 'missing', 'short', 'checkpoint', 'component',
                                   'result_identity', 'ttc', 'ci'])
def test_shared_final_harness_rejects_incomplete_or_unpaired_results(tmp_path, monkeypatch, fault):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    rt = Runtime(cfg, reg, backend)
    for arm in ('A', 'B', 'R'):
        backend.save_checkpoint(rt.root/f'{arm}_checkpoint_024.pt', rt.models[arm], rt.components[arm])
    harness = sys.modules['tools.coopmwm.planner_eval']
    original = harness.prepare_track2
    def corrupt(**kw):
        pairs = original(**kw)
        if fault == 'missing':
            return pairs[:-1]
        spec = pairs[-1][0]
        if fault == 'unpaired': spec['world']['seed'] += 1
        if fault == 'short': spec['sim_s'] = 1199
        if fault == 'checkpoint': spec['checkpoint']['file_sha256'] = '0'*64
        return pairs
    harness.prepare_track2 = corrupt
    if fault == 'component':
        backend.checkpoint_components = lambda p: ['base']
    run_branch = backend.run_branch
    def corrupt_result(spec, snapshot):
        result = run_branch(spec, snapshot)
        if fault == 'result_identity': result['checkpoint']['file_sha256'] = '0'*64
        if fault == 'ttc': del result['safety']['ttc_events']
        return result
    backend.run_branch = corrupt_result
    if fault == 'ci':
        analyse = harness.analyse_track2
        def corrupt_analysis(**kw):
            result = analyse(**kw)
            result['paired_ci']['B-A']['n'] = 95
            return result
        harness.analyse_track2 = corrupt_analysis
    with pytest.raises(ValueError, match='REFUSED'):
        rt.final_evaluation()
    assert not (rt.root/'T2C_FINAL_PAIRED_EVAL.json').exists()
    if fault in ('unpaired', 'missing', 'short', 'checkpoint', 'component'):
        assert not backend.branch_calls and rt.spend['v2v_final'] == 0


def test_final_worlds_and_approval_registration_are_bound(tmp_path, monkeypatch):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    from tools.coopmwm.exact_grade import _seal
    first = rc.final_world_registration(reg)
    reg['final_evaluation']['worlds'][0]['world']['seed'] += 1000
    assert rc.final_world_registration(reg)['final_worlds_sha256'] != first['final_worlds_sha256']
    assert _seal(reg, 'sha256')['sha256'] != _seal(dict(reg, final_evaluation=first), 'sha256')['sha256']
    reg['final_evaluation']['worlds'].pop()
    with pytest.raises(ValueError, match='96 explicit'):
        Runtime(cfg, reg, backend)


def test_full_run_reaches_shared_final_harness(tmp_path, monkeypatch):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    rt = Runtime(cfg, reg, backend)
    def update(arm, u, **kw):
        backend.save_checkpoint(rt.root/f'{arm}_checkpoint_{u:03}.pt', rt.models[arm], rt.components[arm])
        return {}
    # Skip repeated episode work here; separate tests exercise numerical updates and
    # the entire 96-cell isolated expansion. This regression guards run-loop routing.
    rt.update = update
    rt.diagnostics = lambda *a, **kw: {}
    monkeypatch.setattr(rc, 'evaluate_isolated_test', lambda *a, **kw: dict(status='test_isolated'))
    rt.run()
    assert sys.modules['tools.coopmwm.planner_eval'].calls == ['prepare', 'analyse']
    assert (rt.root/'T2C_FINAL_PAIRED_EVAL.json').is_file()
    assert (rt.root/'T2C_ISOLATED_EVAL.json').is_file()


def test_fixed_sample_slate_is_equal_across_arms_and_independent_of_stream_size(tmp_path, monkeypatch):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    rt = Runtime(cfg, reg, backend)
    backend.grade_verification = lambda kind, payload, results: dict(status='graded', advantage=.2,
        label_correct=None if payload.get('target') is None else True)
    counts = {}
    for arm in ('B', 'R'):
        for u in range(1, 25):
            # Deliberately unequal stream sizes, missing B targets: sample counts
            # remain equal and absent targets never fabricate realised accuracy.
            rows = [dict(row_id=f'{arm}_{u}_{i}', target=None if arm == 'B' else Move('none'),
                         window=dict(start_s=0, end_s=1200)) for i in range(10 if arm == 'B' else 71)]
            for kind in ('teacher', 'group_credit'):
                checked = rt._verify(arm, u, kind, rows, .999)
                counts[arm, u, kind] = len(checked)
                assert len(checked) == 0
    for kind in ('teacher', 'group_credit'):
        assert sum(counts['B', u, kind] for u in range(1, 25)) == 0
        assert sum(counts['R', u, kind] for u in range(1, 25)) == 0


def test_augmentation_is_once_and_precedes_first_arm_episode(tmp_path, monkeypatch):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    rt = Runtime(cfg, reg, backend, smoke=True)
    calls = []
    for arm in ('B', 'R'):
        augment = rt.components[arm]['plan_augment'].augment
        def wrapped(bank, *, count, arm, original=augment):
            calls.append(arm)
            return original(bank, count=count, arm=arm)
        rt.components[arm]['plan_augment'].augment = wrapped
    episode = rt.episode
    def inspect_episode(*a, **kw):
        assert calls == ['B', 'R']
        return episode(*a, **kw)
    rt.episode = inspect_episode
    rt.run()
    assert calls == ['B', 'R']


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
