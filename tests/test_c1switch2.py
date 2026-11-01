"""CPU protocol falsifiers: no SUMO, checkpoint load, GPU or subprocess is allowed."""
import copy
from types import SimpleNamespace

import pytest

from tools.c1switch.storage import canonical
from tools.c1switch2 import audit, core, design, gate
from tools.c1switch2.episode import episode
from tools.c1switch2.inference import Predictor, validate_engine
from tools.c1switch2.observable import anonymous_payload, future_seed
from tools.c1switch2.rerun import replay


@pytest.fixture(autouse=True)
def forbid_simulation_and_spawn(monkeypatch):
    import subprocess
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU test attempted process/SUMO allocation')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)


def payload(now=300, incumbent='native'):
    history = []
    for tick in range(121):
        t = now-60+tick*.5
        row = dict(vehicle_id='self-cav', timestamp_s=t, s_m=10., lane=0,
                   v_mps=20., a_mps2=0., age_s=0., source='cav_self')
        detection = dict(row, vehicle_id='hidden-HDV', source='rsu', s_m=40.)
        history.append(dict(now_s=t, self_reports=[row], detections=[detection],
                            coverage=[], counters=None))
    return dict(schema='c1switch_observable/1', history=history,
                controller_state=dict(incumbent=incumbent, now_s=now,
                                      pi={}, commands={}, wall=None))


class Wall:
    @staticmethod
    def validate_memory(memory, own, now):
        if memory is not None and (set(memory) != {'members', 'started_s'} or
                                   set(memory['members'])-own or memory['started_s'] > now):
            raise ValueError('invalid wall memory')
        return memory


class ToyBackend:
    def __init__(self):
        self.now, self.total, self.actual = 300, 0, 300
        self.selected, self.closed = 'native', False
        self.trace, self.wall = [], False
        self.wall_starts, self.wall_stops = 0, 0

    def select(self, action):
        active = action == 'wall' if isinstance(action, str) else 'wall' in action.values()
        if active and not self.wall:
            self.wall_starts += 1
        elif self.wall and not active:
            self.wall_stops += 1
        self.wall = active
        self.selected = copy.deepcopy(action)
        self.trace.append(('select', copy.deepcopy(action), self.now))

    def advance(self, seconds):
        # No imposed picker cost; wall assembly alone causes an endogenous delay.
        rate = 2 if self.wall and self.now % 40 == 0 else 1
        self.total += rate*seconds
        self.now += seconds
        self.actual += seconds
        self.trace.append(('advance', seconds, self.now))

    def metrics(self):
        return dict(team_delay_vs=self.total)

    def episode_metrics(self):
        return dict(team_delay_vs=500+self.total)

    def fingerprint(self):
        return dict(t_s=self.now, total=self.total, action=copy.deepcopy(self.selected))

    def inference_payload(self):
        return payload(self.now)

    def close(self):
        self.closed = True


def test_exact_budget_includes_full_prefixes_and_optional_model():
    assert design.budget()['wm'] == 79800
    assert design.budget()['bb'] == 79800
    assert design.budget()['fleet_ppo'] == 79800
    assert design.budget()['cav_mappo'] == 89900
    assert design.budget()['wall'] == 19200
    assert design.budget()['canary'] == 4680
    assert sum(design.budget().values()) == 468380
    assert sum(design.budget(True).values()) == 548180
    assert sum(design.GPU_SECONDS[a] for a in design.budget(True)) == 126000
    assert design.measurement_indices(969136) == [10]
    for arm, total in design.budget(True).items():
        assert sum(design.local_budget(h, True)[arm] for h in ('cpuQ', 'gpuR')) == total
    assert sum(len(design.measurement_indices(w['seed'])) for w in design.worlds()) == 16


def test_own_durable_stop_reference_is_not_a_new_allocation():
    line = ('- S-C1SWITCH2 06:00 INFRASTRUCTURE STOP / GPU RELEASE: '
            'gpuR own763520 EXITED durableSTOP on wm__s969147; BB0physics/GPU')
    assert audit.registry_hits(line) == []
    assert audit.registry_hits(line + '; another task reserves seed969136')[0]['seeds'] == [969136]
    assert audit.registry_hits(line.replace('own763520', 'own999999'))[0]['seeds'] == [969147]


def test_both_seed_guards_all_seventeen_worlds():
    for world in design.worlds()+[design.canary_world()]:
        design.guard(world)
    for seed in (964000, 962100, 977001, 969135, 969153):
        with pytest.raises(ValueError):
            design.guard(dict(design.worlds()[0], seed=seed))


def test_expectation_not_median_and_no_switch_hysteresis():
    samples = {c: [20.]*8 for c in design.CONTROLLERS}
    samples['native'] = [0.]*7+[160.]
    samples['wall'] = [19.]*8
    assert core.choose(core.means(samples), 'native') == 'wall'
    costs = {c: 2. for c in design.CONTROLLERS}
    costs['wall'] = 2.-1e-10
    assert core.choose(costs, 'native') == 'wall'
    assert core.choose({c: 2. for c in design.CONTROLLERS}, 'wall') == 'wall'


@pytest.mark.parametrize('invalid', ('missing_wall', 'nan', 'short'))
def test_prediction_refuses_incomplete_or_invalid_futures(invalid):
    samples = {c: [1.]*8 for c in design.CONTROLLERS}
    if invalid == 'missing_wall':
        samples.pop('wall')
    elif invalid == 'nan':
        samples['wall'][0] = float('nan')
    else:
        samples['wall'].pop()
    with pytest.raises(ValueError):
        core.means(samples)


def test_receding_horizon_replans_thirty_times_and_only_executes_twenty():
    env, seen, receipts = ToyBackend(), [], []
    class Picker:
        def predict(self, belief, ident):
            now = belief['history'][-1]['now_s']
            seen.append(now)
            costs = {c: 20. for c in design.CONTROLLERS}
            costs['wall' if len(seen) % 2 else 'pi'] = 0.
            return costs
    row = episode(env, 'wm', design.worlds()[0], predictor=Picker(),
                  receipt=lambda kind, value: receipts.append(copy.deepcopy(value)))
    assert seen == list(range(300, 900, 20))
    assert [t[1] for t in env.trace if t[0] == 'advance'] == [20]*30+[300]
    assert env.actual == 1200
    assert row['wall_decisions'] == 15
    assert env.wall_starts == env.wall_stops == 15
    assert row['J900']['team_delay_vs'] == row['J600']['team_delay_vs']+300
    assert [s['j'] for s in row['measurement_specs']] == [10]
    assert receipts[0]['t_s'] == 300 and receipts[-1]['t_s'] == 880
    assert 'branches' not in receipts[0]  # No truth can alter a committed pick.


def test_fresh_replay_uses_twenty_second_prefix_then_sixty_second_alternative():
    reference = ToyBackend()
    for action in ('wall', 'pi', 'wall'):
        reference.select(action)
        reference.advance(20)
    child = ToyBackend()
    result = replay(lambda: child, ['wall', 'pi', 'wall'], 'native',
                    expected_prefix=reference.fingerprint())
    assert result['actual_sim_s'] == 420
    assert [t[1] for t in child.trace if t[0] == 'advance'] == [20, 20, 20, 60]
    assert child.closed and child.wall_stops == 2


def test_prefix_mismatch_refuses_before_any_counterfactual_tick():
    child = ToyBackend()
    with pytest.raises(ValueError, match='bitwise mismatch'):
        replay(lambda: child, [], 'wall', expected_prefix=dict(t_s=301))
    assert child.trace == [] and child.closed and child.actual == 300


@pytest.mark.parametrize('source', design.CONTROLLERS)
@pytest.mark.parametrize('target', design.CONTROLLERS)
def test_all_thirty_six_handover_pairs_retain_or_reset_wall(source, target):
    events = []
    class Controller:
        def __init__(self, name):
            self.name = name
        def start(self, *, now_s):
            events.append(('start', self.name))
        def stop(self, *, now_s):
            events.append(('stop', self.name))
    ownership = SimpleNamespace(release_all=lambda **kw: events.append(('release', None)), controller=None)
    env = SimpleNamespace(ownership=ownership, api=SimpleNamespace(vehicle=SimpleNamespace(getIDList=lambda: ['self-cav'])),
                          _circles=object(), pi_session=object(), _direct_pi=True)
    handover = core.Handover(Controller)
    handover.select(source, env)
    previous, switch_count = handover.controller, handover.switches
    before = list(events)
    handover.select(target, env)
    assert handover.name == target
    assert handover.switches == switch_count+int(source != target)
    if source == target:
        assert handover.controller is previous and events == before
    if source == 'wall' and source != target:
        assert ('stop', 'wall') in events[len(before):]
    if target == 'wall' and source != target:
        assert ('start', 'wall') in events[len(before):]
    if source != target:
        assert env.pi_session is None and env._direct_pi is False


def test_hdv_identity_and_extras_cannot_change_rpc_payload_or_seed():
    a = payload()
    b = copy.deepcopy(a)
    for tick in b['history']:
        tick['detections'][0]['vehicle_id'] = 'different-simulator-roster-id'
        tick['detections'][0]['hidden_extra'] = 1234
        tick['global_truth'] = {'unused': 90}
    x, y = [anonymous_payload(p, Wall.validate_memory) for p in (a, b)]
    assert canonical(x) == canonical(y)
    assert future_seed(x) == future_seed(y)
    assert 'vehicle_id' not in x['history'][0]['detections'][0]


def test_history_cannot_be_tiled_or_shortened():
    a = payload()
    a['history'] = [a['history'][-1]]*121
    with pytest.raises(ValueError, match='genuine 60s'):
        anonymous_payload(a, Wall.validate_memory)


def test_wall_state_is_causal_self_only_and_continues_across_decisions():
    a = payload(320, 'wall')
    a['controller_state']['wall'] = dict(members=['self-cav'], started_s=300)
    clean = anonymous_payload(a, Wall.validate_memory)
    assert clean['controller_state']['wall']['started_s'] == 300
    a['controller_state']['wall']['members'].append('hidden-HDV')
    with pytest.raises(ValueError, match='invalid wall memory'):
        anonymous_payload(a, Wall.validate_memory)


def test_v7_onroad_entries_off_cannot_claim_full_team_support():
    with pytest.raises(ValueError, match='entries-OFF'):
        validate_engine(dict(prediction_objective='on_road_fleet_delay', team_source_entries=False))


def test_six_way_model_feedback_gets_same_anonymous_futures_and_fresh_memory():
    calls = []
    attestation = dict(protocol=design.PROTOCOL, library=list(design.CONTROLLERS), history_s=60,
        horizon_s=60, K=8, objective='full_team_delay_fleet_plus_source', controller_feedback=True,
        wall_start_stop=True, observable_source_admissions=True, anonymous_seed=True)
    class Engine:
        def imagine(self, belief, **kw):
            calls.append((copy.deepcopy(belief), kw))
            belief['history'].clear()  # Must not contaminate the next candidate.
            return [1.]*8
    engine = Engine()
    engine.attestation = attestation
    picker = Predictor(engine, Wall())
    picker.predict(payload(), 'wm_s969136_j0')
    assert [kw['controller'] for _, kw in calls] == list(design.CONTROLLERS)
    assert {kw['horizon_s'] for _, kw in calls} == {60}
    assert len({kw['seed'] for _, kw in calls}) == 1
    assert all(len(belief['history']) == 121 for belief, _ in calls)


def test_rl_pickers_share_cadence_library_and_observable_boundary():
    class Fleet:
        def select(self, belief):
            return 'wall'
    class PerCav:
        def select(self, belief):
            return {'self-cav': 'wall'}
    for arm, selector in [('fleet_ppo', Fleet()), ('cav_mappo', PerCav())]:
        env = ToyBackend()
        row = episode(env, arm, design.worlds()[0], selector=selector)
        assert len(row['schedule']) == 30 and env.actual == 1200
        assert row['wall_decisions'] == 30
    with pytest.raises(ValueError, match='one shared controller'):
        episode(ToyBackend(), 'fleet_ppo', design.worlds()[0], selector=PerCav())


def test_registration_parser_catches_compressed_ranges_but_not_hashes():
    assert audit.registry_hits('- prior reserve TRAIN969130-140')
    assert audit.registry_hits('- prior reserve 969100..969199')
    assert not audit.registry_hits('- eligibility C4_LOOP_TRAIN 966000-969999')
    assert not audit.registry_hits('- SHA256 a969140abc')
    assert not audit.registry_hits('- utc 2026-10-28T15:05:33.969141+00:00')
    assert not audit.registry_hits('- remote ledger: none used in 966000-969699')
    reuse = ('- S-SELMAPPO2 DESIGN REGISTERED: Per S-C1SWITCH2 registration, '
             'leave local14400evaluation reservation UNSTARTED and consume peer-owned '
             'panel969136..969151/J600 receipts')
    assert not audit.registry_hits(reuse)
    assert audit.registry_hits(reuse+'; unrelated new allocation 969152')


def test_only_main_explicit_approval_can_open_budget_gate():
    assert gate.approval('- 17:14 S-C1SWITCH BUDGET APPROVED') is None
    assert gate.approval('- OWNER asks MAIN to append S-C1SWITCH2 BUDGET APPROVED') is None
    assert gate.approval('- MAIN S-C1SWITCH2 BUDGET APPROVED NOT YET') is None
    assert gate.approval('- S-C1SWITCH2 DESIGN REGISTERED: waiting for S-C1SWITCH2 BUDGET APPROVED') is None
    assert gate.approval('- 00:30 MAIN S-C1SWITCH2 BUDGET APPROVED cap 600000')
    assert gate.approval("- 23:48 S-C1SWITCH2 BUDGET APPROVED (owner, chat: 'yes'): cap 600000")


def test_missing_dependency_still_blocks_after_budget_approval(monkeypatch):
    monkeypatch.setattr(gate, 'sha', lambda path: 'd'*64)
    b = dict(protocol=design.PROTOCOL, sim_s=design.budget(), library=list(design.CONTROLLERS),
             gpu_seconds={a: design.GPU_SECONDS[a] for a in design.budget()}, design_sha256='d'*64)
    evidence = {k: dict(status='PASS', binding_sha256=canonical(b)) for k in gate.REQUIRED}
    evidence['wall_model']['status'] = 'PENDING'
    row = gate.admission('- MAIN S-C1SWITCH2 BUDGET APPROVED', b, evidence)
    assert row['missing_gates'] == ['wall_model'] and not row['dispatch_authorized']
    evidence['wall_model']['status'] = 'PASS'
    assert gate.admission('- MAIN S-C1SWITCH2 BUDGET APPROVED', b, evidence)['dispatch_authorized']


def cpu_binding(monkeypatch):
    monkeypatch.setattr(gate, 'sha', lambda path: 'd'*64)
    binding = dict(protocol=design.PROTOCOL, sim_s=design.budget(), library=list(design.CONTROLLERS),
                   gpu_seconds={a: design.GPU_SECONDS[a] for a in design.budget()},
                   design_sha256='d'*64, dispatch_phase='cpu', enabled_arms=list(design.CPU_ARMS))
    text = ('- MAIN S-C1SWITCH2 BUDGET APPROVED\n'
            '- 01:48 MAIN ACKNOWLEDGEMENTS / DECISIONS: S-C1SWITCH2: do not wait for RL checkpoints - '
            'start the CPU-only arms now (6 fixed controllers + random)')
    evidence = {k: dict(status='PASS', binding_sha256=canonical(binding)) for k in gate.CPU_REQUIRED}
    return binding, text, evidence


def test_cpu_subset_needs_no_unused_model_or_selector_but_keeps_scientific_gates(monkeypatch):
    b, text, evidence = cpu_binding(monkeypatch)
    assert gate.admission(text, b, evidence)['dispatch_authorized']
    for key in gate.CPU_REQUIRED:
        broken = copy.deepcopy(evidence)
        broken[key]['status'] = 'PENDING'
        assert not gate.admission(text, b, broken)['dispatch_authorized'], key
        broken[key] = dict(status='PASS', binding_sha256='other-binding')
        assert not gate.admission(text, b, broken)['dispatch_authorized'], key
    assert sum(design.budget()[a] for a in b['enabled_arms']) == 134400


def test_cpu_subset_cannot_smuggle_picker_or_bypass_main_authority(monkeypatch):
    b, text, evidence = cpu_binding(monkeypatch)
    with pytest.raises(ValueError, match='authority'):
        gate.admission('- MAIN S-C1SWITCH2 BUDGET APPROVED', b, evidence)
    for altered in (dict(b, enabled_arms=list(design.CPU_ARMS)+['wm']),
                    dict(b, enabled_arms=['native']), dict(b, dispatch_phase='other')):
        with pytest.raises(ValueError):
            gate.admission(text, altered, evidence)
    from tools.c1switch2 import runner
    monkeypatch.setattr(runner, 'load_plugin', lambda spec: pytest.fail('loaded disabled arm'))
    with pytest.raises(ValueError, match='not enabled'):
        runner.backend(None, b, 'fleet_ppo', design.worlds()[0], 'blocked')
    with pytest.raises(ValueError, match='no oracle branches'):
        runner.run_branch(None, b, {'arm': 'native'})


def test_cpu_dispatch_plans_exact_disjoint_panel_without_repeating_canary(monkeypatch):
    from tools.c1switch2.supervisor import episode_jobs
    b, _, _ = cpu_binding(monkeypatch)
    q, r = episode_jobs(b, 'cpuQ'), episode_jobs(b, 'gpuR')
    assert len(q) == 84 and len(r) == 28
    assert not ({ident for ident, _ in q} & {ident for ident, _ in r})
    for _, args in q+r:
        assert args[2] in design.CPU_ARMS and int(args[4]) in range(969136, 969152)


@pytest.mark.parametrize('model', ('wm', 'bb'))
def test_model_subset_retains_both_gpu_parities_and_only_its_own_model_gate(monkeypatch, model):
    b, text, _ = cpu_binding(monkeypatch)
    b.update(dispatch_phase='model', enabled_arms=[model])
    needed = gate.CPU_REQUIRED+('inference_'+model, 'gpuA_parity', 'gpuR_parity')
    evidence = {k: dict(status='PASS', binding_sha256=canonical(b)) for k in needed}
    assert gate.admission(text, b, evidence)['dispatch_authorized']
    for key in ('inference_'+model, 'gpuA_parity', 'gpuR_parity'):
        broken = copy.deepcopy(evidence)
        broken.pop(key)
        assert not gate.admission(text, b, broken)['dispatch_authorized']
    with pytest.raises(ValueError, match='one frozen model'):
        gate.admission(text, dict(b, enabled_arms=['wm_phys']), evidence)


def test_gpu_parity_rejects_missing_controller_sample_drift_and_lost_wall_memory():
    from tools.c1switch2.modelops import compare_parity, parity_requests
    from tools.c1switch2 import gpuserver, gpuops
    assert callable(gpuserver.main) and callable(gpuops.stage)
    requests = parity_requests({'protocol': design.PROTOCOL})
    assert requests[0]['request_sha256'] != requests[1]['request_sha256']
    results = {h: {name: dict(samples={c: [float(i+1)]*8 for i, c in enumerate(design.CONTROLLERS)})
                   for name in ('fresh', 'continued_wall')} for h in ('gpuA', 'gpuR')}
    for host in results:
        results[host]['continued_wall']['samples']['wall'] = [2.]*8
    compare_parity(results)
    broken = copy.deepcopy(results)
    broken['gpuA']['fresh']['samples'].pop('wall')
    with pytest.raises(ValueError, match='shape'):
        compare_parity(broken)
    broken = copy.deepcopy(results)
    broken['gpuR']['fresh']['samples']['native'][0] = 7.
    with pytest.raises(ValueError, match='sample parity'):
        compare_parity(broken)
    for host in results:
        results[host]['continued_wall'] = copy.deepcopy(results[host]['fresh'])
    with pytest.raises(ValueError, match='continuation'):
        compare_parity(results)


def test_signed_joint_oracle_gap_can_beat_all_fleet_actions():
    from tools.c1switch2.analyze import regret
    branches = {c: dict(prefix={'t_s': 300}, **{'60': {'team_delay_vs': 20.}})
                for c in design.CONTROLLERS}
    branches['wall']['60']['team_delay_vs'] = 10.
    assert regret(branches, 'native')['regret'] == 10
    with pytest.raises(ValueError, match='held-joint-action'):
        regret(branches, {'self-cav': 'wall'})
    joint = dict(prefix={'t_s': 300}, **{'60': {'team_delay_vs': 5.}})
    assert regret(branches, {'self-cav': 'wall'}, joint)['signed_joint_oracle_gap'] == -5


def test_primary_requires_complete_pairing_and_bootstraps_worlds():
    from tools.c1switch2.analyze import primary
    rows = []
    for world in design.worlds():
        for arm in design.ARMS:
            rows.append(dict(arm=arm, world=world, schedule=['native']*30,
                             J600={'team_delay_vs': 10. if arm == 'wm' else 20.},
                             decisions=[{'prefix': {'t_s': 300, 'seed': world['seed']}}]))
    result = primary(rows, draws=100)
    assert result['wm minus fleet_ppo']['mean'] == -10
    assert result['wm minus best_fixed']['hi'] == -10
    with pytest.raises(ValueError, match='complete paired panel'):
        primary(rows[:-1], draws=100)


def test_wm_phys_rule_is_frozen_at_all_three_crps_clocks():
    from tools.c1switch2.qualification import qualifies
    wm = dict(delay_crps={'120': 10., '300': 20., '600': 30.}, breakdown_rate=.8,
              hdv_position_error_60_m=6.)
    phys = dict(delay_crps={'120': 9., '300': 19., '600': 29.}, breakdown_rate=.6,
                hdv_position_error_60_m=6.)
    receipt = dict(panel='S-RB-EVAL', complete=True, wm2=wm, wm_phys=phys,
                   real={'breakdown_rate': .5})
    assert qualifies(receipt)
    phys['delay_crps']['300'] = 20.
    assert not qualifies(receipt)


def test_backend_overlay_leaves_frozen_v7_globals_untouched():
    from tools.c1switch import sumo as v7
    from tools.c1switch2.backend import frozen_overlay
    original = v7.guard, v7.Handover, v7.Controller
    overlay = frozen_overlay(Wall())
    assert overlay.guard is design.guard and overlay.Handover is core.Handover
    assert (v7.guard, v7.Handover, v7.Controller) == original


def test_runner_installs_frozen_paths_before_guards_but_never_sets_up_invalid_world(tmp_path, monkeypatch):
    from tools.c1switch2 import runner
    events = []
    class Runtime:
        def __init__(self, binding):
            events.append('frozen_paths')
        def create_backend(self, *args, **kwargs):
            events.append('create')
            return 'cpu-construction-sentinel'
    def guarded(world):
        assert events == ['frozen_paths']
        events.append('two_frozen_guards')
    monkeypatch.setattr(runner, 'load_plugin', lambda spec: Runtime)
    monkeypatch.setattr(runner, 'guard', guarded)
    (tmp_path/'jobs').mkdir()
    binding = dict(runtime={}, cpu_host='cpuQ')
    with pytest.raises(ValueError, match='unregistered'):
        runner.backend(tmp_path, binding, 'canary', dict(design.canary_world(), seed=964000), 'bad')
    assert events == []
    result = runner.backend(tmp_path, binding, 'canary', design.canary_world(), 'cpu_probe')
    assert result == 'cpu-construction-sentinel'
    assert events == ['frozen_paths', 'two_frozen_guards', 'create']
