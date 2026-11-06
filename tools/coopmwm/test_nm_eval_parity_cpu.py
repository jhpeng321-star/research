"""Training-vs-evaluation parity for the N6 wz3to2 runner (FAIRAUDIT 2026-11-06), CPU only.

For the same rows/natives/seed the eval runner (``nm_eval_runner.eval_policy`` +
``wrap_controller``) and the clean4 TRAINING path (``track2_backend.episode_components`` on the
policy ``accel_plant`` builds from the job weights) must issue IDENTICAL controller events for
A, B and R - including N2 lane-drop moves (``native_commands``), N2's RSU head and, when the
module exists, N9 families. A negative control proves the test catches the missing
``native_commands`` defect. Real EpisodePolicy / TriggerPolicy / MoveExecutor / v2v controller
code; fakes only for the trigger scoring head and registered draws (N2/N9 test fixtures).
"""
from copy import deepcopy
import importlib
import json

import pytest
import torch

from . import launch_production as production, nm_eval_runner as runner, track2_backend as tb
from .acceleration import Controller
from . import test_lane_drop_cpu as _ld, test_track2_backend_cpu as _t2b
from .test_lane_drop_cpu import Prefer, advance, ld_cfg, pair_scene, rsu_scene
from .v2v_intent import IntentPolicy
from .v2v_runtime import controller_class, from_weights

cfg, cpu_threads, fixed = _t2b.cfg, _t2b.cpu_threads, _ld.fixed   # shared fixtures (cpu_threads autouse)
SEED = 989612
UPDATE = 24


def canon(value):
    return json.dumps(value, sort_keys=True, default=repr)


def write_root(tmp_path, cfg, arm):
    """A clean4-shaped run root holding one 'trained' arm: checkpoint + track2 worker sidecar."""
    root = tmp_path/f'root_{arm}'
    root.mkdir()
    torch.manual_seed(7)
    if arm == 'A':
        base = IntentPolicy()
        with torch.no_grad():
            base.actor[-1].bias.fill_(.25)
        torch.save(dict(update=UPDATE, policy=base.state_dict()), root/f'A_checkpoint_{UPDATE:03}.pt')
        return root, base.state_dict(), None
    p = tb.EpisodePolicy(IntentPolicy(), cfg, arm)
    with torch.no_grad():
        p.layer.base.actor[-1].bias.fill_(.25)                    # nonzero frozen MAPPO mean
        for param in p.layer.layer_parameters():
            param.add_(.01)                                       # a non-initial layer state
    worker = p.worker_state()
    torch.save(dict(update=UPDATE, policy=p.layer.base.state_dict()), root/f'{arm}_checkpoint_{UPDATE:03}.pt')
    torch.save(dict(arm=arm, update=UPDATE, worker=worker), root/f'{arm}_track2_{UPDATE:03}.pt')
    return root, p.layer.base.state_dict(), worker


def training(cfg, arm, base_state, worker, head=None):
    """accel_plant: from_weights(job weights) -> track2_backend.episode_components -> controller.
    ``head`` (test-only trigger scoring head) is attached before the controller exists."""
    base = from_weights(deepcopy(base_state), enabled=True)
    base.eval()
    job = dict(track2_backend=True, track2_config=cfg, arm=arm, track2_state=deepcopy(worker), noise_seed=SEED)
    policy, factory = tb.episode_components(job, base, controller_class(Controller))
    if head is not None:
        policy.trigger.trigger_head = head(policy.trigger)
    return policy, factory(policy, torch.Generator().manual_seed(SEED), True)


def evaluation(native, arm, head=None):
    policy = runner.eval_policy(native, arm, UPDATE, SEED)
    if head is not None:
        policy.trigger.trigger_head = head(policy.trigger)
    controller = controller_class(Controller)(policy, torch.Generator().manual_seed(SEED), True)
    return policy, runner.wrap_controller(controller, policy)


def buggy(native, arm, head=None):
    """Pre-fix eval wrapper (observe WITHOUT native_commands): the negative control."""
    policy = runner.eval_policy(native, arm, UPDATE, SEED)
    if head is not None:
        policy.trigger.trigger_head = head(policy.trigger)
    controller = controller_class(Controller)(policy, torch.Generator().manual_seed(SEED), True)
    original = controller.apply

    def apply(rows, commands, now):
        policy.observe(rows, now, command_commitments=controller.active_plans)
        return original(rows, commands, now)
    controller.apply = apply
    return policy, controller


def drive(controllers, scene, natives, ticks, speeds=None):
    logs = [[] for _ in controllers]
    for now in ticks:
        rows = advance(scene, now-ticks[0])
        if speeds is not None:
            rows = [dict(r, v_mps=speeds(now).get(r['vid'], r['v_mps'])) for r in rows]
        for log, c in zip(logs, controllers):
            requests = c.apply(rows, dict(natives), now)
            log.append(canon(dict(requests=requests, event=c.events[-1])))
    return logs


def pair_speeds(now):
    return {'a': 18.4-.02*(now-300), 'b': 16.5-.05*(now-300), 'c': 20.}


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_lane_drop_moves_issue_identical_commands_in_training_and_evaluation(cfg, fixed, tmp_path, arm):
    c4 = ld_cfg(cfg, v2v=False)
    root, base_state, worker = write_root(tmp_path, c4, arm)
    native = dict(c4, run_root=str(root))
    head = lambda trigger: Prefer(trigger, {'ld_kind_closing_pair': 50.})  # noqa: E731
    (pt, ct), (pe, ce), (_, cb) = (training(c4, arm, base_state, worker, head), evaluation(native, arm, head),
                                   buggy(native, arm, head))
    assert pe.lane_drop is not None
    assert canon({k: v for k, v in pe.worker_state()['layer'].items()}) == canon(pt.worker_state()['layer'])
    natives = {'a': .4, 'b': -.2, 'c': .1}
    train, evaluate, old = drive((ct, ce, cb), pair_scene(), natives, list(range(300, 344)), pair_speeds)
    assert [e['move'].family for e in pt.executor.active] == [e['move'].family for e in pe.executor.active]
    assert evaluate == train                                    # identical commands every second
    started = json.loads(train[0])['event']['sample']['track2']['active_moves']
    assert [m['move']['family'] for m in started] == ['closing_pair']
    assert old != train                                         # the pre-fix wrapper diverged (defect caught)
    member = json.loads(train[5])['requests']['a']['requested_raw']
    assert member != pytest.approx(natives['a']+.25)            # the lane-drop law, not native + mean


def test_plain_a_issues_identical_commands(cfg, tmp_path):
    root, base_state, _ = write_root(tmp_path, cfg, 'A')
    native = dict(cfg, run_root=str(root))
    _, ct = training(cfg, 'A', base_state, None)
    pe, ce = evaluation(native, 'A')
    assert not isinstance(pe, tb.EpisodePolicy)
    natives = {'a': .4, 'b': -.2, 'c': .1}
    train, evaluate = drive((ct, ce), pair_scene(), natives, list(range(300, 320)))
    assert evaluate == train


def test_rsu_head_is_restored_and_rsu_suggestions_match(cfg, fixed, tmp_path):
    c4 = ld_cfg(cfg, rsu=dict(range_m=1000.))
    root, base_state, worker = write_root(tmp_path, c4, 'B')
    assert 'rsu' in worker
    native = dict(c4, run_root=str(root))
    with pytest.raises(KeyError):                               # the clean3 loader drops the RSU head
        production.load_policy(production.checkpoint_weights(native, 'B', UPDATE), native)
    (pt, ct), (pe, ce) = training(c4, 'B', base_state, worker), evaluation(native, 'B')
    assert canon(pe.rsu.head.state_dict()) == canon(pt.rsu.head.state_dict())
    natives = {'a': .3, 'b': -.1, 'c': .2}
    train, evaluate = drive((ct, ce), rsu_scene(), natives, list(range(300, 330)))
    assert evaluate == train


def test_n9_families_identical_when_enabled(cfg, monkeypatch, tmp_path):
    # same package object as this module (pytest may import it as ``coopmwm`` or ``tools.coopmwm``)
    nf = pytest.importorskip(f'{__package__}.nm_families', reason='N9 (c4/nm-families) not merged into this tree')
    tnf = importlib.import_module(f'{__package__}.test_nm_families_cpu')

    def common(spec, rng):
        depth = {'pinned_lo': nf.DEPTH[0], 'pinned_hi': nf.DEPTH[1]}.get(spec.depth, .6)
        return dict(tnf.FIXED[spec.name]), 0, 40, depth
    monkeypatch.setattr(nf, '_sample_common', common)
    c4 = tnf.fam_cfg(cfg, dict(platoon_headway=4))
    root, base_state, worker = write_root(tmp_path, c4, 'B')
    native = dict(c4, run_root=str(root))
    head = lambda trigger: tnf.Prefer(trigger, tnf.initiate_then_join('platoon_headway', 'head'))  # noqa: E731
    (_, ct), (_, ce) = training(c4, 'B', base_state, worker, head), evaluation(native, 'B', head)
    scene = [tnf.row('p1', -300, 1, v=20.), tnf.row('p2', -330, 1, v=20.)]
    train, evaluate = drive((ct, ce), scene, dict(p1=.3, p2=-.2), list(range(300, 306)))
    assert evaluate == train
    statuses = [json.loads(x)['event']['sample']['track2']['nm_families']['commands'].get('p2', {}).get('status')
                for x in train]
    assert 'cacc' in statuses                                   # the N9 law actually executed
