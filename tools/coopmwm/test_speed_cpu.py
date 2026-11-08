"""c4/speed CPU contracts: G4 pool size (default 8, configurable), learn-step torch threads (opt-in, float-tolerance
equivalent, restored), chain/prepare plumbing; COOPMWM_SPEED_BENCH=1 prints the micro benchmark."""
# ruff: noqa: F811 (imported pytest fixtures are test parameters)
import os
import time
from copy import deepcopy

import pytest
import torch

from . import clean5b_restart as c5b, launch_training as lt, track2_backend as tb
from .test_track2_backend_cpu import cfg, cpu_threads, fake_load, reward_result, sample  # noqa: F401


def test_c_overrides_g4_default_8_configurable_and_learn_threads():
    o = c5b._c_overrides()
    assert o['g4_workers'] == 8 and o['g4_mem_available_min_gb'] == c5b.F8_G4_MEM_MIN_GB
    assert 'learn_threads' not in o                                          # measured slower: off by default
    assert c5b._c_overrides(8, 16)['learn_threads'] == dict(rule=lt.LEARN_THREADS_RULE, max=16,
                                                            reserve=c5b.F8_WAVE_WORKERS)
    assert c5b._c_overrides(6, 4)['g4_workers'] == 6 and c5b._c_overrides(6, 4)['learn_threads']['max'] == 4
    keep = {k: v for k, v in o.items() if k != 'g4_workers'}
    assert keep == {k: v for k, v in c5b.VARIANT_F8['cfg_overrides'].items() if k != 'g4_workers'}
    assert c5b.VARIANT_F8['cfg_overrides']['g4_workers'] == 4               # F8/F8S unchanged
    from .g4_runtime import g4_settings
    assert g4_settings(dict(g4_workers=8, g4_mem_available_min_gb=20)) == (8, 20)


def test_prepare_c_passes_speed_overrides(monkeypatch):
    seen = {}
    monkeypatch.setattr(c5b.c5, 'prepare', lambda source, *, root, variant, **kw: seen.update(v=variant, kw=kw))
    c5b.prepare_c('src', root='r', contexts=[])
    assert seen['v']['cfg_overrides']['g4_workers'] == 8 and seen['kw'] == dict(contexts=[])
    c5b.prepare_c('src', root='r', g4_workers=5, learn_threads_max=8)
    assert seen['v']['cfg_overrides']['g4_workers'] == 5 and seen['v']['cfg_overrides']['learn_threads']['max'] == 8
    assert set(seen['v']) == set(c5b.c5.VARIANT)


def test_learn_threads_value_and_validation(monkeypatch):
    assert lt.learn_threads({}) is None                                     # absent: unchanged 1-thread path
    monkeypatch.setattr(lt.os, 'cpu_count', lambda: 48)
    v = dict(rule=lt.LEARN_THREADS_RULE, max=16, reserve=8)
    assert lt.learn_threads(dict(learn_threads=v)) == 16
    monkeypatch.setattr(lt.os, 'cpu_count', lambda: 20)
    assert lt.learn_threads(dict(learn_threads=v)) == 12                    # cores - active workers
    monkeypatch.setattr(lt.os, 'cpu_count', lambda: 4)
    assert lt.learn_threads(dict(learn_threads=v)) == 1
    for bad in (dict(v, rule='x'), dict(v, max=0), dict(v, reserve=-1), dict(v, extra=1), dict(v, max=2.)):
        with pytest.raises(ValueError, match='learn_threads'):
            lt.learn_threads(dict(learn_threads=bad))


def test_threads_context_sets_and_restores():
    torch.set_num_threads(1)
    with lt._threads(3):
        assert torch.get_num_threads() == 3
    assert torch.get_num_threads() == 1
    with pytest.raises(RuntimeError):
        with lt._threads(2):
            raise RuntimeError('x')
    assert torch.get_num_threads() == 1
    with lt._threads(None):
        assert torch.get_num_threads() == 1


def _episodes(cfg, n):
    _, s, _ = sample(cfg)
    episodes, rows = [], []
    for k in range(n):
        episodes.append(dict(id=f'ep{k}', arm='B', steps=[dict(time_s=300, sample=deepcopy(s))]))
        for r in reward_result(cfg, s)['rows']:
            rows.append(dict(r, episode_id=f'ep{k}'))
    return episodes, dict(reward_result(cfg, s), rows=rows)


def _train(cfg, monkeypatch, threads, n):
    fake_load(monkeypatch)
    torch.manual_seed(0)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    runtime.safety_checked = True
    monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([], {}))
    episodes, envelope = _episodes(cfg, n)
    t = time.perf_counter()
    with lt._threads(threads):
        report = runtime.train(episodes, rewards=envelope)
    elapsed = time.perf_counter()-t
    params = [p.detach().clone() for p in runtime.policy.parameters()]
    return report, params, elapsed


def _close(a, b, tol):
    if isinstance(a, dict):
        return set(a) == set(b) and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float):
        return abs(a-b) <= tol*max(1., abs(a))
    return a == b


def test_learning_step_threads_equivalent_within_float_tolerance(cfg, monkeypatch):
    """Seeded small batch: the real track2 train step at 1 thread vs 4 threads -> same receipts (float tolerance)
    and the same parameters (allclose)."""
    r1, p1, _ = _train(cfg, monkeypatch, 1, 12)
    r4, p4, _ = _train(cfg, monkeypatch, 4, 12)
    assert torch.get_num_threads() == 1
    assert _close(r1, r4, 1e-5)
    assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-5) for a, b in zip(p1, p4))


def test_update_uses_learn_threads_only_around_learning(monkeypatch, tmp_path):
    """launch_training.update: train_update / bank / FULLPOL supervision run at learn_threads; collection, safety
    and reward building at 1 thread; A never changes threads."""
    from . import fair_vehicle_log, fullpol_learn, practice_starts, replay_bank, g4_runtime
    seen = {}
    rec = lambda name, value=None: (lambda *a, **k: seen.setdefault(name, []).append(torch.get_num_threads()) or value)  # noqa: E731
    hooks = dict(collect_update=rec('collect', []), safety=rec('safety', {}), train_update=rec('train', {}),
                 save_checkpoint=rec('save', str(tmp_path/'ck.pt')), runtime=object())
    (tmp_path/'ck.pt').write_bytes(b'x')
    monkeypatch.setattr(lt, 'augmentation_receipt', lambda cfg: dict(admitted=dict(B=0, R=0)))
    monkeypatch.setattr(lt.production, 'backend_config', lambda cfg, **k: dict(cfg))
    monkeypatch.setattr(lt, 'health', lambda root: None)
    monkeypatch.setattr(lt, 'wait_for', lambda *a: None)
    monkeypatch.setattr(lt, 'write', lambda *a, **k: None)
    monkeypatch.setattr(lt.p1, 'build_arm_runtime', lambda native, arm: hooks)
    monkeypatch.setattr(lt, '_resume_view', lambda *a: None)
    monkeypatch.setattr(fair_vehicle_log, 'update_step', lambda *a: None)
    monkeypatch.setattr(replay_bank, 'ttc_events', lambda *a: 0)
    monkeypatch.setattr(g4_runtime, 'enabled', lambda *a, **k: False)
    monkeypatch.setattr(lt.p2, 'reward_update', rec('reward', {}))
    monkeypatch.setattr(lt.production, 'bank_move_supervision', rec('bank', dict(final_ce=0.)))
    monkeypatch.setattr(fullpol_learn, 'supervise', rec('fullpol', None))
    monkeypatch.setattr(lt.production, 'update_metrics', lambda *a, **k: {})
    monkeypatch.setattr(practice_starts, 'update_report', lambda e: {})
    monkeypatch.setattr(lt.rc, 'sha', lambda p: '0'*64)
    monkeypatch.setattr(lt.os, 'cpu_count', lambda: 12)
    torch.set_num_threads(1)
    cfg = dict(run_root=str(tmp_path), learn_threads=dict(rule=lt.LEARN_THREADS_RULE, max=16, reserve=8))
    lt.update(cfg, 'B', 1)
    assert seen == dict(collect=[1], safety=[1], reward=[1], train=[4], bank=[4], fullpol=[4], save=[1])
    seen.clear()
    lt.update(dict(cfg, learn_threads=None), 'R', 1)                         # absent: all at 1 thread
    assert seen['train'] == [1] and seen['fullpol'] == [1]
    assert torch.get_num_threads() == 1


@pytest.mark.skipif(os.environ.get('COOPMWM_SPEED_BENCH') != '1', reason='micro benchmark: COOPMWM_SPEED_BENCH=1')
def test_micro_benchmark(cfg, monkeypatch, capsys):
    cfg = dict(cfg, layer_config=dict(cfg['layer_config'], hidden_dim=64))
    n = int(os.environ.get('COOPMWM_SPEED_EPISODES', '200'))
    out = {}
    for threads in (1, 4, 8, 16):
        _, _, out[threads] = _train(cfg, monkeypatch, threads, n)
    with capsys.disabled():
        print('\nSPEED learn step (track2 train, %d single-step episodes, hidden 64): ' % n
              + ', '.join(f'{k} thr {v:.2f}s ({out[1]/v:.2f}x)' for k, v in out.items()))
