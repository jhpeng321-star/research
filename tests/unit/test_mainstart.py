"""Amended launch guards: PID reuse, scientific vetoes and no-charge readiness."""
import importlib.util
from pathlib import Path

import pytest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[2]
start = load('mainstart', ROOT / 'scripts/c4/mainstart.py')
try:
    verdict = load('mainstart_verdict', ROOT / 'tools/main14/verdict_snapshot.py')
except FileNotFoundError:
    verdict = load('mainstart_verdict', ROOT.parent / 'c4-main14/tools/main14/verdict_snapshot.py')


def test_concurrency_is_exact_incarnation_and_descendants_only():
    procs = {10: {'ppid': 1, 'start_ticks': '100'}, 11: {'ppid': 10, 'start_ticks': '110'},
             12: {'ppid': 1, 'start_ticks': '120'}}
    lookup = procs.get
    assert start.allowed_pairft(10, {'10': '100'}, lookup)
    assert start.allowed_pairft(11, {'10': '100'}, lookup)
    assert not start.allowed_pairft(10, {'10': '99'}, lookup)
    assert not start.allowed_pairft(12, {'10': '100'}, lookup)
    with pytest.raises(RuntimeError, match='unregistered CUDA'):
        start.gpu_rows_allowed('12, python, 200 MiB', {'10': '100'}, lookup)


def qualification(ratio=1.0, sign='PASS', mag='PASS'):
    return {'core': {'sign:x': sign, 'mag:x': mag}, 'table': {'s1': {'signlate:x': sign}},
            'per_set': {'s1': {'x': {'sign_verdict': sign, 'mag_verdict': mag}}},
            'fid': {'all_pairs': {'ratio': ratio}}}


def candidate(label='14b', ratio=0.9, **kw):
    return {'label': label, 'qualification': qualification(ratio, **kw),
            'actual_sha256': 'abc', 'scored_sha256': 'abc'}


@pytest.mark.parametrize('item', [candidate(sign='FAIL'), candidate(mag='FAIL'), candidate(ratio=1.01),
                                 {**candidate(), 'actual_sha256': 'wrong'}])
def test_swap_veto(item):
    chosen, decisions = start.choose_swap(qualification(), [item], verdict.choose_base, verdict.strict_ratio)
    assert chosen is None
    assert decisions[0]['eligible'] is False


def test_swap_best_ratio_and_stable_tie():
    chosen, _ = start.choose_swap(qualification(), [candidate(), candidate('round16', 0.8)],
                                  verdict.choose_base, verdict.strict_ratio)
    assert chosen['label'] == 'round16'
    chosen, _ = start.choose_swap(qualification(), [candidate('round16'), candidate()],
                                  verdict.choose_base, verdict.strict_ratio)
    assert chosen['label'] == '14b'


def test_incomplete_verdict_does_not_authorise_swap():
    item = candidate()
    del item['qualification']['table']
    del item['qualification']['per_set']
    chosen, decisions = start.choose_swap(qualification(), [item], verdict.choose_base, verdict.strict_ratio)
    assert chosen is None
    assert 'missing' in decisions[0]['error']


def test_later_boundaries_preserve_finetunes_of_already_adopted_base():
    previous = {'sha256': 'abc', 'strict_ttc_ratio': 0.9}
    assert not start.improves_selected_base(candidate(), previous, verdict.strict_ratio)
    assert not start.improves_selected_base({**candidate('round16', 0.95), 'actual_sha256': 'new'},
                                          previous, verdict.strict_ratio)
    assert start.improves_selected_base({**candidate('round16', 0.8), 'actual_sha256': 'better'},
                                      previous, verdict.strict_ratio)


def test_unready_collect_does_not_enter_phase_or_touch_seeds(monkeypatch):
    calls = []

    def unavailable(host, *args):
        calls.append(host)
        raise RuntimeError('host not ready')

    class Loop:
        class Refused(Exception):
            pass

        def __getattr__(self, name):
            pytest.fail(f'unready host must not invoke loop.{name}')

    cfg = {'mainstart': {'collect_hosts': {h: {'python': '/py', 'code_root': '/collect/code'}
                                          for h in ('cpuP', 'cpuQ')}},
           'collect': {'workers': 12, 'work_root': '/collect'}, 'hosts': {'cpu': 'original'}}
    monkeypatch.setattr(start, 'remote_json', unavailable)
    assert not start.collection_ready(cfg, {'native_source_manifest_sha256': 'sha'}, Loop())['ready']
    assert cfg['hosts']['cpu'] == 'original'
    assert calls == ['cpuP', 'cpuQ']


@pytest.mark.parametrize('q_workers', [0, 6, 12])
def test_two_box_readiness_checks_both_and_preserves_pinned_caps(monkeypatch, q_workers):
    calls = []
    def ready(host, *args):
        calls.append(host)
        return {'ready': True, 'workers': 12 if host == 'cpuP' else q_workers}
    class Loop:
        class Refused(Exception):
            pass
        def disk_guard(self, *args):
            return {}
    cfg = {'mainstart': {'collect_hosts': {h: {'python': '/py', 'code_root': '/code'}
                                          for h in ('cpuP', 'cpuQ')}},
           'collect': {'workers': 12, 'work_root': '/collect'}, 'hosts': {'cpu': 'original'},
           'fastloop': {'two_box': True, 'workers_per_box': 7}}
    monkeypatch.setattr(start, 'remote_json', ready)
    result = start.collection_ready(cfg, {'native_source_manifest_sha256': 'sha'}, Loop())
    assert result['ready'] == (q_workers >= 7)
    assert calls == ['cpuP', 'cpuQ']
    assert cfg['hosts']['cpu'] == 'original' and cfg['fastloop']['workers_per_box'] == 7
