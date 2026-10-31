"""Terminal reporting must never turn an interrupted subset into efficacy."""
import pytest

from tools.c1switch import finish
from tools.c1switch.storage import write


def test_terminal_report_refuses_live_owner(tmp_path,monkeypatch):
    monkeypatch.setattr(finish,'EVIDENCE',tmp_path)
    write(tmp_path/'V10_COLLECTION.json',dict(all_cpu_owned_absent=False,gpu=dict(owned_pid_alive=False)))
    with pytest.raises(ValueError,match='closed owned jobs'):
        finish.report()


def test_incomplete_terminal_report_does_not_bootstrap_subset(tmp_path,monkeypatch):
    from tools.c1switch import analyze
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(finish,'EVIDENCE',tmp_path)
    monkeypatch.setattr(finish,'MIRROR',tmp_path/'mirror')
    monkeypatch.setattr(analyze,'summarize',lambda *a:pytest.fail('partial subset must not be bootstrapped'))
    (tmp_path/'docs/paper_results').mkdir(parents=True)
    write(tmp_path/'mirror/STOP',dict(reason='combined owner GPU budget exhausted'))
    write(tmp_path/'V10_COLLECTION.json',dict(all_cpu_owned_absent=True,
        gpu=dict(owned_pid_alive=False,**{'GPU_FINISHED.json':dict(cumulative_gpu_s=14396.3)}),
        charged_sim_s=250000,complete_episodes=10,complete_physics=20,complete_branches=233))
    finish.report()
    text=(tmp_path/'docs/paper_results/C1SWITCH.md').read_text(encoding='utf-8')
    assert 'no full-panel efficacy verdict' in text
    assert 'not evaluated' in text
    assert 'combined owner GPU budget exhausted' in text
    assert not (tmp_path/'docs/paper_results/c1switch/main_results.json').exists()
