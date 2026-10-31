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


def test_gpu_stop_closes_cpu_waiters_without_overwriting_first_cause(tmp_path,monkeypatch):
    import contextlib
    import io
    import json
    from tools.c1switch import transport
    cpu=tmp_path/'cpu/rpc';gpu=tmp_path/'gpu/rpc'
    for root in (cpu,gpu):
        for a in ('wm','bb'):(root/a).mkdir(parents=True)
    for a in ('wm','bb'):write(gpu/a/'STOP',dict(reason='combined cap'))
    write(cpu.parent/'STOP',dict(reason='first causal receipt'))
    monkeypatch.setattr(transport,'CPU',str(cpu))
    monkeypatch.setattr(transport,'GPU',str(gpu))
    def local_ssh(host,code):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):exec(code,{})
        return json.loads(output.getvalue())
    monkeypatch.setattr(transport,'ssh',local_ssh)
    transport.markers()
    assert json.loads((cpu.parent/'STOP').read_text())['reason']=='first causal receipt'
    assert all(json.loads((cpu/a/'STOP').read_text())['reason']=='combined cap' for a in ('wm','bb'))


def test_actual_gpu_segments_keep_every_phase_without_ceil_refund():
    totals=finish.actual_gpu_accounting(dict(elapsed_s=0,cumulative_gpu_s=10279.222073864192))
    assert totals['total_s']==pytest.approx(10265.216253813356,abs=1e-8)
    assert totals['conservative_total_s']-totals['total_s']==pytest.approx(14.005820050836,abs=1e-8)
    assert len(totals['segments_s'])==8
