"""Heartbeat accounting must distinguish finished CPU panels and unused GPU arms."""
from tools.c1switch2 import watch


def test_scan_is_read_only_and_scope_is_explicit():
    compile(watch.SCAN, '<watch-scan>', 'exec')
    assert 'os.scandir' not in watch.SCAN
    assert 'subprocess' not in watch.SCAN
    assert 'kill(' not in watch.SCAN
    assert 'write' not in watch.SCAN


def test_summary_does_not_count_unclaimed_arms_as_finished():
    snapshot = dict(hosts=dict(cpuQ=dict(cpu=dict(
        cpu=dict(process=dict(state='EXITED'), episodes=84, decisions=0,
                 charged_sim_s=100800, closed_actual_sim_s=100800),
        wm=dict(process=dict(state='S'), episodes=2, decisions=88,
                charged_sim_s=3560, closed_actual_sim_s=2400),
        bb=dict(state='UNCLAIMED'))),
        gpuR=dict(cpu=dict(cpu=dict(process=dict(state='EXITED'), episodes=28,
            decisions=0, charged_sim_s=33600, closed_actual_sim_s=33600)))), alerts=[])
    text = watch.summary(snapshot)
    assert 'cpu:episodes=112,decisions=0,charged=134400,closedactual=134400' in text
    assert 'wm:episodes=2,decisions=88,charged=3560,closedactual=2400' in text
    assert 'bb:' not in text


def test_snapshot_refuses_existing_staged_peer_work(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return type('Result', (), dict(stdout='peer_file.md\n'))()

    monkeypatch.setattr(watch.subprocess, 'run', run)
    import pytest
    with pytest.raises(RuntimeError, match='preserves index'):
        watch.commit_snapshot({})
    assert calls == [['git', 'diff', '--cached', '--name-only']]


def test_start_refuses_duplicate_claim_before_spawning(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, 'WATCH', tmp_path)
    (tmp_path / 'RUN_ONCE.json').write_text('{}')
    import pytest
    with pytest.raises(RuntimeError, match='never duplicate'):
        watch.start()


def test_monitor_replacement_refuses_live_original(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(watch, 'WATCH', tmp_path)
    (tmp_path / 'DISPATCH.json').write_text('{"pid":27200}')
    monkeypatch.setattr(watch, 'local_process', lambda pid: dict(state='LIVE'))
    with pytest.raises(RuntimeError, match='must be exited'):
        watch.start(resume=True)
    assert not (tmp_path / 'RESUME_ONCE.json').exists()
