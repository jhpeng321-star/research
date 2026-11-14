from concurrent.futures import ThreadPoolExecutor
from threading import Event

from tools import recover_damping_v2_startup as recovery


def test_real_byte_zero_lock_serializes_without_appending_locked_byte(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    entered = Event()

    def contender():
        with recovery.fixed_launch_lock():
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with recovery.fixed_launch_lock():
            pending = pool.submit(contender)
            assert not entered.wait(0.1)
        pending.result(timeout=4)
    assert entered.is_set()
    assert (tmp_path / "runs/probe/.damping_v2_sumo_launch.lock").read_bytes() == b"0"
