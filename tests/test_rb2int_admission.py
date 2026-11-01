from tools.rb2int.admission import available_memory


def test_idle_archive_cache_is_reclaimable_without_touching_a_peer_process():
    assert available_memory(limit=100, current=95, inactive_file=60, dirty=5,
                            writeback=5, host_available=80) == 55


def test_resident_or_dirty_memory_is_not_counted_as_available():
    assert available_memory(limit=100, current=95, inactive_file=0, host_available=1000) == 5
    assert available_memory(limit=100, current=95, inactive_file=30, dirty=30, host_available=1000) == 5
    assert available_memory(limit=100, current=105, inactive_file=10, host_available=1000) == 5


def test_memory_admission_is_bounded_by_host_and_quota_even_during_snapshot_races():
    assert available_memory(limit=100, current=90, inactive_file=1000, host_available=1000) == 100
    assert available_memory(limit=100, current=90, inactive_file=50, host_available=8) == 8
    assert available_memory(limit=None, current=90, inactive_file=50, host_available=8) == 8
