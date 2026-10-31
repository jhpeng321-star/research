"""Diagnostic monitor cannot confuse admission waits with failed training."""
from copy import deepcopy

from scripts import selmappo2_supervise as supervisor


def parsed():
    return dict(state="PARTIAL", status=dict(timestamp_ns=1_000_000_000_000, alive=True),
                records={18107998: dict(terminal="FAILED"),
                         18107999: dict(terminal="RESULT"),
                         18100000: dict(terminal="RESULT"),
                         18100001: dict(terminal=None)},
                points=[dict(checkpoint_sha256="a" * 64)], reserved_sim_s=3600,
                actual_sim_s=2100, safety=dict(collisions=0, teleports=0), issues=[])


def test_historical_failure_and_open_reservation_are_not_new_failure():
    result = supervisor.summarize(dict(collection="owned"), parsed(), 1001, dict(ProcessId=12432))
    assert result["alerts"] == []
    assert (result["production"], result["open"], result["failures"]) == (1, 1, 1)
    assert result["new_sumo_seconds"] == 0


def test_same_counts_flag_wait_without_authorizing_restart():
    previous = dict(updates=1, production=1)
    result = supervisor.summarize(dict(collection="owned"), parsed(), 1001, {}, previous)
    assert "RESOURCE_WAIT_OR_STALL" in result["alerts"]
    assert "NEW_TRAINING_FAILURE" not in result["alerts"]


def test_stale_remote_and_absent_processes_are_visible():
    data = parsed()
    data["status"]["alive"] = False
    result = supervisor.summarize(dict(collection="owned"), data, 2000, None)
    assert set(result["alerts"]) == {"STALE_OR_FUTURE_REMOTE_OBSERVATION",
                                     "WATCHER_ABSENT", "TRAINING_ABSENT"}


def test_new_failure_log_and_progress_regression_are_visible():
    data = deepcopy(parsed())
    data["records"][18100001]["terminal"] = "FAILED"
    data["status"]["summaries"] = {"training.log": {"tail": "Traceback (most recent call last)"}}
    result = supervisor.summarize(dict(collection="owned"), data, 1001, {"live": True},
                                  dict(updates=2, production=2))
    assert set(result["alerts"]) == {"NEW_TRAINING_FAILURE", "TRAINING_LOG_ERROR", "REGRESSED_PROGRESS"}


def test_final_complete_accepts_normal_process_exit():
    data = parsed()
    data["state"] = "COMPLETE"
    data["status"]["alive"] = False
    result = supervisor.summarize(dict(collection="owned"), data, 1001, None)
    assert result["alerts"] == []
