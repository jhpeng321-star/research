"""Compare non-PI command/physical accounting traces to preserved pre-CV14 code."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from traffic_shepherd import workflow
from tests.multi_cav_runtime import multi_cav_runtime as _runtime

multi_cav_runtime = _runtime
BASE = (
    Path(__file__).resolve().parents[1] / "output/controller_verification_v14/source_preservation"
)


def stub_retired_fs_module(monkeypatch):
    """The preserved pre-CV14 factory imports the retired FollowerStopper
    module at import time; the comparison overrides create_controller with
    the current factory, so a stub satisfies the import without reviving FS."""
    monkeypatch.setitem(sys.modules, "traffic_shepherd.analytical_candidate",
                        SimpleNamespace(ADAPTER_VERSION="retired", LAW_VERSION="retired",
                                        AnalyticalFollowerStopper=object))


def load(name, file):
    spec = importlib.util.spec_from_file_location("traffic_shepherd." + name, BASE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("controller", ["native", "damping", "pi-external-u"])
def test_default_commands_and_costs_match_preserved_workflow(
    tmp_path, monkeypatch, multi_cav_runtime, controller
):
    stub_retired_fs_module(monkeypatch)
    old = load("_cv14_old_workflow", "workflow.py")
    # Resolve transitive source pins against the original runtime location.
    old.__file__ = workflow.__file__
    for key in ["start_sumo", "close_sumo", "observation_map", "SumoLocalSensor", "build_scenario"]:
        setattr(old, key, getattr(workflow, key))
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    kwargs = dict(
        seed=4107,
        loading=2000,
        penetration=0.2,
        controller_name=controller,
        duration=3,
        desired_speed_mps=31.29 if controller == "pi-external-u" else None,
    )
    prior = old.evaluate(output=old_dir, **kwargs)
    old_writes = list(multi_cav_runtime["writes"])
    multi_cav_runtime.update(time=0.0, writes=[], closed=False, callbacks=[])
    current = workflow.evaluate(output=new_dir, **kwargs)
    assert multi_cav_runtime["writes"] == old_writes
    for file in ["telemetry.jsonl", "intervals.jsonl"]:
        assert [json.loads(line) for line in (old_dir / file).read_text().splitlines()] == [
            json.loads(line) for line in (new_dir / file).read_text().splitlines()
        ]
    for key in [
        "totals",
        "minimum_closing_ttc_s",
        "command_counts",
        "fallback_counts",
        "population",
        "system_seconds_per_requested_vehicle",
        "source_wait_vehicle_seconds",
    ]:
        assert current[key] == prior[key]
