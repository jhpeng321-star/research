"""S-MAPPOWZL plateau stop rule (MAIN 04:2x): replay on S-MAPPOWZ, synthetic trends, live watcher."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("nm_mappo_plateau", ROOT / "tools/nm_mappo/plateau_watch.py")
P = importlib.util.module_from_spec(spec)
sys.modules["nm_mappo_plateau"] = P
spec.loader.exec_module(P)
SPEED_LOG = Path("E:/research-dense/runs/active/C4-V7FT-20261017/mappo_wz/run2/production/log.jsonl")


def test_replay_on_s_mappowz_stops_at_48():
    if not SPEED_LOG.exists():
        pytest.skip("S-MAPPOWZ mirror log not present")
    res = P.replay(SPEED_LOG)
    assert res["iteration"] == 48 and res["stop"] and res["window"] == [25, 48]
    assert res["ci95"][0] <= 0 <= res["ci95"][1]


def test_rule_points_and_trends():
    rising = {k: -800 + 2.0 * k + (1 if k % 2 else -1) for k in range(1, 169)}
    assert P.check(rising, 47) is None and P.check(rising, 50) is None
    assert P.check(rising, 48)["stop"] is False
    assert all(not P.check(rising, k)["stop"] for k in range(48, 169, 6))
    falling = {k: -700 - 1.0 * k for k in range(1, 169)}
    assert P.check(falling, 48)["stop"] is True
    b, lo, hi = P.ols_ci(list(range(24)), [3.0 * x + 1 for x in range(24)])
    assert abs(b - 3.0) < 1e-12 and abs(lo - 3.0) < 1e-9 and abs(hi - 3.0) < 1e-9


def test_live_watcher_drops_stop_file_at_first_plateau_check(tmp_path):
    prod = tmp_path / "run/production"
    prod.mkdir(parents=True)
    log = prod / "log.jsonl"
    log.write_text(json.dumps({"type": "config"}) + "\n")
    values = {k: -800 + 3.0 * k for k in range(1, 25)}            # rising until 24 ...
    values.update({k: -700.0 + 5.0 * (-1) ** k for k in range(25, 61)})  # ... flat (noisy) afterwards

    def writer():
        for k in range(1, 61):
            with log.open("a") as fh:
                fh.write(json.dumps({"type": "episode", "iteration": k}) + "\n")
                fh.write(json.dumps({"type": "iter", "iteration": k, "mean_episode_return": values[k]}) + "\n")
            time.sleep(0.1)                                   # the watcher polls every 2 ms
            if (prod / "STOP_REQUESTED").exists():
                (prod / f"EXIT_production_{k}.json").write_text("{}")
                return
        (prod / "EXIT_production_end.json").write_text("{}")

    t = threading.Thread(target=writer)
    t.start()
    assert P.watch(tmp_path, 0.002, do_post=False) == 0
    t.join()
    stop = json.loads((tmp_path / "run/PLATEAU_STOP.json").read_text())
    checks = sorted(p.name for p in (tmp_path / "run").glob("PLATEAU_CHECK_*.json"))
    assert checks == [] and stop["iteration"] == 48 and stop["stop"] and stop["window"] == [25, 48]
    assert (prod / "STOP_REQUESTED").read_text().startswith("plateau rule stop at it48")
