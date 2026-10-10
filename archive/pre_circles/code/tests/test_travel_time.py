"""Tests for travel-time evaluation metrics."""

import tempfile
from pathlib import Path

from evaluation import evaluate_travel_time, percentile


SAMPLE_TRIPINFO = """<?xml version="1.0" encoding="UTF-8"?>
<tripinfos>
    <tripinfo id="main_1" depart="0.00" departLane="1191554636_0" arrival="50.00" duration="50.00" routeLength="1000.00"/>
    <tripinfo id="main_2" depart="5.00" departLane="1191554636_1" arrival="65.00" duration="60.00" routeLength="1000.00"/>
    <tripinfo id="main_3" depart="10.00" departLane="1191554636_0" arrival="80.00" duration="70.00" routeLength="1000.00"/>
    <tripinfo id="ramp_1" depart="2.00" departLane="11553048#1_0" arrival="32.00" duration="30.00" routeLength="500.00"/>
    <tripinfo id="ramp_2" depart="7.00" departLane="11553048#1_0" arrival="52.00" duration="45.00" routeLength="500.00"/>
</tripinfos>
"""


def test_percentile():
    assert percentile([10, 20, 30, 40, 50], 90) == 50.0
    assert percentile([10, 20, 30, 40], 50) == 20.0
    assert percentile([10, 20, 30, 40], 90) == 40.0
    print("test_percentile passed")


def test_travel_time_metrics():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(SAMPLE_TRIPINFO)
        f.flush()
        path = Path(f.name)

    try:
        metrics = evaluate_travel_time(path)
        assert metrics["all"]["count"] == 5
        assert metrics["mainline"]["count"] == 3
        assert metrics["ramp"]["count"] == 2
        assert metrics["mainline"]["p90"] == 70.0
        assert metrics["ramp"]["p90"] == 45.0
        print("test_travel_time_metrics passed")
    finally:
        path.unlink()


if __name__ == "__main__":
    test_percentile()
    test_travel_time_metrics()
