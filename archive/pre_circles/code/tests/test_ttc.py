"""Tests for TTC evaluation metrics."""

import tempfile
from pathlib import Path

from evaluation import evaluate_ttc


SAMPLE_FCD = """<?xml version="1.0" encoding="UTF-8"?>
<fcd-export>
    <timestep time="0.00">
        <vehicle id="A" x="0.00" y="0.00" speed="20.00" lane="edge_0" pos="0.00"/>
        <vehicle id="B" x="50.00" y="0.00" speed="18.00" lane="edge_0" pos="50.00"/>
    </timestep>
    <timestep time="1.00">
        <vehicle id="A" x="0.00" y="0.00" speed="20.00" lane="edge_0" pos="20.00"/>
        <vehicle id="B" x="50.00" y="0.00" speed="18.00" lane="edge_0" pos="68.00"/>
    </timestep>
</fcd-export>
"""


def test_ttc_no_violations():
    """With a large gap and small speed difference, TTC stays well above 1.5s."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(SAMPLE_FCD)
        f.flush()
        path = Path(f.name)

    try:
        metrics = evaluate_ttc(path, threshold=1.5)
        assert metrics["total_events"] == 0
        print("test_ttc_no_violations passed")
    finally:
        path.unlink()


def test_ttc_with_violations():
    """Place B only 3m ahead of A with A 5 m/s faster -> TTC = 0.6s."""
    fcd = """<?xml version="1.0" encoding="UTF-8"?>
    <fcd-export>
        <timestep time="0.00">
            <vehicle id="A" x="0.00" y="0.00" speed="20.00" lane="edge_0" pos="0.00"/>
            <vehicle id="B" x="0.00" y="0.00" speed="15.00" lane="edge_0" pos="3.00"/>
        </timestep>
    </fcd-export>
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(fcd)
        f.flush()
        path = Path(f.name)

    try:
        metrics = evaluate_ttc(path, threshold=1.5)
        assert metrics["total_events"] == 1
        print("test_ttc_with_violations passed")
    finally:
        path.unlink()


if __name__ == "__main__":
    test_ttc_no_violations()
    test_ttc_with_violations()
