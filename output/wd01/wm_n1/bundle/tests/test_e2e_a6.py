"""The bundle executor refuses to train with ``state_global_token`` ON when the
frozen Amendment 6 record is missing (A6).  Synthetic pack, CPU only.
"""
from __future__ import annotations

import pytest

from train import idea_a as ia


def test_run_config_job_refuses_when_a6_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ia._seals, "AMENDMENT_6_PATH",
                        tmp_path / "missing_a6.md")
    from bundle import e2e

    with pytest.raises(ia._seals.Amendment6Error):
        e2e.run_config_job("box-member-1", "member-1", work_dir=tmp_path)
