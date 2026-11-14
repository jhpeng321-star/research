"""Refusal cases: missing checkpoint, sha mismatch, K overflow, horizon,
unknown action via the score interface."""
from __future__ import annotations

import pytest

from wm_api.errors import (CheckpointError, CheckpointNotFoundError,
                           CheckpointShaMismatch, HorizonError,
                           KOverflowError, PlanSchemaError, StateSchemaError,
                           UnknownActionError)
from wm_api.state import ObstacleRecord
from wm_api.checkpoint import sha256_of

from _tiny import default_plans, tiny_scorer, tiny_state


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("refusal_ckpt")
    tiny_scorer(tmp, k=2)
    return tmp


def test_missing_checkpoint_directory(tmp_path):
    from wm_api import WmScorer

    with pytest.raises(CheckpointNotFoundError):
        WmScorer(tmp_path / "nope")


def test_checkpoint_without_members(tmp_path):
    from wm_api import WmScorer

    with pytest.raises(CheckpointNotFoundError):
        WmScorer(tmp_path)


def test_sha_sums_mismatch_refuses(ckpt, tmp_path_factory):
    import shutil
    from wm_api import WmScorer

    bad = tmp_path_factory.mktemp("bad_sha")
    for name in ("member_1.pt", "member_2.pt"):
        shutil.copyfile(ckpt / name, bad / name)
    good = sha256_of(bad / "member_1.pt")
    (bad / "SHA256SUMS").write_text(
        f"{good}  member_1.pt\n"
        f"{'0' * 64}  member_2.pt\n", encoding="utf-8")
    with pytest.raises(CheckpointShaMismatch):
        WmScorer(bad)
    (bad / "SHA256SUMS").write_text(
        f"{good}  member_1.pt\n"
        f"{sha256_of(bad / 'member_2.pt')}  member_2.pt\n", encoding="utf-8")
    scorer = WmScorer(bad)
    assert scorer.checkpoint["sha_verified"] is True


def test_sha_sums_missing_listed_file(ckpt, tmp_path_factory):
    import shutil
    from wm_api import WmScorer

    bad = tmp_path_factory.mktemp("sha_missing_file")
    shutil.copyfile(ckpt / "member_1.pt", bad / "member_1.pt")
    (bad / "SHA256SUMS").write_text(
        f"{sha256_of(bad / 'member_1.pt')}  member_1.pt\n"
        f"{'1' * 64}  member_2.pt\n", encoding="utf-8")
    with pytest.raises(CheckpointNotFoundError):
        WmScorer(bad)


def test_corrupt_member_payload_refuses(ckpt, tmp_path_factory):
    import shutil
    from wm_api import WmScorer

    bad = tmp_path_factory.mktemp("corrupt")
    shutil.copyfile(ckpt / "member_1.pt", bad / "member_1.pt")
    (bad / "member_2.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError):
        WmScorer(bad)


def test_horizon_300_default_and_refusal(ckpt):
    from wm_api import WmScorer

    scorer = WmScorer(ckpt, threads=1)
    st = tiny_state()
    with pytest.raises(HorizonError):
        scorer.score(st, default_plans(), horizon_s=120)
    with pytest.raises(HorizonError):
        scorer.score(st, default_plans(), horizon_s=0)
    other = scorer.score(st, default_plans(), horizon_s=240,
                         allow_other_horizon=True)
    assert other.horizon_s == pytest.approx(240.0)
    assert other.tau_s == pytest.approx(300.0)      # nearest registered tau


def test_k_overflow_reported_never_dropped(tmp_path_factory):

    tmp = tmp_path_factory.mktemp("overflow_ckpt")
    scorer = tiny_scorer(tmp, k=1)
    n = 130
    st = tiny_state(n_cav=3, n_hdv=n - 3)
    assert st.n_model_vehicles == n > 128
    with pytest.raises(KOverflowError) as exc:
        scorer.score(st, default_plans(), horizon_s=60, allow_other_horizon=True)
    assert "never silently dropped" in str(exc.value)
    assert str(n) in str(exc.value)
    st2 = tiny_state(n_cav=3, n_hdv=125, obstacles=1)
    st2.obstacles = st2.obstacles + (ObstacleRecord(
        id="extra", lane_phys=1, s_down_m=0.0),)
    assert st2.n_model_vehicles == 130
    with pytest.raises(KOverflowError):
        scorer.score(st2, default_plans(), horizon_s=60, allow_other_horizon=True)


def test_unknown_action_via_score(ckpt):
    from wm_api import WmScorer

    scorer = WmScorer(ckpt, threads=1)
    with pytest.raises(UnknownActionError):
        scorer.score(tiny_state(), [{"actions": {"cav0": "lurch"}}],
                     horizon_s=60, allow_other_horizon=True)


def test_empty_and_invalid_plans_refuse(ckpt):
    from wm_api import WmScorer

    scorer = WmScorer(ckpt, threads=1)
    st = tiny_state()
    with pytest.raises(PlanSchemaError):
        scorer.score(st, [], horizon_s=60, allow_other_horizon=True)
    with pytest.raises(PlanSchemaError):
        scorer.score(st, "not-a-list", horizon_s=60, allow_other_horizon=True)
    with pytest.raises(PlanSchemaError):
        scorer.score(st, [{"actions": {"cav0": "native"}}, 42],
                     horizon_s=60, allow_other_horizon=True)
    with pytest.raises(PlanSchemaError):
        scorer.score(st, default_plans(), horizon_s=60, allow_other_horizon=True, members=[])
    with pytest.raises(StateSchemaError):
        scorer.score({"ids": ["x"]}, default_plans(), horizon_s=60, allow_other_horizon=True)
