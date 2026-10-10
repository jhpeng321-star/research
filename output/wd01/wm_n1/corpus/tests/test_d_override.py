"""Owner-approved D: output override (FEAT1 blocker 5).

``--allow-d-output-root D:/rd-staging/wmn1_feat`` must permit writing ONLY
under that exact root while ``WD_FORBID_D=1`` keeps blocking every other
D: path; D:/research-dense (and D:/research-dense-backups) are never
overridable.  Policy-level checks touch no D: file; the audit-hook checks
rely on the hook firing before any real I/O.
"""
import os

import pytest

from forbid_d_hook import (ForbiddenDriveDError, allow_root, allowed_roots,
                           assert_not_d, install, is_d_path,
                           reset_allowed_roots)

ROOT = "D:/rd-staging/wmn1_feat"


@pytest.fixture(autouse=True)
def _guard():
    reset_allowed_roots()
    yield
    reset_allowed_roots()


def test_allow_root_registers_normalized():
    got = allow_root("d:\\rd-staging\\wmn1_feat\\")
    assert got == "D:/rd-staging/wmn1_feat"
    assert allowed_roots() == ("D:/rd-staging/wmn1_feat",)
    allow_root(ROOT)
    assert allowed_roots() == ("D:/rd-staging/wmn1_feat",)   # idempotent


def test_allowed_root_paths_are_exempt_for_the_hook():
    allow_root(ROOT)
    install()
    under = f"{ROOT}/shards/chunk4/w8661m2600p10t240__w-mid-early/veh_step.npy"
    assert is_d_path(under)                    # still recognised as D:
    assert not is_d_path("E:/research-dense/runs/x")
    assert_not_d(under, "open")                # no raise: exempt
    # the installed hook must NOT fire for a path under the approved root
    # (FileNotFoundError == the open went through to the OS)
    try:
        open(os.path.join(under, "definitely_missing_probe"), "rb")
    except ForbiddenDriveDError:
        pytest.fail("hook fired inside the approved root")
    except OSError:
        pass


def test_any_other_d_path_refused():
    allow_root(ROOT)
    install()
    for bad in (f"{ROOT}_evil/x", f"{ROOT}2/x", "D:/rd-staging/x",
                "D:/rd-staging/x/y", "D:/tmp/x", "D:/x",
                "D:/rd-staging/wmn1_feat/../other"):
        with pytest.raises(ForbiddenDriveDError):
            assert_not_d(bad, "open")
        with pytest.raises(ForbiddenDriveDError):
            open(os.path.join(bad, "probe"), "rb")


def test_research_dense_refused():
    allow_root(ROOT)
    install()
    for bad in ("D:/research-dense", "D:/research-dense/runs/x",
                "D:/research-dense-backups/fs_removal_20260925",
                "d:\\RESEARCH-DENSE\\x"):
        with pytest.raises(ForbiddenDriveDError):
            assert_not_d(bad, "open")
        with pytest.raises(ForbiddenDriveDError):
            open(os.path.join(bad, "probe"), "rb")


def test_allow_root_validated():
    for bad in ("D:/research-dense", "D:/research-dense/sub",
                "D:/research-dense-backups", "D:/", "D:", "C:/rd-staging",
                "E:/rd-staging/wmn1_feat", "relative/path",
                "D:/rd-staging/../research-dense"):
        with pytest.raises(ValueError):
            allow_root(bad)
    assert allowed_roots() == ()


def test_no_override_blocks_everything():
    install()
    assert allowed_roots() == ()
    with pytest.raises(ForbiddenDriveDError):
        assert_not_d(f"{ROOT}/x", "open")
    with pytest.raises(ForbiddenDriveDError):
        assert_not_d("D:/research-dense/x", "open")


def test_activate_d_output_override_sets_env(monkeypatch):
    from corpus import _util as U

    monkeypatch.delenv("WD_FORBID_D", raising=False)
    monkeypatch.delenv("WD_D_ALLOWED_ROOT", raising=False)
    got = U.activate_d_output_override(ROOT)
    assert got == "D:/rd-staging/wmn1_feat"
    assert os.environ["WD_FORBID_D"] == "1"
    assert os.environ["WD_D_ALLOWED_ROOT"] == "D:/rd-staging/wmn1_feat"
    with pytest.raises(ValueError):
        U.activate_d_output_override("D:/research-dense")


def test_apply_env_d_policy_in_worker(monkeypatch):
    from corpus import _util as U

    monkeypatch.setenv("WD_FORBID_D", "1")
    monkeypatch.setenv("WD_D_ALLOWED_ROOT", ROOT)
    U.apply_env_d_policy()
    assert allowed_roots() == ("D:/rd-staging/wmn1_feat",)
    assert_not_d(f"{ROOT}/x", "open")          # exempt in this process too


def test_featurize_all_refuses_d_out_without_override(tmp_path):
    from corpus.tests import synth
    from corpus import featurize_all as FA

    inv = synth.make_inventory(tmp_path / "inv.parquet", synth.default_runs())
    with pytest.raises((PermissionError, ForbiddenDriveDError)):
        FA.featurize_all(inv, tmp_path, f"{ROOT}/shards", workers=1)


def test_pack_refuses_d_out_without_override(tmp_path):
    from corpus.tests import synth
    from corpus import pack as PK

    data = tmp_path / "shards"
    synth.write_shards(data)
    with pytest.raises((PermissionError, ForbiddenDriveDError)):
        PK.pack_shards(data, f"{ROOT}/pack", workers=1)
