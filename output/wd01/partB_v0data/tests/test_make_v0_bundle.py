"""make_v0_bundle.py tests: verified plain-bundle assembly on a tiny
synthetic dataset root (superseded sources' files omitted, entries kept),
plus one CPU --smoke self-test of the zipped bundle."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
WD01 = HERE.parents[1]
if str(WD01) not in sys.path:
    sys.path.insert(0, str(WD01))

import make_v0_bundle as M  # noqa: E402
import build_v03 as B3  # noqa: E402


@pytest.fixture()
def mini_v03(mini_v02, tmp_path):
    """A v0.3 root from the mini v0.2 (copy + supersession applied + card
    regenerated, no new chunks) -- the shape of root the bundler
    consumes."""
    out = tmp_path / "v03"
    rc = B3.main(["--v02-root", str(mini_v02["root"]), "--out", str(out),
                  "--v02-manifest-sha", "", "--skip-chunk3"])
    assert rc == 0
    import make_data_card as DC
    rc = DC.main(["--out", str(out), "--version", "v0.3"])
    assert rc == 0
    return out


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_bundle_assembles_verified_and_omits_superseded(mini_v02, mini_v03,
                                                        tmp_path):
    out_dir = tmp_path / "bundles"
    rc = M.main(["--data-root", str(mini_v03),
                 "--code-tree", str(WD01), "--code-commit", "testcommit",
                 "--tag", "t0", "--out-dir", str(out_dir)])
    assert rc == 0
    zip_path = out_dir / "partB_v0_bundle_t0.zip"
    assert zip_path.is_file()
    side = out_dir / "partB_v0_bundle_t0.zip.sha256"
    assert side.read_text(encoding="utf-8").split()[0] == \
        M.sha256_file(zip_path)

    top = "partB_v0_bundle_t0"
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        assert all(n.startswith(top + "/") for n in names)
        assert not any("__pycache__" in n for n in names)
        # code side: LF endings only
        assert b"\r\n" not in z.read(f"{top}/code/wdpaths.py")
        assert f"{top}/code/partB_v0/data.py" in names
        assert f"{top}/code/part4/partB/losses.py" in names
        # BUNDLE_MANIFEST re-hash over the zip contents
        bm = json.loads(z.read(f"{top}/BUNDLE_MANIFEST.json")
                        .decode("utf-8"))
        assert bm["schema"] == M.SCHEMA and bm["tag"] == "t0"
        assert bm["code_source"].endswith("testcommit")
        assert bm["value_model"] == "net"      # reserved default
        for rel, sha in bm["files"].items():
            assert rel != "BUNDLE_MANIFEST.json"   # never self-listed
            assert _sha_bytes(z.read(f"{top}/{rel}")) == sha, rel
        # data side: MANIFEST entries kept, superseded FILES omitted
        man = json.loads(z.read(f"{top}/data/MANIFEST.json")
                         .decode("utf-8"))
        sup = [k for k, e in man["worlds"].items()
               if e.get("superseded_by")]
        assert len(sup) == 4          # 2 p40 worlds x base+native
        for k, e in man["worlds"].items():
            for rel in e["files"]:
                in_zip = f"{top}/data/{rel}" in names
                if k in sup:
                    assert not in_zip, rel
                else:
                    assert in_zip, rel
                    assert _sha_bytes(z.read(f"{top}/data/{rel}")) == \
                        e["files"][rel]
        assert f"{top}/data/DATA_CARD_V0.md" in names
        # README: template filled, nohup instructions, no stale placeholders
        readme = z.read(f"{top}/README_PLAIN.md").decode("utf-8")
        assert "nohup python -m partB_v0 train --data ../data" in readme
        assert "tail -f" in readme
        assert "apt-get install -y tmux" in readme       # optional note
        assert "tmux new -s" not in readme               # no longer primary
        assert "__TAG__" not in readme and "__ZIP_NAME__" not in readme
        assert "__VALUE_MODEL__" not in readme
        assert "Value model: net." in readme
        assert "partB_v0_bundle_t0.zip" in readme
        assert "partB_v0_bundle_t0/" in readme
        # requirements pins
        req = z.read(f"{top}/requirements_plain.txt").decode("utf-8")
        assert "pandas==2.2.3" in req and "pyarrow==17.0.0" in req
        assert "torch" not in req.split("\n")[-1]
        # no HOLDOUT world files anywhere in the zip
        for n in names:
            assert mini_v02["holdout"] not in n

    rec = json.loads((out_dir / "partB_v0_bundle_t0.record.json")
                     .read_text(encoding="utf-8"))
    assert rec["zip_sha256"] == M.sha256_file(zip_path)
    assert rec["n_superseded_sources_omitted"] == 4
    assert rec["n_holdout_entries"] == 0
    assert rec["n_holdout_worlds_listed"] == 1


def test_bundle_value_model_passthrough(mini_v03, tmp_path):
    """--value-model is a reserved pass-through: recorded in
    BUNDLE_MANIFEST.json and the record, and filled into README_PLAIN.md
    (no stale __VALUE_MODEL__ placeholder)."""
    out_dir = tmp_path / "bundles_vm"
    rc = M.main(["--data-root", str(mini_v03),
                 "--code-tree", str(WD01), "--code-commit", "testcommit",
                 "--tag", "tvm", "--value-model", "gbm",
                 "--out-dir", str(out_dir)])
    assert rc == 0
    with zipfile.ZipFile(out_dir / "partB_v0_bundle_tvm.zip") as z:
        bm = json.loads(z.read("partB_v0_bundle_tvm/BUNDLE_MANIFEST.json")
                        .decode("utf-8"))
        assert bm["value_model"] == "gbm"
        readme = z.read("partB_v0_bundle_tvm/README_PLAIN.md") \
            .decode("utf-8")
        assert "Value model: gbm." in readme
        assert "__VALUE_MODEL__" not in readme
    rec = json.loads((out_dir / "partB_v0_bundle_tvm.record.json")
                     .read_text(encoding="utf-8"))
    assert rec["value_model"] == "gbm"


def test_bundle_refuses_hash_mismatch(mini_v02, tmp_path):
    mini = mini_v02
    bad_root = tmp_path / "bad_root"
    bad_root.mkdir()
    import shutil
    for p in (mini["root"] / "MANIFEST.json",):
        shutil.copy2(p, bad_root / p.name)
    shutil.copytree(mini["root"] / "worlds", bad_root / "worlds")
    victim = sorted((bad_root / "worlds").rglob("*_feat.npz"))[0]
    victim.write_bytes(victim.read_bytes() + b"x")     # corrupt one file
    out_dir = tmp_path / "bundles_bad"
    with pytest.raises(SystemExit, match="hash"):
        M.main(["--data-root", str(bad_root), "--code-tree", str(WD01),
                "--tag", "tb", "--out-dir", str(out_dir)])
    assert not (out_dir / "partB_v0_bundle_tb.zip").exists()


@pytest.mark.parametrize("tag", ["ts"])
def test_bundle_smoke_self_test(mini_v03, tmp_path, tag):
    """Clean-unzip CPU --smoke self-test of the zipped bundle (2 members
    to keep the unit suite fast; the real bundle run uses the default 5
    via the documented command)."""
    out_dir = tmp_path / "bundles_smoke"
    rc = M.main(["--data-root", str(mini_v03),
                 "--code-tree", str(WD01), "--code-commit", "testcommit",
                 "--tag", tag, "--out-dir", str(out_dir),
                 "--python", sys.executable, "--smoke-members", "2",
                 "--smoke-test"])
    assert rc == 0
    rec = json.loads((out_dir / f"partB_v0_bundle_{tag}.record.json")
                     .read_text(encoding="utf-8"))
    smoke = rec["smoke"]
    assert smoke["rc"] == 0 and smoke["pass"] is True
    assert smoke["done_printed"] and smoke["data_verified_printed"]
    assert smoke["output_manifest_rehash"]["bad"] == 0
    assert smoke["results_zip"]["has_ckpt"] is False
