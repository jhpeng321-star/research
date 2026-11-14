"""Chunk-agnostic builder tests: per-chunk append, chunk/fix-flag columns,
speed-mode resolution (receipt -> observed-cav -> observed -> config), and
the chunk-1 backfill migration."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import backfill_chunk1 as BF
import build_v0data as B


def _manifest(out):
    return json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))


def test_chunk_append_flags_and_speed_mode_sources(synth_world, tmp_path):
    out = tmp_path / "v0c"
    common = ["--plan", str(synth_world["plan"]), "--runs",
              str(synth_world["runs"]), "--out", str(out)]

    # chunk A: the pre-fix world (no speed mode in the receipt ->
    # observed from the physics rows)
    rc = B.main(common + ["--chunk", "chunkA",
                          "--worlds", "wsyntm000p02t240"])
    assert rc == 0
    man = _manifest(out)
    assert set(man["chunks"]) == {"chunkA"}
    a_key = "chunkA/wsyntm000p02t240__base"
    assert a_key in man["worlds"]
    assert man["chunks"]["chunkA"]["world_keys"] == \
        sorted(k for k in man["worlds"] if k.startswith("chunkA/"))

    # chunk B: the post-fix world (receipt records speed_mode 27)
    rc = B.main(common + ["--chunk", "chunkB",
                          "--worlds", "wsynfixm2700p40t240"])
    assert rc == 0
    man = _manifest(out)
    assert set(man["chunks"]) == {"chunkA", "chunkB"}
    assert "chunkA/wsyntm000p02t240__base" in man["worlds"]   # untouched

    for chunk, wname, pen in (("chunkA", "wsyntm000p02t240", 0.02),
                              ("chunkB", "wsynfixm2700p40t240", 0.4)):
        for src in ("base", "native"):
            key = f"{chunk}/{wname}__{src}"
            e = man["worlds"][key]
            assert e["chunk_id"] == chunk
            for f in e["files"]:
                assert f.startswith(f"worlds/{chunk}/")
                assert (out / f).exists()
            lab = pd.read_parquet(
                out / next(f for f in e["files"]
                           if f.endswith("_labels.parquet")))
            td = pd.read_parquet(
                out / next(f for f in e["files"]
                           if f.endswith("_td.parquet")))
            for df in (lab, td):
                assert df["chunk_id"].eq(chunk).all()
                assert df["pre_fix"].dtype == bool
                assert df["p40"].dtype == bool
                assert df["p40"].eq(abs(pen - 0.4) < 1e-9).all()
            assert set(lab["pre_fix"].unique()) == \
                {set(lab["cav_speed_mode"].unique()) == {31}}
            td_rows = man["worlds"][key]["rows"]["n_td_rows"]
            assert len(td) == td_rows

    # speed-mode resolution: chunkA observed from the CAV rows (31,
    # pre-fix), chunkB from the receipt (27, not pre-fix)
    ea = man["worlds"]["chunkA/wsyntm000p02t240__base"]
    eb = man["worlds"]["chunkB/wsynfixm2700p40t240__base"]
    assert ea["cav_speed_mode"] == 31 and \
        ea["speed_mode_source"] == "observed-cav"
    assert ea["pre_fix"] is True and ea["p40"] is False
    assert eb["cav_speed_mode"] == 27 and eb["speed_mode_source"] == "receipt"
    assert eb["pre_fix"] is False and eb["p40"] is True

    # npz carries the same scalars
    z = np.load(out / next(f for f in eb["files"]
                           if f.endswith("_feat.npz")))
    assert str(z["chunk_id"]) == "chunkB"
    assert bool(z["pre_fix"]) is False and bool(z["p40"]) is True

    # per-chunk resume: rerunning chunkA appends nothing and leaves files
    e_before = man["worlds"]["chunkA/wsyntm000p02t240__base"]
    blobs = {f: (out / f).read_bytes() for f in e_before["files"]}
    rc = B.main(common + ["--chunk", "chunkA",
                          "--worlds", "wsyntm000p02t240"])
    assert rc == 0
    man2 = _manifest(out)
    assert man2["build"]["n_sources_built_this_run"] == 0
    for f, blob in blobs.items():
        assert (out / f).read_bytes() == blob
    # config default used when neither receipt nor rows record a mode
    mode, source = B.resolve_speed_mode(None, {"smode_values": []}, 27)
    assert (mode, source) == (27, "config")
    mode, source = B.resolve_speed_mode(None, {"smode_values": [31.0]}, None)
    assert (mode, source) == (31, "observed")
    mode, source = B.resolve_speed_mode(
        {"evaluate_summary": {"config": {"speed_mode": 27}}}, None, 31)
    assert (mode, source) == (27, "receipt")


def _write_old_format_source(out, key, chunk="chunk1", pen=0.4, mode=31):
    """Replicate the pre-chunk layout for one source (old files + done
    marker + flat manifest entry)."""
    import v0lib as V
    old = out / "worlds"
    old.mkdir(parents=True, exist_ok=True)
    lab = pd.DataFrame(dict(
        world=[key.rsplit("__", 1)[0]], source=["base"], split=["TRAIN"],
        demand_veh_h=[2700], penetration=[pen], t0_s=[240.0],
        policy=["x"], t=[250.0], tau=[30.0], tau_available=[590.0],
        cav_speed_mode=[mode], total_delay_ref_s=[10.0],
        wait_mainline_s=[1.0], wait_ramp_s=[0.0], ttc_lt3_veh_s=[0.5],
        hard_brake_s=[2.0], collision_any=[0]))
    td = pd.DataFrame(dict(
        world=[key.rsplit("__", 1)[0]], source=["base"], split=["TRAIN"],
        penetration=[pen], t=[250.0], t_next=[280.0], tau=[60.0],
        tau_available=[590.0], d30_s=[10.0], label_mc_s=[21.0],
        label_mc_complete=[1], cav_speed_mode=[mode]))
    f_lab, f_td = old / f"{key}_labels.parquet", old / f"{key}_td.parquet"
    f_npz = old / f"{key}_feat.npz"
    lab.to_parquet(f_lab, index=False)
    td.to_parquet(f_td, index=False)
    np.savez_compressed(f_npz, t=np.array([250.0]),
                        tau_available=np.array([590.0]),
                        cav_speed_mode=np.int32(mode))
    files = {p.name: B.sha_of(p) for p in (f_npz, f_lab, f_td)}
    B.atomic_write_json(old / f"{key}_done.json",
                        dict(files=files, rows=dict(n_td_rows=1),
                             physics_sha256="x", built_utc="old"))
    return dict(world=key.rsplit("__", 1)[0], source="base",
                files=files, rows=dict(n_decision_times=1, n_label_rows=1,
                                       n_td_rows=1),
                trace_stats=dict(smode_values=[float(mode)]),
                physics_sha256="x")


def test_backfill_chunk_flags_resumable(synth_world, tmp_path):
    out = tmp_path / "v0old"
    out.mkdir()
    old_worlds = {"wsyntm000p02t240__base":
                  _write_old_format_source(out, "wsyntm000p02t240__base",
                                           pen=0.4),
                  "wsynterrp02t240__native":
                  _write_old_format_source(out, "wsynterrp02t240__native",
                                           pen=0.02, mode=27)}
    man = dict(schema=B.SCHEMA, zero_simulation=True, worlds=old_worlds,
               skipped_holdout=[dict(world="wsynholdp02t240")],
               skipped_incomplete=[dict(world="wsynterrp02t240",
                                        source="base", reason="x")])
    B.atomic_write_json(out / "MANIFEST.json", man)

    rc = BF.main(["--out", str(out)])
    assert rc == 0
    man = _manifest(out)
    assert set(man["chunks"]) == {"chunk1"}
    sec = man["chunks"]["chunk1"]
    assert sec["n_sources"] == 2
    assert sec["skipped_holdout"][0]["world"] == "wsynholdp02t240"
    # flat world keys re-prefixed; old locations gone
    assert set(man["worlds"]) == {"chunk1/wsyntm000p02t240__base",
                                  "chunk1/wsynterrp02t240__native"}
    assert not (out / "worlds" / "wsyntm000p02t240__base_feat.npz").exists()
    e = man["worlds"]["chunk1/wsyntm000p02t240__base"]
    for f, sha in e["files"].items():
        assert f.startswith("worlds/chunk1/")
        assert B.sha_of(out / f) == sha
    lab = pd.read_parquet(
        out / next(f for f in e["files"] if f.endswith("_labels.parquet")))
    assert lab["chunk_id"].eq("chunk1").all()
    assert lab["pre_fix"].all() and lab["p40"].all()      # pen 0.4, mode 31
    td = pd.read_parquet(
        out / next(f for f in e["files"] if f.endswith("_td.parquet")))
    assert td["chunk_id"].eq("chunk1").all()
    z = np.load(out / next(f for f in e["files"] if f.endswith("_feat.npz")))
    assert str(z["chunk_id"]) == "chunk1" and bool(z["p40"])
    done = json.loads(
        (out / "worlds/chunk1/wsyntm000p02t240__base_done.json")
        .read_text(encoding="utf-8"))
    assert done["flags_backfilled"] is True
    # the not-pre-fix source keeps pre_fix=False
    e2 = man["worlds"]["chunk1/wsynterrp02t240__native"]
    lab2 = pd.read_parquet(
        out / next(f for f in e2["files"] if f.endswith("_labels.parquet")))
    assert not lab2["pre_fix"].any() and not lab2["p40"].any()

    # resumable: second run is a no-op
    before = {f: (out / f).read_bytes() for f in e["files"]}
    rc = BF.main(["--out", str(out)])
    assert rc == 0
    man2 = _manifest(out)
    assert man2["build"]["n_sources_backfilled_this_run"] == 0
    for f, blob in before.items():
        assert (out / f).read_bytes() == blob


def test_holdout_world_refused_by_build_source(synth_world, tmp_path):
    """In-code guard: build_source must refuse a HOLDOUT world even when
    called directly (main() skips HOLDOUT before any run file is opened;
    this is the second line of defence and never reads run files)."""
    world = dict(world="wsynholdp02t240", split="HOLDOUT", t0_s=240.0,
                 loading_veh_h=2700, penetration=0.02,
                 base_arm=dict(arm="base", run="wsynholdp02t240__base"))
    arm = synth_world["runs"] / "wsynholdp02t240__base"
    with pytest.raises(AssertionError, match="HOLDOUT"):
        B.build_source(world, "base", arm, synth_world["runs"],
                       tmp_path, None, {}, {})
    assert not any(tmp_path.iterdir())      # nothing written
