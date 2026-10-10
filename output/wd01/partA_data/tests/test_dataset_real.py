"""Tests against the built real dataset (skipped if data/ is absent):
MANIFEST sha256 integrity, dedup determinism, s-advance continuity on real
roots incl. junction boundaries, lane-change event consistency, and a
loader smoke call."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import wdpaths  # noqa: E402
import pdl as L  # noqa: E402
import loader as LD  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
MAN = HERE / "MANIFEST.json"

pytestmark = pytest.mark.skipif(
    not MAN.exists(), reason="dataset not built yet (run build.py)")


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MAN.read_text(encoding="utf-8"))


def test_manifest_sha256_and_structure(manifest):
    assert manifest["checkpoint_sha256"] == L.SURV_CKPT_SHA256
    assert len(manifest["roots"]) == 26
    seen = {}
    for name, r in manifest["roots"].items():
        p = Path(wdpaths.remap(r["file"]))
        assert p.exists(), p
        assert L.sha256_file(p) == r["sha256"], name
        assert r["split"] in ("train", "holdout")
        for tag in r["all_tags"]:
            assert tag not in seen, f"tag {tag} in two roots"
            seen[tag] = name
    assert len(seen) == 88
    # dedup table is flat, consistent, and deterministic: re-hash the
    # representative run of two roots and compare
    dt = {d["tag"]: d for d in manifest["dedup_table"]}
    assert set(dt) == set(seen)
    for d in manifest["dedup_table"]:
        assert dt[d["tag"]]["window_sha256"] == d["window_sha256"]
        assert dt[d["tag"]]["root"] == seen[d["tag"]]
        assert dt[d["tag"]]["representative"] == (
            d["tag"] == manifest["roots"][d["root"]]["representative_tag"])
    for name in ("cv17", "cv20_7005m2000"):
        r = manifest["roots"][name]
        run_dir = None
        row = next(d for d in manifest["dedup_table"]
                   if d["tag"] == r["representative_tag"])
        m = L.load_json(L.PART1 / "results" / "tag_run_map.json")
        run_dir = wdpaths.remap(next(x["run_dir"] for x in m["rows"]
                                     if x["tag"] == r["representative_tag"]))
        assert L.window_hash(run_dir) == row["window_sha256"]


def test_root_split_by_root_and_v3_consistency(manifest):
    v3 = L.load_v3_splits()
    dt = manifest["dedup_table"]
    for d in dt:
        assert d["split"] == v3[d["tag"]], d
    for name, r in manifest["roots"].items():
        splits = {v3[t] for t in r["all_tags"]}
        assert len(splits) == 1, name      # no mixed-split root
        assert splits == {r["split"]}


def _walk_pairs(z):
    from collections import defaultdict
    rows = defaultdict(list)
    for i in range(len(z["vid"])):
        rows[int(z["vid"][i])].append(i)
    changes_at = defaultdict(set)
    for vi, tk in zip(z["lc_vid"], z["lc_tick"]):
        changes_at[int(vi)].add(int(tk))
    for vi, idxs in rows.items():
        idxs.sort(key=lambda i: int(z["tick"][i]))
        ch = changes_at.get(vi, ())
        for a, b in zip(idxs, idxs[1:]):
            ta, tb = int(z["tick"][a]), int(z["tick"][b])
            yield vi, a, b, any(ta < t <= tb for t in ch)


@pytest.mark.parametrize("root", ["cv17", "cv20_7005m2000"])
def test_s_advance_continuity_on_real_root(manifest, root):
    z = LD.load_root(root)
    phys, bridge = [], []
    for vi, a, b, is_change in _walk_pairs(z):
        if is_change:
            continue
        r = (float(z["s"][b]) - float(z["s"][a])) - float(z["v"][a]) * 0.5
        (bridge if (z["in_junction"][a] or z["in_junction"][b])
         else phys).append(abs(r))
    rp = np.asarray(phys)
    rb = np.asarray(bridge)
    assert len(rp) > 10000 and len(rb) > 100
    # physical-step continuity: median ~cm; the >1.5 m outlier rate is
    # below the 0.5% sanity bound (tail = SUMO integrator form on braking
    # steps; see REPORT_PART2C.md section Sanity)
    assert float(np.median(rp)) < 0.05
    assert float((rp > 1.5).mean()) < 0.005
    # junction bridges carry the documented boundary-s approximation: the
    # residual is bounded by the junction internal-lane length (~8 m here)
    assert float(rb.max()) < 12.0


def test_junction_rows_sit_on_segment_boundaries(manifest):
    topo, _, internal = L.topology()
    boundaries = set()
    for name in L.LANE_IDS:
        for seg in topo.segments_of(name)[1:]:
            boundaries.add(round(seg.s0, 6))
    for root in ("cv17", "cv20_7005m2000"):
        z = LD.load_root(root)
        inj = z["in_junction"] == 1
        assert inj.any()
        assert (z["lane"][inj] == -1).all()
        svals = {round(float(s), 6) for s in z["s"][inj]}
        unknown = {s for s in svals
                   if min((s - b) % 1e-6 for b in boundaries) != 0
                   and not any(abs(s - b) < 1e-6 for b in boundaries)}
        assert not unknown, f"junction s not on a boundary: {sorted(unknown)[:5]}"
        # physical rows never flagged
        assert (z["in_junction"][z["lane"] >= 0] == 0).all()


def test_lane_change_events_match_lane_rows(manifest):
    z = LD.load_root("cv17")
    # every declared change endpoint has the declared new lane
    for vi, tk, to in zip(z["lc_vid"], z["lc_tick"], z["lc_to"]):
        m = (z["vid"] == vi) & (z["tick"] == tk)
        assert m.any(), (vi, tk)
        assert int(z["lane"][m][0]) == int(to)
    # and every lane flip between consecutive physical rows is declared
    from collections import defaultdict
    rows = defaultdict(list)
    for i in range(len(z["vid"])):
        rows[int(z["vid"][i])].append(i)
    declared = {(int(vi), int(tk)) for vi, tk in zip(z["lc_vid"], z["lc_tick"])}
    n_flips = 0
    for vi, idxs in rows.items():
        idxs.sort(key=lambda i: int(z["tick"][i]))
        prev = None
        for i in idxs:
            li = int(z["lane"][i])
            if li >= 0:
                if prev is not None and prev[1] != li:
                    assert (vi, int(z["tick"][i])) in declared
                    n_flips += 1
                prev = (i, li)
    assert n_flips == len(z["lc_vid"])


def test_loader_windows_on_real_root(manifest):
    w = LD.windows("cv17", 300.0, 30.0)
    assert len(w["state"]["vid"]) > 20
    assert len(w["truth"]) == 60
    assert w["arrivals"]["t_s"].size >= 0
    for r in w["truth"]:
        assert np.isfinite(r["s"]).all()
        assert ((r["lane"] >= -1) & (r["lane"] <= 2)).all()
    # no vType in any input-facing output
    for k in list(w["state"]) + [k for r in w["truth"] for k in r]:
        assert "type" not in k.lower() and "class" not in k.lower()
