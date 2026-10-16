"""WALLREGEN full-history variants (h60full / reqfull): base+branch 60 s driver history.

The h60 states carry only the 5 s of branch tape that exists before their decision time;
``h60full`` / ``reqfull`` keep the h60 / req decision times but prepend the pre-branch
base-run recording, remapped by vehicle id into the t0 slot layout (SUMO saveState keeps
registry ids across the restore, but slot columns may differ between recordings). The
existing h60 / real / req variants must stay byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from traffic_shepherd.c4 import wallregen_states as WS

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "_ref" / "sim"
SAMPLE_WORLDS = ("w962000m1900p10t300", "w962006m2400p10t240", "w962009m2700p10t480")
DT = 0.5
# The s-coordinate re-zeros across the junction (lane -1): same-vehicle transitions there
# move up to ~10.4 m beyond v*dt *inside a single tape* (measured on the sample base runs),
# while ordinary transitions stay within v*dt + 1.5 m.
JUNCTION_SLACK_M = 12.0
PLAIN_SLACK_M = 2.0


# ------------------------------------------------------------------ variant table
def test_variant_table_adds_full_history_without_touching_the_old_rules():
    assert WS.VARIANTS["h60"] == ("tw+4.5", 11, 60.0, True)
    assert WS.VARIANTS["real"] == ("tw+60.5", 121, 60.0, False)
    assert WS.VARIANTS["req"] == ("request", 121, 60.0, False)
    assert WS.VARIANTS["h60full"] == ("tw+4.5", 121, 60.0, False)
    assert WS.VARIANTS["reqfull"] == ("request", 121, 60.0, False)
    assert WS.FULL_HISTORY_VARIANTS == frozenset(("h60full", "reqfull"))


# ------------------------------------------------------------------ synthetic tapes
def _tape(t_start: float, rows: int, k: int, layout: dict, speed: float = 10.0,
          s0: float = 100.0) -> dict:
    """One recording with a constant {slot: vid} layout; slot s-position = s0 + 10*slot."""
    time_s = t_start + DT * np.arange(rows)
    s = np.zeros((rows, k), np.float32)
    lane = np.full((rows, k), -1, np.int8)
    v = np.zeros((rows, k), np.float32)
    a = np.zeros((rows, k), np.float32)
    mask = np.zeros((rows, k), bool)
    vid = np.full((rows, k), -1, np.int32)
    for slot, x in layout.items():
        s[:, slot] = s0 + 10.0 * slot + speed * DT * np.arange(rows)
        lane[:, slot], v[:, slot], mask[:, slot], vid[:, slot] = slot % 3, speed, True, x
    return {"time_s": time_s, "s": s, "lane": lane, "v": v, "a": a, "mask": mask, "vid": vid,
            "first_seen_s": np.array([t_start] * k, np.float64),
            "first_entrance": np.zeros(k, np.int8), "is_cav": np.zeros(k, bool)}


def _full_timeline() -> WS.Timeline:
    """Base (t_w-65 .. t_w) + branch (t_w .. t_w+19.5) with PERMUTED slot columns.

    vids: 0='a', 1='b', 2='c', 3='d', 4='e'; K=4 slots.  Base layout puts a/b in slots
    0/1; the base-only context vehicle d occupies slot 3 until base row 60, then e takes
    slot 3 from row 80 (an identity break in the context column).  The branch restores at
    t_w with a/b moved to slots 2/0 and the new vehicle c in slot 1 - the saveState seam
    with a different slot assignment, exactly what the id remap must undo.
    """
    registry = ["a", "b", "c", "d", "e"]
    base = _tape(235.0, 131, 4, {0: 0, 1: 1, 3: 3})          # t: 235.0 .. 300.0
    base["mask"][60:, 3] = False                              # d leaves the base tape
    base["mask"][80:, 3] = True                               # ... slot reused by e
    base["vid"][80:, 3] = 4
    base["s"][80:, 3] = 500.0 + 10.0 * DT * np.arange(131 - 80)
    arm = _tape(300.0, 40, 4, {2: 0, 0: 1, 1: 2})            # t: 300.0 .. 319.5
    # saveState seam: the restored t_w state equals the base t_w row, slots permuted, and
    # the branch then advances at the same speed -> positions are continuous across the seam
    for bslot, aslot in ((0, 2), (1, 0)):
        arm["s"][:, aslot] = base["s"][-1, bslot] + 10.0 * DT * np.arange(40)
    return WS.Timeline(base, arm, registry, arm["is_cav"])


def _grown_timeline() -> WS.Timeline:
    """K=3 tape whose t0 roster fills every column while 2 base-only vehicles overlap."""
    registry = ["a", "b", "c", "d", "e"]
    base = _tape(235.0, 131, 3, {0: 3, 2: 4})                # d + e overlap all rows
    arm = _tape(300.0, 20, 3, {0: 0, 1: 1, 2: 2})            # full roster at t0
    return WS.Timeline(base, arm, registry, arm["is_cav"])


def test_aligned_history_pins_roster_to_t0_slots_and_keeps_base_context():
    tl = _full_timeline()
    j0 = tl.index_of(304.5)                                   # t_w + 4.5
    assert j0 == tl.n_base + 9
    h = tl.history(j0, 121, 60.0, False, align_ids=True)
    kin, mask, tgt = np.array(h["kin"]), np.array(h["mask"]), np.array(h["target_mask"])
    # 121 native rows spanning exactly 60 s, not run-start limited
    assert h["history_steps"] == 121 and kin.shape == (121, 4, 7)
    assert h["available_span_s"] == 60.0 and h["run_start_limited"] is False
    assert h["history_s"] == 60.0 and h["history_rows_requested"] == 121
    # roster vehicles keep their t0 slots (a->2, b->0, c->1); base-only d is context slot 3
    assert h["ids"] == ["b", "c", "a", "__history_context_slot3"]
    assert len(set(h["ids"])) == len(h["ids"]) == kin.shape[1]
    # id-keyed remap: column of 'a' carries a's BASE rows too (base slot 0 != t0 slot 2)
    col_a = h["ids"].index("a")
    lo = j0 - 120
    for j in (0, 55, tl.n_base - 1 - lo, tl.n_base - lo, 120):   # ... t_w-0.5, t_w, t0
        f = lo + j
        assert kin[j, col_a, 0] == pytest.approx(100.0 + 10.0 * f * DT, abs=1e-4)
        assert mask[j, col_a] and kin[j, col_a, 4] == pytest.approx(10.0, abs=1e-4)
        src = 0 if f < tl.n_base else 2                       # 'a': base slot vs arm slot
        assert int(np.argmax(kin[j, col_a, 1:4])) == src % 3  # row keeps its tape lane
    # the seam transition t_w-0.5 -> t_w is continuous for every identity-continuous column
    i_w = tl.n_base - lo
    same = mask[i_w - 1] & mask[i_w] & tgt[i_w - 1]
    assert same.any()
    ds = np.abs(kin[i_w, :, 0] - kin[i_w - 1, :, 0])[same]
    assert ds.max() == pytest.approx(10.0 * DT, abs=1e-3)
    # the base-only vehicles d then e share the context column with an identity break
    col_d = h["ids"].index("__history_context_slot3")
    assert mask[0, col_d] and mask[40, col_d]                 # d: window rows 0..40
    assert not mask[50, col_d]                                # gap after d leaves
    assert mask[100, col_d]                                   # e: window rows 61..110
    assert kin[100, col_d, 0] == pytest.approx(500.0 + 10.0 * DT * (100 - 61), abs=1e-4)
    assert not tgt[40, col_d] and not tgt[60, col_d]          # d -> gap -> e break target_mask


def test_grown_timeline_appends_anonymous_context_columns():
    tl = _grown_timeline()
    j0 = tl.index_of(304.5)
    h = tl.history(j0, 121, 60.0, False, align_ids=True)
    kin = np.array(h["kin"])
    # roster fills K=3; the overlapping base-only pair forces appended anonymous columns
    assert kin.shape[1] == 5
    assert h["ids"][:3] == ["a", "b", "c"]
    assert h["ids"][3:] == ["__history_context_slot3", "__history_context_slot4"]
    ctx = kin[:, 3:]
    assert ctx[..., 0].any() and bool(np.array(h["mask"])[:, 3:].any())


def test_default_history_path_is_unchanged_by_the_full_variants():
    tl = _full_timeline()
    j0 = tl.index_of(304.5)
    h = tl.history(j0, 11, 60.0, True)
    kin = np.array(h["kin"])
    assert kin.shape == (11, 4, 7) and len(h["ids"]) == 4
    assert h["run_start_limited"] is True and h["available_span_s"] == 5.0
    assert np.array_equal(kin, np.stack([tl.kin(j) for j in range(j0 - 10, j0 + 1)]))


# ------------------------------------------------------------------ sample-world evidence
needs_sim = pytest.mark.skipif(not (SIM / "PLAN.json").is_file(),
                               reason="_ref/sim sample worlds not present")


def _head_module(tmp_path: Path) -> Path | None:
    """The committed (HEAD) converter, extracted read-only for byte-identity comparison."""
    try:
        out = subprocess.run(
            ["git", "show", "HEAD:traffic_shepherd/c4/wallregen_states.py"],
            cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    path = tmp_path / "head_wallregen_states.py"
    path.write_bytes(out.stdout)
    return path


def _run_builder(script: Path, variant: str, out: Path, worlds) -> None:
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(ROOT))
    subprocess.run([sys.executable, str(script),
                    "--run", str(SIM), "--variant", variant, "--out", str(out),
                    "--worlds", ",".join(worlds)],
                   cwd=ROOT, env=env, check=True, capture_output=True)


@needs_sim
def test_h60_byte_identical_to_the_committed_converter_on_a_sample_world(tmp_path):
    head = _head_module(tmp_path)
    if head is None:
        pytest.skip("git show HEAD unavailable (detached source tree)")
    old, new = tmp_path / "old", tmp_path / "new"
    _run_builder(head, "h60", old, SAMPLE_WORLDS[:1])
    _run_builder(ROOT / "traffic_shepherd" / "c4" / "wallregen_states.py", "h60", new,
                 SAMPLE_WORLDS[:1])
    for name in ("wall_same.json", "wall_pilot.json", "wall_native.json", "REJECTIONS.jsonl"):
        a = hashlib.sha256((old / name).read_bytes()).hexdigest()
        b = hashlib.sha256((new / name).read_bytes()).hexdigest()
        assert a == b, f"{name} diverged from the committed converter"


def _world_t_w() -> dict:
    out = {}
    for w in SAMPLE_WORLDS:
        rec = json.loads((SIM / "worlds" / w / "world.json").read_text(encoding="utf-8"))
        out[w] = float(rec["plan_world"]["t_w_s"])
    return out


def _normalize():
    """``(iter_windows, initial_frame, recorded_for)`` from the scorer's loader module.

    ``wms.eval.__init__`` pulls in torch (predictor); when torch is absent the torch-free
    ``normalize.py`` is loaded directly over stub namespace packages (nothing else in a
    torch-less process can import ``wms`` anyway).
    """
    try:
        from wms.eval.normalize import initial_frame, iter_windows, recorded_for
        return iter_windows, initial_frame, recorded_for
    except ModuleNotFoundError:
        pass
    import importlib.util
    import types
    if "wms" in sys.modules:                       # torch-free package init survived
        wms = sys.modules["wms"]
    else:
        wms = types.ModuleType("wms")
        wms.__path__ = [str(ROOT / "src" / "wms")]
        sys.modules["wms"] = wms
    ev = types.ModuleType("wms.eval")
    ev.__path__ = [str(ROOT / "src" / "wms" / "eval")]
    sys.modules["wms.eval"] = ev
    spec = importlib.util.spec_from_file_location("wms.eval.normalize",
                                                  ROOT / "src" / "wms" / "eval" / "normalize.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wms.eval.normalize"] = mod
    spec.loader.exec_module(mod)
    return mod.iter_windows, mod.initial_frame, mod.recorded_for


def _load(path: Path) -> list[dict]:
    iter_windows, _, _ = _normalize()
    return list(iter_windows(path))


def _check_full_record(r: dict, t_w: float) -> None:
    h = r["history"]
    kin, mask, tgt = (np.array(h[k]) for k in ("kin", "mask", "target_mask"))
    ids = h["ids"]
    t = kin.shape[0]
    assert t == 121 and h["run_start_limited"] is False, r["key"]
    assert h["history_s"] == 60.0 and h["history_stride_s"] == 1.0 and h["dt_s"] == DT
    assert abs(h["available_span_s"] - 60.0) < 1e-9 and h["history_rows_requested"] == 121
    assert kin.shape[2] == 7 and mask.shape == kin.shape[:2]
    assert tgt.shape == (t - 1, kin.shape[1]) and len(ids) == kin.shape[1]
    assert len(set(ids)) == len(ids)                           # scorer _by_id requirement
    # history row at t0 == the t0 snapshot, keyed by id (junction rows: lane one-hot empty)
    snap = {x["id"]: x for x in r["initial_state"]["vehicles"]}
    for vid, sv in snap.items():
        col = ids.index(vid)
        row = kin[-1, col]
        lane = int(np.argmax(row[1:4])) if row[1:4].any() else -1
        assert mask[-1, col], (r["key"], vid)
        assert abs(row[0] - sv["s"]) <= 1e-3 and abs(row[4] - sv["v"]) <= 1e-3, (r["key"], vid)
        assert lane == sv["lane"], (r["key"], vid)
    # positions continuous across the t_w seam for identity-continuous columns
    t0 = float(r["t0_s"])
    i_w = t - 1 - int(round((t0 - t_w) / DT))
    if 0 < i_w < t:
        both = mask[i_w - 1] & mask[i_w]
        same = both & tgt[i_w - 1]
        assert same.any(), r["key"]
        ds = np.abs(kin[i_w, :, 0] - kin[i_w - 1, :, 0])[same]
        vmax = np.maximum(kin[i_w, :, 4], kin[i_w - 1, :, 4])[same]
        junc = (kin[i_w - 1, :, 1:4].sum(axis=1) == 0) | (kin[i_w, :, 1:4].sum(axis=1) == 0)
        junc = junc[same]
        assert (ds <= vmax * DT + np.where(junc, JUNCTION_SLACK_M, PLAIN_SLACK_M)).all(), r["key"]


def _check_remap_against_base_tape(r: dict) -> None:
    wdir = SIM / "worlds" / r["world"]
    with np.load(wdir / "base.npz", allow_pickle=False) as z:
        base = {k: z[k] for k in ("time_s", "s", "mask", "vid")}
    wj = json.loads((wdir / "world.json").read_text(encoding="utf-8"))
    registry = wj["arms"][r["source_arm"]]["registry"]
    n_base = len(base["time_s"]) - 1
    h = r["history"]
    kin, ids = np.array(h["kin"]), h["ids"]
    lo = int(round((float(r["t0_s"]) - 60.0 - float(base["time_s"][0])) / DT))
    snap = {x["id"]: x for x in r["initial_state"]["vehicles"]}
    for jw in (0, kin.shape[0] // 2, kin.shape[0] - 2):
        bj = lo + jw
        if not 0 <= bj < n_base:
            continue
        bslot = {int(base["vid"][bj][s]): s
                 for s in np.flatnonzero(base["mask"][bj] & (base["vid"][bj] >= 0))}
        for vid in snap:
            vi = registry.index(vid)
            if vi in bslot:
                assert abs(kin[jw, ids.index(vid), 0]
                           - float(base["s"][bj][bslot[vi]])) <= 1e-3, (r["key"], vid, jw)


@needs_sim
def test_full_variants_on_sample_worlds_121_steps_and_remapped_history(tmp_path):
    t_w = _world_t_w()
    builds = {}
    for variant in ("h60", "req", "h60full", "reqfull"):
        out = tmp_path / variant
        WS.convert(SIM, out, variant, worlds=set(SAMPLE_WORLDS))
        builds[variant] = out
    iter_windows, initial_frame, recorded_for = _normalize()          # scorer loader
    for variant in ("h60full", "reqfull"):
        receipt = json.loads((builds[variant] / "BUILD_RECEIPT.json").read_text("utf-8"))
        n_total = 0
        for name in ("wall_same", "wall_pilot", "wall_native"):
            path = builds[variant] / f"{name}.json"
            if not path.is_file():
                continue
            recs = _load(path)
            assert len(recs) == receipt["n_states"].get(name, len(recs))
            for r in recs:
                _check_full_record(r, t_w[r["world"]])
                _check_remap_against_base_tape(r)
                assert initial_frame(r)["vehicles"]                 # loader round-trip
                assert recorded_for(r, "wall" if r["arm"] == "wall" else "native")
            n_total += len(recs)
        assert n_total >= 20                                        # 3 worlds x 4 wall arms
    # decision times and RSU counters are the corresponding variants' exactly
    for a, b in (("h60full", "h60"), ("reqfull", "req")):
        old = {r["key"]: r for r in _load(builds[b] / "wall_same.json")}
        for r in _load(builds[a] / "wall_same.json"):
            ref = old[r["key"]]
            assert r["t0_s"] == ref["t0_s"] and r["t0_rule"] == ref["t0_rule"]
            assert r["counter"] == ref["counter"]
            assert ref["history"]["history_steps"] in (10, 11, 121)
    full = {r["key"]: r for r in _load(builds["h60full"] / "wall_same.json")}
    for r in _load(builds["h60"] / "wall_same.json"):
        assert r["history"]["history_steps"] == 11                  # h60 unchanged: 5 s tape
        assert full[r["key"]]["history"]["history_steps"] == 121
