"""WD-01-PB-V04A builder / sources / verification tests (synthetic world,
plus read-only checks of the real plans; ZERO SUMO)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import T0
from v04a_fixtures import WORLD, make_world

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_v04a as B4  # noqa: E402
import finalize_backup_v04a as FB  # noqa: E402
import v04a_sources as S  # noqa: E402
import v0lib as V  # noqa: E402


def _spec(arms, src, *, typ, policy, kind, role="primary", chunk="chunk4"):
    return S.SrcSpec(
        chunk=chunk, world=WORLD, source=src, split="TRAIN", role=role,
        policy=policy, verify_kind=kind, type=typ, arm_dir=str(arms[src]),
        base_dir=str(arms["base"]), t0_s=T0, demand_veh_h=2700,
        penetration=0.02, runs_root=str(arms["base"].parent),
        default_speed_mode=27)


def _specs(arms, chunk="chunk4"):
    return [
        _spec(arms, "base", typ="base_like", policy="damping_fleet",
              kind="damping_fleet", chunk=chunk),
        _spec(arms, "native", typ="branch", policy="native_fleet",
              kind="native_fleet", chunk=chunk),
        _spec(arms, "pi", typ="branch", policy="pi_fleet",
              kind="pi_fleet", chunk=chunk),
        _spec(arms, "switching1", typ="branch", policy="switching_mix",
              kind="switching", role="aux", chunk=chunk)]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v04a_world")
    arms = make_world(tmp)
    root = tmp / "root"
    res = B4.build_world((str(root), [s.__dict__ for s in _specs(arms)],
                          True))
    return dict(tmp=tmp, arms=arms, root=root, res=res)


def _entries(built):
    return {e["source"]: e for e in built["res"]
            if "_opened" not in e and not e.get("refused")}


# ---------------------------------------------------------------- labels --

def test_skip_first_tick_puts_base_t0_on_the_branch_convention(built):
    arms = built["arms"]
    rt_b = V.build_run_trace(WORLD, "base", arms["base"], T0, t_ins_pre={})
    pre = V.build_t_ins_pre(rt_b, T0)
    rt_n = V.build_run_trace(WORLD, "native", arms["native"], T0,
                             t_ins_pre=pre, base_arm_dir=arms["base"])
    for tau in (30.0, 120.0):
        plain = V.window_label(rt_b, T0, tau)
        skip = V.window_label(rt_b, T0, tau, skip_first_tick=True)
        branch = V.window_label(rt_n, T0, tau)
        # default is byte-identical to the v0.3.2 label; the flag drops
        # exactly the first tick's interval
        assert skip["time_on_section_s"] < plain["time_on_section_s"]
        # synthetic branch physics == base physics after t0, so the
        # skipped base label must equal the branch label exactly
        for k in ("time_on_section_s", "free_flow_ref_s", "traversed_m",
                  "total_delay_ref_s", "wait_mainline_s", "wait_ramp_s"):
            assert skip[k] == pytest.approx(branch[k], abs=1e-9), k


def test_default_window_label_unchanged(built):
    arms = built["arms"]
    rt_b = V.build_run_trace(WORLD, "base", arms["base"], T0, t_ins_pre={})
    a = V.window_label(rt_b, 120.0, 60.0)
    b = V.window_label(rt_b, 120.0, 60.0, skip_first_tick=False)
    assert a == b


# ----------------------------------------------------------------- build --

def test_all_sources_built_and_labelled(built):
    ents = _entries(built)
    assert set(ents) == {"base", "native", "pi", "switching1"}
    assert ents["switching1"]["aux"] is True
    assert ents["base"]["aux"] is False
    for s, e in ents.items():
        assert e["policy"] == e["policy_v04a"]
        for rel, sha in e["files"].items():
            p = built["root"] / rel
            assert p.is_file() and B4.sha256_file(p) == sha
    lab = {s: pd.read_parquet(built["root"] / "worlds" / "chunk4"
                              / f"{WORLD}__{s}_labels.parquet")
           for s in ents}
    assert set(lab["native"]["policy"]) == {"native_fleet"}
    assert set(lab["pi"]["policy"]) == {"pi_fleet"}
    assert set(lab["switching1"]["policy"]) == {"switching_mix"}
    assert lab["switching1"]["aux"].all() and not lab["base"]["aux"].any()
    # branch arms start with the shared t0 row; the base grid is the
    # chunk-4 "after t0" grid
    assert lab["native"]["t"].min() == T0 and lab["base"]["t"].min() == T0
    # (TD tuples are never emitted at the shared t0 row)
    td = pd.read_parquet(built["root"] / "worlds" / "chunk4"
                         / f"{WORLD}__native_td.parquet")
    assert (td["t"] > T0).all()


def test_t0_rows_agree_across_arms_and_pi_columns(built):
    lab = {s: pd.read_parquet(built["root"] / "worlds" / "chunk4"
                              / f"{WORLD}__{s}_labels.parquet")
           for s in ("base", "native", "pi", "switching1")}
    t0 = {s: d[np.isclose(d["t"], T0)].set_index("tau") for s, d in
          lab.items()}
    # identical synthetic dynamics -> identical t0 labels (skip convention)
    for s in ("native", "pi", "switching1"):
        pd.testing.assert_series_equal(
            t0[s]["total_delay_ref_s"], t0["base"]["total_delay_ref_s"],
            check_names=False, atol=1e-9)
    # pi conditioning columns
    r = t0["base"].iloc[0]
    assert (r.pi_native, r.pi_damping, r.pi_pi) == (0.0, 1.0, 0.0)
    r = t0["native"].iloc[0]
    assert (r.pi_native, r.pi_damping, r.pi_pi) == (1.0, 0.0, 0.0)
    r = t0["pi"].iloc[0]
    assert (r.pi_native, r.pi_damping, r.pi_pi) == (0.0, 0.0, 1.0)
    sw = t0["switching1"]
    for tau, rr in sw.iterrows():                 # switching: a mix
        assert rr.pi_native + rr.pi_damping + rr.pi_pi == pytest.approx(1)
    # CAV a at t0: native 5.3 s, damping 3.6 s, pi thereafter (tau=30)
    rr = sw.loc[30.0]
    assert rr.pi_native == pytest.approx(5.3 / 30)
    assert rr.pi_damping == pytest.approx(3.6 / 30)
    assert rr.pi_pi == pytest.approx(21.1 / 30)


def test_tokens_carry_the_planned_action_pi_is_not_damping(built):
    z = {s: np.load(built["root"] / "worlds" / "chunk4"
                    / f"{WORLD}__{s}_feat_dep_strict.npz")
         for s in ("base", "native", "pi", "switching1")}
    slot = {"native": 0, "damping": 1, "pi": 2, "wall": 3, "unknown": 4}
    for s, want in (("base", "damping"), ("native", "native"),
                    ("pi", "pi")):
        tok, msk = z[s]["cav_tokens"], z[s]["cav_mask"] > 0
        act = tok[..., 5:10].argmax(-1)[msk]
        assert msk.any() and (act == slot[want]).all(), s
    # CAV_RAW 20 and empty wall tokens
    assert z["pi"]["cav_tokens"].shape[-1] == 20
    assert not z["pi"]["wall_mask"].any()
    # the switching arm's CAV a changes action across decision times
    tok, msk = z["switching1"]["cav_tokens"], z["switching1"]["cav_mask"] > 0
    ts = z["switching1"]["t"]
    a0 = tok[0, 0, 5:10].argmax()
    assert a0 == slot["native"]                       # t0 = 240
    i = int(np.flatnonzero(np.isclose(ts, 250.0))[0])
    if msk[i].any():
        assert tok[i, 0, 5:10].argmax() == slot["pi"]


def test_strict_variant_has_no_all_vehicle_counts_and_t0_unrecoverable(built):
    zs = np.load(built["root"] / "worlds" / "chunk4"
                 / f"{WORLD}__native_feat_dep_strict.npz")
    zd = np.load(built["root"] / "worlds" / "chunk4"
                 / f"{WORLD}__native_feat_dep_det.npz")
    assert zs["demand"].shape[1] == 6 and zd["demand"].shape[1] == 12
    assert (zs["context"][:, 1] == 0).all()
    assert (zs["context"][:, 6:8] == 0).all()
    assert str(zs["feature_set"]) == "deployable_strict"


# --------------------------------------------------------------- refusal --

def test_mismatching_arm_is_refused_and_not_built(tmp_path):
    arms = make_world(tmp_path, native_owner="damping")
    spec = _spec(arms, "native", typ="branch", policy="native_fleet",
                 kind="native_fleet")
    res = B4.build_world((str(tmp_path / "root"), [_specs(arms)[0].__dict__,
                                                   spec.__dict__], True))
    refused = [e for e in res if e.get("refused")]
    assert len(refused) == 1
    assert "ArmPolicyMismatch" in refused[0]["reason"]
    assert not list((tmp_path / "root" / "worlds" / "chunk4")
                    .glob("*__native_*"))


# ------------------------------------------------ forbidden log reads ----

def test_builder_never_opens_dispatcher_or_insertions(tmp_path):
    arms = make_world(tmp_path, dispatcher=True)
    opened = []

    def hook(event, args):
        if event == "open":
            opened.append(str(args[0]))
    sys.addaudithook(hook)
    B4.build_world((str(tmp_path / "root"),
                    [s.__dict__ for s in _specs(arms)], True))
    low = [p.replace("\\", "/") for p in opened]
    assert any("physics_steps.jsonl" in p for p in low)
    assert not [p for p in low if "dispatcher_rows" in p], "dispatcher read"
    assert not [p for p in low if p.endswith("insertions.jsonl")], \
        "insertions.jsonl read"


def test_static_no_forbidden_reads_or_hardlinks():
    for name in ("build_v04a.py", "v04a_sources.py", "arm_verify_v04a.py",
                 "finalize_backup_v04a.py", "actions_v04.py"):
        src = (HERE / name).read_text(encoding="utf-8")
        code = "\n".join(line.split("#")[0] for line in src.splitlines())
        assert "insertion_crosscheck" not in code, name
        assert "os.link(" not in code and "os.symlink(" not in code, name
        for banned in ("open(", ".read_text(", "json.load("):
            for line in code.splitlines():
                if banned in line and ("dispatcher_rows" in line
                                       or "insertions.jsonl" in line):
                    raise AssertionError(f"{name}: {line.strip()}")


# ------------------------------------------------------ sources / split --

def test_enumerate_specs_never_holdout_and_counts():
    plan = S.PLAN["chunk4"]
    if not Path(plan).is_file() or not S.V032_MANIFEST.is_file():
        pytest.skip("real plans not present")
    hold = {c: set(S.chunk_worlds(c)[1]) for c in S.ALL_CHUNKS}
    specs = S.enumerate_specs()
    assert specs
    assert all(s.split in ("TRAIN", "CALIBRATION") for s in specs)
    for s in specs:
        assert s.world not in hold[s.chunk], s.key()
    n4 = [s for s in specs if s.chunk == "chunk4"]
    # WM-N1 label coverage (2026-10-04): chunk4 also enumerates the four
    # walls-pilot arms for every TRAIN/CAL world of the walls plan.
    n_wall_worlds = len(S.walls_plan_worlds()[0])
    assert len(n4) == 72 * 9 + n_wall_worlds * 4
    assert {s.source for s in n4} == {
        "base", "native_base", "native", "pi", "fixed1", "fixed2",
        "switching1", "switching2", "switching3",
        "w-lo-early", "w-lo-late", "w-mid-early", "w-mid-late"}
    assert all(s.role == "primary" for s in n4)
    aux = [s for s in specs if s.role == "aux"]
    assert {s.source for s in aux} <= {"base", "swnative", "swpi"}
    assert all(s.chunk in ("chunk1", "chunk1_fix27") or s.source in
               ("swnative", "swpi") for s in aux)
    # chunk-2/3: bases + J3 true arms + the campaign fixed/switching branch
    # arms (WM-N1 label coverage 2026-10-04) are primary
    prim23 = {(s.chunk, s.source) for s in specs
              if s.chunk in ("chunk2", "chunk3") and s.role == "primary"}
    assert prim23 == {(c, x) for c in ("chunk2", "chunk3")
                      for x in ("base", "native", "pi", "fixed1", "fixed2",
                                "switching1", "switching2", "switching3")}


def test_open_audit_flags_a_holdout_path():
    S.OpenAudit.paths = ["E:/x/runs/w1__base/run/physics_steps.jsonl"]
    assert S.OpenAudit.assert_no_holdout({"w2"}) == 1
    with pytest.raises(AssertionError):
        S.OpenAudit.assert_no_holdout({"w1"})
    S.OpenAudit.paths = []


# ------------------------------------------- finalize + copy-only backup --

def test_finalize_backup_copy_only_and_verify(built, tmp_path):
    root = built["root"]
    ver = tmp_path / "arm_verification.json"
    ver.write_text("{}", encoding="utf-8")
    man = dict(schema="wd01_partB_v0data/1", worlds={"x": 1},
               v04a=dict(n_refused=0))
    (root / "MANIFEST.json").write_text(json.dumps(man), encoding="utf-8")
    ns = type("A", (), dict(root=str(root), arm_verification=str(ver)))
    assert FB.cmd_finalize(ns) == 0
    dst = tmp_path / "bk"
    ns2 = type("A", (), dict(root=str(root), dst=str(dst)))
    assert FB.cmd_backup(ns2) == 0
    for line in (root / "SHA256SUMS").read_text().splitlines()[:50]:
        rel = line.split("  ", 1)[1]
        a, b = os.stat(root / rel), os.stat(dst / rel)
        assert b.st_nlink == 1 and (a.st_ino, a.st_dev) != (b.st_ino,
                                                            b.st_dev)
    # refuses a non-empty target and detects corruption
    assert FB.cmd_backup(ns2) == 1
    victim = next(p for p in (dst / "worlds").rglob("*.parquet"))
    victim.write_bytes(b"corrupt")
    n, bad, miss = FB.verify(dst)
    assert bad == 1
