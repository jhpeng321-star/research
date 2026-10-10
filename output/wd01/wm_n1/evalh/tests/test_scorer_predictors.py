"""E7 tests: the real per-item predictors of the CAL-tune scorer.

Everything here is GENERATED under ``tmp_path``: tiny random-init bundles
saved through the checkpoint builder, synthetic shards/release specs and a
synthetic Part A v2g ``params_nominal.json``/``engine.py``.  No registered CAL
split, CAL shard, HOLDOUT/chunk-5/J4B/INC1 data, real checkpoint or real
partA_v2g artifact is ever opened.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from evalh import real_source as RS
from evalh.protocol import N_QUANTILES, OutcomeRow
from evalh.real_source import ReleasedCalSource
from evalh.scorer.actions import MissingActionSchedule
from evalh.scorer.checkpoint import load_scoring_checkpoint
from evalh.scorer.cli import ScorerCliRefusal, build_parser, run_score
from evalh.scorer.predictors import (CAV_MOTION_CONTROLLER, CAV_MOTION_TEACHER,
                                     ModelAPredictor, PredictorRefusal,
                                     build_item_world, build_schedule,
                                     build_teacher, build_v2g_predictor,
                                     expand_plan_to_ticks, rollout_teacher)
from evalh.tests.test_real_source import _file_sha, _write_json
from evalh.tests.test_scorer_checkpoint import _payload, _write as _ckpt_write
from features.constants import ACTIONS

_N = 300
_K = 3
_TAU = (60.0, 120.0)
_TUNE = ("w1",)
_CAL = ("w2",)

_ENGINE_FIXTURE = '''\
"""Generated fixture engine (E7 tests only)."""
import numpy as np


def segment_limits(lane, s, params):
    lane = np.asarray(lane)
    s = np.asarray(s, dtype=float)
    starts = params["geometry"]["lane_segment_starts_m"]
    lims = params["geometry"]["lane_segment_limits_mps"]
    out = np.full(len(lane), np.nan, dtype=float)
    for li in (0, 1, 2):
        m = lane == li
        if not m.any():
            continue
        idx = np.searchsorted(np.asarray(starts[li]), s[m],
                              side="right") - 1
        idx = np.clip(idx, 0, len(lims[li]) - 1)
        out[m] = np.asarray(lims[li])[idx]
    return out
'''

_PARAMS = {
    "dt_s": 0.5,
    "geometry": {
        "lane_segment_starts_m": [[0.0], [0.0], [0.0]],
        "lane_segment_limits_mps": [[35.0], [35.0], [35.0]],
    },
}


# ---------------------------------------------------------------- fixture ----
def _arrays(arm: str):
    n, k = _N, _K
    kin = np.zeros((n, k, 7), np.float32)
    kin[..., 0] = 100.0 + 20.0 * np.arange(n, dtype=np.float32)[:, None]
    kin[..., 1] = 1.0                     # all in physical lane 0
    kin[..., 4] = 20.0
    kin[..., 6] = 5.0
    mask = np.ones((n, k), np.uint8)
    slot = np.broadcast_to(np.arange(k, dtype=np.int32), (n, k)).copy()
    target = np.ones((n, k), np.uint8)
    counter = np.zeros((n, 2, 30), np.float16)
    step = np.zeros((n, k, 16), np.float32)
    step[..., 0] = kin[..., 0]
    step[..., 1] = 1.0
    step[..., 4] = 20.0
    step[..., 6] = 5.0
    step[..., 7] = (slot == 0)            # slot 0 is the CAV
    step[..., 15] = 1.0
    plan = np.zeros((20, k, 19, 8), np.float16)
    action = ACTIONS.index("damping" if arm == "damping" else "native")
    plan[0, 0, :, action] = 1.0           # CAV slot 0 planned action
    cav_plan = np.zeros((n, k, 8), np.float16)
    return dict(veh_kin=kin, veh_mask=mask, veh_step=step, slot_index=slot,
                target_mask=target, counter=counter, cav_plan_ch=cav_plan,
                plan=plan)


def _item(world, arm, first, n_rows):
    return dict(batch="chunk2", chunk="chunk2", world=world, arm=arm,
                policy=arm, split="CALIBRATION", t0_s=300.0, tick=0,
                plan_row=0, outcome_rows=list(range(first, first + n_rows)))


def _run(world, arm, first, n_rows):
    return dict(key=f"chunk2/{world}__{arm}", batch="chunk2", chunk="chunk2",
                world=world, arm=arm, split="CALIBRATION", t0_s=300.0,
                tick_start=0, tick_end=_N, dec_start=0, dec_end=20,
                n_ticks=_N, outcome_first=first, n_outcome_rows=n_rows)


def _label(world, arm, tau, y):
    return dict(world=world, source=arm, split="CALIBRATION", t0_s=300.0,
                t=300.0, tau=tau, total_delay_ref_s=y, collision_any=0.0,
                ttc_lt3_veh_s=1.0, hard_brake_s=0.1, n_vehicles_present=10.0,
                tau_available=100.0)


def _write_shard(d, world, arm, *, n_rows):
    arrays = _arrays(arm)
    for name in ("veh_kin", "veh_mask", "veh_step", "slot_index",
                 "target_mask", "counter", "cav_plan_ch", "plan"):
        np.save(d / f"{name}.npy", arrays[name])
    files = {f"{n}.npy": _file_sha(d / f"{n}.npy") for n in (
        "veh_kin", "veh_mask", "veh_step", "slot_index", "target_mask",
        "counter", "cav_plan_ch", "plan")}
    outputs = {n: files[f"{n}.npy"] for n in (
        "veh_mask", "veh_step", "slot_index", "target_mask", "counter",
        "plan")}
    if n_rows:
        rows = [_label(world, arm, 60.0, 10.0),
                _label(world, arm, 120.0, 12.0)]
        pd.DataFrame(rows).to_parquet(d / "outcomes.parquet", index=False)
        files["outcomes.parquet"] = _file_sha(d / "outcomes.parquet")
        outputs["outcomes.parquet"] = files["outcomes.parquet"]
    done = dict(schema="wm_n1_features_index/1", batch="chunk2",
                chunk="chunk2", world=world, arm=arm, split="CALIBRATION",
                t0_s=300.0, n_outcome_rows=n_rows, outputs=outputs)
    done_sha = _write_json(d / "DONE.json", done)
    idea = dict(schema="wm_n1_features_idea_a/1", batch="chunk2",
                chunk="chunk2", world=world, arm=arm, split="CALIBRATION",
                t0_s=300.0, arrays={
                    "veh_kin": {"sha256": files["veh_kin.npy"]},
                    "cav_plan_ch": {"sha256": files["cav_plan_ch.npy"]}})
    idea_sha = _write_json(d / "IDEA_A_DONE.json", idea)
    return files, done_sha, idea_sha


def _make_pack(tmp_path):
    split_path = tmp_path / "CAL_SPLIT_A4.json"
    split_sha = _write_json(split_path, {"schema": "wm_n1_cal_split_a4/1",
                                         "cal_tune": list(_TUNE),
                                         "cal_cal": list(_CAL)})
    order = [("w1", "native", 0, 2), ("w1", "damping", 2, 2),
             ("w2", "native", 4, 0)]
    manifest = {"schema": "wm_n1_pack/1",
                "items": [_item(w, a, f, n) for w, a, f, n in order]}
    index = {"schema": "wm_n1_corpus_pack/2",
             "runs": [_run(w, a, f, n) for w, a, f, n in order]}
    pack = tmp_path / "pack"
    manifest_sha = _write_json(pack / "manifest.json", manifest)
    index_sha = _write_json(pack / "INDEX.json", index)
    (pack / "outcomes.parquet").write_bytes(b"forbidden mixed table")
    shards = tmp_path / "shards"
    shas = {}
    for world, arm, _f, n_rows in order:
        if world == "w2":
            continue
        d = shards / "chunk2" / world / arm
        d.mkdir(parents=True)
        file_shas, done_sha, idea_sha = _write_shard(d, world, arm,
                                                     n_rows=n_rows)
        shas[f"chunk2/{world}__{arm}"] = (file_shas, done_sha, idea_sha)
    return dict(split_path=split_path, split_sha=split_sha, pack=pack,
                manifest_sha=manifest_sha, index_sha=index_sha, shards=shards,
                shas=shas)


def _release(tmp_path, ctx, *, half, worlds, member_sha):
    selected = {}
    for world, arm in worlds:
        key = f"chunk2/{world}__{arm}"
        if key in ctx["shas"]:
            files, done_sha, idea_sha = ctx["shas"][key]
            selected[key] = dict(done_sha256=done_sha,
                                 idea_a_done_sha256=idea_sha, files=files)
        else:
            selected[key] = dict(done_sha256="0" * 64,
                                 idea_a_done_sha256="0" * 64,
                                 files={"veh_mask.npy": "a" * 64})
    spec = dict(schema="wm_n1_released_cal_source/1",
                allocation_id="generated-allocation",
                data_release_id="generated-release", source_head="synthetic",
                checkpoint_member_sha256=member_sha, half=half,
                cal_split_path=str(ctx["split_path"]),
                pack_dir=str(ctx["pack"]),
                manifest_sha256=ctx["manifest_sha"],
                index_sha256=ctx["index_sha"], shards_root=str(ctx["shards"]),
                selected_shards=selected)
    path = tmp_path / f"RELEASE_{half}.json"
    return path, _write_json(path, spec)


def _checkpoint(tmp_path):
    path = _ckpt_write(tmp_path, _payload())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, sha


def _load(path, **kw):
    return load_scoring_checkpoint(
        path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _source(tmp_path, ctx, *, half, worlds, member_sha):
    path, sha = _release(tmp_path, ctx, half=half, worlds=worlds,
                         member_sha=member_sha)
    return ReleasedCalSource(path, expected_release_sha256=sha)


def _tune_source(tmp_path, monkeypatch, member_sha):
    ctx = _make_pack(tmp_path)
    monkeypatch.setattr(RS, "LF_SHA_CAL_SPLIT_A4", ctx["split_sha"])
    src = _source(tmp_path, ctx, half="cal_tune",
                  worlds=[("w1", "native"), ("w1", "damping")],
                  member_sha=member_sha)
    return ctx, src


def _rows():
    return [OutcomeRow(world="w1", t0_s=300.0, decision_t_s=300.0,
                       tau_s=tau, policy=pol, y_delay_s=1.0,
                       features={"pack_item_position": pos})
            for pos in (0, 1) for pol in ("native", "damping")
            for tau in _TAU]


# ------------------------------------------------------------------ worlds ---
def test_build_item_world_roundtrip(tmp_path, monkeypatch):
    ctx, source = _tune_source(tmp_path, monkeypatch, "a" * 64)
    _pos, item, run = source.selected[0]
    window = build_item_world(source, item, run)
    assert window.local_t0_tick == 0
    assert window.world.batch == 1 and window.world.slots == _K
    assert window.is_cav_t0.tolist() == [True, False, False]
    s = window.world.s_m[0].numpy()
    v = window.world.v_mps[0].numpy()
    assert s.tolist() == [100.0, 100.0, 100.0]
    assert v.tolist() == [20.0, 20.0, 20.0]
    assert window.world.lane[0].tolist() == [0, 0, 0]
    assert window.world.slot_id[0].tolist() == [0, 1, 2]
    assert window.counter_t0.shape == (2, 30)
    assert np.isfinite(window.counter_t0).all()


def test_world_builder_refuses_missing_counter(tmp_path, monkeypatch):
    ctx, source = _tune_source(tmp_path, monkeypatch, "a" * 64)
    _pos, item, run = source.selected[0]
    source.pins["chunk2/w1__native"]["files"].pop("counter.npy")
    with pytest.raises(PredictorRefusal, match="counter"):
        build_item_world(source, item, run)


def test_expand_plan_segment_timing():
    plan = np.zeros((20, _K, 19, 8), np.float64)
    for seg in range(19):
        plan[0, 0, seg, ACTIONS.index("native")] = 1.0
    plan[0, 0, 1, :] = 0.0
    plan[0, 0, 1, ACTIONS.index("damping")] = 1.0
    expanded = expand_plan_to_ticks(plan[0], n_steps=120)
    assert expanded.shape == (120, _K, 8)
    assert expanded[0, 0, ACTIONS.index("native")] == 1.0
    assert expanded[59, 0, ACTIONS.index("native")] == 1.0
    assert expanded[60, 0, ACTIONS.index("damping")] == 1.0
    assert expanded[119, 0, ACTIONS.index("damping")] == 1.0


def test_controller_mode_refuses_missing_schedule(tmp_path, monkeypatch):
    ctx, source = _tune_source(tmp_path, monkeypatch, "a" * 64)
    _pos, item, run = source.selected[0]
    source.pins["chunk2/w1__native"]["files"].pop("plan.npy")
    with pytest.raises(MissingActionSchedule, match="plan"):
        build_schedule(source, item, run, n_steps=20, is_cav_t0=[True, False,
                                                                 False])


# --------------------------------------------------------------- motions ----
def test_teacher_mode_reproduces_recorded_cav_positions(tmp_path, monkeypatch):
    _ctx, source = _tune_source(tmp_path, monkeypatch, "a" * 64)
    _pos, item, run = source.selected[0]
    window = build_item_world(source, item, run)
    seq, seq_mask, lane = build_teacher(source, item, run, n_steps=20)
    rolls = rollout_teacher(
        _load(_checkpoint(tmp_path)[0]).bundle, window.world, seq=seq,
        seq_mask=seq_mask, lane=lane, n_steps=20, integrator="d1",
        residual_enabled=True, counter=np.asarray(window.counter_t0)[None],
        is_cav_t0=window.is_cav_t0)
    recorded = np.asarray(seq[:, 0, 0], dtype=np.float64)
    got = rolls.positions[0, :, 0, 0].detach().cpu().numpy().astype(np.float64)
    assert np.allclose(got, recorded)


def test_two_cav_motion_modes_run_and_differ(tmp_path, monkeypatch):
    _ctx, source = _tune_source(tmp_path, monkeypatch, "a" * 64)
    bound = _load(_checkpoint(tmp_path)[0])
    rows = _rows()
    ctrl = ModelAPredictor(bound.bundle, source=source,
                           cav_motion=CAV_MOTION_CONTROLLER)
    teach = ModelAPredictor(bound.bundle, source=source,
                            cav_motion=CAV_MOTION_TEACHER)
    q_ctrl = ctrl.predict_quantiles(rows)
    q_teach = teach.predict_quantiles(rows)
    assert q_ctrl.shape == (len(rows), N_QUANTILES)
    assert q_teach.shape == (len(rows), N_QUANTILES)
    assert np.isfinite(q_ctrl).all() and np.isfinite(q_teach).all()
    assert not np.allclose(q_ctrl, q_teach)
    # the underlying per-item CAV trajectories differ (damping vs recorded)
    _pos, item, run = source.selected[0]
    window = ctrl._cache.world(0)
    c = ctrl._rollout(window, n_steps=20)
    t = teach._rollout(window, n_steps=20)
    assert not np.allclose(c.positions[0, :, 0, 0].numpy(),
                           t.positions[0, :, 0, 0].numpy())


def test_dynamics_extrapolation_runs_but_never_feeds_head(tmp_path,
                                                          monkeypatch):
    _ctx, source = _tune_source(tmp_path, monkeypatch, "a" * 64)
    bound = _load(_checkpoint(tmp_path)[0])
    rows = _rows()
    plain = ModelAPredictor(bound.bundle, source=source,
                            cav_motion=CAV_MOTION_CONTROLLER)
    extrap = ModelAPredictor(bound.bundle, source=source,
                             cav_motion=CAV_MOTION_CONTROLLER,
                             dynamics_extrapolation=True)
    assert np.allclose(plain.predict_quantiles(rows),
                       extrap.predict_quantiles(rows))
    assert extrap.extrapolation_report["label"] == "extrapolation"
    assert extrap.extrapolation_report["n_steps"] == 240
    assert plain.extrapolation_report is None


# ---------------------------------------------------------------- cli args ---
def _args(tmp_path, *, member, release_spec, release_sha, out, cav_motion=None,
          v2g_option=None, extrapolation=False):
    return SimpleNamespace(
        command="score", release_spec=str(release_spec),
        release_sha=release_sha, checkpoint=str(member), out=str(out),
        params_sha256=None, cav_motion=cav_motion,
        v2g_delay_option=v2g_option, dynamics_extrapolation=extrapolation,
        v2g_engine_sha256=None)


def test_missing_cav_motion_refused(tmp_path, monkeypatch):
    _ctx, source = _tune_source(tmp_path, monkeypatch, "a" * 64)
    member, _sha = _checkpoint(tmp_path)
    args = _args(tmp_path, member=member, release_spec=tmp_path / "r.json",
                 release_sha="a" * 64, out=tmp_path / "out", cav_motion=None,
                 v2g_option="trajectory_window_delay")
    with pytest.raises(ScorerCliRefusal, match="cav-motion"):
        run_score(args, source_factory=lambda p, expected_release_sha256: source)
    assert not (tmp_path / "out").exists()


def test_missing_v2g_delay_option_refused(tmp_path, monkeypatch):
    _ctx, source = _tune_source(tmp_path, monkeypatch, "a" * 64)
    member, _sha = _checkpoint(tmp_path)
    args = _args(tmp_path, member=member, release_spec=tmp_path / "r.json",
                 release_sha="a" * 64, out=tmp_path / "out",
                 cav_motion="controller", v2g_option=None)
    with pytest.raises(ScorerCliRefusal, match="v2g-delay-option"):
        run_score(args, source_factory=lambda p, expected_release_sha256: source)


def test_cal_cal_refused_before_payload(tmp_path, monkeypatch):
    ctx = _make_pack(tmp_path)
    monkeypatch.setattr(RS, "LF_SHA_CAL_SPLIT_A4", ctx["split_sha"])
    source = _source(tmp_path, ctx, half="cal_cal",
                     worlds=[("w2", "native")], member_sha="a" * 64)
    member, _sha = _checkpoint(tmp_path)
    opened = []
    monkeypatch.setattr(RS, "_hash_file",
                        lambda p, *a, **k: opened.append(str(p)) or "x")
    monkeypatch.setattr(np, "load",
                        lambda p, *a, **k: opened.append(str(p)) or np.zeros(1))
    args = _args(tmp_path, member=member, release_spec=tmp_path / "r.json",
                 release_sha="a" * 64, out=tmp_path / "out",
                 cav_motion="controller",
                 v2g_option="trajectory_window_delay")
    with pytest.raises(ScorerCliRefusal, match="cal_tune"):
        run_score(args, source_factory=lambda p, expected_release_sha256: source)
    assert opened == []


# --------------------------------------------------------------- end2end -----
def _v2g_factory(tmp_path):
    from evalh.scorer.comparator_v2g import bind_v2g_artifacts

    root = tmp_path / "partA_v2g"
    root.mkdir()
    params = root / "params_nominal.json"
    params.write_bytes((json.dumps(_PARAMS, indent=2) + "\n").encode("utf-8"))
    engine = root / "engine.py"
    engine.write_text(_ENGINE_FIXTURE, encoding="utf-8")
    raw = params.read_bytes()
    params_lf = hashlib.sha256(
        raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
    artifacts = bind_v2g_artifacts(
        root, expected_params_lf_sha256=params_lf,
        expected_engine_sha256=hashlib.sha256(
            engine.read_bytes()).hexdigest())
    assert artifacts["params_lf_sha256"] == params_lf
    return artifacts


def test_end_to_end_cli_score_generated_release(tmp_path, monkeypatch):
    ctx = _make_pack(tmp_path)
    monkeypatch.setattr(RS, "LF_SHA_CAL_SPLIT_A4", ctx["split_sha"])
    member, member_sha = _checkpoint(tmp_path)
    source = _source(tmp_path, ctx, half="cal_tune",
                     worlds=[("w1", "native"), ("w1", "damping")],
                     member_sha=member_sha)
    bound = _load(member)
    out = tmp_path / "out"
    args = _args(tmp_path, member=member, release_spec=tmp_path / "r.json",
                 release_sha="a" * 64, out=out,
                 cav_motion="controller",
                 v2g_option="trajectory_window_delay")

    def v2g_factory(src):
        artifacts = _v2g_factory(tmp_path)
        return build_v2g_predictor(src, bound.bundle,
                                   option="trajectory_window_delay",
                                   cav_motion="controller",
                                   artifacts=artifacts)

    done = run_score(
        args, source_factory=lambda p, expected_release_sha256: source,
        checkpoint_loader=lambda p, **kw: bound, v2g_factory=v2g_factory,
        git_head="b" * 40)
    assert (out / "RESULTS.json").is_file()
    assert (out / "DONE.json").is_file()
    results = json.loads((out / "RESULTS.json").read_text("utf-8"))
    assert results["inputs"]["cav_motion"] == "controller"
    assert results["inputs"]["v2g_delay_option"] == "trajectory_window_delay"
    assert results["inputs"]["engine_sha256"]
    assert results["predictors"]["A"] == "A:controller"
    assert results["dynamics_rollout_label"] == "extrapolation"
    cell = results["metrics"]["cells"]["damping-native"]["60"]
    assert cell["n_pairs"] == 1 and cell["n_worlds"] == 1
    assert results["metrics"]["taus"] == [30.0, 60.0, 120.0, 300.0, 570.0]
    assert results["metrics"]["counts"]["n_pairs"] == 2
    assert done["inputs"] == results["inputs"]
    assert set(results["metrics"]["cells"]["damping-native"]) == {"60", "120"}


def test_v2g_predictor_runs_both_cav_motion_modes(tmp_path, monkeypatch):
    _ctx, source = _tune_source(tmp_path, monkeypatch, "a" * 64)
    bound = _load(_checkpoint(tmp_path)[0])
    artifacts = _v2g_factory(tmp_path)
    rows = _rows()
    ctrl = build_v2g_predictor(source, bound.bundle,
                               option="trajectory_window_delay",
                               cav_motion=CAV_MOTION_CONTROLLER,
                               artifacts=artifacts)
    teach = build_v2g_predictor(source, bound.bundle,
                                option="trajectory_window_delay",
                                cav_motion=CAV_MOTION_TEACHER,
                                artifacts=artifacts)
    q_ctrl = ctrl.predict_quantiles(rows)
    q_teach = teach.predict_quantiles(rows)
    assert q_ctrl.shape == (len(rows), N_QUANTILES)
    assert q_teach.shape == (len(rows), N_QUANTILES)
    assert np.isfinite(q_ctrl).all() and np.isfinite(q_teach).all()
    assert not np.allclose(q_ctrl, q_teach)


def test_real_default_a_factory_used_by_cli(tmp_path, monkeypatch):
    """The default model A factory is exercised (real world/head path)."""
    from evalh.scorer.cli import _default_a_factory

    _ctx, source = _tune_source(tmp_path, monkeypatch, "a" * 64)
    bound = _load(_checkpoint(tmp_path)[0])
    pred = _default_a_factory(bound.bundle, source=source,
                              cav_motion=CAV_MOTION_TEACHER)
    assert pred.name == "A:teacher"
    assert pred.predict_quantiles(_rows()).shape == (len(_rows()),
                                                     N_QUANTILES)
    with pytest.raises(ScorerCliRefusal, match="cav-motion"):
        _default_a_factory(bound.bundle, source=source, cav_motion=None)


def test_score_help_lists_new_required_args(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["score", "--help"])
    help_text = capsys.readouterr().out
    assert "--cav-motion" in help_text
    assert "--v2g-delay-option" in help_text
    assert "--dynamics-extrapolation" in help_text
