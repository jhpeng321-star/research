"""Bounded synthetic audit verification; no real packs, fitting, GPU or SUMO."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from output.wd01.wm_n1.hdv.config import HdvConfig
from output.wd01.wm_n1.wms import brakefix_audit as audit
from output.wd01.wm_n1.wms.driver_encoder import DriverEncoder
from output.wd01.wm_n1.wms.model import PhysicsConfig, WmSModel
from output.wd01.wm_n1.wms.stability import FeasibilityProjection, ProjectionConfig


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(1)
    return WmSModel(physics=PhysicsConfig(sumo_fidelity=True), rollout_mode="hybrid",
                    hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16)).cpu().eval()


def sequence(n=8, k=2):
    states = np.zeros((n + 1, k, 7), np.float32)
    states[..., 0] = np.arange(k) * 30 - 200
    states[..., 1] = 1
    states[..., 4] = 10
    states[..., 6] = 5
    valid = np.ones((n + 1, k), bool)
    return states, valid, np.zeros_like(valid)


def write_pack(tmp_path, *, synthetic=True, mixed=False):
    # Even if pytest was invoked with its default external temp path, refuse it.
    audit.workspace_path(tmp_path)
    root = tmp_path / "pack"
    root.mkdir()
    states, valid, _ = sequence(8)
    states[1:, :, 5] = -5
    step = np.zeros((*valid.shape, 16), np.float32)
    items = [dict(world="w1", arm="native", chunk="chunk4", split="TRAIN",
                  tick=0, t0_s=0.0, pair_key="w1@0", plan_row=0, outcome_rows=[])]
    runs = [dict(world="w1", arm="native", chunk="chunk4", split="TRAIN",
                 tick_start=0, tick_end=9)]
    if mixed:
        # Deliberately unusable CAL kinematics: must never be sliced/scored.
        states = np.concatenate((states, np.full_like(states, np.nan)), axis=0)
        valid = np.concatenate((valid, valid), axis=0)
        step = np.concatenate((step, step), axis=0)
        items.append(dict(items[0], world="cal", split="CALIBRATION", tick=9))
        runs.append(dict(runs[0], world="cal", split="CALIBRATION", tick_start=9, tick_end=18))
    arrays = {"veh_kin": states, "veh_mask": valid.astype(np.uint8),
              "target_mask": valid.astype(np.uint8), "veh_step": step}
    specs = {}
    for name, array in arrays.items():
        np.save(root / f"{name}.npy", array)
        specs[name] = dict(file=f"{name}.npy", shape=list(array.shape), dtype=str(array.dtype))
    manifest = dict(schema="wm_s_brakefix_synthetic/1" if synthetic else "wm_n1_pack/1",
                    arrays=specs, items=items)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "INDEX.json").write_text(json.dumps(dict(runs=runs)), encoding="utf-8")
    return root


def edit_json(path, edit):
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_global_envelope_is_not_recorded_state_cap_coverage(model):
    states, valid, cav = sequence(1, 1)
    states[..., 4] = 0  # free-road physics accelerates, even though next a=-5
    states[1, 0, 5] = -5
    acc = audit.Accumulator()
    acc.add(states, valid, cav, model=model, chunk=60)
    rep = acc.report()
    row = next(r for r in rep["settings"] if r["setting"] == "hybrid / cap 1.5")
    assert row["global_lower_envelope_mps2"] == pytest.approx(-5.7)
    assert row["within_global_lower_envelope_share_of_braking"]["a<-3"] == 1.0
    need = rep["needed_negative_correction"]["driver_defaults"]["a<-3"]
    assert need["mean_signed_needed_correction_mps2"] == pytest.approx(-7.4897308, abs=1e-5)
    assert need["share_negative_correction_exceeding_cap"]["1.5"] == 1.0
    assert row["recorded_state_raw_accel_interval_share_of_braking"]["a<-3"] == 0.0
    assert rep["acceleration_alignment"] == "assumed_destination_channel_not_verified_real"


def test_driver_scenario_recomputes_idm_not_just_decel_clamp(model):
    states, valid, _ = sequence(1)
    states[0, :, 4] = [20, 19]
    states[0, :, 0] = [-200, -140]
    low = audit.Accumulator._physics(model, states[:1], valid[:1], 2.7)
    high = audit.Accumulator._physics(model, states[:1], valid[:1], 5.7)
    # Both above their clamp floors; b still changes the IDM desired gap.
    assert low[0, 0] > -2.7 and high[0, 0] > -5.7
    assert low[0, 0] < high[0, 0]


def test_chunk_tail_and_cross_chunk_propagation_match_whole_window(model):
    states, valid, cav = sequence(8)
    states[3, 1, 5] = -5  # onset transition 2, supervises follower transitions 3..5
    states[5, 0, 5] = -4
    small, whole = audit.Accumulator(), audit.Accumulator()
    small.add(states, valid, cav, model=model, chunk=3, window_steps=3)
    whole.add(states, valid, cav, model=model, chunk=60, window_steps=3)
    assert small.report() == whole.report()
    assert small.report()["prop_window_hdv_steps_by_threshold"]["-3"] == 3
    assert small.hdv_steps == 16


def test_only_continuous_valid_hdv_transitions_are_counted():
    states, valid, cav = sequence(3, 3)
    states[1:, :, 5] = -5
    valid[0, 0] = False  # first destination alone is not a usable transition
    valid[2, 1] = False  # excludes both sides of internal-lane gap
    cav[:, 2] = True
    acc = audit.Accumulator()
    acc.add(states, valid, cav, chunk=2)
    assert acc.report()["hdv_steps"] == 3
    assert acc.report()["needed_correction_evaluated_hdv_steps"] == 0
    assert acc.report()["needed_negative_correction"]["driver_defaults"]["a<-3"]["steps"] == 0


@pytest.mark.parametrize("n,k,all_invalid", [(0, 2, False), (2, 0, False), (2, 2, True)])
def test_zero_samples_are_null_not_zero_coverage_or_crashing_quantiles(n, k, all_invalid):
    states, valid, cav = sequence(n, k)
    if all_invalid:
        valid[:] = False
        states[:] = np.nan
    acc = audit.Accumulator()
    acc.add(states, valid, cav)
    rep = acc.report()
    assert rep["hdv_steps"] == 0 and rep["min_recorded_accel_mps2"] is None
    assert all(v is None for v in rep["quantiles_mps2_binned_0.5"].values())
    assert all(v is None for v in rep["share_steps_below"].values())
    json.dumps(rep, allow_nan=False)


def test_nonfinite_valid_data_refuses_and_histogram_tails_are_truthful():
    states, valid, cav = sequence(1)
    states[1, :, 5] = [-15, 6]
    acc = audit.Accumulator()
    acc.add(states, valid, cav)
    rep = acc.report()
    assert rep["min_recorded_accel_mps2"] == -15
    assert rep["histogram"]["underflow"] == rep["histogram"]["overflow"] == 1
    assert sum(rep["histogram"]["counts"]) == 0
    assert rep["quantiles_mps2_binned_0.5"]["q0.5"] == {"below": -12.0}
    states[1, 0, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        audit.Accumulator().add(states, valid, cav)


def test_tanh_boundary_is_not_claimed_finite_cap_coverage():
    states, valid, cav = sequence(1, 1)
    states[1, 0, 5] = -4
    fake = SimpleNamespace(physics=PhysicsConfig(sumo_fidelity=True), physics_step=None,
                           physics_accel=lambda state, params: torch.full(state["present"].shape, -2.5))
    acc = audit.Accumulator()
    acc.add(states, valid, cav, model=fake)
    row = acc.report()["needed_negative_correction"]["driver_defaults"]["a<-3"]
    assert row["share_negative_correction_exceeding_cap"]["1.5"] == 0
    assert row["at_negative_tanh_boundary_steps_tol_1e-6"]["1.5"] == 1
    assert any("limiting boundary" in s for s in acc.report()["limitations"])


def test_projection_changes_speed_but_not_reported_acceleration(model):
    states, valid, _ = sequence(1, 1)
    kin = torch.tensor(states[:1])
    state = dict(veh_kin=kin, s_m=kin[..., 0], v_mps=kin[..., 4], a_mps2=kin[..., 5],
                 lane=torch.zeros((1, 1), dtype=torch.long), present=torch.tensor(valid[:1]))
    projection = FeasibilityProjection(ProjectionConfig(v_max_mps=1))
    plain = model.step(state, k_steps=0)
    projected = model.step(state, k_steps=0, projection=projection)
    assert projected["next_v_mps"].item() == 1
    assert plain["next_v_mps"].item() > 1
    torch.testing.assert_close(projected["accel_mps2"], plain["accel_mps2"])
    assert model.gate(0).item() == 0


def test_encoder_decel_outer_envelope_is_shrunk_for_finite_history(model):
    encoder = DriverEncoder()
    defaults = torch.zeros(1, 1, 9)
    defaults[..., 4] = audit.DECEL_DEFAULT
    out = encoder(torch.zeros(1, 1, 10, 6), torch.ones(1, 1, 10, dtype=torch.bool),
                  torch.zeros(1, 1, 4), defaults)
    shrink = 10 / 18
    assert out["shrinkage"].item() == pytest.approx(shrink)
    assert abs(out["values"][0, 0, 4].item() - 4.2) <= shrink * 1.5


def test_pack_is_lazy_mmap_bounded_train_and_never_resurrects_slots(tmp_path, monkeypatch):
    root = write_pack(tmp_path, mixed=True)
    pack = audit.AuditPack(root)
    assert pack._mm == {} and len(pack.items) == 1
    calls = []
    original = audit.np.load

    def load(path, **kwargs):
        calls.append((Path(path).name, kwargs))
        return original(path, **kwargs)

    monkeypatch.setattr(audit.np, "load", load)
    window = pack.window(pack.items[0], horizon_steps=3)
    assert window["states"].shape == (4, 2, 7)
    assert all(isinstance(a, np.memmap) for a in pack._mm.values())
    assert all(c[1] == dict(mmap_mode="r", allow_pickle=False) for c in calls)
    assert {c[0] for c in calls} == {"veh_kin.npy", "veh_mask.npy", "target_mask.npy", "veh_step.npy"}
    assert np.isfinite(window["states"]).all()
    # Release mapping before modifying the fixture on Windows.
    for a in pack._mm.values():
        a._mmap.close()
    pack._mm.clear()
    tm = original(root / "target_mask.npy")
    tm[1, 0] = 0
    np.save(root / "target_mask.npy", tm)
    window = pack.window(pack.items[0], horizon_steps=8)
    assert not window["loss_valid"][2:, 0].any()


@pytest.mark.parametrize("split,chunk", [("TEST", "chunk4"), ("TRAIN", "Chunk - 5"),
                                         ("VAL", "chunk4")])
def test_sealed_or_unknown_metadata_refuses_before_numpy_load(tmp_path, monkeypatch, split, chunk):
    root = write_pack(tmp_path)
    edit_json(root / "manifest.json", lambda m: m["items"][0].update(split=split, chunk=chunk))
    monkeypatch.setattr(audit.np, "load", lambda *a, **kw: pytest.fail("no array may be opened"))
    with pytest.raises(PermissionError):
        audit.AuditPack(root)


def test_index_train_mismatch_and_array_path_escape_refuse_before_load(tmp_path, monkeypatch):
    root = write_pack(tmp_path)
    monkeypatch.setattr(audit.np, "load", lambda *a, **kw: pytest.fail("no array may be opened"))
    edit_json(root / "INDEX.json", lambda m: m["runs"][0].update(split="CALIBRATION"))
    with pytest.raises(PermissionError, match="TRAIN INDEX"):
        audit.AuditPack(root)
    edit_json(root / "manifest.json", lambda m: m["arrays"]["veh_kin"].update(file="../elsewhere.npy"))
    with pytest.raises(PermissionError, match="pack-local"):
        audit.AuditPack(root)


def test_external_paths_refuse_without_stat_or_read(monkeypatch):
    original = Path.lstat
    observed = []

    def lstat(path, *args, **kwargs):
        observed.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(PermissionError, match="outside workspace"):
        audit.workspace_path(audit.WORKSPACE.parent / "external")
    assert observed == []


def test_reparse_ancestor_refuses_without_following_target(tmp_path, monkeypatch):
    root = write_pack(tmp_path)
    original = Path.lstat

    def lstat(path, *args, **kwargs):
        if path == root:
            return SimpleNamespace(st_mode=stat_mode, st_file_attributes=0x400)
        return original(path, *args, **kwargs)

    stat_mode = root.lstat().st_mode
    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: pytest.fail("must not read through reparse"))
    with pytest.raises(PermissionError, match="reparse"):
        audit.AuditPack(root)


def test_real_manifest_dtype_spelling_loads_and_malformed_arrays_fail_closed(tmp_path):
    # The real packs disagree on the manifest dtype token: pack_v3 writes
    # "float16" while pack_dspeed/pack_damping write str(np.float16) =
    # "<class 'numpy.float16'>".  The trainer never compares the token, so the
    # audit must normalise both spellings rather than string-equate them.
    root = write_pack(tmp_path, synthetic=False)

    def to_class_token(m):
        for spec in m["arrays"].values():
            spec["dtype"] = str(np.dtype(spec["dtype"]).type)

    edit_json(root / "manifest.json", to_class_token)
    pack = audit.AuditPack(root)
    window = pack.window(pack.items[0], horizon_steps=3)
    assert window["states"].shape == (4, 2, 7)
    assert np.isfinite(window["states"]).all()
    for array in pack._mm.values():
        array._mmap.close()
    pack._mm.clear()
    # A dtype token that resolves to a different numpy type must still refuse.
    edit_json(root / "manifest.json",
              lambda m: m["arrays"]["veh_kin"].update(dtype="float64"))
    with pytest.raises(ValueError, match="veh_kin"):
        audit.AuditPack(root).array("veh_kin")
    # An unrecognised token fails closed, it does not crash or pass through.
    edit_json(root / "manifest.json",
              lambda m: m["arrays"]["veh_kin"].update(dtype="<class 'numpy.notatype'>"))
    with pytest.raises(ValueError, match="unrecognised manifest dtype"):
        audit.AuditPack(root).array("veh_kin")
    # A genuinely malformed shape is still refused.
    edit_json(root / "manifest.json",
              lambda m: m["arrays"]["veh_kin"].update(dtype="float32",
                                                      shape=[999, 2, 7]))
    with pytest.raises(ValueError, match="veh_kin"):
        audit.AuditPack(root).array("veh_kin")


def test_cli_synthetic_report_zero_samples_and_no_overwrite(tmp_path, capsys):
    root = write_pack(tmp_path)
    out = tmp_path / "audit.json"
    args = ["--pack", str(root), "--n-windows", "0", "--out", str(out)]
    assert audit.main(args) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["evidence_kind"] == "synthetic_only"
    assert rep["windows"] == rep["hdv_steps"] == 0
    assert rep["real_data_alignment_verified"] is False
    assert rep["sampled_windows_by_pack"] == {"pack": 0}
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        audit.main(args)
    assert out.read_bytes() == before


def test_cli_train_filter_records_actual_pack_coverage(tmp_path, capsys):
    root = write_pack(tmp_path)
    keep = tmp_path / "train_windows.json"
    keep.write_text(json.dumps(dict(schema=lpo_schema(), split="TRAIN", n_windows=1,
                                    windows=[dict(pack="pack", chunk="chunk4", world="w1",
                                                  arm="native", t0_s=0)])), encoding="utf-8")
    assert audit.main(["--pack", str(root), "--window-list", str(keep),
                       "--n-windows", "1", "--horizon-steps", "3", "--chunk-steps", "2"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["sampled_windows_by_pack"] == {"pack": 1}
    assert rep["hdv_steps"] == rep["needed_correction_evaluated_hdv_steps"] == 6
    assert rep["prop_window_steps"] == 6
    assert "-3" in rep["prop_window_hdv_steps_by_threshold"]
    edit_json(keep, lambda m: m.update(split="CALIBRATION"))
    with pytest.raises(PermissionError, match="TRAIN"):
        audit.main(["--pack", str(root), "--window-list", str(keep)])


def lpo_schema():
    return audit.lpo.WINDOW_SCHEMA


def test_sampling_validation_and_small_sample_does_not_promise_all_packs():
    assert audit.pick_indices(10, 0) == audit.pick_indices(0, 10) == []
    assert audit.pick_indices(10, 1) == [0]
    assert audit.pick_indices(4, 20) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        audit.pick_indices(10, -1)


def test_alignment_diagnostics_on_consistent_recorded_transitions():
    states, valid, cav = sequence(4, 1)
    states[:, 0, 4] = [10, 9, 7, 6.5, 6.5]
    states[1:, 0, 5] = np.diff(states[:, 0, 4]) / 0.5
    acc = audit.Accumulator()
    acc.add(states, valid, cav)
    diag = acc.report()["acceleration_alignment_diagnostics"]
    assert diag["channel_vs_speed_fd_correlation"] == pytest.approx(1.0)
    assert diag["mean_abs_difference_mps2"] == diag["max_abs_difference_mps2"] == 0.0


def test_external_read_opt_in_keeps_output_workspace_guard(tmp_path, monkeypatch):
    root = write_pack(tmp_path)
    original = audit.workspace_path
    monkeypatch.setattr(audit, "workspace_path", lambda *a, **kw: pytest.fail("input guard bypass only when explicit"))
    pack = audit.AuditPack(root, readonly_external=True)
    assert pack.window(pack.items[0], 2)["states"].shape == (3, 2, 7)
    monkeypatch.setattr(audit, "workspace_path", original)
    with pytest.raises(PermissionError, match="outside workspace"):
        audit.workspace_path(audit.WORKSPACE.parent / "external.json", new=True)


def test_metadata_only_smoke_preserves_identity_and_excludes_pi():
    from output.wd01.wm_n1.wms.brakefix_smoke import tiny_manifest
    rows = [dict(pack="pack_v3", batch="b", world=w, arm=a, t0_s=0)
            for w, a in [("pi", "pi"), ("one", "native"), ("two", "damping"), ("three", "native")]]
    source = dict(schema=lpo_schema(), split="TRAIN", windows=rows, n_windows=len(rows))
    tiny = tiny_manifest(source, ["pack_v3"], {("b", "pi__pi")})
    assert tiny["windows"] == rows[1:3] and tiny["n_windows"] == 2
    assert source["n_windows"] == 4
