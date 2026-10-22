"""R9TDIAG: training-path contract fixes found by the real-batch diagnosis."""
import pytest
import torch
from types import SimpleNamespace

from scripts.r9_tdiag_geometry import capture_rollout_geometry, wm_diag
from test_merge_pivot_training import _fixture
from wms.model.entries import ENTRY_S_M, EntranceHistory
from wms.model.merge_pivot import geometry
from wms.physics.state import state_kin
from wms.train.inputs import prepare_inputs
from wms.train.losses.compose import build_loss
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner


def _runner(model, **extra):
    settings = TrainSettings(rollout_mode="hybrid", lane_model="hazard_v5", merge_decision="first_passage",
                             merge_core_tau_reg_weight=.1, truncation_steps=1, **extra)
    return UnrollRunner(model, settings, build_loss(settings, physics=model.physics),
                        projection=None, device=torch.device("cpu"))


def _batch(inp):
    return dict(states=inp.states, present=inp.present, is_cav=torch.zeros_like(inp.is_cav))


def test_differentiable_unroll_term_with_all_inputs_frozen_does_not_abort_the_backward():
    """Round 9 trains only merge_pivot but inherits differentiable unroll terms (lane_event / stability).

    Their value has no autograd graph (all inputs frozen), so ``value.backward()`` raised "element 0 of
    tensors does not require grad" at the end of EVERY batch, after the chunk gradients were already
    accumulated.  The empty backward is skipped; the merge_pivot gradient is untouched.
    """
    model, inp, _ = _fixture()
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("merge_pivot"))
    runner = _runner(model, stability_weight=.5)
    assert "stability" in runner.composer.names and runner.composer.differentiable("stability")
    result = runner.run(_batch(inp), unroll_steps=4, backward=True)
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert result["stability"] is not None
    grad = model.merge_pivot.tau_raw.grad
    assert grad is not None and grad > 0
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith("merge_pivot"))


def test_differentiable_unroll_term_with_trainable_inputs_is_still_backpropagated():
    model, inp, _ = _fixture()
    runner = _runner(model, stability_weight=.5)
    runner.run(_batch(inp), unroll_steps=4, backward=True)
    stab = [p.grad for n, p in model.named_parameters() if n.startswith("residual_net") and p.grad is not None]
    assert stab and any(bool(g.abs().sum() > 0) for g in stab)


def _entry_fixture():
    model, _, _ = _fixture()
    states = torch.zeros(1, 5, 192, 7)
    states[:, :, :5, 6] = 5.
    states[:, :, :5, 0] = torch.tensor([10., 35., -25., ENTRY_S_M[0] + 6., ENTRY_S_M[1] + 6.])
    states[:, :, 0, 3] = 1.
    states[:, :, 1:3, 2] = 1.
    states[:, :, 3, 1] = 1.
    states[:, :, 4, 2] = 1.
    present = torch.zeros(1, 5, 192, dtype=torch.bool)
    present[:, :, :5] = True
    batch = dict(states=states, present=present, is_cav=torch.zeros_like(present),
                 entry_feat=torch.tensor(EntranceHistory.defaults().to_features())[None],
                 entry_queue=torch.tensor([[1., 0., 0., 0.]]))
    settings = TrainSettings(entries_v2=True, entry_pad=1, entry_pad_auto=True,
                             entry_queue_state=True, entry_insert_rule="sumo_like_v2")
    inp = prepare_inputs(batch, model, settings, torch.device("cpu"), 4)
    # Hold the ramp driver's sampled willingness fixed so it remains in the
    # risk zone while we exercise queue admission across successive chunks.
    inp.initial["merge_pivot_present"] = inp.initial["present"].clone()
    inp.initial["merge_pivot_started"] = inp.initial["lane"] == 2
    inp.initial["merge_pivot_z"] = torch.full_like(inp.initial["s_m"], -1000.)
    # No new requests: exercise only the real queue's one pending arrival.
    inp.entries._emit_due = lambda b: []
    return model, inp


def _entry_rollout(model, inp, state, n=1):
    return model.rollout(state, n_steps=n, entries=inp.entries,
                         present_seq=inp.present_seq[:, :n + 1],
                         lane_uniform=torch.full((1, n, inp.present.shape[-1]), 1.e-8))


def test_pending_entrant_is_absent_and_admission_keeps_actual_length_across_chunks(monkeypatch):
    """Queue membership is not road occupancy; diagnostics must use the insertion snapshot."""
    import wms.model.merge_pivot as pivot

    model, inp = _entry_fixture()
    calls = []
    original = pivot.geometry

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(pivot, "geometry", counted)
    # Arbitrary unused-slot coordinates cannot decide presence.
    inp.initial["s_m"][:, 192:] = -1733.
    assert inp.present_seq[:, :, 192:].all()
    with torch.no_grad(), capture_rollout_geometry(type(model)):
        queued = _entry_rollout(model, inp, inp.initial)
        snap = queued["r9_decision_scenes"][0]
        assert inp.entries.pending[0, 0] == 1
        assert not snap["present"][:, 192:].any()
        assert snap["geometry"]["scene_valid"].all()
        assert snap["geometry"]["eligible"][0, 0]
        assert not queued["masks"][:, :, 192:].any()

        # Open both entrance lanes at the next chunk boundary.
        current = queued["state"]
        current["present"][:, 3:5] = False
        entered = _entry_rollout(model, inp, current, n=2)
        assert inp.entries.pending[0, 0] == 0
        assert entered["entrant"][0, 192]
        for scene in entered["r9_decision_scenes"]:
            assert scene["present"][0, 192]
            assert scene["kin"][0, 192, 6] > 0
            assert scene["geometry"]["scene_valid"].all()
            assert scene["geometry"]["eligible"][0, 0]
        assert not snap["present"][0, 192]  # prior snapshot is immutable

        # Reproduce the old probe: runtime presence + recorded zero length
        # manufactures a scene veto for a perfectly valid admitted vehicle.
        actual = entered["r9_decision_scenes"][1]
        wrong = actual["kin"].clone()
        wrong[..., 6] = inp.states[:, 1, :, 6]
        fabricated = original(wrong, actual["present"], model.physics)
        assert not fabricated["scene_valid"].any()
        assert not fabricated["eligible"][0, 0]
        ctx = SimpleNamespace(pred=entered, targets=SimpleNamespace(states=inp.states), c0=0)
        prep = {"risk": inp.present}
        before = len(calls)
        for j in range(2):
            diag = wm_diag(prep, ctx, 0, j, 0)
            assert diag["scene_valid"] and diag["elig"]
            assert diag["n_unobserved_present"] == 0
        assert len(calls) == before == 3  # reporting never re-enters/corrupts geometry counters

        resumed = _entry_rollout(model, inp, entered["state"])
        assert resumed["state"]["length_m"][0, 192] == actual["kin"][0, 192, 6]
        assert resumed["r9_decision_scenes"][0]["geometry"]["scene_valid"].all()


@pytest.mark.parametrize("position", [-1733., 10.])
@pytest.mark.parametrize("invalid", ["length", "nonfinite"])
def test_genuinely_present_invalid_vehicle_still_vetoes_scene(position, invalid):
    model, inp = _entry_fixture()
    state = inp.initial
    state["present"][0, 192] = True
    state["lane"][0, 192] = 1
    state["s_m"][0, 192] = position
    if invalid == "nonfinite":
        state["length_m"][0, 192] = 5.
        state["v_mps"][0, 192] = torch.nan
    kin = state_kin(state)
    geo = geometry(kin, state["present"], model.physics)
    assert geo["ramp"][0, 0]
    assert not geo["scene_valid"].any()
    assert not geo["eligible"].any()
    assert torch.isneginf(model.merge_pivot(geo)[0, 0])


def test_geometry_probe_restores_methods_after_failure():
    model, _, _ = _fixture()
    cls = type(model)
    before = cls.rollout, cls.pivot_decision_geometry
    with pytest.raises(RuntimeError, match="probe failed"):
        with capture_rollout_geometry(cls):
            raise RuntimeError("probe failed")
    assert (cls.rollout, cls.pivot_decision_geometry) == before
