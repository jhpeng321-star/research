"""Exact P1--P3 sequence scoring, causal frozen replay and opt-in plumbing."""
from dataclasses import replace
import pytest
import torch

from wms.config.train import TrainConfig
from wms.model import ModelSpec, WmSModel
from wms.model.merge_pivot import (FirstPassageNet, advance_longitudinal, geometry,
                                    log_event_mass, log_survival, update_state)
from wms.physics.config import PhysicsConfig
from wms.train.inputs import build_targets, prepare_inputs, state_from_kin
from wms.train.losses.base import LossContext
from wms.train.losses.compose import build_loss
from wms.train.losses.merge_pivot import (MergePivotConfig, MergePivotTerm,
                                         build_recorded_geometry, merge_pivot_metrics, score_chunk)
from wms.train.settings import TrainSettings


def _score(values, events=None, *, risk=None, support=None, stop=None, known=True,
           carry=None, initial=None):
    c = torch.tensor(values, dtype=torch.float64).reshape(1, -1, 1)
    ones = torch.ones_like(c, dtype=torch.bool)
    event = torch.zeros_like(ones) if events is None else torch.tensor(events).reshape_as(ones)
    return score_chunk(c, c.new_tensor(.7), ones if risk is None else torch.tensor(risk).reshape_as(ones),
                       event, known_entry=ones & known, reset=~ones, carry=carry,
                       support=None if support is None else torch.tensor(support).reshape_as(ones),
                       stop=None if stop is None else torch.tensor(stop).reshape_as(ones),
                       initial_maximum=initial)


def test_exact_event_and_censor_and_mass_conservation():
    c = torch.tensor([-2., -1., -.5, .7], dtype=torch.float64)
    tau = c.new_tensor(.7)
    mass = torch.stack([log_event_mass(c.new_tensor(-torch.inf) if i == 0 else c[i-1], x, tau).exp()
                        for i, x in enumerate(c)])
    assert float(mass.sum() + log_survival(c[-1], tau).exp()) == pytest.approx(1., abs=1e-12)
    nll, _, stats = _score(c.tolist(), [False, False, False, True])
    assert float(nll) == pytest.approx(float(-mass[-1].log()), abs=1e-12)
    censored, _, _ = _score(c.tolist())
    assert float(censored) == pytest.approx(float(-log_survival(c[-1], tau)), abs=1e-12)
    assert stats["events"] == 1 and stats["zero_mass_events"] == 0


@pytest.mark.parametrize("split", [1, 2, 3])
@pytest.mark.parametrize("event", [False, True])
def test_chunks_telescope_and_preserve_survival_posterior(split, event):
    values = [-2., -1., -1.5, .4]
    events = [False, False, False, event]
    whole, _, _ = _score(values, events)
    first, carry, _ = _score(values[:split], events[:split])
    last, _, _ = _score(values[split:], events[split:], carry=carry)
    assert float(first + last) == pytest.approx(float(whole), abs=1e-12)
    assert float(carry["maximum"]) == max(values[:split])


def test_explicit_delayed_entry_conditions_on_available_maximum():
    initial = torch.tensor([[-1.2]], dtype=torch.float64)
    nll, _, _ = _score([-.4, .5], [False, True], known=False, initial=initial)
    expected = -log_event_mass(torch.tensor([[-.4]], dtype=torch.float64),
                               torch.tensor([[.5]], dtype=torch.float64), initial.new_tensor(.7))
    expected += log_survival(initial, initial.new_tensor(.7))
    assert float(nll) == pytest.approx(float(expected), abs=1e-12)


def test_unknown_age_masked_and_enumerated_without_population_prior():
    nll, carry, stats = _score([-.5, 1.], [False, True], known=False)
    assert nll == 0 and stats["unknown_age"] == 1 and stats["unknown_age_events"] == 1
    assert stats["scored_intervals"] == 0 and carry["terminal"].all()


@pytest.mark.parametrize("case", ["exit", "support", "stop", "event"])
def test_terminal_paths_cannot_regain_support_or_score_postevent(case):
    kwargs = {}
    events = [False] * 4
    if case == "exit":
        kwargs["risk"] = [True, False, True, True]
    elif case == "support":
        kwargs["support"] = [True, False, True, True]
    elif case == "stop":
        kwargs["stop"] = [True, False, False, False]
    else:
        events[0] = True
    nll, carry, stats = _score([-1., 5., 8., 9.], events, **kwargs)
    expected = -log_survival(torch.tensor(-1., dtype=torch.float64), torch.tensor(.7))
    if case == "event":
        expected = -log_event_mass(torch.tensor(-torch.inf), torch.tensor(-1.), torch.tensor(.7))
    assert float(nll) == pytest.approx(float(expected), abs=1e-6)
    assert stats["scored_intervals"] == 1 and float(carry["maximum"]) == -1.


def test_zero_mass_kept_infinite_and_counted_and_veto_has_no_support():
    nll, _, stats = _score([1., .5], [False, True])
    assert torch.isposinf(nll) and stats["zero_mass_events"] == 1
    veto, _, counts = _score([-torch.inf], [True])
    assert torch.isposinf(veto) and counts["physical_fail_events"] == 1
    assert counts["zero_mass_events"] == 1


def test_deterministic_atom_and_impossible_survival_are_exact():
    nll, _, stats = _score([torch.inf], [True])
    assert nll == 0 and stats["zero_mass_events"] == 0
    impossible, _, stats = _score([torch.inf, torch.inf])
    assert torch.isposinf(impossible) and stats["zero_survival"] == 1
    assert stats["scored_intervals"] == 1
    missed, _, counts = _score([torch.inf, torch.inf], [False, True])
    assert torch.isposinf(missed) and counts["zero_mass_events"] == 1


def test_identity_reset_permits_new_likelihood_after_exit():
    c = torch.tensor([[[-1.], [0.], [1.]]])
    risk = torch.tensor([[[True], [False], [True]]])
    reset = torch.tensor([[[False], [False], [True]]])
    nll, carry, stats = score_chunk(c, torch.tensor(.7), risk, torch.zeros_like(risk),
                                    known_entry=torch.ones_like(risk), reset=reset)
    expected = -log_survival(c[:, 0], torch.tensor(.7)) - log_survival(c[:, 2], torch.tensor(.7))
    assert torch.allclose(nll, expected.squeeze()) and stats["births"] == 2
    assert not carry["terminal"].any()


def test_exact_max_boundary_has_finite_nonzero_all_seven_parameter_gradients():
    net = FirstPassageNet().double()
    # Two drivers: follower-constrained and leader-constrained, both with a
    # positive braking term. Neither min side nor braking coefficient is dormant.
    geo = dict(v=torch.tensor([[[5., 12.], [5., 12.]]], dtype=torch.float64),
               v_f=torch.tensor([[[12., 1.], [12., 1.]]], dtype=torch.float64),
               v_l=torch.tensor([[[20., 4.], [20., 4.]]], dtype=torch.float64),
               gap_f=torch.tensor([[[15., 1000.], [25., 1000.]]], dtype=torch.float64),
               gap_l=torch.tensor([[[1000., 15.], [1000., 25.]]], dtype=torch.float64))
    ones = torch.ones(1, 2, 2, dtype=torch.bool)
    geo.update(eligible=ones, has_f=ones, has_l=ones)
    c = net(geo)
    event = torch.tensor([[[False, False], [True, True]]])
    nll, _, stats = score_chunk(c, net.params()["tau"], ones, event,
                                known_entry=ones, reset=~ones)
    nll.backward()
    assert torch.isfinite(nll) and stats["events"] == 2
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def _nested(**loss):
    return TrainConfig.from_dict(dict(model=dict(rollout_mode="hybrid", lane_model="hazard_v5",
                                                merge_decision="first_passage", merge_st_width=.4),
                                      loss=dict(merge_core=loss)))


def test_nested_flat_settings_and_composer_route_only_new_exact_term():
    cfg = _nested(weight=.4, tf_weight=.6, z_tf=395.2, z_on=2.15, tau_reg_weight=.01)
    settings = TrainSettings.from_config(cfg)
    assert cfg.to_legacy_flat()["merge_st_width"] == settings.merge_st_width == .4
    assert settings.merge_core_tau_reg_weight == .01
    composer = build_loss(settings)
    assert "merge_pivot" in composer.names and "merge_pivot" in cfg.required_log_terms()
    assert not {"merge_core", "merge_first_passage", "merge_frailty", "merge_timing"} & set(composer.names)
    term = next(t for t in composer.chunk_terms if t.name == "merge_pivot")
    assert (term.cfg.weight, term.cfg.tf_weight, term.cfg.z_tf, term.cfg.z_on) == (.4, .6, 395.2, 2.15)
    # Even conflicting historical likelihood flags can never route to a hazard.
    assert "merge_timing" not in build_loss(replace(settings, merge_timing_weight=1.)).names
    assert "merge_frailty" not in build_loss(replace(settings, merge_frailty_weight=1.)).names
    defaults = TrainConfig().to_legacy_flat()
    assert "merge_st_width" not in defaults and "merge_core_tau_reg_weight" not in defaults
    assert build_loss(TrainSettings()).names == ("position", "speed", "lane")


@pytest.mark.parametrize("overrides", [dict(merge_negotiation=True), dict(merge_core_yield_weight=.1),
                                       dict(merge_yield_huber=.1), dict(merge_ramp_adapt_huber=.1),
                                       dict(merge_safety_bce=.1), dict(merge_survival_negative_weight=.1),
                                       dict(merge_st_width=0.), dict(merge_st_width=float("nan")),
                                       dict(merge_core_tau_reg_weight=-1.)])
def test_invalid_first_passage_flat_configs_refused(overrides):
    with pytest.raises(ValueError):
        TrainSettings(merge_decision="first_passage", lane_model="hazard_v5", **overrides)


def test_nested_first_passage_rejects_b_weights_and_negotiation():
    with pytest.raises(ValueError):
        _nested(yield_weight=.1)
    with pytest.raises(ValueError):
        TrainConfig.from_dict(dict(model=dict(rollout_mode="hybrid", lane_model="hazard_v5",
                                              merge_decision="first_passage", merge_negotiation=True)))


def _fixture(passage_law="strict"):
    torch.manual_seed(12)
    model = WmSModel(ModelSpec(rollout_mode="hybrid", lane_model="hazard_v5",
                               merge_decision="first_passage", merge_passage_law=passage_law),
                      physics=PhysicsConfig()).eval()
    # One observed pre-zone state, then entry, then a merge. Full scene includes
    # a CAV neighbor. Its current motion is modeled without the future teacher.
    states = torch.zeros(1, 5, 3, 7)
    states[..., 6] = 5.
    for i in range(5):
        states[0, i, :, 0] = torch.tensor([-1. + i * 4., 35. + i * 6., -25. + i * 3.])
        states[0, i, :, 4] = torch.tensor([8., 12., 6.])
        states[0, i, 0, 3 if i < 3 else 2] = 1.
        states[0, i, 1:, 2] = 1.
    present = torch.ones(1, 5, 3, dtype=torch.bool)
    is_cav = torch.zeros_like(present)
    is_cav[:, :, 1] = True
    batch = dict(states=states, present=present, is_cav=is_cav)
    settings = TrainSettings(rollout_mode="hybrid", lane_model="hazard_v5")
    inp = prepare_inputs(batch, model, settings, torch.device("cpu"), 4)
    targets = build_targets(inp, batch, settings, .5, torch.device("cpu"), physics=model.physics)
    targets.prepared["merge_pivot_source"] = (model, inp)
    return model, inp, targets


def test_frozen_recorded_replay_matches_runtime_causal_update_and_boundary_to_1e6():
    model, inp, targets = _fixture()
    # No CAV overrides in this parity fixture: both execute the same frozen forward.
    inp.is_cav.zero_()
    state = state_from_kin(inp.states[:, 0], inp.present[:, 0])
    state["is_cav"] = inp.is_cav[:, 0]
    from wms.model.history import history_step_features, initial_history
    from wms.physics.neighbours import lead_neighbour
    gap, _ = lead_neighbour(state["veh_kin"], state["present"], model.physics)
    hist, mask = initial_history(history_step_features(state["veh_kin"], gap),
                                 state["present"], model.history_len)
    with torch.no_grad():
        out = model.step(state, history=hist, history_mask=mask,
                         lane_uniform=torch.full((1, 3), .5))
        moved = advance_longitudinal(state["veh_kin"], out["accel_mps2"], .5,
                                     model.physics.stop_time_integration)
        # The recorded scene is evaluated in the lossless route frame (junction lengths);
        # parity means the same geometry() on the same pre-state shift, not a different one.
        from wms.data.route_frame import route_shift
        shift = route_shift(inp.states, inp.present, getattr(inp, "continuity", None),
                            dt_s=model.physics.dt_s)[:, 0]
        expected = geometry(moved, state["present"], model.physics, route_shift=shift)
    assert expected["route_s"].ne(expected["s"]).any(), "fixture must exercise the shift"
    rec = build_recorded_geometry(model, inp, targets)
    for key in expected:
        assert torch.allclose(rec[key][:, 0], expected[key], atol=1e-6, rtol=0), key
    c_rec = model.merge_pivot({key: x[:, :1] for key, x in rec.items()})[:, 0]
    c_run = model.merge_pivot(expected)
    assert torch.allclose(c_rec, c_run, atol=1e-6, rtol=0)
    tau = model.merge_pivot.params()["tau"]
    for fn in (log_survival,):
        assert torch.allclose(fn(c_rec, tau), fn(c_run, tau), atol=1e-6, rtol=0)
    initial = torch.full_like(c_run, -torch.inf)
    assert torch.allclose(log_event_mass(initial, c_rec, tau), log_event_mass(initial, c_run, tau),
                          atol=1e-6, rtol=0)
    a = update_state(c_rec, tau, expected["ramp"], state["present"], uniform=torch.full_like(c_rec, .6))
    b = update_state(c_run, tau, expected["ramp"], state["present"], uniform=torch.full_like(c_run, .6))
    assert torch.equal(a["cross"], b["cross"])


def test_recorded_step_has_no_next_row_motion_or_acceleration_leakage():
    model, inp, targets = _fixture()
    baseline = build_recorded_geometry(model, inp, targets)
    # The output for t=0 is invariant to arbitrary future motion AND the teacher.
    inp.states[:, 1:, :, 0] += 10000.
    inp.states[:, 1:, :, 4:6] += 123.
    inp.teacher = torch.full((1, 5, 3, 4), 99999.)
    changed = build_recorded_geometry(model, inp, targets)
    # ``label_*`` rows are the recorded SUCCESSOR, exposed to label-side audits only; every
    # model-facing key (and therefore the boundary c) must be invariant to the future.
    labels = [key for key in baseline if key.startswith("label_")]
    assert labels and any(not torch.equal(baseline[key][:, 0], changed[key][:, 0])
                          for key in labels)
    for key in baseline:
        if key not in labels:
            assert torch.equal(baseline[key][:, 0], changed[key][:, 0]), key
    model_facing = lambda geo: {key: val[:, :1] for key, val in geo.items()  # noqa: E731
                                if not key.startswith("label_")}
    assert torch.equal(model.merge_pivot(model_facing(baseline)),
                       model.merge_pivot(model_facing(changed)))


def test_recorded_current_cav_actions_match_runtime_and_do_not_advance_rollout_controller():
    from wms.model import cav as cav_motion
    from wms.model.history import history_step_features, initial_history
    from wms.physics.neighbours import lead_neighbour
    from wms.train.cav_drive import TimelineControllers

    model, inp, targets = _fixture()
    ids = torch.zeros(1, 5, 3, dtype=torch.long)
    ids[:, :, 1] = 3  # direct, issued now; no future motion teacher
    inp.action_timeline = dict(controller=ids, acceleration_mps2=torch.full((1, 5, 3), -2.),
                                target_speed_mps=torch.full((1, 5, 3), 1.))
    inp.cav_controller = TimelineControllers(model, inp.action_timeline)
    rec = build_recorded_geometry(model, inp, targets)
    assert inp.cav_controller.step_index == 0
    state = state_from_kin(inp.states[:, 0], inp.present[:, 0])
    state["is_cav"] = inp.is_cav[:, 0]
    gap, _ = lead_neighbour(state["veh_kin"], state["present"], model.physics)
    hist, mask = initial_history(history_step_features(state["veh_kin"], gap),
                                 state["present"], model.history_len)
    action_id, command = cav_motion.timeline_controls(inp.action_timeline, 0, .5, 1.,
                                                      state["present"], inp.states.dtype)
    with torch.no_grad():
        out = model.step(state, history=hist, history_mask=mask,
                         action_id=action_id, action_command=command,
                         cav_controller=TimelineControllers(model, inp.action_timeline))
    moved = state["veh_kin"].clone()
    moved[..., 0], moved[..., 4], moved[..., 5] = out["next_s_m"], out["next_v_mps"], out["accel_mps2"]
    from wms.data.route_frame import route_shift
    shift = route_shift(inp.states, inp.present, getattr(inp, "continuity", None),
                        dt_s=model.physics.dt_s)[:, 0]
    expected = geometry(moved, state["present"], model.physics, route_shift=shift)
    for key in expected:
        assert torch.allclose(rec[key][:, 0], expected[key], atol=1e-6, rtol=0), key


def test_teacher_path_freezes_longitudinal_graph_and_logs_all_recorded_events():
    model, inp, targets = _fixture()
    term = MergePivotTerm(MergePivotConfig(tf_weight=.6, z_tf=395.2, tau_reg_weight=.01))
    term.prepare(targets)
    prep = targets.prepared["merge_pivot"]
    assert all(not val.requires_grad for val in prep["geometry"].values())
    # Widen the event clearances only, making this synthetic event record-setting.
    for side in ("f", "l"):
        prep["geometry"][f"gap_{side}"][:, 2, 0] = 10000.
    before = {name: p.detach().clone() for name, p in model.named_parameters()
              if not name.startswith("merge_pivot.")}
    ctx = LossContext(pred={}, current=inp.initial, targets=targets, c0=0, c1=4)
    value = term(ctx)
    assert torch.isfinite(value.value)
    value.value.backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.merge_pivot.parameters())
    for name, p in model.named_parameters():
        if name in before:
            assert p.grad is None and torch.equal(p.detach(), before[name]), name
    metrics = merge_pivot_metrics(prep, .5)
    assert metrics["merge_pivot_recorded_events"] == 1
    assert metrics["merge_pivot_tf_events"] == 1
    assert metrics["merge_pivot_tf_scored_intervals"] == 3  # stops at event, no post-event row


def test_missing_runtime_keys_fail_closed_instead_of_silently_skipping():
    model, inp, targets = _fixture()
    term = MergePivotTerm(MergePivotConfig(weight=1.))
    term.prepare(targets)
    with pytest.raises(RuntimeError, match="must emit merge_pivot_c"):
        term(LossContext(pred={}, current=inp.initial, targets=targets, c0=0, c1=4))


def _prediction(inp, model):
    states = inp.states
    c = torch.full(states[:, 1:, :, 0].shape, -torch.inf)
    c[0, :, 0] = torch.tensor([-2., -1., .5, 5.])
    return dict(positions=states[:, 1:, :, 0].clone(), speeds=states[:, 1:, :, 4].clone(),
                accel=torch.zeros_like(c), lanes=states[:, 1:, :, 1:4].argmax(-1).clone(),
                masks=inp.present[:, 1:].clone(), merge_pivot_c=c.requires_grad_(),
                merge_pivot_eligible=torch.ones_like(c, dtype=torch.bool),
                **{"merge_pivot_" + name: p for name, p in model.merge_pivot.params().items()})


def test_onpolicy_emitted_boundary_chunk_conditioning_and_absorbing_comparison():
    model, inp, targets = _fixture()
    pred = _prediction(inp, model)
    term = MergePivotTerm(MergePivotConfig(weight=.4, z_on=2.15))
    term.prepare(targets)
    total = 0.
    for c0, c1 in [(0, 1), (1, 2), (2, 4)]:
        current = state_from_kin(inp.states[:, c0], inp.present[:, c0])
        chunk = {key: (val[:, c0:c1] if val.ndim == 3 else val) for key, val in pred.items()}
        total = total + term(LossContext(pred=chunk, current=current, targets=targets,
                                         c0=c0, c1=c1)).value
    expected = -.4 / 2.15 * log_event_mass(pred["merge_pivot_c"][:, 1, 0],
                                          pred["merge_pivot_c"][:, 2, 0],
                                          pred["merge_pivot_tau"])
    assert torch.allclose(total, expected.squeeze(), atol=1e-6, rtol=0)
    metrics = merge_pivot_metrics(targets.prepared["merge_pivot"], .5)
    assert metrics["merge_pivot_on_positive_mass_events"] == 1
    assert metrics["merge_pivot_on_scored_seconds"] == 1.5
    total.backward()
    assert pred["merge_pivot_c"].grad[0, 3, 0] == 0.  # no postevent continuation

    # A single incompatible prestate censors its prefix forever, even after the
    # next chunk returns exactly to recorded geometry.
    term.prepare(targets)
    bad = state_from_kin(inp.states[:, 0].clone(), inp.present[:, 0])
    bad["s_m"] = bad["s_m"] + 10000.
    first = {key: (val[:, :1] if val.ndim == 3 else val) for key, val in pred.items()}
    term(LossContext(pred=first, current=bad, targets=targets, c0=0, c1=1))
    rest = {key: (val[:, 1:] if val.ndim == 3 else val) for key, val in pred.items()}
    later = term(LossContext(pred=rest, current=state_from_kin(inp.states[:, 1], inp.present[:, 1]),
                             targets=targets, c0=1, c1=4))
    assert later.value == 0.
    metrics = merge_pivot_metrics(targets.prepared["merge_pivot"], .5)
    assert metrics["merge_pivot_on_support_failures"] == 1
    assert metrics["merge_pivot_on_scored_intervals"] == 0


def test_zero_mass_recorded_event_is_reported_as_infinite_by_term():
    model, inp, targets = _fixture()
    term = MergePivotTerm(MergePivotConfig(tf_weight=1.))
    term.prepare(targets)
    targets.prepared["merge_pivot"]["geometry"]["eligible"][:, 2, 0] = False
    value = term(LossContext(pred={}, current=inp.initial, targets=targets, c0=0, c1=4))
    assert torch.isposinf(value.value)
    metrics = merge_pivot_metrics(targets.prepared["merge_pivot"], .5)
    assert metrics["merge_pivot_recorded_events"] == 1
    assert metrics["merge_pivot_recorded_physical_fail_events"] == 1
    assert metrics["merge_pivot_tf_zero_mass_events"] == 1


@pytest.mark.parametrize("law", ["strict", "hybrid"])
@pytest.mark.parametrize("chunks", [[(0, 4)], [(0, 1), (1, 2), (2, 4)]])
@pytest.mark.parametrize("zero_mass", [False, True])
def test_disabled_onpolicy_is_diagnostic_only_and_exactly_tf(law, chunks, zero_mass, monkeypatch):
    from wms.train.losses import merge_pivot as loss_module

    model, inp, targets = _fixture(law)
    cfg = MergePivotConfig(weight=0., tf_weight=.6, z_tf=395.2, z_on=2.15,
                           passage_law=law)
    original_score = loss_module.score_chunk
    scored = []

    def observe(*args, **kwargs):
        result = original_score(*args, **kwargs)
        scored.append((torch.is_grad_enabled(), result[0].requires_grad))
        return result

    monkeypatch.setattr(loss_module, "score_chunk", observe)

    def run(metrics):
        model.zero_grad(set_to_none=True)
        term = MergePivotTerm(replace(cfg, metrics=metrics))
        term.prepare(targets)
        prep = targets.prepared["merge_pivot"]
        # Keep TF supported while independently making the on-policy event impossible.
        for side in ("f", "l"):
            prep["geometry"][f"gap_{side}"][:, 2, 0] = 10000.
        pred = _prediction(inp, model)
        # Emit the same synthetic decision geometry so comparison retains the event.
        pred.update({"merge_pivot_geometry_" + key: prep["geometry"][key].clone()
                     for key in ("gap_f", "gap_l", "v_f", "v_l", "has_f", "has_l")})
        if zero_mass:
            pred["merge_pivot_eligible"][:, 2, 0] = False
        losses = []
        scored.clear()
        for c0, c1 in chunks:
            chunk = {key: (val[:, c0:c1] if val.ndim == 3 else val)
                     for key, val in pred.items()}
            current = state_from_kin(inp.states[:, c0], inp.present[:, c0])
            value = term(LossContext(pred=chunk, current=current, targets=targets,
                                     c0=c0, c1=c1)).value
            assert torch.isfinite(value)
            losses.append(value.detach().clone())
            value.backward()
        gradients = {name: p.grad.clone() for name, p in model.merge_pivot.named_parameters()}
        assert pred["merge_pivot_c"].grad is None
        return losses, gradients, merge_pivot_metrics(prep, .5), list(scored)

    expected, expected_grad, _, tf_scored = run(False)
    actual, actual_grad, metrics, diagnostic_scored = run(True)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert all(torch.equal(actual_grad[k], v) for k, v in expected_grad.items())
    assert all(torch.isfinite(g).all() for g in actual_grad.values())
    assert tf_scored == [(True, True)] * len(chunks)
    assert diagnostic_scored == [(True, True), (False, False)] * len(chunks)
    assert metrics["merge_pivot_on_objective_disabled"] == 1  # not the number of chunks
    assert metrics["merge_pivot_on_diagnostic_events"] == 1
    assert metrics["merge_pivot_on_diagnostic_zero_mass_events"] == int(zero_mass)
    assert metrics["merge_pivot_on_diagnostic_scored_seconds"] == 1.5
    assert "merge_pivot_on_nll" not in metrics
    if zero_mass:
        assert metrics["merge_pivot_on_diagnostic_nll"] == float("inf")


@pytest.mark.parametrize("law", ["strict", "hybrid"])
def test_positive_onpolicy_keeps_original_loss_and_gradients_bit_exact(law):
    model, inp, targets = _fixture(law)
    cfg = MergePivotConfig(weight=.4, tf_weight=.6, z_tf=395.2, z_on=2.15,
                           passage_law=law)
    term = MergePivotTerm(cfg)
    term.prepare(targets)
    prep = targets.prepared["merge_pivot"]
    for side in ("f", "l"):
        prep["geometry"][f"gap_{side}"][:, 2, 0] = 10000.
    pred = _prediction(inp, model)
    ctx = LossContext(pred=pred, current=inp.initial, targets=targets, c0=0, c1=4)
    pred.update({"merge_pivot_geometry_" + key: prep["geometry"][key].clone()
                 for key in ("gap_f", "gap_l", "v_f", "v_l", "has_f", "has_l")})
    actual = term(ctx).value
    # Original positive-weight arithmetic and score inputs on the existing fixture.
    from wms.train.losses.merge_pivot import pre_window_maximum, pre_window_hybrid_history
    params = model.merge_pivot.params()
    tau, width = params["tau"], params.get("w")
    like = prep["risk"][:, 0].to(tau)
    initial = pre_window_maximum(model, prep["pre_window_history"], like) if width is None else None
    history = pre_window_hybrid_history(model, prep["pre_window_history"], like) if width is not None else None
    kwargs = dict(known_entry=prep["known"], reset=prep["reset"], initial_maximum=initial,
                  w=width, quadrature_order=cfg.quadrature_order, initial_history=history)
    tf, _, _ = score_chunk(model.merge_pivot(prep["geometry"]), tau, prep["risk"], prep["event"],
                           eligible=prep["geometry"]["eligible"], **kwargs)
    support, early = term._comparison(ctx, prep)
    kwargs["w"] = pred["merge_pivot_w"].mean() if width is not None else None
    on, _, _ = score_chunk(pred["merge_pivot_c"], pred["merge_pivot_tau"].mean(),
                           prep["risk"], prep["event"], eligible=pred["merge_pivot_eligible"],
                           support=support, stop=early, **kwargs)
    expected = tau * 0.0 + cfg.tf_weight * tf / max(prep["n_cars"], 1) / cfg.z_tf
    expected = expected + cfg.weight * on / max(prep["n_cars"], 1) / cfg.z_on
    variables = (*model.merge_pivot.parameters(), pred["merge_pivot_c"])
    actual_grad = torch.autograd.grad(actual, variables, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, variables)
    assert torch.equal(actual, expected)
    assert all(torch.equal(a, b) for a, b in zip(actual_grad, expected_grad))
    assert actual_grad[-1].abs().sum() > 0
    assert merge_pivot_metrics(prep, .5)["merge_pivot_on_objective_disabled"] == 0


def test_unroll_runner_prepares_new_term_and_backpropagates_nonzero_chunk_offsets():
    from wms.train.unroll import UnrollRunner

    model, inp, _ = _fixture()
    batch = dict(states=inp.states, present=inp.present, is_cav=torch.zeros_like(inp.is_cav))
    settings = TrainSettings(rollout_mode="hybrid", lane_model="hazard_v5",
                             merge_decision="first_passage", merge_core_tau_reg_weight=.1,
                             truncation_steps=1)
    runner = UnrollRunner(model, settings, build_loss(settings, physics=model.physics),
                           projection=None, device=torch.device("cpu"))
    result = runner.run(batch, unroll_steps=4, backward=True)
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert result["merge_pivot"] == pytest.approx(.1 * model.merge_pivot.params()["tau"].square().item())
    assert model.merge_pivot.tau_raw.grad is not None
    assert model.merge_pivot.tau_raw.grad > 0


def test_recorded_chunk_gradients_match_exact_whole_sequence_not_detached_posterior():
    model, inp, targets = _fixture()
    term = MergePivotTerm(MergePivotConfig(tf_weight=1.))

    def prepare():
        term.prepare(targets)
        for side in ("f", "l"):
            targets.prepared["merge_pivot"]["geometry"][f"gap_{side}"][:, 2, 0] = 10000.

    prepare()
    full = term(LossContext(pred={}, current=inp.initial, targets=targets, c0=0, c1=4)).value
    full.backward()
    expected = {name: p.grad.clone() for name, p in model.merge_pivot.named_parameters()}
    model.zero_grad(set_to_none=True)
    prepare()
    chunked = 0.
    for c0, c1 in [(0, 1), (1, 2), (2, 4)]:
        loss = term(LossContext(pred={}, current=inp.initial, targets=targets, c0=c0, c1=c1)).value
        chunked += float(loss.detach())
        loss.backward()
    assert chunked == pytest.approx(float(full.detach()), abs=1e-6)
    for name, p in model.merge_pivot.named_parameters():
        assert torch.allclose(p.grad, expected[name], atol=1e-6, rtol=0), name


@pytest.mark.parametrize("split", [1, 2, 3])
@pytest.mark.parametrize("event", [False, True])
def test_hybrid_score_chunks_telescope_with_nonrecord_events(split, event):
    from wms.model.merge_pivot import hybrid_log_masses

    c = torch.tensor([[[-1.], [-.3], [-.7], [-.8]]], dtype=torch.float64)
    ones = torch.ones_like(c, dtype=torch.bool)
    events = torch.zeros_like(ones)
    events[:, -1] = event
    tau, w = c.new_tensor(.7), c.new_tensor(.4)
    def score(a, b, carry=None):
        return score_chunk(c[:, a:b], tau, ones[:, a:b], events[:, a:b],
                           known_entry=ones[:, a:b], reset=~ones[:, a:b],
                           w=w, carry=carry, quadrature_order=64)
    full, _, stats = score(0, 4)
    first, carry, _ = score(0, split)
    last, _, _ = score(split, 4, carry)
    le, ls = hybrid_log_masses(c.transpose(1, 2), tau, w, 64)
    expected = -le[..., -1] if event else -ls[..., -1]
    assert float(full) == pytest.approx(float(expected), abs=1e-6)
    assert float(first + last) == pytest.approx(float(full), abs=1e-6)
    assert stats["zero_mass_events"] == 0


def test_hybrid_recorded_chunk_gradients_all_eight_and_penalty_wiring():
    model, inp, targets = _fixture("hybrid")
    cfg = MergePivotConfig(tf_weight=.6, z_tf=395.2, passage_law="hybrid",
                           quadrature_order=64, tau_reg_weight=.02, w_reg_weight=.03)
    term = MergePivotTerm(cfg)
    def prepare():
        term.prepare(targets)
        geo = targets.prepared["merge_pivot"]["geometry"]
        geo["v"][..., 0] = 8.
        geo["v_f"][..., 0], geo["v_l"][..., 0] = 12., 4.
        geo["has_f"][..., 0] = geo["has_l"][..., 0] = True
        geo["gap_f"][0, :, 0] = torch.tensor([15., 1000., 20., 20.])
        geo["gap_l"][0, :, 0] = torch.tensor([1000., 15., 1000., 1000.])
    prepare()
    full = term(LossContext(pred={}, current=inp.initial, targets=targets, c0=0, c1=4)).value
    full.backward()
    expected = {name: p.grad.clone() for name, p in model.merge_pivot.named_parameters()}
    assert len(expected) == 8
    assert all(torch.isfinite(g) and g.abs() > 0 for g in expected.values())
    assert all(p.grad is None for name, p in model.named_parameters()
               if not name.startswith("merge_pivot."))
    model.zero_grad(set_to_none=True)
    prepare()
    value = 0.
    for c0, c1 in [(0, 1), (1, 2), (2, 4)]:
        chunk = term(LossContext(pred={}, current=inp.initial, targets=targets, c0=c0, c1=c1)).value
        value += float(chunk.detach())
        chunk.backward()
    assert value == pytest.approx(float(full.detach()), abs=1e-6)
    for name, p in model.merge_pivot.named_parameters():
        assert torch.allclose(p.grad, expected[name], atol=1e-6, rtol=0), name
    params = model.merge_pivot.params()
    regularizer = MergePivotTerm(MergePivotConfig(w_reg_weight=.03, tau_reg_weight=.02))
    regularizer.prepare(targets)
    loss = regularizer(LossContext(pred={}, current=inp.initial, targets=targets, c0=0, c1=4)).value
    assert torch.allclose(loss, .03 * params["w"].square() + .02 * params["tau"].square())


def test_hybrid_config_penalties_and_default_omission():
    cfg = TrainConfig.from_dict(dict(
        model=dict(rollout_mode="hybrid", lane_model="hazard_v5", merge_decision="first_passage",
                   merge_passage_law="hybrid", merge_w_init=.3, merge_quadrature_order=64),
        loss=dict(merge_core=dict(weight=.4, tf_weight=.6, w_reg_weight=.07))))
    settings = TrainSettings.from_config(cfg)
    term = next(t for t in build_loss(settings).chunk_terms if t.name == "merge_pivot")
    assert term.cfg.passage_law == "hybrid" and term.cfg.quadrature_order == 64
    assert term.cfg.w_reg_weight == .07 and settings.merge_w_init == .3
    defaults = TrainConfig().to_legacy_flat()
    assert not {"merge_passage_law", "merge_w_init", "merge_quadrature_order", "merge_core_w_reg_weight"} & defaults.keys()


def test_hybrid_initial_history_conditions_on_survival_without_a_prior_reset():
    from wms.model.merge_pivot import hybrid_log_masses

    prefix = torch.tensor([[[-.8, -.4]]], dtype=torch.float64)
    c = torch.tensor([[[-.7], [.1]]], dtype=torch.float64)
    ones = torch.ones_like(c, dtype=torch.bool)
    event = torch.tensor([[[False], [True]]])
    tau, w = c.new_tensor(.7), c.new_tensor(.35)
    nll, carry, stats = score_chunk(c, tau, ones, event, known_entry=~ones, reset=~ones,
                                    w=w, initial_history=prefix, quadrature_order=64)
    le, ls = hybrid_log_masses(torch.cat((prefix, c.transpose(1, 2)), -1), tau, w, 64)
    assert nll == pytest.approx(float(-le[..., -1] + ls[..., 1]), abs=1e-6)
    assert stats["unknown_age"] == 0 and stats["positive_mass_events"] == 1
    assert carry["history"].shape == (1, 1, 4)
    with pytest.raises(ValueError, match="causal history"):
        score_chunk(c, tau, ones, event, known_entry=ones, reset=~ones,
                    w=w, initial_maximum=prefix[..., -1])
