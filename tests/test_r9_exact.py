"""Exact checkpoint/real-runtime contracts, including the legacy seq-TF discrepancy."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import r9_exact as exact
from wms.model.merge_frailty import gauss_hermite
from wms.model.controllers import DIRECT_ID
from wms.physics.config import PhysicsConfig
from wms.train.losses.merge_core_seqtf import build_recorded_scene, seq_tf_chunk


@pytest.fixture(scope="module")
def frozen():
    return exact.load_frozen()


def episode(n=7):
    kin = torch.zeros(n + 1, 3, 7)
    kin[..., 0] = torch.tensor([30.0, 44.0, 12.0])
    kin[..., 0] += torch.arange(n + 1).view(-1, 1) * .8
    kin[:, 0, 3], kin[:, 1:, 2] = 1, 1
    kin[..., 4] = torch.tensor([1.6, 1.3, 2.0])
    kin[..., 5] = torch.tensor([.1, -.1, .2])
    kin[..., 6] = torch.tensor([5.0, 4.6, 5.4])
    present = torch.ones(n + 1, 3, dtype=torch.bool)
    cav = torch.zeros_like(present)
    ids = torch.arange(3).expand(n + 1, 3).clone()
    return dict(kin=kin, present=present, is_cav=cav, identity=ids)


def scene(n=7):
    ep = episode(n)
    # Only the ramp car is supervised; all target-lane neighbors are still present.
    cav = ep["is_cav"].clone()
    cav[:, 1:] = True
    targets = SimpleNamespace(states=ep["kin"].unsqueeze(0), n_steps=n,
                              loss_valid=ep["present"].unsqueeze(0),
                              is_cav=cav.unsqueeze(0), dt_s=.5)
    return build_recorded_scene(targets, merge_lane=2, zone_s0_m=0,
                                 zone_s1_m=75.44, k=3)


def clone_scene(value):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in value.items()
            if k != "_post_identity"}


def test_real_checkpoint_load_is_strict_frozen_and_preserves_rng(frozen, tmp_path):
    before = torch.random.get_rng_state()
    second = exact.load_frozen()
    assert torch.equal(before, torch.random.get_rng_state())
    assert frozen.metadata["stage_s"] == 300
    assert frozen.metadata["sha256"] == "ae094e592027eaa5f2fe4eb7e404b7d9215092a62c23fdb10c1cdbfab8639ed9"
    assert len(frozen.model.state_dict()) == 149
    assert all(torch.equal(v, second.model.state_dict()[k])
               for k, v in frozen.model.state_dict().items())
    assert all(not p.requires_grad and p.device.type == "cpu" for p in frozen.model.parameters())
    assert not frozen.model.training
    assert torch.count_nonzero(frozen.core.alpha_net[-1].weight) > 0
    assert frozen.core.alpha_bias != 0
    # Real residual is zero-effect; a later test explicitly exercises a nonzero residual.
    assert torch.count_nonzero(frozen.core.residual_head[-1].weight) == 0
    assert frozen.core.params()["s0"].item() == pytest.approx(2.371485710144043)
    payload = torch.load(exact.DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
    del payload["model"]["merge_core.alpha_net.2.weight"]
    path = tmp_path / "missing_head.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        exact.load_frozen(path)


@pytest.mark.parametrize("path", ["../outside.pt", "data/sealed/file.pt", "data/holdout/a.pt",
                                      "data/heldout/a.pt", "data/cal_split/a.pt", "data/chunk5/a.pt",
                                      "data/j4b/a.pt", "data/inc1/a.pt", "data/hetdata_truth/a.pt",
                                      "data/world964000/a.pt", "data/world964199/a.pt"])
def test_guard_refuses_before_open(path):
    with pytest.raises(ValueError):
        exact.load_frozen(path)


@pytest.mark.parametrize("nonzero_residual", [False, True])
@pytest.mark.parametrize("n_quad", [1, 8, 16])
def test_fitter_masses_equal_ACTUAL_seq_tf_chunk_real_checkpoint(frozen, nonzero_residual, n_quad):
    """Do not substitute a copied formula for the runtime side of this assertion.

    All real learned alpha weights remain nonzero. Even explicitly nonzero
    residual/location heads do not enter the existing seq-TF implementation.
    Its omission is tested, not silently 'repaired' in the original comparator.
    """
    local = copy.deepcopy(frozen)
    if nonzero_residual:
        with torch.no_grad():
            local.core.residual_head[-1].weight.fill_(.09)
            local.core.residual_head[-1].bias.fill_(.15)
    sc = scene()
    # Exercise the runtime's actual per-car sigma shape too.
    sigma = local.core.alpha_scale().expand(1, 1, 3).clone()
    prepared = exact.seq_tf_logits(sc, local.core, n_quad=n_quad, sigma=sigma)
    law = exact.hazard_distribution(prepared["logits"][0, 0], prepared["weights"],
                                     sc["feasible"][0, :, 0])
    p = local.core.params()
    for event in range(sc["n_steps"]):
        event_scene = clone_scene(sc)
        event_scene["rec_event"][0, 0] = event
        event_scene["last_obs"][0, 0] = event - 1
        nll, _, _, metrics = seq_tf_chunk(
            event_scene, 0, sc["n_steps"], **p, sigma=sigma,
            n_quad=n_quad, log_post=None)
        torch.testing.assert_close(law["log_mass"][event], -nll, rtol=1e-5, atol=1e-5)
        assert metrics["events"] == 1
    nll, _, post, _ = exact.seq_tf_score(sc, local, sigma=sigma, n_quad=n_quad)
    torch.testing.assert_close(law["log_survival"][-1], -nll, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(law["log_post"], post[0, :, 0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(law["log_mass"].exp().sum() + law["log_survival"][-1].exp(),
                               torch.tensor(1.0))


def test_seqtf_chunk_posterior_and_left_truncation_match_runtime(frozen):
    sc = scene()
    prepared = exact.seq_tf_logits(sc, frozen.core)
    logits, weights = prepared["logits"][0, 0], prepared["weights"]
    eligible = sc["feasible"][0, :, 0]
    posterior, history = exact.condition_on_prefix(logits, weights, eligible, 3)
    suffix = exact.hazard_distribution(logits[:, 3:], weights, eligible[3:], log_post=posterior)
    whole = exact.hazard_distribution(logits, weights, eligible)
    torch.testing.assert_close(suffix["log_mass"], whole["log_mass"][3:] - history,
                               atol=1e-5, rtol=1e-5)
    first, _, runtime_post, _ = exact.seq_tf_score(sc, frozen, 0, 3)
    second, _, _, _ = exact.seq_tf_score(sc, frozen, 3, 7, log_post=runtime_post)
    torch.testing.assert_close(posterior, runtime_post[0, :, 0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(first + second, -whole["log_survival"][-1])
    torch.testing.assert_close(second, -suffix["log_survival"][-1])


def test_unsupported_seqtf_event_omission_is_visible_not_invented_support(frozen):
    sc = scene()
    sc["rec_event"][0, 0], sc["last_obs"][0, 0] = 3, 2
    sc["feasible"][0, 3, 0] = False
    prepared = exact.seq_tf_logits(sc, frozen.core)
    law = exact.hazard_distribution(prepared["logits"][0, 0], prepared["weights"],
                                     sc["feasible"][0, :, 0])
    nll, _, _, metrics = exact.seq_tf_score(sc, frozen)
    assert metrics["unsupported_events"] == 1
    assert law["log_mass"][3] == -torch.inf
    # Original runtime does NOT score the unsupported event factor.
    torch.testing.assert_close(nll, -law["log_survival"][2])


@pytest.mark.parametrize("nonzero_residual", [False, True])
def test_full_learned_head_quadrature_matches_deployment_apply_core(frozen, nonzero_residual):
    local = copy.deepcopy(frozen)
    if nonzero_residual:
        with torch.no_grad():
            local.core.residual_head[-1].weight.fill_(.07)
            local.core.residual_head[-1].bias.fill_(.1)
    ep = episode(4)
    # The current encoding changes, but alpha uses only the first ramp encoding.
    cache = exact.precompute_episode(local, **ep, corrected=False)
    for row in range(4):
        state = exact._state(ep["kin"][row:row+1], ep["present"][row:row+1],
                              ep["is_cav"][row:row+1])
        embedding = cache["embedding"][:, row].unsqueeze(0)
        # A varied encoding also makes accidental per-step alpha redraw detectable.
        embedding = embedding + .2 * row
        cache["embedding"][:, row] = embedding[0]
    values = exact.head_logits(local.core, cache["original_runtime"]["features"],
                               cache["embedding"], birth_embedding=cache["birth_embedding"],
                               intent=cache["intent"])
    assert bool((values["location"][:, 1:] != values["birth_location"][:, 1:]).any())
    if nonzero_residual:
        assert values["residual"].abs().max() > 0
    nodes, _ = gauss_hermite(8)
    for row in range(4):
        state = exact._state(ep["kin"][row:row+1], ep["present"][row:row+1],
                              ep["is_cav"][row:row+1])
        core_out = local.model._merge_core_forward(
            state["veh_kin"], state["present"], cache["embedding"][:, row].unsqueeze(0),
            (1, 3), cache["intent"][:, row].unsqueeze(0))
        out = {"merge_core_" + key: value for key, value in core_out.items() if key != "params"}
        out.update({"merge_core_" + key: value for key, value in core_out["params"].items()})
        out["lane_v5_acts"] = torch.ones_like(state["present"])
        birth_loc = local.core.alpha_location(cache["birth_embedding"][:, row].unsqueeze(0), (1, 3))
        for q, z in enumerate(nodes):
            alpha = torch.exp((birth_loc + local.core.alpha_scale() * z).clamp(-8, 8))
            carried = {**state, "merge_core_alpha": alpha, "merge_core_has_alpha": state["present"],
                       "merge_core_present": state["present"]}
            runtime = dict(out)
            local.model._apply_merge_core(carried, state["veh_kin"], state["present"],
                                          runtime, state["lane"].clone())
            torch.testing.assert_close(runtime["merge_core_probability"][0],
                                       values["logits"][:, q, row].sigmoid(), atol=1e-6, rtol=1e-6)


def test_causal_motion_is_actual_WMS_one_step_and_preserves_lanes(frozen):
    local = copy.deepcopy(frozen)
    local.metadata["physics_confirmed"] = True  # comparing the IDENTICAL supplied config
    ep = episode(2)
    state = exact._state(ep["kin"][:1], ep["present"][:1], ep["is_cav"][:1])
    history = torch.zeros(1, 3, local.model.driver_history_len, 9)
    mask = torch.zeros(1, 3, local.model.driver_history_len, dtype=torch.bool)
    kwargs = dict(driver_history=history, driver_history_mask=mask, k_steps=5)
    moved, out = exact.longitudinal_proposal(local, state, forward_kwargs=kwargs)
    with torch.random.fork_rng(devices=[]):
        runtime = local.model.step(state, **kwargs)
    torch.testing.assert_close(moved[..., 0], runtime["next_s_m"])
    torch.testing.assert_close(moved[..., 4], runtime["next_v_mps"])
    torch.testing.assert_close(moved[..., 5], runtime["accel_mps2"])
    assert torch.equal(moved[..., 1:4], state["veh_kin"][..., 1:4])
    # Once an out is supplied, there is no second longitudinal forward evaluation.
    local.model.forward = lambda *a, **kw: pytest.fail("second forward")
    repeated, _ = exact.longitudinal_proposal(local, state, out=out)
    torch.testing.assert_close(repeated, moved)


def test_missing_physics_is_explicit_and_never_silently_claimed_exact(frozen):
    assert frozen.metadata["physics_confirmed"] is False
    ep = episode(1)
    state = exact._state(ep["kin"][:1], ep["present"][:1], ep["is_cav"][:1])
    with pytest.raises(ValueError, match="authoritative run physics"):
        exact.longitudinal_proposal(frozen, state)
    loaded = exact.load_frozen(physics=PhysicsConfig(**frozen.metadata["physics"]))
    assert loaded.metadata["physics_confirmed"]
    assert loaded.metadata["physics_source"] == "explicit"


def test_current_direct_CAV_command_matches_actual_step(frozen):
    local = copy.deepcopy(frozen)
    local.metadata["physics_confirmed"] = True
    ep = episode(2)
    ep["is_cav"][:, 1] = True
    state = exact._state(ep["kin"][:1], ep["present"][:1], ep["is_cav"][:1])
    kwargs = dict(driver_history=torch.zeros(1, 3, local.model.driver_history_len, 9),
                  driver_history_mask=torch.zeros(1, 3, local.model.driver_history_len,
                                                  dtype=torch.bool), k_steps=3)
    action_id = torch.full((1, 3), DIRECT_ID)
    command = torch.tensor([[[0., 20.], [-.3, 20.], [0., 20.]]])
    moved, _ = exact.longitudinal_proposal(local, state, forward_kwargs=kwargs,
                                          action_id=action_id, action_command=command)
    with torch.random.fork_rng(devices=[]):
        actual = local.model.step(state, **kwargs, action_id=action_id, action_command=command)
    torch.testing.assert_close(moved[..., 0], actual["next_s_m"])
    torch.testing.assert_close(moved[..., 4], actual["next_v_mps"])
    torch.testing.assert_close(moved[..., 5], actual["accel_mps2"])
    with pytest.raises(ValueError, match="controller callback"):
        exact.longitudinal_proposal(local, state, forward_kwargs=kwargs,
                                    action_id=torch.zeros_like(action_id), action_command=command)


def test_corrected_full_scene_veto_touching_bumpers_and_reservations(frozen):
    ep = episode(1)
    kin = ep["kin"][:1].clone()
    # Exact contact: leader rear == ego front, no physical overlap.
    kin[0, 1, 6] = 5  # exactly representable endpoint contact in float32
    kin[0, 1, 0] = kin[0, 0, 0] + kin[0, 1, 6]
    ep["is_cav"][:, 1] = True  # CAV is still a physical neighbor.
    geo = exact._corrected_features(kin, ep["present"][:1], ep["is_cav"][:1], frozen.model.physics)
    assert geo["eligible"][0, 0]
    assert geo["gap_l"][0, 0] == 0
    kin[0, 1, 0] -= .01
    geo = exact._corrected_features(kin, ep["present"][:1], ep["is_cav"][:1], frozen.model.physics)
    assert geo["overlap"][0, 0] and not geo["eligible"][0, 0]
    # Simultaneous ramp requests reserve intervals independently of willingness.
    kin[0, 1, 1:4] = torch.tensor([0., 0., 1.])
    kin[0, 1, 0] = kin[0, 0, 0] + 1
    ep["is_cav"][:, 1] = False
    geo = exact._corrected_features(kin, ep["present"][:1], ep["is_cav"][:1], frozen.model.physics)
    assert geo["eligible"][0, 0]
    assert not geo["eligible"][0, 1] and geo["reservation_veto"][0, 1]


def test_episode_replay_is_causal_identity_bound_and_stops_at_event(frozen):
    ep = episode(6)
    ep["kin"][4:, 0, 3] = 0
    ep["kin"][4:, 0, 2] = 1
    cache = exact.precompute_episode(frozen, **ep, allow_unverified_physics=True)
    future = {key: value.clone() for key, value in ep.items()}
    future["kin"][3:, :, 0] += 200
    other = exact.precompute_episode(frozen, **future, allow_unverified_physics=True)
    for mode in ("original_runtime", "corrected"):
        torch.testing.assert_close(cache[mode]["logits"][..., :3], other[mode]["logits"][..., :3])
    cars = exact.episode_sequences(cache)
    assert len(cars) == 1
    assert cars[0]["event_row"] == 4
    assert cars[0]["original_runtime"]["logits"].shape == (8, 4)
    assert cars[0]["unknown_history"]
    # Instant replacement without absence: original runtime keeps alpha, corrected policy resets.
    changed = episode(6)
    changed["identity"][3:, 0] = 88
    changed["kin"][3:, 0, 4] += 1
    legacy = exact.precompute_episode(frozen, **changed, corrected=False)
    corrected = exact.precompute_episode(frozen, **changed, corrected=False, identity_policy="identity")
    torch.testing.assert_close(legacy["birth_embedding"][0, 2], legacy["birth_embedding"][0, 3])
    assert not torch.equal(corrected["birth_embedding"][0, 2], corrected["birth_embedding"][0, 3])
    assert len(exact.episode_sequences(corrected)) == 2


def test_matched_refit_preserves_full_architecture_and_has_meaningful_gradients(frozen):
    cache = exact.precompute_episode(frozen, **episode(4), corrected=False)
    core = exact.matched_refit_core(frozen)
    assert set(core.state_dict()) == set(frozen.core.state_dict())
    for key, value in core.state_dict().items():
        torch.testing.assert_close(value, frozen.core.state_dict()[key], rtol=0, atol=0)
    logits = exact.head_logits(core, cache["original_runtime"]["features"], cache["embedding"],
                               birth_embedding=cache["birth_embedding"], intent=cache["intent"])
    law = exact.hazard_distribution(logits["logits"][:1], logits["weights"],
                                     cache["original_runtime"]["eligible"][:1])
    loss = -law["log_mass"][0, -1]
    loss.backward()
    for name in ("s0_raw", "t_raw", "b_raw", "tau_raw", "temperature_raw", "alpha_bias",
                 "alpha_log_sigma_raw"):
        grad = getattr(core, name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    assert core.alpha_net[-1].weight.grad.abs().sum() > 0
    assert core.residual_head[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in frozen.model.parameters())
    assert not core.yield_scale_raw.requires_grad
    # Refit state is exactly reloadable into the same architecture.
    clone = exact.matched_refit_core(frozen)
    optimizer = torch.optim.Adam([p for p in core.parameters() if p.requires_grad], lr=.001)
    optimizer.step()
    clone.load_state_dict(core.state_dict(), strict=True)
    again = exact.head_logits(clone, cache["original_runtime"]["features"], cache["embedding"],
                              birth_embedding=cache["birth_embedding"], intent=cache["intent"])
    changed = exact.head_logits(core, cache["original_runtime"]["features"], cache["embedding"],
                                birth_embedding=cache["birth_embedding"], intent=cache["intent"])
    torch.testing.assert_close(again["logits"], changed["logits"], rtol=0, atol=0)


def test_exact_atoms_vetoes_extremes_and_zero_conditioning_failure():
    logits = torch.tensor([[-1000., 1000., -1000.], [1000., -1000., 1000.]], dtype=torch.float64)
    weights = torch.tensor([.4, .6], dtype=torch.float64)
    law = exact.hazard_distribution(logits, weights)
    assert torch.isfinite(law["log_mass"]).all()
    torch.testing.assert_close(law["log_mass"].exp().sum() + law["log_survival"][-1].exp(),
                               torch.tensor(1., dtype=torch.float64))
    atoms = torch.tensor([[-torch.inf, torch.inf, 1.]])
    eligible = torch.tensor([False, True, True])
    law = exact.hazard_distribution(atoms, torch.ones(1), eligible)
    assert torch.equal(law["log_mass"], torch.tensor([-torch.inf, 0., -torch.inf]))
    with pytest.raises(ValueError, match="zero history conditioning"):
        exact.condition_on_prefix(atoms, torch.ones(1), eligible, 2)


def test_real_TRAIN_episode_precompute_smoke(frozen):
    path = exact.guard("data/r9_train/v5_native/episodes/w963004m1900p40t300__native")
    loaded = exact.load_episode(path)
    # Whole scene around a REAL observed merge; no neighbor selection/search radius.
    lane = loaded["kin"][..., 1:4].argmax(-1)
    events = ((lane[:-1] == 2) & (lane[1:] == 1) & loaded["present"][:-1].astype(bool)
              & loaded["present"][1:].astype(bool) & ~loaded["is_cav"][:-1].astype(bool)
              & (loaded["identity"][:-1] == loaded["identity"][1:])
              & (loaded["kin"][:-1, :, 0] >= 0) & (loaded["kin"][:-1, :, 0] <= 75.44))
    event = int(np.flatnonzero(events.any(-1))[0])
    lo, hi = max(0, event - 7), event + 2
    ep = {key: loaded[key][lo:hi] for key in ("kin", "present", "is_cav", "identity")}
    cache = exact.precompute_episode(frozen, **ep, corrected=True, allow_unverified_physics=True)
    assert cache["proposals"].shape == ep["kin"][:-1].shape
    assert cache["embedding"].shape[-1] == 102
    assert torch.isfinite(cache["original_runtime"]["logits"]).all()
    assert torch.isfinite(cache["corrected"]["logits"]).all()
    assert np.array_equal(cache["proposals"][..., 1:4].numpy(), ep["kin"][:-1, :, 1:4])
    # Actual runtime scene and NLL on the SAME real data and intact checkpoint.
    targets = SimpleNamespace(states=cache["kin"].unsqueeze(0), n_steps=hi-lo-1,
                              loss_valid=cache["present"].unsqueeze(0),
                              is_cav=cache["is_cav"].unsqueeze(0), dt_s=.5)
    sc = build_recorded_scene(targets, merge_lane=2, zone_s0_m=0, zone_s1_m=75.44,
                              k=cache["kin"].shape[1])
    actual_logits = exact.seq_tf_logits(sc, frozen.core)
    torch.testing.assert_close(cache["seq_tf"]["logits"], actual_logits["logits"][0])
    actual, _, _, metrics = exact.seq_tf_score(sc, frozen)
    assert metrics["events"] >= 1
    total = actual.new_zeros(())
    for slot in torch.where(sc["identity"][0] >= 0)[0]:
        event = int(sc["rec_event"][0, slot])
        stop = event if event >= 0 else int(sc["last_obs"][0, slot])
        eligible = (sc["feasible"][0, :stop+1, slot] & sc["supervised"][0, :stop+1, slot]
                    & (sc["run_start"][0, :stop+1, slot] == sc["identity"][0, slot]))
        law = exact.hazard_distribution(cache["seq_tf"]["logits"][slot, :, :stop+1],
                                         cache["weights"], eligible)
        total -= (law["log_mass"][-1] if event >= 0 and eligible[-1]
                  else law["log_survival"][-1])
    torch.testing.assert_close(total, actual, rtol=1e-5, atol=1e-5)
