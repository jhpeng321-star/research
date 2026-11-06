"""S4 regressions for the shared P1--P3 primitives (no simulation/data inputs)."""
import io
import math
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from wms.model.merge_core import load_init
from wms.model.merge_pivot import (
    FirstPassageNet, advance_longitudinal, geometry, hard_st, log_event_mass,
    log_survival, resolve_conflicts, update_state,
)


PHYSICS = SimpleNamespace(merge_lane=2, merge_target_lane=1,
                          merge_zone_s0_m=0.0, merge_zone_s1_m=75.44)
DTYPE = torch.float64


def scene(*cars):
    """Cars are (front, lane, speed, length), with no IDs/truth/future fields."""
    kin = torch.zeros(1, len(cars), 7, dtype=DTYPE)
    for i, (s, lane, v, length) in enumerate(cars):
        kin[0, i, 0] = s
        kin[0, i, 1 + lane] = 1.0
        kin[0, i, 4] = v
        kin[0, i, 6] = length
    return kin, torch.ones(1, len(cars), dtype=torch.bool)


def inputs(**overrides):
    g = {"gap_l": torch.tensor([20.0], dtype=DTYPE),
         "gap_f": torch.tensor([15.0], dtype=DTYPE),
         "v": torch.tensor([8.0], dtype=DTYPE),
         "v_l": torch.tensor([6.0], dtype=DTYPE),
         "v_f": torch.tensor([12.0], dtype=DTYPE),
         "has_l": torch.tensor([True]), "has_f": torch.tensor([True]),
         "eligible": torch.tensor([True])}
    g.update(overrides)
    return g


def tick(c, *, carry=None, ramp=True, present=True, identity=1, u=0.5, tau=1.0):
    return update_state(torch.tensor([c], dtype=DTYPE), torch.tensor(tau, dtype=DTYPE),
                        torch.tensor([ramp]), torch.tensor([present]), carry=carry,
                        identity=torch.tensor([identity]), uniform=torch.tensor([u], dtype=DTYPE))


def assert_finite_grad(*tensors):
    for tensor in tensors:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_initialization_variance_match_and_seven_parameters_only():
    net, old = FirstPassageNet(), load_init()
    p = net.params()
    assert set(p) == {"s_f", "T_f", "b_f", "s_l", "T_l", "b_l", "tau"}
    assert len(list(net.parameters())) == 7
    for side in ("f", "l"):
        for name, legacy in (("s", "s0_m"), ("T", "T_s"), ("b", "b_mps2")):
            assert p[f"{name}_{side}"].item() == pytest.approx(old[legacy], rel=1e-6)
    assert (p["tau"].square() * math.pi**2 / 3).item() == pytest.approx(
        old["alpha_log_sigma"]**2, rel=1e-6)
    with torch.no_grad():
        net.s_f_raw.fill_(-40)
    assert 0 < net.params()["s_f"].item() < 1e-15  # no epsilon-positive floor


@pytest.mark.parametrize("name", ["s_f", "T_l", "tau"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_bad_parameter_initialization(name, value):
    with pytest.raises(ValueError, match=name):
        FirstPassageNet({name: value})


def test_two_sided_boundary_formula_and_hard_assertiveness_equivalence():
    net = FirstPassageNet({"s_f": 1, "T_f": 2, "b_f": 4,
                           "s_l": 3, "T_l": 0.5, "b_l": 2}).double()
    g = inputs()
    c, p = net.boundary(g), net.params()
    df = p["s_f"] + p["T_f"] * 12 + (12**2 - 8**2) / (2 * p["b_f"])
    dl = p["s_l"] + p["T_l"] * 8 + (8**2 - 6**2) / (2 * p["b_l"])
    torch.testing.assert_close(c, torch.minimum((g["gap_f"] / df).log(), (g["gap_l"] / dl).log()))
    z = torch.linspace(-50, 50, 101, dtype=DTYPE)
    expected = (g["gap_f"] >= df / (p["tau"] * z).exp()) & (
        g["gap_l"] >= dl / (p["tau"] * z).exp())
    for width in (1e-4, 0.1, 1e4):
        actual = hard_st(c.expand_as(z), p["tau"], z, width, torch.ones_like(z, dtype=torch.bool))
        assert torch.equal(actual.bool(), expected)


@pytest.mark.parametrize("gap", [0.0, -1.0, math.nan, math.inf])
def test_invalid_constrained_gap_has_exact_zero_support_and_safe_gradients(gap):
    net = FirstPassageNet().double()
    g = inputs(gap_f=torch.tensor([gap], dtype=DTYPE, requires_grad=True))
    c = net.boundary(g)
    assert torch.isneginf(c).all()
    hard_st(c, net.params()["tau"], torch.zeros_like(c), 0.1, g["eligible"]).sum().backward()
    assert_finite_grad(g["gap_f"], *[p for n, p in net.named_parameters() if n != "tau_raw"])
    assert g["gap_f"].grad.item() == 0.0


def test_absent_sides_are_exactly_free_even_with_nan_padding():
    net = FirstPassageNet().double()
    g = inputs(has_f=torch.tensor([False]), has_l=torch.tensor([False]),
               gap_f=torch.tensor([math.nan], dtype=DTYPE),
               gap_l=torch.tensor([math.nan], dtype=DTYPE),
               v_f=torch.tensor([math.nan], dtype=DTYPE),
               v_l=torch.tensor([math.nan], dtype=DTYPE))
    assert torch.isposinf(net(g)).all()
    g["eligible"] = torch.tensor([False])
    assert torch.isneginf(net(g)).all()


def test_learned_boundary_and_tau_receive_nonzero_exact_likelihood_gradients():
    net = FirstPassageNet().double()
    # Each side is separately active and has an active closing-speed term.
    g = {"gap_l": torch.tensor([1000.0, 8.0], dtype=DTYPE),
         "gap_f": torch.tensor([12.0, 1000.0], dtype=DTYPE),
         "v": torch.tensor([6.0, 12.0], dtype=DTYPE),
         "v_l": torch.tensor([6.0, 4.0], dtype=DTYPE),
         "v_f": torch.tensor([15.0, 4.0], dtype=DTYPE),
         **{key: torch.ones(2, dtype=torch.bool) for key in ("has_l", "has_f", "eligible")}}
    c = net(g)
    loss = -log_event_mass(torch.full_like(c, -torch.inf), c, net.params()["tau"]).sum()
    loss.backward()
    assert_finite_grad(*net.parameters())
    assert all(p.grad.abs().item() > 0.0 for p in net.parameters())


def test_geometry_front_bumper_lengths_and_unlimited_neighbor_range():
    kin, present = scene((20, 2, 8, 8), (-300, 1, 13, 2), (400, 1, 7, 12))
    g = geometry(kin, present, PHYSICS)
    assert g["eligible"][0, 0]
    assert g["has_f"][0, 0] and g["has_l"][0, 0]
    assert g["gap_f"][0, 0].item() == 312  # NOT 318 (follower length)
    assert g["gap_l"][0, 0].item() == 368
    assert g["v_f"][0, 0].item() == 13
    assert g["v_l"][0, 0].item() == 7


def test_long_non_nearest_target_vehicle_can_overlap_ego():
    kin, present = scene((20, 2, 0, 5), (26, 1, 0, 2), (40, 1, 0, 30))
    g = geometry(kin, present, PHYSICS)
    assert g["gap_l"][0, 0].item() == 4  # nearest neighbor itself does not intersect
    assert g["overlap"][0, 0]
    assert not g["eligible"][0, 0]
    assert torch.isneginf(FirstPassageNet()(g)[0, 0])


@pytest.mark.parametrize("front,length", [(15, 3), (25, 5)])
def test_touching_allowed_but_zero_learned_gap_blocks(front, length):
    kin, present = scene((20, 2, 1, 5), (front, 1, 1, length))
    g = geometry(kin, present, PHYSICS)
    assert not g["overlap"][0, 0]
    assert g["eligible"][0, 0]
    assert torch.isneginf(FirstPassageNet()(g)[0, 0])


@pytest.mark.parametrize("front,length", [(20, 5), (19.999, 1), (20.001, 1)])
def test_side_by_side_and_small_exact_intersections_veto(front, length):
    kin, present = scene((20, 2, 0, 5), (front, 1, 0, length))
    g = geometry(kin, present, PHYSICS)
    assert g["overlap"][0, 0] and not g["eligible"][0, 0]


@pytest.mark.parametrize("position,expected", [(-0.001, False), (0, True),
                                              (75.44, True), (75.441, False)])
def test_inclusive_risk_zone_and_eligibility(position, expected):
    kin, present = scene((position, 2, 0, 5))
    g = geometry(kin, present, PHYSICS)
    assert g["ramp"].item() == expected
    assert g["eligible"].item() == expected
    assert torch.isposinf(FirstPassageNet()(g)).item() == expected


def test_nonfinite_present_scene_not_treated_as_empty_road_and_absent_padding_ignored():
    kin, present = scene((20, 2, 1, 5), (40, 1, 1, 5), (0, 0, 1, 5))
    kin[0, 2, 4] = math.nan
    assert not geometry(kin, present, PHYSICS)["eligible"][0, 0]
    present[0, 2] = False
    kin[0, 2] = math.nan
    g = geometry(kin.requires_grad_(), present, PHYSICS)
    assert g["eligible"][0, 0]
    net = FirstPassageNet().double()
    loss = -log_survival(net(g)[0, 0], net.params()["tau"])
    loss.backward()
    assert_finite_grad(kin)
    assert torch.equal(kin.grad[0, 2], torch.zeros(7, dtype=DTYPE))


def test_temporary_invalid_ego_speed_veto_does_not_end_risk_lifecycle():
    kin, present = scene((20, 2, 1, 5), (25.1, 1, 1, 5))
    net = FirstPassageNet().double()
    g = geometry(kin, present, PHYSICS)
    first = update_state(net(g), net.params()["tau"], g["ramp"], present,
                         uniform=torch.full_like(kin[..., 0], 0.5))
    kin[0, 0, 4] = math.nan
    invalid = geometry(kin, present, PHYSICS)
    assert invalid["ramp"][0, 0] and not invalid["eligible"][0, 0]
    second = update_state(net(invalid), net.params()["tau"], invalid["ramp"], present, carry=first)
    assert not second["terminal"][0, 0]
    kin[0, 0, 4] = 1
    kin[0, 1, 0] = 100
    restored = geometry(kin, present, PHYSICS)
    third = update_state(net(restored), net.params()["tau"], restored["ramp"], present, carry=second)
    assert third["cross"][0, 0]


def test_batched_and_empty_geometry():
    kin, present = scene((20, 2, 4, 5), (40, 1, 6, 5))
    many = geometry(kin.expand(2, 3, 2, 7), present.expand(2, 3, 2), PHYSICS)
    single = geometry(kin, present, PHYSICS)
    for key in single:
        torch.testing.assert_close(many[key], single[key].expand(2, 3, 2))
    empty = geometry(kin[:, :0], present[:, :0], PHYSICS)
    assert all(t.shape == (1, 0) for t in empty.values())


def test_exact_endpoint_masses_and_zero_mass_failures():
    prev = torch.tensor([-math.inf, -math.inf, 0, math.inf, -math.inf, 1, 2], dtype=DTYPE,
                        requires_grad=True)
    cur = torch.tensor([-math.inf, 0, math.inf, math.inf, math.inf, 1, 1], dtype=DTYPE,
                       requires_grad=True)
    tau = torch.tensor(0.7, dtype=DTYPE, requires_grad=True)
    mass = log_event_mass(prev, cur, tau)
    expected = torch.tensor([-math.inf, -math.log(2), -math.log(2), -math.inf,
                             0, -math.inf, -math.inf], dtype=DTYPE)
    torch.testing.assert_close(mass, expected)
    # Unsupported events stay infinite loss, but do not contaminate other gradients.
    assert torch.isposinf(-mass[0])
    torch.where(torch.isfinite(mass), mass, torch.zeros_like(mass)).sum().backward()
    assert_finite_grad(prev, cur, tau)
    x = torch.tensor([-math.inf, 0, math.inf], dtype=DTYPE, requires_grad=True)
    survival = log_survival(x, tau)
    torch.testing.assert_close(survival, torch.tensor([0, -math.log(2), -math.inf], dtype=DTYPE))
    torch.where(torch.isfinite(survival), survival, torch.zeros_like(survival)).sum().backward()
    assert_finite_grad(x, tau)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_extreme_log_masses_do_not_saturate_or_clip_and_gradients_are_finite(dtype):
    a = torch.tensor([-10000.0, 9999.0, -10000.0, 0.0], dtype=dtype, requires_grad=True)
    b = torch.tensor([-9999.0, 10000.0, 10000.0, 1e-10], dtype=dtype, requires_grad=True)
    tau = torch.tensor(0.05, dtype=dtype, requires_grad=True)
    mass = log_event_mass(a, b, tau)
    assert torch.isfinite(mass).all()
    assert mass[0] < -100000 and mass[1] < -100000
    assert mass[3] < -20  # very small positive increments keep their mass
    (-mass.sum() - log_survival(b, tau).sum()).backward()
    assert_finite_grad(a, b, tau)


def test_log_mass_gradcheck_including_very_small_increment():
    a = torch.tensor([-25.0, 20.0, 0.1], dtype=DTYPE, requires_grad=True)
    b = torch.tensor([-24.0, 21.0, 0.10001], dtype=DTYPE, requires_grad=True)
    tau = torch.tensor(0.7, dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(log_event_mass, (a, b, tau), eps=1e-8, atol=1e-4)


@pytest.mark.parametrize("dtype,value,scale", [(torch.float32, 1e30, 1e-10),
                                              (torch.float64, 1e300, 1e-10)])
def test_overflowed_scaled_logits_have_exact_zero_saturated_adjoints(dtype, value, scale):
    a = torch.tensor([-value], dtype=dtype, requires_grad=True)
    b = torch.tensor([value], dtype=dtype, requires_grad=True)
    tau = torch.tensor(scale, dtype=dtype, requires_grad=True)
    mass = log_event_mass(a, b, tau)
    survival = log_survival(a, tau)
    assert mass.item() == survival.item() == 0.0
    (mass + survival).sum().backward()
    assert_finite_grad(a, b, tau)
    assert tau.grad.item() == 0.0
    width = torch.tensor(scale, dtype=dtype, requires_grad=True)
    z = torch.ones_like(b, requires_grad=True)
    hard_st(b, tau, z, width, torch.tensor([True])).sum().backward()
    assert_finite_grad(b, z, tau, width)
    assert width.grad.item() == 0.0


@pytest.mark.parametrize("dtype,delta", [(torch.float32, 1e-30), (torch.float64, 1e-300)])
def test_tiny_positive_event_mass_and_gradients_survive_scaled_gap_underflow(dtype, delta):
    a = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    b = torch.tensor(delta, dtype=dtype, requires_grad=True)
    tau = torch.tensor(1e20, dtype=dtype, requires_grad=True)
    score = log_event_mass(a, b, tau)
    expected = math.log(delta) - math.log(1e20) - math.log(4)
    assert score.item() == pytest.approx(expected, rel=1e-7)
    score.backward()
    assert_finite_grad(a, b, tau)
    assert (b.grad * b.detach()).item() == pytest.approx(1.0, rel=1e-6)
    assert (tau.grad * tau.detach()).item() == pytest.approx(-1.0, rel=1e-6)


def test_mass_conservation_running_max_chunk_posterior_and_competing_exit():
    c = torch.tensor([-math.inf, -3, -4, -1, -1, 0.2, -2, 2], dtype=DTYPE)
    maxima = c.cummax(0).values
    previous = torch.cat((torch.full_like(maxima[:1], -torch.inf), maxima[:-1]))
    tau = torch.tensor(0.8, dtype=DTYPE)
    logp = log_event_mass(previous, maxima, tau)
    survival = log_survival(maxima, tau)
    torch.testing.assert_close(logp.exp().sum() + survival[-1].exp(), torch.tensor(1., dtype=DTYPE))
    assert torch.isneginf(logp[[0, 2, 4, 6]]).all()
    # Nonzero chunk offset: condition on survival through the previous chunk.
    start = 3
    conditional_mass = (logp[start:] - survival[start - 1]).exp()
    conditional_exit = (survival[-1] - survival[start - 1]).exp()
    torch.testing.assert_close(conditional_mass.sum() + conditional_exit, torch.tensor(1., dtype=DTYPE))
    # A censor/competing exit consumes the remaining mass at its time, not a
    # made-up merge or a second survival factor after exit.
    exit_index = 5
    torch.testing.assert_close(logp[:exit_index + 1].exp().sum() + survival[exit_index].exp(),
                               torch.tensor(1., dtype=DTYPE))
    torch.testing.assert_close((-survival).exp().reciprocal(), torch.sigmoid(-maxima / tau))


def test_scoring_cdf_and_event_parity_to_direct_formula():
    maxima = torch.linspace(-5, 5, 101, dtype=DTYPE)
    previous = torch.cat((torch.full_like(maxima[:1], -torch.inf), maxima[:-1]))
    tau = torch.tensor(0.73, dtype=DTYPE)
    cdf = torch.sigmoid(maxima / tau)
    direct = cdf - torch.cat((torch.zeros_like(cdf[:1]), cdf[:-1]))
    torch.testing.assert_close(log_event_mass(previous, maxima, tau).exp(), direct, atol=1e-6, rtol=0)
    torch.testing.assert_close(-torch.expm1(log_survival(maxima, tau)), cdf, atol=1e-6, rtol=0)
    torch.testing.assert_close(-log_event_mass(previous, maxima, tau), -direct.log(), atol=1e-6, rtol=0)


def test_state_persistent_chunk_carry_exact_first_crossing_and_no_postevent_survival():
    scores = [-math.inf, -3.0, -4.0, -0.1, 0.0, 20.0]
    carry = None
    for i, c in enumerate(scores):
        previous = carry
        carry = tick(c, carry=carry, ramp=(i < 5), u=0.5 if i == 0 else 0.999)
        assert carry["z"].item() == 0.0
        assert carry["cross"].item() == (i == 4)
        if i == 3:  # explicitly detach a nonzero-offset chunk boundary
            carry = {key: value.detach() for key, value in carry.items()}
        if i == 5:
            assert carry["terminal"].item()
            assert carry["maximum"].item() == previous["maximum"].item() == 0.0


def test_identity_presence_reset_and_neighbor_change_does_not_reset():
    first = tick(-10, u=0.8)
    switched_neighbor = tick(-20, carry=first, u=0.1)
    torch.testing.assert_close(first["z"], switched_neighbor["z"])
    torch.testing.assert_close(first["maximum"], switched_neighbor["maximum"])
    new_identity = tick(-20, carry=switched_neighbor, identity=2, u=0.1)
    assert new_identity["z"].item() != first["z"].item()
    assert new_identity["maximum"].item() == -20
    gone = tick(10, carry=new_identity, present=False, identity=2)
    assert not gone["started"].item() and not gone["terminal"].item()
    assert torch.isneginf(gone["maximum"]).item()
    back = tick(-30, carry=gone, identity=2, u=0.9)
    assert back["maximum"].item() == -30
    assert back["z"].item() == pytest.approx(math.log(9))


def test_leaving_risk_is_absorbing_but_physical_veto_is_not():
    before = tick(-math.inf, ramp=False)
    assert not before["started"].item() and not before["terminal"].item()
    started = tick(-2, carry=before)
    veto = tick(-math.inf, carry=started)
    assert not veto["terminal"].item()
    assert veto["maximum"].item() == -2
    exited = tick(-math.inf, carry=veto, ramp=False)
    assert exited["terminal"].item()
    reentry = tick(math.inf, carry=exited)
    assert not reentry["cross"].item()
    assert reentry["maximum"].item() == -2
    replacement = tick(math.inf, carry=reentry, identity=2)
    assert replacement["cross"].item()


def test_latent_tails_are_not_clipped_and_randomness_is_drawn_only_when_needed(monkeypatch):
    c = torch.full((3,), -torch.inf, dtype=DTYPE)
    ramp = torch.tensor([True, False, True])
    draws = []

    def rand(n, **kwargs):
        draws.append(n)
        return torch.full((n,), 0.75, **kwargs)

    monkeypatch.setattr(torch, "rand", rand)
    state = update_state(c, torch.tensor(1.), ramp, torch.ones(3, dtype=torch.bool))
    assert draws == [2]
    state = update_state(c, torch.tensor(1.), ramp, torch.ones(3, dtype=torch.bool), carry=state)
    assert draws == [2]
    low, high = tick(-math.inf, u=1e-30), tick(-math.inf, u=1 - 1e-14)
    assert low["z"].item() < -60 and high["z"].item() > 30
    assert not low["cross"].item() and not high["cross"].item()


@pytest.mark.parametrize("u", [0, 1, math.nan, -0.1])
def test_supplied_uniform_endpoints_fail_instead_of_behavioral_clipping(u):
    with pytest.raises(ValueError, match="strictly between"):
        tick(-1, u=u)


def test_fixed_latent_exact_runtime_first_passage_and_analytic_support():
    c = torch.tensor([-3.0, -4.0, -1.0, -2.0, 0.5, 0.0, 3.0], dtype=DTYPE)
    u = torch.linspace(0.01, 0.99, 99, dtype=DTYPE)
    z = u.log() - torch.log1p(-u)
    tau = torch.tensor(0.7, dtype=DTYPE)
    carry, already = None, torch.zeros_like(u, dtype=torch.bool)
    maxima = c.cummax(0).values
    for i in range(len(c)):
        carry = update_state(c[i].expand_as(u), tau, ~already, torch.ones_like(already),
                             carry=carry, uniform=u)
        crossing = (c[i] + tau * z >= 0) & ~already
        assert torch.equal(carry["cross"], crossing)
        if i > 0 and maxima[i] == maxima[i - 1]:
            assert not crossing.any()
            assert torch.isneginf(log_event_mass(maxima[i - 1], maxima[i], tau))
        already |= crossing
        assert torch.equal(already, maxima[i] + tau * z >= 0)


def test_st_exact_forward_limits_and_sigmoid_backward():
    c = torch.tensor([-math.inf, -1e6, -0.1, 0, 0.1, 1e6, math.inf, math.nan], dtype=DTYPE,
                     requires_grad=True)
    z = torch.zeros_like(c, requires_grad=True)
    tau = torch.tensor(0.3, dtype=DTYPE, requires_grad=True)
    eligible = torch.ones_like(c, dtype=torch.bool)
    expected = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0], dtype=DTYPE)
    for width in (1e-6, 0.05, 1e6):
        assert torch.equal(hard_st(c, tau, z, width, eligible), expected)
    out = hard_st(c, tau, z, 0.05, eligible)
    out.sum().backward()
    assert_finite_grad(c, z, tau)
    probability = torch.sigmoid(c.detach()[2:5] / 0.05)
    torch.testing.assert_close(c.grad[2:5], probability * (1 - probability) / 0.05)
    assert c.grad[[0, 6, 7]].abs().sum() == 0
    assert torch.equal(hard_st(c, tau, z, 0.1, ~eligible), torch.zeros_like(c))


@pytest.mark.parametrize("width", [0, -1, math.nan, math.inf])
def test_invalid_st_width(width):
    with pytest.raises(ValueError, match="width"):
        hard_st(torch.zeros(1), torch.ones(1), torch.zeros(1), width, torch.tensor([True]))


@pytest.mark.parametrize("stopping", [False, True])
def test_longitudinal_matches_actual_wms_step_once_without_future_input(stopping, monkeypatch):
    from wms.model import ModelSpec, WmSModel
    from wms.physics.config import PhysicsConfig

    kin, present = scene((10, 0, 1, 5), (40, 1, 4, 5), (60, 1, 0, 5))
    kin = kin.float()
    accel = torch.tensor([[-8., 2., 0.]])
    current_lane = kin[..., 1:4].argmax(-1)
    physics = PhysicsConfig(stop_time_integration=stopping)
    model = WmSModel(ModelSpec(), physics=physics)
    count = []

    def forward(*args, **kwargs):
        count.append(1)
        return {"accel_mps2": accel,
                "lane_logits": F.one_hot(current_lane, 3).float(), "hdv_mask": present}

    monkeypatch.setattr(model, "forward", forward)
    out = model.step({"s_m": kin[..., 0], "v_mps": kin[..., 4], "lane": current_lane,
                      "length_m": kin[..., 6], "present": present, "a_mps2": kin[..., 5]})
    advanced = advance_longitudinal(kin, accel, physics.dt_s, stopping)
    assert count == [1]
    torch.testing.assert_close(advanced[..., 0], out["next_s_m"], rtol=0, atol=0)
    torch.testing.assert_close(advanced[..., 4], out["next_v_mps"], rtol=0, atol=0)
    torch.testing.assert_close(advanced[..., 1:4], kin[..., 1:4], rtol=0, atol=0)
    torch.testing.assert_close(advanced[..., 5], accel)
    assert advanced[0, 0, 0].item() == pytest.approx(10.0625 if stopping else 10.25)


def test_decision_geometry_uses_single_causal_advance():
    kin, present = scene((-1, 2, 4, 5), (20, 1, 6, 5))
    original = kin.clone()
    advanced = advance_longitudinal(kin, torch.zeros_like(kin[..., 4]), 0.5)
    assert not geometry(kin, present, PHYSICS)["eligible"][0, 0]
    g = geometry(advanced, present, PHYSICS)
    assert g["eligible"][0, 0]
    assert advanced[0, 0, 0].item() == 1
    assert g["gap_l"][0, 0].item() == 17
    torch.testing.assert_close(kin, original)


def conflicts(cars, current, candidates, learned=None, present=None):
    kin, seen = scene(*cars)
    current = torch.tensor([current])
    candidates = torch.tensor([candidates])
    learned = torch.ones_like(seen) if learned is None else torch.tensor([learned])
    seen = seen if present is None else torch.tensor([present])
    return resolve_conflicts(kin, seen, current, candidates, learned)[0].tolist()


def test_conflict_stationary_occupancy_wins_even_with_higher_slot():
    assert conflicts([(20, 2, 0, 5), (20, 1, 0, 5)], [2, 1], [1, 1]) == [2, 1]


def test_conflict_simultaneous_moves_stable_slot_priority():
    assert conflicts([(20, 2, 0, 5), (20, 0, 0, 5)], [2, 0], [1, 1]) == [1, 0]


def test_conflict_canceled_crossing_can_retry_without_redrawing_or_terminal_lockout():
    kin, present = scene((20, 2, 0, 5), (20, 0, 0, 5))
    ramp = torch.tensor([[True, False]])
    c = torch.tensor([[1.0, -math.inf]], dtype=DTYPE)
    current = torch.tensor([[2, 0]])
    first = update_state(c, torch.tensor(1.), ramp, present, uniform=torch.full_like(c, 0.5))
    # The nonlearned move has fixed final occupancy and cancels this HDV proposal.
    final = resolve_conflicts(kin, present, current, torch.tensor([[1, 1]]),
                              torch.tensor([[True, False]]))
    assert first["cross"][0, 0] and final[0, 0] == 2
    retry = update_state(c, torch.tensor(1.), ramp, present, carry=first,
                          uniform=torch.full_like(c, 0.01))
    assert retry["cross"][0, 0] and not retry["terminal"][0, 0]
    torch.testing.assert_close(retry["z"], first["z"])
    committed = update_state(c, torch.tensor(1.), torch.zeros_like(ramp), present, carry=retry)
    assert committed["terminal"][0, 0] and not committed["cross"][0, 0]


def test_conflict_nonlearned_moves_immutable_and_beat_hdv():
    assert conflicts([(20, 2, 0, 5), (20, 0, 0, 5)], [2, 0], [1, 1], [True, False]) == [2, 1]


@pytest.mark.parametrize("separation,expected", [(5.0, [1, 1]), (5.001, [1, 1]),
                                               (4.999, [2, 1])])
def test_conflict_no_padding_or_approximate_overlap(separation, expected):
    assert conflicts([(20, 2, 0, 5), (20 + separation, 1, 0, 5)], [2, 1], [1, 1]) == expected


def test_conflict_cancellation_cascade_rechecks_restored_origin():
    # B cannot enter C's occupied lane; B's restoration then blocks A.
    assert conflicts([(20, 2, 0, 5), (20, 1, 0, 5), (20, 0, 0, 5)],
                     [2, 1, 0], [1, 0, 0]) == [2, 1, 0]


def test_conflict_simultaneous_chain_keeps_nonconflicting_later_mover():
    assert conflicts([(20, 2, 0, 5), (24, 0, 0, 5), (28, 2, 0, 5)],
                     [2, 0, 2], [1, 1, 1]) == [1, 0, 1]


def test_conflict_swap_allowed_when_final_intervals_do_not_conflict():
    assert conflicts([(20, 2, 0, 5), (20, 1, 0, 5)], [2, 1], [1, 2]) == [1, 2]


def test_conflict_absent_slots_ignored_and_existing_fixed_conflicts_preserved():
    assert conflicts([(20, 2, 0, 5), (20, 1, 0, 5)], [2, 1], [1, 1],
                     present=[True, False]) == [1, 1]
    assert conflicts([(20, 1, 0, 5), (20, 1, 0, 5)], [1, 1], [1, 1]) == [1, 1]


def test_checkpoint_in_memory_roundtrip_preserves_scale_cdf_mass_and_decisions():
    net = FirstPassageNet().double()
    with torch.no_grad():
        net.tau_raw.add_(0.3)
        net.T_f_raw.add_(0.2)
    saved = io.BytesIO()
    torch.save(net.state_dict(), saved)
    saved.seek(0)
    restored = FirstPassageNet().double()
    restored.load_state_dict(torch.load(saved, weights_only=True))
    p, q = net.params(), restored.params()
    torch.testing.assert_close(p["tau"], q["tau"], rtol=0, atol=0)
    c, d = net(inputs()), restored(inputs())
    torch.testing.assert_close(c, d, rtol=0, atol=0)
    for score in (log_survival,):
        torch.testing.assert_close(score(c, p["tau"]), score(d, q["tau"]), rtol=0, atol=0)
    torch.testing.assert_close(log_event_mass(c - 1, c, p["tau"]),
                               log_event_mass(d - 1, d, q["tau"]), rtol=0, atol=0)
    z = torch.linspace(-5, 5, 21, dtype=DTYPE)
    assert torch.equal(hard_st(c, p["tau"], z, 0.1, torch.tensor([True])),
                       hard_st(d, q["tau"], z, 1.0, torch.tensor([True])))
