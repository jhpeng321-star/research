"""Planned-command intent observation contracts, CPU only (numpy; torch tests isolated).

C4 intent-obs part: buffer + features for the ``v2v_intent`` option (DEFAULT OFF).
Covered here: shape/order, one-step V2V latency (t-1, never t), stale/absent
masking, episode reset, despawn drop-out, per-step coverage telemetry, the
option-off byte-identical contract against the plain v2v block, and the
runner/runtime/live call-site gates.
"""
import numpy as np
import pytest

from .v2v_obs import (V2V_DIM, V2V_INTENT_DIM, V2V_INTENT_H, V2VConfig, V2VIntentBuffer,
                      intent_block_width, intent_coverage, intent_coverage_stats,
                      is_intent_block_width, publish_intents, v2v_features,
                      v2v_intent_features, v2v_intent_slot_block, v2v_slot_block)


def state():
    """One road: ego at s=100; p0 (+30, CAV), p1 (-20, CAV), hdv (+10, not CAV), far (+800, CAV)."""
    return {
        "ego": {"s_m": 100.0, "lane": 1, "v_mps": 10.0, "a_mps2": 0.0, "is_cav": True},
        "p0": {"s_m": 130.0, "lane": 1, "v_mps": 12.0, "a_mps2": 0.5, "is_cav": True},
        "p1": {"s_m": 80.0, "lane": 0, "v_mps": 9.0, "a_mps2": -0.5, "is_cav": True},
        "hdv": {"s_m": 110.0, "lane": 1, "v_mps": 10.0, "a_mps2": 0.0, "is_cav": False},
        "far": {"s_m": 900.0, "lane": 1, "v_mps": 11.0, "a_mps2": 0.0, "is_cav": True},
    }


PLAN_A = [0.1, 0.2, 0.3, 0.4, 0.5]
PLAN_B = [-1.0, -0.8, -0.6, -0.4, -0.2]
ON = V2VConfig(v2v=True, v2v_intent=True)


def intent_part(block, k=4, stride=6 + V2V_INTENT_H + 1):
    """(H planned commands, plain 6-wide prefix) of a block, partner-major."""
    arr = np.asarray(block, dtype=np.float32).reshape(-1, k, stride)
    return arr[..., 6:6 + V2V_INTENT_H], arr[..., :6]


def valid_bits(block, k=4, stride=6 + V2V_INTENT_H + 1):
    """intent_valid bit per partner slot."""
    return np.asarray(block, dtype=np.float32).reshape(-1, k, stride)[..., stride - 1]


def test_dimensions_and_option_defaults():
    assert V2V_INTENT_H == 5
    assert V2V_INTENT_DIM == 4 * (6 + 5 + 1) == 48
    assert V2V_DIM == 24
    cfg = V2VConfig()
    assert cfg.v2v is False and cfg.v2v_intent is False and cfg.v2v_intent_horizon == 5
    assert intent_block_width() == V2V_DIM
    assert intent_block_width(V2VConfig(v2v=True)) == V2V_DIM
    assert intent_block_width(ON) == V2V_INTENT_DIM
    assert intent_block_width(V2VConfig(v2v=True, v2v_intent=True, v2v_intent_horizon=3)) == 4 * 10
    with pytest.raises(ValueError, match='horizon'):
        V2VConfig(v2v_intent=True, v2v_intent_horizon=0)
    assert is_intent_block_width(48) and is_intent_block_width(32)
    assert not is_intent_block_width(24) and not is_intent_block_width(33) and not is_intent_block_width(34)


def test_option_off_is_byte_identical_even_with_fresh_broadcasts():
    buf = V2VIntentBuffer()
    publish_intents(buf, 300, {"p0": PLAN_A, "p1": PLAN_B})
    st = state()
    cfg_off = V2VConfig(v2v=True)  # intent option off
    assert np.array_equal(v2v_intent_features(st, "ego", {}, buf, 301, cfg_off),
                          v2v_features(st, "ego", {}, cfg_off))
    assert np.array_equal(v2v_intent_features(st, "ego", {}, buf, 301, V2VConfig()),
                          v2v_features(st, "ego", {}, V2VConfig()))  # both off: zeros
    slots = dict(s_m=[100., 130., 80.], lane=[1, 1, 0], v_mps=[10., 12., 9.],
                 a_mps2=[0., .5, -.5], present=[True, True, True], is_cav=[True, True, True],
                 slot_ids=["ego", "p0", "p1"])
    assert np.array_equal(v2v_intent_slot_block(intent_buffer=buf, step_index=301,
                                                cfg=cfg_off, last_commands={}, **slots),
                          v2v_slot_block(last_commands={}, cfg=cfg_off, **slots))
    assert intent_block_width(cfg_off) == V2V_DIM


def test_intent_layout_matches_plain_prefix_then_plan_then_valid():
    buf = V2VIntentBuffer()
    publish_intents(buf, 300, {"p0": PLAN_A, "p1": PLAN_B})
    st = state()
    block = v2v_intent_features(st, "ego", {}, buf, 301, ON)
    assert block.shape == (V2V_INTENT_DIM,) and block.dtype == np.float32
    plain = v2v_features(st, "ego", {}, V2VConfig(v2v=True))
    _intent, prefix = intent_part(block)
    assert np.array_equal(prefix.reshape(-1), plain)          # identical 6-feature slots
    assert np.array_equal(_intent[0, 0, :], np.array(PLAN_B, np.float32))   # nearest: p1
    assert np.array_equal(_intent[0, 1, :], np.array(PLAN_A, np.float32))   # second: p0
    assert np.array_equal(valid_bits(block)[0], np.array([1., 1., 0., 0.], np.float32))
    assert not _intent[0, 2:, :].any()                        # zero-padded slots, valid=0
    assert block[35] == 0.0 and block[47] == 0.0


def test_latency_reads_t_minus_one_never_t_and_stale_is_masked():
    buf = V2VIntentBuffer()
    publish_intents(buf, 300, {"p0": PLAN_A, "p1": PLAN_B})
    st = state()
    same = v2v_intent_features(st, "ego", {}, buf, 300, ON)     # step t: broadcast NOT visible
    part, _ = intent_part(same)
    assert not part.any() and not valid_bits(same).any()
    fresh = v2v_intent_features(st, "ego", {}, buf, 301, ON)    # step t+1: visible
    part, _ = intent_part(fresh)
    assert np.array_equal(part[0, 1, :], np.array(PLAN_A, np.float32))
    stale = v2v_intent_features(st, "ego", {}, buf, 302, ON)    # step t+2: masked again
    part, _ = intent_part(stale)
    assert not part.any() and not valid_bits(stale).any()
    publish_intents(buf, 301, {"p0": [9.] * 5})                 # latest broadcast wins
    part, _ = intent_part(v2v_intent_features(st, "ego", {}, buf, 302, ON))
    assert np.array_equal(part[0, 1, :], np.full(5, 9., np.float32))
    assert part[0, 0, :].sum() == 0.0                           # p1 absent at 301: masked


def test_absent_stale_and_unknown_ego_masking():
    buf = V2VIntentBuffer()
    publish_intents(buf, 300, {"p0": PLAN_A})                   # p1 never broadcasts
    st = state()
    part, _ = intent_part(v2v_intent_features(st, "ego", {}, buf, 301, ON))
    bits = valid_bits(v2v_intent_features(st, "ego", {}, buf, 301, ON))[0]
    assert part[0, 0, :].sum() == 0.0 and bits[0] == 0.0          # absent partner masked
    assert part[0, 1, :].sum() != 0.0 and bits[1] == 1.0          # fresh partner unmasked
    buf.drop("p0")
    assert "p0" not in buf and buf.fresh("p0", 301) is None
    assert np.array_equal(v2v_intent_features({}, "ego", {}, buf, 301, ON),
                          np.zeros(V2V_INTENT_DIM, np.float32))  # unknown ego: zeros


def test_publish_validation_clip_and_counts():
    buf = V2VIntentBuffer()
    assert publish_intents(buf, 5, {"a": PLAN_A, "b": None, "c": [0.] * 5}) == 2
    with pytest.raises(ValueError, match='horizon'):
        publish_intents(buf, 5, {"a": [1., 2.]})
    with pytest.raises(ValueError, match='horizon'):
        V2VIntentBuffer(horizon=0)
    clipped = V2VIntentBuffer()
    publish_intents(clipped, 5, {"a": [-99.] + [99.] * 4}, clip=(-3.0, 3.0))
    assert np.array_equal(clipped.fresh("a", 6), np.array([-3., 3., 3., 3., 3.], np.float32))
    short = V2VIntentBuffer(horizon=3)
    assert publish_intents(short, 1, {"a": [1., 2., 3.]}) == 1
    assert np.array_equal(short.fresh("a", 2), np.array([1., 2., 3.], np.float32))


def test_reset_and_despawn_drop_out():
    buf = V2VIntentBuffer()
    publish_intents(buf, 300, {"p0": PLAN_A, "p1": PLAN_B, "gone": PLAN_A})
    buf.retain({"p0", "p1"})                                    # despawned CAVs drop out
    assert set(buf) == {"p0", "p1"} and len(buf) == 2
    buf.reset()                                                 # episode start
    assert len(buf) == 0 and buf.fresh("p0", 301) is None
    part, _ = intent_part(v2v_intent_features(state(), "ego", {}, buf, 301, ON))
    assert not part.any()


def test_slot_block_alignment_and_zero_rows():
    buf = V2VIntentBuffer()
    publish_intents(buf, 300, {"ego": PLAN_A, "p0": PLAN_B, "p1": PLAN_A})
    s_m = [100., 130., 80., 500.]
    lane, v_mps, a_mps2 = [1, 1, 0, 1], [10., 12., 9., 8.], [0., .5, -.5, 0.]
    present = [True, True, True, False]
    is_cav = [True, True, True, True]
    slot_ids = ["ego", "p0", "p1", None]
    last = {"ego": .2, "p0": .1, "p1": None}
    block = v2v_intent_slot_block(s_m, lane, v_mps, a_mps2, present, is_cav, slot_ids,
                                  last, buf, 301, ON)
    assert block.shape == (4, V2V_INTENT_DIM) and block.dtype == np.float32
    table = {vid: rec for vid, rec in state().items()}
    table["ego"] = dict(table["ego"], **{"s_m": 100., "lane": 1, "v_mps": 10., "a_mps2": 0.})
    for slot, vid in enumerate(slot_ids[:3]):
        want = v2v_intent_features(state(), vid, last, buf, 301, ON)
        assert np.array_equal(block[slot], want)
    assert not block[3].any()                                   # id-less slot row: zeros


def test_coverage_telemetry():
    buf = V2VIntentBuffer()
    publish_intents(buf, 300, {"p0": PLAN_A})                   # only p0 broadcasts at 300
    args = dict(s_m=[100., 90., 130.], lane=[1, 1, 1], v_mps=[10., 9., 12.],
                a_mps2=[0., 0., .5], present=[True] * 3, is_cav=[True] * 3,
                slot_ids=["ego", "p1", "p0"], last_commands={}, intent_buffer=buf,
                step_index=301, cfg=ON)
    block = v2v_intent_slot_block(**args)
    stats = intent_coverage_stats(block)
    assert stats == dict(coverage=pytest.approx(1 / 3), egos_with_partners=3, rows=3,
                         present_slots=6, fresh_slots=2)
    assert intent_coverage(block) == pytest.approx(1 / 3)
    publish_intents(buf, 300, {"p1": PLAN_B, "ego": PLAN_A})
    assert intent_coverage(v2v_intent_slot_block(**args)) == 1.0
    assert intent_coverage(np.zeros((2, V2V_INTENT_DIM), np.float32)) == 0.0  # no partners
    assert intent_coverage(np.zeros((0, V2V_INTENT_DIM), np.float32)) == 0.0  # empty step


def test_runner_prepare_option_default_off_and_dimension_gate():
    from .v2v_runner import prepare
    kw = dict(root='run/root', legacy_root='legacy/root', source_root='src',
              common_checkpoint_sha256='0' * 64, v2v_dim=V2V_DIM)
    cfg = prepare({}, **kw)
    assert 'v2v_intent' not in cfg and 'v2v_intent_horizon' not in cfg   # byte-identical binding
    intent = prepare({}, v2v_intent=True, **dict(kw, v2v_dim=V2V_INTENT_DIM))
    assert intent['v2v_intent'] is True and intent['v2v_intent_horizon'] == V2V_INTENT_H
    assert intent['v2v_dim'] == V2V_INTENT_DIM
    with pytest.raises(ValueError, match='intent observation dimension'):
        prepare({}, v2v_intent=True, **kw)


def test_live_binding_gate_default_off():
    from .v2v_live import intent_binding_gate
    intent_binding_gate(dict(v2v=True))                          # option absent: passes
    intent_binding_gate(dict(v2v=True, v2v_intent=False))        # explicit off: passes
    intent_binding_gate(dict(v2v=True, v2v_intent=True, v2v_dim=V2V_INTENT_DIM))
    with pytest.raises(ValueError, match='intent observation dimension'):
        intent_binding_gate(dict(v2v=True, v2v_intent=True, v2v_dim=V2V_DIM))


def test_runtime_policy_and_controller_intent_mode():
    torch = pytest.importorskip('torch')
    from .acceleration import Controller, Policy as BasePolicy, encode
    from .v2v_runtime import Policy, controller_class, from_weights

    def rows():
        from .test_acceleration_cpu import rows as fixture
        result = fixture()
        cavs = [v for v in result if v['is_cav']]
        cavs[1]['s_m'] = cavs[0]['s_m'] + 20.
        return result

    torch.manual_seed(3)
    base = BasePolicy()
    import copy
    from .v2v_model import extend_for_v2v
    plain_v2v = copy.deepcopy(base)
    extend_for_v2v(plain_v2v, V2V_DIM)
    wide = from_weights(plain_v2v.state_dict(), enabled=True)     # plain v2v width: unchanged
    assert wide.v2v_dim == V2V_DIM and type(wide) is Policy
    from .v2v_model import load_checkpoint as load_wide
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        from .accel_learner import save
        path = pathlib.Path(tmp) / 'u0.pt'
        opt = torch.optim.Adam(base.parameters(), lr=3e-4)
        gen = torch.Generator().manual_seed(11)
        save(path, base, opt, gen, 0, 0)
        model, model_opt, model_gen, _ = load_wide(path, config=dict(v2v=True, v2v_dim=V2V_INTENT_DIM))
        assert tuple(model.actor[0].weight.shape) == (64, 29 + V2V_INTENT_DIM + 4)  # sensor|intent|latent
        assert tuple(model.critic[0].weight.shape) == (128, 29 + V2V_INTENT_DIM + 12)
        assert torch.equal(model.actor[0].weight[:, 29:29 + V2V_INTENT_DIM],
                           torch.zeros(64, V2V_INTENT_DIM))      # zero-init extension
    sensor = rows()
    members = encode(sensor)['members']
    commands = {r['vid']: 0.1 for r in sensor if r['is_cav']}
    controller = controller_class(Controller, intent=True)(
        Policy(v2v_dim=V2V_INTENT_DIM), torch.Generator().manual_seed(4), True)
    assert controller.intent_buffer is not None
    controller.reset_intents()
    controller.apply(sensor, commands, 301)                       # no broadcasts yet
    blocks = controller.events[-1]['sample']['context']['v2v']
    assert set(blocks) == set(members) and all(len(v) == V2V_INTENT_DIM for v in blocks.values())
    assert intent_coverage(np.asarray([blocks[v] for v in members], dtype=np.float32)) == 0.0
    # Runner hook after the policy step at t=301; readers see it at t+1 only.
    assert controller.publish_planned_intents(
        301, {v: [0.1, 0.2, 0.3, 0.4, 0.5] for v in members}) == len(members)
    controller.apply(sensor, commands, 302)
    fresh = controller.events[-1]['sample']['context']['v2v']
    assert intent_coverage(np.asarray([fresh[v] for v in members], dtype=np.float32)) == 1.0
    # Despawned CAVs drop out of the broadcast table on the next step.
    sensor = [r for r in sensor if r['vid'] != members[0]]
    commands.pop(members[0], None)
    controller.publish_planned_intents(302, {v: PLAN_A for v in members})
    controller.apply(sensor, commands, 303)
    assert members[0] not in controller.intent_buffer
    controller.reset_intents()
    controller.apply(sensor, commands, 304)
    reset_blocks = controller.events[-1]['sample']['context']['v2v']
    assert intent_coverage(np.asarray([reset_blocks[v] for v in reset_blocks],
                                      dtype=np.float32)) == 0.0   # episode reset masks everything
    with pytest.raises(ValueError, match='unsupported V2V checkpoint width'):
        from_weights({'actor.0.weight': torch.zeros(64, 34), 'critic.0.weight': torch.zeros(128, 41)})
