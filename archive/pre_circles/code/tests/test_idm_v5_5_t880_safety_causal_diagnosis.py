from types import SimpleNamespace

from experiments.idm.run_idm_v5_5_t880_safety_causal_diagnosis import (
    _ZeroEffectWall,
    _physical_state,
    _schedule,
)


def test_zero_effect_wall_keeps_actuator_interface_without_commands():
    wall = _ZeroEffectWall(d_lo=200.0, d_hi=400.0)
    assert wall.act({"cav": object()}) == {}
    assert wall.act_with_targets({"cav": object()}, {"cav": 10.0}) == {}
    assert wall.restore_tau(["cav"]) == []
    assert wall.set_tau_blend(["cav"], 0.5) == []


def test_diagnosis_schedules_are_frozen_at_one_or_twenty_steps():
    schedule = _schedule(10, "sham")
    assert [schedule(step) for step in (10, 11, 30, 31)] == [0, 1, 1, 0]
    assembly = _schedule(10, "assembly_only")
    assert [assembly(step) for step in (10, 11, 12)] == [0, 1, 0]


def test_physical_identity_excludes_branch_metadata_fields():
    left = SimpleNamespace(
        relative_step=1,
        sim_time=2.0,
        action=0,
        supervisor_observation=(0.1,),
        vehicles=(("v", "edge", "lane", 1.0),),
        active_cavs=("v",),
        collision_vehicle_ids=(),
        starting_teleport_vehicle_ids=(),
    )
    right = SimpleNamespace(**{**left.__dict__, "action": 1, "supervisor_observation": (0.2,)})
    assert _physical_state(left) == _physical_state(right)

