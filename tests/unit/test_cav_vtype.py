"""Every CAV uses the normal_car vType; HDV type draws are unchanged."""
from traffic_shepherd.scenarios import request_schedule


def test_cavs_are_normal_car_and_hdvs_unchanged():
    legacy = request_schedule(4107, 2400, penetration=0.4)
    uniform = request_schedule(4107, 2400, penetration=0.4, cav_vtype="normal_car")
    assert [r.vehicle_id for r in legacy] == [r.vehicle_id for r in uniform]
    assert any(r.is_cav and r.driver_type != "normal_car" for r in legacy)
    for old, new in zip(legacy, uniform):
        assert old.is_cav == new.is_cav and old.depart_s == new.depart_s
        assert new.driver_type == ("normal_car" if new.is_cav else old.driver_type)


def test_env_default_is_normal_car():
    from traffic_shepherd.c4.env import CAV_VTYPE
    assert CAV_VTYPE == "normal_car"
