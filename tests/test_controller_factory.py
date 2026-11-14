import pytest

from traffic_shepherd import controller_factory as factory
from traffic_shepherd.config import load_config


class Config:
    pass


@pytest.mark.parametrize("name", ["native", "damping"])
def test_default_controllers_reject_u(monkeypatch, name):
    monkeypatch.setattr(factory, "CirclesController", lambda config: object())
    with pytest.raises(ValueError):
        factory.create_controller(name, Config(), desired_speed_mps=10.)


def test_follower_stopper_is_no_longer_a_controller():
    with pytest.raises(ValueError, match="unsupported controller"):
        factory.create_controller("follower-stopper", load_config(), desired_speed_mps=10.)
