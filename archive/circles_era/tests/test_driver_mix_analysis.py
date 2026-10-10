"""Keep easier background traffic distinct from improved incremental control."""
import importlib.util
from pathlib import Path

PATH = Path(__file__).parents[1] / 'output/driver_mix_v1/analyze.py'
spec = importlib.util.spec_from_file_location('oe43_analysis', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_absolute_wall_improvement_can_hide_worse_incremental_effect():
    old_damping, old_wall = {'service': 200.0}, {'service': 190.0}
    new_damping, new_wall = {'service': 150.0}, {'service': 160.0}
    assert module.subtract(new_wall, old_wall)['service'] == -30
    assert module.subtract(module.subtract(new_wall, new_damping),
                           module.subtract(old_wall, old_damping))['service'] == 20


def test_effects_with_different_physical_dimensions_cannot_be_subtracted():
    import pytest
    with pytest.raises(ValueError):
        module.subtract({'service': 1}, {'braking': 1})
