"""Install the pinned additive lane-speed helper without editing a live runtime."""
import importlib.util
from pathlib import Path


def install(path, net_path=None):
    from traffic_shepherd.c4 import geometry
    path = Path(path)
    name = 'traffic_shepherd.c4._coopmwm_nullfix_geometry'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Existing geometry stays authoritative; only the new lookup is installed.
    for field in ('LANE_TABLE', 'HERE_OFFSETS'):
        assert getattr(geometry, field) == getattr(module, field), field
    if net_path is None:
        geometry.lane_allowed_speeds_mps = module.lane_allowed_speeds_mps
    else:
        net_path = Path(net_path)
        assert net_path.is_file()
        table = module.lane_allowed_speeds_mps(net_path)
        assert set(table) == {v[0] for v in geometry.LANE_TABLE.values()}
        # Bind only this lookup's artifact; do not change process-wide data roots.
        geometry.lane_allowed_speeds_mps = lambda path=None: module.lane_allowed_speeds_mps(
            net_path if path is None else path)
    geometry.NET_RELPATH = module.NET_RELPATH
    geometry._LANE_SPEEDS_CACHE = module._LANE_SPEEDS_CACHE
    return module
