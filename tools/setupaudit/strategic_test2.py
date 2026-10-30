"""Load isolated repair modules for CPU checks without touching the active scorer."""
import importlib.util
from pathlib import Path
import sys


def install():
    for name, path in [('wms.control.bridge', 'strategic_bridge.py'),
                       ('wms.control.policy14', 'strategic_policy14.py')]:
        path = Path(__file__).with_name(path)
        assert path.is_file()
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)


if __name__ == '__main__':
    install()
    import pytest
    raise SystemExit(pytest.main(sys.argv[1:]))
