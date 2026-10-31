"""Load the fleet-owned package under an isolated, stable import name."""
import importlib.util
import re
import sys
from pathlib import Path


FORBIDDEN = re.compile(
    r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|"
    r"vehicle_params|driver_params|vtype|\.rou\.xml", re.I)


def load(root):
    root = Path(root)
    if FORBIDDEN.search(str(root)) or FORBIDDEN.search(str(root.resolve())):
        raise ValueError("prohibited shared source path")
    root = root.resolve()
    package_root = root / "tools" / "selmappo"
    if not (package_root / "__init__.py").is_file():
        raise ValueError("shared fleet package missing")
    name = "_selmappo_shared"
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != package_root / "__init__.py":
            raise ValueError("shared package already bound to another root")
        return module
    spec = importlib.util.spec_from_file_location(
        name, package_root / "__init__.py", submodule_search_locations=[str(package_root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
