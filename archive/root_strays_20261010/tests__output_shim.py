"""Import shim: exposes the aggregation helper from the analysis directory."""
import importlib.util
from pathlib import Path

_target = Path(__file__).resolve().parent.parent / "output" / "wall_harm_speed_v1" / "ttc_decompose.py"
_spec = importlib.util.spec_from_file_location("ttc_decompose", str(_target))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
aggregate_encounters = _mod.aggregate_encounters
