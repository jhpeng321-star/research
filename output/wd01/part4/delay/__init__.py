"""WD-01 part 4 delay library.

Total-delay accounting per PREDICT_CONTRACT_v1.md §4.1 (authoritative
copy at RD_WT_ROOT/wd01/output/wd01/PREDICT_CONTRACT_v1.md, sha256
74a1de8edb52437babfda611336e46fb9653f7d1cbd43da878ac0295d5fe0280).
Pure numpy, shared by truth-log labels, the contract trajectory layout
and future Part B label building.  Zero simulation.
"""
from .geometry import (ENTRANCE_MAINLINE, ENTRANCE_NONE, ENTRANCE_RAMP,
                       JUNCTION, SectionGeometry, make_geometry,
                       wm30_geometry)
from .core import (EPS, VehicleDelay, VehicleTrace, WaitRecord, WindowDelay,
                   vehicle_delay, waiting_component, window_totals)
from .truth import (RootData, load_root_data, load_schedule, loading_of,
                    manifest, roots_by_split, root_run_dir, run_map,
                    vtype_speed_factors, window_truth)
from .contract_io import contract_view_of_truth, delay_from_contract_layout

__all__ = [
    "ENTRANCE_MAINLINE", "ENTRANCE_NONE", "ENTRANCE_RAMP", "JUNCTION",
    "SectionGeometry", "make_geometry", "wm30_geometry",
    "EPS", "VehicleTrace", "VehicleDelay", "WaitRecord", "WindowDelay",
    "vehicle_delay", "waiting_component", "window_totals",
    "RootData", "load_root_data", "load_schedule", "loading_of",
    "manifest", "roots_by_split", "root_run_dir", "run_map",
    "vtype_speed_factors", "window_truth",
    "contract_view_of_truth", "delay_from_contract_layout",
]
