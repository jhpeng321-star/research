import json, sys
sys.path.insert(0, 'src')
from wms.control.train_cli import _read_config
from wms.config.schema import deep_merge
raw = _read_config('configs/c4_policy14_c4prep_on.toml')
over = {"trainer": {"native_anchor": False, "native_anchor_calibration": ""},
        "data": {"focus_cell_fraction": 0.6},
        "policy14": {"fd_gate": False, "free_road_gate": False},
        "counterfactual": {"curriculum_updates": [0,1], "w_individual":[8.9,8.9],"w_group":[9.9,9.9],"w_coop":[8.9,8.9],"w_team":[10.3,10.3],"w_all":[1.0,1.0]},
        "horizon": {"enabled": True, "horizons_s":[10,90], "curriculum_updates":[0,1], "entry_pad_max":[32,288], "gate_file":"X"}}
raw = deep_merge(raw, over)
print(json.dumps(raw, indent=1, default=str))
