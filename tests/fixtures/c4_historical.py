"""Reconstruct the one-switch recipes used to capture the C4 window goldens.

The production v4 recipe subsequently gained new sources, losses and scheduling.
Those changes must not alter the inputs of the junction/hygiene regression checks.
"""
from wms.config.load import load_train_config


def load_window_config(name, overrides=()):
    switches = {
        "c4_wm_v3": [],
        "c4_wm_v4": ["data.junction_fix=ffill"],
        "c4_wm_v4_hygiene": ["data.junction_fix=ffill", "data.controller_hygiene=strict"],
    }
    return load_train_config("configs/c4_wm_v3.toml", [
        *switches[name], f"runtime.out_dir=runs/active/{name}", *overrides])
