"""``configs/golden_r3.toml`` is the configuration the golden artifacts were captured under.

``tests/golden/capture_golden.py`` built it from the legacy CLI argv below; the TOML preset must
resolve to the identical flat (legacy) configuration, so the golden tests exercise exactly the
recipe the legacy code ran.  Only run-local values differ (output dir, stage steps) and the
window-list paths, which are logical names in the preset.
"""
from __future__ import annotations

from pathlib import Path

from wms.config.legacy_flags import parse_legacy_argv
from wms.config.load import load_raw
from wms.config.schema import from_mapping
from wms.config.train import TrainConfig

ROOT = Path(__file__).resolve().parents[3]
RUN_LOCAL = {"out_dir", "stage_steps"}
PATH_KEYS = {"window_list", "val_window_list", "exclude_window_list"}


def golden_argv(unroll_s=10.0, history_s=60.0):
    """capture_golden.r3_argv with placeholder paths (parsing never opens them)."""
    argv = ["train", "--pack", "P1", "--identities", "I1", "--outcomes", "O",
            "--pack", "P2", "--identities", "I2", "--overlay", "V2",
            "--pack", "P3", "--identities", "I3", "--overlay", "V3",
            "--window-list", "lists/windows_train_noPI_noWall.json",
            "--val-window-list", "lists/windows_val_noPI_noWall.json",
            "--exclude-window-list", "lists/excluded_runs_PI_wall_v3.json",
            "--cav-drive", "teacher", "--device", "cpu", "--threads", "2",
            "--num-workers", "0", "--batch-size", "2", "--sumo-fidelity",
            "--start-shift-prob", "0.75", "--truncation-steps", "20",
            "--merge-cal-params", "merge_params.json", "--merge-cal", "--loss-share-logging",
            "--residual-soft-bound-mps2", "4", "--residual-mag-weight", "0.01",
            "--lane-rate-weight", "1.0", "--lane-rate-target", "0.0123",
            "--select-by-long-horizon", "--rollout-mode", "residual", "--lane-head", "off",
            "--asym-speed-loss", "2", "--decel-weight", "5", "--decel-threshold", "3",
            "--propagation-loss", "1", "--prop-window", "3",
            "--seed", "101", "--unroll-s", repr(float(unroll_s)), "--out", "out"]
    return argv + ["--history-s", repr(float(history_s)), "--history-stride-s", "1"]


def _basenames(flat):
    out = dict(flat)
    for key in PATH_KEYS & set(out):
        value = out[key]
        out[key] = ([Path(v).name for v in value] if isinstance(value, list)
                    else None if value is None else Path(value).name)
    return out


def test_golden_preset_equals_the_legacy_argv_configuration():
    _, from_argv = parse_legacy_argv(golden_argv())
    raw = load_raw(ROOT / "configs" / "golden_r3.toml")
    raw["curriculum"]["unroll_s"] = [10.0]
    from_toml = from_mapping(TrainConfig, raw)
    a, b = _basenames(from_argv.to_legacy_flat()), _basenames(from_toml.to_legacy_flat())
    for key in RUN_LOCAL:
        a.pop(key, None), b.pop(key, None)
    assert a == b


def test_r3_preset_is_the_golden_recipe_without_the_opt_in_guards():
    r3 = _basenames(from_mapping(TrainConfig, load_raw(ROOT / "configs" / "r3_packs_only.toml"))
                    .to_legacy_flat())
    golden = _basenames(from_mapping(TrainConfig, load_raw(ROOT / "configs" / "golden_r3.toml"))
                        .to_legacy_flat())
    guards = {"residual_soft_bound_mps2", "residual_mag_weight", "lane_rate_weight",
              "select_by_long_horizon", "history_s", "history_stride_s", "batch_size",
              "device", "threads", "stage_steps", "unroll_curriculum_s", "max_hours"}
    assert {k: v for k, v in r3.items() if k not in guards} == \
        {k: v for k, v in golden.items() if k not in guards}
