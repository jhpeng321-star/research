"""Every legacy CLI flag maps onto a config field; the R3 chain and C1FAST receipt are covered."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from wms.config.legacy_flags import FLAG_TO_FIELD, FLAGS, build_parser, parse_legacy_argv
from wms.config.legacy_flat import FLAT_TO_PATH, get_path
from wms.config.load import load_train_config
from wms.config.schema import to_mapping
from wms.config.train import TrainConfig
from wms.core.errors import ConfigError

from .old_code import run_old

ROOT = Path(__file__).resolve().parents[3]
CHAIN = ROOT / "_out" / "train_r3_chain.sh"
RECEIPT = ROOT / "tests" / "fixtures" / "c1fast_receipt_subset.json"
pytestmark = pytest.mark.filterwarnings("ignore:.*REJECTED")
BASE = ["train", "--pack", "P", "--identities", "I"]

SPECIAL = {"--pack", "--identities", "--overlay", "--unroll-s", "--steps", "--num-workers",
           "--gpu-share", "--stage-steps", "--black-box", "--blackbox-traj"}

# option -> (argv fragment incl. companions, expected nested path, expected value)
SAMPLES = {
    "--outcomes": (["--outcomes", "o"], "data.sidecars.outcomes", "o"),
    "--window-list": (["--window-list", "w.json"], "data.windows.train", "w.json"),
    "--exclude-window-list": (["--exclude-window-list", "a", "--exclude-window-list", "b"],
                              "data.windows.exclude", ["a", "b"]),
    "--val-window-list": (["--val-window-list", "v.json"], "data.windows.val", "v.json"),
    "--val-holdout-frac": (["--window-list", "w", "--val-holdout-frac", "0.2"],
                           "data.windows.val_holdout_frac", 0.2),
    "--val-holdout-seed": (["--val-holdout-seed", "4"], "data.windows.val_holdout_seed", 4),
    "--select-by-long-horizon": (["--select-by-long-horizon"],
                                 "selection.select_by_long_horizon", True),
    "--val-every": (["--val-every", "50"], "selection.val_every", 50),
    "--out": (["--out", "run1"], "runtime.out_dir", "run1"),
    "--threads": (["--threads", "3"], "runtime.threads", 3),
    "--batch-size": (["--batch-size", "16"], "optim.batch_size", 16),
    "--sumo-fidelity": (["--sumo-fidelity"], "model.sumo_fidelity", True),
    "--geometry-only": (["--geometry-only"], "model.geometry_only", True),
    "--truncation-steps": (["--truncation-steps", "10"], "curriculum.truncation_steps", 10),
    "--device": (["--device", "cuda"], "runtime.device", "cuda"),
    "--start-shift-prob": (["--start-shift-prob", "0.75"], "data.start_shift_prob", 0.75),
    "--history-s": (["--history-s", "60"], "data.history_s", 60.0),
    "--history-stride-s": (["--history-s", "30", "--history-stride-s", "2"],
                           "data.history_stride_s", 2.0),
    "--lr": (["--lr", "0.0005"], "optim.lr", 0.0005),
    "--seed": (["--seed", "7"], "runtime.seed", 7),
    "--n-controllers": (["--n-controllers", "6"], "model.n_controllers", 6),
    "--controller-embed-dim": (["--controller-embed-dim", "8"], "model.controller_embed_dim", 8),
    "--switch-window-fraction": (["--switch-window-fraction", "0.25"],
                                 "data.switching.window_fraction", 0.25),
    "--checkpoint-residual": (["--checkpoint-residual"], "model.checkpoint_residual", True),
    "--merge-cal": (["--merge-cal"], "model.merge_cal", True),
    "--merge-cal-params": (["--merge-cal-params", "mp.json"], "model.physics_params", "mp.json"),
    "--traffic-loss": (["--traffic-loss"], "loss.traffic.enabled", True),
    "--traffic-weight": (["--traffic-weight", "0.3"], "loss.traffic.weight", 0.3),
    "--traffic-bin-s": (["--traffic-bin-s", "15"], "loss.traffic.bin_s", 15.0),
    "--traffic-ramp-start-s": (["--traffic-ramp-start-s", "0"], "loss.traffic.ramp_start_s", 0.0),
    "--traffic-ramp-end-s": (["--traffic-ramp-end-s", "200"], "loss.traffic.ramp_end_s", 200.0),
    "--state-noise-std-m": (["--state-noise-std-m", "0.5"], "data.state_noise_std_m", 0.5),
    "--state-noise-std-mps": (["--state-noise-std-mps", "0.1"], "data.state_noise_std_mps", 0.1),
    "--entries-v2": (["--entries-v2"], "data.entries_v2", True),
    "--entry-pad": (["--entry-pad", "4"], "data.entry_pad", 4),
    "--rollout-mode": (["--rollout-mode", "hybrid"], "model.rollout_mode", "hybrid"),
    "--lane-head": (["--rollout-mode", "hybrid", "--lane-head", "history"], "model.lane_head",
                    "history"),
    "--residual-magnitude-weight": (["--residual-magnitude-weight", "0.01"],
                                    "loss.residual.hybrid_magnitude_weight", 0.01),
    "--driver-sidecar": (["--driver-sidecar", "d.npz"], "data.sidecars.driver_sidecar", "d.npz"),
    "--projection-enabled": (["--projection-enabled"], "model.projection.enabled", True),
    "--projection": (["--projection"], "model.projection.enabled", True),
    "--projection-min-gap-m": (["--projection-min-gap-m", "3"], "model.projection.min_gap_m", 3.0),
    "--residual-cap-mps2": (["--residual-cap-mps2", "2"], "model.residual_cap_mps2", 2.0),
    "--residual-soft-bound-mps2": (["--residual-soft-bound-mps2", "1.5"],
                                   "model.residual_soft_bound_mps2", 1.5),
    "--residual-soft-bound-neg-mps2": (["--residual-soft-bound-mps2", "1.5",
                                        "--residual-soft-bound-neg-mps2", "2.5"],
                                       "model.residual_soft_bound_neg_mps2", 2.5),
    "--residual-mag-weight": (["--residual-mag-weight", "0.05"],
                              "loss.residual.residual_mag_weight", 0.05),
    "--lane-rate-weight": (["--lane-rate-weight", "0.1"], "loss.lane_rate.weight", 0.1),
    "--lane-rate-target": (["--lane-rate-target", "0.02"], "loss.lane_rate.target", 0.02),
    "--lane-logit-cap": (["--lane-logit-cap", "4"], "model.lane_logit_cap", 4.0),
    "--stability-weight": (["--stability-weight", "0.2"], "loss.stability_weight", 0.2),
    "--stability-margin": (["--stability-margin", "2"], "loss.stability_margin", 2.0),
    "--stability-probes": (["--stability-probes", "3"], "loss.stability_probes", 3),
    "--conservation-metric": (["--conservation-metric"], "loss.conservation_enabled", True),
    "--conservation-weight": (["--conservation-weight", "0.2"], "loss.conservation_weight", 0.2),
    "--loss-share-logging": (["--loss-share-logging"], "loss.loss_share_logging", True),
    "--asym-speed-loss": (["--asym-speed-loss", "2"], "loss.braking.asym_speed_alpha", 2.0),
    "--decel-weight": (["--decel-weight", "5"], "loss.braking.decel_weight", 5.0),
    "--decel-threshold": (["--decel-threshold", "2.5"], "loss.braking.decel_threshold_mps2", 2.5),
    "--propagation-loss": (["--propagation-loss", "1"], "loss.braking.propagation_weight", 1.0),
    "--prop-window": (["--prop-window", "4"], "loss.braking.prop_window_s", 4.0),
    "--prop-max-gap-m": (["--prop-max-gap-m", "40"], "loss.braking.prop_max_gap_m", 40.0),
    "--residual-cap-neg-mps2": (["--residual-cap-neg-mps2", "5"],
                                "model.residual_cap_neg_mps2", 5.0),
    "--cav-drive": (["--cav-drive", "closed_loop"], "data.cav_drive", "closed_loop"),
    "--leave-pi-wall-out": (["--leave-pi-wall-out"], "data.windows.leave_pi_wall_out", True),
    "--heldout-family-policy": (["--heldout-family-policy", "refuse"],
                                "data.switching.heldout_family_policy", "refuse"),
    "--heldout-max-drop-frac": (["--heldout-max-drop-frac", "0.5"],
                                "data.switching.heldout_max_drop_frac", 0.5),
    "--switch-mix-ratio": (["--switch-mix-ratio", "0.3"], "data.switching.mix_ratio", 0.3),
    "--event-oversample": (["--event-oversample", "1"], "data.switching.event_oversample", 1.0),
    "--jam-oversample": (["--jam-oversample", "3"], "data.jam.oversample", 3.0),
    "--jam-speed-mps": (["--jam-speed-mps", "4"], "data.jam.speed_mps", 4.0),
    "--jam-cache-dir": (["--jam-cache-dir", "jc"], "data.jam.cache_dir", "jc"),
    "--switch-val-frac": (["--switch-val-frac", "0.2"], "data.switching.val_frac", 0.2),
    "--switch-val-seed": (["--switch-val-seed", "2"], "data.switching.val_seed", 2),
}
SWITCH = ["--switch-pack", "SP", "--switch-identities", "SI", "--switch-window-list", "SW",
          "--switch-timeline-index", "TI", "--zero-shot-heldout", "ZS"]
SWITCH_SAMPLES = {
    "--switch-pack": ("data.switching.pack", "SP"),
    "--switch-identities": ("data.switching.identities", "SI"),
    "--switch-window-list": ("data.switching.windows_train", "SW"),
    "--switch-timeline-index": ("data.switching.timeline_index", "TI"),
    "--zero-shot-heldout": ("data.switching.zero_shot_heldout", "ZS"),
}


def nested(argv):
    return to_mapping(parse_legacy_argv(argv)[1])


def test_samples_cover_every_non_special_flag():
    covered = set(SAMPLES) | set(SWITCH_SAMPLES) | {"--switch-val-window-list"}
    assert covered == set(FLAG_TO_FIELD) - SPECIAL


@pytest.mark.parametrize("option", sorted(SAMPLES))
def test_flag_sets_its_config_field(option):
    fragment, path, expected = SAMPLES[option]
    assert FLAG_TO_FIELD[option] == path
    assert get_path(nested(BASE + fragment), path) == expected


@pytest.mark.parametrize("option", sorted(SWITCH_SAMPLES))
def test_switching_flags_set_their_field(option):
    path, expected = SWITCH_SAMPLES[option]
    assert FLAG_TO_FIELD[option] == path
    assert get_path(nested(BASE + SWITCH), path) == expected


def test_switch_val_window_list_and_all_or_nothing_rule():
    tree = nested(BASE + SWITCH + ["--switch-val-window-list", "SV"])
    assert get_path(tree, "data.switching.windows_val") == "SV"
    with pytest.raises(ConfigError, match="requires"):
        parse_legacy_argv(BASE + ["--switch-pack", "SP"])
    with pytest.raises(ConfigError, match="without"):
        parse_legacy_argv(BASE + ["--zero-shot-heldout", "ZS"])


def test_special_flags():
    tree = nested(["train", "--pack", "A", "--pack", "B", "--identities", "IA", "--identities", "IB",
                   "--overlay", "OB", "--unroll-s", "10", "--unroll-s", "30",
                   "--stage-steps", "5,6", "--steps", "9", "--gpu-share"])
    assert tree["data"]["packs"] == {"pack0": "A", "pack1": "B"}
    assert tree["data"]["sidecars"]["identities"] == {"pack0": "IA", "pack1": "IB"}
    assert tree["data"]["sidecars"]["overlays"] == {"pack1": "OB"}
    assert tree["curriculum"]["unroll_s"] == [10.0, 30.0]
    assert tree["curriculum"]["stage_steps"] == [5, 6]
    assert tree["curriculum"]["steps_per_epoch"] == 9
    assert tree["runtime"]["num_workers"] == 4
    assert nested(BASE + ["--gpu-share", "--num-workers", "1"])["runtime"]["num_workers"] == 1
    assert nested(BASE + ["--num-workers", "0"])["runtime"]["num_workers"] == 0
    assert nested(BASE + ["--black-box"])["model"]["kind"] == "blackbox_id"
    assert nested(BASE + ["--blackbox-traj"])["model"]["kind"] == "blackbox_traj"
    with pytest.raises(ConfigError, match="mutually exclusive"):
        parse_legacy_argv(BASE + ["--black-box", "--blackbox-traj"])


def test_overlay_and_identity_counts_are_validated():
    with pytest.raises(ConfigError, match="once per --pack"):
        parse_legacy_argv(["train", "--pack", "A", "--pack", "B", "--identities", "IA"])
    with pytest.raises(ConfigError, match="once per extra"):
        parse_legacy_argv(["train", "--pack", "A", "--identities", "IA", "--overlay", "x",
                           "--overlay", "y"])


def test_dry_run_defaults_to_two_steps():
    command, cfg = parse_legacy_argv(["dry-run", "--pack", "P", "--identities", "I"])
    assert command == "dry-run" and cfg.curriculum.steps_per_epoch == 2
    assert parse_legacy_argv(BASE)[1].curriculum.steps_per_epoch == 2000


def test_flag_table_has_no_unmapped_flag_and_paths_exist():
    default = to_mapping(TrainConfig())
    for flag in FLAGS:
        if flag.path is not None:
            get_path(default, flag.path)  # KeyError if the path is not a config field
    assert {o for o, p in FLAG_TO_FIELD.items() if p == "<special>"} == SPECIAL


def test_every_flat_key_path_exists():
    default = to_mapping(TrainConfig())
    for path in FLAT_TO_PATH.values():
        get_path(default, path)


def test_parser_matches_the_legacy_parser_defaults_and_types():
    argvs = [BASE, ["dry-run", "--pack", "P", "--identities", "I"],
             BASE + ["--unroll-s", "10", "--lr", "0.01", "--projection", "--conservation-metric",
                     "--prop-max-gap-m", "5", "--cav-drive", "closed_loop"]]
    code = ("import json; from wms.cli import _parser; p = _parser(); "
            f"print(json.dumps([vars(p.parse_args(a)) for a in {argvs!r}]))")
    old = run_old(code)
    new = [vars(build_parser().parse_args(a)) for a in argvs]
    assert new == old


def test_r3_chain_flags_are_all_mapped():
    if not CHAIN.is_file():
        pytest.skip("R3 chain not present")
    text = CHAIN.read_text(encoding="utf-8")
    flags = set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)(?=[\s\"'|)]|$)", text, re.M))
    # Options of other tools or of the chain's own interface, not of `wms.cli train`.
    foreign = {"--help", "--itemize-changes", "--stages", "--require-terms", "--ramp-end-s", "--max-hours"}
    missing = sorted(flags - foreign - set(FLAG_TO_FIELD))
    assert not missing, f"R3 chain flags without a config field: {missing}"
    assert {"--stage-steps", "--merge-cal", "--loss-share-logging", "--asym-speed-loss",
            "--switch-mix-ratio", "--zero-shot-heldout"} <= flags


def test_c1fast_receipt_keys_are_all_mapped():
    flat = json.loads(RECEIPT.read_text(encoding="utf-8"))["config"]
    assert set(flat) <= set(FLAT_TO_PATH)


R3_ARGV = (
    "train --pack P3 --identities I3 --outcomes OUT "
    "--pack PD --identities ID --overlay OD --pack PM --identities IM --overlay OM "
    "--window-list WT --val-window-list WV --exclude-window-list EX "
    "--switch-pack SP --switch-identities SI --switch-window-list SWT --switch-timeline-index STI "
    "--zero-shot-heldout ZSH --switch-mix-ratio 0.3 --event-oversample 1 "
    "--switch-val-frac .1 --heldout-family-policy drop --heldout-max-drop-frac .85 "
    "--cav-drive teacher --device cuda --threads 2 --num-workers 0 --batch-size 8 --sumo-fidelity "
    "--start-shift-prob 0.75 --truncation-steps 20 --merge-cal-params MP --merge-cal "
    "--loss-share-logging --rollout-mode residual --lane-head off "
    "--unroll-s 10 --unroll-s 30 --unroll-s 60 --unroll-s 120 "
    "--stage-steps 2500,2500,1875,1250 --out RUN"
).split()
R3_BRAKING = {"full": ["--asym-speed-loss", "2", "--decel-weight", "5", "--decel-threshold", "3",
                       "--propagation-loss", "1", "--prop-window", "3"],
              "ctrl": ["--asym-speed-loss", "1", "--decel-weight", "1", "--decel-threshold", "3",
                       "--propagation-loss", "0", "--prop-window", "3"]}
# Fields that hold data locations or run bookkeeping, not the recipe.
LOCATION_PATHS = ("data.packs", "data.sidecars", "data.windows.train", "data.windows.val",
                  "data.windows.exclude", "data.switching.pack", "data.switching.identities",
                  "data.switching.driver_params", "data.switching.windows_train",
                  "data.switching.timeline_index", "data.switching.zero_shot_heldout",
                  "model.physics_params", "runtime.out_dir", "curriculum.max_hours", "name")


def _without_locations(cfg: TrainConfig) -> dict:
    tree = to_mapping(cfg)
    for dotted in LOCATION_PATHS:
        *parents, leaf = dotted.split(".")
        node = tree
        for part in parents:
            node = node[part]
        node[leaf] = None
    return tree


@pytest.mark.parametrize("preset, braking", [("r3", "full"), ("r3_ctrl", "ctrl")])
def test_presets_reproduce_the_r3_chain_recipe(preset, braking):
    _command, from_chain = parse_legacy_argv(R3_ARGV + R3_BRAKING[braking])
    assert _without_locations(load_train_config(preset)) == _without_locations(from_chain)


def test_r3_chain_location_fields_have_matching_shape():
    cfg = load_train_config("r3")
    assert list(cfg.data.packs) == ["pack_v3", "dspeed", "damping"]
    assert set(cfg.data.sidecars.identities) == set(cfg.data.packs)
    assert list(cfg.data.sidecars.overlays) == ["dspeed", "damping"]
    assert cfg.data.sidecars.outcomes is not None and len(cfg.data.windows.exclude) == 1
    assert cfg.required_log_terms() == ("position", "speed", "lane", "asym_speed", "decel_weight",
                                        "propagation")
