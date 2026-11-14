"""Standalone local chain guards. Never imports trainer or opens real packs.

Run: python -B _out/test_train_main_chain.py
BASH_EXE may name Git Bash; fixtures and subprocess logs stay in this workspace.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


OUT = Path(__file__).resolve().parent
CHAIN = OUT / "train_main_chain.sh"
DATA = OUT / "train_main_data.json"
BASH = os.environ.get("BASH_EXE", "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash") or "")


class ContractTests(unittest.TestCase):
    def test_frozen_lpwo_metadata_and_row_level_train_guards(self):
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", CHAIN.read_text(), re.S)
        gate = next(block for block in blocks if "expected_lists =" in block)
        with tempfile.TemporaryDirectory(prefix="lpwo_metadata_", dir=OUT) as tmp:
            root = Path(tmp)
            train, monitor = root / "train.json", root / "monitor.json"
            payload = {"schema": "wm_s_lpo_windows/1", "n_windows": 1,
                       "windows": [{"split": "TRAIN"}]}
            monitor.write_text(json.dumps({**payload, "list": "windows_val_noPI_noWall"}),
                               encoding="utf-8")
            cases = [(payload, None),
                     ({**payload, "split": "CAL"}, "TRAIN-scoped"),
                     ({**payload, "windows": [{"split": "CAL"}]}, "explicitly TRAIN"),
                     ({**payload, "windows": [{}]}, "explicitly TRAIN"),
                     ({**payload, "n_windows": 2}, "exact count"),
                     ({**payload, "schema": "unknown"}, "TRAIN-scoped")]
            for data, error in cases:
                train.write_text(json.dumps({**data, "list": "windows_train_noPI_noWall"}),
                                 encoding="utf-8")
                with self.subTest(error=error), mock.patch("sys.argv", ["gate", str(train), str(monitor)]):
                    if error:
                        with self.assertRaisesRegex(SystemExit, error):
                            exec(compile(gate, str(CHAIN), "exec"), {})
                    else:
                        exec(compile(gate, str(CHAIN), "exec"), {})

    def test_lf_and_fixed_four_stage_budget(self):
        raw = CHAIN.read_bytes()
        self.assertNotIn(b"\r", raw)
        data = json.loads(DATA.read_text())
        self.assertEqual(data["curriculum"]["stages_s"], [10, 30, 60, 120])
        self.assertEqual(data["curriculum"]["stage_steps"], [2500, 2500, 1875, 1250])
        self.assertEqual(data["curriculum"]["smoke_stage_steps"], [6] * 4)
        self.assertEqual(data["execution"]["num_workers"], 0)
        text = raw.decode()
        self.assertIn('MODE=${1:-preflight}', text)
        self.assertIn('STAGES_CSV=10,30,60,120 STAGE_STEPS=2500,2500,1875,1250', text)
        self.assertIn('SMOKE_STAGE_STEPS=6,6,6,6', text)

    def test_stage_allowlist_and_exclusions(self):
        data = json.loads(DATA.read_text())
        self.assertEqual(data["switching"]["stage_paths"], [
            "pack_switching_train", "identities_switching_train", "driver_params_switching_train",
            "ACTION_TIMELINE_INDEX_train.json", "windows_train_hetdata.json", "ZERO_SHOT_HELDOUT.json"])
        text = CHAIN.read_text()
        self.assertIn('--exclude-window-list "$EXCLUDE" --leave-pi-wall-out', text)
        self.assertIn('--val-window-list "$WINDOWS_VAL"', text)
        self.assertIn('--traffic-ramp-start-s 0 --traffic-ramp-end-s 1', text)
        self.assertIn('FLAGS=${FLAGS:-}', text)
        self.assertIn('BRAKING_FLAGS=${BRAKING_FLAGS:-}', text)

    def test_embedded_python_syntax(self):
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", CHAIN.read_text(), re.S)
        self.assertGreaterEqual(len(blocks), 7)
        for block in blocks:
            compile(block, str(CHAIN), "exec")

    def test_smoke_requires_six_steps_per_stage_and_valid_timing(self):
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", CHAIN.read_text(), re.S)
        gate = next(b for b in blocks if "counts = {s: 0" in b)
        terms = ["position", "speed", "lane", "traffic", "stability", "conservation"]
        with tempfile.TemporaryDirectory(prefix="smoke_guard_", dir=OUT) as tmp:
            root = Path(tmp)
            shares, result, log = [root / s for s in ["shares.jsonl", "gate.txt", "train.jsonl"]]
            shares.write_text("\n".join(json.dumps({"stage_s": s, "loss_share": dict.fromkeys(terms, .1)})
                                        for s in [10, 30, 60, 120]), encoding="utf-8")
            result.write_text('{"status":"passed"}', encoding="utf-8")
            valid = [{"stage_s": s, "seconds": 1.0, **dict.fromkeys(terms, .1)}
                     for s in [10, 30, 60, 120] for _ in range(6)]
            cases = [(valid, None), (valid[:-1], "exactly six"),
                     ([{**valid[0], "seconds": -1}] + valid[1:], "invalid smoke step timing")]
            for rows, error in cases:
                with self.subTest(error=error):
                    log.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
                    with mock.patch("sys.argv", ["gate", str(shares), str(result), str(log), ",".join(terms)]):
                        if error:
                            with self.assertRaisesRegex(SystemExit, error):
                                exec(compile(gate, str(CHAIN), "exec"), {})
                        else:
                            exec(compile(gate, str(CHAIN), "exec"), {})

    def test_smoke_requires_enabled_braking_in_shares_and_every_step(self):
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", CHAIN.read_text(), re.S)
        gate = next(b for b in blocks if "counts = {s: 0" in b)
        base = ["position", "speed", "lane", "traffic", "stability", "conservation"]
        with tempfile.TemporaryDirectory(prefix="braking_gate_", dir=OUT) as tmp:
            root = Path(tmp)
            shares, result, log = [root / s for s in ["shares.jsonl", "gate.txt", "train.jsonl"]]
            result.write_text('{"status":"passed"}', encoding="utf-8")
            for brake_term in ["asym_speed", "decel_weight", "propagation"]:
                terms = base + [brake_term]
                for defect in [None, "share_missing", "share_nan", "share_negative", "share_bool",
                               "step_missing", "step_nan", "step_bool"]:
                    with self.subTest(term=brake_term, defect=defect):
                        share_rows = [{"stage_s": s, "loss_share": dict.fromkeys(terms, .1)}
                                      for s in [10, 30, 60, 120]]
                        rows = [{"stage_s": s, "seconds": 1.0, **dict.fromkeys(terms, .1)}
                                for s in [10, 30, 60, 120] for _ in range(6)]
                        if defect:
                            target = share_rows[0]["loss_share"] if defect.startswith("share") else rows[7]
                            if defect.endswith("missing"):
                                del target[brake_term]
                            else:
                                target[brake_term] = {"nan": float("nan"), "negative": -.1,
                                                      "bool": True}[defect.split("_")[1]]
                        shares.write_text("\n".join(map(json.dumps, share_rows)), encoding="utf-8")
                        log.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
                        with mock.patch("sys.argv", ["gate", str(shares), str(result), str(log), ",".join(terms)]):
                            if defect:
                                with self.assertRaisesRegex(SystemExit, "loss-share|smoke term"):
                                    exec(compile(gate, str(CHAIN), "exec"), {})
                            else:
                                exec(compile(gate, str(CHAIN), "exec"), {})


@unittest.skipUnless(BASH, "bash unavailable")
class BashGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="chain_test_", dir=OUT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = os.environ.copy()
        # Do not inherit a user's training configuration.
        for key in list(self.env):
            if key.startswith(("SW_", "PACK_", "IDENTITIES_", "OVERLAY_")) or key in {
                "PY", "W", "BOX", "ORIGINAL_ROOT", "SIDECAR_ROOT", "TRAIN_OUTCOMES", "MERGE_PARAMS",
                "WINDOWS_TRAIN", "WINDOWS_VAL", "EXCLUDE", "ZERO_SHOT_HELDOUT", "NUM_WORKERS",
                "FLAGS", "BRAKING_FLAGS", "MAX_HOURS", "OMP", "CODE_ROOT", "WM_CODE",
                "MODEL_KIND", "ROLLOUT_MODE", "LANE_HEAD", "ASYM_SPEED_LOSS", "DECEL_WEIGHT",
                "DECEL_THRESHOLD", "PROPAGATION_LOSS", "PROP_WINDOW", "PROP_MAX_GAP_M",
                "RESIDUAL_CAP_NEG_MPS2", "SWITCH_MIX_RATIO", "EVENT_OVERSAMPLE", "CAV_DRIVE",
                "STUB_DISK_FAIL", "STUB_CPU_FAIL", "STUB_SMOKE_PASS", "STUB_SMOKE_TERMS"}:
                del self.env[key]
        self.env.update(BOX="gpuA", ORIGINAL_ROOT=str(self.root), SW_ROOT=str(self.root / "switch"),
                        W=str(self.root / "fresh_run"), CODE_ROOT=str(self.root),
                        WM_CODE=str(self.root / "code"), OMP="2")
        dirs = {
            "PACK_V3": "v3", "PACK_DSPEED": "dspeed", "PACK_DAMPING": "damping",
            "IDENTITIES_V3": "id_v3", "IDENTITIES_DSPEED": "id_dspeed", "IDENTITIES_DAMPING": "id_damping",
            "OVERLAY_DSPEED": "overlay_dspeed", "OVERLAY_DAMPING": "overlay_damping", "TRAIN_OUTCOMES": "outcomes",
            "SW_PACK": "switch/pack_switching_train", "SW_IDENTITIES": "switch/identities_switching_train",
            "SW_DRIVER": "switch/driver_params_switching_train"}
        (self.root / "code/wms").mkdir(parents=True)
        for key, name in dirs.items():
            p = self.root / name
            p.mkdir(parents=True)
            self.env[key] = str(p)
        files = {"WINDOWS_TRAIN": "train.json", "WINDOWS_VAL": "monitor.json", "EXCLUDE": "exclude.json",
                 "MERGE_PARAMS": "merge.json", "SW_WINDOWS_TRAIN": "switch/windows_train_hetdata.json",
                 "SW_TIMELINE_INDEX": "switch/ACTION_TIMELINE_INDEX_train.json",
                 "ZERO_SHOT_HELDOUT": "switch/ZERO_SHOT_HELDOUT.json"}
        for key, name in files.items():
            p = self.root / name
            p.write_text('{"split":"TRAIN"}\n', encoding="utf-8")
            self.env[key] = str(p)
        for key, names in {
            "SW_PACK": ["manifest.json", "INDEX.json", "controller_timeline.npy", "issued_acceleration_mps2.npy", "issued_target_speed_mps.npy"],
            "SW_IDENTITIES": ["TRAIN_IDENTITY_RECEIPT.json", "identities.npy", "global_positions.npy", "vehicle_map.json"],
            "SW_DRIVER": ["driver_params.npy"]}.items():
            for name in names:
                (Path(self.env[key]) / name).write_text('{"split":"TRAIN"}\n', encoding="utf-8")
        self.calls = self.root / "stub_calls.txt"
        self.env["STUB_CALLS"] = str(self.calls)
        self.argv = self.root / "stub_argv.txt"
        self.env["STUB_ARGV"] = str(self.argv)
        self.env["CONTRACT_PY"] = sys.executable
        self.stub = self.root / "python_stub.sh"
        # Only the stdlib braking validator runs Python; trainer/help are stubbed.
        flags = sorted(set(re.findall(r"--[a-z][a-z0-9-]*", CHAIN.read_text())) |
                       {"--checkpoint-residual", "--residual-magnitude-weight"})
        self.help = " ".join(flags)
        self.write_stub()
        self.env["PY"] = str(self.stub)

    def write_stub(self, missing=""):
        help_text = " ".join(f for f in self.help.split() if f != missing)
        self.stub.write_text(
            '#!/usr/bin/env bash\nset -eu\nprintf "%s\\n" "$*" >> "$STUB_CALLS"\n'
            'if [[ "$*" == *"--help"* ]]; then\n'
            f"  printf '%s\\n' '{help_text}'\n  exit 0\nfi\n"
            'if [[ "$1" == "-" ]]; then\n'
            '  input=$(cat)\n'
            '  if [[ "$input" == *"parser = argparse.ArgumentParser"* ]]; then\n'
            '    printf "%s\\n" "$input" | "$CONTRACT_PY" -B - "${@:2}"\n    exit $?\n  fi\n'
            '  if [[ "${2:-}" == "disk" && "${STUB_DISK_FAIL:-0}" == 1 ]]; then exit 8; fi\n'
            '  if [[ "${2:-}" == "idle" && "${STUB_CPU_FAIL:-0}" == 1 ]]; then exit 9; fi\n'
            '  if [[ "$input" == *"subprocess.Popen"* ]]; then\n'
            '    printf "%s\\n" "${@:5}" > "$STUB_ARGV"\n'
            '    if [[ "${STUB_SMOKE_PASS:-0}" == 1 ]]; then\n'
            '      "$CONTRACT_PY" -B - "$2/$3" "$STUB_SMOKE_TERMS" <<\'FIXTURE\'\n'
            'import json, pathlib, sys\n'
            'root = pathlib.Path(sys.argv[1]); root.mkdir()\n'
            'terms = sys.argv[2].split(",")\n'
            'rows = [{"stage_s": s, "seconds": 1., **dict.fromkeys(terms, .1)}\n'
            '        for s in [10, 30, 60, 120] for _ in range(6)]\n'
            '(root / "train_log.jsonl").write_text("\\n".join(map(json.dumps, rows)))\n'
            'shares = [{"stage_s": s, "loss_share": dict.fromkeys(terms, .1)}\n'
            '          for s in [10, 30, 60, 120]]\n'
            '(root / "loss_share.jsonl").write_text("\\n".join(map(json.dumps, shares)))\n'
            'FIXTURE\n'
            '      exit 0\n    fi\n'
            '    printf "stub smoke failure\\n" > "$2/$3.log"\n    exit 17\n  fi\n'
            '  if [[ "$input" == *"counts = {s: 0"* ]]; then\n'
            '    printf "%s\\n" "$input" | "$CONTRACT_PY" -B - "${@:2}"\n    exit $?\n  fi\n'
            '  exit 0\nfi\n'
            'if [[ "${STUB_SMOKE_PASS:-0}" == 1 && "$2" == "wms.gate_stages" ]]; then\n'
            '  printf \'{"status":"passed"}\\n\'\n  exit 0\nfi\n'
            'if [[ "${STUB_SMOKE_PASS:-0}" == 1 && "$2" == "wms.plan_steps" ]]; then exit 18; fi\n'
            'exit 99\n', encoding="utf-8", newline="\n")
        self.stub.chmod(0o755)

    def invoke(self, *args):
        return subprocess.run([BASH, str(CHAIN), *args], cwd=OUT.parent, env=self.env,
                              capture_output=True, text=True, timeout=20)

    def assert_no_train(self):
        if self.calls.exists():
            calls = self.calls.read_text()
            self.assertNotIn("--out ", calls)
            for line in calls.splitlines():
                if "wms.cli train" in line:
                    self.assertIn("--help", line)

    def test_bash_syntax(self):
        p = subprocess.run([BASH, "-n", str(CHAIN)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_default_preflight_never_trains_or_creates_output(self):
        p = self.invoke()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("preflight passed", p.stdout)
        self.assertFalse(Path(self.env["W"]).exists())
        self.assert_no_train()

    def stub_run_argv(self):
        # The monitor stub records the proposed argv then fails; no trainer runs.
        self.env["W"] = str(self.root / f"run_{len(list(self.root.glob('run_*')))}")
        p = self.invoke("run")
        self.assertEqual(p.returncode, 17, p.stderr)
        args = self.argv.read_text().splitlines()
        self.assertEqual(args[1:4], ["-m", "wms.cli", "train"])
        return args[4:]

    def test_model_argv_defaults_and_shared_split_mix(self):
        shared = None
        for kind, selected in [("structured", None), ("blackbox_traj", "--blackbox-traj"),
                               ("blackbox_id", "--black-box")]:
            with self.subTest(model=kind):
                self.env["MODEL_KIND"] = kind
                args = self.stub_run_argv()
                self.assertEqual([a for a in args if a.startswith("--black")], [selected] if selected else [])
                self.assertEqual(args[args.index("--rollout-mode") + 1], "residual")
                self.assertEqual(args[args.index("--lane-head") + 1], "off")
                self.assertNotIn("--prop-max-gap-m", args)
                self.assertNotIn("--residual-cap-neg-mps2", args)
                braking_start = args.index("--asym-speed-loss") if kind == "structured" else args.index("--stage-steps")
                # The entire COMMON and curriculum argv are identical across models.
                current = args[:args.index(selected or "--rollout-mode")] + args[args.index("--unroll-s"):braking_start]
                if shared is None:
                    shared = current
                else:
                    self.assertEqual(current, shared)
                for option, value in [("--window-list", self.env["WINDOWS_TRAIN"]),
                                      ("--val-window-list", self.env["WINDOWS_VAL"]),
                                      ("--exclude-window-list", self.env["EXCLUDE"]),
                                      ("--switch-window-list", self.env["SW_WINDOWS_TRAIN"]),
                                      ("--switch-mix-ratio", "0.5"), ("--event-oversample", "2"),
                                      ("--switch-val-frac", ".1"), ("--heldout-family-policy", "drop"),
                                      ("--heldout-max-drop-frac", ".85"), ("--cav-drive", "teacher"),
                                      ("--stage-steps", "6,6,6,6")]:
                    self.assertEqual(args[args.index(option) + 1], value)
                self.assertIn("--leave-pi-wall-out", args)
                if kind == "structured":
                    for option, value in [("--asym-speed-loss", "1"), ("--decel-weight", "1"),
                                          ("--decel-threshold", "3"), ("--propagation-loss", "0"),
                                          ("--prop-window", "3")]:
                        self.assertEqual(args[args.index(option) + 1], value)
                else:
                    self.assertNotIn("--asym-speed-loss", args)

    def test_hybrid_history_and_explicit_braking_argv(self):
        self.env.update(ROLLOUT_MODE="hybrid", LANE_HEAD="history", ASYM_SPEED_LOSS="2",
                        DECEL_WEIGHT="5", DECEL_THRESHOLD="2.5", PROPAGATION_LOSS="1",
                        PROP_WINDOW="4", PROP_MAX_GAP_M="80", RESIDUAL_CAP_NEG_MPS2="3",
                        FLAGS="--checkpoint-residual --residual-magnitude-weight .002")
        args = self.stub_run_argv()
        for option, value in [("--rollout-mode", "hybrid"), ("--lane-head", "history"),
                              ("--asym-speed-loss", "2"), ("--decel-weight", "5"),
                              ("--decel-threshold", "2.5"), ("--propagation-loss", "1"),
                              ("--prop-window", "4"), ("--prop-max-gap-m", "80"),
                              ("--residual-cap-neg-mps2", "3"), ("--residual-magnitude-weight", ".002")]:
            self.assertEqual(args[args.index(option) + 1], value)
        self.assertIn("--checkpoint-residual", args)

    def test_supported_hybrid_modes_and_optional_monitor_share_contract(self):
        monitor = self.root / "switch/monitor.json"
        monitor.write_text('{"split":"TRAIN"}', encoding="utf-8")
        self.env.update(ROLLOUT_MODE="hybrid", SW_WINDOWS_VAL=str(monitor),
                        SWITCH_MIX_RATIO="0.3", EVENT_OVERSAMPLE="4")
        for kind in ["structured", "blackbox_id"]:
            for head in ["off", "history"]:
                with self.subTest(model=kind, head=head):
                    self.env.update(MODEL_KIND=kind, LANE_HEAD=head)
                    args = self.stub_run_argv()
                    for option, value in [("--rollout-mode", "hybrid"), ("--lane-head", head),
                                          ("--switch-val-window-list", str(monitor)),
                                          ("--switch-mix-ratio", "0.3"), ("--event-oversample", "4")]:
                        self.assertEqual(args[args.index(option) + 1], value)

    def test_invalid_model_mode_and_head_combinations(self):
        cases = [({"MODEL_KIND": "other"}, "MODEL_KIND must"),
                 ({"ROLLOUT_MODE": "driver_params"}, "ROLLOUT_MODE must"),
                 ({"LANE_HEAD": "other"}, "LANE_HEAD must"),
                 ({"LANE_HEAD": "history"}, "requires ROLLOUT_MODE=hybrid")]
        cases += [({"MODEL_KIND": "blackbox_traj", "ROLLOUT_MODE": "hybrid", "LANE_HEAD": head},
                   "blackbox_traj requires") for head in ["off", "history"]]
        for config, error in cases:
            with self.subTest(config=config), mock.patch.dict(self.env, config):
                p = self.invoke()
                self.assertNotEqual(p.returncode, 0)
                self.assertIn(error, p.stderr)
        self.assert_no_train()

    def test_mode_head_and_model_extra_overrides_refused(self):
        for key in ["FLAGS", "BRAKING_FLAGS"]:
            for value in ["--rollout-mode hybrid", "--rollout-mode=hybrid", "--lane-head history",
                          "--lane-head=off", "--black-box", "--blackbox-traj"]:
                with self.subTest(key=key, value=value), mock.patch.dict(self.env, {key: value}):
                    p = self.invoke()
                    self.assertNotEqual(p.returncode, 0)
                    self.assertIn("cannot override fixed contract", p.stderr)
        self.assert_no_train()

    def test_braking_terms_follow_effective_env_and_legacy_flags(self):
        base = "position,speed,lane,traffic,stability,conservation"
        cases = [({}, base), ({"ASYM_SPEED_LOSS": "2"}, base + ",asym_speed"),
                 ({"DECEL_WEIGHT": "5"}, base + ",decel_weight"),
                 ({"PROPAGATION_LOSS": ".2"}, base + ",propagation"),
                 ({"RESIDUAL_CAP_NEG_MPS2": "3"}, base),
                 ({"FLAGS": "--asym-speed-loss=2", "BRAKING_FLAGS": "--decel-weight 5 --propagation-loss 1"},
                  base + ",asym_speed,decel_weight,propagation"),
                 ({"ASYM_SPEED_LOSS": "2", "FLAGS": "--asym-speed-loss 3",
                   "BRAKING_FLAGS": "--asym-speed-loss=1"}, base)]
        for config, terms in cases:
            with self.subTest(config=config), mock.patch.dict(self.env, config):
                p = self.invoke()
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertIn("terms=" + terms + "\n", p.stdout)
        self.assert_no_train()

    def test_blackboxes_refuse_enabled_braking_in_env_and_extras(self):
        for kind in ["blackbox_traj", "blackbox_id"]:
            for config in [{"ASYM_SPEED_LOSS": "2"}, {"DECEL_WEIGHT": "5"},
                           {"PROPAGATION_LOSS": "1"}, {"RESIDUAL_CAP_NEG_MPS2": "3"},
                           {"FLAGS": "--asym-speed-loss=2"}, {"BRAKING_FLAGS": "--propagation-loss 1"},
                           {"BRAKING_FLAGS": "--residual-cap-neg-mps2=3"}]:
                with self.subTest(kind=kind, config=config), mock.patch.dict(self.env, {"MODEL_KIND": kind, **config}):
                    p = self.invoke()
                    self.assertNotEqual(p.returncode, 0)
                    self.assertIn("enabled braking requires MODEL_KIND=structured", p.stderr)
                    self.assertFalse(Path(self.env["W"]).exists())
        self.assert_no_train()

    def test_invalid_braking_values_refused(self):
        cases = [("ASYM_SPEED_LOSS", ".5"), ("DECEL_WEIGHT", "0"), ("DECEL_THRESHOLD", "0"),
                 ("PROPAGATION_LOSS", "-1"), ("PROP_WINDOW", "0"), ("PROP_MAX_GAP_M", "-2"),
                 ("RESIDUAL_CAP_NEG_MPS2", "0"), ("ASYM_SPEED_LOSS", "nan"),
                 ("DECEL_WEIGHT", "inf"), ("PROP_WINDOW", "abc"),
                 ("BRAKING_FLAGS", "--decel-weight=nan"), ("FLAGS", "--propagation-loss -1")]
        for key, value in cases:
            with self.subTest(key=key, value=value), mock.patch.dict(self.env, {key: value}):
                p = self.invoke()
                self.assertNotEqual(p.returncode, 0)
                self.assertIn("invalid braking contract", p.stderr)
        self.assert_no_train()

    def test_enabled_braking_gate_wiring_with_fixture_smoke(self):
        base = "position,speed,lane,traffic,stability,conservation"
        required = base + ",asym_speed,decel_weight,propagation"
        self.env.update(ASYM_SPEED_LOSS="2", BRAKING_FLAGS="--decel-weight 5 --propagation-loss 1",
                        STUB_SMOKE_PASS="1")
        for present, expected in [(base, "missing loss-share term"), (required, None)]:
            with self.subTest(present=present):
                self.env["W"] = str(self.root / ("missing_braking" if expected else "valid_braking"))
                self.env["STUB_SMOKE_TERMS"] = present
                if self.calls.exists():
                    self.calls.unlink()
                p = self.invoke("run")
                calls = self.calls.read_text()
                w = Path(self.env["W"])
                self.assertIn("--require-terms " + required + " --ramp-end-s 1", calls)
                self.assertIn(str(w) + "/smoke_stages/train_log.jsonl " + required, calls)
                if expected:
                    self.assertIn(expected, p.stderr)
                    self.assertNotIn("wms.plan_steps " + str(w), calls)
                else:
                    # Stop at the planner stub after the actual inline gate passes.
                    self.assertEqual(p.returncode, 18, p.stderr)
                    self.assertIn("wms.plan_steps " + str(w), calls)
                self.assertNotIn("--stage-steps 2500,2500,1875,1250 --out", calls)
                self.assertFalse((w / "full").exists())

    def test_selected_model_and_braking_flags_require_exact_help(self):
        for config, flag in [({"MODEL_KIND": "blackbox_traj"}, "--blackbox-traj"),
                             ({"MODEL_KIND": "blackbox_id"}, "--black-box"),
                             ({"ROLLOUT_MODE": "hybrid"}, "--rollout-mode"),
                             ({"ROLLOUT_MODE": "hybrid", "LANE_HEAD": "history"}, "--lane-head"),
                             ({"ASYM_SPEED_LOSS": "2"}, "--asym-speed-loss"),
                             ({"PROP_MAX_GAP_M": "80"}, "--prop-max-gap-m"),
                             ({"RESIDUAL_CAP_NEG_MPS2": "3"}, "--residual-cap-neg-mps2")]:
            with self.subTest(config=config), mock.patch.dict(self.env, config):
                self.write_stub(missing=flag)
                p = self.invoke()
                self.assertNotEqual(p.returncode, 0)
                self.assertIn("MISSING FLAG " + flag, p.stderr)
        self.assert_no_train()

    def test_required_variable_exact_name(self):
        del self.env["WINDOWS_TRAIN"]
        self.env["WINDOW_TRAIN"] = "wrong-name"
        p = self.invoke()
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("WINDOWS_TRAIN", p.stderr)
        self.assert_no_train()

    def test_missing_sidecar_refuses(self):
        self.env["TRAIN_OUTCOMES"] = str(self.root / "absent")
        p = self.invoke()
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("missing required directory", p.stderr)
        self.assert_no_train()

    def test_runtime_exclusion_flag_required(self):
        self.write_stub(missing="--leave-pi-wall-out")
        p = self.invoke()
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("MISSING FLAG --leave-pi-wall-out", p.stderr)
        self.assert_no_train()

    def test_extra_flags_also_guarded(self):
        self.env["FLAGS"] = "--new-hybrid-extra 0.5"
        p = self.invoke()
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("MISSING FLAG --new-hybrid-extra", p.stderr)
        self.assert_no_train()

    def test_extras_cannot_override_teacher_or_budget(self):
        for key, value in [("FLAGS", "--cav-drive closed_loop"), ("BRAKING_FLAGS", "--stage-steps 1,1,1,1")]:
            with self.subTest(key=key):
                self.env[key] = value
                p = self.invoke()
                self.assertNotEqual(p.returncode, 0)
                self.assertIn("cannot override fixed contract", p.stderr)
                self.env[key] = ""
        self.assert_no_train()

    def test_disk_and_cpu_refusals_before_training(self):
        for key in ["STUB_DISK_FAIL", "STUB_CPU_FAIL"]:
            with self.subTest(key=key):
                self.env[key] = "1"
                p = self.invoke()
                self.assertNotEqual(p.returncode, 0)
                self.env[key] = "0"
                self.assertFalse(Path(self.env["W"]).exists())
        self.assert_no_train()

    def test_nonzero_workers_refused(self):
        self.env["NUM_WORKERS"] = "4"
        p = self.invoke()
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("NUM_WORKERS must be 0", p.stderr)
        self.assert_no_train()

    def test_existing_outputs_and_stage_destinations_preserved(self):
        w = Path(self.env["W"])
        w.mkdir()
        marker = w / "marker"
        marker.write_text("keep", encoding="utf-8")
        p = self.invoke("run")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("output exists", p.stderr)
        self.assertEqual(marker.read_text(), "keep")
        self.env["SW_STAGE_SRC"] = str(self.root)
        p = self.invoke("stage")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("SW_ROOT already exists", p.stderr)
        self.assert_no_train()

    def test_failed_smoke_preserved_without_full_or_retry(self):
        p = self.invoke("run")
        self.assertEqual(p.returncode, 17, p.stderr)
        w = Path(self.env["W"])
        self.assertIn("stub smoke failure", (w / "smoke_stages.log").read_text())
        self.assertIn("FAILED rc=17", (w / "chain.log").read_text())
        self.assertFalse((w / "full").exists())
        self.assertNotIn("wms.plan_steps " + str(w), self.calls.read_text())
        second = self.invoke("run")
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("output exists", second.stderr)

    def test_actual_monitor_refuses_contaminated_timing_and_high_rss(self):
        # Execute the real embedded monitor with mocked /proc and Popen; no
        # trainer or external data access. Verify its termination path as well.
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", CHAIN.read_text(), re.S)
        monitor = next(b for b in blocks if "subprocess.Popen" in b)
        original_read = Path.read_text
        original_iterdir = Path.iterdir
        for reason in ["contention", "memory"]:
            with self.subTest(reason=reason):
                work = self.root / reason
                work.mkdir()
                readings = iter(["cpu 0 0 0 100 0 0 0 0", "cpu 0 0 0 200 0 0 0 0",
                                 "cpu 0 0 0 300 0 0 0 0",
                                 "cpu 100 0 0 300 0 0 0 0" if reason == "contention" else "cpu 0 0 0 400 0 0 0 0"])
                class Child:
                    pid = 123
                    stopped = False

                    def poll(self):
                        return 0 if self.stopped else None

                    def wait(self, timeout=None):
                        self.stopped = True
                        return 0

                def read(path, *args, **kwargs):
                    if str(path).replace("\\", "/") == "/proc/stat":
                        return next(readings)
                    if path.name == "statm":
                        return "99999999 1464844"  # >6,000,000,000 RSS bytes at 4096 B/page
                    return original_read(path, *args, **kwargs)

                def iterdir(path):
                    if str(path).replace("\\", "/") == "/proc":
                        return iter([Path("/proc/123")] if reason == "memory" else [])
                    return original_iterdir(path)

                with mock.patch("sys.argv", ["monitor", str(work), "smoke_stages", "2", "stub-trainer"]), \
                     mock.patch.object(Path, "read_text", read), mock.patch.object(Path, "iterdir", iterdir), \
                     mock.patch("os.cpu_count", return_value=4), \
                     mock.patch("os.getloadavg", return_value=(0, 0, 0), create=True), \
                     mock.patch("os.getpgid", return_value=123, create=True), \
                     mock.patch("os.sysconf", return_value=4096, create=True), \
                     mock.patch("os.killpg", create=True) as kill, \
                     mock.patch("signal.signal"), mock.patch("time.sleep"), \
                     mock.patch("subprocess.Popen", return_value=Child()) as popen:
                    expected = "CPU contention during timing" if reason == "contention" else "RSS reached 6 GB"
                    with self.assertRaisesRegex(RuntimeError, expected):
                        exec(compile(monitor, str(CHAIN), "exec"), {})
                    self.assertEqual(popen.call_count, 1)
                    kill.assert_called_once()
                samples = (work / "smoke_stages_host.jsonl").read_text()
                self.assertIn('"event": "started"', samples)
                self.assertIn('"busy_cores"', samples)


if __name__ == "__main__":
    unittest.main(verbosity=2)
