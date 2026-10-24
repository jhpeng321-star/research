"""No SSH, training or model imports: exercise waiter control and real metric arrays."""
import importlib.util
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "round13c_waiter", Path(__file__).resolve().parents[1] / "scripts/round13c_waiter.py")
W = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(W)


def test_registered_stop_boundaries_and_fail_closed_numbers():
    assert W.stop_reasons(0.02, 0.01, 0.5, 0.4) == []
    assert len(W.stop_reasons(0.02001, 0.01, 0.50001, 0.4)) == 2
    assert W.stop_reasons(0, 0, 0, 0) == []
    assert len(W.stop_reasons(0.001, 0, 0, 0)) == 1
    with pytest.raises(ValueError):
        W.stop_reasons(float("nan"), 0.01, 0.4, 0.4)


def test_proc_stat_handles_parentheses_in_comm(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: "42 (python ) worker) " + " ".join(
        ["S"] + ["0"] * 18 + ["123456"] + ["0"] * 5))
    assert W.proc_stat(42) == (123456, "S")


@pytest.mark.parametrize("ticks,state", [(99, "S"), (100, "Z")])
def test_reused_or_zombie_pid_never_signalled(monkeypatch, ticks, state):
    monkeypatch.setattr(W, "proc_stat", lambda pid: (ticks, state))
    proc = W.ProductionProcess(12345, "/production", 100)
    monkeypatch.setattr(W.os, "pidfd_open", lambda pid: pytest.fail("must not open pidfd"), raising=False)
    assert proc.terminate().startswith("not sent")


def test_signal_requires_exact_production_argv_and_pins_pidfd(monkeypatch):
    monkeypatch.setattr(W, "proc_stat", lambda pid: (100, "S"))
    proc = W.ProductionProcess(12345, "/production", 100)
    argv = [b"python\0-m\0wms\0train\0--mode\0run\0--out\0/smoke\0"]
    monkeypatch.setattr(Path, "read_bytes", lambda *a: argv[0])
    sent = []
    monkeypatch.setattr(W.os, "pidfd_open", lambda pid: 77, raising=False)
    monkeypatch.setattr(W.os, "close", lambda fd: None)
    monkeypatch.setattr(W.signal, "pidfd_send_signal", lambda *a: sent.append(a), raising=False)
    assert proc.terminate().startswith("not sent")
    argv[0] = b"python\0-m\0wms\0train\0--mode\0run\0--out\0/production\0"
    # Windows resolve prefixes this path; use the resolved production name.
    argv[0] = b"python\0-m\0wms\0train\0--mode\0run\0--out\0" + str(proc.run).encode() + b"\0"
    assert proc.terminate().startswith("SIGTERM sent")
    assert sent == [(77, W.signal.SIGTERM)]
    monkeypatch.setattr(W.os, "pidfd_open", lambda pid: (setattr(proc, "start_ticks", 999) or 77), raising=False)
    assert proc.terminate().startswith("not sent")
    assert len(sent) == 1


def test_paired_mae_rejects_different_baseline_rows(tmp_path):
    cols = np.array(["win", "t", "slot", "lane", "rec", "fin"])
    rows = np.array([[0, i, 0, 1, 0, 0.4] for i in range(5)])
    for arm in ("r10b", "r12c"):
        np.savez(tmp_path / f"onestep_{arm}.npz", cols=cols, rows=rows)
    assert W.paired_mae(tmp_path, ("r10b", "r12c"))["r10b"]["mae"] == pytest.approx(0.4)
    rows[0, 1] = 99
    np.savez(tmp_path / "onestep_r12c.npz", cols=cols, rows=rows)
    with pytest.raises(ValueError, match="identical supervision"):
        W.paired_mae(tmp_path, ("r10b", "r12c"))


def test_hard_brakes_use_full_road_velocity_transitions_not_zone_or_commands(tmp_path):
    directory = tmp_path / "r10b"
    directory.mkdir()
    v = np.full((121, 2), 300.0)
    v[1:, 0] -= 2  # exactly one -4 m/s^2 transition per state
    for i in range(24):
        np.savez(directory / f"s{i:03d}_k0.npz", d_v=v, d_lane=np.ones_like(v),
                 d_mask=np.ones_like(v, dtype=bool), d_cav=np.array([False, True]),
                 d_pos=np.full_like(v, -1000), rows=np.zeros((0, 25)), d_acc=np.zeros_like(v))
    result = W.hard_brakes(tmp_path, ("r10b",))["r10b"]
    assert result == {"n": 24 * 120, "a_lt_minus3": 24, "share": 1 / 120}
    (directory / "s000_k0.npz").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        W.hard_brakes(tmp_path, ("r10b",))


def test_provenance_rejects_eval_seeds_and_restricted_references():
    W.validate_seed(963959)
    with pytest.raises(ValueError):
        W.validate_seed(964000)
    with pytest.raises(ValueError):
        W.check_strings({"source": "/data/" + "sealed" + "/x"})
    with pytest.raises(ValueError):
        W.check_strings({"seed": 964199})


@pytest.mark.parametrize("mode,decision,signals", [("midrun", "STOP", 1), ("postrun", "ERROR", 0)])
def test_measurement_error_is_recorded_and_only_midrun_stops(tmp_path, monkeypatch, mode, decision, signals):
    # Fake only Linux locking and process interaction, keeping actual report persistence.
    monkeypatch.setitem(__import__("sys").modules, "fcntl", SimpleNamespace(LOCK_EX=2, flock=lambda *a: None))
    waiter = object.__new__(W.Waiter)
    waiter.mode, waiter.out, waiter.work = mode, tmp_path / "checks" / mode, tmp_path / "shm"
    sent = []
    waiter.process = SimpleNamespace(pid=123, start_ticks=456, terminate=lambda: (sent.append(123) or "sent"))
    waiter.commands = []
    waiter.wait_checkpoint = lambda: tmp_path / "checkpoint.pt"
    waiter.measure = lambda ck: (_ for _ in ()).throw(RuntimeError("broken measurement"))
    assert waiter.execute() == 1
    report = json.loads((waiter.out / "RESULT.json").read_text())
    assert report["decision"] == decision
    assert "broken measurement" in report["reasons"][0]
    assert len(sent) == signals
    assert (waiter.out / "RESULT.md").is_file()


def test_postrun_requires_summary_and_original_process_gone(tmp_path, monkeypatch):
    waiter = object.__new__(W.Waiter)
    waiter.mode, waiter.run = "postrun", tmp_path
    (tmp_path / "RUN_SUMMARY.json").write_text("{}")
    checkpoint = tmp_path / "wm_s.pt"
    checkpoint.write_bytes(b"complete")
    states = iter([True, False])
    waiter.process = SimpleNamespace(alive=lambda: next(states))
    sleeps = []
    monkeypatch.setattr(W.time, "sleep", lambda seconds: sleeps.append(seconds))
    assert waiter.wait_checkpoint() == checkpoint
    assert sleeps == [W.POLL]


def test_panel_records_internal_val_train_provenance_and_rejects_unmatched_state(tmp_path):
    states = []
    for si, source in enumerate(W.SOURCES):
        directory = tmp_path / "data" / source
        directory.mkdir(parents=True)
        entries = []
        for i in range(96):
            seed = 963000 + 480 * si + i
            episode = f"w{seed}__native"
            entries.append({"episode": episode, "seed": seed, "split_role": "val",
                            "dir": f"episodes/{episode}"})
            states.append({"key": f"state{seed}", "source": source, "episode": episode,
                           "seed": seed, "t0_s": 450, "dt_s": 0.5,
                           "recorded_horizon_steps": {"native": 240}})
        (directory / "C4_EPISODES_MANIFEST.json").write_text(json.dumps({"episodes": entries}))
    source_path = tmp_path / "chk_r10b" / "states192.json.gz"
    source_path.parent.mkdir()
    work = tmp_path / "shm"
    work.mkdir()
    def save():
        with gzip.open(source_path, "wt") as stream:
            json.dump(states, stream)
    save()
    panel, receipt = W.train_panel(tmp_path, work)
    with gzip.open(panel, "rt") as stream:
        assert len(json.load(stream)) == 24
    assert len(receipt["states"]) == 24
    assert {r["split_role"] for r in receipt["states"]} == {"val"}
    states[0]["episode"] = "missing"
    save()
    with pytest.raises(ValueError, match="not matched"):
        W.train_panel(tmp_path, work)
    states[1]["seed"] = 964000  # unselected rows are checked too
    save()
    with pytest.raises(ValueError, match="evaluation seed"):
        W.train_panel(tmp_path, work)


def test_wait_does_not_hold_measurement_lock(tmp_path, monkeypatch):
    events = []
    monkeypatch.setitem(__import__("sys").modules, "fcntl", SimpleNamespace(
        LOCK_EX=2, flock=lambda *a: events.append("lock")))
    waiter = object.__new__(W.Waiter)
    waiter.mode, waiter.out, waiter.work = "postrun", tmp_path / "checks/postrun", tmp_path / "shm"
    waiter.process = SimpleNamespace(pid=123, start_ticks=456)
    waiter.commands = []
    waiter.wait_checkpoint = lambda: (events.append("wait") or tmp_path / "wm_s.pt")
    waiter.measure = lambda ck: (events.append("measure") or {"decision": "CONTINUE", "reasons": []})
    assert waiter.execute() == 0
    assert events == ["wait", "lock", "measure"]


@pytest.mark.parametrize("mode", ["midrun", "postrun"])
def test_measurement_orchestration_preserves_paired_arms_and_postrun_headzero(tmp_path, monkeypatch, mode):
    waiter = object.__new__(W.Waiter)
    waiter.mode, waiter.repo, waiter.root = mode, tmp_path / "repo", tmp_path
    waiter.work, waiter.out = tmp_path / "shm", tmp_path / "persistent"
    waiter.out.mkdir()
    panel = tmp_path / "verified_states24.json.gz"
    monkeypatch.setattr(W, "train_panel", lambda root, work: (panel, {"verified": True}))
    def snapshot(src, name):
        return src, {"source": str(src), "sha256": name,
                     "emissions": dict.fromkeys(W.EMISSIONS, name == "baseline")}
    waiter.snapshot = snapshot
    commands = []
    def command(script, *args):
        args = list(map(str, args))
        commands.append((script.name, args))
        def option(name):
            return args[args.index(name) + 1]
        if script.name == "make_headzero.py":
            Path(args[2]).write_text(json.dumps({"all_other_tensors_equal": True, "zeroed_are_zero": True}))
        elif script.name == "run_onestep.py":
            directory = Path(option("--out"))
            directory.mkdir()
            (directory / "ONESTEP_DONE.json").write_text(json.dumps({
                "replay_present": "filled", "base_cols": True, "windows": 24 if mode == "midrun" else 240}))
            for arm in ("r10b", "r12c", "r9", "headzero"):
                rows = [[0, i, 0, 1, 0, 0.4 if arm == "r10b" else 0.35] for i in range(5)]
                np.savez(directory / f"onestep_{arm}.npz", rows=np.array(rows),
                         cols=np.array(["win", "t", "slot", "lane", "rec", "fin"]))
        elif script.name == "base_calibration.py":
            Path(option("--json")).write_text(json.dumps({
                "replay_present": "filled", "verdict": "REJECT context only",
                "rules": {k: ["PASS", ""] for k in ("BASE-1", "BASE-2", "BASE-3", "BASE-5")},
                "classes": {a: {} for a in ("r10b", "r9", "headzero")}}))
        elif script.name == "midrun_check.py":
            Path(option("--out-json")).write_text(json.dumps({"decision": "STOP"}))
        elif script.name == "d11_run5.py":
            directory = Path(option("--out"))
            directory.mkdir()
            (directory / "ROLLOUT_DONE.json").write_text(json.dumps({"n_states": 24, "k": 1, "steps": 120}))
            for arm in ("r10b", "r12c", "r9"):
                (directory / arm).mkdir()
                for i in range(24):
                    np.savez(directory / arm / f"s{i:03d}_k0.npz", d_v=np.ones((121, 1)),
                             d_lane=np.ones((121, 1)), d_mask=np.ones((121, 1), dtype=bool),
                             d_cav=np.array([False]))
        else:
            pytest.fail(f"unexpected command {script}")
    waiter.command = command
    result = waiter.measure(tmp_path / "candidate.pt")
    assert result["decision"] == "CONTINUE"  # legacy STOP does not replace paired R13C rules
    assert result["one_step"]["r12c"]["mae"] == pytest.approx(0.35)
    one_step = next(args for name, args in commands if name == "run_onestep.py")
    for key in W.EMISSIONS:
        assert f"r12c=model.{key}=true" in one_step
        assert f"r9=model.{key}=false" in one_step
    assert "r9=model.driver_param_mode=legacy" in one_step
    assert "r12c=model.driver_param_mode=physical" in one_step
    assert "--limit" in one_step if mode == "midrun" else "--limit" not in one_step
    assert any(name == "make_headzero.py" for name, args in commands) == (mode == "postrun")
    rollout = next(args for name, args in commands if name == "d11_run5.py")
    assert rollout[rollout.index("--states") + 1] == str(panel)
    assert rollout[rollout.index("--steps") + 1] == "120"
    assert all(int(args[args.index("--workers") + 1]) <= 12
               for name, args in commands if "--workers" in args)


def test_stable_snapshot_retries_and_keeps_checkpoint_physics(tmp_path, monkeypatch):
    waiter = object.__new__(W.Waiter)
    waiter.work = tmp_path / "shm"
    waiter.work.mkdir()
    src = tmp_path / "wm_s_stage120.pt"
    src.write_bytes(b"checkpoint")
    (tmp_path / "physics.json").write_text('{"merge_end_wait": false}')
    attempts = []
    def load(path, **kwargs):
        attempts.append(path)
        if len(attempts) == 1:
            raise EOFError("writer not finished")
        return {"format": "wm_s/1", "model": {"x": 1}, "config": {"temporal_accel_off": True}}
    monkeypatch.setitem(__import__("sys").modules, "torch", SimpleNamespace(
        set_num_threads=lambda n: None, load=load))
    monkeypatch.setattr(W.time, "sleep", lambda seconds: None)
    checkpoint, receipt = waiter.snapshot(src, "candidate")
    assert len(attempts) == 2
    assert checkpoint.read_bytes() == b"checkpoint"
    assert (checkpoint.parent / "physics.json").read_bytes() == (tmp_path / "physics.json").read_bytes()
    assert receipt["emissions"]["temporal_accel_off"] is True
    assert receipt["sha256"] == W.digest(src)
