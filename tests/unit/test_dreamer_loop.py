"""scripts/c4/dreamer_loop.py: the C4 Dreamer-loop state machine with a fake backend (no ssh, no SUMO).

The fake simulates the three hosts on the local tmp filesystem with a fake clock: a SUMO collection
that writes complete raw episodes, a WM refresh that writes logs/checkpoints/RUN_SUMMARY, and a policy
trainer that writes ``policy_<n>.pt`` every 5 updates and honours the STOP file.
"""
from __future__ import annotations

import importlib.util
import json
import shlex
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("dreamer_loop", REPO / "scripts" / "c4" / "dreamer_loop.py")
DL = importlib.util.module_from_spec(_spec)
sys.modules["dreamer_loop"] = DL
_spec.loader.exec_module(DL)


def _arg(argv, flag):
    return argv[argv.index(flag) + 1]


def _convert(tokens):
    """Fake ``convert_cmd``: ``python -B -c CODE LIST DEST [> log 2>&1]``."""
    if ">" in tokens:
        tokens = tokens[:tokens.index(">")]
    listing, dest = Path(tokens[-2]), Path(tokens[-1])
    dest.mkdir(parents=True)
    (dest / "C4_EPISODES_MANIFEST.json").write_text(json.dumps({"episodes": listing.read_text().split()}))


class FakeBackend:
    def __init__(self, tmp):
        self.tmp = tmp
        self.t = 1_000_000.0
        self.procs = {}
        self.next_pid = 5000
        self.mtimes = {}
        self.signals = []
        self.launches = []
        self.commands = []
        self.free = 50.0
        self.busy = []
        self.wm_val = 10.0
        self.wm_loss = 1.0
        self.collect_s = 100
        self.log = None

    # -- clock
    def now(self):
        return self.t

    def sleep(self, seconds):
        step = max(1.0, float(seconds))
        self.t += step
        for proc in list(self.procs.values()):
            if proc["alive"]:
                self._tick(proc)

    def free_gb(self, path):
        return self.free

    # -- files
    def run(self, host, cmd, timeout=300, check=False):
        self.commands.append((host.get("name"), cmd))
        if "wms.data.c4_episodes" in cmd:
            _convert(shlex.split(cmd.split("&&")[-1]))
            return 0, "", ""
        for part in cmd.split("&&"):
            words = shlex.split(part.strip()) if part.strip() else []
            if words[:1] == ["touch"]:
                Path(words[1]).touch()
            elif words[:2] == ["mkdir", "-p"]:
                Path(words[2]).mkdir(parents=True, exist_ok=True)
        return 0, "", ""

    def exists(self, host, path):
        return Path(path).exists()

    def read_text(self, host, path, tail_bytes=0):
        try:
            return Path(path).read_text()
        except OSError:
            return None

    def list_ckpts(self, host, run_dir):
        rows = [(self.mtimes.get(str(p), 0.0), p.stat().st_size, str(p)) for p in Path(run_dir).glob("policy_*.pt")]
        return sorted(rows, reverse=True)

    def copy(self, src_host, src, dst_host, dst):
        if src.endswith("/"):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    def pgrep(self, host, pattern):
        return list(self.busy)

    # -- processes
    def add_policy(self, run_dir, n=52, pid=None):
        pid = pid or self._pid()
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        self.procs[pid] = {"pid": pid, "start": "77", "alive": True, "kind": "policy", "run_dir": run_dir, "n": n,
                           "next": self.t + 20, "cmdline": f"python -m wms.control.train_cli --output-dir {run_dir}",
                           "env": {}, "host": "policy"}
        return pid

    def _pid(self):
        self.next_pid += 1
        return self.next_pid

    def launch(self, host, cwd, cmd, log_path, exit_path, env):
        pid = self._pid()
        argv = shlex.split(cmd)
        proc = {"pid": pid, "start": str(pid * 10), "alive": True, "cmdline": "bash -c " + cmd, "env": dict(env),
                "exit_path": exit_path, "host": host.get("name"), "argv": argv, "t0": self.t}
        if "traffic_shepherd.c4.collect" in argv:
            proc.update(kind="collect", out=_arg(argv, "--out"), seeds=_arg(argv, "--seeds"))
        elif "wms.data.c4_episodes" in cmd:
            proc.update(kind="convert")
        elif "wms" in argv and "train" in argv:
            proc.update(kind="wm", out=_arg(argv, "--out"))
        elif "wms.control.train_cli" in argv:
            proc.update(kind="policy", run_dir=_arg(argv, "--output-dir"), n=int(self.resume_n(argv)),
                        next=self.t + 20)
        self.procs[pid] = proc
        self.launches.append(proc)
        return {"pid": pid, "start": proc["start"]}

    def resume_n(self, argv):
        name = Path(_arg(argv, "--resume")).name
        return name[7:-3] if name[7:-3].isdigit() else 60

    def proc_info(self, host, pid):
        proc = self.procs.get(pid)
        if not proc or not proc["alive"]:
            return None
        return {"state": "S", "pgid": pid, "start": proc["start"], "cmdline": proc["cmdline"]}

    def alive(self, host, proc):
        info = self.proc_info(host, proc["pid"])
        return bool(info) and proc.get("start") in (None, info["start"])

    def signal(self, host, proc, sig, marker, group):
        info = self.proc_info(host, proc["pid"])
        if not info:
            return False
        if proc.get("start") and info["start"] != proc["start"]:
            raise RuntimeError("reused")
        if marker and marker not in info["cmdline"]:
            raise RuntimeError("marker")
        self.signals.append((proc["pid"], sig, group))
        self.procs[proc["pid"]]["alive"] = False
        return True

    # -- simulated jobs
    def _tick(self, proc):
        kind = proc.get("kind")
        if kind == "convert" and self.t - proc["t0"] >= 30:
            _convert(proc["argv"])
            Path(proc["exit_path"]).write_text("0\n")
            proc["alive"] = False
        elif kind == "collect" and self.t - proc["t0"] >= self.collect_s:
            for seed in DL.expand([[int(x) for x in r.split("-")] if "-" in r else [int(r)] * 2
                                   for r in proc["seeds"].split(",")]):
                d = Path(proc["out"]) / "episodes" / "pol" / f"s{seed}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "episode.npz").write_bytes(b"x")
                (d / "manifest.json").write_text(json.dumps({"status": "complete"}))
                (d / "summary.json").write_text("{}")
            Path(proc["exit_path"]).write_text("0\n")
            proc["alive"] = False
        elif kind == "wm" and self.t - proc["t0"] >= 200:
            out = Path(proc["out"])
            out.mkdir(parents=True, exist_ok=True)
            (out / "train_log.jsonl").write_text(json.dumps({"step": 1, "loss": self.wm_loss}) + "\n")
            (out / "val_log.jsonl").write_text("".join(
                json.dumps({"step": 10, "horizon_s": h, "mean_position": self.wm_val}) + "\n" for h in (10.0, 60.0)))
            for name in ("wm_s_best.pt", "physics.json", "TRAIN_RECEIPT.json", "RUN_SUMMARY.json"):
                (out / name).write_text("{}")
            Path(proc["exit_path"]).write_text("0\n")
            proc["alive"] = False
        elif kind == "policy":
            run = Path(proc["run_dir"])
            while proc["alive"] and self.t >= proc["next"]:
                if (run / "STOP").exists():
                    self._ckpt(run / "policy_latest.pt")
                    proc["alive"] = False
                    break
                proc["n"] += 1
                proc["next"] += 20
                with (run / "updates.jsonl").open("a") as f:
                    f.write(json.dumps({"update": proc["n"]}) + "\n")
                if proc["n"] % 5 == 0:
                    self._ckpt(run / f"policy_{proc['n']:06d}.pt")

    def _ckpt(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ckpt")
        self.mtimes[str(path)] = self.t


def _config(tmp, **over):
    cfg = {
        "loop_root": str(tmp / "loop"), "data_root": str(tmp / "data"), "wm_runs_root": str(tmp / "wmruns"),
        "max_iterations": 2, "poll_s": 10, "min_free_gb": 5.0,
        "hosts": {"wm": {"code_root": str(tmp / "repo"), "python": "python"},
                  "policy": {"code_root": str(tmp / "repo"), "python": "python", "work_root": str(tmp / "pol")},
                  "cpu": {"code_root": str(tmp / "boxrepo"), "python": "py", "work_root": str(tmp / "box"),
                          "ssh": ["-i", "k", "u@box"]}},
        "seeds": {"per_iteration": 4, "exclude": [[965186, 965186]]},
        "policy": {"env": {"WMS_DATA_V5_NATIVE": "/d/v5"}, "start_check_s": 30},
    }
    for key, value in over.items():
        cfg[key] = DL.merge(cfg[key], value) if isinstance(value, dict) and isinstance(cfg.get(key), dict) else value
    for d in ("data", "wmruns", "repo"):
        (tmp / d).mkdir(exist_ok=True)
    path = tmp / "loop.json"
    path.write_text(json.dumps(cfg))
    return DL.load_config(path)


def _setup(tmp, **over):
    cfg = _config(tmp, **over)
    fake = FakeBackend(tmp)
    init_wm = tmp / "wm0" / "wm_s_best.pt"
    init_wm.parent.mkdir()
    init_wm.write_text("wm0")
    (init_wm.parent / "val_log.jsonl").write_text("".join(
        json.dumps({"step": 5, "horizon_s": h, "mean_position": 10.0}) + "\n" for h in (10.0, 60.0)))
    run_dir = str(tmp / "pol" / "runs" / "v6")
    pid = fake.add_policy(run_dir)
    fake._ckpt(Path(run_dir) / "policy_000050.pt")
    fake.t += 100
    loop = DL.Loop(cfg, backend=fake, out=lambda *_: None)
    loop.init_state(str(init_wm), str(init_wm.parent / "val_log.jsonl"), run_dir, pid)
    return cfg, fake, loop, pid


def test_one_full_iteration_with_adopted_policy(tmp_path):
    cfg, fake, loop, pid = _setup(tmp_path, max_iterations=1)
    assert loop.run() == 0
    s = loop.state
    assert s["status"] == "finished" and s["iteration"] == 2
    it = s["iters"]["1"]
    assert it["seeds"] == [965184, 965185, 965187, 965188]           # exclusion respected
    assert DL.seed_spec(it["seeds"]) == "965184-965185,965187-965188"
    ledger = [json.loads(x) for x in (tmp_path / "loop" / "seed_ledger.jsonl").read_text().splitlines()]
    assert ledger[0]["seeds"] == "965184-965185,965187-965188" and ledger[0]["n"] == 4
    assert it["pulled"]["complete_episodes"] == 4 and it["converted"]["episodes"] == 4
    assert Path(it["converted"]["dir"]).name == "v6_policy_iter001"
    assert it["wm_verdict"]["accepted"] and Path(s["wm"]["ckpt"]).parent.name == "c4_wm_loop_iter001"
    # adopted trainer: periodic checkpoint + one more update, then SIGTERM of exactly that PID (no group)
    assert fake.signals == [(pid, "TERM", False)]
    assert it["policy_stop"]["how"] == "adopted_checkpoint_then_sigterm"
    assert it["policy_stop"]["resume"].endswith(".pt") and "policy_0000" in it["policy_stop"]["resume"]
    pol = s["policy"]
    assert pol["owned"] and pol["stop_file"] and pol["run_dir"].endswith("runs/c4_policy_loop_iter001")
    launch = [p for p in fake.launches if p["kind"] == "policy"][-1]
    assert launch["env"]["WMS_LOOP_REFRESH"] == "1"
    assert launch["env"]["WMS_DATA_V6_POLICY"] == it["converted"]["dir"]
    assert launch["env"]["WMS_DATA_V5_NATIVE"] == "/d/v5"
    assert _arg(launch["argv"], "--config") == "configs/c4_policy_v6_loop.toml"
    assert _arg(launch["argv"], "--resume") == it["policy_stop"]["resume"]
    assert _arg(launch["argv"], "--wm-ckpt") == s["wm"]["ckpt"]
    wm_launch = [p for p in fake.launches if p["kind"] == "wm"][0]
    assert "model.init_from=" + str(tmp_path / "wm0" / "wm_s_best.pt") in wm_launch["argv"]
    assert "curriculum.stage_steps=[60,40]" in wm_launch["argv"]
    assert wm_launch["env"]["WMS_DATA_V6_POLICY"] == it["converted"]["dir"]
    col = [p for p in fake.launches if p["kind"] == "collect"][0]
    assert "--no-physics-log" in col["argv"] and _arg(col["argv"], "--policy").endswith("iter_001/policy_snapshot.pt")
    events = [json.loads(x)["event"] for x in (tmp_path / "loop" / "loop_log.jsonl").read_text().splitlines()]
    assert events[0] == "init" and "wm_accepted" in events and events[-1] == "finished"


def test_second_iteration_uses_stop_file_and_deletes_superseded_converted(tmp_path):
    cfg, fake, loop, _ = _setup(tmp_path, max_iterations=2)
    assert loop.run() == 0
    s = loop.state
    assert s["iteration"] == 3
    it2 = s["iters"]["2"]
    assert it2["seeds"] == [965189, 965190, 965191, 965192]
    assert it2["policy_stop"]["how"] == "stop_file" and it2["policy_stop"]["resume"].endswith("policy_latest.pt")
    assert len(fake.signals) == 1                       # the second stop needed no signal
    assert it2["converted"]["episodes"] == 8            # all policy episodes so far
    data = sorted(p.name for p in (tmp_path / "data").iterdir())
    assert data == ["v6_policy_iter002"]                # iter001 deleted once the policy moved on
    assert Path(s["wm"]["ckpt"]).parent.name == "c4_wm_loop_iter002"
    wm2 = [p for p in fake.launches if p["kind"] == "wm"][1]
    assert "model.init_from=" + str(Path(s["iters"]["1"]["wm"]["out"]) / "wm_s_best.pt") in wm2["argv"]


def test_crash_resume_continues_without_repeating_steps(tmp_path):
    cfg, fake, loop, _ = _setup(tmp_path, max_iterations=1)
    assert loop.run(max_steps=3) == 0                   # snapshot, collect_launch, collect_wait
    assert loop.state["step"] == "pull"
    loop2 = DL.Loop(cfg, backend=fake, out=lambda *_: None)
    assert loop2.load()
    assert loop2.run() == 0
    assert loop2.state["status"] == "finished"
    assert len([p for p in fake.launches if p["kind"] == "collect"]) == 1
    assert len([p for p in fake.launches if p["kind"] == "wm"]) == 1
    assert loop2.state["iters"]["1"]["seeds"] == [965184, 965185, 965187, 965188]


def test_stop_file_exits_and_keeps_step(tmp_path):
    cfg, fake, loop, _ = _setup(tmp_path)
    assert loop.run(max_steps=2) == 0
    (tmp_path / "loop" / "STOP").write_text("x")
    assert loop.run() == 0
    assert loop.state["status"] == "stopped" and loop.state["step"] == "collect_wait"
    collect = loop.state["iters"]["1"]["collect"]["proc"]["pid"]
    assert fake.procs[collect]["alive"]                 # detached job untouched


def test_rejected_wm_keeps_previous_and_policy_untouched(tmp_path):
    cfg, fake, loop, pid = _setup(tmp_path, max_iterations=1)
    fake.wm_val = 20.0                                  # 2x worse than the reference 10.0
    assert loop.run() == 0
    s = loop.state
    assert not s["iters"]["1"]["wm_verdict"]["accepted"]
    assert s["wm"]["ckpt"] == str(tmp_path / "wm0" / "wm_s_best.pt")
    assert fake.signals == [] and fake.procs[pid]["alive"]
    assert s["policy"]["proc"]["pid"] == pid


def test_nonfinite_loss_rejected(tmp_path):
    cfg, fake, loop, _ = _setup(tmp_path, max_iterations=1)
    fake.wm_loss = float("nan")
    assert loop.run() == 0
    assert "nonfinite loss" in loop.state["iters"]["1"]["wm_verdict"]["why"]


def test_disk_guard_pauses_then_resumes(tmp_path):
    cfg, fake, loop, _ = _setup(tmp_path, max_iterations=1)
    fake.free = 1.0
    real_sleep = fake.sleep
    calls = {"n": 0}

    def sleep(seconds):
        calls["n"] += 1
        if calls["n"] == 30:
            fake.free = 50.0
        real_sleep(seconds)

    fake.sleep = sleep
    assert loop.run() == 0
    events = [json.loads(x)["event"] for x in (tmp_path / "loop" / "loop_log.jsonl").read_text().splitlines()]
    assert "disk_pause" in events and "disk_ok" in events and loop.state["status"] == "finished"


def test_busy_cpu_box_delays_collection(tmp_path):
    cfg, fake, loop, _ = _setup(tmp_path, collect={"wait_for_absent": ["scripts/boxops/eval_v5.sh"]})
    fake.busy = ["123 bash scripts/boxops/eval_v5.sh x"]
    loop.run(max_steps=3)
    assert loop.state["step"] == "collect_launch" and not loop.state["iters"]["1"].get("seeds")
    fake.busy = []
    loop.run(max_steps=1)
    assert loop.state["step"] == "collect_wait"


def test_collect_failure_retries_same_seeds_then_fails(tmp_path):
    cfg, fake, loop, _ = _setup(tmp_path, collect={"max_attempts": 2})
    orig = fake._tick

    def tick(proc):
        if proc.get("kind") == "collect" and fake.t - proc["t0"] >= 50:
            Path(proc["exit_path"]).write_text("1\n")
            proc["alive"] = False
            return
        orig(proc)

    fake._tick = tick
    assert loop.run() == 2
    cols = [p for p in fake.launches if p["kind"] == "collect"]
    assert len(cols) == 2 and cols[0]["seeds"] == cols[1]["seeds"]
    assert loop.state["status"] == "failed"


def test_seed_guards():
    with pytest.raises(ValueError):
        DL.check_seeds([964000])
    with pytest.raises(ValueError):
        DL.check_seeds([965183])
    DL.check_seeds([965184, 965999])
    assert DL.seed_spec([965990, 965991, 965995]) == "965990-965991,965995"


def test_config_refuses_bad_block_and_restricted_paths(tmp_path):
    with pytest.raises(ValueError, match="outside the approved"):
        _config(tmp_path, seeds={"block": [965000, 965300]})
    with pytest.raises(ValueError, match="restricted"):
        _config(tmp_path, data_root=str(tmp_path / "holdout_data"))


def test_seed_block_exhaustion_finishes(tmp_path):
    cfg, fake, loop, _ = _setup(tmp_path, seeds={"block": [965184, 965187], "per_iteration": 4},
                                max_iterations=5)
    assert loop.run() == 0
    assert loop.state["status"] == "finished" and loop.state["iteration"] == 2
    assert loop.state["iters"]["1"]["seeds"] == [965184, 965185, 965187]


def test_wm_sanity_ratio(tmp_path):
    t, v, r = tmp_path / "t.jsonl", tmp_path / "v.jsonl", tmp_path / "r.jsonl"
    t.write_text('{"step": 1, "loss": 2.0}\n')
    v.write_text('{"step": 3, "horizon_s": 60.0, "mean_position": 12.0}\n'
                 '{"step": 3, "horizon_s": 60.0, "mean_position": 99.0, "revalidated": true}\n')
    r.write_text('{"step": 1, "horizon_s": 60.0, "mean_position": 50.0}\n'
                 '{"step": 9, "horizon_s": 60.0, "mean_position": 10.0}\n')
    ok = DL.wm_sanity(t, v, r, "mean_position", 0.3)
    assert ok["ok"] and abs(ok["compare"]["60"]["ratio"] - 1.2) < 1e-9
    assert not DL.wm_sanity(t, v, r, "mean_position", 0.1)["ok"]
    assert DL.wm_sanity(t, v, None, "mean_position", 0.1)["ok"]


def test_real_backend_signal_refuses_foreign_or_reused_pid(monkeypatch):
    b = DL.Backend(retries=1)
    sent = []
    monkeypatch.setattr(b, "run", lambda host, cmd, **k: (sent.append(cmd), (0, "", ""))[1])
    info = {"state": "S", "pgid": 10, "start": "500", "cmdline": "python -m wms.control.train_cli --output-dir /x"}
    monkeypatch.setattr(b, "proc_info", lambda host, pid: dict(info))
    with pytest.raises(RuntimeError, match="reused"):
        b.signal({}, {"pid": 10, "start": "499"}, "TERM", "", group=True)
    with pytest.raises(RuntimeError, match="lacks"):
        b.signal({}, {"pid": 10, "start": "500"}, "TERM", "something_else", group=False)
    with pytest.raises(RuntimeError, match="group leader"):
        b.signal({}, {"pid": 11, "start": "500"}, "TERM", "", group=True)
    assert sent == []
    assert b.signal({}, {"pid": 10, "start": "500"}, "TERM", "wms.control.train_cli", group=False)
    assert b.signal({}, {"pid": 10, "start": "500"}, "TERM", "", group=True)
    assert sent == ["kill -TERM 10", "kill -TERM -- -10"]


def test_real_backend_proc_info_parses_proc_stat(monkeypatch):
    b = DL.Backend(retries=1)
    stat = "123 (python -m x) S 1 123 123 0 -1 4194560 " + " ".join(["0"] * 12) + " 98765 rest"
    monkeypatch.setattr(b, "run", lambda host, cmd, **k: (0, stat + "\n@@python -m wms.control.train_cli ", ""))
    info = b.proc_info({}, 123)
    assert info == {"state": "S", "pgid": 123, "start": "98765", "cmdline": "python -m wms.control.train_cli"}


def test_dry_run_prints_plan(tmp_path, capsys):
    _config(tmp_path)
    path = tmp_path / "loop.json"
    flags = ["--config", str(path), "--init-wm", "/w/wm.pt", "--init-policy-run-dir", "/p/run",
             "--adopt-policy-pid", "42"]
    assert DL.main(["dry-run", "--sequential", *flags]) == 0
    text = capsys.readouterr().out
    assert "965184-965185,965187-965188" in text and "traffic_shepherd.c4.collect" in text
    assert "adopted pid 42" in text and not (tmp_path / "loop" / "state.json").exists()
    assert DL.main(["dry-run", *flags]) == 0
    text = capsys.readouterr().out
    assert text.startswith("PIPELINED") and "965184-965185,965187-965234" in text and "adopted pid 42" in text


# ============================================================ pipelined mode
def _pipe(tmp, **over):
    hosts = over.pop("hosts", ["cpu", "cpu2"])
    over.setdefault("mode", "pipelined")
    cfg = _config(tmp, **over)
    cfg["hosts"]["cpu2"] = dict(cfg["hosts"]["cpu"], work_root=str(tmp / "box2"))
    cfg = DL.load_config_dict({**cfg, "collect": {**cfg["collect"], "hosts": hosts}})
    fake = FakeBackend(tmp)
    init_wm = tmp / "wm0" / "wm_s_best.pt"
    init_wm.parent.mkdir()
    init_wm.write_text("wm0")
    (init_wm.parent / "val_log.jsonl").write_text("".join(
        json.dumps({"step": 5, "horizon_s": h, "mean_position": 10.0}) + "\n" for h in (10.0, 60.0)))
    run_dir = str(tmp / "pol" / "runs" / "v6")
    pid = fake.add_policy(run_dir)
    fake._ckpt(Path(run_dir) / "policy_000050.pt")
    fake.t += 100
    loop = DL.PipeLoop(cfg, backend=fake, out=lambda *_: None)
    loop.init_state(str(init_wm), str(init_wm.parent / "val_log.jsonl"), run_dir, pid, str(tmp / "cal" / "ap.pt"))
    return cfg, fake, loop, pid


def test_pipelined_collects_refreshes_and_switches_until_seeds_run_out(tmp_path):
    cfg, fake, loop, pid = _pipe(tmp_path, seeds={"block": [965184, 965203], "per_iteration": 1},
                                 collect={"batch_size": 3})
    assert loop.run() == 0
    s = loop.state
    assert s["status"] == "finished"
    batches = [b for b in s["batches"].values()]
    seeds = [x for b in batches for x in b["seeds"]]
    assert len(seeds) == len(set(seeds))                               # disjoint across hosts and batches
    assert set(seeds) == set(range(965184, 965204)) - {965186}         # whole block minus exclusion
    assert {b["host"] for b in batches} == {"cpu", "cpu2"}
    assert all(b["status"] == "pooled" for b in batches)
    ledger = (tmp_path / "loop" / "seed_ledger.jsonl").read_text().splitlines()
    assert len(ledger) == len(batches)
    refreshes = list(s["refreshes"].values())
    assert len(refreshes) >= 2 and all(r["status"] == "accepted" for r in refreshes)
    assert refreshes[-1]["pool_episodes"] == 19
    # refreshes chain: each starts from the previous accepted WM
    assert refreshes[1]["init_from"] == str(Path(refreshes[0]["out"]) / "wm_s_best.pt")
    # first switch: the adopted trainer got exactly one SIGTERM; later switches used the STOP file
    assert fake.signals == [(pid, "TERM", False)]
    switches = s["policy_switches"]
    assert switches[0]["how"] == "adopted" and all(x["how"] == "stop_file" for x in switches[1:])
    pol_launches = [p for p in fake.launches if p["kind"] == "policy"]
    assert pol_launches and all(p["env"]["WMS_LOOP_REFRESH"] == "1" for p in pol_launches)
    last = pol_launches[-1]
    assert last["env"]["WMS_DATA_V6_POLICY"] == refreshes[-1]["converted"]
    assert _arg(last["argv"], "--wm-ckpt") == s["wm"]["policy_ckpt"]
    # only the converted pool the policy reads survives
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == [Path(refreshes[-1]["converted"]).name]
    # each batch used the newest policy checkpoint at its start
    assert len({b["snapshot"]["source"] for b in batches}) > 1


def test_pipelined_resume_after_crash(tmp_path):
    cfg, fake, loop, _ = _pipe(tmp_path, seeds={"block": [965184, 965195]}, collect={"batch_size": 3})
    assert loop.run(max_steps=12) == 0
    mid = json.loads((tmp_path / "loop" / "state.json").read_text())
    assert mid["batches"] and mid["status"] == "running"
    loop2 = DL.PipeLoop(cfg, backend=fake, out=lambda *_: None)
    assert loop2.load() and loop2.run() == 0
    s = loop2.state
    seeds = [x for b in s["batches"].values() for x in b["seeds"]]
    assert len(seeds) == len(set(seeds)) == 11
    assert len([p for p in fake.launches if p["kind"] == "collect"]) == len(s["batches"])


def test_pipelined_rejected_refresh_keeps_policy(tmp_path):
    cfg, fake, loop, pid = _pipe(tmp_path, seeds={"block": [965184, 965189]}, collect={"batch_size": 3},
                                 hosts=["cpu"])
    fake.wm_val = 99.0
    assert loop.run() == 0
    s = loop.state
    assert all(r["status"] == "rejected" for r in s["refreshes"].values())
    assert fake.signals == [] and fake.procs[pid]["alive"] and not s.get("policy_switches")
    assert s["wm"]["ckpt"] == str(tmp_path / "wm0" / "wm_s_best.pt")


def test_pipelined_stop_file(tmp_path):
    cfg, fake, loop, _ = _pipe(tmp_path)
    assert loop.run(max_steps=3) == 0
    (tmp_path / "loop" / "STOP").write_text("x")
    assert loop.run() == 0 and loop.state["status"] == "stopped"
    running = [b for b in loop.state["batches"].values() if b["status"] == "running"]
    assert running and all(fake.procs[b["proc"]["pid"]]["alive"] for b in running)


def test_mode_flag_and_state_mismatch(tmp_path, capsys):
    cfg, fake, loop, _ = _pipe(tmp_path)
    path = tmp_path / "loop.json"
    raw = json.loads(path.read_text())
    raw["mode"] = "pipelined"
    path.write_text(json.dumps(raw))
    assert DL.main(["dry-run", "--config", str(path), "--sequential"]) == 1     # state is pipelined
    assert "use another loop_root" in capsys.readouterr().out


def test_pipelined_one_batch_per_policy_checkpoint(tmp_path):
    cfg, fake, loop, pid = _pipe(tmp_path, collect={"batch_size": 2})
    fake.procs[pid]["next"] = 1e18                       # the policy writes no further checkpoint
    loop.run(max_steps=30)
    assert len(loop.state["batches"]) == 1               # the second host waits for a new checkpoint
    (tmp_path / "b").mkdir()
    cfg2, fake2, loop2, pid2 = _pipe(tmp_path / "b", collect={"batch_size": 2, "require_new_policy_ckpt": False})
    fake2.procs[pid2]["next"] = 1e18
    loop2.run(max_steps=30)
    assert len(loop2.state["batches"]) > 2


def test_collection_dirs_are_per_loop_and_never_reused(tmp_path):
    cfg, fake, loop, _ = _pipe(tmp_path, hosts=["cpu"])
    loop.run(max_steps=1)
    batch = loop.state["batches"]["b0001"]
    assert batch["remote_dir"] == f"{tmp_path / 'box'}/{loop.state['loop_id']}/b0001"
    # a second loop whose next batch dir already exists refuses to reuse it
    cfg2, fake2, loop2, _ = _pipe(tmp_path / "x", hosts=["cpu"]) if (tmp_path / "x").mkdir() is None else (None,) * 4
    Path(f"{tmp_path / 'x' / 'box'}/{loop2.state['loop_id']}/b0001").mkdir(parents=True)
    loop2.run(max_steps=1)
    assert not loop2.state["batches"] and not loop2.state["seeds_used"]
    events = [json.loads(x) for x in (tmp_path / "x" / "loop" / "loop_log.jsonl").read_text().splitlines()]
    assert any(e["event"] == "lane_error" and "never reuse" in e["error"] for e in events)
