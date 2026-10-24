"""One-shot operational handoff; polls smoke completion, never retries training."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path("/root/autodl-tmp/c4/v5")
REPO = ROOT / "repo_r13c_14bdd8ec"
SMOKE, GATE = 522790, 522786
sys.path.insert(0, str(REPO / "scripts"))
from round13c_waiter import pidfd_open, pidfd_send_signal, proc_stat  # noqa: E402


def argv(pid):
    return Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")


def main():
    (ROOT / ".round13c_handoff.lock").mkdir()
    smoke_identity, gate_identity = proc_stat(SMOKE), proc_stat(GATE)
    assert smoke_identity and gate_identity and gate_identity[1] == "T"
    assert str(ROOT / "runs/smoke_r13c") in argv(SMOKE)
    assert str(ROOT / "repo_r13c_bd3c16a8/scripts/round13c_gate.sh") in argv(GATE)
    fd = pidfd_open(GATE)
    try:
        deadline = time.monotonic() + 90 * 60
        while True:
            stat = proc_stat(SMOKE)
            assert stat and stat[0] == smoke_identity[0], "smoke process identity lost"
            if stat[1] == "Z":
                fields = Path(f"/proc/{SMOKE}/stat").read_text().rsplit(")", 1)[1].split()
                exit_code = os.waitstatus_to_exitcode(int(fields[49]))
                break
            assert time.monotonic() < deadline, "smoke completion wait timed out"
            time.sleep(10)
        print(f"smoke PID {SMOKE} exited {exit_code}", flush=True)
        assert exit_code == 0, "smoke failed; production prohibited"
        assert (ROOT / "runs/smoke_r13c/RUN_SUMMARY.json").is_file()
        assert not (ROOT / "runs/c4_wm_v10_round13c").exists()
        assert proc_stat(GATE) == gate_identity, "held gate identity/state changed"
        # Retire only the held original handoff, leaving smoke artifacts untouched.
        pidfd_send_signal(fd, signal.SIGTERM)
        try:
            pidfd_send_signal(fd, signal.SIGCONT)
        except ProcessLookupError:
            pass
        for _ in range(50):
            stat = proc_stat(GATE)
            if stat is None or stat[0] != gate_identity[0] or stat[1] == "Z":
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("held gate did not exit; production prohibited")
    finally:
        os.close(fd)
    env = os.environ.copy()
    env.update(PYTHONPATH="src:.", OMP_NUM_THREADS="1")
    command = [sys.executable, "-m", "wms.train.round13c_gate",
               "--smoke-dir", str(ROOT / "runs/smoke_r13c"),
               "--smoke-log", str(ROOT / "smoke_r13c.log"),
               "--train-exit-code", str(exit_code),
               "--init-checkpoint", str(ROOT / "runs/c4_wm_v10_round12c/wm_s.pt"),
               "--config", "configs/c4_wm_v10round13c.toml",
               "--report", str(ROOT / "smoke_r13c.gate.json")]
    with (ROOT / "smoke_r13c.verdict").open("x") as log:
        subprocess.run(command, cwd=REPO, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    print((ROOT / "smoke_r13c.verdict").read_text(), flush=True)
    with (ROOT / "production_r13c.log").open("x") as log:
        process = subprocess.Popen(["bash", str(REPO / "scripts/round13c_production.sh"), str(REPO)],
                                   cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {"production_pid": process.pid, "smoke_exit_code": exit_code,
               "launched_at": datetime.datetime.now().astimezone().isoformat(),
               "repo": str(REPO), "verdict": "PASS"}
    (ROOT / "handoff_r13c.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
