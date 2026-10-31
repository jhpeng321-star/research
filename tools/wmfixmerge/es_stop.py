"""Stop only receipt-bound ES processes after MAIN's generation-two cutoff."""
import argparse
import base64
import json
import subprocess

from tools.wmfixmerge.es_status import ROOTS, PYTHONS


REMOTE = r'''
import json, os, pathlib, signal, socket, time
root = pathlib.Path(ROOT)
assert root.is_dir()
reason = "MAIN 2026-10-31 17:53: stop after generation 2; no A/B dispatch"
stop = root / "STOP"
if not stop.exists():
    with stop.open("x") as f:
        f.write(reason + "\n")
receipts = []
for pattern in ("RUN.json", "ENV_*.json", "FINISH_WATCH_PID.json"):
    for p in root.glob(pattern):
        d = json.loads(p.read_text())
        if p.name.startswith("ENV_") and d.get("host") != socket.gethostname():
            continue
        receipts.append((int(d["pid"]), p.name))
def command(pid):
    p = pathlib.Path("/proc") / str(pid)
    try:
        if (p / "stat").read_text().split()[2] == "Z":
            return None
        cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        owned_es = "-m tools.wmfixmerge.es " in cmd and ("--out " + str(root) + " ") in cmd + " "
        owned_finish = "-m tools.wmfixmerge.es_finish " in cmd and ("--root " + str(root) + " ") in cmd + " "
        return cmd if owned_es or owned_finish else None
    except (FileNotFoundError, ProcessLookupError):
        return None
targets = []
for pid, receipt in receipts:
    cmd = command(pid)
    if cmd:
        targets.append(dict(pid=pid, receipt=receipt, command=cmd))
# Freeze the dispatcher first so it cannot spawn another child while stopping.
for row in targets:
    if row["receipt"] == "RUN.json" and command(row["pid"]):
        os.kill(row["pid"], signal.SIGSTOP)
for row in targets:
    if command(row["pid"]):
        os.kill(row["pid"], signal.SIGTERM)
        if row["receipt"] == "RUN.json":
            os.kill(row["pid"], signal.SIGCONT)
time.sleep(2)
for row in targets:
    if command(row["pid"]):
        os.kill(row["pid"], signal.SIGKILL)
time.sleep(1)
remaining = [row["pid"] for row in targets if command(row["pid"])]
# Detect unreceipted leftovers by exact module and owned output, never signal them.
unreceipted = []
for p in pathlib.Path("/proc").iterdir():
    if p.name.isdigit() and command(int(p.name)):
        unreceipted.append(int(p.name))
doc = dict(reason=reason, host=socket.gethostname(), observed_unix=time.time(),
           completed_generation_2=(root / "GEN_002.json").is_file(),
           targets=targets, remaining=remaining, active_owned_processes=unreceipted,
           partial_generation_3_preserved=True, verdict_dispatched=False)
receipt = root / "STOP_MAIN_1753.json"
if not receipt.exists():
    with receipt.open("x") as f:
        json.dump(doc, f, indent=2)
print(json.dumps(doc, indent=2))
if remaining or unreceipted:
    raise SystemExit("owned processes remain; reconcile before claiming CPU release")
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=tuple(ROOTS), required=True)
    args = parser.parse_args()
    script = "ROOT = " + repr(ROOTS[args.host]) + "\n" + REMOTE
    encoded = base64.b64encode(script.encode()).decode()
    sudo = "sudo " if args.host == "cpuQ" else ""
    remote = sudo + PYTHONS[args.host] + " -c \"import base64;exec(base64.b64decode('" + encoded + "'))\""
    result = subprocess.run(["ssh", "-o", "ConnectTimeout=8", args.host, remote],
                            text=True, capture_output=True, timeout=30)
    print(result.stdout, end="")
    if result.returncode:
        raise RuntimeError(result.stderr or "owned ES processes remain")


if __name__ == "__main__":
    main()
