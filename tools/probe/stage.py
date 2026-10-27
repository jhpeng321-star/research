"""Stage only source/tests to the authorized ~/probe directory; no launchers."""
import subprocess
import tarfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
paths = ["src", "traffic_shepherd", "tests/unit", "scripts/c4", "configs", "tools/fastloop",
         "scripts/round13c_waiter.py", "tools/main14", "tools/loop14_checks", "tools/probe", "pyproject.toml"]
if sys.argv[1:]:
    paths = sys.argv[1:]
names = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
forbidden = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "hetdata_truth", "vtype")
names = [p for p in names if any(p == root or p.startswith(root + "/") for root in paths)
         and not any(word in p.lower() for word in forbidden)]
print(f"Staging {len(names)} files", flush=True)
subprocess.run(["ssh", "-n", "cpuP", "test -d ~/probe"], check=True)
ssh = subprocess.Popen(["ssh", "-T", "cpuP", "tar -xf - -C ~/probe"], stdin=subprocess.PIPE)
with tarfile.open(fileobj=ssh.stdin, mode="w|") as archive:
    for i, name in enumerate(names):
        if i % 100 == 0:
            print(f"Archive {i}: {name}", flush=True)
        archive.add(ROOT / name, arcname=name, recursive=False)
ssh.stdin.close()
if ssh.wait():
    raise SystemExit("source staging failed")
print(f"Staged {len(names)} source/test files to cpuP:~/probe")
