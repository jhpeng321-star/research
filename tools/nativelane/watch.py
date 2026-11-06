"""Run in gpuR memory: read cpuX evidence, touch no armed file unless adoption passes."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

DEADLINE = datetime.fromisoformat("2026-10-26T09:00:00+08:00").timestamp()
ARM = Path("/root/autodl-tmp/c4/v5/main14_arm")


def cpu(code, data=None):
    cmd = '/home/ubuntu/c4/venv/bin/python -c "import base64;exec(base64.b64decode(\'' + base64.b64encode(code.encode()).decode() + '\'))"'
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "cpuX", cmd],
                       input=data, capture_output=True, timeout=180, check=False)
    if p.returncode:
        raise RuntimeError(p.stderr.decode() + p.stdout.decode())
    return p.stdout


def log(event, **kwargs):
    doc = {"time": time.time(), "event": event, **kwargs}
    cpu("from pathlib import Path;import sys;p=Path.home()/'nativelane';assert p.is_dir();f=(p/'gpuR_watch.jsonl').open('ab');f.write(sys.stdin.buffer.read()+b'\\n');f.close()",
        json.dumps(doc).encode())


def main():
    os.nice(19)
    expected = sys.argv[1]
    log("watching", pid=os.getpid(), package_sha256=expected)
    while time.time() < DEADLINE:
        try:
            raw = cpu("from pathlib import Path;import sys;p=Path.home()/'nativelane/DECISION.json';sys.stdout.buffer.write(p.read_bytes() if p.exists() else b'{}')")
            decision = json.loads(raw)
            if "adopt_scientific" not in decision:
                time.sleep(20)
                continue
            if not decision.get("adopt"):
                log("not_adopted_arm_untouched", reason="registered scientific rule failed or late")
                return
            data = cpu("from pathlib import Path;import sys;sys.stdout.buffer.write((Path.home()/'nativelane/rearm_package.tar.gz').read_bytes())")
            assert hashlib.sha256(data).hexdigest() == expected, "registered package changed"
            assert time.time() < DEADLINE - 300
            # First armed-directory write: complete, timely scientific adoption is now recorded.
            package = ARM / "nativelane_adoption_package"
            assert not package.exists()
            package.mkdir()
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
                for m in t:
                    assert m.isfile() and m.name in ("adopt.py", "source.tar.gz", "request.json")
                    t.extract(m, package)
            log("prearm_started")
            with (package / "adoption.log").open("ab") as stream:
                proc = subprocess.run(["/root/miniconda3/bin/python", str(package / "adopt.py"), "--package", str(package)],
                                      cwd=ARM, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                      timeout=5400, check=False)
            log("adoption_finished", rc=proc.returncode)
            return
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as error:
            # Readiness connectivity can recover. Operational/adoption failures do not retry.
            if (ARM / "nativelane_adoption_package").exists():
                log("operational_failure_no_retry", error=repr(error))
                return
            time.sleep(30)
        except (AssertionError, OSError) as error:
            log("adoption_refused_no_retry", error=repr(error))
            return
    log("expired_arm_untouched")


if __name__ == "__main__":
    main()
