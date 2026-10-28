"""Read-only fetch of the already frozen candidate-A WM3 checkpoint (no datasets)."""
import hashlib
import base64
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2] / "_execparity_data"
PATH = "/root/autodl-tmp/c4/restart20261026/arm/loop/wm/c03.a2/wm_s.pt"
SHA = "41e18f09b6060a1f4e4039f02655fcbcd497e32a360cffb71be716fffeaadc84"

if __name__ == "__main__":
    assert ROOT.is_dir()
    code = ("import pathlib,sys,base64,json; "
            f"p=pathlib.Path({PATH!r}); "
            "print(json.dumps({n:base64.b64encode(q.read_bytes()).decode() for n,q in "
            "{'wm3.pt':p,'physics.json':p.with_name('physics.json')}.items()}))")
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "gpuA",
                        "/root/miniconda3/bin/python -B -"], input=code.encode(),
                       capture_output=True, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    files = {n:base64.b64decode(b) for n,b in json.loads(p.stdout).items()}
    assert hashlib.sha256(files['wm3.pt']).hexdigest() == SHA
    for name,data in files.items():
        target = ROOT / name
        if target.exists():
            assert target.read_bytes() == data
        else:
            target.write_bytes(data)
        print(name, hashlib.sha256(data).hexdigest(), len(data))
