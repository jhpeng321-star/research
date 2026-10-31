"""Explicit-host transport; remote Python is fed on stdin (no shell quoting)."""
import argparse
import subprocess

from .audit import safe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("host", choices=("gpuR", "gpuA", "cpuQ"))
    p.add_argument("script")
    a = p.parse_args()
    python = ("/home/ubuntu/miniconda3/bin/python" if a.host == "cpuQ"
              else "/root/miniconda3/bin/python")
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                        a.host, python+" -B -"], input=safe(a.script).read_text(),
                       text=True, timeout=120)
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
