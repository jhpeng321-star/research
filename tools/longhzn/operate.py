"""Sequential, guarded transport for S-LONGHZN-1 (CPU inference only)."""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = {"cpuP": "/home/ubuntu/miniconda3/bin/python", "cpuQ": "/home/ubuntu/miniconda3/bin/python",
          "gpuA": "/root/miniconda3/bin/python"}


def remote(host, source, timeout=180, echo=True):
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
                             PYTHON[host] + " -B -"], input=source, text=True,
                            encoding="utf-8", capture_output=True, timeout=timeout)
    if echo:
        print(result.stdout, end="")
    print(result.stderr, end="")
    if result.returncode:
        raise SystemExit(result.returncode)
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("host", choices=PYTHON)
    parser.add_argument("script")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    remote(args.host, (ROOT / args.script).read_text(encoding="utf-8"), args.timeout)


if __name__ == "__main__":
    main()
