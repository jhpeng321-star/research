#!/usr/bin/env python3
"""Box-local metadata, safe git archives, population adaptation, and spot watcher.

Usage: python ops.py --help. No SUMO or network work occurs on import.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
RUNTIME_PATHS = ("traffic_shepherd", "src", "configs", "scripts", "baselines", "scenarios",
                 "assets", "tools", "output/controller_verification_v7/pi_owner_adapter.py",
                 "output/controller_verification_v2/pi_saturation_v1.py",
                 "output/controller_verification_v4/pi_external_u.py",
                 "pyproject.toml", "requirements.txt")


def safe(path: str | Path) -> Path:
    p = Path(path)
    if any(token in str(p).lower() for token in FORBIDDEN):
        raise ValueError("restricted path refused")
    return p


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with safe(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, text: str) -> None:
    safe(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def bundle(repo: Path, out: Path) -> None:
    safe(repo)
    safe(out)
    if out.exists() or out.with_name(out.name + ".sha256").exists():
        raise ValueError("archive already exists; use a new immutable archive name")
    # Inspect names only, then archive exclusively permitted tracked runtime files.
    listing = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-rz",
                                       "--name-only", "HEAD", "--", *RUNTIME_PATHS])
    names = [p for p in listing.decode("utf-8").split("\0")
             if p and not any(t in p.lower() for t in FORBIDDEN)]
    if not names:
        raise ValueError("no runtime files in HEAD")
    subprocess.run(["git", "-C", str(repo), "archive", "--format=tar.gz",
                    f"--output={out.resolve()}", "HEAD", "--", *names], check=True)
    atomic(out.with_name(out.name + ".sha256"), f"{sha(out)}  {out.name}\n")
    print(json.dumps({"commit": git(repo, "rev-parse", "HEAD"), "archive": str(out),
                      "sha256": sha(out), "files": len(names)}))


def extract(archive: Path, target: Path) -> None:
    safe(archive)
    safe(target)
    receipt = safe(archive.with_name(archive.name + ".sha256")).read_text().strip().split()
    if len(receipt) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", receipt[0]):
        raise ValueError("invalid SHA256 sidecar")
    if receipt[1].lstrip("*") != archive.name or sha(archive) != receipt[0].lower():
        raise ValueError("archive SHA256 verification failed")
    fingerprint = receipt[0].lower()
    if target.exists():
        marker = target / ".BOXOPS_ARCHIVE_SHA256"
        if marker.is_file() and marker.read_text().strip() == fingerprint:
            print("same verified archive already installed")
            return
        raise ValueError("repo destination exists with different/no archive receipt; use a new REPO")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        commit = source.pax_headers.get("comment", "")
        if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            raise ValueError("archive must be made by git archive (commit PAX header missing)")
        seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            safe(member.name)
            if (path.is_absolute() or ".." in path.parts or "\\" in member.name
                    or not path.parts or ".git" in path.parts or path.as_posix() in seen
                    or not (member.isfile() or member.isdir())):
                raise ValueError("unsafe archive member (paths/links/devices/duplicates refused)")
            seen.add(path.as_posix())
        # Preserve partial staging after interruption; a retry uses a new directory.
        stage = Path(tempfile.mkdtemp(prefix=".boxops-extract-", dir=target.parent))
        for member in members:
            dest = stage.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError("archive file has no payload")
            with stream, dest.open("wb") as output:
                while block := stream.read(1024 * 1024):
                    output.write(block)
            dest.chmod(member.mode & 0o777)
        for required in ("traffic_shepherd/c4/collect.py", "scripts/boxops/setup_box.sh"):
            if not (stage / required).is_file():
                raise ValueError(f"runtime archive is missing {required}")
        # Strip CRLF only from runtime shell scripts, before ever executing them.
        for script in (stage / "scripts").rglob("*.sh"):
            safe(script)
            script.write_bytes(script.read_bytes().replace(b"\r\n", b"\n"))
            script.chmod(script.stat().st_mode | 0o111)
        atomic(stage / "BOXOPS_CODE_ID", commit + "\n")
        atomic(stage / ".BOXOPS_ARCHIVE_SHA256", fingerprint + "\n")
        stage.rename(target)
    print(f"SHA256 verified; commit {commit}")


def begin(out: Path, repo: Path, flags: list[str]) -> None:
    safe(out)
    safe(repo)
    code = ((repo / "BOXOPS_CODE_ID").read_text().strip() if (repo / "BOXOPS_CODE_ID").is_file()
            else git(repo, "rev-parse", "HEAD"))
    params = dict(item.split("=", 1) for item in flags[1:])
    # File-content identity matters for configs and warm-start weights, too.
    inputs = {}
    for key in ("config", "init_from"):
        if params.get(key):
            path = safe(params[key])
            path = path if path.is_absolute() else repo / path
            if path.is_file():
                inputs[key] = sha(path)
    for controller in params.get("controllers", "").split(","):
        if controller:
            path = safe(controller)
            path = path if path.is_absolute() else repo / path
            if path.is_file():
                inputs["controller:" + controller] = sha(path)
    spec = {"code_commit": code, "mode": flags[0], "flags": params, "input_sha256": inputs}
    spec_path = out / "RUN_SPEC.json"
    if spec_path.is_file() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("resume refused: code, flags or input hashes changed; use a new RUN_ID")
    atomic(spec_path, json.dumps(spec, indent=2) + "\n")
    # A new invocation may produce newer checkpoints/metadata; it must be pushed anew.
    if (out / "PUSHED").exists():
        (out / "PUSHED").rename(out / f"PUSHED.previous-{time.time_ns()}")
    with (out / "CODE_ID.txt").open("a", encoding="utf-8") as log:
        log.write(f"git_commit={code}\nflags={json.dumps(params, sort_keys=True)}\nstart_utc={now()}\n")


def end(out: Path, rc: int) -> None:
    with safe(out / "CODE_ID.txt").open("a", encoding="utf-8") as log:
        log.write(f"end_utc={now()}\nexit_code={rc}\n")


def summary(out: Path) -> None:
    safe(out)
    rows = {}
    # Scan names before reading; never descend into a forbidden directory.
    for root, dirs, files in os.walk(out):
        dirs[:] = [d for d in dirs if not any(t in d.lower() for t in FORBIDDEN)
                   and ".incomplete-" not in d and not d.startswith("c4_episodes")
                   and d not in ("scenarios", "checkpoints")]
        for name in ("COLLECT_SUMMARY.json", "RUN_SUMMARY.json"):
            path = Path(root) / name
            if name in files and path.parent != out:
                rows[path.relative_to(out).as_posix()] = json.loads(safe(path).read_text())
    if (out / "RUN_SUMMARY.json").is_file():
        rows["RUN_SUMMARY.json"] = json.loads((out / "RUN_SUMMARY.json").read_text())
    atomic(out / "COLLECT_SUMMARY.json", json.dumps({"updated_utc": now(), "phases": rows}, indent=2) + "\n")


def supports_population(repo: Path, env: bool = False) -> bool:
    name = "env.py" if env else "collect.py"
    tree = ast.parse(safe(repo / "traffic_shepherd/c4" / name).read_text(encoding="utf-8-sig"))
    if env:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "CavSumoEnv":
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == "__init__":
                        return any(a.arg == "population" for a in method.args.args + method.args.kwonlyargs)
        return False
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "add_argument"
               and any(isinstance(a, ast.Constant) and a.value == "--population" for a in n.args)
               for n in ast.walk(tree))


def require_population(repo: Path) -> None:
    if not supports_population(repo):
        raise ValueError("collector lacks --population; integrate c4/popfix and publish a new archive")


def mappo_config(repo: Path, config: Path, dest: Path, population: str, required: str) -> None:
    safe(repo)
    safe(config)
    safe(dest)
    if required not in ("0", "1"):
        raise ValueError("REQUIRE_POPULATION must be 0 or 1")
    supported = supports_population(repo, env=True)
    if required == "1" and not supported:
        raise ValueError("CavSumoEnv lacks population kwarg; integrate c4/popfix before fine-tuning")
    # wms's TOML parser is lightweight and supports Python 3.10.
    sys.path.insert(0, str(repo / "src"))
    from wms.core.toml import loads
    text = config.read_text(encoding="utf-8")
    cfg = loads(text)
    kwargs = dict(cfg.get("env", {}).get("kwargs", {}))
    if supported:
        kwargs["population"] = population
    else:
        print("WARNING: explicit REQUIRE_POPULATION=0; running legacy environment", file=sys.stderr)
    def toml_value(value):
        if isinstance(value, dict):
            return "{ " + ", ".join(json.dumps(k) + " = " + toml_value(v) for k, v in value.items()) + " }"
        if isinstance(value, list):
            return "[" + ", ".join(toml_value(v) for v in value) + "]"
        if isinstance(value, (str, bool, int, float)):
            return json.dumps(value, allow_nan=False)
        raise ValueError("unsupported TOML env kwarg")
    # Replace only [env].kwargs in a runtime copy; preserve the rest of the config.
    lines, result, in_env, found_env, inserted = text.splitlines(), [], False, False, False
    for line in lines:
        if re.match(r"\s*\[", line):
            if in_env and not inserted:
                result.append("kwargs = " + toml_value(kwargs))
                inserted = True
            in_env = line.strip().split("#", 1)[0].strip() == "[env]"
            found_env |= in_env
        if in_env and re.match(r"\s*kwargs\s*=", line):
            if not re.fullmatch(r"\s*kwargs\s*=\s*\{.*\}\s*(?:#.*)?", line):
                raise ValueError("env.kwargs must be a one-line inline TOML table")
            line = "kwargs = " + toml_value(kwargs)
            inserted = True
        result.append(line)
    if not inserted:
        if not found_env:
            result.append("[env]")
        result.append("kwargs = " + toml_value(kwargs))
    rendered = "\n".join(result) + "\n"
    if loads(rendered).get("env", {}).get("kwargs", {}) != kwargs:
        raise ValueError("generated env.kwargs verification failed")
    atomic(dest, rendered)
    print(f"population={population}; env_kwarg_supported={supported}")


def genrand(args: list[str]) -> int:
    from traffic_shepherd.c4 import genrand_driver as driver
    require_population(Path(driver.__file__).resolve().parents[2])
    original = driver.collect_cmd
    def collect_cmd(*pos, **kw):
        cmd = original(*pos, **kw)
        if "--population" in cmd:
            cmd[cmd.index("--population") + 1] = os.environ["BOXOPS_POPULATION"]
        else:
            cmd += ["--population", os.environ["BOXOPS_POPULATION"]]
        return cmd
    driver.collect_cmd = collect_cmd
    return driver.main(args)


def check_collection(out: Path, expected: int) -> None:
    doc = json.loads(safe(out / "COLLECT_SUMMARY.json").read_text())
    index = json.loads(safe(out / "INDEX.json").read_text())
    complete = doc["episodes_complete"] + len(index.get("already_complete", []))
    if doc["episodes_failed"] or complete != expected:
        raise ValueError(f"expected {expected} complete episodes, got {complete}")


def convert(source: Path, dest: Path) -> None:
    safe(source)
    safe(dest)
    episodes = []
    for root, dirs, files in os.walk(source):
        dirs[:] = [d for d in dirs if ".incomplete-" not in d
                   and not d.startswith("c4_episodes") and not any(t in d.lower() for t in FORBIDDEN)]
        if all(name in files for name in ("episode.npz", "manifest.json", "summary.json")):
            path = safe(Path(root))
            if json.loads((path / "manifest.json").read_text()).get("status") == "complete":
                episodes.append(str(path))
    if not episodes:
        raise ValueError("no complete episodes to convert")
    # Explicit episode roots exclude preserved incomplete ancestors in paired arms,
    # which the existing converter's immediate-directory-name check cannot exclude.
    subprocess.run([sys.executable, "-m", "wms.data.c4_episodes", "--episodes",
                    *sorted(episodes), "--out", str(dest)], check=True)


def check_spot(poll: float) -> tuple[Path | None, str, datetime | None]:
    if poll <= 0:
        raise ValueError("watcher poll interval must be positive")
    notice_text = os.environ.get("SPOT_NOTICE_FILE", "")
    notice = safe(notice_text) if notice_text else None
    deadline_text = os.environ.get("SPOT_STOP_AT_UTC", "")
    deadline = datetime.fromisoformat(deadline_text.replace("Z", "+00:00")) if deadline_text else None
    if deadline is not None and deadline.tzinfo is None:
        raise ValueError("SPOT_STOP_AT_UTC must include timezone")
    url = os.environ.get("SPOT_METADATA_URL", "")
    if url and not (url.startswith("http://169.254.169.254/")
                    or url.startswith("http://metadata.tencentyun.com/")):
        raise ValueError("spot endpoint must be a local cloud metadata URL")
    return notice, url, deadline


def watch_stop(out: Path, pid: int, poll: float) -> None:
    if sys.platform == "win32":
        raise ValueError("spot watcher is Linux-only (Windows PID checks are not safe)")
    safe(out)
    if pid <= 0:
        raise ValueError("watcher PID must be positive")
    notice, url, deadline = check_spot(poll)
    notice = notice or safe(out / "SPOT_NOTICE")
    # Metadata polling is opt-in: set the provider's documented termination endpoint.
    while True:
        try:
            os.kill(pid, 0)  # existence check only, never terminates the trainer
        except ProcessLookupError:
            return
        stop = notice.exists()
        if deadline is not None:
            stop |= datetime.now(timezone.utc) >= deadline
        if url:
            try:
                with urlopen(Request(url), timeout=2) as response:
                    payload = response.read(4096).decode("utf-8").strip()
                    stop |= response.status == 200 and payload not in ("", "null", "none", "false")
            except (URLError, TimeoutError, OSError):
                pass  # no notice / temporary metadata failure
        if stop:
            atomic(out / "STOP_REQUESTED", f"spot notice {now()}\n")
            print(f"STOP_REQUESTED: {now()}", flush=True)
            return
        time.sleep(poll)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("bundle", "extract", "begin", "end", "summary",
                                          "require-population", "mappo-config", "genrand",
                                          "check-collection", "watch-stop", "check-spot", "convert"))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    a = ns.args
    if ns.action == "genrand":
        return genrand(a)
    counts = {"bundle": 2, "extract": 2, "end": 2, "summary": 1,
              "require-population": 1, "mappo-config": 5, "check-collection": 2, "watch-stop": 3,
              "check-spot": 1, "convert": 2}
    if (ns.action == "begin" and len(a) < 3) or (ns.action in counts and len(a) != counts[ns.action]):
        parser.error(f"wrong argument count for {ns.action}")
    if ns.action == "bundle":
        bundle(Path(a[0]), Path(a[1]))
    elif ns.action == "extract":
        extract(Path(a[0]), Path(a[1]))
    elif ns.action == "begin":
        begin(Path(a[0]), Path(a[1]), a[2:])
    elif ns.action == "end":
        end(Path(a[0]), int(a[1]))
    elif ns.action == "summary":
        summary(Path(a[0]))
    elif ns.action == "require-population":
        require_population(Path(a[0]))
    elif ns.action == "mappo-config":
        mappo_config(Path(a[0]), Path(a[1]), Path(a[2]), a[3], a[4])
    elif ns.action == "check-collection":
        check_collection(Path(a[0]), int(a[1]))
    elif ns.action == "watch-stop":
        watch_stop(Path(a[0]), int(a[1]), float(a[2]))
    elif ns.action == "check-spot":
        check_spot(float(a[0]))
    elif ns.action == "convert":
        convert(Path(a[0]), Path(a[1]))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"boxops: {exc}", file=sys.stderr)
        sys.exit(1)
