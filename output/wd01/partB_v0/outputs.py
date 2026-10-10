"""Output plumbing for the B2 CLI: env report, OUTPUT_MANIFEST,
results.zip (+ sha256 sidecar), stdout tee logging, git commit probe.

Env report records python / torch / CUDA / cuDNN / driver / GPU name +
memory / CPU, the data MANIFEST sha256 and the code git commit -- all
guarded, because the cloud container has no git and CPU hosts have no
CUDA.

OUTPUT_MANIFEST.json carries the sha256 of every DELIVERABLE output
file and covers exactly the contents of results.zip.  B3a (item D19):
the bundle is an EXPLICIT WHITELIST -- member_*/gbm_* model files +
sidecars, stats.json / calibration.json / ablation.json (B3b v0.3 net),
metrics.json, eval_report.md, env_report.json, logs/ and
OUTPUT_MANIFEST.json -- so anything else in OUT_DIR (ckpt/,
ckpt_m0_old/, stray files) is excluded from BOTH the manifest and the
zip.  results.zip excludes itself and its .sha256 sidecar, which
postdate it.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path

MANIFEST_EXCLUDE = {"OUTPUT_MANIFEST.json", "results.zip",
                    "results.zip.sha256"}
RESUME_ONLY_DIR = "ckpt"          # excluded from manifest AND zip

# B3a item D19: explicit deliverables whitelist (glob patterns match
# OUT_DIR-top-level file names; dirs are included with ALL contents).
# B3b additions (v0.3 net): stats.json (A4 statistics shipping with the
# weights), calibration.json (D14 maps) and ablation.json (B7 table).
ZIP_WHITELIST_GLOBS = ("member_*.pt", "member_*.json",
                       "gbm_*.pkl", "gbm_*.json")
ZIP_WHITELIST_FILES = ("metrics.json", "eval_report.md",
                       "env_report.json", "OUTPUT_MANIFEST.json",
                       "stats.json", "calibration.json", "ablation.json")
ZIP_WHITELIST_DIRS = ("logs",)


def whitelist_relpaths(out_dir: Path) -> list[str]:
    """Relative paths of the whitelisted deliverables in OUT_DIR
    (sorted); everything else -- ckpt/, ckpt_m0_old/, stray files -- is
    excluded from the manifest and results.zip."""
    out = Path(out_dir)
    keep: set[str] = set()
    if not out.exists():
        return []
    for p in sorted(out.iterdir()):
        rel = p.relative_to(out).as_posix()
        if p.is_file() and (rel in ZIP_WHITELIST_FILES
                            or any(fnmatch.fnmatch(p.name, g) for g
                                   in ZIP_WHITELIST_GLOBS)):
            keep.add(rel)
        elif p.is_dir() and p.name in ZIP_WHITELIST_DIRS:
            for f in sorted(p.rglob("*")):
                if f.is_file() and not f.name.endswith(".tmp"):
                    keep.add(f.relative_to(out).as_posix())
    return sorted(keep)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: str | None = None) -> str:
    """HEAD sha of the code repo, or 'unknown' (e.g. in a container)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
            text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:                                   # noqa: BLE001
        pass
    return "unknown"


def _gpu_info() -> dict:
    info = dict(cuda_available=False)
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            p = torch.cuda.get_device_properties(0)
            info["gpu_name"] = p.name
            info["gpu_total_memory_gb"] = round(
                p.total_memory / 2 ** 30, 2)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception as e:                              # noqa: BLE001
        info["torch_error"] = f"{type(e).__name__}: {e}"
        return info
    try:                                                # driver via nvidia-smi
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=10)
        if out.returncode == 0:
            info["driver_version"] = out.stdout.strip().splitlines()[0]
    except Exception:                                   # noqa: BLE001
        info["driver_version"] = None
    return info


def write_env_report(out_dir: Path, data_manifest_sha256: str,
                     extra: dict | None = None) -> dict:
    """env_report.json: versions, GPU, CPU, data tag, git commit."""
    info = dict(
        schema="wd01_partB_v0_env/1",
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        python=sys.version.split()[0], platform=platform.platform(),
        cpu_model=platform.processor() or platform.machine(),
        cpu_count=os.cpu_count(),
        data_manifest_sha256=data_manifest_sha256,
        git_commit=git_commit(),
        **_gpu_info())
    if extra:
        info.update(extra)
    path = Path(out_dir) / "env_report.json"
    path.write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def write_output_manifest(out_dir: Path) -> dict:
    """sha256 of every whitelisted deliverable output file (see
    whitelist_relpaths; excludes the manifest itself, the results bundle
    and the resume-only ckpt/ -- the manifest covers exactly the contents
    of results.zip)."""
    out = Path(out_dir)
    entries = {}
    for rel in whitelist_relpaths(out):
        if rel in MANIFEST_EXCLUDE:
            continue
        entries[rel] = sha256_of(out / rel)
    doc = dict(
        schema="wd01_partB_v0_output_manifest/1",
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        whitelist=list(ZIP_WHITELIST_GLOBS) + list(ZIP_WHITELIST_FILES)
        + [d + "/" for d in ZIP_WHITELIST_DIRS],
        note=(f"covers exactly the contents of results.zip; whitelist "
              f"deliverables only -- '{RESUME_ONLY_DIR}/' (resume-only "
              f"state) and anything else in OUT_DIR are excluded"),
        files=entries)
    (out / "OUTPUT_MANIFEST.json").write_text(
        json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    return doc


def write_results_zip(out_dir: Path) -> tuple[Path, str]:
    """results.zip of the whitelisted deliverables (+ .sha256 sidecar).

    ckpt/ and any other non-whitelisted OUT_DIR contents are deliberately
    NOT zipped (B3a item D19)."""
    out = Path(out_dir)
    zpath = out / "results.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in whitelist_relpaths(out):
            p = out / rel
            if p == zpath or p.name == "results.zip.sha256" \
                    or p.name.endswith(".tmp"):
                continue          # OUTPUT_MANIFEST.json ships inside
            z.write(p, rel)
    digest = sha256_of(zpath)
    (out / "results.zip.sha256").write_text(digest + "\n",
                                            encoding="utf-8")
    return zpath, digest


class TeeLogger:
    """stdout + out/logs/train.log."""

    def __init__(self, out_dir: Path, name: str = "partB_v0"):
        logs = Path(out_dir) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self.path = logs / f"{name}.log"
        self.fh = open(self.path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:                               # noqa: BLE001
            pass


def finalize_outputs(out_dir: Path, log=print) -> dict:
    """OUTPUT_MANIFEST.json -> results.zip -> results.zip.sha256.

    Call with log=None for the silent final pass (the CLI rebuilds the
    bundle once AFTER the last log line so every manifest hash matches
    the final file contents)."""
    man = write_output_manifest(out_dir)
    zpath, digest = write_results_zip(out_dir)
    if log is not None:
        log(f"outputs: OUTPUT_MANIFEST.json with {len(man['files'])} "
            f"whitelisted deliverable files ('{RESUME_ONLY_DIR}/' and "
            f"non-whitelisted OUT_DIR content are not bundled)")
        log(f"outputs: {zpath.name} "
            f"({zpath.stat().st_size / 2 ** 20:.1f} MiB) sha256 {digest}")
    return dict(n_files=len(man["files"]), results_zip=str(zpath),
                results_zip_sha256=digest,
                results_zip_mib=round(zpath.stat().st_size / 2 ** 20, 1))


def verify_outputs(out_dir: Path) -> tuple[bool, list[str]]:
    """Re-hash every OUTPUT_MANIFEST entry, check the zip contents match
    the manifest exactly and results.zip.sha256 pins the zip."""
    out = Path(out_dir)
    problems: list[str] = []
    man_path = out / "OUTPUT_MANIFEST.json"
    if not man_path.exists():
        return False, ["OUTPUT_MANIFEST.json missing"]
    doc = json.loads(man_path.read_text(encoding="utf-8"))
    for rel, sha in sorted(doc.get("files", {}).items()):
        p = out / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
        elif sha256_of(p) != sha:
            problems.append(f"sha mismatch: {rel}")
    zpath = out / "results.zip"
    sha_path = out / "results.zip.sha256"
    if not zpath.exists():
        problems.append("results.zip missing")
    else:
        with zipfile.ZipFile(zpath) as z:
            zipped = {n for n in z.namelist() if not n.endswith("/")}
        # the manifest cannot list itself, but it ships inside the zip
        expected = set(doc.get("files", {})) | {"OUTPUT_MANIFEST.json"}
        if zipped != expected:
            problems.append(
                f"zip/manifest contents differ: zip-only="
                f"{sorted(zipped - expected)}, "
                f"manifest-only={sorted(expected - zipped)}")
        if not sha_path.exists():
            problems.append("results.zip.sha256 missing")
        elif sha_path.read_text(encoding="utf-8").strip() \
                != sha256_of(zpath):
            problems.append("results.zip.sha256 does not match zip")
    return (not problems), problems


def keep_alive_if_requested(log=print) -> None:
    """Addendum 3a: keep the container alive after DONE until stopped.

    Only sleeps when KEEP_ALIVE=1 is set explicitly; the Docker
    entrypoint (separate task) will set it on cloud images, so local
    runs and CI never hang here.
    """
    if os.environ.get("KEEP_ALIVE", "").strip() == "1":
        log("keep-alive: KEEP_ALIVE=1 -> sleeping until stopped "
            "(Ctrl-C / container stop to exit)")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            log("keep-alive: interrupted; exiting")
