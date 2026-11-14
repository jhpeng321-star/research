"""WD-01 Part B B1: build the verified PLAIN training bundle in ONE command.

    python make_v0_bundle.py --data-root <versioned dataset root> \
        --code-tree <output/wd01 tree> [--code-commit <sha>] --tag v0.3 \
        [--value-model {net,gbm}] \
        --out-dir E:/research-dense/output/partB_bundles [--smoke-test]

Assembles (reference: the hand-built v0.2 bundle + BUNDLE_V0.2_RECORD.md):

    partB_v0_bundle_<tag>/
      code/     wdpaths.py + partB_v0/ + part4/partB/  (LF endings,
                no __pycache__)
      data/     MANIFEST.json, DATA_CARD_V0.md, worlds/...  -- ONLY the
                NON-superseded, non-HOLDOUT sources' files; superseded
                MANIFEST entries are kept (provenance) but their files are
                omitted and the trainer's default --exclude-superseded
                tolerates their absence
      requirements_plain.txt   (pandas/pyarrow pins; torch & numpy NOT pinned)
      README_PLAIN.md          (nohup instructions; tag/sha placeholders filled)
      BUNDLE_MANIFEST.json     (sha256 per file + tag + code commit + data
                                MANIFEST sha)
    partB_v0_bundle_<tag>.zip + <zip>.sha256

Verification: the data MANIFEST and EVERY included file hash are checked
before zipping (refusal on any mismatch or HOLDOUT file); with
--smoke-test the zip is unzipped into a temp dir and
`python -m partB_v0 train --smoke --device cpu --data ../data` is run from
code/ (KEEP_ALIVE=0); checks: rc=0, DONE printed, OUTPUT_MANIFEST re-hash
OK, results.zip present with no ckpt/ inside.

ZERO SIMULATION.  CPU work runs at below-normal priority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wdpaths  # noqa: E402

RD = Path(wdpaths.RESEARCH_DENSE_ROOT)
DEFAULT_OUT_DIR = str(RD / "output" / "partB_bundles")
README_TEMPLATE_DEFAULT = str(RD / "output" / "partB_bundles" / "staging_v0.2"
                              / "partB_v0_bundle_v0.2" / "README_PLAIN.md")
SCHEMA = "wd01_partB_v0_bundle/1"
CODE_PARTS = ("wdpaths.py", "partB_v0", Path("part4") / "partB")
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "results"}
SKIP_SUFFIX = (".pyc", ".pyo", ".log")
T0W = time.time()


def log(msg):
    print(f"[{time.time() - T0W:7.1f}s] {msg}", flush=True)


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def lower_priority():
    BELOW_NORMAL = 0x4000
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        log("priority set to below-normal (psutil)")
        return
    except Exception:
        pass
    try:
        import ctypes
        if os.name == "nt":
            h = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(h, BELOW_NORMAL)
        else:
            os.nice(10)
    except Exception as e:
        log(f"priority unchanged ({e})")


# ------------------------------------------------------------ data side --

def select_data_files(data_root: Path):
    """(included rels with expected sha, n_superseded_entries_omitted,
    n_holdout_listed, n_holdout_worlds, data_manifest_sha).  Superseded
    entries keep their MANIFEST row (provenance) but contribute NO files;
    HOLDOUT sources have no entries (never built -- they appear only in
    the per-chunk skipped_holdout listings, counted separately)."""
    man_path = data_root / "MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man_sha = sha256_file(man_path)
    holdouts = {s["world"] for sec in man.get("chunks", {}).values()
                for s in sec.get("skipped_holdout", [])}
    included: dict[str, str] = {"MANIFEST.json": man_sha}
    card = data_root / "DATA_CARD_V0.md"
    if not card.is_file():
        raise SystemExit(f"REFUSAL: {card} missing -- run make_data_card.py "
                         f"on the data root first")
    included["DATA_CARD_V0.md"] = sha256_file(card)
    n_sup = 0
    n_hold = 0
    for key, e in sorted(man["worlds"].items()):
        if str(e["split"]).upper() == "HOLDOUT":
            n_hold += 1
            assert not e.get("files"), f"HOLDOUT entry with files: {key}"
            continue
        if e.get("superseded_by"):
            n_sup += 1
            continue                        # provenance kept, files omitted
        for rel, sha in sorted(e["files"].items()):
            if any(f"/{w}__" in rel for w in holdouts):
                raise SystemExit(f"REFUSAL: HOLDOUT file listed: {rel}")
            included[rel] = sha
    return man, included, n_sup, n_hold, len(holdouts), man_sha


# ------------------------------------------------------------ code side --

def iter_code_files(code_tree: Path):
    for part in CODE_PARTS:
        base = code_tree / part
        if base.is_file():
            yield base
            continue
        assert base.is_dir(), f"code tree missing {base}"
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if any(d in p.parts for d in SKIP_DIRS):
                continue
            if p.suffix in SKIP_SUFFIX or p.name.endswith(".pyc"):
                continue
            yield p


def stage_code(staging: Path, code_tree: Path) -> int:
    n = 0
    for p in iter_code_files(code_tree):
        rel = p.relative_to(code_tree)
        dst = staging / "code" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw = p.read_bytes().replace(b"\r\n", b"\n")     # LF endings
        dst.write_bytes(raw)
        n += 1
    return n


# ---------------------------------------------------------------- readme --

README_TEMPLATE = """# Part B V0 training bundle __TAG__: step-by-step guide

This zip trains the Part B "value function" (5 small neural networks) on the
current data: __DATA_DESC__, dataset MANIFEST sha256 `__MANIFEST_SHA16__…`.
Value model: __VALUE_MODEL__.

A single command does everything:

- trains the 5 networks;
- scores them on the calibration worlds;
- writes one `results.zip` for you to download.

It needs an NVIDIA GPU. The target is about 75 minutes of training plus a
few minutes of scoring.

**Before you start:** check that the zip arrived intact. Its SHA-256 must
equal the value we gave you (`__ZIP_NAME__.sha256`). On Linux:
`sha256sum __ZIP_NAME__`.

---

## A. AutoDL (recommended)

1. **Rent a machine.**
   - GPU: 1 × RTX 4090 (or 3090).
   - Image: a **PyTorch** base image with PyTorch ≥ 2.1, Python ≥ 3.10 and
     CUDA 12.x. For example, PyTorch 2.3.0 / Python 3.12 / CUDA 12.1.
   - Start the instance and open **JupyterLab → Terminal**.
2. **Upload** `__ZIP_NAME__` to `/root/autodl-tmp` (the data
   disk) with JupyterLab's upload button or AutoDL's file transfer.
3. **Unzip and install:**
   ```bash
   cd /root/autodl-tmp
   unzip -q __ZIP_NAME__
   cd __BUNDLE_DIR__
   pip install -r requirements_plain.txt
   ```
   The requirements do NOT install or upgrade PyTorch. The image's PyTorch
   is used.
4. **Quick check (about 2 min):**
   ```bash
   cd /root/autodl-tmp/__BUNDLE_DIR__/code
   python -m partB_v0 train --smoke --data ../data --out /root/autodl-tmp/smoke_out
   ```
   It must end with a line starting `DONE`. Near the top, the line starting
   `probe:` must say `device=cuda`. Near the end, the `env:` line must say
   `cuda=True` and name your GPU. If you see `device=cpu` or `cuda=False`,
   stop and send us the output. (Add `--device cuda` to force the GPU; it
   will then fail loudly instead of silently using the CPU.)
5. **Real training.** Run it with `nohup` so a closed browser tab does not
   stop it (tmux is not installed on AutoDL by default):
   ```bash
   cd /root/autodl-tmp/__BUNDLE_DIR__/code
   nohup python -m partB_v0 train --data ../data \
       --out /root/autodl-tmp/out___TAG_NODOT__ \
       > /root/autodl-tmp/train___TAG_NODOT__.log 2>&1 &
   tail -f /root/autodl-tmp/train___TAG_NODOT__.log
   ```
   (`tail -f` shows progress; Ctrl-C stops only the tail, not the training.
   If you prefer tmux: `apt-get install -y tmux` is optional.)
   - **If anything interrupts it** (disconnect, instance restart), run
     **exactly the same nohup command** again. It resumes from the last
     checkpoint (saved every 10 min) and does not redo finished networks.
6. **When the log prints `DONE — download …/results.zip`,** download
   `/root/autodl-tmp/out___TAG_NODOT__/results.zip` and
   `/root/autodl-tmp/out___TAG_NODOT__/results.zip.sha256` through
   JupyterLab (right-click → Download).
7. **SHUT DOWN the instance** in the AutoDL console. It bills while it is
   running.

## B. Google Colab (alternative)

1. Runtime → Change runtime type → GPU (T4 or better).
2. In a cell:
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   ```
3. Upload the zip to your Google Drive, for example `MyDrive/partB/`, then:
   ```bash
   !cd /content && unzip -q /content/drive/MyDrive/partB/__ZIP_NAME__
   !pip install -r /content/__BUNDLE_DIR__/requirements_plain.txt
   !cd /content/__BUNDLE_DIR__/code && python -m partB_v0 train --data ../data --out /content/drive/MyDrive/partB/out___TAG_NODOT__
   ```
   The output goes to Drive, so a Colab disconnect loses at most 10
   minutes. Re-run the last line to resume.
4. Download `results.zip` from `MyDrive/partB/out___TAG_NODOT__/`.

## What is (and is not) in results.zip

- **Included:** weights, scores, the machine report and the logs.
- **Not included:** the resume checkpoints in
  `out___TAG_NODOT__/ckpt/`. They stay on the machine and are not
  needed by us.
- `OUTPUT_MANIFEST.json` lists a checksum for every file except the logs,
  which are still being written when the checksums are taken.

## What to send back

Send `results.zip` and `results.zip.sha256`. Nothing else is needed. The
zip contains:

- the 5 trained networks;
- the scores;
- the logs;
- a machine report;
- a checksum list.

## If something goes wrong

- **`pip` reports a version conflict** with pandas or pyarrow: run
  `pip install pandas pyarrow` (without versions), then continue.
- **"CUDA out of memory":** re-run the same nohup command with
  `--batch-size 256` added (or 128). It resumes; nothing is lost.
- **Training takes far longer than ~90 min:** the program sizes itself to the
  GPU's speed at the start (a "probe" line). You can shorten it with
  `--time-budget-min 45`.
- **Anything else:** send us the last 50 lines of
  `/root/autodl-tmp/train___TAG_NODOT__.log`.
"""

REQUIREMENTS_PLAIN = """# Part B V0 bundle __TAG__: Python dependencies EXCEPT torch.
# torch comes from the host image (PyTorch >= 2.1, CUDA 12.x). Do not reinstall it.
# numpy is deliberately NOT pinned: keep the host image's numpy, which its torch
# build was compiled against. torch < 2.3 needs numpy 1.x; torch >= 2.3 works with 1.x or 2.x.
# pandas/pyarrow pins support numpy 1.26 and 2.x and Python 3.9-3.12.
pandas==2.2.3
pyarrow==17.0.0
"""


def fill_template(tag: str, data_desc: str, man_sha: str, zip_name: str,
                  value_model: str = "net") -> str:
    return (README_TEMPLATE
            .replace("__TAG_NODOT__", tag.replace(".", ""))
            .replace("__TAG__", tag)
            .replace("__DATA_DESC__", data_desc)
            .replace("__MANIFEST_SHA16__", man_sha[:16])
            .replace("__ZIP_NAME__", zip_name)
            .replace("__BUNDLE_DIR__", f"partB_v0_bundle_{tag}")
            .replace("__VALUE_MODEL__", value_model))


# ------------------------------------------------------------------ zip --

def write_zip(staging_bundle: Path, zip_path: Path, top_dir: str) -> int:
    files = sorted(p for p in staging_bundle.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            rel = p.relative_to(staging_bundle)
            zi = zipfile.ZipInfo(f"{top_dir}/{rel.as_posix()}",
                                 date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, p.read_bytes())
    return len(files)


# ----------------------------------------------------------------- smoke --

def run_smoke(zip_path: Path, tag: str, out_dir: Path, python: str,
              members: int | None = None, timeout_s: int = 2400) -> dict:
    """Unzip into a scratch dir and run the CPU --smoke self-test from
    code/.  Nothing is deleted: the scratch dir is reused across runs and
    the smoke OUT_DIR is timestamped fresh each run."""
    top_dir = f"partB_v0_bundle_{tag}"
    scratch = out_dir / f"smoke_{tag}"
    scratch.mkdir(parents=True, exist_ok=True)
    log(f"smoke: unzipping into {scratch} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(scratch)
    code = scratch / top_dir / "code"
    out_sub = scratch / f"out_{tag.replace('.', '')}_" \
                        f"{time.strftime('%H%M%S')}"
    env = dict(os.environ)
    env.update(KEEP_ALIVE="0", WD_DEVICE="cpu",
               CUDA_VISIBLE_DEVICES=env.get("CUDA_VISIBLE_DEVICES", ""),
               PYTHONPATH=str(code))
    cmd = [python, "-m", "partB_v0", "train", "--smoke", "--device", "cpu",
           "--data", "../data", "--out", str(out_sub)]
    if members is not None:
        cmd += ["--members", str(members)]
    log(f"smoke: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=code, env=env, capture_output=True,
                          text=True, timeout=timeout_s)
    wall = time.time() - t0
    out_txt = (proc.stdout or "") + (proc.stderr or "")
    log_path = scratch / "smoke_last_output.txt"
    log_path.write_text(out_txt, encoding="utf-8")

    checks: dict = {"rc": proc.returncode,
                    "done_printed": "DONE" in out_txt,
                    "data_verified_printed": "data: verified" in out_txt}
    tail = out_txt.strip().splitlines()[-15:]
    print("--- smoke output tail " + "-" * 40, flush=True)
    for ln in tail:
        print(f"  | {ln}", flush=True)
    print("-" * 62, flush=True)

    om_path = out_sub / "OUTPUT_MANIFEST.json"
    if om_path.is_file():
        om = json.loads(om_path.read_text(encoding="utf-8"))
        entries = om.get("files", {})
        n_ok = n_bad = n_skip = 0
        for rel, sha in sorted(entries.items()):
            if rel.startswith(("logs/", "ckpt/")):
                n_skip += 1
                continue
            p = out_sub / rel
            if p.is_file() and sha256_file(p) == sha:
                n_ok += 1
            else:
                n_bad += 1
        checks["output_manifest_rehash"] = \
            dict(ok=n_ok, bad=n_bad, skipped_logs_ckpt=n_skip)
    else:
        checks["output_manifest_rehash"] = "OUTPUT_MANIFEST.json missing"

    rz = out_sub / "results.zip"
    if rz.is_file():
        with zipfile.ZipFile(rz) as z:
            names = z.namelist()
        checks["results_zip"] = dict(
            size=rz.stat().st_size, n_entries=len(names),
            sha256=sha256_file(rz), has_ckpt=any("ckpt/" in n
                                                 for n in names))
    else:
        checks["results_zip"] = "results.zip missing"
    checks["pass"] = (proc.returncode == 0 and checks["done_printed"]
                      and checks["data_verified_printed"]
                      and isinstance(checks["output_manifest_rehash"], dict)
                      and checks["output_manifest_rehash"]["bad"] == 0
                      and isinstance(checks["results_zip"], dict)
                      and not checks["results_zip"]["has_ckpt"])
    checks["wall_s"] = round(wall, 1)
    checks["scratch"] = str(scratch)
    (out_dir / f"partB_v0_bundle_{tag}.smoke.json").write_text(
        json.dumps(checks, indent=1), encoding="utf-8")
    return checks


# ------------------------------------------------------------------ main --

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data-root", required=True,
                    help="versioned dataset root (e.g. "
                         "output/wd01_partB_v0data_v03)")
    ap.add_argument("--code-tree", default=str(HERE),
                    help="an output/wd01 tree (wdpaths.py, partB_v0/, "
                         "part4/partB/)")
    ap.add_argument("--code-commit", default=None,
                    help="commit recorded in BUNDLE_MANIFEST.json "
                         "(default: git -C code-tree rev-parse HEAD, "
                         "suffix -dirty when the tree is dirty)")
    ap.add_argument("--value-model", choices=("net", "gbm"), default="net",
                    help="value-model family this bundle is labelled for "
                         "(reserved pass-through to README_PLAIN.md / "
                         "BUNDLE_MANIFEST.json; only 'net' ships a trainer "
                         "today)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--readme-template", default=None,
                    help="optional README template with __TAG__-style "
                         "placeholders (default: the embedded v0.2-derived "
                         "template)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the smoke self-test")
    ap.add_argument("--smoke-members", type=int, default=None,
                    help="pass --members N to the smoke train (default: "
                         "the trainer default, 5)")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args(argv)

    lower_priority()
    data_root = Path(wdpaths.remap(args.data_root))
    code_tree = Path(wdpaths.remap(args.code_tree))
    out_dir = Path(wdpaths.remap(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    top_dir = f"partB_v0_bundle_{tag}"
    zip_path = out_dir / f"{top_dir}.zip"

    man, included, n_sup, n_hold, n_hold_worlds, man_sha = \
        select_data_files(data_root)
    log(f"data: MANIFEST {len(man['worlds'])} sources; including "
        f"{len(included)} files ({n_sup} superseded sources' files "
        f"omitted, entries kept; {n_hold_worlds} HOLDOUT worlds listed "
        f"across chunks, {n_hold} HOLDOUT entries in the MANIFEST); "
        f"MANIFEST sha {man_sha[:12]}...")
    n_bad = 0
    for rel, sha in sorted(included.items()):
        p = data_root / rel
        if not p.is_file() or sha256_file(p) != sha:
            print(f"DATA HASH MISMATCH: {rel}")
            n_bad += 1
    if n_bad:
        raise SystemExit(f"REFUSAL: {n_bad} data files failed hash "
                         f"verification; nothing was built")
    log(f"data: {len(included)} files verified against MANIFEST "
        f"(0 mismatches)")

    staging_root = out_dir / f"staging_{tag}"
    staging = staging_root / top_dir
    staging.mkdir(parents=True, exist_ok=True)
    n_code = stage_code(staging, code_tree)
    log(f"code: {n_code} files staged from {code_tree} (LF endings, no "
        f"__pycache__)")

    data_dir = staging / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for rel, _sha in included.items():
        dst = data_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data_root / rel, dst)
        assert sha256_file(dst) == _sha

    chunks_present = sorted(man.get("chunks", {}))
    data_desc = (", ".join(chunks_present) + " (TRAIN + CALIBRATION only, "
                 "superseded sources excluded)") if chunks_present \
        else "the V0 dataset"
    (staging / "requirements_plain.txt").write_text(
        REQUIREMENTS_PLAIN.replace("__TAG__", tag), encoding="utf-8")
    if args.readme_template:
        tpl = Path(args.readme_template).read_text(encoding="utf-8")
        readme = (tpl.replace("__TAG_NODOT__", tag.replace(".", ""))
                  .replace("__TAG__", tag)
                  .replace("__DATA_DESC__", data_desc)
                  .replace("__MANIFEST_SHA16__", man_sha[:16])
                  .replace("__ZIP_NAME__", zip_path.name)
                  .replace("__BUNDLE_DIR__", top_dir)
                  .replace("__VALUE_MODEL__", args.value_model))
    else:
        readme = fill_template(tag, data_desc, man_sha, zip_path.name,
                               value_model=args.value_model)
    (staging / "README_PLAIN.md").write_text(readme, encoding="utf-8")

    code_commit = args.code_commit
    if not code_commit:
        head = subprocess.run(["git", "-C", str(code_tree), "rev-parse",
                               "HEAD"], capture_output=True, text=True)
        code_commit = head.stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "-C", str(code_tree), "status",
                                "--porcelain"], capture_output=True,
                               text=True)
        if dirty.stdout.strip():
            code_commit += "-dirty"
    bundle_files = {p.relative_to(staging).as_posix(): sha256_file(p)
                    for p in sorted(staging.rglob("*")) if p.is_file()}
    n_staged = len(bundle_files)     # BUNDLE_MANIFEST.json cannot list itself
    bundle_man = dict(
        schema=SCHEMA, tag=tag,
        built_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        code_source=f"{code_tree} @ {code_commit}",
        value_model=args.value_model,
        data_manifest_sha256=man_sha,
        data_source=f"{data_root} (V0 dataset; chunks {chunks_present}; "
                    f"TRAIN+CALIBRATION, no HOLDOUT; {n_sup} superseded "
                    f"sources' files omitted, entries kept)",
        entry_point="cd code && python -m partB_v0 train --data ../data "
                    "--out <OUT_DIR>",
        n_files=len(bundle_files),
        files=bundle_files)
    (staging / "BUNDLE_MANIFEST.json").write_text(
        json.dumps(bundle_man, indent=1), encoding="utf-8")

    n_files = write_zip(staging, zip_path, top_dir)
    zip_sha = sha256_file(zip_path)
    zip_size = zip_path.stat().st_size
    (out_dir / f"{zip_path.name}.sha256").write_text(
        f"{zip_sha}  {zip_path.name}\n", encoding="utf-8")
    log(f"zip: {zip_path} ({zip_size} bytes, {n_files} entries, "
        f"sha256 {zip_sha})")
    record = dict(schema="wd01_partB_v0_bundle_record/1", tag=tag,
                  zip=str(zip_path), zip_bytes=zip_size, zip_sha256=zip_sha,
                  n_zip_entries=n_files, n_staged_files=n_staged,
                  n_data_files=len(included),
                  value_model=args.value_model,
                  n_superseded_sources_omitted=n_sup,
                  n_holdout_entries=n_hold,
                  n_holdout_worlds_listed=n_hold_worlds,
                  data_manifest_sha256=man_sha,
                  code_source=bundle_man["code_source"],
                  built_utc=bundle_man["built_utc"])
    (out_dir / f"{top_dir}.record.json").write_text(
        json.dumps(record, indent=1), encoding="utf-8")

    smoke = None
    if args.smoke_test:
        log("smoke self-test requested")
        smoke = run_smoke(zip_path, tag, out_dir, args.python,
                          members=args.smoke_members)
        log(f"smoke: rc={smoke['rc']} pass={smoke['pass']} "
            f"({smoke.get('wall_s')}s; OUTPUT_MANIFEST re-hash "
            f"{smoke['output_manifest_rehash']}; results.zip "
            f"{smoke['results_zip']})")
        record["smoke"] = smoke
        (out_dir / f"{top_dir}.record.json").write_text(
            json.dumps(record, indent=1), encoding="utf-8")

    print("BUNDLE BUILD SUMMARY", flush=True)
    print(f"  zip: {zip_path}", flush=True)
    print(f"  size: {zip_size} bytes", flush=True)
    print(f"  sha256: {zip_sha}", flush=True)
    print(f"  entries: {n_files}; staged files: {len(bundle_files)}; "
          f"data files included: {len(included)}; superseded sources' "
          f"files omitted: {n_sup} (MANIFEST entries kept)", flush=True)
    print(f"  sidecars: {zip_path.name}.sha256, "
          f"{top_dir}.record.json", flush=True)
    if smoke is not None:
        print(f"  smoke: rc={smoke['rc']} pass={smoke['pass']} "
              f"checks={ {k: v for k, v in smoke.items() if k != 'scratch'} }",
              flush=True)
    ok = smoke is None or smoke["pass"]
    print("BUNDLE_BUILD: " + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
