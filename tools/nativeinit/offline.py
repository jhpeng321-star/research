"""Stage approved source-only offline verification in this task's own gpuR scratch."""
from __future__ import annotations
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("E:/agent-work/c4-mainstart")
REMOTE = "/root/autodl-tmp/c4/nativeinit_s1"
DENIED = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "hetdata_truth")


def files():
    paths = []
    for directory in ("src", "traffic_shepherd", "configs", "scripts/c4", "tests/unit",
                      "tools/fastloop", "tools/main14"):
        for p in (BASE / directory).rglob("*"):
            rel = p.relative_to(BASE)
            if any(word in rel.as_posix().lower() for word in DENIED):
                continue
            if p.is_file() and p.suffix in (".py", ".toml", ".json") and not p.is_symlink():
                paths.append(rel)
    return paths


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("inspect", "status"):
        script = '''import json,pathlib
p=pathlib.Path("/root/autodl-tmp/c4/v5/mainstart/mainr16/restart/CONFIG.json")
d=json.loads(p.read_text())
print(json.dumps({k:d[k] for k in ("policy","loop_root")}))
s=json.loads(pathlib.Path("/root/main14_loop/LOOP14_STATE.json").read_text())
print(json.dumps({k:s.get(k) for k in ("accepted_wm","real_returns")}))
'''
        if sys.argv[1] == "status":
            script = '''import json,pathlib
r=pathlib.Path('/root/autodl-tmp/c4/nativeinit_s1')
crn=(r/'MEASURE_CRN_PID.json').exists()
final=(r/'MEASURE_FINAL_PID.json').exists()
tag='measure30_final' if final else 'measure30_crn' if crn else 'measure30'
pid=json.loads((r/('MEASURE_FINAL_PID.json' if final else 'MEASURE_CRN_PID.json' if crn else 'MEASURE_PID.json')).read_text())
proc=pathlib.Path('/proc')/str(pid['pid'])/'stat'
print(json.dumps({'owned':pid,'live':proc.exists() and proc.read_text().split()[21]==pid['start']}))
print((r/(tag+'.log')).read_text())
p=r/tag/'RUN_SUMMARY.json'
if p.exists():
 d=json.loads(p.read_text()); print(json.dumps({k:d.get(k) for k in ('status','error','native_curve','recommended_updates_per_cycle')}))
p=r/tag/'native_curve.jsonl'
if p.exists(): print(p.read_text())
p=r/tag/'updates.jsonl'
if p.exists():
 rows=[json.loads(s) for s in p.read_text().splitlines()]
 print(json.dumps({'updates_recorded':len(rows),'last':{k:rows[-1].get(k) for k in ('update','actor_loss','actor_grad_norm')}}))
'''
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "gpuR", "/root/miniconda3/bin/python -"],
                                input=script.encode(), capture_output=True, check=True)
        print(result.stdout.decode())
        return
    release = "runtime_" + sys.argv[1] if len(sys.argv) > 1 else "runtime"
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        for rel in files():
            # Original sources support exact OFF comparisons, never import restricted data.
            tar.add(BASE / rel, arcname="baseline/" + rel.as_posix())
            path = ROOT / rel if (ROOT / rel).is_file() else BASE / rel
            tar.add(path, arcname=release + "/" + rel.as_posix())
        for rel in ("src/wms/control/nativeinit.py", "tests/unit/test_nativeinit.py"):
            tar.add(ROOT / rel, arcname=release + "/" + rel)
        for rel in ("tests/__init__.py", "tests/unit/__init__.py"):
            tar.addfile(tarfile.TarInfo(release + "/" + rel), io.BytesIO(b""))
    remote = ("import io, pathlib, sys, tarfile; p=pathlib.Path(" + repr(REMOTE) + "); "
              "p.mkdir(parents=True, exist_ok=True); "
              "t=tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz'); "
              "assert all(not m.issym() and not m.islnk() and not m.name.startswith('/') "
              "and '..' not in pathlib.PurePosixPath(m.name).parts for m in t.getmembers()); "
              "t.extractall(p, filter='data'); print('source-only staged')")
    import shlex
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "gpuR",
                             "/root/miniconda3/bin/python -c " + shlex.quote(remote)],
                            input=payload.getvalue(), capture_output=True, check=True)
    print(result.stdout.decode())
    print(json.dumps({"remote": REMOTE, "source_archive_bytes": len(payload.getvalue())}))


if __name__ == "__main__":
    main()
