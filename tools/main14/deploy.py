"""Stage committed, allowlisted MAIN14 code on the two authorised hosts, then arm explicitly."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V5 = "/root/autodl-tmp/c4/v5"
GPU_CODE = V5 + "/repo_main14"
CPU_CODE = "/home/ubuntu/c4/repo-main14"
TOKENS = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "hetdata_truth", "vtype")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def remote(host, code, args=(), data=None):
    python = "/root/miniconda3/bin/python" if host == "gpuR" else "/home/ubuntu/c4/venv/bin/python"
    import base64
    payload = base64.b64encode(code.encode()).decode()
    command = f"{python} -c \"import base64;exec(base64.b64decode('{payload}'))\" " + " ".join(args)
    proc = subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], input=data,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=False)
    if proc.returncode:
        raise RuntimeError(f"{host}: {proc.stderr.decode()} {proc.stdout.decode()}")
    return proc.stdout.decode()


def export(commit):
    prefixes = ["src", "traffic_shepherd", "tools", "scripts", "configs", "output", "protocols",
                "tests/unit", "baselines/circles/adapter.json", "pyproject.toml",
                "assets/research/highway_merge_shepherd_long/osm.net.xml.gz"]
    names = git("ls-tree", "-r", "--name-only", commit, "--", *prefixes).decode().splitlines()
    selected = []
    for name in names:
        if any(token in name.lower() for token in TOKENS):
            continue
        if name.startswith("output/") and not name.endswith(".py"):
            continue
        if Path(name).suffix not in (".py", ".json", ".toml", ".sh", ".gz"):
            continue
        # No SUMO vehicle/driver assignment XML is opened or exported.
        selected.append(name)
    # One Git batch avoids thousands of slow subprocess startups on Windows.
    objects = subprocess.check_output(["git", "cat-file", "--batch"], cwd=ROOT,
                                      input="".join(f"{commit}:{name}\n" for name in selected).encode())
    source = io.BytesIO(objects)
    files = {}
    for name in selected:
        header = source.readline().decode().split()
        if len(header) != 3 or header[1] != "blob":
            raise RuntimeError(f"not a regular committed source file: {name}")
        files[name] = source.read(int(header[2]))
        assert source.read(1) == b"\n"
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue(), manifest


def stage():
    commit = git("rev-parse", "HEAD").decode().strip()
    bundle, manifest = export(commit)
    extract = """
import io,json,pathlib,sys,tarfile,hashlib
root=pathlib.Path(sys.argv[1]); assert root.parent.is_dir(), 'parent missing'
arm=pathlib.Path('/root/autodl-tmp/c4/v5/main14_arm')
if str(root).startswith('/root/') and arm.is_dir() and (arm/'waiter.pid').exists():
 p=(arm/'waiter.pid').read_text().strip()
 assert not pathlib.Path('/proc/'+p).exists(), 'already armed; refuse source mutation'
root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as t:
 for member in t:
  assert member.isfile() and (root/member.name).resolve().is_relative_to(root.resolve())
  t.extract(member,root)
print(json.dumps({'root':str(root),'staged':True}))
"""
    print(remote("gpuR", extract, [GPU_CODE], bundle))
    print(remote("cpuX", extract, [CPU_CODE], bundle))
    # The existing tested SUMO fixture is a server-local runtime dependency.  Only link
    # the directory; do not open, copy, hash or inspect vehicle-type assignments.
    print(remote("cpuX", """
import pathlib,json
root=pathlib.Path('/home/ubuntu/c4/repo-main14')
src=pathlib.Path('/home/ubuntu/sumocredit/code_d/scenarios')
assert src.is_dir()
dst=root/'scenarios'
if not dst.exists(): dst.symlink_to(src,target_is_directory=True)
assert dst.resolve()==src.resolve()
print(json.dumps({'scenario_fixture_directory_linked':True}))
"""))
    cfg = json.loads(git("show", f"{commit}:configs/c4_loop14_main14_gpur.json"))
    config_text = json.dumps(cfg, indent=1) + "\n"
    config_sha = hashlib.sha256(config_text.encode()).hexdigest()
    reg = json.loads(git("show", f"{commit}:configs/c4_main14_registration.json"))
    reg["sources"].update(code_sha=commit, config_sha256=config_sha,
                          source_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())
    reg["registered_at"] = datetime.now(timezone.utc).isoformat()
    reg_text = json.dumps(reg, indent=1) + "\n"
    reg_sha = hashlib.sha256(reg_text.encode()).hexdigest()
    request = {"code_sha": commit, "registration": reg_text, "config": config_text,
               "registration_sha256": reg_sha, "config_sha256": config_sha, "source_files": manifest}
    out = remote("gpuR", """
import json,os,pathlib,sys,hashlib
v5=pathlib.Path('/root/autodl-tmp/c4/v5'); assert v5.is_dir()
req=json.load(sys.stdin); arm=v5/'main14_arm'; arm.mkdir(exist_ok=True)
for name in ('WAIT_OWNER.json','LAUNCHED.json','waiter.pid'):
 assert not (arm/name).exists(), 'prior arm exists; manual inspection required'
for key,name in [('registration','main14_REGISTRATION.json'),('config','main14_config.json')]:
 (v5/name).write_text(req[key])
fields=pathlib.Path('/proc/636520/stat').read_text().rsplit(')',1)[1].split()
record={'schema':'main14_arm/1','code_sha':req['code_sha'],'source_files':req['source_files'],
 'registration':str(v5/'main14_REGISTRATION.json'),'registration_sha256':req['registration_sha256'],
 'config':str(v5/'main14_config.json'),'config_sha256':req['config_sha256'],
 'round14b_start_ticks':fields[19],
 'round14b_summary':str(v5/'runs/c4_wm_v10_round14_b/RUN_SUMMARY.json'),
 'verdict_deadline':'2026-10-26T13:00:00+08:00'}
(arm/'ARMED.json').write_text(json.dumps(record,indent=1)+'\\n')
(v5/'repo_main14/MAIN14_SOURCE.json').write_text(json.dumps({'commit':req['code_sha'],'files':req['source_files']},indent=1))
print(json.dumps({k:v for k,v in record.items() if k!='source_files'}))
""", data=json.dumps(request).encode())
    receipt = json.loads(out)
    receipt.update(exported_files=len(manifest), bundle_bytes=len(bundle))
    (ROOT / "tools/main14/deployment_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    (ROOT / "tools/main14/REGISTERED.json").write_text(reg_text)
    print(json.dumps(receipt, indent=1))


def arm():
    print(remote("gpuR", """
import json,os,pathlib,subprocess,time
v5=pathlib.Path('/root/autodl-tmp/c4/v5'); arm=v5/'main14_arm'; code=v5/'repo_main14'
assert not (arm/'waiter.pid').exists(), 'already armed'
argv=['/root/miniconda3/bin/python',str(code/'scripts/c4/main14_autolaunch.py'),'--arm',str(arm/'ARMED.json')]
subprocess.run([*argv,'--check-only'],cwd=code,check=True)
with (v5/'main14_autolaunch.log').open('ab') as log:
 p=subprocess.Popen(['setsid','nohup',*argv],cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
time.sleep(3)
assert p.poll() is None, 'waiter died; inspect main14_autolaunch.log'
assert (arm/'waiter.pid').read_text().strip()==str(p.pid)
stat=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()
record={'pid':p.pid,'session':int(stat[3]),'ppid':int(stat[1]),'start_ticks':stat[19],
 'log':str(v5/'main14_autolaunch.log'),'stop_marker':str(arm/'STOP_ARM')}
(arm/'WAITER_INSTALLED.json').write_text(json.dumps(record,indent=1))
print(json.dumps(record))
"""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("stage", "arm"))
    args = p.parse_args()
    stage() if args.action == "stage" else arm()
