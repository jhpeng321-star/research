"""Scoped stdlib-only SSH transport. No remote shell quoting of Python source."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/root/autodl-tmp/c4/longcritic"


def ssh(host, source):
    return subprocess.run(["ssh", host, "/root/miniconda3/bin/python -B -"], input=source, text=True,
                          encoding="utf-8", capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["probe", "stage", "deps", "tests", "evidence", "inventory", "inventory_a", "benchmark", "pull_cost", "final_launch", "final_status", "final_pull", "bitid", "lint"])
    args = ap.parse_args()
    if args.action == "probe":
        r = ssh("gpuA", "import torch, pathlib\nprint(torch.__version__)\n"
                "print(pathlib.Path('/root/autodl-tmp/c4').is_dir())\n")
    elif args.action == "stage":
        files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                                        cwd=ROOT, text=True).splitlines()
        # Source-only explicit roots: no archive, no data, no binary checkpoints.
        selected = [p for p in files if p.split('/')[0] in
                    {"src", "traffic_shepherd", "scripts", "tests", "configs", "tools"}
                    and (Path(p).suffix in {".py", ".toml", ".json"} or p == 'tools/longcritic/TRANSFER_RECEIPT.txt')
                    and (ROOT / p).is_file() and not (ROOT / p).is_symlink()
                    and not any(x in p.lower() for x in
                                ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1"))]
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as tar:
            for name in selected:
                # Normalize source newlines across the Windows/Linux boundary.
                data = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        import base64
        blob = base64.b64encode(stream.getvalue()).decode()
        r = ssh("gpuA", "import pathlib,tarfile,io,base64\n"
                "assert pathlib.Path('/root/autodl-tmp/c4').is_dir()\n"
                f"root=pathlib.Path('{REMOTE}/runtime'); root.mkdir(parents=True,exist_ok=True)\n"
                f"tarfile.open(fileobj=io.BytesIO(base64.b64decode('{blob}')),mode='r:gz').extractall(root,filter='data')\n"
                f"print('STAGED {len(selected)} source files')\n")
    elif args.action == "deps":
        r = ssh("gpuA", "import os,pathlib,subprocess,sys\n"
                f"root=pathlib.Path('{REMOTE}'); assert root.is_dir()\n"
                "(root/'tmp').mkdir(exist_ok=True); os.environ['TMPDIR']=str(root/'tmp')\n"
                "sys.exit(subprocess.call([sys.executable,'-m','pip','install','--no-cache-dir',"
                "'--target',str(root/'deps'),'pytest','ruff']))\n")
    elif args.action == "tests":
        r = ssh("gpuA", f"import os,sys,pathlib\nos.chdir('{REMOTE}/runtime')\n"
                f"tmp=pathlib.Path('{REMOTE}/tmp'); tmp.mkdir(exist_ok=True); os.environ['TMPDIR']=str(tmp)\n"
                "os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',"
                "PYTHONDONTWRITEBYTECODE='1',C4_GPU_MEM_FRACTION='0.3')\n"
                f"sys.path[:0]=['{REMOTE}/deps','src','.']\nimport torch; torch.set_num_threads(4)\n"
                "import pytest\nsys.exit(pytest.main(['-q','-ra','-p','no:cacheprovider',"
                "'tests/unit/test_longcritic.py','tests/unit/test_longcritic_followup.py','tests/unit/test_coopw.py',"
                "'tests/unit/test_policy14_critic.py','tests/unit/test_sumocredit.py',"
                "'tests/unit/test_sumocredit_kcheck.py','tests/unit/test_fastloop.py',"
                "'tests/unit/test_c4_dreamer.py','tests/unit/test_loop14.py','tests/unit/test_loop14c.py',"
                "'tests/unit/test_policy14_integration.py','tests/unit/test_policy14e.py',"
                "'tests/unit/test_policy14_lanes.py']))\n")
    elif args.action == "bitid":
        names = subprocess.check_output(["git", "diff", "--diff-filter=M", "--name-only", "d6a426c6"], cwd=ROOT, text=True).splitlines()
        originals = {name: subprocess.check_output(["git", "show", f"d6a426c6:{name}"], cwd=ROOT).decode()
                     for name in names if name.endswith('.py')}
        snapshot = '''import importlib.util,pathlib,sys,torch
root=pathlib.Path.cwd(); sys.path[:0]=[str(root/'src'),str(root)]
torch.set_num_threads(4)
def load(name):
 spec=importlib.util.spec_from_file_location(name,root/'tests/unit'/(name+'.py'))
 m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
D=load('test_c4_dreamer'); C=load('test_coopw'); L=load('test_longcritic')
from wms.control.coopw import ExplorationConfig
out={}
for kind in ('plain','coop','real_wm'):
 D._seed(965140)
 t=(D._trainer() if kind=='plain' else L._real_wm_trainer()[0] if kind=='real_wm'
    else C.trainer_for(C.coop_reward(1.,1.)))
 if kind=='coop': t.exploration=ExplorationConfig(enabled=True,rho=[.5])
 metrics=[]
 for _ in range(2):
  m=t.update(); m.pop('wall_time_s'); metrics.append(m)
 out[kind]={n:getattr(t,n).state_dict() for n in ('policy','group_encoder','critic','target_critic','actor_optimizer','critic_optimizer')}
 out[kind]['metrics']=metrics; out[kind]['rng']=torch.get_rng_state()
torch.save(out,sys.argv[1])
'''
        r = ssh("gpuA", "import pathlib,shutil,json,subprocess,os,sys\n"
                f"root=pathlib.Path('{REMOTE}'); parent=root/'parent'; source=root/'runtime'\n"
                "shutil.copytree(source,parent,dirs_exist_ok=True)\n"
                f"originals=json.loads({json.dumps(json.dumps(originals))})\n"
                "for name,text in originals.items(): (parent/name).write_text(text)\n"
                "os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')\n"
                "os.environ['PYTHONPATH']=str(root/'deps'); os.environ['TMPDIR']=str(root/'tmp')\n"
                "for name,code in [('parent',parent),('current',source)]:\n"
                f" subprocess.run([sys.executable,'-B','-c',{snapshot!r},str(root/(name+'.pt'))],check=True,cwd=code)\n"
                "import torch\na=torch.load(root/'parent.pt',weights_only=True); b=torch.load(root/'current.pt',weights_only=True)\n"
                "count=0\ndef equal(a,b,path=''):\n"
                " global count\n"
                " if torch.is_tensor(a):\n"
                "  assert torch.equal(a,b),path; count+=1\n"
                " elif isinstance(a,dict):\n"
                "  assert a.keys()==b.keys(),path\n"
                "  for k in a: equal(a[k],b[k],path+'/'+str(k))\n"
                " elif isinstance(a,(list,tuple)):\n"
                "  assert len(a)==len(b),path\n"
                "  for i,(x,y) in enumerate(zip(a,b)): equal(x,y,path+'/'+str(i))\n"
                " else: assert a==b,(path,a,b)\n"
                "equal(a,b)\nprint(json.dumps({'parent':'d6a426c6','exact_tensor_comparisons':count,'updates_per_fixture':2,'fixtures':['plain','coop','real_wm'],'status':'PASS'}))\n")
    elif args.action == "lint":
        names = subprocess.check_output(["git", "diff", "--name-only", "d6a426c6"], cwd=ROOT, text=True).splitlines()
        names += ['src/wms/control/longcritic.py','tests/unit/test_longcritic.py','tools/longcritic/remote.py',
                  'tools/longcritic/benchmark.py','src/wms/control/real_share.py',
                  'tests/unit/test_longcritic_followup.py','tools/longcritic/prepare_inputs.py',
                  'tools/longcritic/prepare_policy.py','tools/longcritic/prepare_native.py']
        r = ssh("gpuA", "import subprocess,os,sys\n"
                f"os.chdir('{REMOTE}/runtime')\nos.environ['PYTHONPATH']='{REMOTE}/deps'\n"
                f"sys.exit(subprocess.call([sys.executable,'-m','ruff','check','--no-cache','--select','E4,E7,E9,F','--per-file-ignores','traffic_shepherd/c4/collect.py:E402',*{[n for n in names if n.endswith('.py')]!r}]))\n")
    elif args.action == 'final_launch':
        r = ssh('gpuA', "import pathlib,os,subprocess,sys,json\n"
                f"root=pathlib.Path('{REMOTE}'); assert root.is_dir(); assert not (root/'BENCHMARK_FINAL_LAUNCH.json').exists()\n"
                "env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',"
                "'PYTHONDONTWRITEBYTECODE':'1','C4_GPU_MEM_FRACTION':'0.3','WMS_SPEED_PIVOT_DIAG_OFF':'1','WMS_SPEED_GRAD_AUDIT_EVERY':'10',"
                f"'PYTHONPATH':'{REMOTE}/runtime/src:{REMOTE}/runtime','C4_ALLOWED_ROOTS':'/root/autodl-tmp/c4/v5:{REMOTE}',"
                f"'WMS_DATA_V5_NATIVE_Q':'{REMOTE}/data/v5_native_q','WMS_DATA_GENRAND_PLAIN_Q':'{REMOTE}/data/genrand_plain_q'}}\n"
                "argv=[sys.executable,'-B','tools/longcritic/benchmark.py','--transfer-receipt','tools/longcritic/TRANSFER_RECEIPT.txt',"
                "'--wm-ckpt','/root/autodl-tmp/c4/v5/runs/c4_wm_v10_round16_b/wm_s.pt',"
                "'--wm-config','/root/autodl-tmp/c4/v5/runs/c4_wm_v10_round16_b/config.resolved.json',"
                "'--native-returns',str(root/'paired_native_returns.npz'),"
                "'--reliability','tools/longcritic/reliability_r16.json','--calibrate-share',"
                "'--real-returns',str(root/'current_real_returns.npz'),'--policy-state',str(root/'COLLECTED_POLICY_STATE.pt'),"
                "'--output',str(root/'COST_FINAL.json')]\n"
                "with (root/'final_benchmark.log').open('xb') as log:\n"
                " p=subprocess.Popen(argv,cwd=root/'runtime',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\n"
                "receipt={'pid':p.pid,'start_ticks':pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],'argv':argv}\n"
                "(root/'BENCHMARK_FINAL_LAUNCH.json').write_text(json.dumps(receipt,indent=2)); print(json.dumps(receipt))\n")
        if r.returncode == 0:
            (ROOT/'tools/longcritic/BENCHMARK_FINAL_LAUNCH.json').write_text(r.stdout, encoding='utf-8')
    elif args.action == 'final_status':
        r = ssh('gpuA', "import pathlib,json\n"
                f"root=pathlib.Path('{REMOTE}'); d=json.loads((root/'BENCHMARK_FINAL_LAUNCH.json').read_text())\n"
                "p=pathlib.Path('/proc/'+str(d['pid'])+'/stat'); live=p.exists() and p.read_text().split()[21]==d['start_ticks']\n"
                "print(json.dumps({'own_pid_live':live,'result_exists':(root/'COST_FINAL.json').exists()}))\n"
                "print((root/'final_benchmark.log').read_text()[-2000:])\n")
    elif args.action == 'final_pull':
        r = ssh('gpuA', "import pathlib,json\n"
                f"root=pathlib.Path('{REMOTE}'); print(json.dumps({{n:json.loads((root/n).read_text()) for n in ['COST_FINAL.json','REAL_SHARE_MEASURED.json']}}))\n")
        if r.returncode == 0:
            for name, doc in json.loads(r.stdout).items():
                (ROOT/'tools/longcritic'/name).write_text(json.dumps(doc, indent=2), encoding='utf-8')
    elif args.action == 'pull_cost':
        r = ssh('gpuA', f"from pathlib import Path\nprint(Path('{REMOTE}/COST_INITIAL.json').read_text())\n")
        if r.returncode == 0:
            (ROOT/'tools/longcritic/COST_INITIAL.json').write_text(r.stdout, encoding='utf-8')
    elif args.action == "benchmark":
        r = ssh("gpuA", "import os,subprocess,sys\n"
                f"os.chdir('{REMOTE}/runtime')\n"
                f"os.environ.update(PYTHONPATH='{REMOTE}/runtime/src:{REMOTE}/runtime',C4_ALLOWED_ROOTS='/root/autodl-tmp/c4/v5:{REMOTE}',"
                f"WMS_DATA_V5_NATIVE_Q='{REMOTE}/data/v5_native_q',"
                f"WMS_DATA_GENRAND_PLAIN_Q='{REMOTE}/data/genrand_plain_q',"
                "OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',C4_GPU_MEM_FRACTION='0.3',PYTHONDONTWRITEBYTECODE='1')\n"
                "sys.exit(subprocess.call([sys.executable,'-B','tools/longcritic/benchmark.py',"
                "'--transfer-receipt','tools/longcritic/TRANSFER_RECEIPT.txt',"
                "'--wm-ckpt','/root/autodl-tmp/c4/v5/runs/c4_wm_v10_round16_b/wm_s.pt',"
                "'--wm-config','/root/autodl-tmp/c4/v5/runs/c4_wm_v10_round16_b/config.resolved.json',"
                "'--native-returns','/root/autodl-tmp/c4/v5/policy14_stage/policy14_native_returns.npz',"
                "'--reliability','tools/longcritic/reliability_r16.json',"
                f"'--output','{REMOTE}/COST_INITIAL.json']))\n")
    elif args.action == "inventory_a":
        r = ssh("gpuA", "import pathlib,json,os\n"
                "root=pathlib.Path('/root/autodl-tmp/c4/v5')\n"
                "print('DATA_DIRS',[p.name for p in (root/'data').iterdir() if not any(t in p.name.lower() for t in ['sealed','holdout','heldout','cal_split','chunk5','j4b','inc1'])])\n"
                "for rel in ['runs/c4_wm_v10_round16_b','policy14_stage','data/v5_native_q','data/genrand_plain_q']:\n"
                " p=root/rel; print(str(p),p.is_dir())\n"
                " if p.is_dir() and not rel.startswith('data'): print([x.name for x in p.iterdir()])\n"
                "os.environ.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')\n"
                "import numpy as np\np=root/'policy14_stage/policy14_native_returns.npz'\n"
                "if p.exists():\n"
                " with np.load(p,allow_pickle=False) as d: print(json.dumps({k:d[k].tolist() for k in ['source_kind','seeds','horizon_s']}))\n")
    elif args.action == "inventory":
        r = ssh("gpuR", "import pathlib,json\n"
                "p=pathlib.Path('/root/autodl-tmp/c4/v5/mainstart/mainr16/restart/CONFIG.json')\n"
                "d=json.loads(p.read_text()); print(json.dumps({k:d[k] for k in ('policy','collect','wm')},indent=2))\n")
    else:
        r = ssh("gpuR", "import json,pathlib,hashlib\n"
                "root=pathlib.Path('/root/autodl-tmp/c4/v5/mainstart')\n"
                "paths=list(root.glob('horizon_*.json'))+list((root/'mainr16/restart').glob('horizon_*.json'))\n"
                "for p in sorted(paths):\n"
                " data=p.read_bytes(); d=json.loads(data); d.pop('rows',None)\n"
                " print(json.dumps({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'doc':d}))\n")
    if args.action == "evidence" and r.returncode == 0:
        for line in r.stdout.splitlines():
            doc = json.loads(line)
            print(json.dumps({"path": doc["path"], "sha256": doc["sha256"], **doc["doc"]}))
    else:
        print(r.stdout, end="")
    print(r.stderr, end="")
    if args.action in ("tests", "evidence", "bitid", "lint", "benchmark"):
        (ROOT / "tools/longcritic" / (args.action + ".log")).write_text(r.stdout + r.stderr, encoding="utf-8")
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
