"""gpuR-resident, signal-free preparation of collector runtimes and scored base packages."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

V5 = Path('/root/autodl-tmp/c4/v5')
A = V5 / 'mainstart'
CODE = V5 / 'repo_main14'
sys.path[:0] = [str(CODE), str(CODE / 'scripts/c4')]
from tools.main14 import verdict_snapshot as verdict
import loop14 as loop


def query(alias, script, args=()):
    command = ' '.join(map(shlex.quote, ['/usr/bin/python3', '-c', script, *args]))
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', alias, command],
                       capture_output=True, text=True, timeout=120, check=False)
    if p.returncode:
        raise RuntimeError(f'{alias}: {p.stderr[-500:]}')
    return json.loads(p.stdout)


def push(alias, source, destination):
    subprocess.run(['rsync', '-a', '--timeout=120', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
                    str(source), f'{alias}:{destination}'], check=True, timeout=300)


def runtime_bundle(arm):
    target = A / 'collector_runtime.tgz'
    if not target.exists():
        with tarfile.open(target, 'w:gz') as tar:
            for rel, expected in arm['source_files'].items():
                loop.guard(rel)
                if 'vtype' in rel.lower() or Path(rel).suffix == '.xml':
                    raise RuntimeError('assignment file in registered source manifest')
                path = CODE / rel
                if path.is_symlink() or loop.sha256(path) != expected:
                    raise RuntimeError(f'native source changed: {rel}')
                tar.add(path, arcname=rel, recursive=False)
    return target


PREPARE = r'''
import json,pathlib,sys,tarfile,hashlib,subprocess,os
h=pathlib.Path.home(); root=h/'c4/main14'; assert root.is_dir()
archive=root/'collector_runtime.tgz'; sha=sys.argv[1]
assert hashlib.sha256(archive.read_bytes()).hexdigest()==sha
runtime=root/'runtime'; manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
if not runtime.exists():
 runtime.mkdir()
 with tarfile.open(archive,'r:gz') as tar:
  for m in tar:
   assert m.isfile() and (runtime/m.name).resolve().is_relative_to(runtime.resolve())
   tar.extract(m,runtime)
for rel,sha in manifest['files'].items():
 p=runtime/rel; assert not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==sha
fixture=h/'aq_code/code_r13e_65cc714e/scenarios'
assert fixture.is_dir(), 'existing tested fixture directory not ready'
link=runtime/'scenarios'
if not link.exists(): link.symlink_to(fixture,target_is_directory=True)
assert link.resolve()==fixture.resolve()
assert (h/'newcpu/SETUP_READY').is_file(), 'scoring environment not ready'
py=h/'miniconda3/bin/python'
probe="import json,torch,numpy,libsumo,traci,sumolib; from traffic_shepherd.config import load_config; c=load_config(); assert c.network_path.is_file() and c.scenario_template_path.is_file(); print(json.dumps({'torch':torch.__version__,'numpy':numpy.__version__,'network':str(c.network_path),'template':str(c.scenario_template_path)}))"
env={**os.environ,'PYTHONPATH':str(runtime/'src')+':'+str(runtime),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
p=subprocess.run([str(py),'-c',probe],capture_output=True,text=True,env=env,cwd=runtime)
assert p.returncode==0,p.stderr[-1000:]
receipt={'source_manifest_sha256':manifest['sha256'],'source_files':manifest['files'],
 'fixture_verified_without_new_sumo':True,'fixture_check':'directory link, configured file presence and existing registered fixture; assignment contents never opened by preparation',
 'dependencies':json.loads(p.stdout),'new_sumo_sim_s':0}
(root/'COLLECTION_READY.json').write_text(json.dumps(receipt,indent=1))
print(json.dumps({'ready':True,'dependencies':receipt['dependencies']}))
'''


def prepare_collectors(arm):
    bundle = runtime_bundle(arm)
    source = A / 'COLLECTOR_SOURCE.json'
    if not source.exists():
        loop.write_atomic(source, {'sha256': arm['native_source_manifest_sha256'], 'files': arm['source_files']})
    for alias in ('cpuP', 'cpuQ'):
        try:
            query(alias, "import pathlib,json; h=pathlib.Path.home(); assert (h/'c4').is_dir(); (h/'c4/main14').mkdir(exist_ok=True); print('{}')")
            push(alias, bundle, '/home/ubuntu/c4/main14/collector_runtime.tgz')
            push(alias, source, '/home/ubuntu/c4/main14/SOURCE_MANIFEST.json')
            receipt = query(alias, PREPARE, (loop.sha256(bundle),))
            loop.write_atomic(A / f'COLLECTOR_{alias}.json', receipt)
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(json.dumps({'event': 'collector_setup_wait', 'host': alias, 'error': str(error)}), flush=True)


SCORED = r'''
import pathlib,json,sys,hashlib
h=pathlib.Path.home(); label=sys.argv[1]
q=h/f'actqual/runs/{label}/QUAL_{label}.json'
if not q.is_file(): q=h/f'autopilot_r10/runs/{label}/QUAL_{label}.json'
assert q.is_file(), 'complete qualification absent'
ck=h/f'autopilot_r10/runs/{label}/ck/wm_s_best.pt'; assert ck.is_file()
configs=[ck.parent/'config.resolved.json',ck.parent/'config.json',ck.parent/'config.resolved.toml']
config=next((p for p in configs if p.is_file()),None)
assert config is not None, 'FINAL config absent'
sha=hashlib.sha256(ck.read_bytes()).hexdigest()
print(json.dumps({'label':label,'qualification':json.loads(q.read_text()),'qualification_path':str(q),
 'qualification_sha256':hashlib.sha256(q.read_bytes()).hexdigest(),
 'checkpoint':str(ck),'scored_checkpoint_sha256':sha,'config':str(config),
 'sidecars':[str(p) for p in [ck.parent/'physics.json',ck.parent/'TRAIN_RECEIPT.json',ck.parent/'lane_calib.json'] if p.is_file()]}))
'''


def stage_scored(alias, label, public_label):
    snap = query(alias, SCORED, (label,))
    qual = verdict.select_label(snap['qualification'], label)
    verdict.reaction_items(qual)
    verdict.strict_ratio(qual)
    if label == 'r14b' and loop.sha256(V5 / 'runs/c4_wm_v10_round14_b/wm_s.pt') != snap['scored_checkpoint_sha256']:
        raise RuntimeError('14b scored checkpoint differs from gpuR FINAL')
    if label == 'r16':
        actual = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'gpuB',
                                         'sha256sum /root/autodl-tmp/c4/v5/runs/c4_wm_v10_round16_b/wm_s.pt'],
                                        text=True, timeout=90).split()[0]
        if actual != snap['scored_checkpoint_sha256']:
            raise RuntimeError('round16 scored checkpoint differs from gpuB FINAL')
    dest = A / 'bases' / label / snap['scored_checkpoint_sha256']
    dest.mkdir(parents=True, exist_ok=True)
    qpath = dest / 'QUAL.json'
    loop.write_atomic(qpath, snap['qualification'])
    for src in [snap['checkpoint'], snap['config'], *snap['sidecars']]:
        loop.guard(src)
        subprocess.run(['rsync', '-a', '--timeout=120', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
                        f'{alias}:{src}', str(dest) + '/'], check=True, timeout=300)
    ckpt = dest / Path(snap['checkpoint']).name
    if loop.sha256(ckpt) != snap['scored_checkpoint_sha256']:
        raise RuntimeError(f'{label} FINAL hash mismatch after copy')
    loop.write_atomic(dest / 'SCORE_PROVENANCE.json', {'host': alias, **snap})
    return {'label': public_label, 'score_label': label, 'checkpoint': str(ckpt),
            'scored_checkpoint_sha256': snap['scored_checkpoint_sha256'],
            'config': str(dest / Path(snap['config']).name), 'qualification_path': str(qpath)}


def stage_bases():
    reference = stage_scored('cpuP', 'r13e', '13E')
    if reference['scored_checkpoint_sha256'] != '952ec059de5a9c0285d2327bcea57c836fd5d736c9975592ffbb932a506e84c8':
        raise RuntimeError('regenerated 13E qualification uses different checkpoint')
    candidates = []
    for alias, label, public in [('cpuP', 'r14b', '14b'), ('cpuQ', 'r16', 'round16')]:
        try:
            candidates.append(stage_scored(alias, label, public))
        except (RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
            print(json.dumps({'event': 'candidate_not_ready', 'label': label, 'error': str(error)}), flush=True)
    if candidates:
        loop.write_atomic(A / 'BASE_CANDIDATES.json', {'reference_qualification': reference['qualification_path'],
                          'reference_checkpoint_sha256': reference['scored_checkpoint_sha256'], 'candidates': candidates})


def main():
    arm = loop.read_json(A / 'ARMED.json')
    loop.write_atomic(A / 'HANDOFF_PID.json', {'pid': os.getpid(), 'sid': os.getsid(0)})
    while not (A / 'STOP_HANDOFF').exists() and not (A / 'STOP_ARM').exists():
        prepare_collectors(arm)
        try:
            stage_bases()
        except (RuntimeError, ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
            print(json.dumps({'event': 'base_verdict_wait', 'error': str(error)}), flush=True)
        if (Path('/root/main14_loop/LOOP14_STATE.json')).exists():
            state = loop.read_json('/root/main14_loop/LOOP14_STATE.json')
            if state.get('stopped') or (A / 'WAIT_OWNER.json').exists():
                return
        time.sleep(120)


if __name__ == '__main__':
    main()
