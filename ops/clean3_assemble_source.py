import hashlib, json, os, shutil, subprocess, tarfile
from pathlib import Path
commit = '162b35c98116a6b7f4cb4af27610868c490a2d35'
up = Path('/home/ubuntu/cpuv_setup_20261105/clean3_upload_162b35c9')
old = Path('/home/ubuntu/coopmwm20261105_clean2_src_285663a1')
src = Path('/home/ubuntu/coopmwm20261106_clean3_src_162b35c9')
assert not src.exists()
src.mkdir()
for name in ('pytools.tgz', 'top.tgz'):
    with tarfile.open(up/name) as t:
        t.extractall(src)
os.symlink(os.readlink(old/'WM_PACKAGE'), src/'WM_PACKAGE')
shutil.copytree(Path(os.readlink(old/'runtime_src')), src/'runtime_src', symlinks=True)
with tarfile.open(up/'runtime.tgz') as t:
    t.extractall(src)  # G5 env.py (traffic_shepherd/c4/env.py) from the commit
shutil.copytree(old/'qualification_baselines', src/'qualification_baselines')
for name in ('F4_CONTINUATION_RECIPE.json', 'FIX_ACCEPTANCE.json', 'pytools/F4_CONTINUATION_RECIPE.json'):
    shutil.copy2(old/name, src/name)
for p in (old/'acceptance').iterdir():
    shutil.copy2(p, src/'acceptance'/p.name)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
fix = json.loads((src/'FIX_ACCEPTANCE.json').read_text())
for k, e in fix.items(): assert sha(src/e['path']) == e['sha256'], k
g = {k: dict(path=f'acceptance/{k}_ACCEPTED.json', sha256=sha(src/f'acceptance/{k}_ACCEPTED.json')) for k in ('G1','G2','G4','G5')}
(src/'G_ACCEPTANCE.json').write_text(json.dumps(g, indent=1, sort_keys=True))
sel = json.loads((old/'TEST_SELECTION.json').read_text())
paths = list(sel['paths']) if isinstance(sel, dict) and 'paths' in sel else list(sel)
paths += ['tools/coopmwm/'+n for n in ('test_executable_labels_cpu.py','test_coordination_opportunities_cpu.py',
  'test_sparse_synergy_credit_cpu.py','test_g4_runtime_cpu.py','test_clean3_budget_cpu.py','test_group_lane_cpu.py',
  'test_g1_certification_cpu.py','test_bank_perseed_cpu.py','test_bank_imitation_cpu.py','test_replay_bank_v2v_cpu.py',
  'test_clean3_restart_cpu.py','test_g5_runtime_cpu.py')]
paths = [p for p in dict.fromkeys(paths) if (src/'pytools'/p).is_file()]
(src/'TEST_SELECTION.json').write_text(json.dumps(dict(paths=paths), indent=1))
files = {}
for p in sorted(src.rglob('*')):
    if p.is_file() and not p.is_symlink() and not any(x.is_symlink() for x in p.relative_to(src).parents if str(x) != '.' and (src/x).is_symlink()):
        rel = str(p.relative_to(src))
        if rel.split('/')[0] == 'WM_PACKAGE' or '__pycache__' in rel: continue
        files[rel] = sha(p)
for base in ('', 'pytools', 'pytools/tools', 'pytools/tests', 'pytools/tests/unit'):
    if (old/base).is_dir():
        for p in (old/base).iterdir():
            if p.is_symlink() and not os.path.lexists(src/base/p.name) and p.name not in ('runtime_src',):
                os.symlink(os.readlink(p), src/base/p.name)
print(json.dumps(dict(files=len(files), tests=len(paths), sel_type=type(sel).__name__)))
(src/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(commit=commit, experiment='clean3', files=files), indent=1, sort_keys=True)); print(hashlib.sha256((src/'SOURCE_MANIFEST.json').read_bytes()).hexdigest())
