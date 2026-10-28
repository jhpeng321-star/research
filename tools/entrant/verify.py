"""Run focused tests on the staged, CPU-only task runtime."""
import hashlib
import io
from pathlib import Path
import sys
import tarfile
from remote import ROOT, ssh

files = ['src/wms/control/entrant.py', 'src/wms/control/dreamer.py',
         'src/wms/eval/predictor.py', 'tests/unit/test_entrant.py']
stream = io.BytesIO()
with tarfile.open(fileobj=stream, mode='w:gz') as tar:
    for name in files:
        tar.add(ROOT/name, arcname=name, recursive=False)
digest = hashlib.sha256(stream.getvalue()).hexdigest()

ssh(sys.argv[1], '''
import hashlib,importlib.util,io,json,os,sys,tarfile
from pathlib import Path
p=Path.home()/'entrant';runtime=p/'runtime';assert runtime.is_dir()
snapshot=p/'verify'/DIGEST;snapshot.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
    for item in tar:
        assert item.isfile() and (snapshot/item.name).resolve().is_relative_to(snapshot.resolve())
        tar.extract(item,snapshot,filter='data')
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
sys.path[:0]=[str(runtime),str(runtime/'src')]
import wms.control
for short in ('entrant','dreamer'):
    name='wms.control.'+short
    spec=importlib.util.spec_from_file_location(name,snapshot/f'src/wms/control/{short}.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    setattr(wms.control,short,module)
compile((snapshot/'src/wms/eval/predictor.py').read_text(),str(snapshot/'src/wms/eval/predictor.py'),'exec')
import pytest
os.chdir(runtime)
code=pytest.main([str(snapshot/'tests/unit/test_entrant.py'),'-q'])
receipt={'returncode':int(code),'snapshot':str(snapshot),'sha256':DIGEST,
         'files':{str(f.relative_to(snapshot)):hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in snapshot.rglob('*.py')}}
(p/'UNIT_TESTS.json').write_text(json.dumps(receipt,indent=1))
raise SystemExit(code)
'''.replace('DIGEST', repr(digest)), stream.getvalue())
