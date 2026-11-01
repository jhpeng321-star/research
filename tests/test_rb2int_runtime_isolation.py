"""Exercise the real launcher import boundary in a clean Python subprocess."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tools.rb2int.audit import ROOT


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_recipe_wins_over_new_control_stage_recipe(tmp_path):
    control, runtime = tmp_path/'control', tmp_path/'frozen'
    entry = control/'tools/rb2int/evaluate.py'
    put(entry, (ROOT/'tools/rb2int/evaluate.py').read_text(encoding='utf-8'))
    put(control/'tools/rb2int/inputs.py', 'import json\ndef validate_panel(p): return json.load(open(p))\n')
    put(control/'tools/rbint/recipe.py', 'raise RuntimeError("NEW CONTROL RECIPE LEAKED INTO BASELINE")\n')
    recipe = runtime/'tools/rbint/recipe.py'
    put(recipe, 'def load_candidate(*args): pass\ndef candidate_future(*args): pass\n')
    put(runtime/'tools/rbeval/core.py', '''import hashlib,json
from pathlib import Path
def safe(p): return Path(p)
def load(p): return json.loads(Path(p).read_text())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
''')
    put(control/'tools/rbeval/run_wm.py', '''import inspect,json
def main(**kwargs):
 print(json.dumps(dict(recipe=inspect.getfile(kwargs['model_loader']), sources=[str(p) for p in kwargs['identity_sources']])))
''')
    config = runtime/'config.toml'
    put(config, '# frozen\n')
    checkpoint = runtime/'wm.pt'
    put(checkpoint, 'frozen checkpoint fixture\n')
    release = tmp_path/'release.json'
    put(release, json.dumps(dict(models=dict(example=dict(runtime_root=str(runtime),
        config_sha256=sha(config), wm_sha256=sha(checkpoint), variant='FULL', sources={str(recipe): sha(recipe)})))))
    panel = tmp_path/'panel.json'
    put(panel, json.dumps(dict(release_path=str(release), registration_path=str(tmp_path/'registration.json'))))
    result = subprocess.run([sys.executable, '-B', str(entry), '--variant', 'FULL',
        '--runtime-root', str(runtime), '--wm-config', str(config), '--wm', str(checkpoint), '--model-name', 'example',
        '--registered-panel', str(panel)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout+result.stderr
    output = json.loads(result.stdout)
    assert Path(output['recipe']) == recipe
    assert str(recipe) in output['sources']
