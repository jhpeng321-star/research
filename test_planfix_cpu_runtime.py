"""Protect recovered-config equality and path/device-only transformation."""
import ast
import json
from pathlib import Path
import pytest
from planfix_cpu_runtime import cpu_tree,execute_tree
from planfix_recover_bootstrap_config import recover

def test_original_inspections_agree_with_preserved_recovery():
    recorded=json.loads(Path(__file__).with_name('PLANFIX_BOOTSTRAP_CONFIG_RECOVERED.json').read_text())
    assert recover()==recorded
    assert recorded['independent_config_observations']==4
    assert recorded['config']['cav_exec_cap']=='idm'

def test_path_device_port_preserves_scientific_literals_and_operations():
    source="""def run(x):
    source = Path('/root/wmcalib/data/jobs')
    x = x.to('cuda')
    return (x * .5 + 1.2).clamp_min(0), 300, 960, 8, 4, 'FULL'
"""
    tree=cpu_tree(source,{'cuda':'cpu','/root/wmcalib/data/jobs':'/cpu/priors'})
    expected=ast.parse(source.replace("'/root/wmcalib/data/jobs'","'/cpu/priors'").replace("'cuda'","'cpu'"))
    assert ast.dump(tree)==ast.dump(expected)

def test_unknown_checkpoint_load_abi_refuses_instead_of_substituting():
    with pytest.raises(AssertionError):
        cpu_tree('candidate = torch.load(other_path)',{},recovered_config=True)

def test_cpu_install_maps_start_design_root_and_history_root():
    tree=ast.parse(Path(__file__).with_name('planfix_cpu_runtime.py').read_text())
    install=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
    paths=next(n.value for n in ast.walk(install) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='paths' for t in n.targets))
    assert {k.value for k in paths.keys}=={'/root/wmcalib','/root/wmcalib/data','/root/wmcalib/data/jobs'}

def test_original_introspection_adapter_can_recover_transformed_prepare_source():
    import inspect
    source="def prepare_batch(rt, source):\n    return with_entry_slots(source,300,960,device='cuda')\n"
    namespace={}
    execute_tree(cpu_tree(source,{'cuda':'cpu'}),'<planfix-test-prepare>',namespace)
    recovered=ast.parse(inspect.getsource(namespace['prepare_batch']))
    values=[n.value for n in ast.walk(recovered) if isinstance(n,ast.Constant)]
    assert values.count(300)==1 and values.count(960)==1 and 'cpu' in values and 'cuda' not in values
