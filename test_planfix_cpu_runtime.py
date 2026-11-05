"""Protect recovered-config equality and path/device-only transformation."""
import ast
import json
from pathlib import Path
import pytest
from planfix_cpu_runtime import cpu_tree
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
