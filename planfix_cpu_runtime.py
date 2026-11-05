"""Device/path-only port of the original WMTEACH runtime, with recovered config.

This operator module never edits the frozen scientific source. The same source
is compiled with explicit path/device literals remapped; only the unused legacy
candidate tensor load becomes its independently recovered saved CONFIG.
"""
import ast
import hashlib
import json
from pathlib import Path

RUNTIME_SHA='2893e762741a61a791ce467493e0a35290e6ad4c893e2909b70bfa0fe1df50d8'

def cpu_tree(text, mapping, *, recovered_config=False):
    tree=ast.parse(text)
    class Port(ast.NodeTransformer):
        recovered=0
        allocations=0
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in mapping:
                return ast.copy_location(ast.Constant(mapping[node.value]),node)
            return node
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='torch.cuda.set_per_process_memory_fraction':
                self.allocations+=1
                return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if recovered_config and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='candidate':
                assert isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='torch.load'
                assert len(node.value.args)==1 and ast.unparse(node.value.args[0])=="BASE / 'setupaudit20261030/policy_000190.pt'"
                self.recovered+=1
                return ast.copy_location(ast.parse("candidate = {'config': RECOVERED_CONFIG}").body[0],node)
            return self.generic_visit(node)
    port=Port();tree=port.visit(tree);ast.fix_missing_locations(tree)
    if recovered_config:
        assert port.recovered==1 and port.allocations==1, 'frozen bootstrap ABI changed'
    return tree

def install(worker):
    import sys,importlib
    worker=Path(worker);physical=worker/'runtime'
    priors=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix/cpu_priors_97fa5403964f')
    recovered=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix/cpu_bootstrap_recovered')
    old=Path('/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa')
    sys.path[:0]=[str(physical/'src'),str(physical),str(old/'pytools')]
    import tools
    tools.__path__.insert(0,str(physical/'tools'))
    # observations has two frozen absolute path literals, neither changes any
    # identities, fit windows, posterior construction or allowed-world checks.
    observations=importlib.import_module('tools.rbint.observations')
    paths={'/root/wmcalib':str(priors),'/root/wmcalib/data/jobs':str(priors/'data/jobs')}
    source=Path(observations.__file__).read_text()
    exec(compile(cpu_tree(source,paths),'<planfix-cpu-observable-paths>','exec'),observations.__dict__)
    from tools.wmteach import runtime
    text=Path(runtime.__file__).read_text()
    assert hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest()==RUNTIME_SHA
    values=json.loads((recovered/'RECOVERED.json').read_text())
    binding=dict(values['original_export'],path=values['original_export']['source'])
    destination=worker/'EXPORT_BOUND_CPU.json'
    if destination.exists():assert json.loads(destination.read_text())==binding
    else:
        with destination.open('x') as out:json.dump(binding,out,sort_keys=True)
    mapping=dict(paths,**{
        'cuda':'cpu',
        '/root/rbeval20261031/phys_code_22cc32f65a80':str(physical),
        '/root/autodl-tmp/c4/setupaudit20261030/fix':str(worker/'fix'),
        '/root/autodl-tmp/c4':str(worker/'legacy'),
        '/root/autodl-tmp:/root/wmcalib:/root/rbeval20261031':'/home/ubuntu',
        'wmcombined_wm2/wm_s.pt':'/home/ubuntu/c4/main14/restart20261026/work/w968195/C4-LOOP14-COOPON-20261026/c03.a2/inputs/wm/wm_s.pt',
        'speedhead20261030/EXPORT_BOUND.json':str(destination),
    })
    runtime.RECOVERED_CONFIG=values['config']
    tree=cpu_tree(text,mapping,recovered_config=True)
    exec(compile(tree,'<planfix-cpu-original-wmteach>','exec'),runtime.__dict__)
    return runtime
