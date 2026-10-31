"""Consume the completed S-RB-FIT reader/artifact; no fitting or new data reads."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch

from .operate import RECEIPTS


def main():
    out = RECEIPTS/'MOTION_POINT_TABLE.json'
    assert out.parent.is_dir() and not out.exists()
    torch.set_num_threads(1)
    helper = Path('E:/agent-work/c4-rb-fit/src/wms/physics/motion_fit.py')
    root = Path('E:/agent-work/c4-rb-fit/artifacts/rbfit/finite_mixture_v4')
    assert (root/'idm_posteriors.json').stat().st_size < 2_000_000_000
    provenance = json.loads((root/'provenance.json').read_text())
    assert {r['world'] for r in provenance['sources']} == set(range(969400,969432))
    bindings = {}
    for row in provenance['sources']:
        assert row['episode'].endswith('__native')
        old = json.loads((RECEIPTS/'panel9_results'/(row['episode']+'.json')).read_text())
        assert row['sha256'] == old['input_sha256']
        bindings[row['episode']] = row['sha256']
    spec = importlib.util.spec_from_file_location('rblane_published_motion_reader',helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # completed OWN project reader, not downloaded code
    reader = module.MotionFit(root)
    assert reader.posterior['conditioning_end_s'] == 300
    assert reader.population['delta'] == 4
    worlds = {int(key.split(':')[0]) for key in reader.posterior['drivers']}
    assert worlds == set(range(969400,969432))
    drivers = {}
    order = module.IDM_ORDER
    for world in sorted(worlds):
        ids = sorted(int(key.split(':')[1]) for key in reader.posterior['drivers'] if key.startswith(str(world)+':'))
        values = reader.idm(world,torch.tensor(ids),t0_s=300,sample=False,use_posterior=True)
        for index, vid in enumerate(ids):
            drivers[f'{world}:{vid}'] = {key:float(values[key][index]) for key in order}
    first = next(iter(reader.posterior['drivers']))
    world,vid = map(int,first.split(':'))
    pop = reader.idm(world,torch.tensor([vid]),t0_s=300,sample=False,use_posterior=False)
    result = dict(schema='rblane.motion_point_table.v1',conditioning_end_s=300,
        fit_worlds=sorted(worlds),native_only=True,parameter_order=list(order),
        point_rule='published MotionFit.idm(sample=False); modal_parameters when supplied',
        population={key:float(pop[key][0]) for key in order},drivers=drivers,
        input_sha256=bindings,helper_source=str(helper),
        helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        population_sha256=hashlib.sha256((root/'idm_population.json').read_bytes()).hexdigest(),
        posterior_sha256=hashlib.sha256((root/'idm_posteriors.json').read_bytes()).hexdigest(),
        training_performed=False)
    assert all(v>0 and math.isfinite(v) for row in [result['population'],*drivers.values()] for v in row.values())
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),drivers=len(drivers),worlds=len(worlds),
        population=result['population'],helper_sha256=result['helper_sha256'],training_performed=False),indent=2))


if __name__ == '__main__':
    main()
