"""Preserve a readable, byte-bound final-only selector delivery for admission."""
import json

from tools.c1switch.design import checked, sha
from tools.c1switch.storage import write
from .operations import EVIDENCE, ssh


def per_cav_specs(value, frozen_sha256):
    if (value['status'] != 'COMPLETE' or value['iteration'] != 169 or
            value['variant'] != 'per_cav' or value['host'] != 'cpuQ' or
            value['local_evaluation_sim_s'] != 0 or value['evaluation_owner'] != 'S-C1SWITCH2' or
            value['panel_sha256'] != '7c780c25237dde73a657937ab577b8e48f71766817f4363f3298982e90144827' or
            value['checkpoint_sha256'] != value['frozen']['checkpoint_sha256']):
        raise ValueError('genuine final169 on the original peer-owned panel required')
    packages = {}
    for key, folder in [('shared', 'selmappo'), ('per_cav', 'selmappo2')]:
        prefix = 'tools/'+folder+'/'
        sources = {n.removeprefix(prefix): v for n, v in value['sources'].items()
                   if n.startswith(prefix) and n.endswith('.py')}
        if '__init__.py' not in sources or any('/' in n for n in sources):
            raise ValueError('complete flat Python peer package required')
        packages[key] = dict(status='FROZEN_FINAL', root=value['source_root']+'/'+prefix.rstrip('/'), sources=sources)
    for stem in ('observations', 'learning', 'protocol', 'runtime', 'environment'):
        if packages['shared']['sources'][stem+'.py'] != value['frozen']['sources'][stem]:
            raise ValueError('training and execution sources differ')
    if packages['per_cav']['sources']['mixed.py'] != value['frozen']['sources']['selmappo2.mixed']:
        raise ValueError('mixed executor differs from training')
    spec = dict(packages['shared'], checkpoint=value['checkpoint'],
                checkpoint_sha256=value['checkpoint_sha256'],
                frozen_receipt=value['root']+'/training/FROZEN_FINAL.json',
                frozen_receipt_sha256=frozen_sha256, checkpoint_sources=value['frozen']['sources'],
                checkpoint_bindings=value['frozen']['bindings'])
    return spec, packages


def load_per_cav():
    value = json.loads(checked('SELMAPPO2_FINAL.json').read_text())
    # Validate the delivery before opening any remote path, then attest only its
    # final checkpoint/freeze and declared Python sources. No training restart.
    spec, packages = per_cav_specs(value, '')
    proof = ssh('cpuQ', f'''import hashlib,json
from pathlib import Path
spec={spec!r};packages={packages!r};freeze=Path(spec['frozen_receipt'])
assert json.loads(freeze.read_text())=={value['frozen']!r}
assert hashlib.sha256(Path(spec['checkpoint']).read_bytes()).hexdigest()==spec['checkpoint_sha256']
for package in packages.values():
 for name,digest in package['sources'].items():
  assert hashlib.sha256((Path(package['root'])/name).read_bytes()).hexdigest()==digest
print(json.dumps(dict(frozen_receipt_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest(),status='PASS',new_sim_s=0,new_gpu_s=0)))
''', timeout=45)
    return per_cav_specs(value, proof['frozen_receipt_sha256'])


def preserve():
    path = checked('SELMAPPO2_FINAL.json')
    value = json.loads(path.read_text())
    if (value['frozen']['iteration'] != 169 or
            value['checkpoint_sha256'] != value['frozen']['checkpoint_sha256'] or
            value['evaluation_owner'] != 'S-C1SWITCH2'):
        raise ValueError('genuine final169 peer-owned evaluation delivery required')
    target = EVIDENCE/'SELECTOR_DELIVERY_cav_mappo.json'
    if target.exists():
        raise ValueError('preserve existing selector delivery receipt')
    write(target, dict(delivery_sha256=sha(path), delivery=value, new_sim_s=0, new_gpu_s=0))
    print(json.dumps(dict(path=str(target), delivery_sha256=sha(path))))


if __name__ == '__main__':
    preserve()
