"""Copy the existing frozen WM2 checkpoint for CPU encoder inference only.

Exact allowlisted model asset on gpuA; no scenario/driver files, no training,
no model rollout. Download into a NEW EMPTY approved temporary directory.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .operate import RECEIPTS

SOURCE = '/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt'
LOCAL = Path('E:/Temp/opencode/rblane-wm2-encoder')


def main():
    receipt = RECEIPTS / 'ENCODER_SOURCE.json'
    assert receipt.parent.is_dir() and not receipt.exists()
    assert LOCAL.parent.is_dir() and not LOCAL.exists()
    code = f'''import os,pathlib,hashlib,json
os.environ['CUDA_VISIBLE_DEVICES']=''
assert os.getloadavg()[0]<20
import torch
torch.set_num_threads(1)
p=pathlib.Path({SOURCE!r})
payload=torch.load(p,map_location='cpu',weights_only=True)
assert payload['format']=='wm_s/1'
cfg=payload['config']; state=payload['model']
encoder={{k:list(v.shape) for k,v in state.items() if k.startswith('driver_encoder.')}}
assert encoder
print(json.dumps(dict(source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    bytes=p.stat().st_size,load=os.getloadavg(),
    config={{k:cfg.get(k) for k in ('history_s','history_stride_s','history_features',
                                    'driver_param_mode','rollout_mode')}},
    encoder_shapes=encoder)))
'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        'gpuA', '/root/miniconda3/bin/python -B -'], input=code, text=True,
        capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    source = json.loads(result.stdout)
    # The registered WM2 identity, as used by EVAL; refuse a replacement.
    assert source['source_sha256'].startswith('b8797242'), 'not the registered frozen WM2'
    LOCAL.mkdir()
    assert not any(LOCAL.iterdir())
    subprocess.run(['scp', '-q', 'gpuA:' + SOURCE, str(LOCAL / 'wm_s.pt')],
                   check=True, timeout=90)
    assert hashlib.sha256((LOCAL/'wm_s.pt').read_bytes()).hexdigest() == source['source_sha256']
    source.update(local_path=str(LOCAL/'wm_s.pt'), copied_at=time.time(),
                  purpose='frozen motion encoder inference; no WM rollout or training')
    receipt.write_text(json.dumps(source, indent=2) + '\n')
    print(json.dumps(source, indent=2))


if __name__ == '__main__':
    main()
