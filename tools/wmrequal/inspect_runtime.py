from pathlib import Path
base=Path('/root/autodl-tmp/c4')
for rel in ('overnight20261028/W5_COMBINED/code/common.py',
            'cct2inc20261029/SMOKE_RECEIPT.json',
            'cct2mix20261029/run2.sh'):
    p=base/rel
    print('FILE',p,p.is_file())
    if p.is_file():print(p.read_text()[:12000])
for rel in ('overnight20261028/runtime','overnight20261028/W5_COMBINED/runtime',
            'wmcombined20261027/runtime','restart20261026/runtime_longint',
            'cct2mix20261029/rt_mix'):
    p=base/rel
    print('RUNTIME',p,'src',(p/'src/wms').is_dir())
