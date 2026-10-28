"""Reuse finished LRDIAG OFF trajectories after exact full-horizon parity proof.

The primary 120/300 values and every retained sensor/bin remain untouched. Missing
secondary WM horizons are exact sums of retained HDV bins (a different reduction
order from the old float32 global sum); summarization applies this reduction to
all variants. Recorded targets are recomputed with the original delay function.
"""
import sys
from remote import detach

detach(sys.argv[1], 'reuse_off', '''
import copy,hashlib,json,math,sys
from pathlib import Path
p=Path.home()/'entrant';old=Path.home()/'lrdiag';out=p/'results_v3'
assert p.is_dir() and out.is_dir()
sys.path[:0]=[str(p/'runtime'),str(p/'runtime/src')]
from tools.lrdiag.replay import build_source,WM_SHA
from tools.wm_eval_common import delay_proxy
assert hashlib.sha256((p/'wm2/wm_s.pt').read_bytes()).hexdigest()==WM_SHA
for panel in ('e3','pairs300'):
    source,manifest=build_source(p/('e3data' if panel=='e3' else 'pairs300'),panel)
    sha=hashlib.sha256((p/('e3data' if panel=='e3' else 'pairs300')/'C4_EPISODES_MANIFEST.json').read_bytes()).hexdigest()
    count=0
    for gi,g in enumerate(source.groups):
        # Each queue owns only its registered parity, even when prior evidence is gathered.
        if gi%2!=PARITY:continue
        owner=json.loads((p/'OWNER.json').read_text())
        assert owner['parity']==PARITY
        # All already completed new jobs must still be entrant-ON: OFF is queued,
        # never overwrite an output that a worker may have begun producing.
        assert all(json.loads(f.read_text())['variant'].startswith('entrant_on') or
                   json.loads(f.read_text()).get('reused_from')
                   for f in out.glob('e3_*_s600.json'))
        records=source.group_records(gi,g['rows'],base_row=0)
        real={}
        for record in records:
            arm=record['meta']['arm'];sel=record['present'][0]&~record['is_cav'][0]
            real[arm]={str(h):delay_proxy(record['states'][...,4],record['present'],sel,h)
                       for h in (10,30,60,90,120,180,240,300)}
        for variant in ('baseline','ramp_adapt_off'):
            for draw in range(8):
                name=('' if panel=='e3' else 'p300_')+f'g{gi:02d}_{variant}_k{draw}_s600.json'
                f=old/'results'/name;assert f.is_file(),f
                d=json.loads(f.read_text())
                assert d['pair_key']==g['pair_key'] and d['variant']==variant and d['draw']==draw and d['steps']==600
                target=out/f'{panel}_g{gi:02d}_{variant}_k{draw}_s600.json'
                if target.exists():continue
                for arm,a in d['arms'].items():
                    assert all(a['delay'][h][1]==real[arm][h] for h in ('120','300')),(panel,gi,arm,'real target changed')
                    for h in (10,30,60,90,180,240):
                        wm=math.fsum(r[3] for r in a['timeline'] if r[1]=='HDV' and r[2]<=h)
                        a['delay'][str(h)]=[wm,real[arm][str(h)]]
                d.update(entrant_control=False,controller_specs=None,
                    controller_audit={'entrant_cav_steps':[],'non_native_command_steps':[]},
                    reused_from=str(f),reused_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                    panel_manifest_sha256=sha,secondary_horizons_from_retained_bins=True,
                    implementation_manifest_sha256=hashlib.sha256((p/'SOURCE.json').read_bytes()).hexdigest())
                tmp=target.with_suffix('.reuse.tmp');tmp.write_text(json.dumps(d,allow_nan=False));tmp.replace(target)
                count+=1
    print(panel,'reused',count,flush=True)
receipt={'parity':PARITY,'wm_sha256':WM_SHA,'status':'PASS',
         'primary_120_300_unchanged':True,'secondary_reduction':'math.fsum retained10s HDV bins; recorded targets original delay_proxy'}
(p/'REUSE_OFF.json').write_text(json.dumps(receipt,indent=1));print(json.dumps(receipt))
'''.replace('PARITY',str(int(sys.argv[1]=='cpuQ'))))
