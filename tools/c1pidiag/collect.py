"""Write-once compact export after all inference phases have completed."""
from pathlib import Path
import gzip
import hashlib
import json
import argparse
import platform
import sys
import time

ROOT=Path('/root/autodl-tmp/c4/c1pidiag20261031')


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--primary',action='store_true');args=ap.parse_args()
    phases=(('original','units_v2','evidence_v2/TF_DONE.json'),
            ('original','feedback_units','feedback_evidence/COMPLETE.json'),
            ('aux','aux_units','aux_evidence/COMPLETE.json'),
            ('original','extra_units','extra_evidence/COMPLETE.json'),
            ('legacy','legacy_units','legacy_evidence/COMPLETE.json'))
    if args.primary:phases=phases[:2]
    for _,_,e in phases:
        done=json.loads((ROOT/e).read_text())
        assert not done['errors']
        if 'sources' in done:
            assert all(digest(Path(p))==h for p,h in done['sources'].items()),'source changed after scoring'
    target=ROOT/('RAW_PRIMARY.jsonl.gz' if args.primary else 'RAW_UNITS.jsonl.gz')
    assert not target.exists()
    manifest={};counts={}
    with target.open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz:
            for panel,folder,_ in phases:
                paths=sorted((ROOT/folder).glob('*.json'))
                counts[folder]=0
                for p in paths:
                    doc=json.loads(p.read_text())
                    if folder=='units_v2' and doc['mode']!='tf':continue
                    assert not doc['error'],p
                    counts[folder]+=1
                    doc['_panel']=panel;doc['_source']=str(p)
                    gz.write(json.dumps(doc,allow_nan=False,separators=(',',':')).encode()+b'\n')
                    manifest[str(p.relative_to(ROOT))]=dict(bytes=p.stat().st_size,sha256=digest(p))
    import torch
    value=dict(schema='c1pidiag_outputs/1',counts=counts,units=sum(counts.values()),
        raw_sha256=digest(target),raw_bytes=target.stat().st_size,outputs=manifest,
        python=sys.version,torch=torch.__version__,platform=platform.platform(),
        cuda_inference=False,threads_per_worker=1,workers=24,finished=time.time(),
        panel_sha256=digest(ROOT/'PANEL.json'))
    with (ROOT/('OUTPUTS_PRIMARY.json' if args.primary else 'OUTPUTS_MANIFEST.json')).open('x') as f:json.dump(value,f,indent=1)
    receipts={}
    for e in ('evidence','evidence_v2'):
        for p in sorted((ROOT/e).glob('*OWNED.json')):
            receipt=json.loads(p.read_text());pid=receipt['pid'];q=Path(f'/proc/{pid}/stat')
            fields=q.read_text().split(') ',1)[1].split() if q.exists() else None
            current=fields[19] if fields else None
            receipts[str(p.relative_to(ROOT))]=dict(pid=pid,start_ticks=receipt['start_ticks'],
                state=fields[0] if fields else None,
                still_running_with_same_start=current==receipt['start_ticks'] and fields[0] not in ('Z','X'))
    with (ROOT/('PID_PRIMARY.json' if args.primary else 'PID_AUDIT.json')).open('x') as f:json.dump(receipts,f,indent=2)
    print(json.dumps({k:v for k,v in value.items() if k!='outputs'},indent=2))


if __name__=='__main__':main()
