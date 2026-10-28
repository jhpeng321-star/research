"""Verify task-owned staged input bytes and validation-role fences."""
import hashlib
import json
from pathlib import Path

WM='b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
BAD=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth')


def check(root,out):
    receipt=json.loads((root/'STAGING.json').read_text());checked=0
    for name,expected in receipt['files'].items():
        p=root/name
        assert p.resolve().is_relative_to(root.resolve())
        assert not any(x in str(p).lower() for x in BAD),p
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,p
        checked+=1
    split=json.loads((root/'e3data/SPLIT.json').read_text())
    m=json.loads((root/'e3data/C4_EPISODES_MANIFEST.json').read_text())
    assert len(m['episodes'])==108 and len({e['world'] for e in m['episodes']})==36
    assert all(split['worlds'][str(e['seed'])]=='validation' and not 964000<=e['seed']<=964199
               for e in m['episodes'])
    assert {e['arm'] for e in m['episodes']}=={'native','mappo','ours_c1'}
    assert hashlib.sha256((root/'wm2/wm_s.pt').read_bytes()).hexdigest()==WM
    doc={'status':'PASS','checked_staged_files':checked,'role':'validation','worlds':36,'episodes':108,
         'wm_sha256':WM,'split_sha256':hashlib.sha256((root/'e3data/SPLIT.json').read_bytes()).hexdigest(),
         'filtered_manifest_sha256':hashlib.sha256((root/'e3data/C4_EPISODES_MANIFEST.json').read_bytes()).hexdigest(),
         'scope':'Only predeclared staged source/code/array files read; no driver-type assignment or truth files.'}
    out.mkdir(exist_ok=True);(out/'INPUTS_VERIFIED.json').write_text(json.dumps(doc,indent=1));return doc


if __name__=='__main__':
    root=Path.home()/'lrdiag';print(json.dumps(check(root,root/'evidence')))
