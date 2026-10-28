"""Inspect observable insertion/controller metadata, without running rollouts."""
import sys
from remote import ssh

ssh(sys.argv[1], '''
import sys,json
from pathlib import Path
p=Path.home()/'entrant';sys.path[:0]=[str(p/'runtime'),str(p/'runtime/src')]
from tools.lrdiag.replay import build_source
from wms.model.entries import EntranceHistory
for panel in ('e3','pairs300'):
    source,manifest=build_source(p/('e3data' if panel=='e3' else 'pairs300'),panel)
    print('PANEL',panel,source.receipt)
    records=source.group_records(1,source.groups[1]['rows'],base_row=0)
    for r in records:
        print(r['meta'],'queue',r.get('entry_queue'),'feature cav',EntranceHistory.from_features(r['entry_feat']).cav_share)
        if 'entry_feat_control' in r: print('control cav',EntranceHistory.from_features(r['entry_feat_control']).cav_share)
        print('algorithms',set(r['action_timeline']['controller'].reshape(-1).tolist()))
for f in (p/'results').glob('*.json'):
    r=json.loads(f.read_text());print('RECEIPT',f.name,'repeat',r.get('repeat_identical'),'audit',r.get('controller_audit'))
''')
