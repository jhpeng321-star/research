"""Print only task panel manifests and observable controller metadata."""
import sys
from remote import ssh

ssh(sys.argv[1], '''
import json
from pathlib import Path
p=Path.home()/'entrant'
for panel in ('pairs300','e3data'):
    root=p/panel;doc=json.loads((root/'C4_EPISODES_MANIFEST.json').read_text())
    print(panel,'manifest keys',list(doc),'first episode',doc['episodes'][0])
    if panel=='pairs300': print('random branch',next(e for e in doc['episodes'] if e['branch'].get('policy_params')))
    q=root/doc['episodes'][0]['dir']
    print('panel episode allowed metadata names',[f.name for f in q.iterdir() if f.suffix=='.json'])
    for name in ('meta.json','metadata.json','manifest.json'):
        f=q/name
        if f.exists(): print(name,f.read_text()[:16000])
''')
