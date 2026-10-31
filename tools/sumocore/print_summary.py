"""Small read-only report view; scores/summary.json retains every group."""
from remote import ssh

SOURCE=r'''
from pathlib import Path
import json
p=Path('/home/ubuntu/sumocore20261031/scores/summary.json')
d=json.loads(p.read_text());print('MATCHED_WM2',json.dumps(d['wm2_matched']))
print('INVALID',json.dumps(d['collision_flagged_attempts']))
for s in d['summary']:
 if (s['split']=='all8' and (s['arm']=='all' or (s['horizon']==600 and s['arm'] in ('native','mappo','pi')))) or (s['split']!='all8' and s['horizon']==600 and s['arm']=='all'):
  print(json.dumps(s))
'''
if __name__=='__main__':print(ssh(SOURCE).decode())
