"""Checkpoint-linked exposure audit using S-C1GAP's metadata exports only."""
from pathlib import Path
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent
GAP=Path('E:/agent-work/c4-c1gap/docs/paper_results')
p=GAP/'c1gap/wm2_remote_provenance.json'
x=json.loads(p.read_text())
training=set()
sources=[]
for ck in x['checkpoints']:
    for name,src in ck.get('sources',{}).items():
        train=src.get('train',{})
        positive=train.get('positive_episode_worlds',[])
        training.update(positive)
        arms=train.get('positive_episode_arm_counts',{})
        assert not any(any(bad in arm.lower() for bad in ('mappo','pi-external','external-u','wall')) for arm in arms)
        sources.append(dict(checkpoint_sha256=ck['sha256'],source_name=name,
            manifest_sha256=src.get('manifest_sha256'),positive_worlds=len(positive),arms=arms))
    paired=ck.get('other_data',{}).get('paired_batches',{}).get('source',{})
    training.update(paired.get('worlds',[]))
seeds={int(re.match(r'w(\d+)m',w).group(1)) for w in training}
overlap=sorted(seeds&set(range(969400,969440)))
assert x['checkpoints'][0]['sha256']=='b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
assert not overlap
out=dict(task='S-C1CLOSED',source='S-C1GAP hash-linked metadata export; no raw training arrays opened',
    source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    gap_report_sha256=hashlib.sha256((GAP/'C1_GAPS.md').read_bytes()).hexdigest(),
    audited_training_worlds=len(training),eval_seeds=list(range(969400,969440)),
    audited_training_seed_overlap=overlap,sources=sources,
    primary_controller_novelty='PI/MAPPO absent in audited algorithm corpora; unresolved pre-R16 initialization exclusion',
    caveat='provisional algorithm novelty, not an unbroken initialization-lineage certificate; CTRL0 exposed, CIRCLES diagnostic',
    wmfix_secondary_status='no independent passing verdict supplied in read v2-DATA / ES reports',
    dependencies={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [
        Path('E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_PILOT_V2DATA.md'),
        Path('E:/agent-work/c4-wmfixes/docs/paper_results/WMFIX_ES.md'),
        Path('E:/agent-work/c4-c1floor/docs/paper_results/C1FLOOR_PREREQUISITES.md')]})
with (HERE/'evidence/ELIGIBILITY.json').open('x') as f:json.dump(out,f,indent=1)
print(json.dumps({k:out[k] for k in ('audited_training_worlds','audited_training_seed_overlap','primary_controller_novelty')}))
