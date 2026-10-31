"""Checkpoint-specific training-world/algorithm comparison from safe audited metadata."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('E:/agent-work/c4-c1gap/docs/paper_results/c1gap')


def main():
    panel=json.loads((ROOT/'docs/paper_results/c1pidiag/PANEL.json').read_text())['states']
    old=json.loads((SOURCE/'exposure.json').read_text())
    pools={'c1struct_and_frozen_BBs':set(old['historical']['effective_worlds'])}
    hashes={}
    for tag in ('13e','wm2'):
        path=SOURCE/f'{tag}_remote_provenance.json';doc=json.loads(path.read_text())
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        train=set()
        for checkpoint in doc['checkpoints']:
            for src in checkpoint.get('sources',{}).values():
                train.update(src.get('train',{}).get('positive_episode_worlds',[]))
                train.update(src.get('train',{}).get('worlds',[]))
        pools[tag]=train
    results={}
    for cohort in ('pi_u_same','wall_same'):
        worlds={e['world'] for e in panel if e['cohort']==cohort}
        results[cohort]={tag:dict(n_eval_worlds=len(worlds),n_audited_train_worlds=len(train),
            overlap=sorted(worlds & train),overlap_count=len(worlds & train)) for tag,train in pools.items()}
    value=dict(sources=hashes,cohorts=results,old_effective_controllers=old['historical']['effective_policy_counts'],
        caveat='Audited post-initialization corpora only for newer models; unresolved copied initialization lineage remains. World exposure is not controller exposure, and different training data/physics cannot identify an architecture effect.')
    with (ROOT/'docs/paper_results/c1pidiag/COVERAGE.json').open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
