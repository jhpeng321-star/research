"""Bounded metadata inspection of explicitly allowed cached analysis inputs."""
import json
from pathlib import Path

BASE = Path('E:/agent-work')
BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')


def load(p):
    assert not any(s in str(p).lower() for s in BAD), p
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    refs = load(BASE/'c4-bdrank/tools/rbeval/references.json')
    print('DETECTOR', json.dumps(refs['breakdown']))
    audit = load(BASE/'c4-rb-int/docs/rbint/noon/20261101T093000+0800/FULL-v4/distribution/audit.json')
    p = Path(audit['inputs'][0]['task'])
    task = load(p)
    print('V4_TASK', p, list(task), json.dumps(task['episodes'][0])[:5000])
    for version, name in [('v4', 'SNAPSHOT_1793521615.json'), ('v5', 'V5_QUALITY_SNAPSHOT_1793521619.json')]:
        d = load(BASE/'c4-wmteach/docs/paper_results/wmteach'/name)
        print('TEACH_SNAPSHOT', version, list(d))
        print('FILES', list(d.get('files', {})))
        for key in ('labels', 'rows', 'source_root', 'root'):
            if key in d:
                value = d[key]
                print(key, json.dumps(value[:1] if isinstance(value, list) else value)[:6000])
        for key in ('QUALITY_RECORDS.json', 'IDENTITY.json'):
            if key in d.get('files', {}):
                value = d['files'][key]
                print(key, json.dumps(value[:1] if isinstance(value, list) else value)[:3000])
    for name in ('COLLECT_20261101T161643+0800_gpuA.json', 'RUN_REGISTRY_FIRST_CACHE.json', 'FINAL_RUN_PLAN.json'):
        d = load(BASE/'c4-rb2-int/docs/rb2int'/name)
        print('RB2', name, json.dumps(d)[:5000])


if __name__ == '__main__':
    main()
