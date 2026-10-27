"""Detached input staging; only converted numeric arrays and return receipts."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from common import CONFIG, OLD, OUT, disk, event, write
from guard import safe


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    disk()
    old = json.loads((OLD/'DESIGN.json').read_text())
    split = json.loads((OLD/'SPLIT.json').read_text())
    excluded = set(split['score_pairs_seeds']) | set(split['loop_validation_seeds'])
    # Stream directly gpuA -> gpuR, no raw simulation artifacts or laptop relay.
    source = (OUT/'code/export_pairs.py').read_text()
    guard = (OUT/'code/guard.py').read_text()
    source = source.replace('from guard import install, safe', guard)
    encoded = base64.b64encode(source.encode()).decode()
    cmd = f"/root/miniconda3/bin/python -B -c \"import base64; exec(base64.b64decode('{encoded}'))\""
    with (OUT/'export.log').open('x') as log:
        p = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                              'gpuA', cmd], stdout=subprocess.PIPE, stderr=log)
        try:
            with tarfile.open(fileobj=p.stdout, mode='r|gz') as tar:
                for member in tar:
                    disk()
                    dest = safe(OUT/member.name)
                    assert dest.resolve().is_relative_to(OUT) and member.isfile()
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with dest.open('xb') as f:
                        shutil.copyfileobj(tar.extractfile(member), f)
            assert p.wait(timeout=60) == 0, 'gpuA export failed'
        finally:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=30)
    doc = json.loads((OUT/'LOOP_EXPORT.json').read_text())
    train, val = set(split['train_seeds']), set()
    # Recover original global validation roles, not merely each source's local role.
    for root in old['sources'].values():
        d = json.loads((safe(root)/'C4_EPISODES_MANIFEST.json').read_text())
        val.update(int(e['seed']) for e in d['episodes'] if e['split_role'] == 'val')
    assert not train & val
    new_val = set()
    for e in doc['episodes']:
        seed = int(e['seed'])
        assert seed not in excluded and not 964000 <= seed <= 964199
        if seed not in train | val:
            assert 968170 <= seed <= 968182
            if int(hashlib.sha256(f'longloss20261027:{seed}'.encode()).hexdigest(), 16) % 4 == 0:
                new_val.add(seed)
                val.add(seed)
            else:
                train.add(seed)
        e['split_role'] = 'val' if seed in val else 'train'
        e['dir'] = str(safe(OUT/e['dir']))
    doc.update(n_train=sum(e['split_role'] == 'train' for e in doc['episodes']))
    doc['n_val'] = len(doc['episodes'])-doc['n_train']
    write('views/loop14_pairs/C4_EPISODES_MANIFEST.json', doc)
    proofs = []
    for cycle in ('c00.a1', 'c01.a1', 'c02.a1'):
        d = json.loads((OUT/'provenance'/f'pairing_{cycle}.json').read_text())
        assert d['policy_sha256'] == sha(OUT/'provenance'/f'returns_{cycle}.npz')
        assert d['native_sha256'] == sha(OUT/'provenance'/f'native_returns_{cycle}.npz')
        assert all(p['horizon_s'] == 300 and p['same_saved_start'] and
                   p['observable_start_sha256'] for p in d['pairs'])
        proofs.append({'cycle': cycle, 'sha256': sha(OUT/'provenance'/f'pairing_{cycle}.json'),
                       'pairs': len(d['pairs'])})
    (OUT/'wm1').mkdir()
    for name in ('wm_s.pt', 'physics.json'):
        shutil.copy2(OLD/'wm1'/name, OUT/'wm1'/name)
    assert sha(OUT/'wm1/wm_s.pt') == '5257a64c7cbb6a8cb828cb26230e97c686d7715cc33825bd3e1ca95f7bf51d8e'
    for name in ('wm1_gate.json', 'accept_panel.json', 'SPLIT.json'):
        shutil.copy2(OLD/name, OUT/name)
    sources = {**old['sources'], 'loop14_pairs': str(OUT/'views/loop14_pairs')}
    write('DESIGN.json', {**old, 'id': 'S-LONGLOSSTEST-1', 'config': str(CONFIG),
                         'runtime': str(OUT/'runtime'), 'sources': sources,
                         'validation_sources': old['sources'], 'longwm': True,
                         'training_cap_s': 10800, 'unroll_s': 300, 'truncation_ticks': 600,
                         'new_validation_seeds': sorted(new_val), 'pair_proofs': proofs,
                         'original_split_sha256': sha(OLD/'SPLIT.json'),
                         'loop_episodes': len(doc['episodes']), 'free_gib': disk(),
                         'verdict': 'absolute 300s bias reduction > 0, world bootstrap CI low > 0, short signs pass'})
    event(phase='prepared', episodes=len(doc['episodes']), proofs=proofs)


if __name__ == '__main__':
    main()
