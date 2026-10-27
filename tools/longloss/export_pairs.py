"""gpuA: stream only converter arrays and paired-return provenance, never raw runs."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

from guard import install, safe

install()
BASE = Path('/root/autodl-tmp/c4/restart20261026/arm/loop')
OLD = Path('/root/autodl-tmp/c4/wmdeep20261027')
ARRAYS = ('veh_kin', 'veh_mask', 'target_mask', 'counter', 'vid', 'is_cav', 'time_s',
          'control_time_s', 'controller', 'issued_acceleration_mps2',
          'issued_target_speed_mps', 'entry_backlog', 'request_counter', 'entry_cav_counter')


def main():
    split = json.loads((OLD/'SPLIT.json').read_text())
    excluded = set(split['score_pairs_seeds']) | set(split['loop_validation_seeds'])
    entries, provenance = [], []
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as tar:
        def blob(name, data):
            info = tarfile.TarInfo(name); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for cycle in ('c00.a1', 'c01.a1', 'c02.a1'):
            root = BASE/'data'/('pairs_'+cycle)
            doc = json.loads(safe(root/'C4_EPISODES_MANIFEST.json').read_text())
            for e0 in doc['episodes']:
                seed = int(e0['seed'])
                assert not 964000 <= seed <= 964199
                if seed in excluded:
                    continue
                assert e0['branch']['kind'] in ('B', 'POLICY')
                assert e0['branch']['horizon_s'] == 300
                e = dict(e0)
                src = safe(root/e['dir'])
                dest = Path('inputs')/cycle/e['episode']
                for name in ARRAYS:
                    p = safe(src/(name+'.npy'))
                    if p.is_file():
                        tar.add(p, arcname=str(dest/p.name), recursive=False)
                    else:
                        assert name in ('entry_backlog', 'request_counter', 'entry_cav_counter')
                e['dir'] = str(dest)
                e['cycle'] = cycle
                entries.append(e)
            for name in ('returns_'+cycle+'.npz', 'native_returns_'+cycle+'.npz',
                         'pairing_'+cycle+'.json'):
                p = safe(BASE/'returns'/name)
                data = p.read_bytes()
                blob('provenance/'+name, data)
                provenance.append({'name': name, 'sha256': hashlib.sha256(data).hexdigest()})
        doc['episodes'] = entries
        doc.update(n_episodes=len(entries), roots=[], split='TRAIN')
        blob('LOOP_EXPORT.json', json.dumps(doc).encode())
        blob('PROVENANCE.json', json.dumps(provenance).encode())


if __name__ == '__main__':
    main()
