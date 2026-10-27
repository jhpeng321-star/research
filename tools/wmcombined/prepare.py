"""Detached input staging; only receipts and the WM2 start checkpoint (no bulk copies).

Reuses the S-LONGLOSSTEST artifacts read-only: DESIGN/SPLIT, accept panel and the
exported ``views/loop14_pairs`` paired tapes stay in the longloss OUT (referenced by
path, never copied); the S-E3DATA-1 dataset stays at ``/root/autodl-tmp/c4/e3data``.
The only new bytes are the WM2 start checkpoint (3.6 MiB, streamed from gpuA with the
registered sha asserted) and JSON receipts.
"""
import hashlib
import json
from pathlib import Path
import subprocess

from common import GPUA_WM2, ID, OLD, OUT, VARIANTS, WM2_SHA256, disk, event, write
from guard import safe


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch_wm2():
    (OUT/'wm2').mkdir(parents=True, exist_ok=True)
    target = OUT/'wm2/wm_s.pt'
    assert not target.exists(), 'WM2 already staged'
    with target.open('wb') as f:
        subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                        'gpuA', 'cat', GPUA_WM2], stdout=f, check=True, timeout=120)
    got = sha(target)
    assert got == WM2_SHA256, f'WM2 sha mismatch: {got}'
    with (OUT/'wm2/config.resolved.loop.json').open('wb') as f:
        subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'gpuA',
                        'cat', str(Path(GPUA_WM2).parent/'config.resolved.json')],
                       stdout=f, check=True, timeout=60)
    return got


def main():
    disk()
    assert (OLD/'DONE.json').is_file(), 'S-LONGLOSSTEST DONE.json missing (wait condition)'
    old = json.loads((OLD/'DESIGN.json').read_text())
    split = json.loads(safe(OLD/'SPLIT.json').read_text())
    excluded = set(split['score_pairs_seeds']) | set(split['loop_validation_seeds'])
    assert not excluded & set(range(964000, 964200))

    proofs = []
    for cycle in ('c00.a1', 'c01.a1', 'c02.a1'):
        d = json.loads((safe(OLD/'provenance')/f'pairing_{cycle}.json').read_text())
        assert d['policy_sha256'] == sha(safe(OLD/'provenance')/f'returns_{cycle}.npz')
        assert d['native_sha256'] == sha(safe(OLD/'provenance')/f'native_returns_{cycle}.npz')
        assert all(p['horizon_s'] == 300 and p['same_saved_start'] and
                   p['observable_start_sha256'] for p in d['pairs'])
        proofs.append({'cycle': cycle,
                       'sha256': sha(safe(OLD/'provenance')/f'pairing_{cycle}.json'),
                       'pairs': len(d['pairs'])})
    loop_manifest = json.loads(
        (safe(OLD/'views/loop14_pairs/C4_EPISODES_MANIFEST.json')).read_text())
    e3_split = json.loads((safe('/root/autodl-tmp/c4/e3data/SPLIT.json')).read_text())
    counts = {r: sum(v == r for v in e3_split['worlds'].values())
              for r in ('fit', 'calibration', 'validation')}
    assert counts == {'fit': 156, 'calibration': 36, 'validation': 36}, counts

    for name in ('accept_panel.json', 'SPLIT.json'):
        (OUT/name).write_bytes((OLD/name).read_bytes())

    wm2_sha = fetch_wm2()
    sources = {**old['sources'], 'loop14_pairs': str(OLD/'views/loop14_pairs'),
               'e3_pairs': '/root/autodl-tmp/c4/e3data'}
    write('DESIGN.json', {**old, 'id': ID, 'config': str(OUT/'runtime/configs/c4_wm_combined.toml'),
                          'runtime': str(OUT/'runtime'), 'sources': sources,
                          'validation_sources': old['sources'], 'longwm': True,
                          'longwm_extra_paired': 'e3_pairs',
                          'training_cap_s': 12600, 'unroll_s': 300, 'truncation_ticks': 600,
                          'variants': list(VARIANTS), 'pair_proofs': proofs,
                          'original_split_sha256': sha(OLD/'SPLIT.json'),
                          'loop_episodes': len(loop_manifest['episodes']),
                          'e3_fit_groups': 156, 'e3_split_counts': counts,
                          'e3_split_sha256': sha('/root/autodl-tmp/c4/e3data/SPLIT.json'),
                          'wm2_sha256': wm2_sha,
                          'selection': {'criterion': 'held-out 300 s paired dJ error on the '
                                                     'E3DATA calibration worlds (36)',
                                        'primary': 'abs(mean signed error), strict improvement',
                                        'reported': 'MAE, sign agreement',
                                        'every_steps': 10, 'patience': 4,
                                        'diagnostic_only': 'fixed16 120 s mean_position',
                                        'final_step_checkpoint': 'saved and scored'},
                          'free_gib': disk(),
                          'verdict': "owner-amended (registered 21:38): helps iff 300 s "
                                     "|mean paired error| falls with world-bootstrap CI "
                                     "excluding no change OR mean sign agreement over "
                                     "120/180/300 s rises with CI excluding no change; "
                                     "short horizons reported, non-gating; original rule "
                                     "reported alongside; primary panel = pairs300 48-group "
                                     "registered panel; E3 validation worlds reported too"})
    event(phase='prepared', loop_episodes=len(loop_manifest['episodes']), proofs=proofs,
          wm2_sha256=wm2_sha)


if __name__ == '__main__':
    main()
