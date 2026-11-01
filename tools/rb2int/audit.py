"""Outcome-blind recorded-world panel audit; no inference, fitting or simulator.

Only the two named collection designs and named provenance/handoff files are
read. No trajectory, future target, assignment file or directory tree is opened.
Worker attestations are a separate prerequisite: this audit does not invent them.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
ARMS = ['native', 'ctrl0', 'mappo', 'pi', 'circles']
FORBIDDEN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b',
             'inc1', 'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')


def checked(path):
    path = Path(path)
    if any(part in str(path).lower() for part in FORBIDDEN):
        raise ValueError('prohibited path')
    return path


def sha(path):
    return hashlib.sha256(checked(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(checked(path).read_text(encoding='utf-8'))


def git(path, *args):
    return subprocess.check_output(['git', *args], cwd=path, text=True).strip()


def select_panel(first, second):
    """Selection depends ONLY on world identities and collection-design cells."""
    if first['arms'] != ARMS or second['arms'][:5] != ARMS:
        raise ValueError('five original recorded arms required')
    if (first['warmup_s'], first['control_s']) != (300, 600) or (
            second['warmup_s'], second['control_s']) != (300, 600):
        raise ValueError('recorded clock contract changed')
    reserved = [w for w in first['worlds'] if 969432 <= w['seed'] <= 969439]
    added = second['worlds']
    if sorted(w['seed'] for w in reserved) != list(range(969432, 969440)):
        raise ValueError('reserved-eight metadata incomplete')
    if sorted(w['seed'] for w in added) != list(range(969550, 969600)):
        raise ValueError('WMDATA2 metadata incomplete')
    worlds = []
    for source, rows in (('WMDATA', reserved), ('WMDATA2', added)):
        for row in rows:
            worlds.append(dict(row, collection=source))
    cells = defaultdict(list)
    for w in worlds:
        cells[w['penetration'], w['demand']].append(w['seed'])
    replicated = sorted(s for ids in cells.values() if len(ids) >= 2 for s in ids)
    singletons = sorted(s for ids in cells.values() if len(ids) == 1 for s in ids)
    return dict(worlds=worlds, final_comparison_worlds=sorted(w['seed'] for w in worlds),
        replicated_primary_worlds=replicated, descriptive_singleton_worlds=singletons,
        reserved8=list(range(969432, 969440)),
        cells=[dict(penetration=p, demand=d, seeds=sorted(ids), n=len(ids),
                    replicated=len(ids) >= 2) for (p, d), ids in sorted(cells.items())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    designs = [WORK/'c4-wmdata/tools/wmdata/design.json',
               WORK/'c4-wmdata2/tools/wmdata2/design.json']
    panel = select_panel(*(load(p) for p in designs))
    fit_path = ROOT/'artifacts/rbfit/shrinkage_v1/idm_shrinkage.json'
    fit = load(fit_path)
    if fit['fit_worlds'] != list(range(969400, 969432)):
        raise ValueError('inherited shrinkage fit lineage differs')
    if set(fit['fit_worlds']) & set(panel['final_comparison_worlds']):
        raise ValueError('fit/evaluation overlap')
    peers = []
    for name in ('lane', 'break', 'cal', 'lphys'):
        peer = WORK/('c4-rb2-'+name)
        path = peer/'HANDOFF.md'
        peers.append(dict(worker=name, head=git(peer, 'rev-parse', 'HEAD'),
            status=git(peer, 'status', '--short'), handoff=str(path),
            handoff_sha256=sha(path), text=path.read_text(encoding='utf-8'),
            attestation_accepted=False))
    result = dict(schema='rb2int-panel-metadata/1',
        assembled_at=datetime.now().astimezone().isoformat(),
        selection='all recorded WMDATA reserved8 + all WMDATA2; primary cells require n>=2',
        selection_used_outcomes=False, **panel, arms=ARMS, K=8, horizons_s=[120, 300, 600],
        designs={str(p): sha(p) for p in designs},
        inherited_fit=dict(path=str(fit_path), sha256=sha(fit_path),
                           fit_worlds=fit['fit_worlds']), worker_snapshots=peers,
        historical_exposure='Prior aggregate reports and corpus analyses disclosed; not historically untouched.',
        final_inference_authorized=False,
        pending=['four worker fit/selection and inherited artifact exposure attestations',
                 'identity-bound remote record availability and ownership audit',
                 'candidate freeze and development-only smoke'],
        full12_gate='cannot qualify from four singleton cells; replicated subset is explicitly scoped')
    out = checked(args.out)
    if not out.parent.is_dir():
        raise ValueError('receipt parent must exist')
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(receipt=str(out), sha256=sha(out),
        final_worlds=len(panel['final_comparison_worlds']),
        primary_worlds=len(panel['replicated_primary_worlds']),
        singleton_worlds=panel['descriptive_singleton_worlds'],
        cells=panel['cells'], final_inference_authorized=False), indent=1))


if __name__ == '__main__':
    main()
